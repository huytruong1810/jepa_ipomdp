# ABSOLUTE PATH: main.py
# ==============================================================================
# HIGH-PERFORMANCE VECTORIZED TRAINING ORCHESTRATION DAEMON
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Selective PyTorch Compilation for CUDA Graphs:
#    - Selectively compiles fixed-shape neural towers (jepa_model, value_head,
#      reward_head, opponent_head) while keeping dynamic MCTS search in eager Python.
#    - Issues torch.compiler.cudagraph_mark_step_begin() prior to trainer updates for CUDA Graph safety.
#
# 2. Canonical Interaction Timing and Memory Storage:
#    - The domain (src/ipomdp/domain) emits no observation at reset. Each episode starts
#      with the all-zero "empty history" observation (see src/ipomdp/agents/jepa_agent.py);
#      then, every step: agent.observe(o_t) -> a_t = agent.act() -> (o_{t+1}, r_t) = env.step(a_t).
#    - The buffer stores (o_t, a_t, r_t) per step; when a row truncates, the true final
#      observation o_T is passed to buffer.end_episode() BEFORE the row is reset, so the
#      terminal transition pair is never corrupted by the next episode's start.
#
# 3. Row Isolation on Truncation:
#    - Truncated rows are reset in the simulator, the agent's recurrent state, and the
#      pending observation, without touching the other rows.
#
# 4. Single Source of Truth for the Domain:
#    - |A|, |O|, action names and the discount gamma are read from the FinitePOMDP; gamma
#      is passed explicitly to both the planner and the trainer so they cannot disagree
#      with the benchmark solver.
#
# 5. Graceful Shutdown & Checkpoint Integrity:
#    - Intercepts SIGINT / KeyboardInterrupt, saving an emergency interrupt_checkpoint.pt
#      before cleanly closing the TensorBoard SummaryWriter.
# ==============================================================================

from collections import deque
import gc
from pathlib import Path
import sys
import warnings
import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor")

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ipomdp.telemetry import (
    MetricsLogger,
    setup_logger,
    ModelCheckpointer,
    MCTSGraphVisualizer,
    LatentSpaceVisualizer,
    RewardTrajectoryVisualizer,
    SystemTelemetryMonitor,
    PipelineProfiler,
    ExecutionGuardrail,
)
from ipomdp.domain import BatchedPOMDPEnv, FinitePOMDP, build_tiger_pomdp
from ipomdp.models import (
    MLPFeatureExtractor,
    RecurrentContextEncoder,
    CausalRelationalPredictor,
    RecurrentJEPABase,
    ValueHead,
    RewardHead,
    DiscretePolicyHead,
)
from ipomdp.planning import DiscreteLatentOpenLoopSearch
from ipomdp.training import PrioritizedSequenceBuffer, DiscreteRecurrentIPOMDPTrainer
from ipomdp.agents import DiscreteJEPAAgent

# Domains selectable through conf/env/<name>.yaml. Each builder returns the exact model.
DOMAIN_BUILDERS = {"tiger": build_tiger_pomdp}

# The canonical single-agent POMDP has no opponent. Until the opponent machinery of the
# world model and planner is reviewed (Phases 3-4), it is fed a singleton opponent action
# space {0}: the opponent head then predicts a constant and contributes no information.
OPPONENT_ACTION_DIM = 1

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    logger = setup_logger("IPOMDP_Experiment")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    logger.info(f"Compute Device: {device} | Vector Batch Size: {cfg.training.env_batch_size} | Seed: {cfg.seed}")

    pomdp: FinitePOMDP = DOMAIN_BUILDERS[cfg.env.name]()
    num_obs, num_actions = pomdp.num_observations, pomdp.num_actions
    action_map = dict(enumerate(pomdp.action_names))
    batch_size = cfg.training.env_batch_size
    env = BatchedPOMDPEnv(pomdp, batch_size, max_steps=cfg.env.max_steps, seed=cfg.seed, device=device)

    extractor = MLPFeatureExtractor(
        obs_dim=num_obs,
        hidden_dim=cfg.model.latent_dim,
        num_objects=cfg.model.num_objects,
        num_blocks=cfg.model.num_blocks
    )

    encoder = RecurrentContextEncoder(
        extractor,
        action_dim=num_actions,
        latent_dim=cfg.model.latent_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_blocks=cfg.model.num_blocks
    )

    predictor = CausalRelationalPredictor(
        num_objects=cfg.model.num_objects,
        latent_dim=cfg.model.latent_dim,
        action_dim_i=num_actions,
        action_dim_j=OPPONENT_ACTION_DIM,
        hidden_dim=cfg.model.hidden_dim,
        num_blocks=cfg.model.num_blocks
    )

    jepa_model = RecurrentJEPABase(encoder, predictor).to(device)

    value_head = ValueHead(
        latent_dim=cfg.model.latent_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_blocks=cfg.model.num_blocks
    ).to(device)

    reward_head = RewardHead(
        latent_dim=cfg.model.latent_dim,
        action_dim_i=num_actions,
        action_dim_j=OPPONENT_ACTION_DIM,
        hidden_dim=cfg.model.hidden_dim,
        num_blocks=cfg.model.num_blocks
    ).to(device)

    opponent_head = DiscretePolicyHead(
        latent_dim=cfg.model.latent_dim,
        action_dim=OPPONENT_ACTION_DIM,
        num_opponents=1,
        hidden_dim=cfg.model.hidden_dim,
        num_blocks=cfg.model.num_blocks
    ).to(device)

    # Selective compilation of the fixed-shape networks; MCTS control flow stays eager.
    use_compile = cfg.training.compile and device.type == "cuda"
    networks = (jepa_model, value_head, reward_head, opponent_head)
    if use_compile:
        logger.info("Selective PyTorch compilation enabled...")
        networks = tuple(torch.compile(net) for net in networks)
    run_jepa, run_value, run_reward, run_opponent = networks

    planner = DiscreteLatentOpenLoopSearch(
        jepa_model=run_jepa,
        value_head=run_value,
        reward_head=run_reward,
        opponent_head=run_opponent,
        action_dim_i=num_actions,
        action_dim_j=OPPONENT_ACTION_DIM,
        num_simulations=cfg.mcts.num_simulations,
        num_latent_obs=cfg.mcts.num_latent_obs,
        discount=pomdp.discount
    )
    trainer = DiscreteRecurrentIPOMDPTrainer(
        jepa_model=run_jepa,
        value_head=run_value,
        reward_head=run_reward,
        opponent_head=run_opponent,
        logger=logger,
        device=device,
        latent_dim=cfg.model.latent_dim,
        action_dim_i=num_actions,
        action_dim_j=OPPONENT_ACTION_DIM,
        num_objects=cfg.model.num_objects,
        gamma=pomdp.discount,
        detach_belief_for_rl=False
    )

    agent = DiscreteJEPAAgent(
        planner=planner,
        batch_size=batch_size,
        num_actions=num_actions,
        num_objects=cfg.model.num_objects,
        latent_dim=cfg.model.latent_dim,
        device=device,
        temperature=cfg.agent.temperature,
        temperature_min=cfg.agent.temperature_min,
        temperature_decay=cfg.agent.temperature_decay
    )

    buffer = PrioritizedSequenceBuffer(
        capacity=2048,
        burn_in=cfg.training.burn_in,
        seq_len=cfg.training.train_seq_len
    )

    checkpointer = ModelCheckpointer(f"{cfg.env.name}_checkpoints", logger)
    metrics_logger = MetricsLogger(log_dir=f"{cfg.env.name}_tensorboard", experiment_name="jepa_ipomdp")
    latent_viz = LatentSpaceVisualizer(save_dir=f"{cfg.env.name}_plots/latent")
    tree_viz = MCTSGraphVisualizer(save_dir=f"{cfg.env.name}_plots/trees")
    reward_viz = RewardTrajectoryVisualizer(save_dir=f"{cfg.env.name}_plots")

    system_monitor = SystemTelemetryMonitor(thermal_threshold_c=82.0)
    profiler = PipelineProfiler(window_size=100)

    models_dict = {
        "jepa": jepa_model,
        "value": value_head,
        "reward": reward_head,
        "opponent": opponent_head
    }

    current_step_holder = [0]
    guardrail = ExecutionGuardrail(
        logger=logger,
        system_monitor=system_monitor,
        profiler=profiler,
        thermal_trip_c=82.0,
        thermal_recovery_c=72.0,
        vram_trip_mb=13500.0,
        rss_trip_mb=18000.0,
        emergency_save_fn=lambda: checkpointer.save(
            current_step_holder[0], models_dict, trainer.optimizer, float('inf'), filename="emergency_guardrail_checkpoint.pt"
        )
    )


    start_step = 0
    latest_path = Path(f"{cfg.env.name}_checkpoints/latest_checkpoint.pt")
    if latest_path.exists():
        start_step = checkpointer.load(str(latest_path), models_dict, trainer.optimizer, device)

    # o_t fed to the filter: the all-zero "empty history" at the start of every episode.
    observation = torch.zeros(batch_size, num_obs, device=device)
    no_opponent_action = torch.zeros(1, dtype=torch.int64)

    ep_returns = np.zeros(batch_size, dtype=np.float32)
    recent_ep_returns = deque(maxlen=100)
    ep_lengths = np.zeros(batch_size, dtype=np.int32)
    recent_ep_lengths = deque(maxlen=100)

    # Within-episode cumulative trajectory tracking
    active_trajectories = [
        {"cum_rewards": [0.0], "actions": [], "rewards": []}
        for _ in range(batch_size)
    ]
    completed_trajectories = deque(maxlen=64)

    logger.info("Entering Vectorized Training Loop...")
    latest_metrics = {}
    episodes_completed = 0
    do_viz = False
    viz_beliefs, viz_true_states, viz_actions = [], [], []

    try:
        pbar = tqdm(range(start_step, cfg.training.total_steps), desc="Steps", initial=start_step, total=cfg.training.total_steps)
        for step in pbar:
            current_step_holder[0] = step

            with profiler.profile("update_belief"):
                agent.observe(observation)

            if do_viz:
                viz_beliefs.append(agent.belief[0].clone().detach().unsqueeze(0))
                viz_true_states.append(env.state[0:1])

            with profiler.profile("mcts_search"):
                if episodes_completed < cfg.training.warmup_episodes:
                    action = agent.act_uniformly()
                else:
                    action = agent.act()

            if do_viz:
                viz_actions.append(int(action[0].item()))
                if len(viz_actions) == 1 and planner.root is not None:
                    tree_viz.visualize(planner.root, filename=f"mcts_tree_ep{episodes_completed}")

            with profiler.profile("env_step"):
                out = env.step(action)

            next_observation = F.one_hot(out.observation, num_classes=num_obs).float()
            rewards = out.reward.tolist()
            actions_taken = action.tolist()
            truncated = out.truncated.tolist()

            for i in range(batch_size):
                step_reward = rewards[i]
                act_taken = actions_taken[i]
                ep_returns[i] += step_reward
                ep_lengths[i] += 1

                active_trajectories[i]["actions"].append(act_taken)
                active_trajectories[i]["rewards"].append(step_reward)
                active_trajectories[i]["cum_rewards"].append(active_trajectories[i]["cum_rewards"][-1] + step_reward)

                buffer.push(
                    env_idx=i,
                    obs=observation[i],
                    act_i=action[i].view(1),
                    act_j=no_opponent_action,
                    reward=step_reward
                )

                if truncated[i]:
                    recent_ep_returns.append(ep_returns[i])
                    recent_ep_lengths.append(ep_lengths[i])
                    ep_returns[i] = 0.0
                    ep_lengths[i] = 0

                    completed_trajectories.append({
                        "cum_rewards": list(active_trajectories[i]["cum_rewards"]),
                        "actions": list(active_trajectories[i]["actions"]),
                        "rewards": list(active_trajectories[i]["rewards"]),
                    })
                    active_trajectories[i] = {"cum_rewards": [0.0], "actions": [], "rewards": []}

                    # Truncation is not termination: the transition bootstraps through o_T.
                    buffer.end_episode(env_idx=i, final_obs=next_observation[i], terminated=False)

                    if i == 0:
                        if do_viz and len(viz_beliefs) > 1:
                            latent_viz.plot_trajectory(
                                beliefs=viz_beliefs,
                                true_states=viz_true_states,
                                actions=viz_actions,
                                filename=f"trajectory_ep{episodes_completed}",
                                value_head=value_head,
                                action_map=action_map
                            )

                        if do_viz and completed_trajectories:
                            recent_trajs = list(completed_trajectories)[-min(16, len(completed_trajectories)):]
                            fig = reward_viz.plot_cumulative_rewards(
                                trajectories=recent_trajs,
                                filename="canonical_episode_cumulative_rewards",
                                title_suffix=f"Step {step:,}, Ep {episodes_completed}",
                                action_map=action_map
                            )
                            metrics_logger.log_figure("Visuals/within_episode_cumulative_reward", fig, step)
                            plt.close(fig)

                        episodes_completed += 1
                        agent.anneal_temperature()
                        do_viz = (episodes_completed % cfg.training.viz_freq == 0) and (episodes_completed > 0)
                        viz_beliefs.clear()
                        viz_true_states.clear()
                        viz_actions.clear()

            # Start new episodes in truncated rows: simulator, recurrent state, empty history.
            truncated_mask = out.truncated
            env.reset_rows(truncated_mask)
            agent.reset_rows(truncated_mask)
            next_observation[truncated_mask] = 0.0
            observation = next_observation

            performed_train = False
            if buffer.tree.size >= cfg.training.batch_size and step % cfg.training.update_freq == 0:
                if use_compile and device.type == "cuda":
                    torch.compiler.cudagraph_mark_step_begin()

                with profiler.profile("train_sequence"):
                    batch, is_weights, tree_indices = buffer.sample_sequence(cfg.training.batch_size)
                    metrics, td_errors = trainer.train_sequence(batch, is_weights)
                    buffer.update_priorities(tree_indices, td_errors)
                performed_train = True

                if len(recent_ep_returns) > 0:
                    metrics["episode_reward_mean"] = float(np.mean(recent_ep_returns))
                    metrics["episode_reward_min"] = float(np.min(recent_ep_returns))
                    metrics["episode_reward_max"] = float(np.max(recent_ep_returns))
                    metrics["episode_reward_std"] = float(np.std(recent_ep_returns))
                    metrics["episode_length_mean"] = float(np.mean(recent_ep_lengths))

                if completed_trajectories:
                    all_final_returns = [t["cum_rewards"][-1] for t in completed_trajectories]
                    metrics["episode_reward_max_within_ep"] = float(np.max(all_final_returns))
                    all_recent_rewards = [r for t in completed_trajectories for r in t["rewards"]]
                    all_recent_actions = [a for t in completed_trajectories for a in t["actions"]]
                    door_openings = sum(1 for a in all_recent_actions if a != 0)
                    treasures = sum(1 for r in all_recent_rewards if r > 0)
                    tigers = sum(1 for r in all_recent_rewards if r <= -10)
                    if door_openings > 0:
                        metrics["treasure_accuracy_pct"] = float(treasures / door_openings * 100.0)
                        metrics["tiger_penalty_pct"] = float(tigers / door_openings * 100.0)

                metrics["step_reward_mean"] = float(out.reward.mean().item())
                metrics["episodes_completed"] = episodes_completed
                metrics["agent_temperature"] = float(agent.temperature)

                act_data = action
                for a_idx, a_name in action_map.items():
                    act_key = f"action_pct_{a_name.lower().replace(' ', '_')}"
                    metrics[act_key] = float((act_data == a_idx).float().mean().item())

                if hasattr(planner, 'last_avg_depth'):
                    metrics["mcts_avg_depth"] = float(planner.last_avg_depth)
                    metrics["mcts_max_depth"] = float(planner.last_max_depth)
                    metrics["mcts_q_spread"] = float(planner.last_q_spread)
                    metrics["mcts_entropy"] = float(planner.last_entropy)

                latest_metrics = metrics
                metrics_logger.log_metrics(metrics, step, prefix="Train")
                metrics_logger.log_metrics(system_monitor.get_metrics(), step, prefix="System")
                metrics_logger.log_metrics(profiler.get_all_metrics(), step, prefix="Profiler")
                metrics_logger.log_metrics(guardrail.get_incident_metrics(), step, prefix="Guardrail")

            profiler.record_step(transitions=batch_size, train_steps=1 if performed_train else 0)

            # Autonomous Execution Guardrail Supervision
            if step % 25 == 0 and step > 0:
                is_healthy = guardrail.check_system_health(step)
                if not is_healthy:
                    logger.critical("[!] ExecutionGuardrail triggered emergency abort. Halting training.")
                    break
                guardrail.check_step_latency(step, "mcts_search")
                guardrail.check_value_bounds(step, agent.last_root_value)

            if step % 10 == 0:
                pbar_dict = {}
                if latest_metrics:
                    pbar_dict["jepa"] = f"{latest_metrics.get('loss_jepa', 0.0):.3f}"
                    if "loss_reward" in latest_metrics:
                        pbar_dict["r_loss"] = f"{latest_metrics['loss_reward']:.3f}"
                if len(recent_ep_returns) > 0:
                    pbar_dict["ep_rew"] = f"{np.mean(recent_ep_returns):.1f}"
                pbar_dict["tps"] = f"{profiler.get_tps():.0f}"
                m_sys = system_monitor.get_metrics()
                temp = m_sys.get("gpu/temp_celsius", 0.0)
                pbar_dict["gpu"] = f"{temp:.0f}°C" if temp > 0 else "Active"
                vram_used = m_sys.get("gpu/physical_vram_used_mb", m_sys.get("gpu/vram_reserved_mb", 0.0)) / 1024.0
                pbar_dict["vram"] = f"{vram_used:.1f}G"
                pbar.set_postfix(pbar_dict)


            if step % 1000 == 0 and step > 0 and len(recent_ep_returns) > 0:
                logger.info(
                    f"Step {step:7d} | Ep Return: {np.mean(recent_ep_returns):6.2f} "
                    f"(min: {np.min(recent_ep_returns):6.1f}, max: {np.max(recent_ep_returns):6.1f}, "
                    f"std: {np.std(recent_ep_returns):6.2f}) | "
                    f"Losses: JEPA={latest_metrics.get('loss_jepa', 0.0):.4f}, "
                    f"Reward={latest_metrics.get('loss_reward', 0.0):.4f}, "
                    f"Value={latest_metrics.get('loss_value', 0.0):.4f}"
                )

            if step % cfg.training.save_freq == 0 and step > 0:
                if not latest_metrics:
                    loss_total = float('inf')
                else:
                    loss_total = float(latest_metrics.get("loss_jepa", float('inf')) + latest_metrics.get("loss_rl", float('inf')))
                checkpointer.save(step, models_dict, trainer.optimizer, loss_total, metrics=latest_metrics)
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    except KeyboardInterrupt:
        logger.info("\n[!] KeyboardInterrupt detected. Executing graceful shutdown...")
        if not latest_metrics:
            loss_total = float('inf')
        else:
            loss_total = float(latest_metrics.get("loss_jepa", float('inf')) + latest_metrics.get("loss_rl", float('inf')))
        checkpointer.save(step, models_dict, trainer.optimizer, loss_total, filename="interrupt_checkpoint.pt", metrics=latest_metrics)
        metrics_logger.close()
        logger.info("Graceful shutdown complete. Model saved. Exiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()