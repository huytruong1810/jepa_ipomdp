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
# 2. Strict Information Filtering Causality & Memory Storage:
#    - Advances recurrent belief state b_t = Filter(b_{t-1}, a_{t-1}, o_t) prior to
#      MCTS search tree action selection a_t ~ pi(a | b_t).
#    - When an environment channel truncates or terminates, retrieves the true terminal
#      observation o_T from infos["agent_0"]["terminal_obs"] to push to buffer.end_episode(),
#      preventing reset observation o_0 from corrupting terminal transition pairs.
#
# 3. Channel Isolation on Truncation/Reset:
#    - Zeroes prev_actions and invokes agent.reset_index(i) on terminated channels,
#      maintaining strict information isolation across asynchronous vector environments.
#
# 4. Graceful Shutdown & Checkpoint Integrity:
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
    register_env,
    make_env,
    register_extractor,
    make_extractor,
)
import ipomdp.telemetry.registry as registry

import ipomdp.envs
from ipomdp.envs import SyncVectorEnv

from ipomdp.models import (
    RecurrentContextEncoder,
    CausalRelationalPredictor,
    RecurrentJEPABase,
    ValueHead,
    RewardHead,
    DiscretePolicyHead,
)
from ipomdp.planning import DiscreteLatentOpenLoopSearch
from ipomdp.training import PrioritizedSequenceBuffer, DiscreteRecurrentIPOMDPTrainer
from ipomdp.agents import DiscreteJEPAAgent, StatelessAgent
from ipomdp.types import Action

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    logger = setup_logger("IPOMDP_Experiment")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device} | Vector Batch Size: {cfg.training.env_batch_size}")

    env_fn = lambda: registry.make_env(cfg.env.name, **dict(cfg.env.get("env_kwargs", {})))
    vec_env = SyncVectorEnv(env_fn, cfg.training.env_batch_size)
    action_map = {int(k): v for k, v in cfg.env.action_map.items()}

    extractor = registry.make_extractor(
        cfg.env.extractor_name,
        obs_dim=cfg.env.obs_dim,
        hidden_dim=cfg.model.latent_dim,
        num_objects=cfg.model.num_objects,
        num_blocks=cfg.model.num_blocks
    )

    encoder = RecurrentContextEncoder(
        extractor,
        action_dim=cfg.env.action_dim_i,
        latent_dim=cfg.model.latent_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_blocks=cfg.model.num_blocks
    )

    predictor = CausalRelationalPredictor(
        num_objects=cfg.model.num_objects,
        latent_dim=cfg.model.latent_dim,
        action_dim_i=cfg.env.action_dim_i,
        action_dim_j=cfg.env.action_dim_j,
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
        action_dim_i=cfg.env.action_dim_i,
        action_dim_j=cfg.env.action_dim_j,
        hidden_dim=cfg.model.hidden_dim,
        num_blocks=cfg.model.num_blocks
    ).to(device)

    opponent_head = DiscretePolicyHead(
        latent_dim=cfg.model.latent_dim,
        action_dim=cfg.env.action_dim_j,
        num_opponents=1,
        hidden_dim=cfg.model.hidden_dim,
        num_blocks=cfg.model.num_blocks
    ).to(device)

    # Optional PyTorch Compilation (Eager mode is default for minimal memory footprint)
    use_compile = getattr(cfg.training, "compile", False) and hasattr(torch, "compile") and device.type == "cuda"
    if use_compile:
        logger.info("Selective PyTorch compilation enabled...")
        compiled_jepa = torch.compile(jepa_model)
        compiled_value = torch.compile(value_head)
        compiled_reward = torch.compile(reward_head)
        compiled_opp = torch.compile(opponent_head)

        planner = DiscreteLatentOpenLoopSearch(
            jepa_model=compiled_jepa,
            value_head=compiled_value,
            reward_head=compiled_reward,
            opponent_head=compiled_opp,
            action_dim_i=cfg.env.action_dim_i,
            action_dim_j=cfg.env.action_dim_j,
            num_simulations=cfg.mcts.num_simulations,
            num_latent_obs=cfg.mcts.num_latent_obs
        )
        trainer = DiscreteRecurrentIPOMDPTrainer(
            jepa_model=compiled_jepa,
            value_head=compiled_value,
            reward_head=compiled_reward,
            opponent_head=compiled_opp,
            logger=logger,
            device=device,
            latent_dim=cfg.model.latent_dim,
            action_dim_i=cfg.env.action_dim_i,
            action_dim_j=cfg.env.action_dim_j,
            num_objects=cfg.model.num_objects,
            detach_belief_for_rl=False
        )
    else:
        planner = DiscreteLatentOpenLoopSearch(
            jepa_model=jepa_model,
            value_head=value_head,
            reward_head=reward_head,
            opponent_head=opponent_head,
            action_dim_i=cfg.env.action_dim_i,
            action_dim_j=cfg.env.action_dim_j,
            num_simulations=cfg.mcts.num_simulations,
            num_latent_obs=cfg.mcts.num_latent_obs
        )
        trainer = DiscreteRecurrentIPOMDPTrainer(
            jepa_model=jepa_model,
            value_head=value_head,
            reward_head=reward_head,
            opponent_head=opponent_head,
            logger=logger,
            device=device,
            latent_dim=cfg.model.latent_dim,
            action_dim_i=cfg.env.action_dim_i,
            action_dim_j=cfg.env.action_dim_j,
            num_objects=cfg.model.num_objects,
            detach_belief_for_rl=False
        )

    agents = {
        "agent_0": DiscreteJEPAAgent(
            agent_id="agent_0",
            planner=planner,
            latent_dim=cfg.model.latent_dim,
            action_dim=cfg.env.action_dim_i,
            device=device,
            num_objects=cfg.model.num_objects
        ),
        "agent_1": StatelessAgent(
            agent_id="agent_1",
            action_dim=cfg.env.action_dim_j,
            action_idx=0
        )
    }

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

    models_dict = {
        "jepa": jepa_model,
        "value": value_head,
        "reward": reward_head,
        "opponent": opponent_head
    }

    start_step = 0
    try:
        start_step = checkpointer.load(
            f"{cfg.env.name}_checkpoints/latest_checkpoint.pt",
            models_dict,
            trainer.optimizer,
            device
        )
    except Exception as e:
        logger.warning(f"Starting from step 0 (no checkpoint loaded): {e}")

    observations, infos = vec_env.reset()
    for agent in agents.values():
        agent.reset(batch_size=cfg.training.env_batch_size)

    prev_actions = {
        aid: Action(data=torch.zeros(
            cfg.training.env_batch_size,
            cfg.env.action_dim_i if aid == "agent_0" else cfg.env.action_dim_j,
            device=device
        ))
        for aid in agents
    }

    ep_returns = np.zeros(cfg.training.env_batch_size, dtype=np.float32)
    recent_ep_returns = deque(maxlen=100)
    ep_lengths = np.zeros(cfg.training.env_batch_size, dtype=np.int32)
    recent_ep_lengths = deque(maxlen=100)

    # Within-episode cumulative trajectory tracking
    active_trajectories = [
        {"cum_rewards": [0.0], "actions": [], "rewards": []}
        for _ in range(cfg.training.env_batch_size)
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
            agents["agent_0"].update_belief(observations["agent_0"], prev_actions["agent_0"])
            agents["agent_1"].update_belief(observations["agent_1"], prev_actions["agent_1"])

            if do_viz:
                viz_beliefs.append(agents["agent_0"].belief[0].clone().detach().unsqueeze(0))
                ts = infos["agent_0"]["true_state"]
                ts_val = ts.data[0] if isinstance(ts, State) else ts[0]
                viz_true_states.append(ts_val.clone().detach())



            actions = {aid: agent.act(observations[aid]) for aid, agent in agents.items()}

            if do_viz:
                viz_actions.append(int(actions["agent_0"].data[0].item()))
                if len(viz_actions) == 1 and hasattr(planner, 'root') and planner.root is not None:
                    try:
                        tree_viz.visualize(planner.root, filename=f"mcts_tree_ep{episodes_completed}")
                    except Exception as e:
                        logger.warning(f"MCTS tree visualization skipped: {e}")

            next_obs, rews, terms, truncs, infos = vec_env.step(actions)

            for i in range(cfg.training.env_batch_size):
                step_reward = float(rews["agent_0"][i].item())
                act_taken = int(actions["agent_0"].data[i].item())
                ep_returns[i] += step_reward
                ep_lengths[i] += 1

                active_trajectories[i]["actions"].append(act_taken)
                active_trajectories[i]["rewards"].append(step_reward)
                active_trajectories[i]["cum_rewards"].append(active_trajectories[i]["cum_rewards"][-1] + step_reward)

                buffer.push(
                    env_idx=i,
                    obs=observations["agent_0"].data[i],
                    act_i=actions["agent_0"].data[i],
                    act_j=actions["agent_1"].data[i],
                    reward=step_reward
                )

                if terms["agent_0"][i].item() or truncs["agent_0"][i].item():
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

                    term_obs = infos["agent_0"]["terminal_obs"][i]
                    buffer.end_episode(env_idx=i, final_obs=term_obs)

                    prev_actions["agent_0"].data[i].zero_()
                    prev_actions["agent_1"].data[i].zero_()

                    agents["agent_0"].reset_index(i)
                    agents["agent_1"].reset_index(i)

                    if i == 0:
                        if do_viz and len(viz_beliefs) > 1:
                            try:
                                latent_viz.plot_trajectory(
                                    beliefs=viz_beliefs,
                                    true_states=viz_true_states,
                                    actions=viz_actions,
                                    filename=f"trajectory_ep{episodes_completed}",
                                    value_head=value_head,
                                    action_map=action_map
                                )
                            except Exception as e:
                                logger.error(f"Latent trajectory visualization failed: {e}")

                        if do_viz and completed_trajectories:
                            try:
                                recent_trajs = list(completed_trajectories)[-min(16, len(completed_trajectories)):]
                                fig = reward_viz.plot_cumulative_rewards(
                                    trajectories=recent_trajs,
                                    filename="canonical_episode_cumulative_rewards",
                                    title_suffix=f"Step {step:,}, Ep {episodes_completed}",
                                    action_map=action_map
                                )
                                metrics_logger.log_figure("Visuals/within_episode_cumulative_reward", fig, step)
                                plt.close(fig)
                            except Exception as e:
                                logger.error(f"Reward trajectory visualization failed: {e}")

                        episodes_completed += 1
                        do_viz = (episodes_completed % cfg.training.viz_freq == 0) and (episodes_completed > 0)
                        viz_beliefs.clear()
                        viz_true_states.clear()
                        viz_actions.clear()

            observations = next_obs
            prev_actions = actions

            if buffer.tree.size >= cfg.training.batch_size and step % cfg.training.update_freq == 0:
                if device.type == "cuda":
                    torch.compiler.cudagraph_mark_step_begin()

                batch, is_weights, tree_indices = buffer.sample_sequence(cfg.training.batch_size)
                metrics, td_errors = trainer.train_sequence(batch, is_weights)
                buffer.update_priorities(tree_indices, td_errors)

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

                metrics["step_reward_mean"] = float(rews["agent_0"].mean().item())
                metrics["episodes_completed"] = episodes_completed

                act_data = actions["agent_0"].data.view(-1)
                for a_idx, a_name in action_map.items():
                    act_key = f"action_pct_{a_name.lower().replace(' ', '_')}"
                    metrics[act_key] = float((act_data == a_idx).float().mean().item())

                latest_metrics = metrics
                metrics_logger.log_metrics(metrics, step, prefix="Train")

            if step % 10 == 0:
                pbar_dict = {}
                if latest_metrics:
                    pbar_dict["jepa"] = f"{latest_metrics.get('loss_jepa', 0.0):.3f}"
                    if "loss_reward" in latest_metrics:
                        pbar_dict["r_loss"] = f"{latest_metrics['loss_reward']:.3f}"
                if len(recent_ep_returns) > 0:
                    pbar_dict["ep_rew"] = f"{np.mean(recent_ep_returns):.1f}"
                if pbar_dict:
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