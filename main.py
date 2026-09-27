# ABSOLUTE PATH: main.py
# ==============================================================================
# TRAINING ENTRY POINT: EPISODE-MAJOR COLLECTION + WHOLE-EPISODE WORLD-MODEL UPDATES
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Canonical Interaction Timing:
#    - The domain (src/ipomdp/domain) emits no observation at reset. Every collection:
#      env.reset(); agent.reset() (z = z_0); then for each step
#         a_t = agent.act()  ->  (o_{t+1}, r_t) = env.step(a_t)  ->  agent.update(a_t, o_{t+1}).
#    - All env_batch_size episodes start together and truncate together after
#      env.max_steps steps, so episodes are collected whole and stored whole
#      (src/ipomdp/training/episode_buffer.py); the trainer never sees partial histories.
#
# 2. Single Source of Truth for the Domain:
#    - |A|, |O|, action names and the discount gamma are read from the FinitePOMDP; gamma is
#      passed explicitly to both the planner and the trainer so neither can disagree with
#      the exact benchmark solver.
#
# 3. Single-Agent Bridge:
#    - The canonical single-agent POMDP has no opponent, but the world model, heads and
#      planner still carry opponent inputs (reviewed in Phases 3-4). They receive a singleton
#      opponent action space {0} (OPPONENT_ACTION_DIM = 1), which carries no information.
#
# 4. Selective Compilation:
#    - With training.compile on CUDA, the fixed-shape networks are compiled while MCTS
#      control flow stays eager. The compiled callables are used for search and training;
#      the uncompiled modules are what gets checkpointed.
#
# 5. Failure Is Loud, Shutdown Is Graceful:
#    - Errors (including non-finite losses and visualisation failures) propagate and stop
#      the run. SIGINT saves interrupt_checkpoint.pt and closes the TensorBoard writer.
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

from ipomdp.agents import DiscreteJEPAAgent
from ipomdp.domain import BatchedPOMDPEnv, FinitePOMDP, build_tiger_pomdp
from ipomdp.models import (
    BeliefFilter,
    LatentPredictor,
    ObservationHead,
    OpponentPolicyHead,
    TwoHotSymlog,
    RecurrentJEPA,
    RewardHead,
    ValueHead,
)
from ipomdp.planning import LatentBeliefTreeSearch
from ipomdp.telemetry import (
    ExecutionGuardrail,
    LatentSpaceVisualizer,
    MCTSGraphVisualizer,
    MetricsLogger,
    ModelCheckpointer,
    PipelineProfiler,
    RewardTrajectoryVisualizer,
    SystemTelemetryMonitor,
    setup_logger,
)
from ipomdp.training import EpisodeBatch, EpisodeBuffer, TrainerConfig, WorldModelTrainer

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

# Domains selectable through conf/env/<name>.yaml. Each builder returns the exact model.
DOMAIN_BUILDERS = {"tiger": build_tiger_pomdp}

# Singleton opponent action space for the single-agent POMDP (module header, section 3).
OPPONENT_ACTION_DIM = 1


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    logger = setup_logger("IPOMDP_Experiment")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    logger.info(f"Compute Device: {device} | Parallel episodes: {cfg.training.env_batch_size} | Seed: {cfg.seed}")

    pomdp: FinitePOMDP = DOMAIN_BUILDERS[cfg.env.name]()
    num_obs, num_actions = pomdp.num_observations, pomdp.num_actions
    action_map = dict(enumerate(pomdp.action_names))
    batch_size = cfg.training.env_batch_size
    episode_length = cfg.env.max_steps
    env = BatchedPOMDPEnv(pomdp, batch_size, max_steps=episode_length, seed=cfg.seed, device=device)

    m = cfg.model
    world_model = RecurrentJEPA(
        BeliefFilter(num_actions, num_obs, m.latent_dim, m.hidden_dim, m.num_blocks),
        LatentPredictor(m.latent_dim, num_actions, OPPONENT_ACTION_DIM, m.hidden_dim, m.num_blocks),
        ema_momentum=m.ema_momentum,
    ).to(device)
    value_head = ValueHead(m.latent_dim, m.hidden_dim, m.num_blocks, m.num_bins).to(device)
    reward_head = RewardHead(m.latent_dim, num_actions, OPPONENT_ACTION_DIM, m.hidden_dim, m.num_blocks,
                             m.num_bins).to(device)
    opponent_head = OpponentPolicyHead(m.latent_dim, OPPONENT_ACTION_DIM, m.hidden_dim, m.num_blocks).to(device)
    observation_head = ObservationHead(m.latent_dim, num_actions, OPPONENT_ACTION_DIM, num_obs, m.hidden_dim,
                                       m.num_blocks).to(device)
    codec = TwoHotSymlog(m.num_bins, pomdp.value_bound).to(device)  # shared by planner and trainer

    use_compile = cfg.training.compile and device.type == "cuda"
    networks = (world_model, value_head, reward_head, opponent_head, observation_head)
    if use_compile:
        logger.info("Selective PyTorch compilation enabled...")
        networks = tuple(torch.compile(net) for net in networks)
    run_world_model, run_value, run_reward, run_opponent, run_observation = networks

    planner = LatentBeliefTreeSearch(
        world_model=run_world_model,
        value_head=run_value,
        reward_head=run_reward,
        opponent_head=run_opponent,
        observation_head=run_observation,
        codec=codec,
        action_dim_i=num_actions,
        action_dim_j=OPPONENT_ACTION_DIM,
        num_simulations=cfg.mcts.num_simulations,
        num_observation_samples=cfg.mcts.num_observation_samples,
        discount=pomdp.discount,
    )
    trainer = WorldModelTrainer(
        world_model=world_model,
        value_head=value_head,
        reward_head=reward_head,
        opponent_head=opponent_head,
        observation_head=observation_head,
        codec=codec,
        num_actions=num_actions,
        num_observations=num_obs,
        num_opponent_actions=OPPONENT_ACTION_DIM,
        discount=pomdp.discount,
        config=TrainerConfig(**cfg.trainer),
        device=device,
    )
    agent = DiscreteJEPAAgent(
        belief_filter=world_model.belief_filter,
        planner=planner,
        batch_size=batch_size,
        num_actions=num_actions,
        num_observations=num_obs,
        device=device,
        temperature=cfg.agent.temperature,
        temperature_min=cfg.agent.temperature_min,
        temperature_decay=cfg.agent.temperature_decay,
    )
    buffer = EpisodeBuffer(cfg.training.buffer_capacity, episode_length, device, seed=cfg.seed)

    checkpointer = ModelCheckpointer(f"{cfg.env.name}_checkpoints", logger)
    metrics_logger = MetricsLogger(log_dir=f"{cfg.env.name}_tensorboard", experiment_name="jepa_ipomdp")
    latent_viz = LatentSpaceVisualizer(save_dir=f"{cfg.env.name}_plots/latent")
    tree_viz = MCTSGraphVisualizer(save_dir=f"{cfg.env.name}_plots/trees")
    reward_viz = RewardTrajectoryVisualizer(save_dir=f"{cfg.env.name}_plots")
    system_monitor = SystemTelemetryMonitor(thermal_threshold_c=82.0)
    profiler = PipelineProfiler(window_size=100)

    models_dict = {"world_model": world_model, "value": value_head, "reward": reward_head, "opponent": opponent_head,
                   "observation": observation_head}
    current_collection = [0]
    guardrail = ExecutionGuardrail(
        logger=logger,
        system_monitor=system_monitor,
        profiler=profiler,
        thermal_trip_c=82.0,
        thermal_recovery_c=72.0,
        vram_trip_mb=13500.0,
        rss_trip_mb=18000.0,
        emergency_save_fn=lambda: checkpointer.save(
            current_collection[0], models_dict, trainer.optimizer, float('inf'),
            filename="emergency_guardrail_checkpoint.pt"),
    )

    start_collection = 0
    latest_path = Path(f"{cfg.env.name}_checkpoints/latest_checkpoint.pt")
    if latest_path.exists():
        start_collection = checkpointer.load(str(latest_path), models_dict, trainer.optimizer, device)

    total_collections = cfg.training.total_episodes // batch_size
    no_opponent_action = torch.zeros(batch_size, episode_length, dtype=torch.int64, device=device)
    recent_returns = deque(maxlen=100)
    recent_trajectories = deque(maxlen=64)
    latest_metrics: dict[str, float] = {}

    logger.info("Entering training loop...")
    collection = start_collection
    try:
        pbar = tqdm(range(start_collection, total_collections), desc="Collections", initial=start_collection,
                    total=total_collections)
        for collection in pbar:
            current_collection[0] = collection
            episodes_so_far = collection * batch_size
            do_viz = collection % cfg.training.viz_every == 0 and collection > 0
            warmup = episodes_so_far < cfg.training.warmup_episodes

            # ---------------- Collect env_batch_size complete episodes ----------------
            env.reset()
            agent.reset()
            actions, observations, rewards = [], [], []
            viz_beliefs, viz_states = [], []
            for t in range(episode_length):
                if do_viz:
                    viz_beliefs.append(agent.belief[0:1].clone())
                    viz_states.append(env.state[0:1])
                with profiler.profile("mcts_search"):
                    action = agent.act_uniformly() if warmup else agent.act()
                if do_viz and t == 0 and not warmup:
                    tree_viz.visualize(planner.root, filename=f"mcts_tree_c{collection}")
                with profiler.profile("env_step"):
                    out = env.step(action)
                with profiler.profile("update_belief"):
                    agent.update(action, out.observation)
                actions.append(action)
                observations.append(out.observation)
                rewards.append(out.reward)
            if not bool(out.truncated.all()):
                raise RuntimeError("Episodes must truncate exactly at env.max_steps.")

            episodes = EpisodeBatch(torch.stack(actions, 1), torch.stack(observations, 1),
                                    torch.stack(rewards, 1), no_opponent_action)
            buffer.add(episodes)
            if not warmup:
                agent.anneal_temperature()

            episode_returns = episodes.rewards.sum(dim=1).tolist()
            recent_returns.extend(episode_returns)
            action_lists, reward_lists = episodes.actions.tolist(), episodes.rewards.tolist()
            for acts, rews in zip(action_lists, reward_lists):
                recent_trajectories.append(
                    {"actions": acts, "rewards": rews, "cum_rewards": np.concatenate([[0.0], np.cumsum(rews)]).tolist()})

            if do_viz:
                latent_viz.plot_trajectory(beliefs=viz_beliefs, true_states=viz_states, actions=action_lists[0],
                                           filename=f"trajectory_c{collection}", value_head=value_head,
                                           action_map=action_map)
                fig = reward_viz.plot_cumulative_rewards(
                    trajectories=list(recent_trajectories)[-16:], filename="canonical_episode_cumulative_rewards",
                    title_suffix=f"Collection {collection:,}", action_map=action_map)
                metrics_logger.log_figure("Visuals/within_episode_cumulative_reward", fig, collection)
                plt.close(fig)

            # ---------------- Gradient updates on whole episodes ----------------
            for _ in range(cfg.training.updates_per_collection):
                if use_compile:
                    torch.compiler.cudagraph_mark_step_begin()
                with profiler.profile("train_step"):
                    latest_metrics = trainer.train_step(buffer.sample(cfg.training.batch_size))
            profiler.record_step(transitions=batch_size * episode_length,
                                 train_steps=cfg.training.updates_per_collection)

            metrics = dict(latest_metrics)
            metrics["episode_return_mean"] = float(np.mean(recent_returns))
            metrics["episode_return_std"] = float(np.std(recent_returns))
            metrics["agent_temperature"] = agent.temperature
            flat_actions = episodes.actions.reshape(-1)
            for a_idx, a_name in action_map.items():
                metrics[f"action_pct_{a_name.lower()}"] = float((flat_actions == a_idx).float().mean())
            if not warmup:
                metrics["mcts_avg_depth"] = planner.last_avg_depth
                metrics["mcts_max_depth"] = planner.last_max_depth
                metrics["mcts_q_spread"] = planner.last_q_spread
                metrics["mcts_entropy"] = planner.last_entropy
            metrics_logger.log_metrics(metrics, collection, prefix="Train")
            metrics_logger.log_metrics(system_monitor.get_metrics(), collection, prefix="System")
            metrics_logger.log_metrics(profiler.get_all_metrics(), collection, prefix="Profiler")
            metrics_logger.log_metrics(guardrail.get_incident_metrics(), collection, prefix="Guardrail")

            if not guardrail.check_system_health(collection):
                logger.critical("[!] ExecutionGuardrail triggered emergency abort. Halting training.")
                break
            guardrail.check_step_latency(collection, "mcts_search")
            guardrail.check_value_bounds(collection, agent.last_root_value)

            pbar.set_postfix({
                "return": f"{metrics['episode_return_mean']:.1f}",
                "loss": f"{latest_metrics['loss_total']:.3f}",
                "tps": f"{profiler.get_tps():.0f}",
            })

            if collection % cfg.training.save_every == 0 and collection > 0:
                checkpointer.save(collection, models_dict, trainer.optimizer, latest_metrics["loss_total"],
                                  metrics=latest_metrics)
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    except KeyboardInterrupt:
        logger.info("\n[!] KeyboardInterrupt detected. Executing graceful shutdown...")
        checkpointer.save(collection, models_dict, trainer.optimizer,
                          latest_metrics.get("loss_total", float('inf')),
                          filename="interrupt_checkpoint.pt", metrics=latest_metrics)
        metrics_logger.close()
        logger.info("Graceful shutdown complete. Model saved. Exiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()
