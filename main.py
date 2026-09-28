# ABSOLUTE PATH: main.py
# ==============================================================================
# TRAINING ENTRY POINT
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Thin Shell Over TrainingRun:
#    - All experiment state and logic live in ipomdp.training.TrainingRun (build, collect,
#      train, evaluate, state_dict). This file only reads the Hydra config, wires telemetry,
#      schedules evaluation / checkpointing / visualisation, and handles SIGINT.
#
# 2. One Directory per Run:
#    - Every artifact of a run (checkpoints, TensorBoard events, plots, JSONL log) is written
#      under Hydra's run output directory, so runs never overwrite each other. A run is
#      continued bit-for-bit with `resume=<run dir>/checkpoints/latest.pt` (which restores the
#      replay buffer and every random generator, see training/run.py).
#
# 3. Single Source of Truth for the Domain:
#    - The FinitePOMDP supplies |A|, |O|, action/observation names, the discount and the
#      value bound; nothing domain-specific is configured here.
#
# 4. Failure Is Loud, Shutdown Is Graceful:
#    - Errors (including non-finite losses) propagate and stop the run. SIGINT writes
#      latest.pt and closes the TensorBoard writer.
# ==============================================================================

import logging
from pathlib import Path
import sys

import hydra
from hydra.core.hydra_config import HydraConfig
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ipomdp.domain import FinitePOMDP, build_tiger_pomdp
from ipomdp.telemetry import (
    ExecutionGuardrail,
    LatentSpaceVisualizer,
    MCTSGraphVisualizer,
    MetricsLogger,
    PipelineProfiler,
    RewardTrajectoryVisualizer,
    SystemTelemetryMonitor,
)
from ipomdp.training import (RunConfig, TrainerConfig, TrainingRun, load_checkpoint, play_episodes,
                             save_checkpoint)

torch.set_float32_matmul_precision('high')

# Domains selectable through conf/env/<name>.yaml. Each builder returns the exact model.
DOMAIN_BUILDERS = {"tiger": build_tiger_pomdp}


def build_run_config(cfg: DictConfig) -> RunConfig:
    """Flattens the Hydra config into the typed RunConfig."""
    t, m, a, s = cfg.training, cfg.model, cfg.agent, cfg.mcts
    return RunConfig(
        seed=cfg.seed, episode_length=cfg.env.max_steps, env_batch_size=t.env_batch_size,
        warmup_episodes=t.warmup_episodes, buffer_capacity=t.buffer_capacity, batch_size=t.batch_size,
        updates_per_collection=t.updates_per_collection, eval_episodes=t.eval_episodes,
        latent_dim=m.latent_dim, hidden_dim=m.hidden_dim, num_blocks=m.num_blocks, ema_momentum=m.ema_momentum,
        num_bins=m.num_bins, trainer=TrainerConfig(**cfg.trainer), num_simulations=s.num_simulations,
        c_puct=s.c_puct, dirichlet_alpha=s.dirichlet_alpha, dirichlet_epsilon=s.dirichlet_epsilon,
        temperature=a.temperature, temperature_min=a.temperature_min, temperature_decay=a.temperature_decay)


def visualize(run: TrainingRun, pomdp: FinitePOMDP, collection: int, plots: Path, metrics_logger: MetricsLogger) -> None:
    """Latent trajectory, search tree and cumulative rewards of greedy evaluation episodes."""
    episodes, states = play_episodes(run.eval_env, run.eval_agent, uniform=False)
    with torch.no_grad():
        latents = run.world_model.belief_filter.unroll(F.one_hot(episodes.actions[:1], pomdp.num_actions).float(),
                                                       F.one_hot(episodes.observations[:1], pomdp.num_observations).float())
    action_map = dict(enumerate(pomdp.action_names))
    LatentSpaceVisualizer(save_dir=str(plots / "latent")).plot_trajectory(
        beliefs=list(latents[0, :-1].unsqueeze(1)), true_states=list(states[0, :-1].unsqueeze(1)),
        actions=episodes.actions[0].tolist(), filename=f"trajectory_c{collection}", action_map=action_map)
    MCTSGraphVisualizer(save_dir=str(plots / "trees")).visualize(
        run.eval_agent.planner.roots[0], pomdp.discount, pomdp.action_names, pomdp.observation_names,
        filename=f"mcts_tree_c{collection}")
    trajectories = [{"actions": acts, "rewards": rews, "cum_rewards": np.concatenate([[0.0], np.cumsum(rews)]).tolist()}
                    for acts, rews in zip(episodes.actions[:16].tolist(), episodes.rewards[:16].tolist())]
    figure = RewardTrajectoryVisualizer(save_dir=str(plots)).plot_cumulative_rewards(
        trajectories=trajectories, filename=f"eval_cumulative_rewards_c{collection}",
        title_suffix=f"Collection {collection:,}", action_map=action_map)
    metrics_logger.log_figure("Visuals/eval_cumulative_reward", figure, collection)
    plt.close(figure)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    logger = logging.getLogger("ipomdp")  # Hydra writes it to the console and <run dir>/main.log
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pomdp = DOMAIN_BUILDERS[cfg.env.name]()
    run = TrainingRun(pomdp, build_run_config(cfg), device)
    if cfg.resume is not None:
        run.load_state_dict(load_checkpoint(Path(cfg.resume)))
        logger.info(f"Resumed from {cfg.resume} at collection {run.collection}")
    logger.info(f"Run directory: {run_dir} | device: {device} | seed: {cfg.seed}")

    checkpoints = run_dir / "checkpoints"
    metrics_logger = MetricsLogger(log_dir=str(run_dir / "tensorboard"))
    system_monitor = SystemTelemetryMonitor(thermal_threshold_c=cfg.guardrail.thermal_trip_c)
    profiler = PipelineProfiler(window_size=100)
    guardrail = ExecutionGuardrail(
        logger=logger, system_monitor=system_monitor, thermal_trip_c=cfg.guardrail.thermal_trip_c,
        thermal_recovery_c=cfg.guardrail.thermal_recovery_c, vram_trip_mb=cfg.guardrail.vram_trip_mb,
        rss_trip_mb=cfg.guardrail.rss_trip_mb,
        emergency_save_fn=lambda: save_checkpoint(checkpoints / "emergency.pt", run.state_dict()))

    t = cfg.training
    total_collections = t.total_episodes // t.env_batch_size
    transitions_per_collection = t.env_batch_size * cfg.env.max_steps
    pbar = tqdm(total=total_collections, initial=run.collection, desc="Collections")
    try:
        while run.collection < total_collections:
            with profiler.profile("collect_and_train"):
                metrics = run.collect_and_train()
            profiler.record_step(transitions=transitions_per_collection, train_steps=t.updates_per_collection)
            collection = run.collection
            metrics_logger.log_metrics(metrics, collection, prefix="Train")

            if collection % t.eval_every == 0:
                with profiler.profile("evaluate"):
                    evaluation = run.evaluate()
                metrics_logger.log_metrics(evaluation, collection, prefix="Eval")
                logger.info(f"collection {collection}: greedy eval discounted return "
                            f"{evaluation['eval_discounted_return']:.2f} +- {evaluation['eval_discounted_return_stderr']:.2f} "
                            f"(best {run.best_eval_return:.2f})")
                if evaluation["eval_discounted_return"] >= run.best_eval_return:
                    save_checkpoint(checkpoints / "best.pt", {
                        "collection": collection, "eval_return": evaluation["eval_discounted_return"],
                        "networks": {name: net.state_dict() for name, net in run.networks.items()}})
                pbar.set_postfix({"eval": f"{evaluation['eval_discounted_return']:.2f}",
                                  "best": f"{run.best_eval_return:.2f}", "loss": f"{metrics['loss_total']:.3f}"})
            if collection % t.viz_every == 0:
                visualize(run, pomdp, collection, run_dir / "plots", metrics_logger)
            if collection % t.save_every == 0:
                save_checkpoint(checkpoints / "latest.pt", run.state_dict())

            metrics_logger.log_metrics(system_monitor.get_metrics(), collection, prefix="System")
            metrics_logger.log_metrics(profiler.get_all_metrics(), collection, prefix="Profiler")
            metrics_logger.log_metrics(guardrail.get_incident_metrics(), collection, prefix="Guardrail")
            if not guardrail.check_system_health(collection):
                logger.critical("[!] ExecutionGuardrail triggered emergency abort. Halting training.")
                break
            pbar.update(1)
        save_checkpoint(checkpoints / "latest.pt", run.state_dict())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt: saving latest.pt and shutting down.")
        save_checkpoint(checkpoints / "latest.pt", run.state_dict())
        metrics_logger.close()
        sys.exit(0)
    metrics_logger.close()


if __name__ == "__main__":
    main()
