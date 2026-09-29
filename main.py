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
#      value bound; nothing domain-specific is configured here. The config -> run mapping
#      (domain builders, typed RunConfig) lives in ipomdp.experiments, shared with analyze.py
#      and sweep.py.
#
# 4. Failure Is Loud, Shutdown Is Graceful:
#    - Errors (including non-finite losses) propagate and stop the run. SIGINT writes
#      latest.pt, closes the TensorBoard writer and exits with status 130; a guardrail abort
#      writes latest.pt (the guardrail itself writes emergency.pt) and exits with status 1.
#      Only a run that completed every collection exits with 0, which is what sweep.py relies
#      on before analysing a run.
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
from tqdm import tqdm

from ipomdp.domain import FinitePOMDP
from ipomdp.experiments import build_training_run
from ipomdp.interpretability import build_probe_dataset
from ipomdp.telemetry import (
    BeliefGeometryVisualizer,
    ExecutionGuardrail,
    MCTSGraphVisualizer,
    MetricsLogger,
    PipelineProfiler,
    RewardTrajectoryVisualizer,
    SystemTelemetryMonitor,
)
from ipomdp.training import TrainingRun, load_checkpoint, play_episodes, save_checkpoint

torch.set_float32_matmul_precision('high')

def visualize(run: TrainingRun, pomdp: FinitePOMDP, collection: int, plots: Path, metrics_logger: MetricsLogger) -> None:
    """Belief geometry, search tree and cumulative rewards of greedy evaluation episodes."""
    episodes, _ = play_episodes(run.eval_env, run.eval_agent)
    dataset = build_probe_dataset(pomdp, run.world_model.belief_filter, episodes)
    figure = BeliefGeometryVisualizer(save_dir=str(plots / "geometry")).plot(
        dataset.latents, dataset.posteriors[:, 0], pomdp.state_names[0], filename=f"geometry_c{collection}")
    metrics_logger.log_figure("Visuals/belief_geometry", figure, collection)
    plt.close(figure)
    MCTSGraphVisualizer(save_dir=str(plots / "trees")).visualize(
        run.eval_agent.planner.roots[0], pomdp.discount, pomdp.action_names, pomdp.observation_names,
        filename=f"mcts_tree_c{collection}")
    action_map = dict(enumerate(pomdp.action_names))
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
    pomdp, run = build_training_run(cfg, device)
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
                save_checkpoint(checkpoints / "latest.pt", run.state_dict())
                metrics_logger.close()
                sys.exit(1)
            pbar.update(1)
        save_checkpoint(checkpoints / "latest.pt", run.state_dict())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt: saving latest.pt and shutting down.")
        save_checkpoint(checkpoints / "latest.pt", run.state_dict())
        metrics_logger.close()
        sys.exit(130)
    metrics_logger.close()


if __name__ == "__main__":
    main()
