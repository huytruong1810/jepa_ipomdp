# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/reference_probe.py
# ==============================================================================
# PROBES OF THE LATENT ON REFERENCE-AGENT HISTORIES (LONG, DIRECTED EXCURSIONS)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-02): the agent-history probes (analyze.py) look good, but the planner's
#    own games rarely travel far. Does the latent still know the hunter's cell and heading, and
#    the danger ahead, on histories that do travel (the reference agent's)?
#    - Usage: uv run python <this> <run dir> <games> <checkpoint name>. MLP probe fitted on one
#      half of the games, evaluated on the other; the cell and heading majority baselines are
#      printed next to the accuracies.
# ==============================================================================

import sys
from pathlib import Path

import torch

from ipomdp.agents import WumpusReferenceAgent
from ipomdp.domain import WumpusSimulator
from ipomdp.experiments.runs import build_training_run
from ipomdp.interpretability import build_wumpus_dataset, evaluate_wumpus_probe, fit_mlp_wumpus_probe
from ipomdp.training import EpisodeBatch, load_checkpoint, play_episodes, stream_seed
from omegaconf import OmegaConf

run_dir, n, name = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
dev = torch.device("cuda")
cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
setup, run = build_training_run(cfg, dev)
for key, net in run.networks.items():
    net.load_state_dict(load_checkpoint(run_dir / "checkpoints" / name)["networks"][key])
world = setup.domain
episodes, _ = play_episodes(WumpusSimulator(world, n, cfg.env.max_steps, stream_seed(4321, 1), dev),
                            WumpusReferenceAgent(world, n, 0.4))
half = n // 2
pick = lambda rows: EpisodeBatch(episodes.actions[rows], episodes.observations[rows], episodes.rewards[rows],  # noqa: E731
                                 episodes.initial_observations[rows])
fit = build_wumpus_dataset(world, run.acting_filter, pick(slice(0, half)))
held = build_wumpus_dataset(world, run.acting_filter, pick(slice(half, None)))
report = evaluate_wumpus_probe(fit_mlp_wumpus_probe(fit), fit, held)
majority = lambda x: float(torch.bincount(x).max()) / len(x)  # noqa: E731
print(f"{run_dir.name}/{name} on reference histories ({report.rows} held-out states):")
print("  hazard error probe/constant: " + " ".join(f"{k} {report.hazard_error[k]:.4f}/{report.constant_error[k]:.4f}"
                                                 for k in report.hazard_error))
print(f"  cell acc {report.cell_accuracy:.3f} (majority {majority(held.cell):.3f}), "
      f"heading acc {report.heading_accuracy:.3f} (majority {majority(held.heading):.3f})")
