# ABSOLUTE PATH: reports/2026-10-07_wumpus_reviewer_page/scripts/dump_latents.py
# ==============================================================================
# LATENTS AND EXACT-BELIEF LABELS FOR THE UMAP FIGURE (step 1 of 2)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. What is embedded:
#    - The ACTING filter's latents (what the planner uses) of a run's best.pt (the analysed
#      checkpoint), at every alive step of 256 greedy-planner games and 256 uniform-random games
#      on a fixed seed. Random histories cover situations the agent avoids; the agent's own
#      histories are what it actually plans from.
#    - Labels from the exact enumeration filter on the same histories: hunter cell and heading,
#      the exact danger of a forward step, log10 of the number of consistent layouts (how
#      uncertain the exact belief still is), the game step and the history source; plus the last
#      action (-1 before the first) and the percept that just arrived, to test whether the latent
#      is organised by its most recent input.
#    - Usage: uv run python <this> <run dir> <out .npz>; then umap_latents.py (step 2) runs UMAP in
#      a separate environment, so the project does not depend on umap-learn.
# ==============================================================================

import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from ipomdp.agents import PlanningAgent, UniformRandomAgent
from ipomdp.domain import WumpusSimulator
from ipomdp.domain.wumpus import GRID
from ipomdp.experiments.runs import build_training_run
from ipomdp.interpretability import build_wumpus_dataset
from ipomdp.planning import BeliefTreeSearch
from ipomdp.training import load_checkpoint, play_episodes, stream_seed

run_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
dev = torch.device("cuda")
cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
setup, run = build_training_run(cfg, dev)
best = load_checkpoint(run_dir / "checkpoints" / "best.pt")
for key, net in run.networks.items():
    net.load_state_dict(best["networks"][key])
world, model, n = setup.domain, run.eval_agent.model, 256
planner = PlanningAgent(model, BeliefTreeSearch(model, cfg.mcts.num_simulations, cfg.mcts.c_puct, cfg.mcts.dirichlet_alpha,
                                                dirichlet_epsilon=0.0, seed=stream_seed(31, 2),
                                                observation_prune=cfg.mcts.observation_prune),
                        n, temperature=0.0, seed=stream_seed(31, 3), device=dev)
parts = {}
for source, agent in (("agent", planner), ("random", UniformRandomAgent(world.num_actions, n, stream_seed(31, 4), dev))):
    episodes, _ = play_episodes(WumpusSimulator(world, n, cfg.env.max_steps, stream_seed(31, 1), dev), agent)
    d = build_wumpus_dataset(world, run.acting_filter, episodes)
    delta = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=dev)
    x, y = d.cell % GRID + delta[d.heading, 0], d.cell // GRID + delta[d.heading, 1]
    inside = (x >= 0) & (x < GRID) & (y >= 0) & (y < GRID)
    ahead = (x + GRID * y).clamp(0, 15)
    danger = torch.where(inside, d.hazards[:, :16].gather(1, ahead[:, None]).squeeze(1)
                         + d.wumpus_alive.double() * d.hazards[:, 16:32].gather(1, ahead[:, None]).squeeze(1), 0.0)
    rows_e, rows_t = d.episode, d.step
    last_action = torch.where(rows_t > 0, episodes.actions[rows_e, (rows_t - 1).clamp(min=0)], torch.full_like(rows_t, -1))
    percept = torch.where(rows_t > 0, episodes.observations[rows_e, (rows_t - 1).clamp(min=0)], episodes.initial_observations[rows_e])
    parts[source] = {"last_action": last_action.cpu().numpy(), "percept": percept.cpu().numpy(),"latents": d.latents.cpu().numpy(), "cell": d.cell.cpu().numpy(), "heading": d.heading.cpu().numpy(),
                     "danger": danger.cpu().numpy(), "log_layouts": d.log_layouts.cpu().numpy(), "step": d.step.cpu().numpy(),
                     "episode": d.episode.cpu().numpy()}
    print(f"{source}: {len(d.latents)} alive states")
arrays = {f"{k}_{s}": v for s, p in parts.items() for k, v in p.items()}
np.savez_compressed(out, **arrays)
print(f"wrote {out}")
