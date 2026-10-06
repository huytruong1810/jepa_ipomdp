# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/greedy_deaths.py
# ==============================================================================
# DANGER AND PREDICTED DEATH AT THE FORWARD STEPS THE GREEDY PLANNER TAKES
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-05, combined-fix run eps_decoder_k5_u256_seed0 dying in ~50% of games):
#    - death_calibration.py scores the head on reference and random histories. This script scores
#      it on the GREEDY PLANNER'S OWN histories, only at the FORWARD steps it actually takes:
#      exact danger (enumeration filter) vs the acting observation head's P(GAME_OVER | z, FORWARD).
#    - Usage: uv run python <this> <run dir> <checkpoint name>. The run's evaluation agent plays
#      eval_episodes games on a fixed seed.
#    - Result 2026-10-05 (latest.pt, ~collection 40): 28 FORWARDs at exact danger >= 0.6 and 55 at
#      0.4-0.6, all predicted at P(death) <= 0.01; death rate 0.47. The planner steps exactly where
#      the head wrongly says "safe" (optimizer's curse on a miscalibrated head). Older runs show the
#      same at a smaller scale (north50_decoder_seed0 34, north50_seed0 12 such steps).
# ==============================================================================

import sys
from pathlib import Path
import torch, torch.nn.functional as F
from ipomdp.domain import WumpusSimulator
from ipomdp.domain.wumpus import GAME_OVER, GRID
from ipomdp.experiments.runs import build_training_run
from ipomdp.interpretability import build_wumpus_dataset
from ipomdp.training import load_checkpoint, play_episodes, stream_seed
from omegaconf import OmegaConf
run_dir, name = Path(sys.argv[1]), sys.argv[2]
dev = torch.device("cuda"); cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
setup, run = build_training_run(cfg, dev)
for k, net in run.networks.items(): net.load_state_dict(load_checkpoint(run_dir / "checkpoints" / name)["networks"][k])
world, model, n = setup.domain, run.eval_agent.model, run.eval_agent.batch_size
ep, _ = play_episodes(WumpusSimulator(world, n, cfg.env.max_steps, stream_seed(4321, 1), dev), run.eval_agent)
d = build_wumpus_dataset(world, model.belief_filter, ep)
keep = d.step < ep.actions.shape[1]
import dataclasses
d = type(d)(**{k: getattr(d, k)[keep] for k in [f.name for f in dataclasses.fields(d)]})
delta = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=dev)
with torch.no_grad():
    x, y = d.cell % GRID + delta[d.heading, 0], d.cell // GRID + delta[d.heading, 1]
    inside = (x >= 0) & (x < GRID) & (y >= 0) & (y < GRID); ahead = (x + GRID * y).clamp(0, 15)
    exact = torch.where(inside, d.hazards[:, :16].gather(1, ahead[:, None]).squeeze(1)
                        + d.wumpus_alive.double() * d.hazards[:, 16:32].gather(1, ahead[:, None]).squeeze(1), 0.0)
    act = ep.actions[d.episode, d.step]
    pred = F.softmax(model.observation_head(d.latents, F.one_hot(torch.zeros_like(act), world.num_actions).float()).float(), -1)[:, GAME_OVER]
deaths = (ep.rewards <= -1000).any(1).float().mean(); wins = (ep.rewards >= 998).any(1).float().mean()
f = act == 0
print(f"{run_dir.name} {name}: games {n}, death {deaths:.2f}, win {wins:.2f}; alive steps {len(act)}, FORWARD {int(f.sum())}")
print("  action counts:", torch.bincount(act, minlength=5).tolist())
for lo, hi in ((0, 1e-9), (1e-9, 0.4), (0.4, 0.6), (0.6, 1.01)):
    m = f & (exact >= lo) & (exact < hi)
    print(f"  FORWARD taken at exact danger [{lo:.1g},{hi}): n={int(m.sum())}, predicted P(death) {pred[m].mean():.3f}")
first_fwd_step = [int(d.step[(d.episode == e) & f].min()) if ((d.episode == e) & f).any() else -1 for e in range(n)]
import collections; print("  step of first FORWARD (count):", sorted(collections.Counter(first_fwd_step).items())[:10])
