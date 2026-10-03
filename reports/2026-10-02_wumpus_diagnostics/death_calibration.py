# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/death_calibration.py
# ==============================================================================
# CALIBRATION OF THE LEARNED DEATH PREDICTION AND OF Q(FORWARD)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-02, after the off-policy diagnostic made the planner die more):
#    - Does the observation head predict death correctly? The exact probability that FORWARD
#      kills is P(pit ahead) + P(live wumpus ahead) from the enumeration filter (0 at a wall).
#      The model's prediction is P_hat(GAME_OVER | z, FORWARD) from the acting observation head.
#    - Usage: uv run python <this> <run dir> <games> <checkpoint name, e.g. best.pt>. States come
#      from reference-agent (risk 0.4) and random games on the analysis seed, alive states only.
#    - Reported per exact-danger bin: mean exact danger vs mean predicted P(GAME_OVER), and the
#      learned one-step Q(FORWARD) - Q(TURN_LEFT) (positive = the model prefers moving).
# ==============================================================================

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from ipomdp.agents import UniformRandomAgent, WumpusReferenceAgent
from ipomdp.domain import WumpusSimulator
from ipomdp.domain.wumpus import GAME_OVER, GRID
from ipomdp.experiments.runs import build_training_run
from ipomdp.interpretability import build_wumpus_dataset
from ipomdp.training import load_checkpoint, play_episodes, stream_seed
from omegaconf import OmegaConf

run_dir, n, name = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
dev = torch.device("cuda")
cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
setup, run = build_training_run(cfg, dev)
for key, net in run.networks.items():
    net.load_state_dict(load_checkpoint(run_dir / "checkpoints" / name)["networks"][key])
world, model = setup.domain, run.eval_agent.model
delta = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=dev)
for label, agent in (("reference", WumpusReferenceAgent(world, n, 0.4)),
                     ("random", UniformRandomAgent(world.num_actions, n, stream_seed(4321, 2), dev))):
    episodes, _ = play_episodes(WumpusSimulator(world, n, cfg.env.max_steps, stream_seed(4321, 1), dev), agent)
    d = build_wumpus_dataset(world, model.belief_filter, episodes)
    with torch.no_grad():
        x, y = d.cell % GRID + delta[d.heading, 0], d.cell // GRID + delta[d.heading, 1]
        inside = (x >= 0) & (x < GRID) & (y >= 0) & (y < GRID)
        ahead = (x + GRID * y).clamp(0, 15)
        pit, wumpus = d.hazards[:, :16], d.hazards[:, 16:32]
        exact = torch.where(inside, pit.gather(1, ahead[:, None]).squeeze(1)
                            + d.wumpus_alive.double() * wumpus.gather(1, ahead[:, None]).squeeze(1), 0.0)
        forward = F.one_hot(torch.zeros(len(d.latents), dtype=torch.int64, device=dev), world.num_actions).float()
        predicted = F.softmax(model.observation_head(d.latents, forward).float(), -1)[:, GAME_OVER].double()
        q = torch.cat([(lambda e: e.rewards + world.discount * (e.observation_probs * e.next_values).sum(-1))(
            model.expand(d.latents[i:i + 4096])) for i in range(0, len(d.latents), 4096)]).double()
    print(f"== {name} on {label} games: {len(exact)} alive states")
    for lo, hi in ((0, 1e-9), (1e-9, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.0001)):
        m = (exact >= lo) & (exact < hi)
        if m.any():
            print(f"  exact danger [{lo:.1g},{hi:.2g}) n={int(m.sum()):6d}: exact {exact[m].mean():.3f}  "
                  f"predicted P(GAME_OVER) {predicted[m].mean():.3f}  Q(FORWARD)-Q(LEFT) {(q[m, 0] - q[m, 1]).mean():8.1f}")
