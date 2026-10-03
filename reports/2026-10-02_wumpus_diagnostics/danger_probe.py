# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/danger_probe.py
# ==============================================================================
# IS THE DANGER AHEAD DECODABLE FROM THE LATENT? (LATENT VS OBSERVATION-HEAD BOTTLENECK)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-02): the observation head's P(GAME_OVER | z, FORWARD) barely separates
#    provably safe steps from certain death (death_calibration.py). If a fresh MLP probe can
#    decode the exact danger ahead from the same latents, the head (its training signal) is the
#    bottleneck; if it cannot, the latent is.
#    - Usage: uv run python <this> <run dir> <games> <checkpoint name>. Reference-agent (risk
#      0.4) and random games, pooled; probe fitted on the first half of the games, binned
#      calibration and mean absolute error on the second half, next to the head's prediction.
# ==============================================================================

import sys
from pathlib import Path

import torch
import torch.nn as nn
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


def states(seed):
    out = []
    for agent in (WumpusReferenceAgent(world, n, 0.4), UniformRandomAgent(world.num_actions, n, stream_seed(seed, 2), dev)):
        d = build_wumpus_dataset(world, model.belief_filter,
                                 play_episodes(WumpusSimulator(world, n, cfg.env.max_steps, stream_seed(seed, 1), dev), agent)[0])
        x, y = d.cell % GRID + delta[d.heading, 0], d.cell // GRID + delta[d.heading, 1]
        inside = (x >= 0) & (x < GRID) & (y >= 0) & (y < GRID)
        ahead = (x + GRID * y).clamp(0, 15)[:, None]
        exact = torch.where(inside, d.hazards[:, :16].gather(1, ahead).squeeze(1)
                            + d.wumpus_alive.double() * d.hazards[:, 16:32].gather(1, ahead).squeeze(1), 0.0)
        out.append((d.latents, exact.float()))
    return torch.cat([z for z, _ in out]), torch.cat([e for _, e in out])


fit_z, fit_e = states(4321)
held_z, held_e = states(8765)
mean, std = fit_z.mean(0), fit_z.std(0) + 1e-6
torch.manual_seed(0)
probe = nn.Sequential(nn.Linear(fit_z.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 1)).to(dev)
optimizer = torch.optim.Adam(probe.parameters(), 1e-3)
generator = torch.Generator(device=dev).manual_seed(0)
for step in range(4000):
    rows = torch.randint(0, len(fit_z), (512,), device=dev, generator=generator)
    loss = F.binary_cross_entropy_with_logits(probe((fit_z[rows] - mean) / std).squeeze(1), fit_e[rows])
    optimizer.zero_grad(); loss.backward(); optimizer.step()
with torch.no_grad():
    decoded = torch.sigmoid(probe((held_z - mean) / std).squeeze(1))
    forward = F.one_hot(torch.zeros(len(held_z), dtype=torch.int64, device=dev), world.num_actions).float()
    head = F.softmax(model.observation_head(held_z, forward).float(), -1)[:, GAME_OVER]
print(f"{run_dir.name}/{name}: held-out states {len(held_z)}; mean |error| probe {(decoded - held_e).abs().mean():.3f}, "
      f"observation head {(head - held_e).abs().mean():.3f}, constant {(fit_e.mean() - held_e).abs().mean():.3f}")
for lo, hi in ((0, 1e-9), (1e-9, 0.4), (0.4, 0.6), (0.6, 1.0001)):
    m = (held_e >= lo) & (held_e < hi)
    print(f"  exact [{lo:.1g},{hi:.2g}) n={int(m.sum()):6d}: exact {held_e[m].mean():.3f}  probe {decoded[m].mean():.3f}  head {head[m].mean():.3f}")
