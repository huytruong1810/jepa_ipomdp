# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/supervised_filter.py
# ==============================================================================
# SUPERVISED UPPER BOUND FOR THE BELIEF-FILTER ARCHITECTURE ON WUMPUS WORLD
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (user decision, 2026-10-03):
#    - Neither JEPA nor the decoder makes the danger ahead decodable from the latent
#      (danger_probe.py). Is that the TRAINING OBJECTIVE or the ARCHITECTURE / CAPACITY?
#    - Train the very same BeliefFilter (the trained runs' latent_dim, hidden_dim, num_blocks)
#      directly on privileged exact targets from the enumeration filter. If it learns them well,
#      the architecture suffices and the objectives are the bottleneck; if it cannot, the
#      architecture (or its capacity) is.
#    - Diagnostic only: the targets are privileged, so this never becomes a training method.
#
# 2. Targets (per alive step, from WumpusBeliefTracker):
#    - Hazard marginals P(pit / wumpus / gold at c), 48 soft binary targets (BCE).
#    - Hunter cell (16 classes) and heading (4 classes), cross-entropy.
#    - Danger ahead P(pit ahead) + P(live wumpus ahead) (0 at a wall), soft binary target (BCE).
#      This is exactly what the observation head must know for FORWARD.
#    - A small MLP readout maps the latent to all targets; gradients flow into the filter.
#
# 3. Data and Evaluation:
#    - Reference-agent (risk 0.4) and uniformly random games in equal numbers (the same mix as
#      danger_probe.py), a fixed training pool and a disjoint held-out pool on other seeds.
#    - Reported on held-out alive states with the metrics of danger_probe.py and the
#      analysis probes: hazard error vs the constant, cell / heading accuracy, and the binned
#      calibration of the danger ahead, so the numbers sit next to the learned latents'.
#
# 4. Usage:
#    uv run python reports/2026-10-02_wumpus_diagnostics/supervised_filter.py [--steps 3000]
# ==============================================================================

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from ipomdp.agents import UniformRandomAgent, WumpusReferenceAgent
from ipomdp.domain import WumpusSimulator, WumpusWorld
from ipomdp.domain.wumpus import GRID, NUM_CELLS, WumpusBeliefTracker
from ipomdp.models import BeliefFilter
from ipomdp.training import play_episodes, stream_seed

DELTA = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]])          # E, N, W, S


def pool(world: WumpusWorld, games: int, length: int, seed: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Episodes plus exact per-step targets, shape (B, T + 1, ...), with an alive mask."""
    parts = []
    for agent in (WumpusReferenceAgent(world, games, 0.4),
                  UniformRandomAgent(world.num_actions, games, stream_seed(seed, 2), device)):
        episodes, _ = play_episodes(WumpusSimulator(world, games, length, stream_seed(seed, 1), device), agent)
        tracker, steps = WumpusBeliefTracker(world), []
        for start in range(0, games, 512):
            rows = slice(start, start + 512)
            steps.append(tracker.filter_episodes(episodes.initial_observations[rows], episodes.actions[rows],
                                                 episodes.observations[rows]))
        stack = lambda field: torch.cat([torch.stack([getattr(m, field) for m in chunk], 1) for chunk in steps])  # noqa: E731
        pit, wumpus, cell, heading = stack("pit"), stack("wumpus"), stack("cell"), stack("heading")
        delta = DELTA.to(device)
        x, y = cell % GRID + delta[heading, 0], cell // GRID + delta[heading, 1]
        inside = (x >= 0) & (x < GRID) & (y >= 0) & (y < GRID)
        ahead = (x + GRID * y).clamp(0, NUM_CELLS - 1).unsqueeze(-1)
        danger = torch.where(inside, pit.gather(2, ahead).squeeze(-1)
                             + stack("wumpus_alive").double() * wumpus.gather(2, ahead).squeeze(-1), 0.0)
        parts.append({"actions": episodes.actions, "observations": episodes.observations,
                      "initial": episodes.initial_observations, "alive": ~stack("done"),
                      "hazards": torch.cat([pit, wumpus, stack("gold")], -1).float(),
                      "cell": cell, "heading": heading, "danger": danger.float()})
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="runs/wumpus/north50_seed0/.hydra/config.yaml")
    parser.add_argument("--games", type=int, default=4096, help="Training games per behaviour policy.")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda")
    cfg = OmegaConf.load(args.config)
    world = WumpusWorld(device)
    length, num_a, num_o = cfg.env.max_steps, world.num_actions, world.num_observations
    t0 = time.time()
    train, held = pool(world, args.games, length, 2026, device), pool(world, 512, length, 2027, device)
    print(f"pools built in {time.time() - t0:.0f}s: {train['actions'].shape[0]} training, {held['actions'].shape[0]} held-out games")

    torch.manual_seed(0)
    belief_filter = BeliefFilter(num_a, num_o, cfg.model.latent_dim, cfg.model.hidden_dim, cfg.model.num_blocks).to(device)
    readout = nn.Sequential(nn.Linear(cfg.model.latent_dim, 256), nn.SiLU(),
                            nn.Linear(256, 3 * NUM_CELLS + NUM_CELLS + 4 + 1)).to(device)
    optimizer = torch.optim.Adam([*belief_filter.parameters(), *readout.parameters()], lr=3e-4)

    def forward(data, rows):
        latents = belief_filter.unroll(F.one_hot(data["actions"][rows], num_a).float(),
                                       F.one_hot(data["observations"][rows], num_o).float(),
                                       F.one_hot(data["initial"][rows], num_o).float())      # (b, T+1, D)
        alive = data["alive"][rows]
        out = readout(latents[alive])
        return out, {k: data[k][rows][alive] for k in ("hazards", "cell", "heading", "danger")}

    def losses(out, target):
        hazards, cell = out[:, :48], out[:, 48:64]
        heading, danger = out[:, 64:68], out[:, 68]
        return (F.binary_cross_entropy_with_logits(hazards, target["hazards"])
                + F.cross_entropy(cell, target["cell"]) + F.cross_entropy(heading, target["heading"])
                + F.binary_cross_entropy_with_logits(danger, target["danger"]))

    generator = torch.Generator(device=device).manual_seed(0)
    for step in range(1, args.steps + 1):
        rows = torch.randint(0, train["actions"].shape[0], (args.batch,), device=device, generator=generator)
        loss = losses(*forward(train, rows))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*belief_filter.parameters(), *readout.parameters()], 1.0)
        optimizer.step()
        if step % 500 == 0:
            print(f"step {step}: training loss {loss.item():.4f}", flush=True)

    with torch.no_grad():
        outs, targets = [], []
        for start in range(0, held["actions"].shape[0], 256):
            o, tgt = forward(held, slice(start, start + 256))
            outs.append(o); targets.append(tgt)
        out = torch.cat(outs)
        target = {k: torch.cat([t[k] for t in targets]) for k in targets[0]}
        hazards = torch.sigmoid(out[:, :48])
        constant = train["hazards"][train["alive"]].mean(0)
        names = ("pit", "wumpus", "gold")
        errors = " ".join(f"{n} {(hazards[:, 16 * i:16 * i + 16] - target['hazards'][:, 16 * i:16 * i + 16]).abs().mean():.4f}/"
                          f"{(constant[16 * i:16 * i + 16] - target['hazards'][:, 16 * i:16 * i + 16]).abs().mean():.4f}"
                          for i, n in enumerate(names))
        majority = lambda x: float(torch.bincount(x).max()) / len(x)  # noqa: E731
        cell_acc = float((out[:, 48:64].argmax(1) == target["cell"]).float().mean())
        heading_acc = float((out[:, 64:68].argmax(1) == target["heading"]).float().mean())
        danger = torch.sigmoid(out[:, 68])
        print(f"held-out alive states {len(danger)}")
        print(f"  hazard error supervised/constant: {errors}")
        print(f"  cell acc {cell_acc:.3f} (majority {majority(target['cell']):.3f}), "
              f"heading acc {heading_acc:.3f} (majority {majority(target['heading']):.3f})")
        print(f"  danger ahead mean |error| {(danger - target['danger']).abs().mean():.3f} "
              f"(constant {(train['danger'][train['alive']].mean() - target['danger']).abs().mean():.3f})")
        for lo, hi in ((0, 1e-9), (1e-9, 0.4), (0.4, 0.6), (0.6, 1.0001)):
            m = (target["danger"] >= lo) & (target["danger"] < hi)
            print(f"  exact [{lo:.1g},{hi:.2g}) n={int(m.sum()):6d}: exact {target['danger'][m].mean():.3f}  "
                  f"supervised {danger[m].mean():.3f}")


if __name__ == "__main__":
    main()
