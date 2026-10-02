# ABSOLUTE PATH: src/ipomdp/interpretability/wumpus_latents.py
# ==============================================================================
# WUMPUS WORLD: WHAT THE LEARNED LATENT KNOWS, AGAINST THE EXACT ENUMERATION FILTER
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Ground Truth Without a Solver:
#    - Wumpus World has no exact solver here (user decision), but its exact posterior over the
#      16,380 layouts is computable by enumeration (domain/wumpus.py, section 5). The learned
#      latent is probed against that posterior's marginals -- P(pit at c), P(wumpus at c),
#      P(gold at c) for the 16 cells -- and against the hunter's own state (cell, heading,
#      arrow, wumpus life), which is a deterministic function of the history.
#    - Only histories whose game is still running are used: after GAME_OVER the belief no
#      longer matters for any decision.
#
# 2. One Probe Decodes Everything:
#    - A probe maps a latent to 48 hazard logits (sigmoid; binary cross-entropy against the
#      exact marginals, a proper scoring rule, so the optimum is the exact marginal), 16 cell
#      logits and 4 heading logits (softmax) and 2 binary logits (arrow, wumpus alive).
#    - Linear probe (full-batch L-BFGS): "linearly readable". MLP probe (two layers, Adam): an
#      upper bound on the information present. Fit and evaluation use disjoint episode sets.
#
# 3. Metrics (held-out):
#    - Hazard marginals: mean |P_exact - P_probe| per object (pit, wumpus, gold), and the same
#      error of the best CONSTANT predictor (the data set's mean marginal per cell). The
#      ratio says how much of the history-dependent belief the probe recovers; Tiger's KL/L1
#      metrics do not transfer because the marginals are not one categorical distribution.
#    - Own state: accuracy of cell, heading, arrow and wumpus-life decoding.
#
# 4. Figures:
#    - belief_maps: one game, several time steps; rows are exact and MLP-decoded 4x4 maps of
#      P(pit), P(wumpus), P(gold), with the hunter's cell and heading drawn on every map.
#    - latent_pca: all latents in one PCA plane, coloured by the hunter's cell and by the
#      remaining uncertainty (log10 of the number of consistent layouts).
# ==============================================================================

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..domain.wumpus import NUM_CELLS, WumpusBeliefTracker, WumpusWorld
from ..models.world_model import BeliefFilter
from ..training.episode_buffer import EpisodeBatch

HAZARDS = ("pit", "wumpus", "gold")
_HEADINGS = 4


@dataclass(frozen=True)
class WumpusLatentDataset:
    """Latents of running games and their exact targets; N rows (section 1)."""

    latents: Tensor          # (N, D) float32
    hazards: Tensor          # (N, 48) float64: P(pit), P(wumpus), P(gold) per cell
    cell: Tensor             # (N,) int64
    heading: Tensor          # (N,) int64
    arrow: Tensor            # (N,) bool
    wumpus_alive: Tensor     # (N,) bool
    log_layouts: Tensor      # (N,) float64: log10 of the number of consistent layouts
    episode: Tensor          # (N,) int64 row of the episode batch
    step: Tensor             # (N,) int64 time t of the history h_t


@torch.no_grad()
def build_wumpus_dataset(world: WumpusWorld, belief_filter: BeliefFilter, episodes: EpisodeBatch,
                         chunk: int = 128) -> WumpusLatentDataset:
    """Replays episodes through the exact enumeration filter and the learned filter (section 1)."""
    if episodes.initial_observations is None:
        raise ValueError("Wumpus episodes carry the initial observation o_0.")
    tracker = WumpusBeliefTracker(world)
    num_a, num_o = world.num_actions, world.num_observations
    parts: list[dict[str, Tensor]] = []
    for start in range(0, episodes.actions.shape[0], chunk):
        rows = slice(start, start + chunk)
        actions, observations = episodes.actions[rows], episodes.observations[rows]
        initial = episodes.initial_observations[rows]
        latents = belief_filter.unroll(F.one_hot(actions, num_a).float(), F.one_hot(observations, num_o).float(),
                                       F.one_hot(initial, num_o).float())                    # (b, T+1, D)
        exact = tracker.filter_episodes(initial, actions, observations)                       # T+1 marginals
        batch = actions.shape[0]
        for t, m in enumerate(exact):
            running = ~m.done
            parts.append({
                "latents": latents[:, t][running].float(),
                "hazards": torch.cat([m.pit, m.wumpus, m.gold], dim=1)[running],
                "cell": m.cell[running], "heading": m.heading[running], "arrow": m.arrow[running],
                "wumpus_alive": m.wumpus_alive[running],
                "log_layouts": m.consistent_layouts[running].double().log10(),
                "episode": (torch.arange(batch, device=running.device) + start)[running],
                "step": torch.full((int(running.sum()),), t, dtype=torch.int64, device=running.device),
            })
    return WumpusLatentDataset(**{key: torch.cat([p[key] for p in parts]) for key in parts[0]})


class WumpusProbe(nn.Module):
    """Latent -> hazard marginals, cell, heading, arrow, wumpus life (section 2)."""

    OUTPUTS = 3 * NUM_CELLS + NUM_CELLS + _HEADINGS + 2

    def __init__(self, network: nn.Module, mean: Tensor, std: Tensor):
        super().__init__()
        self.network = network
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def logits(self, latents: Tensor) -> dict[str, Tensor]:
        out = self.network((latents.float() - self.mean) / self.std)
        hazards, cell, heading, binary = out.split([3 * NUM_CELLS, NUM_CELLS, _HEADINGS, 2], dim=-1)
        return {"hazards": hazards, "cell": cell, "heading": heading, "arrow": binary[:, 0], "alive": binary[:, 1]}

    @torch.no_grad()
    def hazards(self, latents: Tensor) -> Tensor:
        """Decoded P(pit), P(wumpus), P(gold) per cell, (N, 48) float64."""
        return torch.sigmoid(self.logits(latents)["hazards"]).double()


def _loss(probe: WumpusProbe, data: WumpusLatentDataset) -> Tensor:
    out = probe.logits(data.latents)
    return (F.binary_cross_entropy_with_logits(out["hazards"], data.hazards.float())
            + F.cross_entropy(out["cell"], data.cell) + F.cross_entropy(out["heading"], data.heading)
            + F.binary_cross_entropy_with_logits(out["arrow"], data.arrow.float())
            + F.binary_cross_entropy_with_logits(out["alive"], data.wumpus_alive.float()))


def _standardisation(latents: Tensor) -> tuple[Tensor, Tensor]:
    return latents.mean(0), latents.std(0).clamp(min=1e-6)


def fit_linear_wumpus_probe(data: WumpusLatentDataset) -> WumpusProbe:
    """Linear probe, full-batch L-BFGS (section 2)."""
    probe = WumpusProbe(nn.Linear(data.latents.shape[1], WumpusProbe.OUTPUTS), *_standardisation(data.latents))
    probe.to(data.latents.device)
    optimizer = torch.optim.LBFGS(probe.parameters(), max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = _loss(probe, data)
        loss.backward()
        return loss

    optimizer.step(closure)
    return probe


def fit_mlp_wumpus_probe(data: WumpusLatentDataset, steps: int = 3000, hidden: int = 256,
                         seed: int = 0) -> WumpusProbe:
    """Two-layer MLP probe, Adam, full batch (section 2)."""
    torch.manual_seed(seed)
    network = nn.Sequential(nn.Linear(data.latents.shape[1], hidden), nn.ELU(), nn.Linear(hidden, WumpusProbe.OUTPUTS))
    probe = WumpusProbe(network, *_standardisation(data.latents)).to(data.latents.device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=3e-3)
    for _ in range(steps):
        optimizer.zero_grad()
        _loss(probe, data).backward()
        optimizer.step()
    return probe


@dataclass(frozen=True)
class WumpusProbeReport:
    """Held-out probe quality (section 3)."""

    rows: int
    hazard_error: dict[str, float]            # mean |P_exact - P_probe| per object
    constant_error: dict[str, float]          # same error of the best constant predictor (fit-set mean)
    hazard_error_p99: dict[str, float]        # 99th percentile of the per-cell absolute error
    cell_accuracy: float
    heading_accuracy: float
    arrow_accuracy: float
    alive_accuracy: float


@torch.no_grad()
def evaluate_wumpus_probe(probe: WumpusProbe, fit: WumpusLatentDataset,
                          held_out: WumpusLatentDataset) -> WumpusProbeReport:
    """Scores a probe on held-out histories against exact targets and the constant baseline."""
    decoded = probe.hazards(held_out.latents)
    constant = fit.hazards.mean(0, keepdim=True)
    out = probe.logits(held_out.latents)
    error, baseline, p99 = {}, {}, {}
    for k, name in enumerate(HAZARDS):
        cells = slice(k * NUM_CELLS, (k + 1) * NUM_CELLS)
        absolute = (decoded[:, cells] - held_out.hazards[:, cells]).abs()
        error[name] = float(absolute.mean())
        p99[name] = float(absolute.flatten().quantile(0.99)) if absolute.numel() < 2 ** 24 else float("nan")
        baseline[name] = float((constant[:, cells] - held_out.hazards[:, cells]).abs().mean())
    accuracy = lambda logits, target: float((logits.argmax(-1) == target).double().mean())  # noqa: E731
    binary = lambda logits, target: float(((logits > 0) == target).double().mean())  # noqa: E731
    return WumpusProbeReport(int(held_out.latents.shape[0]), error, baseline, p99,
                             accuracy(out["cell"], held_out.cell), accuracy(out["heading"], held_out.heading),
                             binary(out["arrow"], held_out.arrow), binary(out["alive"], held_out.wumpus_alive))


_ARROWS = {0: (0.32, 0.0), 1: (0.0, 0.32), 2: (-0.32, 0.0), 3: (0.0, -0.32)}   # E, N, W, S in grid units


def plot_belief_maps(data: WumpusLatentDataset, probe: WumpusProbe, episode: int, num_steps: int = 6) -> plt.Figure:
    """Exact vs decoded hazard maps along one game (section 4). Grid row y = 0 is drawn at the bottom."""
    rows = (data.episode == episode).nonzero().flatten()
    if len(rows) == 0:
        raise ValueError(f"episode {episode} has no running steps in the data set.")
    chosen = rows[np.unique(np.linspace(0, len(rows) - 1, num_steps).round().astype(int))]
    decoded = probe.hazards(data.latents[chosen]).cpu().numpy()
    exact = data.hazards[chosen].cpu().numpy()
    fig, axes = plt.subplots(6, len(chosen), figsize=(2.0 * len(chosen), 12.4), dpi=130, squeeze=False)
    for j, row in enumerate(chosen.tolist()):
        cell, heading = int(data.cell[row]), int(data.heading[row])
        x, y = cell % 4, cell // 4
        for k, name in enumerate(HAZARDS):
            for i, (label, values) in enumerate((("exact", exact), ("decoded", decoded))):
                ax = axes[2 * k + i, j]
                grid = values[j, k * NUM_CELLS:(k + 1) * NUM_CELLS].reshape(4, 4)
                ax.imshow(grid, origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
                dx, dy = _ARROWS[heading]
                ax.annotate("", xy=(x + dx, y + dy), xytext=(x - dx, y - dy),
                            arrowprops=dict(arrowstyle="-|>", color="#2ec4b6", lw=2))
                ax.set_xticks([])
                ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel(f"{label}\nP({name})", fontsize=9)
                if 2 * k + i == 0:
                    ax.set_title(f"t = {int(data.step[row])}", fontsize=9)
    fig.suptitle(f"Exact vs MLP-decoded beliefs, held-out game {episode} (arrow: hunter, heading)", fontsize=10)
    fig.tight_layout()
    return fig


def plot_latent_pca(data: WumpusLatentDataset) -> plt.Figure:
    """Latents in their first two principal components, coloured by cell and by uncertainty (section 4)."""
    centred = data.latents.double() - data.latents.double().mean(0)
    _, singular, vh = torch.linalg.svd(centred, full_matrices=False)
    coords = (centred @ vh[:2].T).cpu().numpy()
    ratios = (singular ** 2 / (singular ** 2).sum())[:2].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=130)
    panels = ((data.cell, "tab20", "hunter cell (x + 4y)"),
              (data.log_layouts, "viridis", "log10 consistent layouts (uncertainty)"))
    for ax, (colour, cmap, label) in zip(axes, panels):
        points = ax.scatter(coords[:, 0], coords[:, 1], c=colour.cpu().numpy(), cmap=cmap, s=3, alpha=0.5)
        fig.colorbar(points, ax=ax, label=label)
        ax.set_xlabel(f"PC1 ({ratios[0]:.1%} of variance)")
        ax.set_ylabel(f"PC2 ({ratios[1]:.1%} of variance)")
        ax.grid(True, alpha=0.3, linestyle="--")
    fig.suptitle("Belief-filter latents of running games")
    fig.tight_layout()
    return fig
