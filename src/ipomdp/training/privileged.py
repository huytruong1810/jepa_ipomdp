# ABSOLUTE PATH: src/ipomdp/training/privileged.py
# ==============================================================================
# PRIVILEGED EXACT-BELIEF SUPERVISION OF THE FILTER (THE "ORACLE-LATENT" DIAGNOSTIC ARM)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (user decision 2026-10-11, HANDOFF section 6):
#    - On Wumpus World the greedy planner idles in front of provably safe cells. Exact-belief
#      overrides (reports/2026-10-02_wumpus_diagnostics/override_diag.py) price the idling at +300 per
#      game, most of it from turning towards an adjacent safe cell and stepping in. The agent's latent
#      decodes the hunter's heading and cell only ~58% of the time on random histories, against
#      99.9-100% for the same filter trained on exact targets (supervised_filter.py).
#    - Is the idling a REPRESENTATION failure (a turn looks like a no-op because the latent does not
#      know where the hunter faces) or a VALUE / PLANNING failure? Give the agent a latent that knows
#      the exact belief and see whether it still idles.
#
# 2. Method:
#    - A readout (residual MLP, models/layers.py) maps every latent z_0..z_T of the training batch to
#      the domain's exact-belief targets; its loss is ADDED to the trainer's losses with weight 1 and
#      its gradients flow INTO the filter, like the decoder's observation term (trainer.py, section 2e).
#      Everything else (value targets, heads, search, acting copies) is unchanged.
#    - Targets are computed per batch from the episode's own history by the exact Bayes filter (for
#      Wumpus: WumpusBeliefTracker.supervision_targets, domain/wumpus.py section 6). No buffer change is
#      needed and no hidden state is read; the privilege is the known model.
#    - Loss: cross-entropy for each categorical target (Wumpus: cell, heading) plus binary
#      cross-entropy averaged over the soft targets (Wumpus: 48 hazard marginals and the danger ahead),
#      on live steps only. This is supervised_filter.py's loss.
#
# 3. Scope:
#    - A diagnostic arm, never an agent to report as a method: it uses the exact model.
#    - Domains opt in by giving DomainSetup a PrivilegedSpec (experiments/runs.py); Tiger has none and
#      the option raises there. Off (the default) builds nothing and draws nothing: canonical Tiger and
#      every existing run are unchanged.
# ==============================================================================

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..models.layers import build_residual_stack
from .episode_buffer import EpisodeBatch


@dataclass(frozen=True)
class PrivilegedTargets:
    """Exact targets for every step 0..T of a batch: class indices, soft binaries and a live mask (section 2)."""

    classes: tuple[Tensor, ...]          # each (B, T + 1) int64
    soft: Tensor                         # (B, T + 1, S) float32 in [0, 1]
    live: Tensor                         # (B, T + 1) bool


@dataclass(frozen=True)
class PrivilegedSpec:
    """What a domain supervises: the number of classes of each categorical target, the soft-target count, the targets."""

    class_sizes: tuple[int, ...]
    num_soft: int
    targets: Callable[[EpisodeBatch], PrivilegedTargets]


class PrivilegedBeliefLoss(nn.Module):
    """The readout and its loss (section 2); its parameters train with the filter."""

    def __init__(self, spec: PrivilegedSpec, latent_dim: int, hidden_dim: int, num_blocks: int):
        super().__init__()
        self.spec = spec
        self.readout = build_residual_stack(latent_dim, hidden_dim, sum(spec.class_sizes) + spec.num_soft, num_blocks)

    def forward(self, latents: Tensor, episodes: EpisodeBatch) -> Tensor:
        """Loss of latents (B, T + 1, D) against the exact targets of `episodes` (section 2)."""
        with torch.no_grad():
            targets = self.spec.targets(episodes)
        out = self.readout(latents[targets.live]).float()
        loss, start = out.new_zeros(()), 0
        for size, classes in zip(self.spec.class_sizes, targets.classes):
            loss = loss + F.cross_entropy(out[:, start:start + size], classes[targets.live])
            start += size
        return loss + F.binary_cross_entropy_with_logits(out[:, start:], targets.soft[targets.live])
