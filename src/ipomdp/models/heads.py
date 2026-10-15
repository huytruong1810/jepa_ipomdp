# ABSOLUTE PATH: src/ipomdp/models/heads.py
# ==============================================================================
# PREDICTION HEADS OVER THE BELIEF LATENT
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Vector Beliefs, Agent-Side Inputs Only:
#    - Every head reads the single belief latent z_t of shape (B, D) (world_model.py,
#      section 2) and, where needed, the agent's own action a_t as a one-hot.
#    - There is no opponent input. In an I-POMDP the agent never observes the other agent's
#      action, only its effects on observations and rewards; the learned model therefore
#      describes the POMDP the agent actually faces, with the opponent folded into the
#      environment exactly as in the exact solver's S x M_j reduction (domain/pomdp.py,
#      section 4). What the latent knows about the opponent is read out by probes
#      (src/ipomdp/interpretability), never fed in. (Phase-4 decision; an earlier design fed
#      the opponent's true action -- privileged information -- into every head.)
#
# 2. Distributional Value and Reward:
#    - Two-hot distributions over bins bounded by the domain's value bound
#      (models/distributions.py); the decoded mean is unbiased. The final projection is
#      zero-initialised so the initial prediction is uniform, whose mean is exactly 0.
#
# 2b. Horizon Value Head (optional, model.time_aware_value; user decision 2026-10-07):
#    - V(z_t, k_t) where k_t = H - t is the number of steps left in the agent's fixed window of H
#      steps. k is the agent's own step counter, not privileged information. It enters as k / H
#      concatenated to the latent.
#    - Why: Wumpus World is scored on the return inside a 50-step window, but the window is a
#      truncation and a stationary V(z) bootstraps through it, i.e. it optimises an infinite-
#      horizon game in which a wasted step costs only about 1 + (1 - gamma) * V ~ 7.5. The
#      greedy planner then idled in front of provably safe unvisited cells whenever V-hat errors
#      between neighbouring latents exceeded that (~15 measured;
#      reports/2026-10-02_wumpus_diagnostics/unfinished_diag.py). With k the value is the
#      finite-horizon value of the window, so a wasted step costs a step of a scarce budget.
#    - Only the value depends on time: the dynamics (reward and observation heads) are
#      stationary and get no k. The terminal rule V(z, 0) = 0 is applied by the callers
#      (planning/search_model.py, section 5; training/trainer.py, section 3b), never learned.
#
# 2c. Value Pair for Double and Clipped-Double Estimation (optional, model.value_estimator; 2026-10-15):
#    - Two independent value heads A and B read as one value: their mean (double, van Hasselt 2010) or
#      their minimum (clipped double, Fujimoto et al. 2018). The trainer gives each its own targets and
#      rows (training/trainer.py, section 3e); the planner and the backup read the combined value.
#    - Why: on Wumpus the value backed up through the learned model, max over actions of noisy child
#      values, is flat and impossibly optimistic (V-hat > 999, the most any game pays, on 40-59% of the
#      oracle-latent agent's live steps; reports/2026-10-02_wumpus_diagnostics/phantom_reward_diag.py).
#      Iterated on frozen latents the plain max diverges (to ~50,000), while double and clipped-double
#      targets remove most impossible values and gain +136 to +212 per game at 50 simulations on 2 of 3
#      oracle-latent networks (fvi_screen.py, section 4).
#    - Every value head exposes value(latent, codec), the decoded V, so the search model reads one
#      head or a pair the same way. A pair has no horizon variant (it raises in training/run.py).
#
# 3. Observation Head (the planning model):
#    - P(o_{t+1} | z_t, a_t). The planner branches over its outcomes and steps the belief
#      filter (world_model.py, section 6). It is trained on DETACHED latents, so it never
#      shapes the representation. On canonical Tiger it reaches a mean KL to the exact
#      P(o' | b, a) of 0.001-0.007 nats (LISTEN) and < 0.0002 (door actions).
#
# 4. Multi-Step Observation Predictor (optional representation term, model.observation_horizon):
#    - P(o_{t+j} | z_t, a_t..a_{t+j-1}) for j = 1..k, by an open-loop action-conditioned
#      rollout h_0 = z_t, h_j = GRUCell(a_{t+j-1}, h_{j-1}), read out by a residual MLP. Its
#      cross-entropy flows INTO the belief filter (training/trainer.py, section 2c).
#    - Why: on Wumpus World neither JEPA nor one-step observation prediction makes the latent
#      track the hunter's cell, heading and the danger ahead, although the same filter learns
#      them almost perfectly from privileged supervision (HANDOFF.md, section 4c, 2026-10-03).
#      Predicting future percepts under arbitrary future actions requires exactly that
#      information. It is a self-supervised, non-privileged signal: actions and observations only.
#    - It is not a planning model: the planner never reads it. With k = 0 it is not built at all.
# ==============================================================================

from enum import StrEnum

import torch
import torch.nn as nn
from torch import Tensor

from .distributions import TwoHotSymlog
from .layers import build_residual_stack


def _zero_init_last_layer(net: nn.Sequential) -> nn.Sequential:
    """Zero weights and bias of the final Linear: uniform logits, decoded mean 0."""
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


class ValueHead(nn.Module):
    """V(z_t) as two-hot logits."""

    def __init__(self, latent_dim: int, hidden_dim: int, num_blocks: int, num_bins: int):
        super().__init__()
        self.net = _zero_init_last_layer(build_residual_stack(latent_dim, hidden_dim, num_bins, num_blocks))

    def forward(self, latent: Tensor) -> Tensor:
        """(B, D) -> (B, num_bins) logits."""
        return self.net(latent)

    def value(self, latent: Tensor, codec: TwoHotSymlog) -> Tensor:
        """(B, D) -> (B,) decoded V (section 2c)."""
        return codec.mean(self(latent))


class ValueEstimator(StrEnum):
    """How the value target backs up child values (section 2c; training/trainer.py, section 3e)."""

    MAX = "max"                        # one head, max_a over its own child values (the default)
    DOUBLE = "double"                  # pair: A picks the action, B evaluates it, and vice versa
    CLIPPED_DOUBLE = "clipped_double"  # pair: max_a over child values min(A, B)


class ValuePair(nn.Module):
    """Two value heads read as one value: their mean (double) or their minimum (clipped double); section 2c."""

    def __init__(self, a: ValueHead, b: ValueHead, estimator: ValueEstimator):
        super().__init__()
        if estimator is ValueEstimator.MAX:
            raise ValueError("a value pair needs the double or clipped-double estimator.")
        self.a, self.b, self.estimator = a, b, estimator

    def head_values(self, latent: Tensor, codec: TwoHotSymlog) -> tuple[Tensor, Tensor]:
        """(B, D) -> the decoded V of A and of B, each (B,)."""
        return self.a.value(latent, codec), self.b.value(latent, codec)

    def value(self, latent: Tensor, codec: TwoHotSymlog) -> Tensor:
        """(B, D) -> (B,) the combined V: (A + B) / 2 for double, min(A, B) for clipped double."""
        va, vb = self.head_values(latent, codec)
        return 0.5 * (va + vb) if self.estimator is ValueEstimator.DOUBLE else torch.minimum(va, vb)


class HorizonValueHead(nn.Module):
    """V(z_t, k_t) as two-hot logits, k_t the steps left in the window (section 2b)."""

    def __init__(self, latent_dim: int, hidden_dim: int, num_blocks: int, num_bins: int, horizon: int):
        super().__init__()
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}.")
        self.horizon = horizon
        self.net = _zero_init_last_layer(build_residual_stack(latent_dim + 1, hidden_dim, num_bins, num_blocks))

    def forward(self, latent: Tensor, steps_left: Tensor) -> Tensor:
        """(B, D) latents and (B,) steps left -> (B, num_bins) logits."""
        fraction = (steps_left / self.horizon).to(latent.dtype).unsqueeze(-1)
        return self.net(torch.cat([latent, fraction], dim=-1))


class RewardHead(nn.Module):
    """R(z_t, a_t) as two-hot logits."""

    def __init__(self, latent_dim: int, num_actions: int, hidden_dim: int, num_blocks: int, num_bins: int):
        super().__init__()
        self.net = _zero_init_last_layer(build_residual_stack(latent_dim + num_actions, hidden_dim, num_bins, num_blocks))

    def forward(self, latent: Tensor, action: Tensor) -> Tensor:
        """(B, D), one-hot (B, |A|) -> (B, num_bins) logits."""
        return self.net(torch.cat([latent, action], dim=-1))


class ObservationHead(nn.Module):
    """P(o_{t+1} | z_t, a_t) as logits; the planning model, trained on detached latents."""

    def __init__(self, latent_dim: int, num_actions: int, num_observations: int, hidden_dim: int, num_blocks: int):
        super().__init__()
        self.net = build_residual_stack(latent_dim + num_actions, hidden_dim, num_observations, num_blocks)

    def forward(self, latent: Tensor, action: Tensor) -> Tensor:
        """(B, D), one-hot (B, |A|) -> (B, |O|) logits."""
        return self.net(torch.cat([latent, action], dim=-1))


class MultiStepObservationPredictor(nn.Module):
    """Open-loop P(o_{t+j} | z_t, a_t..a_{t+j-1}), j = 1..k (section 4)."""

    def __init__(self, latent_dim: int, num_actions: int, num_observations: int, hidden_dim: int, num_blocks: int,
                 horizon: int):
        super().__init__()
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}.")
        self.horizon = horizon
        self.cell = nn.GRUCell(num_actions, latent_dim)
        self.readout = build_residual_stack(latent_dim, hidden_dim, num_observations, num_blocks)

    def forward(self, latents: Tensor, actions: Tensor) -> list[Tensor]:
        """
        Args:
            latents: z_0..z_{T-1} of shape (B, T, D), the start of every rollout.
            actions: One-hot a_0..a_{T-1} of shape (B, T, |A|).

        Returns:
            For j = 1..min(k, T): logits of o_{t+j} for t = 0..T-j, shape (B, T - j + 1, |O|).
        """
        batch, steps, dim = latents.shape
        hidden = latents
        out = []
        for j in range(1, min(self.horizon, steps) + 1):
            valid = steps - j + 1                                     # rollouts that still have a target
            hidden = self.cell(actions[:, j - 1:j - 1 + valid].reshape(batch * valid, -1),
                               hidden[:, :valid].reshape(batch * valid, dim)).view(batch, valid, dim)
            out.append(self.readout(hidden))
        return out
