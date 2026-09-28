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
# 3. Observation Head (the planning model):
#    - P(o_{t+1} | z_t, a_t). The planner branches over its outcomes and steps the belief
#      filter (world_model.py, section 6). It is trained on DETACHED latents, so it never
#      shapes the representation. On canonical Tiger it reaches a mean KL to the exact
#      P(o' | b, a) of 0.001-0.007 nats (LISTEN) and < 0.0002 (door actions).
# ==============================================================================

import torch
import torch.nn as nn
from torch import Tensor

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
