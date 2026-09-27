# ABSOLUTE PATH: src/ipomdp/models/heads.py
# ==============================================================================
# PREDICTION HEADS OVER THE BELIEF LATENT
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Vector Beliefs:
#    - Every head reads the single belief latent z_t of shape (B, D) (see world_model.py,
#      section 2). The attention poolers and cross-attention "swarm" action encoders that
#      served the multi-slot design were removed; actions enter as concatenated one-hots.
#
# 2. DreamerV3 Distributional Value and Reward (reviewed in Phase 3):
#    - Value and reward are 255-bin two-hot distributions in symlog space
#      (src/ipomdp/models/distributions.py). The final projection is zero-initialised so the
#      initial prediction is the uniform distribution, whose decoded expectation is exactly 0.
#
# 3. Opponent Head:
#    - Predicts the opponent's action distribution from z_t. In the single-agent POMDP the
#      opponent action space is a singleton, so the head is inert until Phases 3-4.
#
# 4. Observation Probe Head:
#    - P(o_{t+1} | z_t, a_t), trained only on detached latents for interpretability; it never
#      shapes the representation (reviewed in Phase 6).
# ==============================================================================

import torch
import torch.nn as nn
from torch import Tensor

from .layers import build_residual_stack


def _zero_init_last_layer(net: nn.Sequential) -> nn.Sequential:
    """Zero weights and bias of the final Linear: uniform logits, decoded expectation 0."""
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


class ValueHead(nn.Module):
    """V(z_t) as two-hot symlog logits."""

    def __init__(self, latent_dim: int, hidden_dim: int, num_blocks: int, num_bins: int):
        super().__init__()
        self.net = _zero_init_last_layer(build_residual_stack(latent_dim, hidden_dim, num_bins, num_blocks))

    def forward(self, latent: Tensor) -> Tensor:
        """(B, D) -> (B, num_bins) logits."""
        return self.net(latent)


class RewardHead(nn.Module):
    """R(z_t, a_t, a^j_t) as two-hot symlog logits."""

    def __init__(self, latent_dim: int, num_actions: int, num_opponent_actions: int, hidden_dim: int,
                 num_blocks: int, num_bins: int):
        super().__init__()
        self.net = _zero_init_last_layer(
            build_residual_stack(latent_dim + num_actions + num_opponent_actions, hidden_dim, num_bins, num_blocks))

    def forward(self, latent: Tensor, action: Tensor, opponent_action: Tensor) -> Tensor:
        """(B, D), one-hot (B, |A_i|), one-hot (B, |A_j|) -> (B, num_bins) logits."""
        return self.net(torch.cat([latent, action, opponent_action], dim=-1))


class OpponentPolicyHead(nn.Module):
    """pi_j(a^j | z_t) as logits over the opponent's actions."""

    def __init__(self, latent_dim: int, num_opponent_actions: int, hidden_dim: int, num_blocks: int):
        super().__init__()
        self.net = build_residual_stack(latent_dim, hidden_dim, num_opponent_actions, num_blocks)

    def forward(self, latent: Tensor) -> Tensor:
        """(B, D) -> (B, |A_j|) logits."""
        return self.net(latent)


class ObservationProbeHead(nn.Module):
    """P(o_{t+1} | z_t, a_t) as logits; an interpretability probe trained on detached latents."""

    def __init__(self, latent_dim: int, num_actions: int, num_observations: int, hidden_dim: int, num_blocks: int):
        super().__init__()
        self.net = build_residual_stack(latent_dim + num_actions, hidden_dim, num_observations, num_blocks)

    def forward(self, latent: Tensor, action: Tensor) -> Tensor:
        """(B, D), one-hot (B, |A|) -> (B, |O|) logits."""
        return self.net(torch.cat([latent, action], dim=-1))
