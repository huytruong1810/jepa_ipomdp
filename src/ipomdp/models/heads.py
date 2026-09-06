# ABSOLUTE PATH: src/ipomdp/models/heads.py
# ==============================================================================
# NEURAL PREDICTION & DIAGNOSTIC PROBE HEADS
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Permutation-Invariant Attention Pooling:
#    - All downstream regression heads process structured multi-object beliefs
#      b_t in R^(B x N_obj x D) via AttentionPooler. A learnable [CLS] query token
#      attends across object slots, producing a permutation-invariant summary vector.
#    - Query tensor expansion invokes .contiguous() to ensure native PyTorch SDPA
#      FlashAttention kernel dispatch without C++ fallback.
#
# 2. Swarm Action Encoder Rank Safety & Validation:
#    - SwarmActionEncoder validates and reshapes integer indices, 1D batches, 2D singletons,
#      or 3D multi-opponent tensors into (B, M, action_dim_j), preventing shape crashes.
#
# 3. Distributional Value & Immediate Reward Heads:
#    - ValueHead outputs 255-bin symlog categorical logits predicting discounted
#      lambda-returns from current latent information states.
#    - RewardHead outputs 255-bin symlog logits predicting physical transition rewards
#      given belief and joint multi-agent actions (a_i, a_j).
#
# 4. Opponent Policy Head:
#    - DiscretePolicyHead utilizes learnable Agent ID queries to predict categorical
#      action probabilities for surrounding agents: pi_eta(a_j | b_t).
#
# 5. Robust Action-Conditioned Observation Probe Head:
#    - ObservationProbeHead decodes discrete observation logits P_theta(o_{t+1} | b_{t+1}, a_t)
#      from frozen JEPA latent representations for offline information-theoretic evaluation.
#    - Handles 1D integer index inputs, unbatched tensors, and float one-hot action vectors
#      without batch dimension mismatches or group convolution errors.
# ==============================================================================

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import build_residual_stack, RMSNorm, AttentionPooler, SwarmActionEncoder
from ..types import Action


class ValueHead(nn.Module):
    """Predicts 255-bin symlog categorical value distributions from pooled belief states."""

    def __init__(self, latent_dim: int, hidden_dim: int = 128, num_blocks: int = 2, num_bins: int = 255):
        super().__init__()
        self.pooler = AttentionPooler(latent_dim)
        self.net = build_residual_stack(latent_dim, hidden_dim, num_bins, num_blocks)

    def forward(self, belief: torch.Tensor) -> torch.Tensor:
        """
        Computes value distribution logits.

        Args:
            belief: Belief state tensor of shape (B, N_obj, D_latent).

        Returns:
            Symlog value logits of shape (B, num_bins).
        """
        pooled = self.pooler(belief)
        return self.net(pooled)


class RewardHead(nn.Module):
    """Predicts 255-bin symlog immediate reward distributions given belief and joint actions."""

    def __init__(
        self,
        latent_dim: int,
        action_dim_i: int,
        action_dim_j: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        num_bins: int = 255
    ):
        super().__init__()
        self.action_dim_j = int(action_dim_j)
        self.pooler = AttentionPooler(latent_dim)
        self.swarm_encoder = SwarmActionEncoder(action_dim_i, action_dim_j, hidden_dim=hidden_dim, num_heads=4)
        self.net = build_residual_stack(latent_dim + self.swarm_encoder.hidden_dim, hidden_dim, num_bins, num_blocks)

    def forward(
        self,
        belief: torch.Tensor,
        ego_action: torch.Tensor,
        opp_actions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes immediate reward distribution logits.

        Args:
            belief: Belief state tensor of shape (B, N_obj, D_latent).
            ego_action: Ego action tensor of shape (B, action_dim_i).
            opp_actions: Opponent action tensor of shape (B, M, action_dim_j) or (B, action_dim_j).

        Returns:
            Symlog reward logits of shape (B, num_bins).
        """
        b = belief.size(0)
        pooled = self.pooler(belief)

        if opp_actions is None or opp_actions.numel() == 0 or (opp_actions.dim() >= 2 and opp_actions.size(1) == 0):
            opp_actions = torch.zeros(b, 1, self.action_dim_j, device=belief.device, dtype=ego_action.dtype)

        joint_features = self.swarm_encoder(ego_action, opp_actions)
        x = torch.cat([pooled, joint_features], dim=-1)
        return self.net(x)


class DiscretePolicyHead(nn.Module):
    """Opponent Policy Head predicting discrete action distributions for the surrounding swarm."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        num_opponents: int = 1,
        hidden_dim: int = 128,
        num_blocks: int = 2
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.num_opponents = int(num_opponents)
        self.latent_dim = int(latent_dim)
        self.norm = RMSNorm(latent_dim)

        self.agent_queries = nn.Parameter(torch.randn(1, num_opponents, latent_dim))
        nn.init.normal_(self.agent_queries, std=0.02)

        self.to_q = nn.Linear(latent_dim, latent_dim, bias=False)
        self.to_k = nn.Linear(latent_dim, latent_dim, bias=False)
        self.to_v = nn.Linear(latent_dim, latent_dim, bias=False)

        self.net = build_residual_stack(latent_dim, hidden_dim, action_dim, num_blocks)

    def forward(self, belief: torch.Tensor) -> torch.Tensor:
        """
        Predicts action logits for all tracked opponents.

        Args:
            belief: Belief state tensor of shape (B, N_obj, D_latent).

        Returns:
            Opponent action logits of shape (B, num_opponents, action_dim).
        """
        b = belief.size(0)
        x_norm = self.norm(belief)

        queries_expanded = self.agent_queries.expand(b, self.num_opponents, self.latent_dim).contiguous()
        q = self.to_q(queries_expanded)
        k = self.to_k(x_norm)
        v = self.to_v(x_norm)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        return self.net(attn_out)


class ObservationProbeHead(nn.Module):
    """
    Non-Intrusive Observation Classifier Probe.
    Decodes discrete observation logits P_theta(o_{t+1} | b_{t+1}, a_t) directly from
    latent belief states and action context without generative decoders.
    Trained strictly with stop-gradients on frozen JEPA representations.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int = 3,
        num_obs_classes: int = 6,
        hidden_dim: int = 128,
        num_blocks: int = 2
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.pooler = AttentionPooler(latent_dim)
        self.net = build_residual_stack(latent_dim + self.action_dim, hidden_dim, num_obs_classes, num_blocks)

    def forward(self, belief: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Decodes observation class logits conditioned on belief and transition action.

        Args:
            belief: Latent belief state tensor of shape (B, N_obj, D_latent).
            action: Action tensor of shape (B, action_dim) or (B, 1) integer index or (B,).

        Returns:
            Observation category logits of shape (B, num_obs_classes).
        """
        pooled = self.pooler(belief)
        b = pooled.size(0)

        # Rank and dtype safety: Ensure action is one-hot float tensor of shape (B, action_dim)
        action_onehot = Action.to_one_hot(action, self.action_dim, device=pooled.device)
        if action_onehot.size(0) != b:
            action_onehot = action_onehot.expand(b, -1)


        x = torch.cat([pooled, action_onehot], dim=-1)
        return self.net(x)
