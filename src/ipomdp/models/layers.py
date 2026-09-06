# ABSOLUTE PATH: src/ipomdp/models/layers.py
# ==============================================================================
# SOTA NEURAL BUILDING BLOCKS & ATTENTION LAYERS
# ==============================================================================
#
# DESIGN DECISIONS & ARCHITECTURAL HIGHLIGHTS:
# 1. Root Mean Square Normalization (RMSNorm):
#    - Float32 precision invariance with epsilon=1e-6 to ensure robust normalization
#      across large transformer/ResMLP representations without mean-centering overhead.
#
# 2. SwiGLU Gated Multi-Layer Perceptrons:
#    - Employs gated linear unit activation: SwiGLU(x) = (w1(x) * swish(w2(x))) @ w3
#    - Zero-initialized final projection w3 ensures residual identity mapping at step 0.
#
# 3. Permutation-Equivariant Swarm Action Fusion:
#    - Cross-attention transformer layer pooling variable numbers of opponent action
#      embeddings (M >= 0) without fixed-width tensor slicing.
#
# 4. Multi-Slot Attention Pooler:
#    - Scaled dot-product attention mapping multi-slot representations (B, N_obj, D)
#      to compact state vectors (B, D) using a learned CLS token query.
# ==============================================================================

import math
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Normalizes inputs by root-mean-square without mean-centering, reducing memory
    bandwidth and compute overhead while stabilizing deep representations.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Promote to float32 internally to guarantee numerical stability
        orig_dtype = x.dtype
        x_f32 = x.float()
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        normed = x_f32 * torch.rsqrt(variance + self.eps)
        return (normed * self.scale).to(orig_dtype)


class SwiGLUResidualBlock(nn.Module):
    """
    SwiGLU Gated Feed-Forward Residual Block with Zero-Initialized Residual Projection.
    """

    def __init__(self, dim: int, expansion_factor: float = 2.0):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.norm = RMSNorm(dim)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

        # Variance scaling initialization
        nn.init.kaiming_normal_(self.w1.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.w2.weight, nonlinearity='linear')
        # Identity mapping at step 0 via zero-initialization
        nn.init.zeros_(self.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        normed = self.norm(x)
        gated = F.silu(self.w1(normed)) * self.w2(normed)
        out = self.w3(gated)
        return residual + out


def build_residual_stack(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_blocks: int = 2
) -> nn.Sequential:
    """
    Constructs a SOTA SwiGLU Gated ResMLP network.

    Args:
        input_dim: Input feature dimensionality.
        hidden_dim: Hidden dimension of residual blocks.
        output_dim: Final output feature dimensionality.
        num_blocks: Number of sequential SwiGLU residual blocks.

    Returns:
        nn.Sequential container implementing the deep ResMLP pipeline.
    """
    layers: List[nn.Module] = [
        nn.Linear(input_dim, hidden_dim),
        RMSNorm(hidden_dim)
    ]
    for _ in range(num_blocks):
        layers.append(SwiGLUResidualBlock(hidden_dim))
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class LearnedPositionalEncoding2D(nn.Module):
    """
    Learned 2D spatial positional encodings for feature maps.
    """

    def __init__(self, dim: int, height: int, width: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, height, width, dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map tensor of shape (B, C, H, W).
        Returns:
            Flattened spatial sequence with positional embeddings added: (B, H*W, C).
        """
        B, C, H, W = x.shape
        x_perm = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_pos = x_perm + self.pos_embed
        return x_pos.view(B, H * W, C)


class AttentionPooler(nn.Module):
    """
    Learned Multi-Object Attention Pooler.

    Uses a learned query token (CLS) to pool multi-object latent representations
    (B, N_obj, D) into a global context vector (B, D).
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm_tokens = RMSNorm(dim)
        self.norm_cls = RMSNorm(dim)

    def forward(self, object_slots: torch.Tensor) -> torch.Tensor:
        """
        Args:
            object_slots: Multi-object tensor of shape (B, N_obj, D).

        Returns:
            Pooled feature tensor of shape (B, D).
        """
        B = object_slots.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        normed_slots = self.norm_tokens(object_slots)
        normed_cls = self.norm_cls(cls_tokens)

        pooled, _ = self.multihead_attn(
            query=normed_cls,
            key=normed_slots,
            value=normed_slots
        )
        return pooled.squeeze(1)


class SwarmActionEncoder(nn.Module):
    """
    Cross-Attention Joint Action Fusion Module.

    Permutation-invariant cross-attention fusion over arbitrary numbers of opponent actions.
    """

    def __init__(
        self,
        action_dim_i: int,
        action_dim_j: int,
        hidden_dim: int,
        num_heads: int = 4
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.ego_proj = nn.Linear(action_dim_i, hidden_dim)
        self.opp_proj = nn.Linear(action_dim_j, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = RMSNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self,
        ego_action: torch.Tensor,
        opp_actions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            ego_action: One-hot or continuous ego action tensor of shape (B, action_dim_i).
            opp_actions: Opponent action tensor of shape (B, num_opps, action_dim_j) or (B, action_dim_j).

        Returns:
            Fused action embedding of shape (B, hidden_dim).
        """
        B = ego_action.size(0)
        e_emb = self.ego_proj(ego_action).unsqueeze(1)  # (B, 1, H)

        if opp_actions is None or opp_actions.numel() == 0 or (opp_actions.dim() >= 2 and opp_actions.size(1) == 0):
            o_emb = torch.zeros_like(e_emb)
            fused_context = o_emb
        else:
            if opp_actions.dim() == 2:
                opp_actions = opp_actions.unsqueeze(1)


            o_tokens = self.opp_proj(opp_actions)  # (B, M, H)
            attn_out, _ = self.cross_attn(
                query=e_emb,
                key=o_tokens,
                value=o_tokens
            )
            fused_context = self.norm(attn_out)

        combined = torch.cat([e_emb, fused_context], dim=-1).squeeze(1)  # (B, 2*H)
        return self.out_proj(combined)
