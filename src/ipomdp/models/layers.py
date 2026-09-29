# ABSOLUTE PATH: src/ipomdp/models/layers.py
# ==============================================================================
# NEURAL BUILDING BLOCKS: RMSNorm AND SwiGLU RESIDUAL MLPs
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. RMSNorm:
#    - Scale-only normalisation (no mean centring), computed in float32 with eps = 1e-6 so it
#      is stable under bfloat16 autocast.
#
# 2. SwiGLU Residual Blocks:
#    - x + w3( silu(w1(RMSNorm(x))) * w2(RMSNorm(x)) ).
#    - w3 is zero-initialised, so every block is the identity at initialisation and a stack
#      starts as its input and output projections.
#
# 3. Scope:
#    - Attention poolers, swarm action encoders and 2D positional encodings belonged to the
#      removed multi-slot / image-observation design and were deleted with it.
# ==============================================================================

from typing import List
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
    Input projection + RMSNorm, num_blocks SwiGLU residual blocks, output projection.

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
