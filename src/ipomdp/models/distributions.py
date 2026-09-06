# ABSOLUTE PATH: src/ipomdp/models/distributions.py
# ==============================================================================
# DISTRIBUTIONAL TWO-HOT SYMLOG REGRESSION MODULE (DREAMERV3 STANDARD)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Float32 Internal Numerical Safeguards:
#    - All logarithmic, exponential, and normalization operations execute strictly
#      in torch.float32 before casting back to input dtypes. This eliminates NaN
#      instability and mantissa underflow under CUDA bfloat16/float16 execution.
#
# 2. Asymmetric Dynamic Range Clamping (symexp):
#    - Clamps input magnitudes to max_val = 11.0 for fp16/bf16 (exp(11) ≈ 5.98e4, safely
#      below fp16 overflow 6.55e4) and max_val = 88.0 for fp32 (exp(88) ≈ 1.65e38, safely
#      below fp32 overflow 3.4e38).
#
# 3. Exact Two-Hot Categorical Mapping:
#    - Discretizes continuous target y into K=255 bins spanning [min_val, max_val] in
#      symlog space. Continuous values are linearly interpolated between two adjacent bins:
#         below = floor((y - min_val) / step)
#         above = below + 1
#         w_above = (y - b_below) / step,  w_below = 1 - w_above
#    - Computing loss via F.cross_entropy(logits, target_probs) strictly implements
#      the DreamerV3 distributional cross-entropy objective.
#
# 4. Rank-Safe Target Alignment:
#    - Dynamically handles target tensors of shape (B, 1), (B,), or (B, T, 1), guaranteeing
#      the returned loss tensor strictly preserves leading batch/sequence dimensions
#      with a trailing singleton channel (..., 1).
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    """
    Symmetric logarithmic compression: sign(x) * ln(|x| + 1).

    Compresses wide-ranging continuous rewards and returns into a stable scale
    while preserving sign symmetry around zero. Computes internally in float32.

    Args:
        x: Continuous numerical input tensor of arbitrary shape.

    Returns:
        Symlog-compressed tensor with matching dynamic range and input dtype.
    """
    orig_dtype = x.dtype
    x_f32 = x.float()
    res = torch.sign(x_f32) * torch.log1p(torch.abs(x_f32))
    return res.to(dtype=orig_dtype)


def symexp(x: torch.Tensor) -> torch.Tensor:
    """
    Symmetric exponential decompression: sign(x) * (exp(|x|) - 1).

    Inverts symlog-compressed values back to their original physical scale.
    Applies dtype-aware clamping to prevent exponent overflow.

    Args:
        x: Symlog-compressed input tensor of arbitrary shape.

    Returns:
        Decompressed continuous tensor in original physical scale and dtype.
    """
    orig_dtype = x.dtype
    x_f32 = x.float()

    # Dtype-aware safe threshold bounds
    if orig_dtype in (torch.float16, torch.bfloat16):
        max_val = 11.0
    else:
        max_val = 88.0

    safe_x = torch.clamp(torch.abs(x_f32), max=max_val)
    res = torch.sign(x_f32) * torch.expm1(safe_x)
    return res.to(dtype=orig_dtype)


class TwoHotSymlog(nn.Module):
    """
    DreamerV3 Distributional Two-Hot Categorical Regression Module.

    Maps continuous scalar targets onto a discrete grid of K bins in symlog space.
    Eliminates target scale sensitivity without requiring dynamic return normalization.
    """

    def __init__(self, min_val: float = -20.0, max_val: float = 20.0, num_bins: int = 255):
        """
        Initializes discrete bin centers in symlog space.

        Args:
            min_val: Minimum representable value in symlog space (symexp(-20) ≈ -4.85e8).
            max_val: Maximum representable value in symlog space (symexp(20) ≈ 4.85e8).
            num_bins: Total discrete categorical bins (default: 255).
        """
        super().__init__()
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.num_bins = int(num_bins)
        self.step_size = (self.max_val - self.min_val) / (self.num_bins - 1)

        # Register linearly spaced bin centers as persistent non-trainable buffer
        bins = torch.linspace(self.min_val, self.max_val, self.num_bins, dtype=torch.float32)
        self.register_buffer('bins', bins)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        auto_symlog: bool = True
    ) -> torch.Tensor:
        """
        Calculates cross-entropy loss between predicted bin logits and two-hot target distribution.

        Args:
            logits: Predicted distribution logits of shape (..., num_bins).
            targets: Continuous target values (returns or rewards) of shape (..., 1) or (...).
            auto_symlog: If True (default), compresses raw continuous targets via symlog()
                         prior to two-hot bin projection.

        Returns:
            Cross-entropy loss tensor of shape (..., 1) with matching input dtype.
        """
        orig_dtype = logits.dtype
        logits_f32 = logits.float()
        targets_f32 = targets.float()

        # Rank safety: Normalize target to match logits batch dimensions
        if targets_f32.dim() == logits_f32.dim() and targets_f32.size(-1) == 1:
            targets_f32 = targets_f32.squeeze(-1)

        # Ensure bins match device
        bins = self.bins.to(device=logits.device, dtype=torch.float32)

        # Compress targets to symlog space if requested
        if auto_symlog:
            targets_symlog = symlog(targets_f32)
        else:
            targets_symlog = targets_f32

        targets_clamped = torch.clamp(targets_symlog, self.min_val, self.max_val)

        below = torch.floor((targets_clamped - self.min_val) / self.step_size).long()
        below = torch.clamp(below, 0, self.num_bins - 2)
        above = below + 1

        b_val = self.min_val + below.float() * self.step_size

        # Linear probability weights
        weight_above = (targets_clamped - b_val) / self.step_size
        weight_below = 1.0 - weight_above

        # Construct target probability tensor
        target_probs = torch.zeros_like(logits_f32)
        target_probs.scatter_add_(-1, below.unsqueeze(-1), weight_below.unsqueeze(-1))
        target_probs.scatter_add_(-1, above.unsqueeze(-1), weight_above.unsqueeze(-1))

        # Flatten leading dimensions for fused cross-entropy execution
        flat_logits = logits_f32.view(-1, self.num_bins)
        flat_targets = target_probs.view(-1, self.num_bins)

        loss = F.cross_entropy(flat_logits, flat_targets, reduction='none')

        # Restore original leading dimensions with trailing singleton
        out_shape = list(logits.shape[:-1]) + [1]
        return loss.view(*out_shape).to(dtype=orig_dtype)

    def decode(self, logits: torch.Tensor, real_scale: bool = True) -> torch.Tensor:
        """
        Decodes predicted categorical bin logits into expected continuous scalar values.

        Args:
            logits: Predicted distribution logits of shape (..., num_bins).
            real_scale: If True (default), inverts expected symlog values to real physical scale
                        via symexp(). If False, returns expected value in compressed symlog space.

        Returns:
            Continuous expected scalar tensor of shape (..., 1) with matching input dtype.
        """
        orig_dtype = logits.dtype
        logits_f32 = logits.float()
        bins = self.bins.to(device=logits.device, dtype=torch.float32)

        probs = F.softmax(logits_f32, dim=-1)
        symlog_expectation = torch.sum(probs * bins, dim=-1, keepdim=True)

        if real_scale:
            decoded = symexp(symlog_expectation)
        else:
            decoded = symlog_expectation

        return decoded.to(dtype=orig_dtype)

