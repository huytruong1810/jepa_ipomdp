# ABSOLUTE PATH: src/ipomdp/models/distributions.py
# ==============================================================================
# TWO-HOT DISTRIBUTIONAL REGRESSION ON SYMLOG-SPACED BINS
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Purpose:
#    - Value and reward heads predict a categorical distribution over K fixed bins; the
#      prediction used by the planner and the Bellman value targets is the distribution's MEAN. The
#      categorical cross-entropy is scale-free, so rewards of -100 and -1 train equally
#      well without return normalisation (DreamerV3, Hafner et al. 2023).
#
# 2. Bins Are Symlog-Spaced, Live in Real Space, and Are Bounded by the Domain:
#      B = symexp(linspace(-R, +R, K)),   R = symlog(max_abs_value),  K = num_bins (odd)
#    - Resolution is fine near zero and coarse for large magnitudes, as in DreamerV3. The
#      bins are built as an exactly antisymmetric float64 grid with an exact 0 bin.
#    - max_abs_value must bound every target. For a FinitePOMDP it is exact:
#      |r| <= max|R| and |V| <= max|R| / (1 - gamma) (2000 for canonical Tiger).
#    - Why bounded: DreamerV3's fixed R = 20 puts bins at +-4.85e8. In real space the mean
#      is then dominated by the residual softmax mass on those bins (1e-7 of probability on
#      4.85e8 shifts the mean by 48), measured as a -3.8 bias on the -100/+10 test below and
#      a 0.5 offset for uniform logits in float32. Bounding the grid by the domain's value
#      bound removes both effects without introducing any tunable range.
#
# 3. Encoding and Decoding Are Both Linear in Real Space (the unbiased form):
#      twohot(y): weight (B_{k+1} - y) / (B_{k+1} - B_k) on bin k and the rest on bin k+1,
#                 where B_k <= y < B_{k+1}. Its mean  sum_i twohot(y)_i B_i  equals y exactly.
#      mean(logits) = softmax(logits) . B
#    - The cross-entropy minimiser for a random target Y is p = E[twohot(Y)], whose mean is
#      E[Y] by linearity: the decoded prediction is an unbiased estimate of the expected
#      reward/return even when Y is multimodal.
#    - An earlier iteration interpolated in symlog space and decoded symexp(E[symlog Y]).
#      That is Jensen-biased toward the median in symlog space: fitted to Tiger's
#      door-opening reward (-100 or +10 with probability 1/2, mean -45) it decoded -2.9, and
#      on a trained model the expected reward of opening a door was off by ~39 on average.
#      The planner therefore saw a -45 gamble as nearly free. This form removes that bias.
#
# 4. Strictness:
#    - Non-finite targets raise FloatingPointError (searchsorted of NaN is an arbitrary index
#      and surfaced as unrelated CUDA device asserts); so do targets outside
#      [-max_abs_value, max_abs_value], which would violate the bound in section 2.
#    - All arithmetic is float32 regardless of autocast.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TwoHotSymlog(nn.Module):
    """Two-hot categorical codec over symlog-spaced real-valued bins (module header)."""

    def __init__(self, num_bins: int, max_abs_value: float):
        """
        Args:
            num_bins: Number of bins K (odd, so that 0 is exactly representable).
            max_abs_value: Bound on |target|; the outermost bins sit at +-max_abs_value.
        """
        super().__init__()
        if num_bins < 3 or num_bins % 2 == 0:
            raise ValueError(f"num_bins must be an odd number >= 3, got {num_bins}.")
        if max_abs_value <= 0:
            raise ValueError(f"max_abs_value must be positive, got {max_abs_value}.")
        self.num_bins = num_bins
        self.max_abs_value = max_abs_value
        symlog_max = torch.log1p(torch.tensor(max_abs_value, dtype=torch.float64))
        half = torch.expm1(torch.linspace(0.0, 1.0, (num_bins + 1) // 2, dtype=torch.float64) * symlog_max)
        half[-1] = max_abs_value
        self.register_buffer("bins", torch.cat([-half[1:].flip(0), half]).float())

    def encode(self, targets: Tensor) -> Tensor:
        """
        Two-hot encoding with an exact mean.

        Args:
            targets: Real-valued targets of any shape (...).

        Returns:
            Probabilities of shape (..., K) with sum_i p_i B_i = targets.
        """
        targets = targets.float()
        if not torch.isfinite(targets).all():
            raise FloatingPointError("TwoHotSymlog received non-finite targets.")
        if (targets.abs() > self.max_abs_value).any():
            raise FloatingPointError(f"TwoHotSymlog target outside +-{self.max_abs_value}: {float(targets.abs().max())}.")
        above = torch.searchsorted(self.bins, targets.contiguous(), right=True).clamp(1, self.num_bins - 1)
        below = above - 1
        weight_above = (targets - self.bins[below]) / (self.bins[above] - self.bins[below])
        probabilities = torch.zeros(*targets.shape, self.num_bins, device=targets.device)
        probabilities.scatter_(-1, below.unsqueeze(-1), (1.0 - weight_above).unsqueeze(-1))
        probabilities.scatter_add_(-1, above.unsqueeze(-1), weight_above.unsqueeze(-1))
        return probabilities

    def loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Cross-entropy between softmax(logits) and twohot(targets).

        Args:
            logits: Shape (..., K).
            targets: Real-valued targets of shape (...).

        Returns:
            Per-element loss of shape (...), float32.
        """
        return -(self.encode(targets) * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1)

    def mean(self, logits: Tensor) -> Tensor:
        """Mean of the predicted distribution, softmax(logits) . B; shape (...), float32."""
        return F.softmax(logits.float(), dim=-1) @ self.bins
