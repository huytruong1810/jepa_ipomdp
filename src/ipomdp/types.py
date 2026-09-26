# ABSOLUTE PATH: src/ipomdp/types.py
# ==============================================================================
# ACTION TENSOR CONTAINER AND RANK-SAFE ONE-HOT ENCODING
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Scope:
#    - Environment-side containers (State, Observation, StepResult) were removed with the
#      move to the tensor-based domain layer (src/ipomdp/domain). Action remains only for
#      its one-hot encoder, which the world model and heads use; it is reviewed with those
#      networks in Phase 2/3.
#
# 2. Universal Rank-Safe One-Hot Encoding:
#    - Action.to_one_hot handles integer scalars, 1D batches (B,), 2D singletons (B, 1),
#      and 3D temporal sequences (B, T, 1) without dynamic shape failures or rank loss
#      when B = 1.
# ==============================================================================

from dataclasses import dataclass, replace
from typing import Optional

import torch


@dataclass(frozen=True)
class Action:
    """
    Encapsulates the action intent of an agent (A_i).

    Shape Contract:
        data: (B, action_dim) for one-hot vectors, or (B, 1) / (B,) for discrete integer indices.
    """
    data: torch.Tensor

    def __post_init__(self):
        assert isinstance(self.data, torch.Tensor), f"Action data must be torch.Tensor, got {type(self.data)}"
        assert self.data.dim() >= 1, f"Action tensor must possess a leading batch dimension, got shape: {self.data.shape}"

    def to(self, device: torch.device, non_blocking: bool = True) -> 'Action':
        """Transfers action tensor to target hardware device."""
        return replace(self, data=self.data.to(device, non_blocking=non_blocking))

    @staticmethod
    def to_one_hot(
        action_idx: int | torch.Tensor,
        num_classes: int,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Universal rank-preserving one-hot action encoder.

        Guarantees that input shape (B, 1) or (B,) transforms to (B, num_classes) without
        squeezing leading batch dimensions when B = 1.

        Args:
            action_idx: Integer scalar or action index tensor of shape (B, 1), (B,), or (B, T, 1).
            num_classes: Total discrete action choices (|A_i|).
            device: Optional target torch.device.

        Returns:
            One-hot float32 action tensor of shape (..., num_classes).
        """
        # Idempotent guard: return if input is already a valid one-hot float tensor
        if (
            isinstance(action_idx, torch.Tensor)
            and action_idx.is_floating_point()
            and action_idx.dim() >= 2
            and action_idx.size(-1) == num_classes
        ):
            return action_idx.to(device=device, non_blocking=True) if device is not None else action_idx

        if isinstance(action_idx, (int, float)):
            action_idx = torch.tensor([int(action_idx)], device=device, dtype=torch.long)
        elif not isinstance(action_idx, torch.Tensor):
            action_idx = torch.tensor(action_idx, device=device, dtype=torch.long)
        else:
            if device is not None and action_idx.device != device:
                action_idx = action_idx.to(device=device, non_blocking=True)

        # Ensure at least 1D for scalar tensors
        if action_idx.dim() == 0:
            action_idx = action_idx.unsqueeze(0)
        elif action_idx.dim() >= 2 and action_idx.size(-1) == 1:
            action_idx = action_idx.squeeze(-1)

        one_hot = torch.nn.functional.one_hot(
            action_idx.to(dtype=torch.long), num_classes=num_classes
        ).float()

        # Guarantee leading batch dimension exists even for scalar inputs
        if one_hot.dim() == 1:
            one_hot = one_hot.unsqueeze(0)

        return one_hot
