# ABSOLUTE PATH: src/ipomdp/types.py
# ==============================================================================
# IMMUTABLE TENSOR DATA STRUCTURES & BATCH CONTRACT VALIDATORS
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Structural Tensor Invariant Contracts:
#    - All domain dataclasses (State, Observation, Action, StepResult) are strictly
#      immutable (frozen=True) and validate leading batch dimension presence (B >= 1)
#      in __post_init__ to catch unbatched tensors before neural propagation.
#
# 2. Non-Blocking Device Migration:
#    - Provides unified, recursive .to(device, non_blocking=True) methods to streamline
#      asynchronous PCIe memory transfers between host pinned RAM and GPU VRAM.
#
# 3. Universal Rank-Safe One-Hot Encoding:
#    - Action.to_one_hot handles integer scalars, 1D batches (B,), 2D singletons (B, 1),
#      and 3D temporal sequences (B, T, 1) without dynamic shape failures or rank loss
#      when B = 1.
#
# 4. Complete Recursive StepResult Device Migration:
#    - StepResult.to recursively transfers all enclosed dictionary fields including
#      observations, rewards, terminations, truncations, and tensor metadata in infos
#      (e.g., terminal_obs, terminal_state, true_state) to prevent host-device PCIe drops.
# ==============================================================================

from dataclasses import dataclass, replace
from typing import Dict, TypeAlias, Optional, Any
import torch

AgentID: TypeAlias = str


@dataclass(frozen=True)
class State:
    """
    Encapsulates the true, hidden global environment state (S).

    Shape Contract:
        data: (B, *state_shape) where B >= 1 is the number of parallel environment channels.
    """
    data: torch.Tensor

    def __post_init__(self):
        assert isinstance(self.data, torch.Tensor), f"State data must be torch.Tensor, got {type(self.data)}"
        assert self.data.dim() >= 1, f"State tensor must possess a leading batch dimension, got shape: {self.data.shape}"

    def to(self, device: torch.device, non_blocking: bool = True) -> 'State':
        """Transfers state tensor to target hardware device."""
        return replace(self, data=self.data.to(device, non_blocking=non_blocking))


@dataclass(frozen=True)
class Observation:
    """
    Encapsulates a masked local observation available to an agent (O_i).

    Shape Contract:
        data: (B, *obs_shape) where B >= 1 is the number of parallel environment channels.
    """
    data: torch.Tensor

    def __post_init__(self):
        assert isinstance(self.data, torch.Tensor), f"Observation data must be torch.Tensor, got {type(self.data)}"
        assert self.data.dim() >= 1, f"Observation tensor must possess a leading batch dimension, got shape: {self.data.shape}"

    def to(self, device: torch.device, non_blocking: bool = True) -> 'Observation':
        """Transfers observation tensor to target hardware device."""
        return replace(self, data=self.data.to(device, non_blocking=non_blocking))


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


@dataclass(frozen=True)
class StepResult:
    """
    Standardized result container returned by environment step operations.
    All enclosed tensors adhere to the (B, ...) leading batch contract.
    """
    observations: Dict[AgentID, Observation]
    rewards: Dict[AgentID, torch.Tensor]
    terminations: Dict[AgentID, torch.Tensor]
    truncations: Dict[AgentID, torch.Tensor]
    infos: Dict[AgentID, dict]

    def __iter__(self):
        yield self.observations
        yield self.rewards
        yield self.terminations
        yield self.truncations
        yield self.infos

    def to(self, device: torch.device, non_blocking: bool = True) -> 'StepResult':
        """
        Recursively transfers all enclosed observation, reward, termination, truncation,
        and info metadata tensors to target hardware device in a single operation.
        """
        def _migrate(val: Any) -> Any:
            if isinstance(val, (torch.Tensor, State, Observation, Action)):
                return val.to(device, non_blocking=non_blocking)
            if isinstance(val, dict):
                return {k: _migrate(v) for k, v in val.items()}
            if isinstance(val, list):
                return [_migrate(v) for v in val]
            if isinstance(val, tuple):
                return tuple(_migrate(v) for v in val)
            return val

        new_obs = {
            agent_id: obs.to(device, non_blocking=non_blocking)
            for agent_id, obs in self.observations.items()
        }
        new_rews = {
            agent_id: rew.to(device, non_blocking=non_blocking)
            for agent_id, rew in self.rewards.items()
        }
        new_terms = {
            agent_id: term.to(device, non_blocking=non_blocking)
            for agent_id, term in self.terminations.items()
        }
        new_truncs = {
            agent_id: trunc.to(device, non_blocking=non_blocking)
            for agent_id, trunc in self.truncations.items()
        }
        new_infos = {
            agent_id: _migrate(info_dict)
            for agent_id, info_dict in self.infos.items()
        }

        return replace(
            self,
            observations=new_obs,
            rewards=new_rews,
            terminations=new_terms,
            truncations=new_truncs,
            infos=new_infos
        )

