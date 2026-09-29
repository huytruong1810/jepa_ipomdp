# ABSOLUTE PATH: src/ipomdp/domain/env.py
# ==============================================================================
# BATCHED, SEEDED SIMULATOR FOR A FinitePOMDP
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. The Simulator Samples From the Specification, Nothing Else:
#    - s_0 ~ b0,  s_{t+1} ~ T(. | s_t, a_t),  o_{t+1} ~ O(. | s_{t+1}, a_t),  r_t = R(s_t, a_t).
#      No dynamics are re-implemented here, so the simulator cannot disagree with the
#      exact Bayes filter or the exact solver (see pomdp.py, section 1).
#
# 2. Interaction Protocol (canonical POMDP timing):
#    - reset() samples s_0 and returns NOTHING observable: the agent's first decision is
#      made from b0 alone. step(a_t) returns (o_{t+1}, r_t, truncated_{t+1}).
#    - The learned agent represents the empty history by the belief filter's learned initial
#      latent z_0 (models/world_model.py, section 1); no placeholder observation exists on
#      either side.
#
# 3. Continuing Task, Artificial Truncation:
#    - A FinitePOMDP has no terminal states. `max_steps` only cuts the infinite interaction
#      into episodes for training-data segmentation and evaluation. Truncation is NOT
#      termination: value targets must bootstrap through it, never zero it.
#    - Rows are not auto-reset. The caller reads the final observation of a truncated row
#      and then calls reset_rows(mask); this keeps the true o_T available without the
#      "terminal_obs" side-channel that auto-resetting wrappers need.
#
# 4. Batched and Reproducible:
#    - All B episodes advance with one tensor operation per quantity; there is no Python
#      loop over environments. All randomness comes from a private torch.Generator seeded
#      at construction, so a (seed, action sequence) pair reproduces a run bit-for-bit
#      on a given device.
#
# 5. Privileged Diagnostics:
#    - `state` exposes the hidden state for diagnostics only (training/rollout.py returns it
#      next to the episodes). It must never be fed to the agent or its losses; the probes of
#      ipomdp.interpretability target the exact posterior, not the hidden state.
# ==============================================================================

from dataclasses import dataclass

import torch
from torch import Tensor

from .pomdp import FinitePOMDP


@dataclass(frozen=True)
class StepOutput:
    """
    Result of one batched environment step.

    Attributes:
        observation: o_{t+1}, int64 tensor of shape (B,).
        reward: r_t = R(s_t, a_t), float32 tensor of shape (B,).
        truncated: True where the episode reached max_steps, bool tensor of shape (B,).
    """

    observation: Tensor
    reward: Tensor
    truncated: Tensor


class BatchedPOMDPEnv:
    """Simulates B independent episodes of a FinitePOMDP in lock-step."""

    def __init__(self, model: FinitePOMDP, batch_size: int, max_steps: int, seed: int, device: torch.device):
        """
        Args:
            model: Exact POMDP specification to sample from.
            batch_size: Number of parallel episodes B.
            max_steps: Truncation length of an episode (>= 1).
            seed: Seed of the private random generator.
            device: Device holding all simulator tensors.
        """
        if batch_size < 1 or max_steps < 1:
            raise ValueError(f"batch_size and max_steps must be >= 1, got {batch_size}, {max_steps}.")
        self.model = model
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.device = device
        self._generator = torch.Generator(device=device)
        self._generator.manual_seed(seed)
        self._transition = model.transition.to(device)
        self._observation = model.observation.to(device)
        self._reward = model.reward.to(device=device, dtype=torch.float32)
        self._initial_belief = model.initial_belief.to(device)
        self._state = torch.empty(batch_size, dtype=torch.int64, device=device)
        self._elapsed = torch.zeros(batch_size, dtype=torch.int64, device=device)
        self.reset()

    @property
    def state(self) -> Tensor:
        """Hidden state s_t of every episode, int64 shape (B,). Diagnostics only."""
        return self._state.clone()

    @property
    def elapsed_steps(self) -> Tensor:
        """Number of steps taken in the current episode of every row, int64 shape (B,)."""
        return self._elapsed.clone()

    def _sample(self, probabilities: Tensor) -> Tensor:
        """Draws one index per row of a (N, K) probability matrix."""
        return torch.multinomial(probabilities, num_samples=1, generator=self._generator).squeeze(-1)

    def state_dict(self) -> dict:
        """Everything needed to continue the simulation bit-for-bit (checkpointing)."""
        return {"generator": self._generator.get_state(), "state": self._state.clone(), "elapsed": self._elapsed.clone()}

    def load_state_dict(self, state: dict) -> None:
        """Restores a state produced by state_dict()."""
        self._generator.set_state(state["generator"])
        self._state = state["state"].to(self.device)
        self._elapsed = state["elapsed"].to(self.device)

    def reset(self) -> None:
        """Starts a new episode in every row: s_0 ~ b0. Emits no observation."""
        self.reset_rows(torch.ones(self.batch_size, dtype=torch.bool, device=self.device))

    def reset_rows(self, mask: Tensor) -> None:
        """
        Starts a new episode in the rows where `mask` is True.

        Args:
            mask: bool tensor of shape (B,).
        """
        count = int(mask.sum())
        if count == 0:
            return
        self._state[mask] = self._sample(self._initial_belief.expand(count, -1))
        self._elapsed[mask] = 0

    def step(self, action: Tensor) -> StepOutput:
        """
        Advances every row by one step.

        Args:
            action: a_t, int64 tensor of shape (B,) with values in [0, |A|).

        Returns:
            StepOutput with o_{t+1}, r_t and the truncation flag.
        """
        if action.shape != (self.batch_size,) or action.dtype != torch.int64:
            raise ValueError(f"action must be int64 of shape ({self.batch_size},), got {action.dtype} {tuple(action.shape)}.")
        if (action < 0).any() or (action >= self.model.num_actions).any():
            raise ValueError(f"action values must lie in [0, {self.model.num_actions}).")
        if (self._elapsed >= self.max_steps).any():
            raise RuntimeError("step() called on a truncated row; call reset_rows() first.")

        reward = self._reward[action, self._state]
        next_state = self._sample(self._transition[action, self._state])
        observation = self._sample(self._observation[action, next_state])

        self._state = next_state
        self._elapsed += 1
        return StepOutput(observation=observation, reward=reward, truncated=self._elapsed >= self.max_steps)
