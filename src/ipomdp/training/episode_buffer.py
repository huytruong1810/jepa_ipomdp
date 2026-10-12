# ABSOLUTE PATH: src/ipomdp/training/episode_buffer.py
# ==============================================================================
# FIXED-LENGTH EPISODE REPLAY BUFFER
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Whole Episodes, Not Chunks:
#    - The recurrent belief filter is trained on complete episodes starting from its learned
#      z_0, exactly as the agent runs it online. The former chunked buffer started mid-episode
#      chunks from a zero belief with a short burn-in, so the trained filter saw histories the
#      online filter never sees (and lost evidence older than the burn-in window). Canonical
#      Tiger episodes are env.max_steps = 100 steps long, so whole-episode storage is exact and cheap.
#
# 2. Storage on the Training Device, Uniform Sampling:
#    - Episodes are stored as dense int64/float32 tensors on the same device as the model:
#      no host round-trips, and in particular no device-to-host copies (a non-blocking copy
#      in the former buffer returned stale memory and corrupted stored transitions).
#    - Sampling is uniform with a private seeded generator. Prioritised replay was removed
#      together with the chunked buffer; it will be reintroduced only if the Phase-5
#      experiments show a measurable benefit that justifies importance-sampling corrections.
#
# 3. Layout (C = capacity, T = episode length):
#      actions[C, T]           a_0 .. a_{T-1}
#      observations[C, T]      o_1 .. o_T
#      rewards[C, T]           r_0 .. r_{T-1}
#      initial_observations[C] o_0             only for domains that emit one (Wumpus World);
#                                              canonical Tiger emits none and stores None
#      greedy[C]               whether the episode was played by the greedy evaluation planner
#    Every stored episode ends by truncation, never termination (continuing task).
#
# 4. Episode Source (2026-10-08):
#    - add(episodes, greedy) records whether a batch came from the greedy evaluation planner
#      (training/run.py, section 8) or from collection. sample() returns the flag with the batch.
#      The trainer's optional n-step greedy-return value target (trainer.py, section 3c) uses it;
#      nothing else reads it, and storing it draws no random numbers.
#
# 5. Slot Indices (2026-10-11):
#    - add() returns the slots it wrote and sample() returns the slots it drew (EpisodeBatch.index).
#      The privileged exact-belief term (training/privileged.py, section 2c) caches its targets per
#      slot with them: the targets are a fixed function of the stored episode, and recomputing the
#      exact filter on every update cost 39 s per 64-update collection (0.6 s per update).
#    - Returning indices draws no random numbers and changes no stored tensor.
# ==============================================================================

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class EpisodeBatch:
    """
    A batch of complete episodes: (B, T) tensors and o_0 of shape (B,) or None (section 3).

    greedy (B,) bool and index (B,) int64 (the buffer slots, section 5) are set on batches sampled
    from an EpisodeBuffer (section 4); freshly played episodes carry None.
    """

    actions: Tensor
    observations: Tensor
    rewards: Tensor
    initial_observations: Tensor | None
    greedy: Tensor | None = None
    index: Tensor | None = None


class EpisodeBuffer:
    """Circular buffer of fixed-length episodes with uniform sampling."""

    def __init__(self, capacity: int, episode_length: int, initial_observation: bool, device: torch.device,
                 seed: int):
        """
        Args:
            capacity: Maximum number of stored episodes C.
            episode_length: Steps per episode T.
            initial_observation: Whether the domain emits o_0 (stored per episode).
            device: Device holding the storage and the sampling generator.
            seed: Seed of the sampling generator.
        """
        self.capacity = capacity
        self.episode_length = episode_length
        self.device = device
        shape = (capacity, episode_length)
        self._actions = torch.zeros(shape, dtype=torch.int64, device=device)
        self._observations = torch.zeros(shape, dtype=torch.int64, device=device)
        self._rewards = torch.zeros(shape, dtype=torch.float32, device=device)
        self._initial_observations = (torch.zeros(capacity, dtype=torch.int64, device=device)
                                      if initial_observation else None)
        self._greedy = torch.zeros(capacity, dtype=torch.bool, device=device)
        self._generator = torch.Generator(device=device)
        self._generator.manual_seed(seed)
        self._next = 0
        self.size = 0

    def add(self, episodes: EpisodeBatch, greedy: bool) -> Tensor:
        """
        Stores a batch of complete episodes, overwriting the oldest when full; greedy: their source (section 4).

        Returns:
            The slots written, (B,) int64 (section 5).
        """
        count = episodes.actions.shape[0]
        if episodes.actions.shape != (count, self.episode_length):
            raise ValueError(f"episodes must have shape (B, {self.episode_length}), got {tuple(episodes.actions.shape)}.")
        if (episodes.initial_observations is None) != (self._initial_observations is None):
            raise ValueError("episodes and buffer disagree on whether the domain emits an initial observation.")
        slots = (self._next + torch.arange(count, device=self.device)) % self.capacity
        self._actions[slots] = episodes.actions
        self._observations[slots] = episodes.observations
        self._rewards[slots] = episodes.rewards
        if self._initial_observations is not None:
            self._initial_observations[slots] = episodes.initial_observations
        self._greedy[slots] = greedy
        self._next = (self._next + count) % self.capacity
        self.size = min(self.size + count, self.capacity)
        return slots

    def state_dict(self) -> dict:
        """Stored episodes, write position and sampling-generator state (checkpointing)."""
        return {"actions": self._actions, "observations": self._observations, "rewards": self._rewards,
                "initial_observations": self._initial_observations, "greedy": self._greedy, "next": self._next, "size": self.size, "generator": self._generator.get_state()}

    def load_state_dict(self, state: dict) -> None:
        """Restores a state produced by state_dict()."""
        self._actions = state["actions"].to(self.device)
        self._observations = state["observations"].to(self.device)
        self._rewards = state["rewards"].to(self.device)
        if (state["initial_observations"] is None) != (self._initial_observations is None):
            raise ValueError("checkpoint and buffer disagree on whether the domain emits an initial observation.")
        if self._initial_observations is not None:
            self._initial_observations = state["initial_observations"].to(self.device)
        self._greedy = state["greedy"].to(self.device)
        self._next, self.size = state["next"], state["size"]
        self._generator.set_state(state["generator"])

    def sample(self, batch_size: int) -> EpisodeBatch:
        """Uniformly samples `batch_size` stored episodes (with replacement)."""
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty EpisodeBuffer.")
        index = torch.randint(0, self.size, (batch_size,), generator=self._generator, device=self.device)
        initial = None if self._initial_observations is None else self._initial_observations[index]
        return EpisodeBatch(self._actions[index], self._observations[index], self._rewards[index], initial,
                            self._greedy[index], index)
