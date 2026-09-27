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
#      Tiger episodes are max_steps long (20), so whole-episode storage is exact and cheap.
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
#      observations[C, T]      o_1 .. o_T      (no o_0: the canonical POMDP emits none)
#      rewards[C, T]           r_0 .. r_{T-1}
#      opponent_actions[C, T]  a^j_0 .. a^j_{T-1}
#    Every stored episode ends by truncation, never termination (continuing task).
# ==============================================================================

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class EpisodeBatch:
    """A batch of complete episodes; every tensor has shape (B, T)."""

    actions: Tensor
    observations: Tensor
    rewards: Tensor
    opponent_actions: Tensor


class EpisodeBuffer:
    """Circular buffer of fixed-length episodes with uniform sampling."""

    def __init__(self, capacity: int, episode_length: int, device: torch.device, seed: int):
        """
        Args:
            capacity: Maximum number of stored episodes C.
            episode_length: Steps per episode T.
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
        self._opponent_actions = torch.zeros(shape, dtype=torch.int64, device=device)
        self._generator = torch.Generator(device=device)
        self._generator.manual_seed(seed)
        self._next = 0
        self.size = 0

    def add(self, episodes: EpisodeBatch) -> None:
        """Stores a batch of complete episodes, overwriting the oldest when full."""
        count = episodes.actions.shape[0]
        if episodes.actions.shape != (count, self.episode_length):
            raise ValueError(f"episodes must have shape (B, {self.episode_length}), got {tuple(episodes.actions.shape)}.")
        slots = (self._next + torch.arange(count, device=self.device)) % self.capacity
        self._actions[slots] = episodes.actions
        self._observations[slots] = episodes.observations
        self._rewards[slots] = episodes.rewards
        self._opponent_actions[slots] = episodes.opponent_actions
        self._next = (self._next + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def sample(self, batch_size: int) -> EpisodeBatch:
        """Uniformly samples `batch_size` stored episodes (with replacement)."""
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty EpisodeBuffer.")
        index = torch.randint(0, self.size, (batch_size,), generator=self._generator, device=self.device)
        return EpisodeBatch(self._actions[index], self._observations[index], self._rewards[index],
                            self._opponent_actions[index])
