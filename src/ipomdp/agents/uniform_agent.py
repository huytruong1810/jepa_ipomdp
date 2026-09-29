# ABSOLUTE PATH: src/ipomdp/agents/uniform_agent.py
# ==============================================================================
# UNIFORMLY RANDOM AGENT
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Purpose:
#    - Fills the replay buffer before planning starts (training/run.py warm-up) and generates
#      broad-coverage histories for the interpretability probes. It keeps no belief; update()
#      exists only to satisfy the rollout protocol (training/rollout.py).
#
# 2. Reproducibility:
#    - Actions come from a private seeded generator whose state is checkpointed.
# ==============================================================================

import torch
from torch import Tensor


class UniformRandomAgent:
    """Chooses every action uniformly at random, independently per episode and step."""

    def __init__(self, num_actions: int, batch_size: int, seed: int, device: torch.device):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.device = device
        self._generator = torch.Generator(device=device)
        self._generator.manual_seed(seed)

    def reset(self) -> None:
        """Nothing to reset: the agent is memoryless."""

    def act(self) -> Tensor:
        """a_t ~ Uniform(A), int64 shape (B,)."""
        return torch.randint(0, self.num_actions, (self.batch_size,), generator=self._generator, device=self.device)

    def update(self, action: Tensor, observation: Tensor) -> None:
        """Memoryless: observations are ignored."""

    def state_dict(self) -> dict:
        return {"generator": self._generator.get_state()}

    def load_state_dict(self, state: dict) -> None:
        self._generator.set_state(state["generator"])
