# ABSOLUTE PATH: src/ipomdp/agents/jepa_agent.py
# ==============================================================================
# BATCHED JEPA + LATENT-MCTS AGENT
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Canonical POMDP Timing:
#    - reset() sets z_0, the filter's learned representation of the prior b0 (the domain
#      emits no observation before the first decision). Each step: a_t = act() plans from
#      z_t; after the environment returns o_{t+1}, update(a_t, o_{t+1}) applies
#         z_{t+1} = BeliefFilter.step(z_t, a_t, o_{t+1}),
#      the learned counterpart of the exact update b_{t+1} = tau(b_t, a_t, o_{t+1}).
#
# 2. Opponent Lives in the Environment:
#    - The agent acts in the (possibly opponent-folded) POMDP presented by the simulator and
#      never receives the opponent's action as an input; multi-agent reasoning enters only
#      through the learned opponent head used by the planner.
#
# 3. Plain Tensors at the Boundary:
#    - Actions and observations are int64 index tensors of shape (B,); the agent one-hots
#      them for the filter.
# ==============================================================================

import torch
import torch.nn.functional as F
from torch import Tensor

from ..interfaces import AbstractPlanner
from ..models.world_model import BeliefFilter


class DiscreteJEPAAgent:
    """Batched agent that filters beliefs with the JEPA belief filter and acts by latent MCTS."""

    def __init__(
        self,
        belief_filter: BeliefFilter,
        planner: AbstractPlanner,
        batch_size: int,
        num_actions: int,
        num_observations: int,
        device: torch.device,
        temperature: float,
        temperature_min: float,
        temperature_decay: float,
    ):
        """
        Args:
            belief_filter: Online belief filter of the world model.
            planner: Latent planner providing search().
            batch_size: Number of parallel episodes B.
            num_actions: |A|.
            num_observations: |O|.
            device: Device of the belief tensors.
            temperature: Initial MCTS visit-count sampling temperature.
            temperature_min: Floor of the annealed temperature.
            temperature_decay: Multiplicative decay applied by anneal_temperature().
        """
        self.belief_filter = belief_filter
        self.planner = planner
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.num_observations = num_observations
        self.device = device
        self.temperature = temperature
        self.temperature_min = temperature_min
        self.temperature_decay = temperature_decay
        self.last_root_value = 0.0
        self.belief = torch.empty(0)
        self.reset()

    @torch.no_grad()
    def reset(self) -> None:
        """Starts new episodes in every row: z = z_0."""
        self.belief = self.belief_filter.initial(self.batch_size).clone()

    @torch.no_grad()
    def update(self, action: Tensor, observation: Tensor) -> None:
        """
        Advances the belief filter with (a_t, o_{t+1}).

        Args:
            action: a_t, int64 shape (B,).
            observation: o_{t+1}, int64 shape (B,).
        """
        self.belief = self.belief_filter.step(
            self.belief,
            F.one_hot(action, self.num_actions).float(),
            F.one_hot(observation, self.num_observations).float(),
        )

    def act(self) -> Tensor:
        """Plans from the current belief and samples a_t; returns int64 shape (B,)."""
        policy = self.planner.search(root_state=self.belief, temperature=self.temperature)
        self.last_root_value = float(sum(root.value for root in self.planner.roots) / len(self.planner.roots))
        return torch.multinomial(policy, num_samples=1).squeeze(-1)

    def act_uniformly(self) -> Tensor:
        """Samples a_t uniformly at random (buffer warm-up); returns int64 shape (B,)."""
        return torch.randint(0, self.num_actions, (self.batch_size,), device=self.device)

    def anneal_temperature(self) -> float:
        """Decays the sampling temperature towards its floor and returns the new value."""
        self.temperature = max(self.temperature_min, self.temperature * self.temperature_decay)
        return self.temperature
