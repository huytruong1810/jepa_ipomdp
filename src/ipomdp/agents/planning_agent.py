# ABSOLUTE PATH: src/ipomdp/agents/planning_agent.py
# ==============================================================================
# BATCHED PLANNING AGENT (BELIEF TRACKING + BELIEF-TREE SEARCH)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. One Agent for Every Model:
#    - The agent tracks its state with the SearchModel's own filter and plans with a
#      BeliefTreeSearch over the same model (planning/search_model.py). With
#      ExactSearchModel it is an exact-belief planner (the Phase-4 reference); with
#      LearnedSearchModel it is the JEPA agent. Nothing else differs, so any gap between the
#      two measures the learned model, not the agent code.
#
# 2. Canonical POMDP Timing:
#    - reset() sets every row to the model's prior state (b0, or the filter's learned z_0);
#      the domain emits no observation before the first decision. Each step:
#         a_t = act()  ->  environment returns o_{t+1}  ->  update(a_t, o_{t+1}).
#
# 3. Exploration:
#    - act() samples from the visit-count distribution shaped by `temperature` (0 = greedy);
#      act_uniformly() is used to fill the replay buffer before planning starts. Root
#      Dirichlet noise is a property of the planner (training vs evaluation planners).
# ==============================================================================

import torch
from torch import Tensor

from ..planning.mcts import BeliefTreeSearch
from ..planning.search_model import SearchModel


class PlanningAgent:
    """Tracks B parallel beliefs with a SearchModel and acts by BeliefTreeSearch."""

    def __init__(self, model: SearchModel, planner: BeliefTreeSearch, batch_size: int, temperature: float,
                 seed: int, device: torch.device):
        """
        Args:
            model: Belief tracker and planning model.
            planner: Search over `model`.
            batch_size: Number of parallel episodes B.
            temperature: Visit-count sampling temperature (0 = greedy).
            seed: Seed of the action-sampling generator.
            device: Device of the action tensors.
        """
        self.model = model
        self.planner = planner
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self._generator = torch.Generator(device=device)
        self._generator.manual_seed(seed)
        self.state = model.initial_states(batch_size)

    def state_dict(self) -> dict:
        """Temperature and action-sampling generator state (checkpointing)."""
        return {"temperature": self.temperature, "generator": self._generator.get_state(),
                "planner": self.planner.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        """Restores a state produced by state_dict()."""
        self.temperature = state["temperature"]
        self._generator.set_state(state["generator"])
        self.planner.load_state_dict(state["planner"])

    def reset(self) -> None:
        """Starts new episodes in every row from the prior state."""
        self.state = self.model.initial_states(self.batch_size)

    def update(self, action: Tensor, observation: Tensor) -> None:
        """Filter step with (a_t, o_{t+1}); int64 tensors of shape (B,)."""
        self.state = self.model.update(self.state, action, observation)

    def act(self) -> Tensor:
        """Plans from the current states and samples a_t; returns int64 shape (B,)."""
        policy = self.planner.search(self.state, self.temperature)
        return torch.multinomial(policy, num_samples=1, generator=self._generator).squeeze(-1)

    def act_uniformly(self) -> Tensor:
        """Uniformly random a_t (replay warm-up); returns int64 shape (B,)."""
        return torch.randint(0, self.model.num_actions, (self.batch_size,), generator=self._generator,
                             device=self.device)
