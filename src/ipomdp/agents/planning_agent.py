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
#    - act() samples from the visit-count distribution shaped by `temperature` (0 = greedy). Root
#      Dirichlet noise is a property of the planner (training vs evaluation planners). Replay
#      warm-up uses UniformRandomAgent (agents/uniform_agent.py), not this class.
#    - epsilon > 0 replaces each row's sampled action, independently with probability epsilon,
#      by a uniformly random one (epsilon-uniform exploration; training/run.py, section 7).
#      Visit-count sampling alone rarely tries an action the model already scores badly, so the
#      model never sees the outcomes that would correct it (Wumpus: dangerous FORWARD steps are
#      4.7% of FORWARD transitions, and the observation head's death prediction stays
#      miscalibrated). With epsilon = 0 no extra random numbers are drawn, so runs without it
#      (canonical Tiger) are bit-identical to runs before the option existed.
# ==============================================================================

import torch
from torch import Tensor

from ..planning.mcts import BeliefTreeSearch
from ..planning.search_model import SearchModel


class PlanningAgent:
    """Tracks B parallel beliefs with a SearchModel and acts by BeliefTreeSearch."""

    def __init__(self, model: SearchModel, planner: BeliefTreeSearch, batch_size: int, temperature: float,
                 seed: int, device: torch.device, epsilon: float = 0.0):
        """
        Args:
            model: Belief tracker and planning model.
            planner: Search over `model`.
            batch_size: Number of parallel episodes B.
            temperature: Visit-count sampling temperature (0 = greedy).
            seed: Seed of the action-sampling generator.
            device: Device of the action tensors.
            epsilon: Probability of replacing each sampled action by a uniform one (0 = off).
        """
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must lie in [0, 1], got {epsilon}.")
        self.model = model
        self.planner = planner
        self.batch_size = batch_size
        self.temperature = temperature
        self.epsilon = epsilon
        self.device = device
        self._generator = torch.Generator(device=device)
        self._generator.manual_seed(seed)
        self.state: Tensor | None = None      # set by reset() at the start of every episode

    def state_dict(self) -> dict:
        """Temperature, epsilon and action-sampling generator state (checkpointing)."""
        return {"temperature": self.temperature, "epsilon": self.epsilon, "generator": self._generator.get_state(),
                "planner": self.planner.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        """Restores a state produced by state_dict()."""
        self.temperature = state["temperature"]
        self.epsilon = state["epsilon"]
        self._generator.set_state(state["generator"])
        self.planner.load_state_dict(state["planner"])

    def reset(self, initial_observation: Tensor | None) -> None:
        """Starts new episodes in every row from the model's prior state, conditioned on o_0 if any."""
        self.state = self.model.initial_states(self.batch_size, initial_observation)

    def update(self, action: Tensor, observation: Tensor) -> None:
        """Filter step with (a_t, o_{t+1}); int64 tensors of shape (B,)."""
        self.state = self.model.update(self.state, action, observation)

    def act(self) -> Tensor:
        """Plans from the current states and samples a_t; returns int64 shape (B,)."""
        if self.state is None:
            raise RuntimeError("act() before reset(): the agent has no belief state.")
        policy = self.planner.search(self.state, self.temperature)
        action = torch.multinomial(policy, num_samples=1, generator=self._generator).squeeze(-1)
        if self.epsilon > 0.0:
            explore = torch.rand(action.shape, generator=self._generator, device=self.device) < self.epsilon
            uniform = torch.randint(0, policy.shape[-1], action.shape, generator=self._generator, device=self.device)
            action = torch.where(explore, uniform, action)
        return action

