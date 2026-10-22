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
#
# 4. Own Turn-then-Step Trials (trial_prob; agent.trial_prob, 2026-10-22; training/run.py, section 17):
#    - With a RiskSpec (planning/search_model.py, section 6) and trial_prob > 0, a row whose action is not
#      the step action (Wumpus: FORWARD) and whose epsilon draw did not fire plays, with probability
#      trial_prob, the turn with the higher root Q (ties: the first turn), and the step action at its next
#      decision. The pending step overrides everything at that decision (planner, epsilon, a new trial).
#    - Why: the agent's model is wrong where the agent does not go (the feedback loop; feature_localise.py,
#      coverage_games.py). Trials the agent picks itself, unfiltered, put the deaths it risks into its data
#      (model-based DAgger, Ross & Bagnell 2012; temporally extended epsilon, Dabney et al. 2021), and keep
#      the risk calibrated (risk_localise.py: no selection bias), unlike safety-filtered privileged data.
#      Offline (coverage_games.py, "own" arm, which this reproduces in-agent) they played +66 over best.pt on
#      5 of 5 seeds, from safety; with the risk margin +74 over control.
#    - The trial draw comes from the same generator after the epsilon draws; with trial_prob = 0 nothing is
#      drawn. Pending steps are cleared by reset(); collections play whole episodes, so none is pending at a
#      checkpoint and the agent's state_dict is unchanged.
#    - trials counts the trials started since the last reset_trial_count() (a dose metric).
# ==============================================================================

import torch
from torch import Tensor

from ..planning.mcts import BeliefTreeSearch
from ..planning.search_model import RiskSpec, SearchModel


class PlanningAgent:
    """Tracks B parallel beliefs with a SearchModel and acts by BeliefTreeSearch."""

    def __init__(self, model: SearchModel, planner: BeliefTreeSearch, batch_size: int, temperature: float,
                 seed: int, device: torch.device, epsilon: float = 0.0, trial: RiskSpec | None = None,
                 trial_prob: float = 0.0):
        """
        Args:
            model: Belief tracker and planning model.
            planner: Search over `model`.
            batch_size: Number of parallel episodes B.
            temperature: Visit-count sampling temperature (0 = greedy).
            seed: Seed of the action-sampling generator.
            device: Device of the action tensors.
            epsilon: Probability of replacing each sampled action by a uniform one (0 = off).
            trial: The domain's step and turn actions (section 4); needed when trial_prob > 0.
            trial_prob: Probability of a turn-then-step trial at a non-step decision (0 = off).
        """
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must lie in [0, 1], got {epsilon}.")
        if not 0.0 <= trial_prob <= 1.0:
            raise ValueError(f"trial_prob must lie in [0, 1], got {trial_prob}.")
        if trial_prob > 0.0 and trial is None:
            raise ValueError("trial_prob > 0 needs the domain's RiskSpec (step and turn actions).")
        self.model = model
        self.planner = planner
        self.batch_size = batch_size
        self.temperature = temperature
        self.epsilon = epsilon
        self.trial, self.trial_prob = trial, trial_prob
        self.device = device
        self.trials = 0
        self._pending = torch.zeros(batch_size, dtype=torch.bool, device=device)   # step action due next (section 4)
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
        self._pending.zero_()

    def reset_trial_count(self) -> int:
        """Returns the trials started since the last call and zeroes the count (section 4)."""
        count, self.trials = self.trials, 0
        return count

    def update(self, action: Tensor, observation: Tensor) -> None:
        """Filter step with (a_t, o_{t+1}); int64 tensors of shape (B,)."""
        self.state = self.model.update(self.state, action, observation)

    def act(self) -> Tensor:
        """Plans from the current states and samples a_t; returns int64 shape (B,)."""
        if self.state is None:
            raise RuntimeError("act() before reset(): the agent has no belief state.")
        policy = self.planner.search(self.state, self.temperature)
        action = torch.multinomial(policy, num_samples=1, generator=self._generator).squeeze(-1)
        explore = torch.zeros_like(action, dtype=torch.bool)
        if self.epsilon > 0.0:
            explore = torch.rand(action.shape, generator=self._generator, device=self.device) < self.epsilon
            uniform = torch.randint(0, policy.shape[-1], action.shape, generator=self._generator, device=self.device)
            action = torch.where(explore, uniform, action)
        if self.trial_prob > 0.0:
            action = self._trial(action, explore)
        return action

    def _trial(self, action: Tensor, explore: Tensor) -> Tensor:
        """Section 4: pending steps first, then new turn-then-step trials."""
        step, (first, second) = self.trial.step_action, self.trial.turn_actions
        draw = torch.rand(action.shape, generator=self._generator, device=self.device) < self.trial_prob
        pending = self._pending
        start = draw & ~pending & ~explore & (action != step)
        q = torch.as_tensor(self.planner.statistics.q_values, device=self.device)
        turn = torch.where(q[:, first] >= q[:, second], first, second)
        action = torch.where(pending, step, torch.where(start, turn, action))
        self._pending = start
        self.trials += int(start.sum())
        return action

