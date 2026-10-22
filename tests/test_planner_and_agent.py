# ABSOLUTE PATH: tests/test_planner_and_agent.py
"""
Phase-4 checks for the belief-tree search and the planning agent.

The search is validated on the EXACT model (ExactSearchModel over the canonical Tiger), with the
exact solver's value functions as leaf estimates, so every expected number is exact and no
learned component is involved. The learned model is then checked only for interface
consistency (its planning quality is measured by the slow acceptance tests).
"""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ipomdp.agents import PlanningAgent
from ipomdp.domain import (BatchedPOMDPEnv, TigerAction, belief_update, build_tiger_pomdp, observation_distribution,
                           solve_finite_horizon, solve_infinite_horizon)
from ipomdp.models import BeliefFilter, HorizonValueHead, ObservationHead, RewardHead, TwoHotSymlog, ValueHead
from ipomdp.planning import (BeliefTreeSearch, ExactSearchModel, Expansion, LearnedSearchModel, MinMaxStats, RiskMarginModel,
                             RiskSpec, SearchStatistics)

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def tiger():
    return build_tiger_pomdp()


@pytest.fixture(scope="module")
def value_functions(tiger):
    return solve_finite_horizon(tiger, 11)


def _beliefs(*p_left: float) -> torch.Tensor:
    p = torch.tensor(p_left, dtype=torch.float64)
    return torch.stack([p, 1 - p], dim=-1)


def _exact_q(tiger, leaf, beliefs):
    """Q(b, a) = b . R[a] + gamma sum_o P(o | b, a) V_leaf(tau(b, a, o)), shape (N, A)."""
    rows = []
    for a in range(tiger.num_actions):
        action = torch.full((len(beliefs),), a)
        probs = observation_distribution(tiger, beliefs, action)
        q = beliefs @ tiger.reward[a]
        for o in range(tiger.num_observations):
            posterior = belief_update(tiger, beliefs, action, torch.full((len(beliefs),), o))
            q = q + tiger.discount * probs[:, o] * leaf.value(posterior)
        rows.append(q)
    return torch.stack(rows, dim=-1)


class TestExactSearchModel:

    def test_expansion_matches_exact_filter(self, tiger, value_functions):
        model = ExactSearchModel(tiger, value_functions[4], CPU)
        beliefs = _beliefs(0.5, 0.85, 0.03)
        expansion = model.expand(beliefs)
        for a in range(tiger.num_actions):
            action = torch.full((3,), a)
            assert torch.allclose(expansion.rewards[:, a], beliefs @ tiger.reward[a])
            assert torch.allclose(expansion.observation_probs[:, a], observation_distribution(tiger, beliefs, action))
            for o in range(tiger.num_observations):
                posterior = belief_update(tiger, beliefs, action, torch.full((3,), o))
                assert torch.allclose(expansion.next_states[:, a, o], posterior)
                assert torch.allclose(expansion.next_values[:, a, o], value_functions[4].value(posterior))

    def test_zero_probability_observations_yield_valid_beliefs(self, tiger):
        # Tiger's emissions are never deterministic, so build a model whose LISTEN observation is
        # exact: from a certain belief the "wrong" growl then has probability 0.
        certain = _beliefs(1.0)
        deterministic = type(tiger)(
            transition=tiger.transition, observation=torch.stack([torch.eye(2, dtype=torch.float64)] * 3),
            reward=tiger.reward, initial_belief=tiger.initial_belief, discount=tiger.discount,
            state_names=tiger.state_names, action_names=tiger.action_names,
            observation_names=tiger.observation_names)
        expansion = ExactSearchModel(deterministic, None, CPU).expand(certain)
        assert expansion.observation_probs[0, TigerAction.LISTEN, 1] == 0.0
        assert torch.allclose(expansion.next_states.sum(-1), torch.ones(1, 3, 2, dtype=torch.float64))
        assert torch.equal(expansion.next_values, torch.zeros(1, 3, 2, dtype=torch.float64))


class TestBeliefTreeSearch:

    def test_root_q_equals_exact_one_step_lookahead_before_search(self, tiger, value_functions):
        # Expanding the root alone gives R + gamma * E[V_h(tau)] exactly, i.e. Q_{h+1}.
        leaf = value_functions[4]
        beliefs = _beliefs(0.5, 0.85, 0.97, 0.03)
        planner = BeliefTreeSearch(ExactSearchModel(tiger, leaf, CPU), num_simulations=1, c_puct=1.25,
                                   dirichlet_alpha=0.3, dirichlet_epsilon=0.0, seed=0)
        planner.search(beliefs, temperature=0.0)
        expected = _exact_q(tiger, leaf, beliefs).numpy()
        for i, root in enumerate(planner.roots):
            for a, edge in enumerate(root.edges):
                if all(child.edges is None for child in edge.children):
                    assert edge.q == pytest.approx(float(expected[i, a]), abs=1e-9)

    def test_expanded_child_takes_exact_expectimax_value(self, tiger, value_functions):
        # A child expanded with V_h leaves must hold max_a Q = V_{h+1} at its belief exactly
        # (expectimax backup); the earlier mean backup violated this.
        leaf, next_horizon = value_functions[4], value_functions[5]
        planner = BeliefTreeSearch(ExactSearchModel(tiger, leaf, CPU), num_simulations=1, c_puct=1.25,
                                   dirichlet_alpha=0.3, dirichlet_epsilon=0.0, seed=0)
        planner.search(_beliefs(0.5, 0.97), temperature=0.0)
        expanded = [child for root in planner.roots for edge in root.edges for child in edge.children
                    if child.edges is not None]
        assert len(expanded) == 2  # one simulation per root
        for child in expanded:
            assert child.value() == pytest.approx(float(next_horizon.value(child.state.unsqueeze(0))), abs=1e-9)

    @pytest.mark.parametrize("p_left, optimal", [
        (0.5, TigerAction.LISTEN), (0.85, TigerAction.LISTEN), (0.15, TigerAction.LISTEN),
        (0.97, TigerAction.OPEN_RIGHT), (0.03, TigerAction.OPEN_LEFT)])
    def test_greedy_action_is_optimal(self, tiger, value_functions, p_left, optimal):
        # With V_9 at the leaves, the search must choose the optimal first action of the
        # 10-step problem (checked against the solver's own greedy action).
        beliefs = _beliefs(p_left)
        assert int(value_functions[9].greedy_action(beliefs)) == optimal
        planner = BeliefTreeSearch(ExactSearchModel(tiger, value_functions[8], CPU), num_simulations=100,
                                   c_puct=1.25, dirichlet_alpha=0.3, dirichlet_epsilon=0.0, seed=0)
        policy = planner.search(beliefs, temperature=0.0)
        assert int(policy.argmax()) == optimal

    def test_visit_counts_and_depth(self, tiger, value_functions):
        planner = BeliefTreeSearch(ExactSearchModel(tiger, value_functions[4], CPU), num_simulations=40,
                                   c_puct=1.25, dirichlet_alpha=0.3, dirichlet_epsilon=0.0, seed=0)
        policy = planner.search(_beliefs(0.5, 0.9), temperature=1.0)
        assert torch.allclose(policy.sum(-1), torch.ones(2))
        assert (planner.statistics.visit_counts.sum(axis=1) == 40).all()
        assert planner.statistics.max_depth >= 2

    def test_search_is_reproducible(self, tiger, value_functions):
        def run(seed):
            planner = BeliefTreeSearch(ExactSearchModel(tiger, value_functions[4], CPU), num_simulations=30,
                                       c_puct=1.25, dirichlet_alpha=0.3, dirichlet_epsilon=0.25, seed=seed)
            planner.search(_beliefs(0.5, 0.7), temperature=1.0)
            return planner.statistics.visit_counts
        assert np.array_equal(run(3), run(3))

    def test_rejects_zero_simulations(self, tiger):
        with pytest.raises(ValueError):
            BeliefTreeSearch(ExactSearchModel(tiger, None, CPU), 0, 1.25, 0.3, 0.0, 0)

    def test_min_max_stats(self):
        stats = MinMaxStats()
        assert stats.normalize(3.0) == 0.5
        stats.update(10.0)
        stats.update(-10.0)
        assert (stats.normalize(-10.0), stats.normalize(0.0), stats.normalize(10.0)) == (0.0, 0.5, 1.0)


class TestLearnedSearchModel:

    def test_expansion_is_consistent_with_filter_and_heads(self):
        torch.manual_seed(0)
        num_a, num_o, dim, hidden = 3, 2, 16, 32
        belief_filter = BeliefFilter(num_a, num_o, dim, hidden, 1)
        codec = TwoHotSymlog(255, 2000.0)
        model = LearnedSearchModel(belief_filter, RewardHead(dim, num_a, hidden, 1, 255),
                                   ObservationHead(dim, num_a, num_o, hidden, 1), ValueHead(dim, hidden, 1, 255),
                                   codec, num_a, num_o, discount=0.95)
        states = torch.randn(4, dim)
        expansion = model.expand(states)
        assert expansion.rewards.shape == (4, num_a)
        assert torch.allclose(expansion.observation_probs.sum(-1), torch.ones(4, num_a))
        for a in range(num_a):
            for o in range(num_o):
                expected = belief_filter.step(states, F.one_hot(torch.full((4,), a), num_a).float(),
                                              F.one_hot(torch.full((4,), o), num_o).float())
                assert torch.allclose(expansion.next_states[:, a, o], expected, atol=1e-6)
        stepped = model.update(states, torch.tensor([0, 1, 2, 0]), torch.tensor([1, 0, 1, 0]))
        assert stepped.shape == (4, dim)

    def test_finite_horizon_counts_down_to_a_terminal_zero(self):
        # search_model.py, section 5: state [z, k]; k counts down; reward and value are exactly 0 past the window.
        torch.manual_seed(0)
        num_a, num_o, dim, hidden, horizon = 3, 2, 16, 32, 3
        belief_filter = BeliefFilter(num_a, num_o, dim, hidden, 1)
        reward_head, value_head = RewardHead(dim, num_a, hidden, 1, 255), HorizonValueHead(dim, hidden, 1, 255, horizon)
        for head in (reward_head, value_head):                       # leave the zero initialisation
            torch.nn.init.normal_(head.net[-1].weight)
        codec = TwoHotSymlog(255, 2000.0)
        model = LearnedSearchModel(belief_filter, reward_head, ObservationHead(dim, num_a, num_o, hidden, 1), value_head,
                                   codec, num_a, num_o, discount=0.95, horizon=horizon)
        states = model.initial_states(4, None)
        assert states.shape == (4, dim + 1) and torch.equal(states[:, -1], torch.full((4,), 3.0))
        actions, observations = torch.tensor([0, 1, 2, 0]), torch.tensor([1, 0, 1, 0])
        stepped = model.update(states, actions, observations)
        assert torch.equal(stepped[:, -1], torch.full((4,), 2.0))
        assert torch.allclose(stepped[:, :-1], belief_filter.step(states[:, :-1], F.one_hot(actions, num_a).float(),
                                                                  F.one_hot(observations, num_o).float()))
        expansion = model.expand(stepped)                            # k = 2: children at k = 1 are valued by the head
        child = expansion.next_states[:, 1, 0]
        assert torch.equal(child[:, -1], torch.ones(4))
        assert torch.allclose(expansion.next_values[:, 1, 0], codec.mean(value_head(child[:, :-1], child[:, -1])))
        assert (expansion.rewards != 0).any()
        last = stepped.clone()
        last[:, -1] = 1.0                                            # last step: children are terminal
        assert torch.equal(model.expand(last).next_values, torch.zeros(4, num_a, num_o))
        ended = stepped.clone()
        ended[:, -1] = 0.0                                           # past the window: no reward either
        assert torch.equal(model.expand(ended).rewards, torch.zeros(4, num_a))
        with pytest.raises(RuntimeError):
            model.update(ended, actions, observations)
        with pytest.raises(ValueError):
            LearnedSearchModel(belief_filter, reward_head, ObservationHead(dim, num_a, num_o, hidden, 1),
                               ValueHead(dim, hidden, 1, 255), codec, num_a, num_o, discount=0.95, horizon=horizon)


class TestUniformRandomAgent:

    def test_actions_are_uniform_and_seeded(self):
        from ipomdp.agents import UniformRandomAgent
        first = UniformRandomAgent(3, 30000, seed=1, device=CPU).act()
        again = UniformRandomAgent(3, 30000, seed=1, device=CPU).act()
        assert torch.equal(first, again)
        counts = torch.bincount(first, minlength=3).double() / 30000
        assert torch.allclose(counts, torch.full((3,), 1 / 3, dtype=torch.float64), atol=0.02)


class TestPlanningAgent:

    def test_tracks_exact_beliefs_with_canonical_timing(self, tiger, value_functions):
        model = ExactSearchModel(tiger, value_functions[4], CPU)
        planner = BeliefTreeSearch(model, 10, 1.25, 0.3, 0.0, seed=0)
        agent = PlanningAgent(model, planner, batch_size=3, temperature=0.0, seed=0, device=CPU)
        with pytest.raises(RuntimeError):
            agent.act()                     # no belief before the first reset()
        agent.reset(None)
        assert torch.equal(agent.state, model.initial_states(3, None))
        with pytest.raises(ValueError):
            model.initial_states(3, torch.zeros(3, dtype=torch.int64))   # a FinitePOMDP has no o_0
        action = agent.act()
        assert action.tolist() == [TigerAction.LISTEN] * 3
        observation = torch.tensor([0, 1, 0])
        agent.update(action, observation)
        assert torch.allclose(agent.state, belief_update(tiger, model.initial_states(3, None), action, observation))
        agent.reset(None)
        assert torch.equal(agent.state, model.initial_states(3, None))

    def test_epsilon_uniform_exploration(self, tiger, value_functions):
        """epsilon = 1 acts uniformly; epsilon = 0 draws no extra random numbers (bit-identical runs)."""
        model = ExactSearchModel(tiger, value_functions[4], CPU)
        batch = 6000
        greedy = PlanningAgent(model, BeliefTreeSearch(model, 4, 1.25, 0.3, 0.0, seed=0), batch, 0.0, seed=0, device=CPU)
        greedy.reset(None)
        assert greedy.act().eq(TigerAction.LISTEN).all()
        before = greedy._generator.get_state()
        reference = torch.Generator().manual_seed(0)
        torch.multinomial(torch.ones(batch, 3), 1, generator=reference)
        assert torch.equal(before, reference.get_state())   # only the visit-count draw consumed the stream
        explorer = PlanningAgent(model, BeliefTreeSearch(model, 4, 1.25, 0.3, 0.0, seed=0), batch, 0.0, seed=0,
                                 device=CPU, epsilon=1.0)
        explorer.reset(None)
        counts = torch.bincount(explorer.act(), minlength=3).double() / batch
        assert torch.allclose(counts, torch.full((3,), 1 / 3, dtype=torch.float64), atol=5 * math.sqrt(2 / 9 / batch))
        with pytest.raises(ValueError):
            PlanningAgent(model, BeliefTreeSearch(model, 4, 1.25, 0.3, 0.0, seed=0), 1, 0.0, seed=0, device=CPU,
                          epsilon=1.5)


# A 5-action, 3-observation stand-in for Wumpus: step action 0, turns 1 and 2, fatal observation 2.
_RISK = RiskSpec(step_action=0, turn_actions=(1, 2), fatal_observation=2, fatal_reward=-1000.0, step_cost=-1.0)


class _FixedModel:
    """A SearchModel stub whose every expansion is the given one."""

    num_actions, num_observations, discount = 5, 3, 0.9

    def __init__(self, expansion):
        self.expansion = expansion

    def initial_states(self, batch_size, initial_observations):
        return torch.zeros(batch_size, 1)

    def update(self, states, actions, observations):
        return states

    def expand(self, states):
        return self.expansion


class TestRiskMarginModel:
    """planning/search_model.py, section 6 (reproduces pessimism_play.py's PessimisticModel)."""

    @staticmethod
    def _expansion():
        probs = torch.softmax(torch.randn(2, 5, 3, generator=torch.Generator().manual_seed(0)), dim=-1)
        probs[0, 0] = torch.tensor([0.6, 0.3, 0.1])
        probs[1, 0] = torch.tensor([0.0, 0.0, 1.0])                       # a certain death
        rewards = torch.tensor([[-1.0 - 1000.0 * 0.1, -1.0, -1.0, -1.0, -10.0], [-1001.0, -1.0, -1.0, 999.0, -10.0]])
        return Expansion(rewards, probs, torch.randn(2, 5, 3, 4), torch.randn(2, 5, 3))

    def test_inflates_only_the_step_actions_fatal_outcome(self):
        e = self._expansion()
        out = RiskMarginModel(_FixedModel(e), _RISK, 2.0).expand(torch.zeros(2, 1))
        p = e.observation_probs[0, 0, 2]
        assert out.observation_probs[0, 0, 2] == pytest.approx(float(2 * p))
        assert torch.allclose(out.observation_probs[0, 0, :2], e.observation_probs[0, 0, :2] * (1 - 2 * p) / (1 - p))
        assert torch.allclose(out.observation_probs.sum(-1), torch.ones(2, 5))
        assert torch.equal(out.observation_probs[1, 0], torch.tensor([0.0, 0.0, 1.0]))   # clamped at 1
        assert out.rewards[0, 0] == pytest.approx(-1.0 - 1000.0 * 0.2)
        assert out.rewards[1, 0] == pytest.approx(-1001.0)
        assert torch.equal(out.rewards[:, 1:], e.rewards[:, 1:])
        assert torch.equal(out.observation_probs[:, 1:], e.observation_probs[:, 1:])
        assert out.next_states is e.next_states and out.next_values is e.next_values

    @pytest.mark.parametrize("margin", [1.0, 0.5])
    def test_rejects_a_margin_that_is_not_above_one(self, margin):
        with pytest.raises(ValueError):
            RiskMarginModel(_FixedModel(self._expansion()), _RISK, margin)


class _FixedPlanner:
    """A planner stub: a fixed one-hot policy and fixed root Q values."""

    def __init__(self, actions, q):
        self.actions, self.statistics = actions, SearchStatistics(q_values=np.asarray(q, dtype=float))

    def search(self, states, temperature):
        return F.one_hot(self.actions, 5).float()

    def state_dict(self):
        return {}


class TestOwnTrials:
    """agents/planning_agent.py, section 4."""

    def test_turn_with_higher_q_then_the_step_action(self):
        q = [[0.0, 1.0, 2.0, 0.0, 0.0], [0.0, 3.0, 3.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
        planner = _FixedPlanner(torch.tensor([3, 4, 0]), q)
        agent = PlanningAgent(_FixedModel(None), planner, 3, temperature=0.0, seed=0, device=CPU,
                              trial=_RISK, trial_prob=1.0)
        agent.reset(None)
        assert agent.act().tolist() == [2, 1, 0]           # better turn (ties: the first); a step row is not trialled
        assert agent.act().tolist() == [0, 0, 0]           # the pending steps override the planner's GRAB / SHOOT
        assert agent.act().tolist() == [2, 1, 0]           # a new trial after the step
        assert agent.reset_trial_count() == 4 and agent.trials == 0
        agent.act()
        agent.reset(None)                                  # reset clears the pending steps
        assert agent.act().tolist() == [2, 1, 0]

    def test_epsilon_rows_are_not_trialled(self):
        planner = _FixedPlanner(torch.full((64,), 3), np.zeros((64, 5)))
        agent = PlanningAgent(_FixedModel(None), planner, 64, temperature=0.0, seed=0, device=CPU, epsilon=1.0,
                              trial=_RISK, trial_prob=1.0)
        agent.reset(None)
        agent.act()
        assert agent.trials == 0

    def test_off_draws_nothing(self):
        planner = _FixedPlanner(torch.tensor([3, 4]), np.zeros((2, 5)))
        off = PlanningAgent(_FixedModel(None), planner, 2, temperature=0.0, seed=0, device=CPU, trial=_RISK)
        plain = PlanningAgent(_FixedModel(None), planner, 2, temperature=0.0, seed=0, device=CPU)
        for agent in (off, plain):
            agent.reset(None)
            agent.act()
        assert torch.equal(off.state_dict()["generator"], plain.state_dict()["generator"])

    def test_trials_need_a_risk_spec(self):
        with pytest.raises(ValueError):
            PlanningAgent(_FixedModel(None), _FixedPlanner(torch.tensor([3]), np.zeros((1, 5))), 1, temperature=0.0,
                          seed=0, device=CPU, trial_prob=0.2)


def _discounted_returns(tiger, agent, steps, seed):
    env = BatchedPOMDPEnv(tiger, agent.batch_size, steps, seed, CPU)
    agent.reset(None)
    returns = torch.zeros(agent.batch_size, dtype=torch.float64)
    for t in range(steps):
        action = agent.act()
        out = env.step(action)
        agent.update(action, out.observation)
        returns += tiger.discount ** t * out.reward.double()
    return returns


@pytest.mark.slow
def test_exact_model_planner_matches_optimal_policy_return(tiger):
    """
    End-to-end on the exact model. (1) With V* leaves, root Q equals Q* at any search depth.
    (2) The planning agent (exact beliefs, V* leaves, 50 simulations, greedy) earns the same
    discounted return as the exact optimal policy on identical simulator seeds, and that return
    matches V*(b0) = 19.37 (100 steps; 0.95^100 = 0.006).
    """
    solution = solve_infinite_horizon(tiger, tolerance=1e-3, prune_epsilon=1e-6)
    batch, steps = 256, 100

    class OptimalAgent:
        batch_size = batch

        def reset(self, initial_observation):
            self.state = tiger.initial_belief.expand(batch, -1).clone()

        def act(self):
            return solution.value_function.greedy_action(self.state)

        def update(self, action, observation):
            self.state = belief_update(tiger, self.state, action, observation)

    # V* is a fixed point of the expectimax backup, so with V* leaves the root Q must equal the
    # exact Q*(b, a) no matter how deep the search goes (up to the solver's certified error).
    model = ExactSearchModel(tiger, solution.value_function, CPU)
    beliefs = _beliefs(0.5, 0.85, 0.97)
    exact_q = _exact_q(tiger, solution.value_function, beliefs).numpy()
    for sims in (1, 200):
        planner = BeliefTreeSearch(model, sims, 1.25, 0.3, 0.0, seed=0)
        planner.search(beliefs, temperature=0.0)
        assert np.allclose(planner.statistics.q_values, exact_q, atol=2 * solution.error_bound + 1e-6)

    planner_agent = PlanningAgent(model, BeliefTreeSearch(model, 50, 1.25, 0.3, 0.0, seed=0), batch,
                                  temperature=0.0, seed=0, device=CPU)
    optimal = _discounted_returns(tiger, OptimalAgent(), steps, seed=11)
    planned = _discounted_returns(tiger, planner_agent, steps, seed=11)
    stderr = float(optimal.std()) / math.sqrt(batch)
    assert float(optimal.mean()) == pytest.approx(19.37, abs=5 * stderr + 0.1)
    assert abs(float(planned.mean()) - float(optimal.mean())) < 3 * stderr
