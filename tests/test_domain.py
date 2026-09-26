# ABSOLUTE PATH: tests/test_domain.py
"""
Phase-1 small-scale checks for the exact domain layer (no learned components involved).

What is verified, and why it matters downstream:
    1. The Tiger tensors are entry-for-entry the reference `tiger.95.POMDP`.
    2. The simulator's empirical frequencies match T, O, R and b0 (statistical z-tests).
    3. The batched Bayes filter matches brute-force enumeration over hidden state paths.
    4. The alpha-vector solver matches an independent belief-tree expectimax and the
       published infinite-horizon value V*(b0) = 19.37.
    5. Simulator + filter + solver agree end-to-end: Monte Carlo returns of the optimal
       policy match the solver's value within sampling error.
These are the ground-truth instruments later phases use to grade the learned agent, so any
error here would silently invalidate every later measurement.
"""

import itertools
import math

import pytest
import torch

from ipomdp.domain import (
    EXACT_PRUNE_EPSILON,
    AlphaVectorSet,
    BatchedPOMDPEnv,
    FinitePOMDP,
    TigerAction,
    TigerObservation,
    TigerState,
    belief_update,
    build_tiger_pomdp,
    initial_beliefs,
    observation_distribution,
    prune,
    solve_finite_horizon,
    solve_infinite_horizon,
    sup_norm_distance,
)

CPU = torch.device("cpu")
# Statistical tests fail only if an estimate is > Z_MAX standard errors from the truth.
# With fixed seeds they are deterministic; 5 SE would be a ~6e-7 false-alarm rate otherwise.
Z_MAX = 5.0

# V_h(b0) for the canonical Tiger, gamma = 0.95, computed by exhaustive recursion over the
# reachable beliefs (independent of the alpha-vector code). V_inf is the published value.
REFERENCE_VALUES = {1: -1.0, 2: -1.95, 3: 2.3098, 4: 1.7955442187, 5: 2.7630961931, 10: 6.6933684318}
REFERENCE_V_INFINITE = 19.3713


@pytest.fixture(scope="module")
def tiger() -> FinitePOMDP:
    return build_tiger_pomdp()


@pytest.fixture(scope="module")
def tiger_value_functions(tiger) -> list[AlphaVectorSet]:
    return solve_finite_horizon(tiger, 10)


# ----------------------------------------------------------------------------------------------
# 1. Specification
# ----------------------------------------------------------------------------------------------

class TestTigerSpecification:
    """The Tiger tensors are the reference `tiger.95.POMDP`, entry for entry."""

    def test_matches_reference_file(self, tiger):
        uniform = torch.full((2, 2), 0.5, dtype=torch.float64)
        assert tiger.discount == 0.95
        assert torch.equal(tiger.initial_belief, torch.tensor([0.5, 0.5], dtype=torch.float64))
        assert torch.equal(tiger.transition[TigerAction.LISTEN], torch.eye(2, dtype=torch.float64))
        assert torch.equal(tiger.transition[TigerAction.OPEN_LEFT], uniform)
        assert torch.equal(tiger.transition[TigerAction.OPEN_RIGHT], uniform)
        assert torch.allclose(tiger.observation[TigerAction.LISTEN],
                              torch.tensor([[0.85, 0.15], [0.15, 0.85]], dtype=torch.float64), atol=0.0, rtol=0.0)
        assert torch.equal(tiger.observation[TigerAction.OPEN_LEFT], uniform)
        assert torch.equal(tiger.observation[TigerAction.OPEN_RIGHT], uniform)
        assert torch.equal(tiger.reward, torch.tensor([[-1.0, -1.0], [-100.0, 10.0], [10.0, -100.0]],
                                                      dtype=torch.float64))

    def test_name_order_matches_enums(self, tiger):
        assert tiger.state_names == ("TIGER_LEFT", "TIGER_RIGHT")
        assert tiger.action_names == ("LISTEN", "OPEN_LEFT", "OPEN_RIGHT")
        assert tiger.observation_names == ("GROWL_LEFT", "GROWL_RIGHT")


class TestFinitePOMDPValidation:
    """Malformed specifications are rejected eagerly; nothing is silently renormalised."""

    @staticmethod
    def _fields(tiger: FinitePOMDP) -> dict:
        return dict(transition=tiger.transition, observation=tiger.observation, reward=tiger.reward,
                    initial_belief=tiger.initial_belief, discount=tiger.discount,
                    state_names=tiger.state_names, action_names=tiger.action_names,
                    observation_names=tiger.observation_names)

    @pytest.mark.parametrize("field, value", [
        ("transition", torch.full((3, 2, 2), 0.4, dtype=torch.float64)),          # rows sum to 0.8
        ("observation", torch.full((3, 2, 3), 1 / 3, dtype=torch.float64)),       # wrong |O|
        ("reward", torch.zeros(3, 2, dtype=torch.float32)),                       # wrong dtype
        ("initial_belief", torch.tensor([1.5, -0.5], dtype=torch.float64)),        # negative
        ("discount", 1.0),                                                        # not < 1
        ("action_names", ("LISTEN", "LISTEN", "OPEN_RIGHT")),                     # duplicate
    ])
    def test_rejects_invalid_field(self, tiger, field, value):
        fields = self._fields(tiger)
        fields[field] = value
        with pytest.raises(ValueError):
            FinitePOMDP(**fields)


# ----------------------------------------------------------------------------------------------
# 2. Simulator
# ----------------------------------------------------------------------------------------------

def _assert_frequency(successes: int, trials: int, probability: float) -> None:
    """Binomial z-test of an empirical frequency against its exact probability."""
    if probability in (0.0, 1.0):
        assert successes == trials * probability
        return
    z = (successes / trials - probability) / math.sqrt(probability * (1 - probability) / trials)
    assert abs(z) < Z_MAX, f"frequency {successes / trials:.5f} vs {probability:.5f} (z={z:.2f})"


class TestBatchedPOMDPEnv:
    """The simulator samples exactly from T, O, R and b0, and is reproducible."""

    BATCH = 200_000

    def test_initial_state_distribution(self, tiger):
        env = BatchedPOMDPEnv(tiger, self.BATCH, max_steps=1, seed=0, device=CPU)
        _assert_frequency(int((env.state == TigerState.TIGER_LEFT).sum()), self.BATCH, 0.5)

    @pytest.mark.parametrize("action", list(TigerAction))
    @pytest.mark.parametrize("start_state", list(TigerState))
    def test_transition_observation_reward(self, tiger, action, start_state):
        env = BatchedPOMDPEnv(tiger, self.BATCH, max_steps=5, seed=int(action) * 2 + int(start_state), device=CPU)
        env._state.fill_(int(start_state))  # condition on s_t
        out = env.step(torch.full((self.BATCH,), int(action), dtype=torch.int64))
        next_state = env.state

        assert torch.all(out.reward == float(tiger.reward[action, start_state]))
        _assert_frequency(int((next_state == 0).sum()), self.BATCH, float(tiger.transition[action, start_state, 0]))
        for s_next in TigerState:
            rows = next_state == s_next
            if not rows.any():  # e.g. LISTEN from TIGER_LEFT never reaches TIGER_RIGHT
                continue
            _assert_frequency(int((out.observation[rows] == 0).sum()), int(rows.sum()),
                              float(tiger.observation[action, s_next, 0]))

    def test_reset_emits_no_observation_and_truncation_protocol(self, tiger):
        env = BatchedPOMDPEnv(tiger, 4, max_steps=3, seed=0, device=CPU)
        assert env.reset() is None
        listen = torch.zeros(4, dtype=torch.int64)
        flags = [env.step(listen).truncated for _ in range(3)]
        assert [bool(f.all()) for f in flags] == [False, False, True]
        with pytest.raises(RuntimeError):
            env.step(listen)
        mask = torch.tensor([True, False, True, False])
        env.reset_rows(mask)
        assert env.elapsed_steps.tolist() == [0, 3, 0, 3]

    def test_rejects_invalid_actions(self, tiger):
        env = BatchedPOMDPEnv(tiger, 2, max_steps=3, seed=0, device=CPU)
        with pytest.raises(ValueError):
            env.step(torch.tensor([0, 3]))
        with pytest.raises(ValueError):
            env.step(torch.tensor([0.0, 1.0]))

    def test_seed_reproducibility(self, tiger):
        def rollout(seed: int) -> list[torch.Tensor]:
            env = BatchedPOMDPEnv(tiger, 64, max_steps=20, seed=seed, device=CPU)
            actions = torch.Generator().manual_seed(123)
            trace = [env.state]
            for _ in range(20):
                out = env.step(torch.randint(0, 3, (64,), generator=actions))
                trace += [out.observation, env.state]
            return trace

        first, second, other = rollout(7), rollout(7), rollout(8)
        assert all(torch.equal(a, b) for a, b in zip(first, second))
        assert not all(torch.equal(a, b) for a, b in zip(first, other))


# ----------------------------------------------------------------------------------------------
# 3. Exact Bayes filter
# ----------------------------------------------------------------------------------------------

def _enumerated_posterior(model: FinitePOMDP, actions: list[int], observations: list[int]):
    """
    P(s_T | a_{0:T-1}, o_{1:T}) and P(o_T | a, o_{1:T-1}) by summing over all hidden state
    paths s_0..s_T. Deliberately shares no code with belief.py.
    """
    num_s = model.num_states
    horizon = len(actions)
    joint = torch.zeros(num_s, dtype=torch.float64)           # P(s_T, o_{1:T})
    joint_prefix = torch.zeros(num_s, dtype=torch.float64)    # P(s_T, o_{1:T-1})
    for path in itertools.product(range(num_s), repeat=horizon + 1):
        p = float(model.initial_belief[path[0]])
        for t in range(horizon):
            p *= float(model.transition[actions[t], path[t], path[t + 1]])
            if t < horizon - 1:
                p *= float(model.observation[actions[t], path[t + 1], observations[t]])
        joint_prefix[path[-1]] += p
        joint[path[-1]] += p * float(model.observation[actions[-1], path[-1], observations[-1]])
    evidence_prefix = joint_prefix.sum()
    predictive = torch.stack([
        (joint_prefix * model.observation[actions[-1], :, o]).sum() / evidence_prefix
        for o in range(model.num_observations)
    ])
    return joint / joint.sum(), predictive


class TestBeliefFilter:
    """The batched filter equals brute-force enumeration over state paths."""

    def test_matches_enumeration_on_all_short_histories(self, tiger):
        # Every (action, observation) history of length 3: 6^3 = 216 histories, one batch.
        histories = list(itertools.product(itertools.product(range(3), range(2)), repeat=3))
        actions = torch.tensor([[a for a, _ in h] for h in histories])
        observations = torch.tensor([[o for _, o in h] for h in histories])

        belief = initial_beliefs(tiger, len(histories), CPU)
        for t in range(3):
            predictive = observation_distribution(tiger, belief, actions[:, t])
            belief = belief_update(tiger, belief, actions[:, t], observations[:, t])

        for i, h in enumerate(histories):
            exact_posterior, exact_predictive = _enumerated_posterior(
                tiger, [a for a, _ in h], [o for _, o in h])
            assert torch.allclose(belief[i], exact_posterior, atol=1e-12, rtol=0)
            assert torch.allclose(predictive[i], exact_predictive, atol=1e-12, rtol=0)

    def test_door_opening_resets_belief_to_prior(self, tiger):
        belief = torch.tensor([[0.99, 0.01], [0.2, 0.8]], dtype=torch.float64)
        for action in (TigerAction.OPEN_LEFT, TigerAction.OPEN_RIGHT):
            for obs in TigerObservation:
                a = torch.full((2,), int(action))
                o = torch.full((2,), int(obs))
                assert torch.equal(belief_update(tiger, belief, a, o), initial_beliefs(tiger, 2, CPU))
                assert torch.equal(observation_distribution(tiger, belief, a),
                                   torch.full((2, 2), 0.5, dtype=torch.float64))

    def test_listen_update_closed_form(self, tiger):
        # b'(TL) = 0.85 b / (0.85 b + 0.15 (1 - b)) after GROWL_LEFT.
        b = torch.tensor([0.5, 0.3, 0.9], dtype=torch.float64)
        belief = torch.stack([b, 1 - b], dim=-1)
        updated = belief_update(tiger, belief, torch.zeros(3, dtype=torch.int64), torch.zeros(3, dtype=torch.int64))
        expected = 0.85 * b / (0.85 * b + 0.15 * (1 - b))
        assert torch.allclose(updated[:, 0], expected, atol=1e-15, rtol=0)

    def test_impossible_observation_raises(self, tiger):
        deterministic = FinitePOMDP(
            transition=tiger.transition, observation=torch.stack([torch.eye(2, dtype=torch.float64)] * 3),
            reward=tiger.reward, initial_belief=torch.tensor([1.0, 0.0], dtype=torch.float64),
            discount=0.95, state_names=tiger.state_names, action_names=tiger.action_names,
            observation_names=tiger.observation_names)
        belief = initial_beliefs(deterministic, 1, CPU)
        with pytest.raises(ValueError):
            belief_update(deterministic, belief, torch.tensor([0]), torch.tensor([1]))


# ----------------------------------------------------------------------------------------------
# 4. Exact solver
# ----------------------------------------------------------------------------------------------

def _expectimax(model: FinitePOMDP, belief: torch.Tensor, horizon: int) -> float:
    """Exhaustive finite-horizon expectimax over the belief tree (independent of alpha vectors)."""
    if horizon == 0:
        return 0.0
    best = -math.inf
    for a in range(model.num_actions):
        action = torch.tensor([a])
        value = float(belief[0] @ model.reward[a])
        predictive = observation_distribution(model, belief, action)[0]
        for o in range(model.num_observations):
            if predictive[o] > 0:
                posterior = belief_update(model, belief, action, torch.tensor([o]))
                value += model.discount * float(predictive[o]) * _expectimax(model, posterior, horizon - 1)
        best = max(best, value)
    return best


class TestExactSolver:
    """Alpha-vector value iteration reproduces independent references."""

    @pytest.mark.parametrize("horizon", sorted(REFERENCE_VALUES))
    def test_matches_reference_values_at_b0(self, tiger, tiger_value_functions, horizon):
        b0 = tiger.initial_belief.unsqueeze(0)
        assert float(tiger_value_functions[horizon - 1].value(b0)) == pytest.approx(REFERENCE_VALUES[horizon], abs=1e-9)

    @pytest.mark.parametrize("horizon", [1, 2, 3, 4, 5])
    def test_matches_expectimax_across_beliefs(self, tiger, tiger_value_functions, horizon):
        for p in torch.linspace(0, 1, 11, dtype=torch.float64):
            belief = torch.stack([p, 1 - p]).unsqueeze(0)
            expected = _expectimax(tiger, belief, horizon)
            assert float(tiger_value_functions[horizon - 1].value(belief)) == pytest.approx(expected, abs=1e-9)

    def test_optimal_actions(self, tiger, tiger_value_functions):
        gamma_10 = tiger_value_functions[9]
        beliefs = torch.tensor([[0.5, 0.5], [0.99, 0.01], [0.01, 0.99]], dtype=torch.float64)
        assert gamma_10.greedy_action(beliefs).tolist() == [
            TigerAction.LISTEN, TigerAction.OPEN_RIGHT, TigerAction.OPEN_LEFT]

    @pytest.mark.parametrize("epsilon", [EXACT_PRUNE_EPSILON, 1e-2, 1.0])
    def test_prune_error_guarantee(self, epsilon):
        generator = torch.Generator().manual_seed(0)
        vectors = torch.randn(300, 3, generator=generator, dtype=torch.float64)
        kept, kept_actions = prune(vectors, torch.arange(300), epsilon)
        full = AlphaVectorSet(vectors, torch.arange(300))
        pruned = AlphaVectorSet(kept, kept_actions)
        # Kept vectors are a subset of the input, and the envelope loses at most epsilon.
        assert all(any(torch.equal(k, v) for v in vectors) for k in kept)
        gap = sup_norm_distance(full, pruned)
        assert -1e-12 <= gap <= epsilon + 1e-12

    def test_sup_norm_distance_is_exact(self, tiger, tiger_value_functions):
        # V_1 vs V_2 on a fine grid lower-bounds the LP result, which must be attained.
        grid = torch.linspace(0, 1, 20001, dtype=torch.float64)
        beliefs = torch.stack([grid, 1 - grid], dim=-1)
        gap = (tiger_value_functions[1].value(beliefs) - tiger_value_functions[0].value(beliefs)).abs().max()
        assert sup_norm_distance(tiger_value_functions[1], tiger_value_functions[0]) == pytest.approx(float(gap), abs=1e-6)

    def test_unreachable_tolerance_is_rejected(self, tiger):
        with pytest.raises(ValueError):
            solve_infinite_horizon(tiger, tolerance=1e-6, prune_epsilon=1e-6)


# ----------------------------------------------------------------------------------------------
# 5. End-to-end agreement: simulator + filter + solver
# ----------------------------------------------------------------------------------------------

def _monte_carlo_return(model: FinitePOMDP, policy_for_step, horizon: int, batch: int, seed: int):
    """Mean and standard error of the discounted return of a belief-based policy."""
    env = BatchedPOMDPEnv(model, batch, max_steps=horizon, seed=seed, device=CPU)
    belief = initial_beliefs(model, batch, CPU)
    returns = torch.zeros(batch, dtype=torch.float64)
    for t in range(horizon):
        action = policy_for_step(t, belief)
        out = env.step(action)
        returns += model.discount ** t * out.reward.to(torch.float64)
        belief = belief_update(model, belief, action, out.observation)
    return float(returns.mean()), float(returns.std() / math.sqrt(batch))


class TestEndToEnd:

    def test_optimal_finite_horizon_policy_achieves_solver_value(self, tiger, tiger_value_functions):
        horizon = 5
        mean, stderr = _monte_carlo_return(
            tiger, lambda t, b: tiger_value_functions[horizon - t - 1].greedy_action(b),
            horizon, batch=200_000, seed=0)
        assert abs(mean - REFERENCE_VALUES[horizon]) < Z_MAX * stderr

    @pytest.mark.slow
    def test_infinite_horizon_value_and_policy(self, tiger):
        solution = solve_infinite_horizon(tiger, tolerance=1e-4, prune_epsilon=1e-6)
        b0 = tiger.initial_belief.unsqueeze(0)
        assert solution.error_bound <= 1e-4
        assert float(solution.value_function.value(b0)) == pytest.approx(REFERENCE_V_INFINITE, abs=1e-4 + 5e-5)

        # Truncating at 400 steps changes the discounted return by < 0.95^400 * 110 / 0.05 < 1e-5.
        mean, stderr = _monte_carlo_return(
            tiger, lambda t, b: solution.value_function.greedy_action(b), 400, batch=50_000, seed=1)
        assert abs(mean - REFERENCE_V_INFINITE) < Z_MAX * stderr + 1e-4
