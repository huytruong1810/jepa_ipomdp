# ABSOLUTE PATH: tests/test_interpretability.py
"""
Phase-6 checks for the interpretability layer: the span-Hoelder lemma behind the error bounds,
the per-action Q* vector sets, soundness of the bounds, probes on a perfectly informative latent,
and an end-to-end smoke run of the belief analysis.
"""

import pytest
import torch

from ipomdp.agents import PlanningAgent, UniformRandomAgent
from ipomdp.domain import (AlphaVectorSet, BatchedPOMDPEnv, TigerAction, action_value_functions, backup,
                           belief_update, build_tiger_pomdp, observation_distribution, solve_finite_horizon)
from ipomdp.domain.solver import EXACT_PRUNE_EPSILON
from ipomdp.interpretability import (ExactBeliefAgent, analyze_beliefs, build_probe_dataset, error_bounds,
                                     evaluate_probe, fit_linear_probe, lipschitz_constant, q_values)
from ipomdp.interpretability.belief_probe import ProbeDataset
from ipomdp.models import BeliefFilter, ObservationHead, RewardHead, TwoHotSymlog, ValueHead
from ipomdp.planning import BeliefTreeSearch, LearnedSearchModel
from ipomdp.training import play_episodes

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def tiger():
    return build_tiger_pomdp()


@pytest.fixture(scope="module")
def value_functions(tiger):
    return solve_finite_horizon(tiger, 10)


def _random_beliefs(n: int, states: int) -> torch.Tensor:
    """Uniform beliefs on the simplex (Dirichlet(1)); callers seed torch."""
    return torch.distributions.Dirichlet(torch.ones(states, dtype=torch.float64)).sample((n,))


class TestLipschitzLemma:

    @pytest.mark.parametrize("states", [2, 3, 5])
    def test_span_hoelder_bound_holds_and_is_tight(self, states):
        torch.manual_seed(states)
        vectors = torch.randn(7, states, dtype=torch.float64) * 10
        value = AlphaVectorSet(vectors, torch.zeros(7, dtype=torch.int64))
        lipschitz = lipschitz_constant(value)
        b, c = _random_beliefs(4000, states), _random_beliefs(4000, states)
        gap = (value.value(b) - value.value(c)).abs()
        assert (gap <= lipschitz * (b - c).abs().sum(-1) + 1e-9).all()
        # Tight: moving mass between the argmax and argmin coordinates of the widest vector.
        k = int((vectors.max(1).values - vectors.min(1).values).argmax())
        single = AlphaVectorSet(vectors[k:k + 1], torch.zeros(1, dtype=torch.int64))
        hi, lo = int(vectors[k].argmax()), int(vectors[k].argmin())
        e_hi, e_lo = torch.zeros(1, states, dtype=torch.float64), torch.zeros(1, states, dtype=torch.float64)
        e_hi[0, hi], e_lo[0, lo] = 1.0, 1.0
        assert float((single.value(e_hi) - single.value(e_lo)).abs()) == pytest.approx(2 * lipschitz_constant(single))


class TestActionValueFunctions:

    def test_matches_exact_lookahead_and_backup(self, tiger, value_functions):
        leaf = value_functions[4]
        per_action = action_value_functions(tiger, leaf, EXACT_PRUNE_EPSILON)
        beliefs = torch.stack([torch.linspace(0, 1, 21, dtype=torch.float64),
                               1 - torch.linspace(0, 1, 21, dtype=torch.float64)], -1)
        q = q_values(per_action, beliefs)
        for a in range(tiger.num_actions):
            action = torch.full((21,), a)
            expected = beliefs @ tiger.reward[a]
            probs = observation_distribution(tiger, beliefs, action)
            for o in range(tiger.num_observations):
                posterior = belief_update(tiger, beliefs, action, torch.full((21,), o))
                expected = expected + tiger.discount * probs[:, o] * leaf.value(posterior)
            assert torch.allclose(q[:, a], expected, atol=1e-8)
        assert torch.allclose(q.max(-1).values, backup(tiger, leaf, EXACT_PRUNE_EPSILON).value(beliefs), atol=1e-8)


class TestErrorBounds:

    def test_zero_error_gives_zero_measurements(self, tiger, value_functions):
        per_action = action_value_functions(tiger, value_functions[8], EXACT_PRUNE_EPSILON)
        beliefs = _random_beliefs(500, 2)
        report = error_bounds(value_functions[9], per_action, tiger.discount, beliefs, beliefs)
        assert report.belief_l1_max == 0.0 and report.value_error_max == 0.0 and report.regret_max == 0.0
        assert report.suboptimal_decision_rate == 0.0

    @pytest.mark.parametrize("noise", [0.01, 0.05, 0.2])
    def test_measurements_never_exceed_bounds(self, tiger, value_functions, noise):
        torch.manual_seed(0)
        per_action = action_value_functions(tiger, value_functions[8], EXACT_PRUNE_EPSILON)
        true = _random_beliefs(2000, 2)
        shifted = (true[:, 0] + noise * torch.randn(2000, dtype=torch.float64)).clamp(0, 1)
        decoded = torch.stack([shifted, 1 - shifted], -1)
        report = error_bounds(value_functions[9], per_action, tiger.discount, true, decoded)
        assert report.value_error_max <= report.value_error_bound_max + 1e-9
        assert report.regret_max <= report.regret_bound_max + 1e-9
        assert report.policy_loss_bound == pytest.approx(report.regret_bound_max / (1 - tiger.discount))


class TestProbes:

    def test_dataset_replays_exact_posteriors(self, tiger):
        belief_filter = BeliefFilter(3, 2, 8, 16, 1)
        episodes, _ = play_episodes(BatchedPOMDPEnv(tiger, 6, 5, 0, CPU), UniformRandomAgent(3, 6, 0, CPU))
        dataset = build_probe_dataset(tiger, belief_filter, episodes)
        assert dataset.latents.shape == (6 * 6, 8) and dataset.posteriors.shape == (6 * 6, 2)
        posteriors = dataset.posteriors.view(6, 6, 2)
        assert torch.allclose(posteriors[:, 0], tiger.initial_belief.expand(6, -1))
        expected = belief_update(tiger, posteriors[:, 0], episodes.actions[:, 0], episodes.observations[:, 0])
        assert torch.allclose(posteriors[:, 1], expected)

    def test_linear_probe_recovers_a_perfectly_informative_latent(self):
        # latent = [log-odds, noise]: a linear-softmax probe can represent b* exactly.
        torch.manual_seed(0)
        def dataset(n):
            p = torch.rand(n, dtype=torch.float64) * 0.98 + 0.01
            log_odds = (p / (1 - p)).log()
            latents = torch.stack([log_odds, torch.randn(n, dtype=torch.float64)], -1).float()
            return ProbeDataset(latents, torch.stack([p, 1 - p], -1))
        report = evaluate_probe(fit_linear_probe(dataset(4000)), dataset(4000))
        assert report.mean_kl < 1e-4 and report.max_l1 < 0.02


class TestBeliefAnalysis:

    def test_exact_belief_agent_listens_at_the_prior(self, tiger, value_functions):
        per_action = action_value_functions(tiger, value_functions[8], EXACT_PRUNE_EPSILON)
        agent = ExactBeliefAgent(tiger, per_action, batch_size=3, device=CPU)
        assert agent.act().tolist() == [TigerAction.LISTEN] * 3

    def test_analysis_runs_end_to_end(self, tiger, value_functions):
        torch.manual_seed(0)
        num_a, num_o, dim, hidden, episodes = 3, 2, 8, 16, 8
        belief_filter = BeliefFilter(num_a, num_o, dim, hidden, 1)
        model = LearnedSearchModel(belief_filter, RewardHead(dim, num_a, hidden, 1, 255),
                                   ObservationHead(dim, num_a, num_o, hidden, 1), ValueHead(dim, hidden, 1, 255),
                                   TwoHotSymlog(255, tiger.value_bound), num_a, num_o, tiger.discount)
        planner = PlanningAgent(model, BeliefTreeSearch(model, 3, 1.25, 0.3, 0.0, seed=0), episodes, 0.0, 0, CPU)
        per_action = action_value_functions(tiger, value_functions[8], EXACT_PRUNE_EPSILON)
        analysis = analyze_beliefs(tiger, belief_filter, planner, value_functions[9], per_action, 0.0,
                                   episode_length=6, num_episodes=episodes, mlp_probe_steps=50, seed=0,
                                   device=CPU)
        assert set(analysis.probes) == {"linear/random", "linear/agent", "mlp/random", "mlp/agent"}
        assert set(analysis.returns) == {"optimal", "decoded_linear", "decoded_mlp", "learned_planner"}
        assert len(analysis.geometry.explained_variance_ratio) == 5
        for report in analysis.bounds.values():
            assert report.regret_max <= report.regret_bound_max + 1e-9


class TestGeometry:

    def test_minimality_ratio_is_zero_iff_latent_is_a_function_of_the_posterior(self):
        from ipomdp.interpretability import latent_geometry
        torch.manual_seed(0)
        p = torch.randint(1, 10, (3000,), dtype=torch.float64) / 10
        posteriors = torch.stack([p, 1 - p], -1)
        log_odds = (p / (1 - p)).log()
        # z = (log-odds, 0.01 |log-odds|): a function of b* alone whose dominant direction is the side.
        minimal = ProbeDataset(torch.stack([log_odds, 0.01 * log_odds.abs()], -1).float(), posteriors)
        report = latent_geometry(minimal)
        assert report.minimality_ratio < 1e-6
        assert report.log_odds_spearman[0] == pytest.approx(1.0, abs=1e-6)
        noisy = ProbeDataset(torch.stack([log_odds, torch.randn(3000, dtype=torch.float64) * 3], -1).float(), posteriors)
        assert latent_geometry(noisy).minimality_ratio > 0.5

    def test_expected_bounds_dominate_expected_measurements(self, tiger, value_functions):
        torch.manual_seed(1)
        per_action = action_value_functions(tiger, value_functions[8], EXACT_PRUNE_EPSILON)
        true = _random_beliefs(2000, 2)
        shifted = (true[:, 0] + 0.05 * torch.randn(2000, dtype=torch.float64)).clamp(0, 1)
        report = error_bounds(value_functions[9], per_action, tiger.discount, true, torch.stack([shifted, 1 - shifted], -1))
        assert report.value_error_mean <= report.value_error_bound_mean + 1e-9
        assert report.regret_mean <= report.regret_bound_mean + 1e-9
