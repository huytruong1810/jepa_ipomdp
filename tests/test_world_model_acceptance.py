# ABSOLUTE PATH: tests/test_world_model_acceptance.py
"""
Phase-2/3/5 acceptance tests: the world model trained by the real WorldModelTrainer on canonical
Tiger, from uniformly-random-policy data only (off-policy), must reproduce exact quantities
computed from the FinitePOMDP and must plan optimally.

Protocol (fixed seeds, GPU, ~20 min):
    - 3000 updates, each on 64 fresh random-policy episodes of 100 steps (no MCTS in training).
      100-step episodes: conf/env/tiger.yaml, section 3. 3000 updates: the fitted value
      iteration of the value head needs them (|V - V*| 26 after 1000 updates, 4.9 after 3000).
    - Held-out data: 1024 random-policy episodes of 100 steps; latents from the online filter run
      from its learned z_0; exact posteriors b* from the batched Bayes filter.

Checked against exact references (measured at 3000 updates in brackets):
    Belief       probes of z_t vs b*(h_t), KL(b* || probe)            [linear 0.0007, MLP 0.00004]
    Reward       decoded R(z_t, a) vs b* . R[a]                        [doors 1.7-2.2, listen < 0.05]
    Observation  P(o' | z_t, a) vs exact P(o' | b*, a), KL             [0.0005-0.01]
    Value        decoded V(z_t) vs the exact optimum V*(b*)            [mean |V - V*| 4.9]
                 (Bellman optimality targets aim at V*, not at the random behaviour policy.)
    Planning     greedy agent over the LEARNED model (50 simulations), [21.19 +- 1.69 vs 19.28]
                 256 episodes, discounted return vs V*(b0)
"""

import pytest
import torch
import torch.nn.functional as F

from ipomdp.agents import PlanningAgent
from ipomdp.domain import BatchedPOMDPEnv, build_tiger_pomdp, observation_distribution, solve_infinite_horizon
from ipomdp.planning import BeliefTreeSearch, LearnedSearchModel
from ipomdp.interpretability import collect_probe_dataset, linear_probe, mlp_probe, uniform_random_policy
from ipomdp.models import BeliefFilter, LatentPredictor, ObservationHead, RecurrentJEPA, RewardHead, TwoHotSymlog, ValueHead
from ipomdp.training import EpisodeBatch, TrainerConfig, WorldModelTrainer, discounted_returns, play_episodes

UPDATES, BATCH, LENGTH, DIM, HIDDEN = 3000, 64, 100, 32, 64


def train_and_measure() -> dict:
    """Trains the world model with the acceptance protocol and returns all measured errors."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    pomdp = build_tiger_pomdp()
    num_a, num_o = pomdp.num_actions, pomdp.num_observations
    world_model = RecurrentJEPA(BeliefFilter(num_a, num_o, DIM, HIDDEN, 1),
                                LatentPredictor(DIM, num_a, HIDDEN, 1), 0.99).to(device)
    value_head = ValueHead(DIM, HIDDEN, 1, 255).to(device)
    reward_head = RewardHead(DIM, num_a, HIDDEN, 1, 255).to(device)
    observation_head = ObservationHead(DIM, num_a, num_o, HIDDEN, 1).to(device)
    codec = TwoHotSymlog(255, pomdp.value_bound).to(device)
    trainer = WorldModelTrainer(
        world_model, value_head, reward_head, observation_head, codec, num_a, num_o, pomdp.discount,
        TrainerConfig(learning_rate=3e-4, weight_decay=1e-4, grad_clip_norm=1.0, value_target_momentum=0.9), device)

    def probe_data():
        return collect_probe_dataset(pomdp, world_model.belief_filter, uniform_random_policy(num_a, 999, device),
                                     1024, LENGTH, seed=999, device=device)

    untrained_linear = linear_probe(probe_data())
    behaviour = uniform_random_policy(num_a, 123, device)
    for update in range(UPDATES):
        env = BatchedPOMDPEnv(pomdp, BATCH, LENGTH, seed=update, device=device)
        actions, observations, rewards = [], [], []
        for t in range(LENGTH):
            action = behaviour(t, torch.empty(BATCH, pomdp.num_states, device=device))
            out = env.step(action)
            actions.append(action)
            observations.append(out.observation)
            rewards.append(out.reward)
        trainer.train_step(EpisodeBatch(torch.stack(actions, 1), torch.stack(observations, 1),
                                        torch.stack(rewards, 1)))

    dataset = probe_data()
    latents, beliefs = dataset.latents, dataset.posteriors
    measured = {"untrained_linear": untrained_linear, "linear": linear_probe(dataset), "mlp": mlp_probe(dataset)}
    solution = solve_infinite_horizon(pomdp, tolerance=0.1, prune_epsilon=1e-6)
    with torch.no_grad():
        measured["optimal_value"] = solution.value_function.value(beliefs.cpu()).to(device)
        measured["value"] = codec.mean(value_head(latents)).double()
        measured["posterior"] = beliefs[:, 0]
        for a in range(num_a):
            action = torch.full((len(latents),), a, device=device)
            onehot = F.one_hot(action, num_a).float()
            measured[f"reward_error_{a}"] = (codec.mean(reward_head(latents, onehot)).double()
                                             - beliefs @ pomdp.reward.to(device)[a]).abs()
            exact = observation_distribution(pomdp, beliefs, action)
            log_model = F.log_softmax(observation_head(latents, onehot).float(), -1).double()
            measured[f"observation_kl_{a}"] = (exact * (exact.clamp(min=1e-12).log() - log_model)).sum(-1)
    # Greedy planning with the learned model (50 simulations, argmax Q), 256 episodes of 100 steps.
    model = LearnedSearchModel(world_model.belief_filter, reward_head, observation_head, value_head, codec,
                               num_a, num_o, pomdp.discount)
    agent = PlanningAgent(model, BeliefTreeSearch(model, 50, 1.25, 0.3, 0.0, seed=0), 256, 0.0, 0, device)
    episodes, _ = play_episodes(BatchedPOMDPEnv(pomdp, 256, LENGTH, seed=4242, device=device), agent, uniform=False)
    returns = discounted_returns(episodes.rewards, pomdp.discount)
    measured["planning_return"] = (float(returns.mean()), float(returns.std()) / 16.0)
    measured["optimal_return"] = float(solution.value_function.value(pomdp.initial_belief.unsqueeze(0)))
    return measured


@pytest.fixture(scope="module")
def measured():
    if not torch.cuda.is_available():
        pytest.skip("acceptance run is sized for a GPU")
    return train_and_measure()


pytestmark = pytest.mark.slow


def test_belief_filter_encodes_exact_posterior(measured):
    """Measured: linear 0.0013-0.0023, MLP 0.0001-0.0003, worst belief 0.002-0.007; untrained linear 0.022."""
    linear, nonlinear = measured["linear"], measured["mlp"]
    worst_belief = max(kl for _, kl in nonlinear.mean_kl_by_posterior.values())
    assert linear.mean_kl < 0.005, linear
    assert linear.mean_kl * 4 < measured["untrained_linear"].mean_kl
    assert nonlinear.mean_kl < 0.001, nonlinear
    assert worst_belief < 0.02, nonlinear.mean_kl_by_posterior


def test_reward_head_decodes_expected_reward(measured):
    """Exact E[r | b, a] = b . R[a]; LISTEN is deterministic (-1), door rewards are -100/+10 gambles."""
    assert float(measured["reward_error_0"].mean()) < 0.05
    for a in (1, 2):
        assert float(measured[f"reward_error_{a}"].mean()) < 8.0, float(measured[f"reward_error_{a}"].mean())


def test_observation_head_matches_exact_predictive(measured):
    """Measured mean KL: 0.007 (LISTEN), 0.0002 (door actions)."""
    assert float(measured["observation_kl_0"].mean()) < 0.02
    for a in (1, 2):
        assert float(measured[f"observation_kl_{a}"].mean()) < 0.002


def test_value_head_approaches_optimal_value(measured):
    """Fitted value iteration through the learned model; measured mean |V - V*| = 4.9."""
    error = (measured["value"] - measured["optimal_value"]).abs()
    assert float(error.mean()) < 8.0, float(error.mean())


def test_learned_model_planner_is_near_optimal(measured):
    """Greedy search over the learned model must match V*(b0) within sampling error (+1 slack)."""
    mean, stderr = measured["planning_return"]
    assert mean > measured["optimal_return"] - 3 * stderr - 1.0, (mean, stderr, measured["optimal_return"])
