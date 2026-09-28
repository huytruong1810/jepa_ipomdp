# ABSOLUTE PATH: tests/test_world_model_acceptance.py
"""
Phase-2/3 acceptance tests: the world model trained by the real WorldModelTrainer on canonical
Tiger must reproduce exact quantities computed from the FinitePOMDP.

Protocol (fixed seeds, GPU):
    - 1000 updates, each on 64 fresh uniformly-random-policy episodes of 100 steps (no MCTS).
      100-step episodes are required for the value check (conf/env/tiger.yaml, section 3).
    - Held-out data: 1024 random-policy episodes of 100 steps; latents from the online filter
      run from its learned z_0; exact posteriors b* from the batched Bayes filter.

Checked against exact references:
    Belief (Phase 2)       probes of z_t vs b*(h_t)                   KL(b* || probe)
    Reward (Phase 3)       decoded mean of R(z_t, a) vs b* . R[a]     absolute error
    Observation (Phase 3)  P(o' | z_t, a) vs exact P(o' | b*, a)      KL
    Value (Phase 3)        decoded V(z_t) vs V^pi = r_bar / (1 - gamma)
                           For the uniform random policy on Tiger the expected immediate
                           reward is (-1 + (-45) + (-45)) / 3 = -30.33 at EVERY belief, so
                           V^pi(b) = -606.67 for all b. This checks that lambda-returns
                           bootstrap correctly through truncation and that the bootstrap latent
                           z_T is anchored (with 20-step episodes V was biased by 8%).
Thresholds are set from measured runs with margin; see each test's docstring.
"""

import pytest
import torch
import torch.nn.functional as F

from ipomdp.domain import BatchedPOMDPEnv, build_tiger_pomdp, observation_distribution
from ipomdp.interpretability import collect_probe_dataset, linear_probe, mlp_probe, uniform_random_policy
from ipomdp.models import BeliefFilter, LatentPredictor, ObservationHead, RecurrentJEPA, RewardHead, TwoHotSymlog, ValueHead
from ipomdp.training import EpisodeBatch, TrainerConfig, WorldModelTrainer

UPDATES, BATCH, LENGTH, DIM, HIDDEN = 1000, 64, 100, 32, 64


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
        TrainerConfig(learning_rate=3e-4, weight_decay=1e-4, grad_clip_norm=1.0, lambda_return=0.95), device)

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
    with torch.no_grad():
        random_policy_value = float(torch.stack([beliefs @ pomdp.reward.to(device)[a] for a in range(num_a)]).mean(0)[0])
        measured["random_policy_value"] = random_policy_value / (1.0 - pomdp.discount)
        measured["value"] = codec.mean(value_head(latents)).double()
        for a in range(num_a):
            action = torch.full((len(latents),), a, device=device)
            onehot = F.one_hot(action, num_a).float()
            measured[f"reward_error_{a}"] = (codec.mean(reward_head(latents, onehot)).double()
                                             - beliefs @ pomdp.reward.to(device)[a]).abs()
            exact = observation_distribution(pomdp, beliefs, action)
            log_model = F.log_softmax(observation_head(latents, onehot).float(), -1).double()
            measured[f"observation_kl_{a}"] = (exact * (exact.clamp(min=1e-12).log() - log_model)).sum(-1)
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


def test_value_head_matches_exact_random_policy_value(measured):
    """V^pi(b) = -606.67 for every belief under the uniform random policy. Measured: -604 to -612."""
    exact = measured["random_policy_value"]
    assert exact == pytest.approx(-606.6667, abs=1e-3)
    value = measured["value"]
    assert abs(float(value.mean()) - exact) < 0.02 * abs(exact), float(value.mean())
    assert float(value.std()) < 0.01 * abs(exact), float(value.std())  # flat across beliefs and time
