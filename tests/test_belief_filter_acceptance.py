# ABSOLUTE PATH: tests/test_belief_filter_acceptance.py
"""
Phase-2 acceptance test: the JEPA belief filter, trained by the real WorldModelTrainer on
canonical Tiger, must encode the exact Bayes posterior.

Protocol (fixed seeds):
    - 1000 updates, each on 256 fresh random-policy episodes of 20 steps (no MCTS involved).
    - Probe data: 4096 held-out random-policy episodes; latents from the online filter run
      from its learned z_0; targets are exact posteriors from the batched Bayes filter.
Thresholds (nats), set from the Phase-2 study where the trained filter reached
linear 0.0013-0.0023 / MLP 0.0001-0.0003 / worst-per-belief 0.002-0.007 and the untrained
filter reached linear 0.0216:
    - linear-probe mean KL < 0.005 and at least 4x below the untrained filter's;
    - MLP-probe mean KL < 0.001;
    - MLP-probe mean KL of every distinct exact posterior value < 0.02.
"""

import pytest
import torch

from ipomdp.domain import BatchedPOMDPEnv, build_tiger_pomdp
from ipomdp.interpretability import collect_probe_dataset, linear_probe, mlp_probe, uniform_random_policy
from ipomdp.models import BeliefFilter, LatentTransition, OpponentPolicyHead, RecurrentJEPA, RewardHead, ValueHead
from ipomdp.training import EpisodeBatch, TrainerConfig, WorldModelTrainer

UPDATES, BATCH, LENGTH = 1000, 256, 20


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="acceptance run is sized for a GPU")
def test_trained_filter_encodes_exact_posterior():
    device = torch.device("cuda")
    torch.manual_seed(0)
    pomdp = build_tiger_pomdp()
    num_a, num_o, dim, hidden = pomdp.num_actions, pomdp.num_observations, 32, 64
    world_model = RecurrentJEPA(BeliefFilter(num_a, num_o, dim, hidden, 1),
                                LatentTransition(dim, num_a, 1, hidden, 1, 4, 4, 0.8), 0.99).to(device)
    trainer = WorldModelTrainer(
        world_model, ValueHead(dim, hidden, 1, 255).to(device), RewardHead(dim, num_a, 1, hidden, 1, 255).to(device),
        OpponentPolicyHead(dim, 1, hidden, 1).to(device), num_a, num_o, 1, pomdp.discount,
        TrainerConfig(learning_rate=3e-4, weight_decay=1e-4, grad_clip_norm=1.0, lambda_return=0.95,
                      kl_free_nats=1.0, kl_scale=0.1, imagination_horizon=3, consistency_scale=0.5),
        device)

    def probe_data():
        return collect_probe_dataset(pomdp, world_model.belief_filter, uniform_random_policy(num_a, 999, device),
                                     4096, LENGTH, seed=999, device=device)

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
        trainer.train_step(EpisodeBatch(torch.stack(actions, 1), torch.stack(observations, 1), torch.stack(rewards, 1),
                                        torch.zeros(BATCH, LENGTH, dtype=torch.int64, device=device)))

    dataset = probe_data()
    linear, nonlinear = linear_probe(dataset), mlp_probe(dataset)
    worst_belief = max(kl for _, kl in nonlinear.mean_kl_by_posterior.values())
    assert linear.mean_kl < 0.005, linear
    assert linear.mean_kl * 4 < untrained_linear.mean_kl, (linear, untrained_linear)
    assert nonlinear.mean_kl < 0.001, nonlinear
    assert worst_belief < 0.02, nonlinear.mean_kl_by_posterior
