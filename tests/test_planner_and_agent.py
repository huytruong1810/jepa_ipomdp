# ABSOLUTE PATH: tests/test_planner_and_agent.py
"""Unit tests for the latent MCTS planner and the batched belief-filtering agent (reviewed in Phase 4)."""

import pytest
import torch
import torch.nn.functional as F

from ipomdp.agents import DiscreteJEPAAgent
from ipomdp.models import BeliefFilter, LatentTransition, OpponentPolicyHead, RecurrentJEPA, RewardHead, ValueHead
from ipomdp.planning import DiscreteLatentOpenLoopSearch, MinMaxStats

CPU = torch.device("cpu")
A, O, AJ, D, H = 3, 2, 1, 16, 32


@pytest.fixture
def components():
    torch.manual_seed(0)
    world_model = RecurrentJEPA(BeliefFilter(A, O, D, H, 1), LatentTransition(D, A, AJ, H, 1, 4, 4, 0.8), 0.99)
    planner = DiscreteLatentOpenLoopSearch(
        jepa_model=world_model, value_head=ValueHead(D, H, 1, 255), reward_head=RewardHead(D, A, AJ, H, 1, 255),
        opponent_head=OpponentPolicyHead(D, AJ, H, 1), action_dim_i=A, action_dim_j=AJ,
        num_simulations=10, num_latent_obs=2, discount=0.95)
    return world_model, planner


class TestPlanner:

    def test_search_returns_distributions(self, components):
        _, planner = components
        policy = planner.search(torch.randn(2, D), temperature=1.0)
        assert policy.shape == (2, A)
        assert torch.allclose(policy.sum(dim=-1), torch.ones(2), atol=1e-4)

    def test_min_max_stats(self):
        stats = MinMaxStats()
        stats.update(10.0)
        stats.update(-10.0)
        assert (stats.maximum, stats.minimum) == (10.0, -10.0)
        assert stats.normalize(0.0) == 0.5
        assert stats.normalize(10.0) == 1.0
        assert stats.normalize(-10.0) == 0.0


class TestAgent:

    def test_lifecycle_follows_canonical_timing(self, components):
        world_model, planner = components
        agent = DiscreteJEPAAgent(world_model.belief_filter, planner, batch_size=2, num_actions=A, num_observations=O,
                                  device=CPU, temperature=1.0, temperature_min=0.1, temperature_decay=0.5)
        # reset() places every row at the learned z_0 (the prior), with no observation.
        assert torch.equal(agent.belief, world_model.belief_filter.initial(2))

        action = agent.act()
        assert action.shape == (2,) and action.dtype == torch.int64
        observation = torch.tensor([0, 1])
        agent.update(action, observation)
        expected = world_model.belief_filter.step(
            world_model.belief_filter.initial(2), F.one_hot(action, A).float(), F.one_hot(observation, O).float())
        assert torch.allclose(agent.belief, expected)

        agent.reset()
        assert torch.equal(agent.belief, world_model.belief_filter.initial(2))

        uniform = agent.act_uniformly()
        assert uniform.shape == (2,) and ((uniform >= 0) & (uniform < A)).all()

    def test_temperature_annealing(self, components):
        world_model, planner = components
        agent = DiscreteJEPAAgent(world_model.belief_filter, planner, batch_size=1, num_actions=A, num_observations=O,
                                  device=CPU, temperature=1.0, temperature_min=0.1, temperature_decay=0.5)
        assert agent.anneal_temperature() == 0.5
        for _ in range(10):
            agent.anneal_temperature()
        assert agent.temperature == 0.1
