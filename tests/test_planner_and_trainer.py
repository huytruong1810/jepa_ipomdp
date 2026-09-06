# ABSOLUTE PATH: tests/test_planner_and_trainer.py
"""Unit tests for MCTS latent planner, DiscreteJEPAAgent, and sequence trainer."""

import pytest
import logging
import torch
import torch.nn.functional as F

from ipomdp.models.extractors import MLPFeatureExtractor
from ipomdp.models.heads import ValueHead, RewardHead, DiscretePolicyHead
from ipomdp.models.world_model import RecurrentContextEncoder, CausalRelationalPredictor, RecurrentJEPABase
from ipomdp.planning.mcts import DiscreteLatentOpenLoopSearch, MinMaxStats, LatentSearchNode
from ipomdp.agents.jepa_agent import DiscreteJEPAAgent, StatelessAgent
from ipomdp.types import Action, Observation
from ipomdp.training.trainer import DiscreteRecurrentIPOMDPTrainer



@pytest.fixture
def test_setup():
    latent_dim = 32
    num_objects = 2
    action_dim_i = 3
    action_dim_j = 3
    device = torch.device("cpu")

    extractor = MLPFeatureExtractor(obs_dim=2, hidden_dim=latent_dim, num_objects=num_objects)
    encoder = RecurrentContextEncoder(extractor, action_dim=action_dim_i, latent_dim=latent_dim, hidden_dim=latent_dim)
    predictor = CausalRelationalPredictor(
        num_objects=num_objects, latent_dim=latent_dim,
        action_dim_i=action_dim_i, action_dim_j=action_dim_j,
        hidden_dim=latent_dim
    )
    jepa_model = RecurrentJEPABase(encoder, predictor)

    value_head = ValueHead(latent_dim=latent_dim, hidden_dim=latent_dim, num_bins=255)
    reward_head = RewardHead(latent_dim=latent_dim, action_dim_i=action_dim_i, action_dim_j=action_dim_j, hidden_dim=latent_dim, num_bins=255)
    opponent_head = DiscretePolicyHead(latent_dim=latent_dim, action_dim=action_dim_j, num_opponents=1, hidden_dim=latent_dim)

    planner = DiscreteLatentOpenLoopSearch(
        jepa_model=jepa_model,
        value_head=value_head,
        reward_head=reward_head,
        opponent_head=opponent_head,
        action_dim_i=action_dim_i,
        action_dim_j=action_dim_j,
        num_simulations=10,
        num_latent_obs=2
    )

    logger = logging.getLogger("test_trainer")
    trainer = DiscreteRecurrentIPOMDPTrainer(
        jepa_model=jepa_model,
        value_head=value_head,
        reward_head=reward_head,
        opponent_head=opponent_head,
        logger=logger,
        device=device,
        latent_dim=latent_dim,
        action_dim_i=action_dim_i,
        action_dim_j=action_dim_j,
        num_objects=num_objects
    )

    return {
        "jepa_model": jepa_model,
        "value_head": value_head,
        "reward_head": reward_head,
        "opponent_head": opponent_head,
        "planner": planner,
        "trainer": trainer,
        "latent_dim": latent_dim,
        "action_dim_i": action_dim_i,
        "action_dim_j": action_dim_j,
        "num_objects": num_objects,
        "device": device
    }


class TestAgentAndPlanner:
    """Rigorous tests for DiscreteJEPAAgent and Latent MCTS planner."""

    def test_discrete_jepa_agent_lifecycle(self, test_setup):
        agent = DiscreteJEPAAgent(
            agent_id="agent_0",
            planner=test_setup["planner"],
            latent_dim=test_setup["latent_dim"],
            action_dim=test_setup["action_dim_i"],
            device=test_setup["device"],
            num_objects=test_setup["num_objects"]
        )

        b_batch = 2
        agent.reset(batch_size=b_batch)
        assert agent.belief.shape == torch.Size([b_batch, 2, 32])
        assert agent.prev_action.shape == torch.Size([b_batch, 3])

        # Test single-channel reset isolation
        agent.belief[0].fill_(1.0)
        agent.belief[1].fill_(2.0)
        agent.reset_index(0)
        assert (agent.belief[0] == 0.0).all()
        assert (agent.belief[1] == 2.0).all()

        # Update belief
        obs = Observation(torch.randn(b_batch, 2))
        prev_a = Action(torch.zeros(b_batch, 1))
        agent.update_belief(obs, prev_a)
        assert agent.belief.shape == torch.Size([b_batch, 2, 32])

        # Select action
        action = agent.act(obs)
        assert action.data.shape == torch.Size([b_batch, 1])

        # Temperature decay
        t_init = agent.temperature
        t_decay = agent.anneal_temperature()
        assert t_decay <= t_init

    def test_latent_mcts_search(self, test_setup):
        planner = test_setup["planner"]
        b_batch = 2
        root_belief = torch.randn(b_batch, test_setup["num_objects"], test_setup["latent_dim"])

        policy_dist = planner.search(root_belief, temperature=1.0)
        assert policy_dist.shape == torch.Size([b_batch, test_setup["action_dim_i"]])
        assert torch.allclose(policy_dist.sum(dim=-1), torch.ones(b_batch), atol=1e-4)

    def test_min_max_stats(self):
        stats = MinMaxStats()
        stats.update(10.0)
        stats.update(-10.0)
        assert stats.maximum == 10.0
        assert stats.minimum == -10.0
        assert stats.normalize(0.0) == 0.5
        assert stats.normalize(10.0) == 1.0
        assert stats.normalize(-10.0) == 0.0


class TestDiscreteRecurrentTrainer:
    """Rigorous tests for sequence optimization trainer."""

    def test_lambda_returns_computation(self, test_setup):
        trainer = test_setup["trainer"]
        b_batch, t_steps = 2, 5
        rewards = torch.tensor([[[-1.0], [-1.0], [-1.0], [-1.0], [10.0]], [[-1.0], [-1.0], [-1.0], [-1.0], [-100.0]]])
        values = torch.zeros(b_batch, t_steps + 1, 1)
        mask = torch.ones(b_batch, t_steps, 1)

        returns = trainer.compute_lambda_returns(rewards, values, mask)
        assert returns.shape == torch.Size([b_batch, t_steps, 1])

    def test_train_sequence_step(self, test_setup):
        trainer = test_setup["trainer"]
        b_batch = 4
        chunk_len = 6
        obs_dim = 2
        latent_dim = test_setup["latent_dim"]

        batch = {
            "obs": torch.randn(b_batch, chunk_len + 1, obs_dim),
            "act_i": torch.randint(0, 3, (b_batch, chunk_len, 1)),
            "act_j": torch.randint(0, 3, (b_batch, chunk_len, 1)),
            "rewards": torch.randn(b_batch, chunk_len, 1),
            "mask": torch.ones(b_batch, chunk_len, 1),
            "prev_act_i": torch.randint(0, 3, (b_batch, 1))
        }
        is_weights = torch.ones(b_batch, 1)

        metrics, td_errors = trainer.train_sequence(batch, is_weights)

        assert "loss_jepa" in metrics
        assert "loss_vicreg" in metrics
        assert "loss_rl" in metrics
        assert "loss_consistency" in metrics
        assert "mean_td_error" in metrics
        assert td_errors.shape == torch.Size([b_batch])
