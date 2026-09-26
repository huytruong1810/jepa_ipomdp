# ABSOLUTE PATH: tests/test_integration_e2e.py
"""
End-to-end smoke test of the training loop on the canonical Tiger: simulator, belief
filtering agent, latent MCTS, sequence buffer and trainer, wired exactly as in main.py.
"""

import logging

import torch
import torch.nn.functional as F

from ipomdp.agents.jepa_agent import DiscreteJEPAAgent
from ipomdp.domain import BatchedPOMDPEnv, build_tiger_pomdp
from ipomdp.models.extractors import MLPFeatureExtractor
from ipomdp.models.heads import DiscretePolicyHead, RewardHead, ValueHead
from ipomdp.models.world_model import CausalRelationalPredictor, RecurrentContextEncoder, RecurrentJEPABase
from ipomdp.planning.mcts import DiscreteLatentOpenLoopSearch
from ipomdp.training.replay_buffer import PrioritizedSequenceBuffer
from ipomdp.training.trainer import DiscreteRecurrentIPOMDPTrainer

# Mirrors main.OPPONENT_ACTION_DIM: the single-agent POMDP has a singleton opponent action space.
OPPONENT_ACTION_DIM = 1


class TestEndToEndIntegration:
    """Full system integration test simulating active vectorized training cycles."""

    def test_tiger_training_cycle_e2e(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(0)
        pomdp = build_tiger_pomdp()
        num_envs, latent_dim, num_objects = 2, 32, 2
        num_actions, num_obs = pomdp.num_actions, pomdp.num_observations
        burn_in, seq_len = 2, 4

        env = BatchedPOMDPEnv(pomdp, num_envs, max_steps=6, seed=0, device=device)

        extractor = MLPFeatureExtractor(obs_dim=num_obs, hidden_dim=latent_dim, num_objects=num_objects)
        encoder = RecurrentContextEncoder(extractor, action_dim=num_actions, latent_dim=latent_dim, hidden_dim=latent_dim)
        predictor = CausalRelationalPredictor(
            num_objects=num_objects, latent_dim=latent_dim,
            action_dim_i=num_actions, action_dim_j=OPPONENT_ACTION_DIM, hidden_dim=latent_dim
        )
        jepa_model = RecurrentJEPABase(encoder, predictor).to(device)
        value_head = ValueHead(latent_dim=latent_dim, hidden_dim=latent_dim, num_bins=255).to(device)
        reward_head = RewardHead(latent_dim=latent_dim, action_dim_i=num_actions, action_dim_j=OPPONENT_ACTION_DIM,
                                 hidden_dim=latent_dim, num_bins=255).to(device)
        opponent_head = DiscretePolicyHead(latent_dim=latent_dim, action_dim=OPPONENT_ACTION_DIM,
                                           num_opponents=1, hidden_dim=latent_dim).to(device)

        planner = DiscreteLatentOpenLoopSearch(
            jepa_model=jepa_model, value_head=value_head, reward_head=reward_head, opponent_head=opponent_head,
            action_dim_i=num_actions, action_dim_j=OPPONENT_ACTION_DIM,
            num_simulations=5, num_latent_obs=1, discount=pomdp.discount
        )
        agent = DiscreteJEPAAgent(
            planner=planner, batch_size=num_envs, num_actions=num_actions, num_objects=num_objects,
            latent_dim=latent_dim, device=device, temperature=1.0, temperature_min=0.1, temperature_decay=0.999
        )
        buffer = PrioritizedSequenceBuffer(capacity=64, burn_in=burn_in, seq_len=seq_len)
        trainer = DiscreteRecurrentIPOMDPTrainer(
            jepa_model=jepa_model, value_head=value_head, reward_head=reward_head, opponent_head=opponent_head,
            logger=logging.getLogger("e2e_logger"), device=device, latent_dim=latent_dim,
            action_dim_i=num_actions, action_dim_j=OPPONENT_ACTION_DIM, num_objects=num_objects,
            gamma=pomdp.discount
        )

        observation = torch.zeros(num_envs, num_obs, device=device)
        no_opponent_action = torch.zeros(1, dtype=torch.int64)
        trained = False
        for _ in range(14):
            agent.observe(observation)
            action = agent.act()
            out = env.step(action)
            next_observation = F.one_hot(out.observation, num_classes=num_obs).float()

            for i in range(num_envs):
                buffer.push(env_idx=i, obs=observation[i], act_i=action[i].view(1),
                            act_j=no_opponent_action, reward=float(out.reward[i]))
                if out.truncated[i]:
                    buffer.end_episode(env_idx=i, final_obs=next_observation[i], terminated=False)

            env.reset_rows(out.truncated)
            agent.reset_rows(out.truncated)
            next_observation[out.truncated] = 0.0
            observation = next_observation

            if buffer.tree.size >= 2:
                batch, is_weights, tree_indices = buffer.sample_sequence(batch_size=2)
                metrics, td_errors = trainer.train_sequence(batch, is_weights)
                buffer.update_priorities(tree_indices, td_errors)
                assert "loss_jepa" in metrics and "loss_rl" in metrics
                trained = True

        assert trained
