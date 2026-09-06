# ABSOLUTE PATH: tests/test_integration_e2e.py
"""End-to-end integration tests for JEPA-IPOMDP training loop and components."""

import pytest
import logging
import torch
import torch.nn.functional as F

from ipomdp.envs.tiger import MultiAgentTigerEnv
from ipomdp.envs.vector import SyncVectorEnv
from ipomdp.models.extractors import MLPFeatureExtractor
from ipomdp.models.heads import ValueHead, RewardHead, DiscretePolicyHead
from ipomdp.models.world_model import RecurrentContextEncoder, CausalRelationalPredictor, RecurrentJEPABase
from ipomdp.planning.mcts import DiscreteLatentOpenLoopSearch
from ipomdp.agents.jepa_agent import DiscreteJEPAAgent, StatelessAgent
from ipomdp.types import Action, Observation
from ipomdp.training.replay_buffer import PrioritizedSequenceBuffer
from ipomdp.training.trainer import DiscreteRecurrentIPOMDPTrainer



class TestEndToEndIntegration:
    """Full system integration test simulating active vectorized training cycles."""

    def test_vectorized_training_cycle_e2e(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_envs = 2
        latent_dim = 32
        num_objects = 2
        action_dim_i = 3
        action_dim_j = 3
        burn_in = 2
        seq_len = 4

        env_fn = lambda: MultiAgentTigerEnv(max_steps=5)
        vec_env = SyncVectorEnv(env_fn, num_envs=num_envs)

        extractor = MLPFeatureExtractor(obs_dim=2, hidden_dim=latent_dim, num_objects=num_objects)
        encoder = RecurrentContextEncoder(extractor, action_dim=action_dim_i, latent_dim=latent_dim, hidden_dim=latent_dim)
        predictor = CausalRelationalPredictor(
            num_objects=num_objects, latent_dim=latent_dim,
            action_dim_i=action_dim_i, action_dim_j=action_dim_j,
            hidden_dim=latent_dim
        )
        jepa_model = RecurrentJEPABase(encoder, predictor).to(device)

        value_head = ValueHead(latent_dim=latent_dim, hidden_dim=latent_dim, num_bins=255).to(device)
        reward_head = RewardHead(latent_dim=latent_dim, action_dim_i=action_dim_i, action_dim_j=action_dim_j, hidden_dim=latent_dim, num_bins=255).to(device)
        opponent_head = DiscretePolicyHead(latent_dim=latent_dim, action_dim=action_dim_j, num_opponents=1, hidden_dim=latent_dim).to(device)

        planner = DiscreteLatentOpenLoopSearch(
            jepa_model=jepa_model,
            value_head=value_head,
            reward_head=reward_head,
            opponent_head=opponent_head,
            action_dim_i=action_dim_i,
            action_dim_j=action_dim_j,
            num_simulations=5,
            num_latent_obs=2
        )

        agents = {
            "agent_0": DiscreteJEPAAgent(
                agent_id="agent_0",
                planner=planner,
                latent_dim=latent_dim,
                action_dim=action_dim_i,
                device=device,
                num_objects=num_objects
            ),
            "agent_1": StatelessAgent(agent_id="agent_1", action_dim=action_dim_j, action_idx=0)
        }

        buffer = PrioritizedSequenceBuffer(capacity=64, burn_in=burn_in, seq_len=seq_len)
        logger = logging.getLogger("e2e_logger")

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

        observations, infos = vec_env.reset()
        for agent in agents.values():
            agent.reset(batch_size=num_envs)

        prev_actions = {
            aid: Action(data=torch.zeros(num_envs, action_dim_i if aid == "agent_0" else action_dim_j, device=device))
            for aid in agents
        }

        for step in range(12):
            agents["agent_0"].update_belief(observations["agent_0"], prev_actions["agent_0"])
            agents["agent_1"].update_belief(observations["agent_1"], prev_actions["agent_1"])

            actions = {aid: agent.act(observations[aid]) for aid, agent in agents.items()}
            next_obs, rews, terms, truncs, infos = vec_env.step(actions)

            for i in range(num_envs):
                r_val = float(rews["agent_0"][i].item())
                buffer.push(
                    env_idx=i,
                    obs=observations["agent_0"].data[i],
                    act_i=actions["agent_0"].data[i],
                    act_j=actions["agent_1"].data[i],
                    reward=r_val
                )

                if terms["agent_0"][i].item() or truncs["agent_0"][i].item():
                    term_obs = infos["agent_0"]["terminal_obs"][i]
                    buffer.end_episode(env_idx=i, final_obs=term_obs)
                    prev_actions["agent_0"].data[i].zero_()
                    prev_actions["agent_1"].data[i].zero_()
                    agents["agent_0"].reset_index(i)
                    agents["agent_1"].reset_index(i)

            observations = next_obs
            prev_actions = actions

            if buffer.tree.size >= 2:
                batch, is_weights, tree_indices = buffer.sample_sequence(batch_size=2)
                metrics, td_errors = trainer.train_sequence(batch, is_weights)
                buffer.update_priorities(tree_indices, td_errors)
                assert "loss_jepa" in metrics
                assert "loss_rl" in metrics

    def test_wumpus_vectorized_training_cycle_e2e(self):
        from ipomdp.envs.wumpus import MultiAgentWumpusEnv
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_envs = 2
        latent_dim = 32
        num_objects = 2
        obs_dim = 5
        action_dim_i = 5
        action_dim_j = 4
        burn_in = 2
        seq_len = 4

        env_fn = lambda: MultiAgentWumpusEnv(grid_size=4, max_steps=5)
        vec_env = SyncVectorEnv(env_fn, num_envs=num_envs)

        extractor = MLPFeatureExtractor(obs_dim=obs_dim, hidden_dim=latent_dim, num_objects=num_objects)
        encoder = RecurrentContextEncoder(extractor, action_dim=action_dim_i, latent_dim=latent_dim, hidden_dim=latent_dim)
        predictor = CausalRelationalPredictor(
            num_objects=num_objects, latent_dim=latent_dim,
            action_dim_i=action_dim_i, action_dim_j=action_dim_j,
            hidden_dim=latent_dim
        )
        jepa_model = RecurrentJEPABase(encoder, predictor).to(device)

        value_head = ValueHead(latent_dim=latent_dim, hidden_dim=latent_dim, num_bins=255).to(device)
        reward_head = RewardHead(latent_dim=latent_dim, action_dim_i=action_dim_i, action_dim_j=action_dim_j, hidden_dim=latent_dim, num_bins=255).to(device)
        opponent_head = DiscretePolicyHead(latent_dim=latent_dim, action_dim=action_dim_j, num_opponents=1, hidden_dim=latent_dim).to(device)

        planner = DiscreteLatentOpenLoopSearch(
            jepa_model=jepa_model,
            value_head=value_head,
            reward_head=reward_head,
            opponent_head=opponent_head,
            action_dim_i=action_dim_i,
            action_dim_j=action_dim_j,
            num_simulations=5,
            num_latent_obs=2
        )

        agents = {
            "agent_0": DiscreteJEPAAgent(
                agent_id="agent_0",
                planner=planner,
                latent_dim=latent_dim,
                action_dim=action_dim_i,
                device=device,
                num_objects=num_objects
            ),
            "agent_1": StatelessAgent(agent_id="agent_1", action_dim=action_dim_j, action_idx=0)
        }

        buffer = PrioritizedSequenceBuffer(capacity=64, burn_in=burn_in, seq_len=seq_len)
        logger = logging.getLogger("wumpus_e2e_logger")

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

        observations, infos = vec_env.reset()
        for agent in agents.values():
            agent.reset(batch_size=num_envs)

        prev_actions = {
            aid: Action(data=torch.zeros(num_envs, action_dim_i if aid == "agent_0" else action_dim_j, device=device))
            for aid in agents
        }

        for step in range(12):
            agents["agent_0"].update_belief(observations["agent_0"], prev_actions["agent_0"])
            agents["agent_1"].update_belief(observations["agent_1"], prev_actions["agent_1"])

            actions = {aid: agent.act(observations[aid]) for aid, agent in agents.items()}
            next_obs, rews, terms, truncs, infos = vec_env.step(actions)

            for i in range(num_envs):
                r_val = float(rews["agent_0"][i].item())
                buffer.push(
                    env_idx=i,
                    obs=observations["agent_0"].data[i],
                    act_i=actions["agent_0"].data[i],
                    act_j=actions["agent_1"].data[i],
                    reward=r_val
                )

                if terms["agent_0"][i].item() or truncs["agent_0"][i].item():
                    term_obs = infos["agent_0"]["terminal_obs"][i]
                    buffer.end_episode(env_idx=i, final_obs=term_obs)
                    prev_actions["agent_0"].data[i].zero_()
                    prev_actions["agent_1"].data[i].zero_()
                    agents["agent_0"].reset_index(i)
                    agents["agent_1"].reset_index(i)

            observations = next_obs
            prev_actions = actions

            if buffer.tree.size >= 2:
                batch, is_weights, tree_indices = buffer.sample_sequence(batch_size=2)
                metrics, td_errors = trainer.train_sequence(batch, is_weights)
                buffer.update_priorities(tree_indices, td_errors)
                assert "loss_jepa" in metrics
                assert "loss_rl" in metrics

