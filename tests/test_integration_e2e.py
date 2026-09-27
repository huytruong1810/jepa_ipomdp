# ABSOLUTE PATH: tests/test_integration_e2e.py
"""
End-to-end smoke test on the canonical Tiger, wired exactly as main.py: episode-major
collection with the belief-filtering MCTS agent, whole-episode replay, world-model updates.
"""

import torch

from ipomdp.agents import DiscreteJEPAAgent
from ipomdp.domain import BatchedPOMDPEnv, build_tiger_pomdp
from ipomdp.models import (BeliefFilter, LatentPredictor, ObservationHead, OpponentPolicyHead, RecurrentJEPA, RewardHead,
                           TwoHotSymlog, ValueHead)
from ipomdp.planning import LatentBeliefTreeSearch
from ipomdp.training import EpisodeBatch, EpisodeBuffer, TrainerConfig, WorldModelTrainer

# Mirrors main.OPPONENT_ACTION_DIM: the single-agent POMDP has a singleton opponent action space.
OPPONENT_ACTION_DIM = 1


def test_tiger_collection_and_training_cycle():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    pomdp = build_tiger_pomdp()
    num_envs, episode_length, latent_dim, hidden = 2, 6, 16, 32
    num_actions, num_obs = pomdp.num_actions, pomdp.num_observations

    world_model = RecurrentJEPA(
        BeliefFilter(num_actions, num_obs, latent_dim, hidden, 1),
        LatentPredictor(latent_dim, num_actions, OPPONENT_ACTION_DIM, hidden, 1), 0.99).to(device)
    value_head = ValueHead(latent_dim, hidden, 1, 255).to(device)
    reward_head = RewardHead(latent_dim, num_actions, OPPONENT_ACTION_DIM, hidden, 1, 255).to(device)
    opponent_head = OpponentPolicyHead(latent_dim, OPPONENT_ACTION_DIM, hidden, 1).to(device)
    observation_head = ObservationHead(latent_dim, num_actions, OPPONENT_ACTION_DIM, num_obs, hidden, 1).to(device)
    codec = TwoHotSymlog(255, pomdp.value_bound).to(device)
    planner = LatentBeliefTreeSearch(
        world_model=world_model, value_head=value_head, reward_head=reward_head, opponent_head=opponent_head, observation_head=observation_head, codec=codec,
        action_dim_i=num_actions, action_dim_j=OPPONENT_ACTION_DIM, num_simulations=5, num_observation_samples=1,
        discount=pomdp.discount)
    agent = DiscreteJEPAAgent(world_model.belief_filter, planner, num_envs, num_actions, num_obs, device,
                              temperature=1.0, temperature_min=0.1, temperature_decay=0.999)
    trainer = WorldModelTrainer(
        world_model, value_head, reward_head, opponent_head, observation_head, codec, num_actions, num_obs, OPPONENT_ACTION_DIM,
        pomdp.discount,
        TrainerConfig(learning_rate=3e-4, weight_decay=1e-4, grad_clip_norm=1.0, lambda_return=0.95),
        device)
    env = BatchedPOMDPEnv(pomdp, num_envs, max_steps=episode_length, seed=0, device=device)
    buffer = EpisodeBuffer(capacity=16, episode_length=episode_length, device=device, seed=0)

    for _ in range(2):
        env.reset()
        agent.reset()
        actions, observations, rewards = [], [], []
        for _ in range(episode_length):
            action = agent.act()
            out = env.step(action)
            agent.update(action, out.observation)
            actions.append(action)
            observations.append(out.observation)
            rewards.append(out.reward)
        assert bool(out.truncated.all())
        buffer.add(EpisodeBatch(torch.stack(actions, 1), torch.stack(observations, 1), torch.stack(rewards, 1),
                                torch.zeros(num_envs, episode_length, dtype=torch.int64, device=device)))
        metrics = trainer.train_step(buffer.sample(4))
        assert {"loss_prediction", "loss_reward", "loss_value"} <= metrics.keys()
