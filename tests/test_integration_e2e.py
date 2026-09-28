# ABSOLUTE PATH: tests/test_integration_e2e.py
"""
End-to-end smoke test on the canonical Tiger, wired exactly as main.py: episode-major collection
with the planning agent over the learned model, whole-episode replay, world-model updates.
"""

import torch

from ipomdp.agents import PlanningAgent
from ipomdp.domain import BatchedPOMDPEnv, build_tiger_pomdp
from ipomdp.models import BeliefFilter, LatentPredictor, ObservationHead, RecurrentJEPA, RewardHead, TwoHotSymlog, ValueHead
from ipomdp.planning import BeliefTreeSearch, LearnedSearchModel
from ipomdp.training import EpisodeBatch, EpisodeBuffer, TrainerConfig, WorldModelTrainer


def test_tiger_collection_and_training_cycle():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    pomdp = build_tiger_pomdp()
    num_envs, episode_length, latent_dim, hidden = 2, 6, 16, 32
    num_actions, num_obs = pomdp.num_actions, pomdp.num_observations

    world_model = RecurrentJEPA(BeliefFilter(num_actions, num_obs, latent_dim, hidden, 1),
                                LatentPredictor(latent_dim, num_actions, hidden, 1), 0.99).to(device)
    value_head = ValueHead(latent_dim, hidden, 1, 255).to(device)
    reward_head = RewardHead(latent_dim, num_actions, hidden, 1, 255).to(device)
    observation_head = ObservationHead(latent_dim, num_actions, num_obs, hidden, 1).to(device)
    codec = TwoHotSymlog(255, pomdp.value_bound).to(device)
    trainer = WorldModelTrainer(
        world_model, value_head, reward_head, observation_head, codec, num_actions, num_obs, pomdp.discount,
        TrainerConfig(learning_rate=3e-4, weight_decay=1e-4, grad_clip_norm=1.0, value_target_momentum=0.99), device)
    model = LearnedSearchModel(world_model.belief_filter, reward_head, observation_head, value_head, codec,
                               num_actions, num_obs, pomdp.discount)
    agent = PlanningAgent(model, BeliefTreeSearch(model, 5, 1.25, 0.3, 0.25, seed=0), num_envs, temperature=1.0,
                          seed=0, device=device)
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
        buffer.add(EpisodeBatch(torch.stack(actions, 1), torch.stack(observations, 1), torch.stack(rewards, 1)))
        metrics = trainer.train_step(buffer.sample(4))
        assert {"loss_prediction", "loss_reward", "loss_value", "loss_observation"} <= metrics.keys()
