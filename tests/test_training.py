# ABSOLUTE PATH: tests/test_training.py
"""Unit tests for the whole-episode replay buffer and the world-model trainer."""

import pytest
import torch

from ipomdp.domain import BatchedPOMDPEnv, build_tiger_pomdp
from ipomdp.models import (BeliefFilter, LatentTransition, OpponentPolicyHead, RecurrentJEPA, RewardHead, TwoHotSymlog,
                           ValueHead)
from ipomdp.training import EpisodeBatch, EpisodeBuffer, TrainerConfig, WorldModelTrainer

CPU = torch.device("cpu")
A, O, AJ, D, H, T = 3, 2, 1, 16, 32, 6
CONFIG = TrainerConfig(learning_rate=3e-4, weight_decay=1e-4, grad_clip_norm=1.0, lambda_return=0.95,
                       kl_dynamics_scale=0.5, kl_representation_scale=0.1, kl_free_nats=1.0, imagination_horizon=3, consistency_scale=0.5)


def _episodes(batch: int, seed: int, device: torch.device = CPU) -> EpisodeBatch:
    pomdp = build_tiger_pomdp()
    env = BatchedPOMDPEnv(pomdp, batch, T, seed, device)
    generator = torch.Generator(device=device).manual_seed(seed)
    actions, observations, rewards = [], [], []
    for _ in range(T):
        action = torch.randint(0, A, (batch,), generator=generator, device=device)
        out = env.step(action)
        actions.append(action)
        observations.append(out.observation)
        rewards.append(out.reward)
    return EpisodeBatch(torch.stack(actions, 1), torch.stack(observations, 1), torch.stack(rewards, 1),
                        torch.zeros(batch, T, dtype=torch.int64, device=device))


def _trainer(device: torch.device = CPU) -> WorldModelTrainer:
    torch.manual_seed(0)
    world_model = RecurrentJEPA(BeliefFilter(A, O, D, H, 1), LatentTransition(D, A, AJ, H, 1, 4, 4, 0.01), 0.99)
    return WorldModelTrainer(world_model.to(device), ValueHead(D, H, 1, 255).to(device),
                             RewardHead(D, A, AJ, H, 1, 255).to(device), OpponentPolicyHead(D, AJ, H, 1).to(device),
                             TwoHotSymlog(255, 2000.0).to(device), A, O, AJ, discount=0.95, config=CONFIG, device=device)


class TestEpisodeBuffer:

    def test_add_sample_and_wraparound(self):
        buffer = EpisodeBuffer(capacity=5, episode_length=T, device=CPU, seed=0)
        with pytest.raises(RuntimeError):
            buffer.sample(1)
        first, second = _episodes(3, 0), _episodes(3, 1)
        buffer.add(first)
        assert buffer.size == 3
        buffer.add(second)  # 6 episodes into capacity 5: the oldest is overwritten
        assert buffer.size == 5
        assert torch.equal(buffer._actions[0], second.actions[2])
        assert torch.equal(buffer._actions[1], first.actions[1])
        sample = buffer.sample(8)
        assert sample.actions.shape == (8, T) and sample.rewards.dtype == torch.float32

    def test_rejects_wrong_episode_length(self):
        buffer = EpisodeBuffer(capacity=5, episode_length=T + 1, device=CPU, seed=0)
        with pytest.raises(ValueError):
            buffer.add(_episodes(2, 0))

    def test_sampling_is_seeded(self):
        def draw(seed):
            buffer = EpisodeBuffer(capacity=16, episode_length=T, device=CPU, seed=seed)
            buffer.add(_episodes(16, 0))
            return buffer.sample(8).actions
        assert torch.equal(draw(3), draw(3))


class TestWorldModelTrainer:

    def test_lambda_returns_match_hand_computation(self):
        trainer = _trainer()
        rewards = torch.tensor([[1.0, 2.0]])
        values = torch.tensor([[0.5, 1.5, 3.0]])
        g1 = 2.0 + 0.95 * 3.0                                     # G_1 bootstraps V(z_2) (truncation)
        g0 = 1.0 + 0.95 * ((1 - 0.95) * 1.5 + 0.95 * g1)
        assert torch.allclose(trainer.lambda_returns(rewards, values), torch.tensor([[g0, g1]]))

    def test_train_step_updates_parameters_and_target(self):
        trainer = _trainer()
        filt, target = trainer.world_model.belief_filter, trainer.world_model.target_filter
        before = [p.clone() for p in filt.parameters()]
        target_before = [p.clone() for p in target.parameters()]
        metrics = trainer.train_step(_episodes(8, 0))
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
        assert any(not torch.equal(a, b) for a, b in zip(before, filt.parameters()))
        assert any(not torch.equal(a, b) for a, b in zip(target_before, target.parameters()))

    def test_non_finite_loss_raises(self):
        trainer = _trainer()
        batch = _episodes(4, 0)
        broken = EpisodeBatch(batch.actions, batch.observations, batch.rewards * float("nan"), batch.opponent_actions)
        with pytest.raises(FloatingPointError):
            trainer.train_step(broken)

    def test_imagination_loss_trains_only_latents_with_multistep_targets(self):
        # With T = 2 and H = 3, only t = 0 has L_0 = 2 > 1, so exactly one imagined latent
        # (d_1 from z_0) is trained; t = 1 has L_1 = 1 and contributes nothing.
        trainer = _trainer()
        latents = torch.randn(1, 2, D)
        actions = torch.nn.functional.one_hot(torch.tensor([[0, 0]]), A).float()
        loss = trainer._imagination_loss(latents, actions, torch.ones(1, 2, AJ))
        assert torch.isfinite(loss) and loss > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_train_step_on_cuda_with_bfloat16_autocast(self):
        trainer = _trainer(torch.device("cuda"))
        metrics = trainer.train_step(_episodes(8, 0, torch.device("cuda")))
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
