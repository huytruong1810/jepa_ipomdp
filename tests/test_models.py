# ABSOLUTE PATH: tests/test_models.py
"""Unit tests for the model layer: building blocks, two-hot symlog, belief filter, transition, heads."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ipomdp.models import (
    BeliefFilter,
    LatentTransition,
    ObservationProbeHead,
    OpponentPolicyHead,
    RecurrentJEPA,
    RewardHead,
    SwiGLUResidualBlock,
    TwoHotSymlog,
    ValueHead,
    build_residual_stack,
    symexp,
    symlog,
)

A, O, AJ, D, H = 3, 2, 1, 16, 32


def _world_model() -> RecurrentJEPA:
    torch.manual_seed(0)
    return RecurrentJEPA(BeliefFilter(A, O, D, H, 1), LatentTransition(D, A, AJ, H, 1, 4, 4, 0.8), ema_momentum=0.9)


class TestBuildingBlocks:

    def test_swiglu_block_is_identity_at_initialisation(self):
        block = SwiGLUResidualBlock(dim=32)
        x = torch.randn(4, 32)
        assert torch.allclose(block(x), x, atol=1e-6)

    def test_residual_stack_shapes(self):
        net = build_residual_stack(input_dim=16, hidden_dim=32, output_dim=8, num_blocks=2)
        assert net(torch.randn(5, 16)).shape == (5, 8)


class TestTwoHotSymlog:

    def test_symlog_symexp_are_inverse(self):
        x = torch.tensor([-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0])
        assert torch.allclose(symexp(symlog(x)), x, atol=1e-4)

    def test_regression_recovers_targets(self):
        twohot = TwoHotSymlog(min_val=-20.0, max_val=20.0, num_bins=255)
        targets = torch.tensor([[-100.0], [0.0], [10.0], [50.0]])
        logits = nn.Parameter(torch.zeros(4, 255))
        optimizer = torch.optim.Adam([logits], lr=0.1)
        for _ in range(200):
            optimizer.zero_grad()
            twohot(logits, targets, auto_symlog=True).mean().backward()
            optimizer.step()
        assert torch.allclose(twohot.decode(logits, real_scale=False), symlog(targets), atol=0.05)
        assert torch.allclose(twohot.decode(logits, real_scale=True), targets, atol=5.0)


class TestBeliefFilter:

    def test_unroll_equals_repeated_steps_from_learned_initial_latent(self):
        model = _world_model().belief_filter
        actions = F.one_hot(torch.randint(0, A, (5, 7)), A).float()
        observations = F.one_hot(torch.randint(0, O, (5, 7)), O).float()
        unrolled = model.unroll(actions, observations)
        assert unrolled.shape == (5, 8, D)
        assert torch.equal(unrolled[:, 0], model.initial(5))
        latent = model.initial(5)
        for t in range(7):
            latent = model.step(latent, actions[:, t], observations[:, t])
            assert torch.allclose(unrolled[:, t + 1], latent, atol=1e-6)

    def test_latent_depends_on_history(self):
        # Different observation histories must yield different latents (no collapse at init).
        model = _world_model().belief_filter
        listen = F.one_hot(torch.zeros(2, 3, dtype=torch.long), A).float()
        growls = F.one_hot(torch.tensor([[0, 0, 0], [1, 1, 1]]), O).float()
        final = model.unroll(listen, growls)[:, -1]
        assert not torch.allclose(final[0], final[1])

    def test_initial_latent_is_trainable(self):
        model = _world_model().belief_filter
        model.initial(3).sum().backward()
        assert model.initial_latent.grad is not None


class TestRecurrentJEPA:

    def test_target_filter_is_frozen_copy_and_ema_updates(self):
        world_model = _world_model()
        online, target = world_model.belief_filter, world_model.target_filter
        assert all(not p.requires_grad for p in target.parameters())
        assert all(torch.equal(p, q) for p, q in zip(online.parameters(), target.parameters()))

        with torch.no_grad():
            for p in online.parameters():
                p.add_(1.0)
        before = [q.clone() for q in target.parameters()]
        world_model.update_target()
        for p, q, q0 in zip(online.parameters(), target.parameters(), before):
            assert torch.allclose(q, 0.9 * q0 + 0.1 * p)

    def test_transition_shapes_and_kl(self):
        transition = _world_model().transition
        latent = torch.randn(6, D)
        action = F.one_hot(torch.randint(0, A, (6,)), A).float()
        opponent = torch.ones(6, AJ)
        predicted, kl = transition.predict_train(latent, action, opponent, torch.randn(6, D))
        assert predicted.shape == (6, D) and kl.shape == (6,)
        assert (kl >= -1e-6).all()
        assert transition.imagine(latent, action, opponent).shape == (6, D)

    def test_kl_is_zero_for_identical_distributions(self):
        logits = torch.randn(3, 4, 4)
        assert torch.allclose(LatentTransition._categorical_kl(logits, logits), torch.zeros(3), atol=1e-6)


class TestHeads:

    def test_value_and_reward_heads_start_at_zero(self):
        twohot = TwoHotSymlog()
        latent = torch.randn(4, D)
        action = F.one_hot(torch.tensor([0, 1, 2, 0]), A).float()
        value = twohot.decode(ValueHead(D, H, 1, 255)(latent))
        reward = twohot.decode(RewardHead(D, A, AJ, H, 1, 255)(latent, action, torch.ones(4, AJ)))
        assert torch.allclose(value, torch.zeros_like(value), atol=1e-5)
        assert torch.allclose(reward, torch.zeros_like(reward), atol=1e-5)

    def test_policy_and_probe_head_shapes(self):
        latent = torch.randn(4, D)
        assert OpponentPolicyHead(D, 5, H, 1)(latent).shape == (4, 5)
        probe = ObservationProbeHead(D, A, O, H, 1)
        assert probe(latent, F.one_hot(torch.tensor([0, 1, 2, 0]), A).float()).shape == (4, O)
