# ABSOLUTE PATH: tests/test_models.py
"""Unit tests for the model layer: building blocks, two-hot symlog, belief filter, transition, heads."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ipomdp.models import (
    BeliefFilter,
    LatentPredictor,
    ObservationHead,
    RecurrentJEPA,
    RewardHead,
    SwiGLUResidualBlock,
    TwoHotSymlog,
    ValueHead,
    build_residual_stack,
    symexp,
    symlog,
)

A, O, D, H = 3, 2, 16, 32


def _world_model() -> RecurrentJEPA:
    torch.manual_seed(0)
    return RecurrentJEPA(BeliefFilter(A, O, D, H, 1), LatentPredictor(D, A, H, 1), ema_momentum=0.9)


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

    def test_bins_are_symmetric_and_include_zero(self):
        bins = TwoHotSymlog(255, 2000.0).bins
        assert bins[127] == 0.0
        assert torch.allclose(bins, -bins.flip(0))

    @pytest.mark.parametrize("num_bins", [2, 254])
    def test_rejects_even_bin_counts(self, num_bins):
        with pytest.raises(ValueError):
            TwoHotSymlog(num_bins, 2000.0)

    def test_loss_is_cross_entropy_to_encoding(self):
        codec = TwoHotSymlog(255, 2000.0)
        logits, targets = torch.randn(3, 255), torch.tensor([-100.0, 0.0, 10.0])
        expected = -(codec.encode(targets) * torch.log_softmax(logits, -1)).sum(-1)
        assert torch.allclose(codec.loss(logits, targets), expected)

    def test_encoding_has_exact_mean(self):
        codec = TwoHotSymlog(255, 2000.0)
        targets = torch.tensor([-100.0, -45.0, -1.0, 0.0, 0.37, 10.0, 19.3713, 1234.5])
        encoded = codec.encode(targets)
        assert torch.allclose(encoded.sum(-1), torch.ones(len(targets)))
        assert (encoded >= 0).all() and ((encoded > 0).sum(-1) <= 2).all()
        assert torch.allclose(encoded @ codec.bins, targets, rtol=1e-5, atol=1e-4)

    def test_mean_is_unbiased_for_multimodal_targets(self):
        # Tiger's door-opening reward at b = 0.5: -100 or +10 with probability 1/2 (mean -45).
        # The former symlog-space decoder returned -2.9 here. The expected loss is optimised
        # exactly (no outcome sampling), so the fitted mean must converge to -45.
        codec = TwoHotSymlog(255, 2000.0)
        outcomes = torch.tensor([-100.0, 10.0])
        logits = nn.Parameter(torch.zeros(1, 255))
        optimizer = torch.optim.Adam([logits], lr=0.1)
        target = codec.encode(outcomes)  # identical to codec.loss, encoded once for speed
        for _ in range(2000):
            optimizer.zero_grad()
            (-(target * torch.log_softmax(logits.expand(2, -1), dim=-1)).sum(-1)).mean().backward()
            optimizer.step()
        assert float(codec.mean(logits.detach())) == pytest.approx(-45.0, abs=0.1)
        # The exact minimiser E[twohot(Y)] has mean E[Y] by linearity of the encoding.
        assert float(codec.encode(outcomes).mean(0) @ codec.bins) == pytest.approx(-45.0, abs=1e-4)

    def test_regression_recovers_deterministic_targets(self):
        codec = TwoHotSymlog(255, 2000.0)
        targets = torch.tensor([-100.0, 0.0, 10.0, 50.0])
        logits = nn.Parameter(torch.zeros(4, 255))
        optimizer = torch.optim.Adam([logits], lr=0.1)
        encoded = codec.encode(targets)  # identical to codec.loss, encoded once for speed
        for _ in range(2000):
            optimizer.zero_grad()
            (-(encoded * torch.log_softmax(logits, dim=-1)).sum(-1)).mean().backward()
            optimizer.step()
        assert torch.allclose(codec.mean(logits), targets, atol=0.1)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), 2000.5])
    def test_invalid_targets_raise(self, bad):
        with pytest.raises(FloatingPointError):
            TwoHotSymlog(255, 2000.0).encode(torch.tensor([1.0, bad]))

    def test_bound_targets_are_encoded_exactly(self):
        codec = TwoHotSymlog(255, 2000.0)
        targets = torch.tensor([-2000.0, 2000.0])
        assert torch.allclose(codec.encode(targets) @ codec.bins, targets)


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

    def test_predictor_shape(self):
        predictor = _world_model().predictor
        action = F.one_hot(torch.randint(0, A, (6,)), A).float()
        assert predictor(torch.randn(6, D), action).shape == (6, D)


class TestHeads:

    def test_value_and_reward_heads_start_at_zero(self):
        twohot = TwoHotSymlog(255, 2000.0)
        latent = torch.randn(4, D)
        action = F.one_hot(torch.tensor([0, 1, 2, 0]), A).float()
        value = twohot.mean(ValueHead(D, H, 1, 255)(latent))
        reward = twohot.mean(RewardHead(D, A, H, 1, 255)(latent, action))
        assert torch.allclose(value, torch.zeros_like(value), atol=1e-5)
        assert torch.allclose(reward, torch.zeros_like(reward), atol=1e-5)

    def test_observation_head_shape(self):
        latent = torch.randn(4, D)
        head = ObservationHead(D, A, O, H, 1)
        assert head(latent, F.one_hot(torch.tensor([0, 1, 2, 0]), A).float()).shape == (4, O)
