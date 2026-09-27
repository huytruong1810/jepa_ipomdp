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
    return RecurrentJEPA(BeliefFilter(A, O, D, H, 1), LatentTransition(D, A, AJ, H, 1, 4, 4, 0.01), ema_momentum=0.9)


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
        for _ in range(2000):
            optimizer.zero_grad()
            codec.loss(logits.expand(2, -1), outcomes).mean().backward()
            optimizer.step()
        assert float(codec.mean(logits)) == pytest.approx(-45.0, abs=0.1)
        # The exact minimiser E[twohot(Y)] has mean E[Y] by linearity of the encoding.
        assert float(codec.encode(outcomes).mean(0) @ codec.bins) == pytest.approx(-45.0, abs=1e-4)

    def test_regression_recovers_deterministic_targets(self):
        codec = TwoHotSymlog(255, 2000.0)
        targets = torch.tensor([-100.0, 0.0, 10.0, 50.0])
        logits = nn.Parameter(torch.zeros(4, 255))
        optimizer = torch.optim.Adam([logits], lr=0.1)
        for _ in range(2000):
            optimizer.zero_grad()
            codec.loss(logits, targets).mean().backward()
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

    def test_transition_shapes_and_kl(self):
        transition = _world_model().transition
        latent = torch.randn(6, D)
        action = F.one_hot(torch.randint(0, A, (6,)), A).float()
        opponent = torch.ones(6, AJ)
        predicted, kl_dynamics, kl_representation = transition.predict_train(latent, action, opponent, torch.randn(6, D))
        assert predicted.shape == (6, D) and kl_dynamics.shape == (6,) and kl_representation.shape == (6,)
        assert torch.allclose(kl_dynamics, kl_representation)  # same value, different gradient routes
        assert (kl_dynamics >= -1e-6).all()
        assert transition.imagine(latent, action, opponent).shape == (6, D)

    def test_kl_gradients_are_routed(self):
        # Dynamics KL trains only the prior; representation KL trains only the posterior.
        transition = _world_model().transition
        latent, target = torch.randn(6, D), torch.randn(6, D)
        action, opponent = F.one_hot(torch.randint(0, A, (6,)), A).float(), torch.ones(6, AJ)
        _, kl_dynamics, kl_representation = transition.predict_train(latent, action, opponent, target)
        prior_grad = torch.autograd.grad(kl_dynamics.sum(), list(transition.prior_net.parameters()), allow_unused=True)
        post_grad = torch.autograd.grad(kl_dynamics.sum(), list(transition.posterior_net.parameters()),
                                        retain_graph=True, allow_unused=True)
        assert any(g is not None and g.abs().sum() > 0 for g in prior_grad)
        assert all(g is None or g.abs().sum() == 0 for g in post_grad)

    def test_unimix_bounds_probabilities_and_kl(self):
        transition = _world_model().transition
        with torch.no_grad():
            transition.prior_net[-1].bias.fill_(0.0)
            transition.prior_net[-1].bias[0::4] = 1e4  # try to force a one-hot prior
        log_probs = transition._log_probs(transition.prior_net, torch.zeros(1, D + A + AJ))
        assert log_probs.exp().min() >= 0.01 / 4 - 1e-7

    def test_kl_is_zero_for_identical_distributions(self):
        log_probs = torch.randn(3, 4, 4).log_softmax(-1)
        assert torch.allclose(LatentTransition.categorical_kl(log_probs, log_probs), torch.zeros(3), atol=1e-6)


class TestHeads:

    def test_value_and_reward_heads_start_at_zero(self):
        twohot = TwoHotSymlog(255, 2000.0)
        latent = torch.randn(4, D)
        action = F.one_hot(torch.tensor([0, 1, 2, 0]), A).float()
        value = twohot.mean(ValueHead(D, H, 1, 255)(latent))
        reward = twohot.mean(RewardHead(D, A, AJ, H, 1, 255)(latent, action, torch.ones(4, AJ)))
        assert torch.allclose(value, torch.zeros_like(value), atol=1e-5)
        assert torch.allclose(reward, torch.zeros_like(reward), atol=1e-5)

    def test_policy_and_probe_head_shapes(self):
        latent = torch.randn(4, D)
        assert OpponentPolicyHead(D, 5, H, 1)(latent).shape == (4, 5)
        probe = ObservationProbeHead(D, A, O, H, 1)
        assert probe(latent, F.one_hot(torch.tensor([0, 1, 2, 0]), A).float()).shape == (4, O)
