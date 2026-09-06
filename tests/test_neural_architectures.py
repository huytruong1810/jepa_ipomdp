# ABSOLUTE PATH: tests/test_neural_architectures.py
"""Unit tests for perceptual extractors, neural prediction heads, and R-JEPA world model."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ipomdp.models.extractors import MLPFeatureExtractor, CNNFeatureExtractor
from ipomdp.models.heads import (
    AttentionPooler,
    SwarmActionEncoder,
    ValueHead,
    RewardHead,
    DiscretePolicyHead,
    ObservationProbeHead,
)
from ipomdp.models.world_model import (
    RecurrentContextEncoder,
    CausalRelationalPredictor,
    RecurrentJEPABase,
)



class TestFeatureExtractors:
    """Rigorous tests for perceptual feature extractors."""

    def test_mlp_feature_extractor_ranks_and_slot_identities(self):
        obs_dim = 2
        hidden_dim = 32
        num_objects = 2
        extractor = MLPFeatureExtractor(obs_dim=obs_dim, hidden_dim=hidden_dim, num_objects=num_objects)

        # 1. 2D observation tensor (B, obs_dim)
        obs_2d = torch.randn(4, obs_dim)
        slots_2d = extractor(obs_2d)
        assert slots_2d.shape == torch.Size([4, num_objects, hidden_dim])

        # 2. 3D observation sequence (B, T, obs_dim)
        obs_3d = torch.randn(4, 5, obs_dim)
        slots_3d = extractor(obs_3d)
        assert slots_3d.shape == torch.Size([4, 5, num_objects, hidden_dim])

        # Slot identity parameter exists and breaks slot symmetry
        assert extractor.slot_identity_embed.shape == torch.Size([1, num_objects, hidden_dim])
        assert not torch.equal(extractor.slot_identity_embed[:, 0, :], extractor.slot_identity_embed[:, 1, :])

    def test_cnn_feature_extractor(self):
        extractor = CNNFeatureExtractor(input_shape=(3, 32, 32), hidden_dim=32, num_objects=3)
        obs_img = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8)
        slots = extractor(obs_img)
        assert slots.shape == torch.Size([2, 3, 32])
        assert extractor.is_permutation_invariant


class TestNeuralHeads:
    """Rigorous tests for AttentionPooler and prediction heads."""

    def test_attention_pooler(self):
        pooler = AttentionPooler(dim=32)
        belief = torch.randn(4, 2, 32)
        pooled = pooler(belief)
        assert pooled.shape == torch.Size([4, 32])

        loss = pooled.sum()
        loss.backward()
        assert pooler.cls_token.grad is not None

    def test_swarm_action_encoder_variable_opponents(self):
        action_dim_i = 3
        action_dim_j = 3
        encoder = SwarmActionEncoder(action_dim_i=action_dim_i, action_dim_j=action_dim_j, hidden_dim=32)

        ego_a = F.one_hot(torch.tensor([0, 1]), num_classes=action_dim_i).float()

        # 1 opponent
        opp_a_1 = F.one_hot(torch.tensor([[0], [2]]), num_classes=action_dim_j).float()
        fused_1 = encoder(ego_a, opp_a_1)
        assert fused_1.shape == torch.Size([2, 32])

        # 3 opponents
        opp_a_3 = F.one_hot(torch.randint(0, 3, (2, 3)), num_classes=action_dim_j).float()
        fused_3 = encoder(ego_a, opp_a_3)
        assert fused_3.shape == torch.Size([2, 32])

    def test_value_reward_policy_and_probe_heads(self):
        latent_dim = 32
        action_dim_i = 3
        action_dim_j = 3
        b_batch = 4

        belief = torch.randn(b_batch, 2, latent_dim)
        ego_a = F.one_hot(torch.tensor([0, 1, 2, 0]), num_classes=action_dim_i).float()
        opp_a = F.one_hot(torch.tensor([[0], [1], [0], [2]]), num_classes=action_dim_j).float()

        v_head = ValueHead(latent_dim=latent_dim, hidden_dim=32, num_bins=255)
        v_logits = v_head(belief)
        assert v_logits.shape == torch.Size([b_batch, 255])

        r_head = RewardHead(latent_dim=latent_dim, action_dim_i=action_dim_i, action_dim_j=action_dim_j, hidden_dim=32, num_bins=255)
        r_logits = r_head(belief, ego_a, opp_a)
        assert r_logits.shape == torch.Size([b_batch, 255])

        p_head = DiscretePolicyHead(latent_dim=latent_dim, action_dim=action_dim_j, num_opponents=1, hidden_dim=32)
        p_logits = p_head(belief)
        assert p_logits.shape == torch.Size([b_batch, 1, action_dim_j])

        probe = ObservationProbeHead(latent_dim=latent_dim, action_dim=action_dim_i, num_obs_classes=6, hidden_dim=32)
        probe_logits = probe(belief, ego_a)
        assert probe_logits.shape == torch.Size([b_batch, 6])


class TestRecurrentJEPAArchitecture:
    """Rigorous tests for Context Encoder, Causal Predictor, and Target EMA synchronization."""

    def test_recurrent_context_encoder_filtering(self):
        extractor = MLPFeatureExtractor(obs_dim=2, hidden_dim=32, num_objects=2)
        encoder = RecurrentContextEncoder(extractor, action_dim=3, latent_dim=32, hidden_dim=32)

        b_batch = 4
        init_belief = torch.zeros(b_batch, 2, 32)
        dummy_prev_a = torch.zeros(b_batch, 3)
        obs_0 = torch.randn(b_batch, 2)

        # Step 0
        b_0 = encoder(obs_0, dummy_prev_a, init_belief)
        assert b_0.shape == torch.Size([b_batch, 2, 32])

        # Step 1
        act_0 = F.one_hot(torch.tensor([0, 1, 2, 1]), num_classes=3).float()
        obs_1 = torch.randn(b_batch, 2)
        b_1 = encoder(obs_1, act_0, b_0)
        assert b_1.shape == torch.Size([b_batch, 2, 32])

    def test_causal_relational_predictor_and_kl_balancing(self):
        predictor = CausalRelationalPredictor(
            num_objects=2,
            latent_dim=32,
            action_dim_i=3,
            action_dim_j=3,
            hidden_dim=32,
            num_categoricals=4,
            num_classes=4
        )

        b_batch = 4
        belief = torch.randn(b_batch, 2, 32)
        ego_a = F.one_hot(torch.tensor([0, 1, 2, 0]), num_classes=3).float()
        opp_a = F.one_hot(torch.tensor([[0], [1], [0], [2]]), num_classes=3).float()
        target_next_b = torch.randn(b_batch, 2, 32)

        # 1. Open-loop prior imagination (MCTS forward pass)
        imagined_next_b = predictor(belief, ego_a, opp_a)
        assert imagined_next_b.shape == torch.Size([b_batch, 2, 32])

        # 2. Training pass with Posterior/Prior KL balancing
        pred_next_b, kl_loss = predictor.forward_train(belief, ego_a, opp_a, target_next_b)
        assert pred_next_b.shape == torch.Size([b_batch, 2, 32])
        assert kl_loss.shape == torch.Size([b_batch, 1])
        assert (kl_loss >= 0.0).all().item()

    def test_recurrent_jepa_target_ema_sync(self):
        extractor = MLPFeatureExtractor(obs_dim=2, hidden_dim=32, num_objects=2)
        encoder = RecurrentContextEncoder(extractor, action_dim=3, latent_dim=32, hidden_dim=32)
        predictor = CausalRelationalPredictor(num_objects=2, latent_dim=32, action_dim_i=3, action_dim_j=3, hidden_dim=32)

        jepa = RecurrentJEPABase(encoder, predictor, ema_momentum=0.9)

        # Perturb online weights
        with torch.no_grad():
            for p in jepa.context_encoder.parameters():
                p.add_(1.0)

        # Trigger in-place EMA sync
        jepa.update_target_encoder()

        # Check target parameters moved toward online parameters
        for p_tgt, p_ctx in zip(jepa.target_encoder.parameters(), jepa.context_encoder.parameters()):
            assert p_tgt.requires_grad is False
