# ABSOLUTE PATH: tests/test_core_utils.py
"""Unit tests for core data structures, utilities, segment trees, and replay buffer."""

import pytest
import numpy as np
import torch
import torch.nn as nn
import tempfile
import logging
from pathlib import Path

from ipomdp.types import Action
from ipomdp.training.replay_buffer import SumTree, PrioritizedSequenceBuffer
from ipomdp.models.distributions import symlog, symexp, TwoHotSymlog
from ipomdp.models.layers import RMSNorm, SwiGLUResidualBlock, build_residual_stack
from ipomdp.telemetry.metrics import compute_distribution_kl_divergence, compute_observation_accuracy_metrics
from ipomdp.telemetry.checkpointer import ModelCheckpointer



class TestDataTypes:
    """Rank-safe one-hot action encoding."""

    def test_action_to_one_hot_ranks(self):
        # 1. Scalar int
        oh_scalar = Action.to_one_hot(1, num_classes=3)
        assert oh_scalar.shape == torch.Size([1, 3])
        assert torch.equal(oh_scalar, torch.tensor([[0.0, 1.0, 0.0]]))

        # 2. 1D tensor batch (B,)
        t_1d = torch.tensor([0, 2, 1])
        oh_1d = Action.to_one_hot(t_1d, num_classes=3)
        assert oh_1d.shape == torch.Size([3, 3])
        assert torch.equal(oh_1d, torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0]
        ]))

        # 3. 2D tensor singleton (B, 1)
        t_2d = torch.tensor([[0], [2]])
        oh_2d = Action.to_one_hot(t_2d, num_classes=3)
        assert oh_2d.shape == torch.Size([2, 3])

        # 4. 3D temporal tensor (B, T, 1)
        t_3d = torch.randint(0, 3, (2, 5, 1))
        oh_3d = Action.to_one_hot(t_3d, num_classes=3)
        assert oh_3d.shape == torch.Size([2, 5, 3])


class TestSumTree:
    """Rigorous double-precision segment tree verification."""

    def test_zero_drift_and_min_tracking(self):
        capacity = 128
        tree = SumTree(capacity)

        priorities = np.random.uniform(0.1, 5.0, size=capacity)
        for idx, p in enumerate(priorities):
            tree.add(p, idx)

        assert tree.size == capacity
        assert pytest.approx(tree.total_priority, rel=1e-7) == float(priorities.sum())
        assert pytest.approx(tree.get_min_priority(), rel=1e-7) == float(priorities.min())

        # Perform 1000 random leaf priority updates
        for _ in range(1000):
            target_leaf_idx = np.random.randint(0, capacity)
            new_p = float(np.random.uniform(0.01, 10.0))
            priorities[target_leaf_idx] = new_p

            global_tree_idx = target_leaf_idx + tree.tree_capacity - 1
            tree.update(global_tree_idx, new_p)

        # Zero-drift validation
        assert pytest.approx(tree.total_priority, rel=1e-7) == float(priorities.sum())
        assert pytest.approx(tree.get_min_priority(), rel=1e-7) == float(priorities.min())

    def test_vectorized_leaf_search(self):
        capacity = 64
        tree = SumTree(capacity)
        for i in range(capacity):
            tree.add(1.0, i)

        batch_vals = np.array([0.5, 10.5, 30.2, 63.8])
        leaf_indices, priorities, payloads = tree.get_leaf_batch(batch_vals)

        assert len(leaf_indices) == 4
        assert len(priorities) == 4
        assert len(payloads) == 4
        assert payloads[0] == 0
        assert payloads[1] == 10
        assert payloads[2] == 30
        assert payloads[3] == 63


class TestSymlogAndTwoHot:
    """Rigorous DreamerV3 TwoHotSymlog tests."""

    def test_symlog_symexp_invertibility(self):
        x = torch.tensor([-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0])
        compressed = symlog(x)
        recovered = symexp(compressed)
        assert torch.allclose(x, recovered, atol=1e-4)

    def test_twohot_symlog_loss_and_decoding(self):
        twohot = TwoHotSymlog(min_val=-20.0, max_val=20.0, num_bins=255)
        targets = torch.tensor([[-100.0], [0.0], [10.0], [50.0]])

        # Initialize trainable logits
        logits = nn.Parameter(torch.zeros(4, 255))
        optimizer = torch.optim.Adam([logits], lr=0.1)

        # Train logits on targets for 200 steps
        for _ in range(200):
            optimizer.zero_grad()
            loss = twohot(logits, targets, auto_symlog=True).mean()
            loss.backward()
            optimizer.step()

        # Decoded expectations should closely approximate targets in symlog space and physical space
        decoded_symlog = twohot.decode(logits, real_scale=False)
        assert torch.allclose(decoded_symlog, symlog(targets), atol=0.05)

        decoded_real = twohot.decode(logits, real_scale=True)
        assert torch.allclose(decoded_real, targets, atol=5.0)


class TestGatedResidualBuilder:
    """Tests for SOTA SwiGLU Gated ResMLP stack."""

    def test_identity_initialization(self):
        block = SwiGLUResidualBlock(dim=32)
        x = torch.randn(4, 32)
        # Because w3 is zero-initialized, block(x) must exactly equal x at start
        out = block(x)
        assert torch.allclose(out, x, atol=1e-6)

    def test_residual_stack_forward(self):
        net = build_residual_stack(input_dim=16, hidden_dim=32, output_dim=8, num_blocks=2)
        x = torch.randn(5, 16)
        out = net(x)
        assert out.shape == torch.Size([5, 8])


class TestPrioritizedSequenceBuffer:
    """Rigorous tests for contiguous Seq-PER replay buffer."""

    def test_sequence_push_unrolling_and_sampling(self):
        buffer = PrioritizedSequenceBuffer(capacity=64, burn_in=2, seq_len=4)
        obs_dim = 2
        act_dim_i = 3
        act_dim_j = 3

        # Simulate 2 complete episodes of length 6
        for ep in range(2):
            for t in range(6):
                obs = torch.randn(obs_dim)
                act_i = torch.tensor([t % act_dim_i])
                act_j = torch.tensor([0])
                rew = 1.0 if t == 5 else -0.1
                buffer.push(env_idx=0, obs=obs, act_i=act_i, act_j=act_j, reward=rew)
            buffer.end_episode(env_idx=0, final_obs=torch.randn(obs_dim))

        assert buffer.tree.size > 0

        batch_size = 4
        batch, is_weights, tree_indices = buffer.sample_sequence(batch_size)

        chunk_len = buffer.burn_in + buffer.seq_len  # 2 + 4 = 6
        assert batch["obs"].shape == torch.Size([batch_size, chunk_len + 1, obs_dim])
        assert batch["act_i"].shape == torch.Size([batch_size, chunk_len, 1])
        assert batch["act_j"].shape == torch.Size([batch_size, chunk_len, 1])
        assert batch["rewards"].shape == torch.Size([batch_size, chunk_len, 1])
        assert batch["mask"].shape == torch.Size([batch_size, chunk_len, 1])
        assert is_weights.shape == torch.Size([batch_size, 1])
        assert len(tree_indices) == batch_size

        # Update priorities with dummy TD errors
        dummy_td_errors = torch.tensor([0.5, 1.2, 0.1, 0.8])
        buffer.update_priorities(tree_indices, dummy_td_errors)


class TestMetricsAndCheckpointer:
    """Tests for KL metrics, accuracy metrics, and checkpointer serialization."""

    def test_kl_divergence_metrics(self):
        p1 = torch.tensor([[0.5, 0.5], [0.9, 0.1]])
        p2 = torch.tensor([[0.5, 0.5], [0.1, 0.9]])

        kl_sample, metrics = compute_distribution_kl_divergence(p1, p1)
        assert pytest.approx(metrics["kl_mean"], abs=1e-6) == 0.0

        kl_sample, metrics = compute_distribution_kl_divergence(p1, p2)
        assert metrics["kl_mean"] > 0.0

    def test_obs_accuracy_metrics(self):
        p_oracle = torch.tensor([[0.8, 0.2]])
        p_jepa = torch.tensor([[0.7, 0.3]])
        target = torch.tensor([0])

        metrics = compute_observation_accuracy_metrics(p_oracle, p_jepa, target)
        assert metrics["acc_oracle"] == 1.0
        assert metrics["acc_jepa"] == 1.0
        assert metrics["efficiency_ratio"] == 100.0

    def test_checkpointer_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = logging.getLogger("test_logger")
            checkpointer = ModelCheckpointer(tmpdir, logger)

            model = nn.Linear(4, 2)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            checkpointer.save(step=10, models={"linear": model}, optimizer=optimizer, current_loss=0.42)
            assert (Path(tmpdir) / "latest_checkpoint.pt").exists()
            assert (Path(tmpdir) / "best_model.pt").exists()

            new_model = nn.Linear(4, 2)
            new_opt = torch.optim.Adam(new_model.parameters(), lr=1e-3)

            step = checkpointer.load(str(Path(tmpdir) / "latest_checkpoint.pt"), {"linear": new_model}, new_opt)
            assert step == 10
            assert torch.allclose(model.weight, new_model.weight)
