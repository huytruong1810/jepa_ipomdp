# ABSOLUTE PATH: tests/test_telemetry.py
"""Unit tests for telemetry metrics and checkpoint serialisation (reviewed in Phases 5-6)."""

import logging
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from ipomdp.telemetry.checkpointer import ModelCheckpointer
from ipomdp.telemetry.metrics import compute_distribution_kl_divergence, compute_observation_accuracy_metrics


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
