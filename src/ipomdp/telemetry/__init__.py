# ABSOLUTE PATH: src/ipomdp/telemetry/__init__.py
"""Telemetry, structured logging, high-precision metrics, visualizers, and checkpointing for JEPA-IPOMDP."""

from .logger import JSONFormatter, setup_logger, log_telemetry
from .metrics import (
    MetricsLogger,
    compute_distribution_kl_divergence,
    compute_observation_accuracy_metrics,
)
from .visualizers import (
    LatentSpaceVisualizer,
    MCTSGraphVisualizer,
    JEPASemanticsProbe,
    RewardTrajectoryVisualizer,
)
from .checkpointer import ModelCheckpointer
from .registry import register_env, make_env, register_extractor, make_extractor

__all__ = [
    "JSONFormatter",
    "setup_logger",
    "log_telemetry",
    "MetricsLogger",
    "compute_distribution_kl_divergence",
    "compute_observation_accuracy_metrics",
    "LatentSpaceVisualizer",
    "MCTSGraphVisualizer",
    "JEPASemanticsProbe",
    "RewardTrajectoryVisualizer",
    "ModelCheckpointer",
    "register_env",
    "make_env",
    "register_extractor",
    "make_extractor",
]
