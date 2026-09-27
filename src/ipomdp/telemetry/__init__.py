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
    RewardTrajectoryVisualizer,
)
from .checkpointer import ModelCheckpointer
from .system_monitor import SystemTelemetryMonitor
from .profiler import PipelineProfiler
from .guardrails import ExecutionGuardrail

__all__ = [
    "JSONFormatter",
    "setup_logger",
    "log_telemetry",
    "MetricsLogger",
    "compute_distribution_kl_divergence",
    "compute_observation_accuracy_metrics",
    "LatentSpaceVisualizer",
    "MCTSGraphVisualizer",
    "RewardTrajectoryVisualizer",
    "ModelCheckpointer",
    "SystemTelemetryMonitor",
    "PipelineProfiler",
    "ExecutionGuardrail",
]

