# ABSOLUTE PATH: tests/test_system_monitoring.py
# ==============================================================================
# RIGOROUS UNIT & INTEGRATION TESTS FOR SYSTEM MONITORING & GUARDRAILS
# ==============================================================================

import logging
import time
import pytest
import torch

from ipomdp.telemetry.system_monitor import SystemTelemetryMonitor
from ipomdp.telemetry.profiler import PipelineProfiler
from ipomdp.telemetry.guardrails import ExecutionGuardrail


class TestSystemTelemetryMonitor:
    """Validates host OS and GPU hardware metric polling."""

    def test_metrics_collection_and_schema(self):
        monitor = SystemTelemetryMonitor(cache_interval_sec=0.0)
        metrics = monitor.get_metrics(force_refresh=True)

        assert isinstance(metrics, dict)
        assert "sys/process_rss_mb" in metrics
        assert metrics["sys/process_rss_mb"] > 0.0
        assert "sys/process_cpu_pct" in metrics
        assert "sys/process_threads" in metrics
        assert metrics["sys/process_threads"] >= 1.0

        if torch.cuda.is_available():
            assert "gpu/vram_allocated_mb" in metrics
            assert "gpu/vram_reserved_mb" in metrics

    def test_caching_behavior(self):
        monitor = SystemTelemetryMonitor(cache_interval_sec=10.0)
        m1 = monitor.get_metrics(force_refresh=True)
        # Directly mutate cached metric to verify caching hit
        monitor._cached_metrics["sys/test_cached_marker"] = 999.0

        m2 = monitor.get_metrics(force_refresh=False)
        assert m2.get("sys/test_cached_marker") == 999.0

        m3 = monitor.get_metrics(force_refresh=True)
        assert "sys/test_cached_marker" not in m3

    def test_thermal_check(self):
        # Unreachable threshold should not trip
        monitor_cold = SystemTelemetryMonitor(thermal_threshold_c=200.0)
        assert not monitor_cold.is_thermally_throttled()

        # Artificially low threshold should trip if GPU temp is reported > 0
        monitor_hot = SystemTelemetryMonitor(thermal_threshold_c=-10.0)
        metrics = monitor_hot.get_metrics()
        if "gpu/temp_celsius" in metrics and metrics["gpu/temp_celsius"] > 0:
            assert monitor_hot.is_thermally_throttled()

    def test_progress_string_formatting(self):
        monitor = SystemTelemetryMonitor()
        progress_str = monitor.get_progress_string()
        assert isinstance(progress_str, str)
        assert "RSS:" in progress_str


class TestPipelineProfiler:
    """Validates wall-clock timing, percentiles, and throughput metrics."""

    def test_context_manager_timing(self):
        profiler = PipelineProfiler(window_size=10)

        with profiler.profile("test_stage"):
            time.sleep(0.01)  # 10 ms sleep

        stats = profiler.get_stage_stats("test_stage")
        assert "profiler/test_stage_mean_ms" in stats
        assert stats["profiler/test_stage_mean_ms"] >= 8.0  # Allow slight timing jitter
        assert "profiler/test_stage_p50_ms" in stats
        assert "profiler/test_stage_p99_ms" in stats

    def test_throughput_and_step_recording(self):
        profiler = PipelineProfiler()
        profiler.record_step(transitions=16, train_steps=1)
        profiler.record_step(transitions=16, train_steps=1)

        metrics = profiler.get_all_metrics()
        assert metrics["profiler/total_transitions"] == 32.0
        assert metrics["profiler/total_train_steps"] == 2.0
        assert metrics["profiler/overall_tps"] > 0.0

        progress_str = profiler.get_progress_string()
        assert "TPS:" in progress_str


class TestExecutionGuardrail:
    """Validates autonomous circuit breakers and health supervisors."""

    def test_healthy_system_passes(self):
        logger = logging.getLogger("TestGuardrail")
        guardrail = ExecutionGuardrail(
            logger=logger,
            thermal_trip_c=120.0,
            vram_trip_mb=100000.0,
            rss_trip_mb=100000.0
        )
        is_healthy = guardrail.check_system_health(step_idx=1)
        assert is_healthy

    def test_emergency_save_on_vram_tripwire(self):
        logger = logging.getLogger("TestGuardrail")
        saved_flag = [False]

        def mock_emergency_save():
            saved_flag[0] = True

        monitor = SystemTelemetryMonitor()
        # Mock monitor metrics returning extreme VRAM
        monitor.get_metrics = lambda force_refresh=False: {
            "gpu/vram_reserved_mb": 20000.0,
            "gpu/physical_vram_used_mb": 20000.0
        }

        guardrail = ExecutionGuardrail(
            logger=logger,
            system_monitor=monitor,
            vram_trip_mb=10000.0,
            emergency_save_fn=mock_emergency_save
        )

        is_healthy = guardrail.check_system_health(step_idx=50)
        assert not is_healthy
        assert saved_flag[0]
        assert guardrail.incident_counts["vram_pressure_events"] >= 1

    def test_latency_spike_detection(self):
        logger = logging.getLogger("TestGuardrail")
        profiler = PipelineProfiler(window_size=50)

        # Populate normal latencies (around 10ms)
        for _ in range(20):
            profiler._history.setdefault("mcts_search", []).append(10.0)

        # Add an extreme 100ms latency spike
        profiler._history["mcts_search"].append(100.0)

        guardrail = ExecutionGuardrail(
            logger=logger,
            profiler=profiler,
            latency_spike_multiplier=3.0
        )
        guardrail.check_step_latency(step_idx=10, stage_name="mcts_search")
        assert guardrail.incident_counts["latency_spike_events"] == 1

    def test_value_divergence_detection(self):
        logger = logging.getLogger("TestGuardrail")
        guardrail = ExecutionGuardrail(logger=logger)

        # Within bounds [-150, 150]
        guardrail.check_value_bounds(step_idx=1, predicted_value=5.0)
        assert guardrail.incident_counts["value_divergence_events"] == 0

        # Out of bounds
        guardrail.check_value_bounds(step_idx=2, predicted_value=-9999.0)
        assert guardrail.incident_counts["value_divergence_events"] == 1

