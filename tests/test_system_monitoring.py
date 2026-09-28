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


class TestExecutionGuardrail:
    """Hardware and memory circuit breakers."""

    @staticmethod
    def _guardrail(metrics: dict, saved: list, **thresholds) -> ExecutionGuardrail:
        monitor = SystemTelemetryMonitor()
        monitor.get_metrics = lambda force_refresh=False: dict(metrics)
        limits = dict(thermal_trip_c=120.0, thermal_recovery_c=100.0, vram_trip_mb=100000.0, rss_trip_mb=100000.0)
        limits.update(thresholds)
        return ExecutionGuardrail(logging.getLogger("TestGuardrail"), monitor,
                                  emergency_save_fn=lambda: saved.append(True), **limits)

    def test_healthy_system_passes(self):
        saved = []
        guardrail = self._guardrail({"gpu/temp_celsius": 50.0, "sys/process_rss_mb": 500.0}, saved)
        assert guardrail.check_system_health(step_idx=1)
        assert not saved

    def test_emergency_save_on_persistent_vram_pressure(self):
        saved = []
        guardrail = self._guardrail({"gpu/vram_reserved_mb": 20000.0, "gpu/physical_vram_used_mb": 20000.0},
                                    saved, vram_trip_mb=10000.0)
        assert not guardrail.check_system_health(step_idx=50)
        assert saved and guardrail.incident_counts["vram_pressure_events"] == 1

    def test_emergency_save_on_rss_pressure(self):
        saved = []
        guardrail = self._guardrail({"sys/process_rss_mb": 30000.0}, saved, rss_trip_mb=18000.0)
        assert not guardrail.check_system_health(step_idx=3)
        assert saved and guardrail.incident_counts["rss_pressure_events"] == 1

    def test_failed_emergency_save_propagates(self):
        monitor = SystemTelemetryMonitor()
        monitor.get_metrics = lambda force_refresh=False: {"sys/process_rss_mb": 30000.0}

        def failing_save():
            raise OSError("disk full")

        guardrail = ExecutionGuardrail(logging.getLogger("TestGuardrail"), monitor, thermal_trip_c=120.0,
                                       thermal_recovery_c=100.0, vram_trip_mb=1e5, rss_trip_mb=18000.0,
                                       emergency_save_fn=failing_save)
        with pytest.raises(OSError):
            guardrail.check_system_health(step_idx=3)
