# ABSOLUTE PATH: src/ipomdp/telemetry/guardrails.py
# ==============================================================================
# AUTONOMOUS PERFORMANCE GUARDRAILS & EXECUTION CIRCUIT BREAKERS
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Multi-Tier Circuit Breakers:
#    - Monitors system health (Thermal, VRAM, RSS) and algorithmic integrity (latency
#      spikes, value divergence) in real-time.
#    - Automatically triggers corrective actions (cooldown sleeps, CUDA cache clearing,
#      or emergency checkpoint dumps) before catastrophic process death or silent drift.
#
# 2. Thermal Protection Governor:
#    - Intercepts GPU core temperatures approaching thermal limit (default 82.0°C).
#      Pauses execution in an adaptive sleep cycle until temperatures decline below
#      safe recovery threshold (default 72.0°C), protecting hardware longevity.
#
# 3. Virtualization VRAM Safety Margin:
#    - Monitors physical and allocated VRAM against safe headroom bound (default 13.5GB
#      on 16GB RTX 5080). Accounts for WSL2 D3D12 driver context overhead to prevent
#      triggering WSL2 kernel OOM reaping.
#
# 4. Scope:
#    - Hardware and memory protection only. Latency "spike" alerts (noise under a shared CPU)
#      and a hard-coded value-divergence bound (superseded by the two-hot codec, which raises
#      on any target outside the domain's exact value bound) were removed in Phase 5.
#    - A failing emergency save raises: it runs exactly when the process is in danger, so a
#      swallowed failure would lose the run silently.
# ==============================================================================

import logging
import time
from typing import Callable, Dict
import torch

from .system_monitor import SystemTelemetryMonitor


class ExecutionGuardrail:
    """
    Autonomous supervisor maintaining execution safety, hardware protection,
    and performance guardrails across long-running training loops.
    """

    def __init__(
        self,
        logger: logging.Logger,
        system_monitor: SystemTelemetryMonitor,
        thermal_trip_c: float,
        thermal_recovery_c: float,
        vram_trip_mb: float,
        rss_trip_mb: float,
        emergency_save_fn: Callable[[], None],
    ):
        """
        Initializes Execution Guardrail.

        Args:
            logger: Logging instance for structured alert dispatch.
            system_monitor: Hardware/OS metrics monitor instance.
            thermal_trip_c: GPU temperature in Celsius triggering thermal backoff.
            thermal_recovery_c: GPU temperature in Celsius allowing execution resumption.
            vram_trip_mb: VRAM usage in MB triggering memory compaction or emergency dump.
            rss_trip_mb: Process RSS memory in MB triggering memory leak tripwire.
            emergency_save_fn: Callback function to serialize weights on critical failure.
        """
        self.logger = logger
        self.system_monitor = system_monitor
        self.thermal_trip_c = float(thermal_trip_c)
        self.thermal_recovery_c = float(thermal_recovery_c)
        self.vram_trip_mb = float(vram_trip_mb)
        self.rss_trip_mb = float(rss_trip_mb)
        self.emergency_save_fn = emergency_save_fn

        self.incident_counts: Dict[str, int] = {
            "thermal_throttle_events": 0,
            "vram_pressure_events": 0,
            "rss_pressure_events": 0,
        }

    def check_system_health(self, step_idx: int) -> bool:
        """
        Evaluates hardware and OS health metrics.
        Returns True if system is healthy; False if emergency intervention was executed.
        """
        metrics = self.system_monitor.get_metrics()

        # ---------------------------------------------------------------------
        # 1. Thermal Governor Check
        # ---------------------------------------------------------------------
        temp = metrics.get("gpu/temp_celsius", 0.0)
        if temp >= self.thermal_trip_c:
            self.incident_counts["thermal_throttle_events"] += 1
            self.logger.warning(
                f"[GUARDRAIL ALERT] GPU Temperature ({temp:.1f}°C) exceeded thermal threshold "
                f"({self.thermal_trip_c:.1f}°C) at step {step_idx}. Initiating thermal cooldown cycle..."
            )
            self._execute_thermal_cooldown()

        # ---------------------------------------------------------------------
        # 2. VRAM Headroom & Memory Leak Check
        # ---------------------------------------------------------------------
        vram_reserved = metrics.get("gpu/vram_reserved_mb", 0.0)
        vram_physical = metrics.get("gpu/physical_vram_used_mb", 0.0)
        curr_vram = max(vram_reserved, vram_physical)

        if curr_vram >= self.vram_trip_mb:
            self.incident_counts["vram_pressure_events"] += 1
            self.logger.warning(
                f"[GUARDRAIL ALERT] VRAM usage ({curr_vram:.0f} MB) exceeded safety threshold "
                f"({self.vram_trip_mb:.0f} MB) at step {step_idx}. Triggering CUDA cache compaction..."
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Re-check after cache flush
            post_metrics = self.system_monitor.get_metrics(force_refresh=True)
            post_vram = max(post_metrics.get("gpu/vram_reserved_mb", 0.0), post_metrics.get("gpu/physical_vram_used_mb", 0.0))
            if post_vram >= self.vram_trip_mb:
                self.logger.critical(
                    f"[CRITICAL GUARDRAIL] Persistent VRAM pressure ({post_vram:.0f} MB) detected! "
                    "Triggering emergency checkpoint serialization..."
                )
                self._trigger_emergency_save()
                return False

        # ---------------------------------------------------------------------
        # 3. Process RSS Memory Leak Check
        # ---------------------------------------------------------------------
        rss = metrics.get("sys/process_rss_mb", 0.0)
        if rss >= self.rss_trip_mb:
            self.incident_counts["rss_pressure_events"] += 1
            self.logger.critical(
                f"[CRITICAL GUARDRAIL] Process RSS ({rss:.0f} MB) exceeded memory budget "
                f"({self.rss_trip_mb:.0f} MB) at step {step_idx}! Memory leak suspected."
            )
            self._trigger_emergency_save()
            return False

        return True

    def _execute_thermal_cooldown(self, poll_interval_sec: float = 2.0, max_wait_sec: float = 60.0):
        """Pauses execution in adaptive backoff loop until GPU temperature subsides."""
        start_wait = time.perf_counter()
        while time.perf_counter() - start_wait < max_wait_sec:
            time.sleep(poll_interval_sec)
            m = self.system_monitor.get_metrics(force_refresh=True)
            temp = m.get("gpu/temp_celsius", 0.0)
            if temp <= self.thermal_recovery_c:
                self.logger.info(
                    f"[THERMAL RECOVERY] GPU temperature cooled to {temp:.1f}°C (<= {self.thermal_recovery_c:.1f}°C). "
                    "Resuming active execution."
                )
                return
        self.logger.warning("[THERMAL TIMEOUT] Maximum cooldown time elapsed; resuming execution.")

    def _trigger_emergency_save(self):
        """Executes the emergency save callback (failures propagate, see module header)."""
        self.emergency_save_fn()
        self.logger.info("[GUARDRAIL] Emergency checkpoint successfully persisted.")

    def get_incident_metrics(self) -> Dict[str, float]:
        """Returns total counts of tripped guardrails for telemetry logging."""
        return {f"guardrail/{k}": float(v) for k, v in self.incident_counts.items()}
