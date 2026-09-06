# ABSOLUTE PATH: src/ipomdp/telemetry/system_monitor.py
# ==============================================================================
# HARDWARE & OPERATING SYSTEM TELEMETRY MONITOR (WSL2 & NVIDIA BLACKWELL)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Zero-Jitter Cached Polling:
#    - Hardware metric queries (via nvidia-smi / OS procfs) are cached with a minimum
#      refresh interval (default 1.0s) to prevent sub-process spawning or sysfs ioctls
#      from introducing latency bubbles into high-frequency MCTS step loops.
#
# 2. Dual-Layer GPU Memory Accounting:
#    - Measures both PyTorch CUDA allocator metrics (torch.cuda.memory_allocated,
#      memory_reserved) and physical device-level VRAM via nvidia-smi. This captures
#      the WSL2 D3D12/dxgkrnl paravirtualization context memory tax (~1-2GB on Blackwell).
#
# 3. Hybrid CPU Architecture & Process Footprint:
#    - Tracks process Resident Set Size (RSS), Virtual Memory Size (VMS), and CPU
#      utilization across Intel Core Ultra hybrid P-cores and E-cores using psutil.
#
# 4. Thermal & Power Degradation Detection:
#    - Evaluates GPU temperature against critical threshold (default 82.0°C) and power
#      draw (Watts) to detect dynamic SM clock throttling before performance collapses.
# ==============================================================================

import os
import shutil
import subprocess
import time
from typing import Dict, Any, Optional
import psutil
import torch


class SystemTelemetryMonitor:
    """
    Monitors host operating system, memory subsystems, and NVIDIA GPU telemetry.
    Designed for WSL2 Ubuntu on Windows 11 with NVIDIA RTX 50-series hardware.
    """

    def __init__(
        self,
        gpu_device_idx: int = 0,
        cache_interval_sec: float = 1.0,
        thermal_threshold_c: float = 82.0
    ):
        """
        Initializes system telemetry monitor.

        Args:
            gpu_device_idx: Index of GPU device to query (default 0).
            cache_interval_sec: Minimum seconds between hardware queries to avoid loop overhead.
            thermal_threshold_c: Temperature threshold in Celsius triggering thermal alerts.
        """
        self.gpu_device_idx = int(gpu_device_idx)
        self.cache_interval_sec = float(cache_interval_sec)
        self.thermal_threshold_c = float(thermal_threshold_c)

        self._process = psutil.Process(os.getpid())
        self._has_cuda = torch.cuda.is_available()
        self._has_nvismi = shutil.which("nvidia-smi") is not None

        self._last_poll_time = 0.0
        self._cached_metrics: Dict[str, float] = {}

    def get_metrics(self, force_refresh: bool = False) -> Dict[str, float]:
        """
        Retrieves current operating system, process, and GPU hardware metrics.
        Returns cached dictionary if queried within cache_interval_sec.

        Returns:
            Dictionary mapping metric keys to float values.
        """
        now = time.perf_counter()
        if not force_refresh and (now - self._last_poll_time) < self.cache_interval_sec and self._cached_metrics:
            return dict(self._cached_metrics)

        metrics: Dict[str, float] = {}

        # ---------------------------------------------------------------------
        # 1. Process & Host OS Memory Metrics (psutil)
        # ---------------------------------------------------------------------
        try:
            mem_info = self._process.memory_info()
            metrics["sys/process_rss_mb"] = float(mem_info.rss / (1024 * 1024))
            metrics["sys/process_vms_mb"] = float(mem_info.vms / (1024 * 1024))
            metrics["sys/process_cpu_pct"] = float(self._process.cpu_percent())
            metrics["sys/process_threads"] = float(self._process.num_threads())

            sys_mem = psutil.virtual_memory()
            metrics["sys/host_ram_used_mb"] = float((sys_mem.total - sys_mem.available) / (1024 * 1024))
            metrics["sys/host_ram_avail_mb"] = float(sys_mem.available / (1024 * 1024))
            metrics["sys/host_ram_pct"] = float(sys_mem.percent)

            swap = psutil.swap_memory()
            metrics["sys/host_swap_used_mb"] = float(swap.used / (1024 * 1024))
        except Exception:
            pass

        # ---------------------------------------------------------------------
        # 2. PyTorch CUDA Allocator Memory Metrics
        # ---------------------------------------------------------------------
        if self._has_cuda:
            try:
                allocated = torch.cuda.memory_allocated(self.gpu_device_idx) / (1024 * 1024)
                reserved = torch.cuda.memory_reserved(self.gpu_device_idx) / (1024 * 1024)
                max_allocated = torch.cuda.max_memory_allocated(self.gpu_device_idx) / (1024 * 1024)

                metrics["gpu/vram_allocated_mb"] = float(allocated)
                metrics["gpu/vram_reserved_mb"] = float(reserved)
                metrics["gpu/vram_max_allocated_mb"] = float(max_allocated)
                metrics["gpu/vram_fragmentation_pct"] = float(
                    ((reserved - allocated) / max(reserved, 1.0)) * 100.0 if reserved > 0 else 0.0
                )
            except Exception:
                pass

        # ---------------------------------------------------------------------
        # 3. Physical GPU Telemetry via nvidia-smi
        # ---------------------------------------------------------------------
        if self._has_cuda and self._has_nvismi:
            try:
                cmd = [
                    "nvidia-smi",
                    f"--id={self.gpu_device_idx}",
                    "--query-gpu=temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw",
                    "--format=csv,noheader,nounits"
                ]
                output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=1.0).strip()
                parts = [p.strip() for p in output.split(",")]
                if len(parts) >= 7:
                    metrics["gpu/temp_celsius"] = float(parts[0])
                    metrics["gpu/util_gpu_pct"] = float(parts[1])
                    metrics["gpu/util_mem_pct"] = float(parts[2])
                    metrics["gpu/physical_vram_total_mb"] = float(parts[3])
                    metrics["gpu/physical_vram_free_mb"] = float(parts[4])
                    metrics["gpu/physical_vram_used_mb"] = float(parts[5])
                    metrics["gpu/power_draw_watts"] = float(parts[6])
            except Exception:
                pass

        self._last_poll_time = now
        self._cached_metrics = metrics
        return dict(metrics)

    def is_thermally_throttled(self) -> bool:
        """Evaluates whether GPU temperature exceeds the critical thermal guardrail threshold."""
        metrics = self.get_metrics()
        temp = metrics.get("gpu/temp_celsius", 0.0)
        return temp >= self.thermal_threshold_c

    def get_progress_string(self) -> str:
        """
        Formats core system metrics into a compact, human-readable summary
        suitable for tqdm progress bar postfixes.
        """
        m = self.get_metrics()
        temp = m.get("gpu/temp_celsius", 0.0)
        pwr = m.get("gpu/power_draw_watts", 0.0)
        vram_used = m.get("gpu/physical_vram_used_mb", m.get("gpu/vram_reserved_mb", 0.0)) / 1024.0
        vram_total = m.get("gpu/physical_vram_total_mb", 16384.0) / 1024.0
        rss = m.get("sys/process_rss_mb", 0.0) / 1024.0
        cpu = m.get("sys/process_cpu_pct", 0.0)

        gpu_str = f"GPU: {temp:.0f}°C ({pwr:.0f}W)" if temp > 0 else "GPU: Active"
        vram_str = f"VRAM: {vram_used:.1f}/{vram_total:.1f}GB"
        host_str = f"RSS: {rss:.1f}GB | CPU: {cpu:.0f}%"

        return f"{gpu_str} | {vram_str} | {host_str}"
