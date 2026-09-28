# ABSOLUTE PATH: src/ipomdp/telemetry/metrics.py
# ==============================================================================
# TENSORBOARD METRIC WRITER
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Scope:
#    - A thin TensorBoard writer for scalar metrics and figures. The KL / accuracy helpers that
#      used to live here were superseded by ipomdp.interpretability (exact-posterior probes);
#      they also silently softmaxed any input that did not sum to one.
# ==============================================================================

from pathlib import Path
from typing import Dict, Union
from torch.utils.tensorboard import SummaryWriter


class MetricsLogger:
    """Domain-agnostic time-series metric tracker using TensorBoard."""

    def __init__(self, log_dir: str):
        """Writes TensorBoard events to `log_dir` (the run directory already carries the timestamp)."""
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_metrics(self, metrics_dict: Dict[str, Union[float, int]], step: int, prefix: str = ""):
        """Logs a dictionary of scalar metrics with optional UI category prefix."""
        for key, value in metrics_dict.items():
            if value is None:
                continue
            tag = f"{prefix}/{key}" if prefix else key
            self.writer.add_scalar(tag, float(value), int(step))

    def log_figure(self, tag: str, figure, step: int):
        """Logs a matplotlib figure directly into TensorBoard."""
        self.writer.add_figure(tag, figure, global_step=int(step))

    def close(self):
        """Closes SummaryWriter handle."""
        self.writer.close()
