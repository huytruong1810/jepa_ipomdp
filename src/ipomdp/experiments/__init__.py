"""
Experiments: building a run from its config, reloading and analysing a run directory, and
aggregating analyses over training seeds. The scripts main.py, analyze.py and sweep.py are thin
shells around this package.
"""

from .aggregate import PAIRED_GAPS, REPORT_METRICS, SeedStatistic, aggregate_reports, seed_statistic
from .runs import (DOMAIN_BUILDERS, AnalysisSettings, TrainedRun, analyze_run, build_run_config, build_training_run,
                   load_trained_run)

__all__ = [
    "DOMAIN_BUILDERS",
    "PAIRED_GAPS",
    "REPORT_METRICS",
    "AnalysisSettings",
    "SeedStatistic",
    "TrainedRun",
    "aggregate_reports",
    "analyze_run",
    "build_run_config",
    "build_training_run",
    "load_trained_run",
    "seed_statistic",
]
