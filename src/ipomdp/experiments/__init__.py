"""
Experiments: building a run from its config, reloading and analysing a run directory (Tiger with the
exact solver, Wumpus World without one), and aggregating analyses over training seeds. The scripts main.py, analyze.py and sweep.py are thin
shells around this package.
"""

from .aggregate import PAIRED_GAPS, REPORT_METRICS, SeedStatistic, aggregate_reports, seed_statistic
from .runs import (DOMAIN_BUILDERS, AnalysisSettings, TrainedRun, analyze_run, build_run_config, build_training_run,
                   latent_figure, load_trained_run, tiger_setup, wumpus_setup)
from .wumpus import WumpusAnalysisSettings, analyze_wumpus_run

__all__ = [
    "DOMAIN_BUILDERS",
    "PAIRED_GAPS",
    "REPORT_METRICS",
    "AnalysisSettings",
    "SeedStatistic",
    "TrainedRun",
    "WumpusAnalysisSettings",
    "aggregate_reports",
    "analyze_run",
    "analyze_wumpus_run",
    "build_run_config",
    "build_training_run",
    "latent_figure",
    "load_trained_run",
    "seed_statistic",
    "tiger_setup",
    "wumpus_setup",
]
