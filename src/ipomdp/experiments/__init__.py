"""
Experiments: building a run from its config, reloading and analysing a run directory (Tiger with the
exact solver, Wumpus World without one), and aggregating analyses over training seeds. The scripts main.py, analyze.py and sweep.py are thin
shells around this package.
"""

from .aggregate import (DOMAIN_METRICS, TIGER_GAPS, TIGER_METRICS, WUMPUS_GAME_METRICS, WUMPUS_GAPS, WUMPUS_METRICS,
                        HierarchicalStatistic, SeedStatistic, aggregate_games, aggregate_reports, hierarchical_bootstrap,
                        interquartile_mean, seed_statistic)
from .runs import (DOMAIN_BUILDERS, AnalysisSettings, TrainedRun, analyze_run, build_run_config, build_training_run,
                   latent_figure, load_trained_run, tiger_setup, wumpus_setup)
from .wumpus import WumpusAnalysisSettings, analyze_wumpus_run

__all__ = [
    "DOMAIN_BUILDERS",
    "DOMAIN_METRICS",
    "TIGER_GAPS",
    "TIGER_METRICS",
    "WUMPUS_GAME_METRICS",
    "WUMPUS_GAPS",
    "WUMPUS_METRICS",
    "AnalysisSettings",
    "HierarchicalStatistic",
    "SeedStatistic",
    "TrainedRun",
    "WumpusAnalysisSettings",
    "aggregate_games",
    "aggregate_reports",
    "analyze_run",
    "analyze_wumpus_run",
    "build_run_config",
    "build_training_run",
    "hierarchical_bootstrap",
    "interquartile_mean",
    "latent_figure",
    "load_trained_run",
    "seed_statistic",
    "tiger_setup",
    "wumpus_setup",
]
