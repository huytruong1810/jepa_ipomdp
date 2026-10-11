# ABSOLUTE PATH: src/ipomdp/experiments/aggregate.py
# ==============================================================================
# SEED AGGREGATION: CONFIDENCE INTERVALS OVER INDEPENDENT TRAINING RUNS
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. The Seed Is the Unit of Replication:
#    - Every Phase 1-6 result came from seed 0. The per-run standard errors in report.json
#      measure only the evaluation noise of ONE trained agent; they say nothing about how much
#      a different training seed would change the result. A claim about the METHOD needs the
#      between-seed spread, so each seed contributes exactly one number per metric (its
#      analysis estimate), and the interval is over those numbers.
#
# 2. Student-t Intervals:
#    - With n seeds (typically 5-10) the 95% interval is mean +- t_{0.975, n-1} * s / sqrt(n),
#      s the sample standard deviation. The normal quantile (1.96) would understate the width
#      badly at small n (t_{0.975,4} = 2.78). Fewer than two seeds has no spread estimate and
#      raises.
#
# 3. Paired Gaps Within a Seed:
#    - In each run's analysis all return estimates share one simulator seed (common random
#      numbers, interpretability/analysis.py), so the per-seed DIFFERENCE learned_planner -
#      optimal has far less variance than either return. The gaps are therefore aggregated as
#      their own metrics instead of being derived from the two aggregated means.
#    - Every seed is analysed with the same analysis seed (runs.AnalysisSettings), i.e. on the
#      same simulator episodes. The optimal agent's return is therefore identical across seeds
#      (std 0), and the between-seed spread of any return is the spread due to TRAINING,
#      conditional on that fixed evaluation set. Per-seed evaluation noise (report.json
#      stderr) is not in the interval; the gaps, which cancel most of it, are the quantities to
#      cite.
#
# 4. The Selection Score Is Reported, Labelled:
#    - checkpoint_eval_return is the best greedy evaluation during training, i.e. the max over
#      noisy estimates, and is biased upwards (runs.py, section 3). It is aggregated for
#      reference only; the unbiased return of the same checkpoint is returns/learned_planner.
#
# 5. One Metric Table per Domain (2026-10-10):
#    - Tiger reports come from experiments/runs.py (exact optimum, probes in KL, error bounds);
#      Wumpus reports from experiments/wumpus.py (no optimum: the exact-belief reference agent at
#      risk 0.4 and the random hunter; outcome rates; cell/heading probes and hazard errors).
#    - DOMAIN_METRICS selects the table. The Wumpus gaps to cite are planner - reference and
#      planner - random, paired within seed for the same reason as in section 3: every seed's
#      analysis plays the same 512 layouts, so the reference's return is identical across seeds.
# ==============================================================================

from dataclasses import dataclass
import math

from scipy import stats

# Metric name -> path into a Tiger report.json (runs.py, section 4).
TIGER_METRICS: dict[str, tuple[str, ...]] = {
    "return/optimal": ("returns", "optimal", "mean"),
    "return/learned_planner": ("returns", "learned_planner", "mean"),
    "return/decoded_mlp": ("returns", "decoded_mlp", "mean"),
    "return/decoded_linear": ("returns", "decoded_linear", "mean"),
    "probe_kl/mlp_random": ("probes", "mlp/random", "mean_kl"),
    "probe_kl/linear_random": ("probes", "linear/random", "mean_kl"),
    "probe_kl/mlp_agent": ("probes", "mlp/agent", "mean_kl"),
    "probe_kl/linear_agent": ("probes", "linear/agent", "mean_kl"),
    "suboptimal_decisions/mlp_agent": ("bounds", "mlp/agent", "suboptimal_decision_rate"),
    "suboptimal_decisions/linear_agent": ("bounds", "linear/agent", "suboptimal_decision_rate"),
    "minimality_ratio": ("geometry", "minimality_ratio"),
    "selection_score (biased)": ("checkpoint_eval_return",),
}

# Paired within-seed differences (section 3): name -> (minuend, subtrahend) metric names.
TIGER_GAPS: dict[str, tuple[str, str]] = {
    "gap/learned_planner - optimal": ("return/learned_planner", "return/optimal"),
    "gap/decoded_mlp - optimal": ("return/decoded_mlp", "return/optimal"),
    "gap/learned_planner - decoded_mlp": ("return/learned_planner", "return/decoded_mlp"),
}

# Metric name -> path into a Wumpus report.json (experiments/wumpus.py; section 5).
WUMPUS_METRICS: dict[str, tuple[str, ...]] = {
    "return/learned_planner": ("returns", "learned_planner", "discounted_return"),
    "return/reference_risk0.4": ("returns", "reference_risk0.4", "discounted_return"),
    "return/random": ("returns", "random", "discounted_return"),
    "win_rate/learned_planner": ("returns", "learned_planner", "win_rate"),
    "death_rate/learned_planner": ("returns", "learned_planner", "death_rate"),
    "unfinished_rate/learned_planner": ("returns", "learned_planner", "unfinished_rate"),
    "probe_cell_acc/mlp_random": ("probes", "mlp_random", "cell_accuracy"),
    "probe_heading_acc/mlp_random": ("probes", "mlp_random", "heading_accuracy"),
    "probe_cell_acc/mlp_agent": ("probes", "mlp_agent", "cell_accuracy"),
    "probe_heading_acc/mlp_agent": ("probes", "mlp_agent", "heading_accuracy"),
    "probe_pit_error/mlp_random": ("probes", "mlp_random", "hazard_error", "pit"),
    "selection_score (biased)": ("checkpoint_eval_return",),
}

WUMPUS_GAPS: dict[str, tuple[str, str]] = {
    "gap/learned_planner - reference_risk0.4": ("return/learned_planner", "return/reference_risk0.4"),
    "gap/learned_planner - random": ("return/learned_planner", "return/random"),
}

# conf/env/<name>.yaml -> (metrics, paired gaps) (section 5).
DOMAIN_METRICS: dict[str, tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, str]]]] = {
    "tiger": (TIGER_METRICS, TIGER_GAPS),
    "wumpus": (WUMPUS_METRICS, WUMPUS_GAPS),
}


@dataclass(frozen=True)
class SeedStatistic:
    """Mean, sample standard deviation and 95% Student-t half-width over n seeds."""

    n: int
    mean: float
    std: float
    ci95: float
    values: list[float]


def seed_statistic(values: list[float]) -> SeedStatistic:
    """Section 2."""
    n = len(values)
    if n < 2:
        raise ValueError(f"A confidence interval over seeds needs at least 2 seeds, got {n}.")
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    return SeedStatistic(n, mean, std, float(stats.t.ppf(0.975, n - 1)) * std / math.sqrt(n), list(values))


def _lookup(report: dict, path: tuple[str, ...]) -> float:
    value = report
    for key in path:
        value = value[key]
    return float(value)


def aggregate_reports(reports: list[dict], domain: str) -> dict[str, SeedStatistic]:
    """Per-metric statistics over one report.json per seed of `domain`, including the paired gaps."""
    metrics, gaps = DOMAIN_METRICS[domain]
    per_seed = [{name: _lookup(report, path) for name, path in metrics.items()} for report in reports]
    for seed_metrics in per_seed:
        for name, (minuend, subtrahend) in gaps.items():
            seed_metrics[name] = seed_metrics[minuend] - seed_metrics[subtrahend]
    return {name: seed_statistic([m[name] for m in per_seed]) for name in per_seed[0]}
