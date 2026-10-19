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
#
# 6. Robust and Hierarchical Statistics (2026-10-19; reports/2026-10-19_eval_protocol/PROTOCOL.md):
#    - The Student-t interval assumes roughly normal seed values; with 5-10 seeds one bad seed
#      dominates it. Every SeedStatistic therefore also carries the interquartile mean over seeds
#      (scipy trim_mean with 25% cut from each end: at n = 5 the mean of the middle 3) and a
#      percentile-bootstrap 95% interval over seeds, the robust aggregate recommended by Agarwal
#      et al. (2021, "Deep RL at the edge of the statistical precipice", rliable). Implemented here
#      (about ten lines) rather than adding rliable as a dependency.
#    - The IQM is taken over SEED-level values, never over games: a Wumpus game pays about +900
#      (win), -40 (unfinished) or -1000 (death), so a game-level IQM would cut away the wins and
#      deaths, which are the outcomes that matter.
#    - The Student-t and IQM intervals condition on the evaluation games (section 3). The
#      hierarchical bootstrap (Wumpus, from each seed's games.npz) resamples seeds and then games
#      within each resampled seed, so its interval holds BOTH training and evaluation noise; for
#      paired gaps it resamples per-game differences (every agent of a seed plays the same
#      layouts, experiments/wumpus.py section 2), which keeps the pairing.
#    - The bootstrap streams are fixed (BOOTSTRAP_SEED) so an aggregate is reproducible; 10,000
#      resamples put the Monte-Carlo error of a 95% bound well below the reported precision.
#    - The exact-belief planner (planning/wumpus_exact_model.py) is a reference played ONCE per
#      analysis seed (reports/2026-10-19_eval_protocol/exact_planner_play.py), not per training
#      seed. aggregate_games(..., exact_planner=rows) pairs its per-game rows with every seed's
#      games.npz and adds EXACT_PLANNER_GAME_METRICS. In the hierarchical bootstrap its return
#      is the same row for every resampled seed, so its interval holds evaluation noise only,
#      and the paired gap holds both. Pairing is verified, not assumed: the planner file carries
#      the heuristic's per-game returns on the same rows, which must equal every seed's
#      reference_risk0.4.return exactly (the heuristic is deterministic given the layout).
#    - The model-free baseline (baselines/recurrent_dqn.py; PROTOCOL.md C-c) is trained per
#      training seed: aggregate_games(..., baseline=[rows per seed]) merges seed k's baseline rows
#      into seed k's games (same pairing check) and adds BASELINE_GAME_METRICS.
# ==============================================================================

from dataclasses import dataclass
import math

import numpy as np
from scipy import stats

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261019

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


# Wumpus per-game metrics (section 6): name -> function of one seed's games.npz rows -> (games,) float64.
WUMPUS_GAME_METRICS = {
    "return/learned_planner": lambda g: g["learned_planner.return"],
    "return/reference_risk0.4": lambda g: g["reference_risk0.4.return"],
    "return/random": lambda g: g["random.return"],
    "win_rate/learned_planner": lambda g: g["learned_planner.won"].astype(np.float64),
    "death_rate/learned_planner": lambda g: g["learned_planner.died"].astype(np.float64),
    "unfinished_rate/learned_planner": lambda g: (~g["learned_planner.won"] & ~g["learned_planner.died"]).astype(np.float64),
    "gap/learned_planner - reference_risk0.4": lambda g: g["learned_planner.return"] - g["reference_risk0.4.return"],
    "gap/learned_planner - random": lambda g: g["learned_planner.return"] - g["random.return"],
}


@dataclass(frozen=True)
class SeedStatistic:
    """
    Over n seeds: mean, sample standard deviation and 95% Student-t half-width (section 2), and the
    interquartile mean with its percentile-bootstrap 95% interval (section 6).
    """

    n: int
    mean: float
    std: float
    ci95: float
    iqm: float
    iqm_ci95: tuple[float, float]
    values: list[float]


@dataclass(frozen=True)
class HierarchicalStatistic:
    """Mean over seeds x games with a 95% interval from resampling seeds, then games (section 6)."""

    seeds: int
    games: int
    mean: float
    ci95: tuple[float, float]


def interquartile_mean(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Mean of the middle 50% along `axis` (scipy trim_mean, 25% cut from each end; section 6)."""
    return stats.trim_mean(values, 0.25, axis=axis)


def seed_statistic(values: list[float]) -> SeedStatistic:
    """Sections 2 and 6."""
    n = len(values)
    if n < 2:
        raise ValueError(f"A confidence interval over seeds needs at least 2 seeds, got {n}.")
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    resampled = interquartile_mean(array[rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))])
    low, high = np.quantile(resampled, [0.025, 0.975])
    return SeedStatistic(n, mean, std, float(stats.t.ppf(0.975, n - 1)) * std / math.sqrt(n),
                         float(interquartile_mean(array)), (float(low), float(high)), list(values))


def hierarchical_bootstrap(per_seed: np.ndarray, chunk: int = 500) -> HierarchicalStatistic:
    """
    Section 6: per_seed (seeds, games) per-game values (or paired per-game differences). Each
    resample draws seeds with replacement, then games with replacement within every drawn seed;
    the statistic is the grand mean. Chunked so the index tensor stays ~10 MB.
    """
    seeds, games = per_seed.shape
    if seeds < 2:
        raise ValueError(f"A hierarchical bootstrap over seeds needs at least 2 seeds, got {seeds}.")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = []
    for start in range(0, BOOTSTRAP_RESAMPLES, chunk):
        size = min(chunk, BOOTSTRAP_RESAMPLES - start)
        seed_draw = rng.integers(0, seeds, size=(size, seeds))
        game_draw = rng.integers(0, games, size=(size, seeds, games))
        means.append(per_seed[seed_draw[:, :, None], game_draw].mean(axis=(1, 2)))
    low, high = np.quantile(np.concatenate(means), [0.025, 0.975])
    return HierarchicalStatistic(seeds, games, float(per_seed.mean()), (float(low), float(high)))


# Added when the exact-planner reference rows are given (section 6); "exact_planner.*" come from its games.npz.
EXACT_PLANNER_GAME_METRICS = {
    "return/exact_planner": lambda g: g["exact_planner.return"],
    "win_rate/exact_planner": lambda g: g["exact_planner.won"].astype(np.float64),
    "death_rate/exact_planner": lambda g: g["exact_planner.died"].astype(np.float64),
    "unfinished_rate/exact_planner": lambda g: (~g["exact_planner.won"] & ~g["exact_planner.died"]).astype(np.float64),
    "gap/learned_planner - exact_planner": lambda g: g["learned_planner.return"] - g["exact_planner.return"],
    "gap/exact_planner - reference_risk0.4": lambda g: g["exact_planner.return"] - g["reference_risk0.4.return"],
}


# Added when baseline rows are given per seed (section 6); "recurrent_dqn.*" come from each seed's baseline games.npz.
BASELINE_GAME_METRICS = {
    "return/recurrent_dqn": lambda g: g["recurrent_dqn.return"],
    "win_rate/recurrent_dqn": lambda g: g["recurrent_dqn.won"].astype(np.float64),
    "death_rate/recurrent_dqn": lambda g: g["recurrent_dqn.died"].astype(np.float64),
    "unfinished_rate/recurrent_dqn": lambda g: (~g["recurrent_dqn.won"] & ~g["recurrent_dqn.died"]).astype(np.float64),
    "gap/learned_planner - recurrent_dqn": lambda g: g["learned_planner.return"] - g["recurrent_dqn.return"],
}


def _check_pairing(seed_games: dict[str, np.ndarray], other: dict[str, np.ndarray], what: str) -> None:
    if not np.array_equal(seed_games["reference_risk0.4.return"], other["reference_risk0.4.return"]):
        raise ValueError(f"{what} was not played on the same games: the heuristic's per-game returns differ "
                         "(another analysis seed or game count).")


def aggregate_games(games: list[dict[str, np.ndarray]],
                    exact_planner: dict[str, np.ndarray] | None = None,
                    baseline: list[dict[str, np.ndarray]] | None = None) -> dict[str, HierarchicalStatistic]:
    """
    Section 6: one games.npz (as a dict of arrays) per seed of a Wumpus sweep; optionally the
    exact planner's games.npz of the same analysis seed and one baseline games.npz per seed (in
    the sweep's seed order). Both are paired game by game, which is checked here.
    """
    lengths = {len(seed_games["learned_planner.return"]) for seed_games in games}
    if len(lengths) != 1:
        raise ValueError(f"Every seed must be analysed on the same number of games, got {sorted(lengths)}.")
    metrics = dict(WUMPUS_GAME_METRICS)
    if exact_planner is not None:
        for index, seed_games in enumerate(games):
            _check_pairing(seed_games, exact_planner, f"Seed entry {index} vs the exact planner")
        games = [{**seed_games, **{k: v for k, v in exact_planner.items() if k.startswith("exact_planner.")}}
                 for seed_games in games]
        metrics.update(EXACT_PLANNER_GAME_METRICS)
    if baseline is not None:
        if len(baseline) != len(games):
            raise ValueError(f"{len(baseline)} baseline seeds for {len(games)} sweep seeds.")
        for index, (seed_games, rows) in enumerate(zip(games, baseline)):
            _check_pairing(seed_games, rows, f"Seed entry {index} vs its baseline")
        games = [{**seed_games, **{k: v for k, v in rows.items() if k.startswith("recurrent_dqn.")}}
                 for seed_games, rows in zip(games, baseline)]
        metrics.update(BASELINE_GAME_METRICS)
    return {name: hierarchical_bootstrap(np.stack([metric(seed_games) for seed_games in games]))
            for name, metric in metrics.items()}


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
