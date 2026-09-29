# ABSOLUTE PATH: analyze.py
# ==============================================================================
# BELIEF ANALYSIS OF A TRAINED RUN
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Input Is a Run Directory:
#    - `uv run analyze.py runs/tiger/<run>` reloads the run from its resolved Hydra config and
#      its best.pt and runs the interpretability analysis (ipomdp.experiments.runs.analyze_run,
#      which documents the outputs under <run>/analysis/).
#
# 2. Exact References:
#    - V* and the per-action Q* vector sets are computed by the certified solver; the certified
#      error is written into the report and enters the error bounds as documented in
#      interpretability/error_bounds.py.
#
# 3. Thin Script:
#    - Argument parsing and the console summary only. sweep.py calls the same analyze_run for
#      every seed, so a single run and a sweep are analysed identically.
# ==============================================================================

import argparse
from pathlib import Path

import torch

from ipomdp.experiments import AnalysisSettings, analyze_run
from ipomdp.interpretability import BeliefAnalysis


def print_summary(analysis: BeliefAnalysis) -> None:
    print(f"V*(b0) = {analysis.optimal_value_b0:.3f} (certified error {analysis.solver_error_bound:.3g})")
    for name, estimate in analysis.returns.items():
        print(f"  {name:<16} discounted return {estimate.mean:8.3f} +- {estimate.stderr:.3f}")
    for name, probe in analysis.probes.items():
        print(f"  probe {name:<14} KL mean {probe.mean_kl:.5f}  L1 mean {probe.mean_l1:.4f}  L1 max {probe.max_l1:.4f}")
    for name, bound in analysis.bounds.items():
        print(f"  bounds {name:<12} L_V {bound.lipschitz_value:.1f} L_Q {bound.lipschitz_q:.1f} | "
              f"worst case: value error {bound.value_error_max:.3f} <= {bound.value_error_bound_max:.3f}, "
              f"regret {bound.regret_max:.3f} <= {bound.regret_bound_max:.3f} | "
              f"expected: value error {bound.value_error_mean:.3f} <= {bound.value_error_bound_mean:.3f}, "
              f"regret {bound.regret_mean:.4f} <= {bound.regret_bound_mean:.4f} | "
              f"suboptimal decisions {bound.suboptimal_decision_rate:.4f}")
    geometry = analysis.geometry
    print(f"  geometry: explained variance {[round(r, 3) for r in geometry.explained_variance_ratio]}; "
          f"minimality ratio {geometry.minimality_ratio:.3f}")
    if geometry.log_odds_spearman is not None:
        print(f"    |Spearman(PC1..3, log-odds)| {[round(r, 3) for r in geometry.log_odds_spearman]}, "
              f"|Spearman(PC1..3, |log-odds|)| {[round(r, 3) for r in geometry.confidence_spearman]}")


def main() -> None:
    defaults = AnalysisSettings()
    parser = argparse.ArgumentParser(description="Belief analysis of a trained run (see module header).")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--episodes", type=int, default=defaults.episodes,
                        help="Episodes per data set and return estimate.")
    parser.add_argument("--solver-tolerance", type=float, default=defaults.solver_tolerance,
                        help="Certified ||V* - V_n|| bound.")
    parser.add_argument("--mlp-probe-steps", type=int, default=defaults.mlp_probe_steps)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    args = parser.parse_args()

    settings = AnalysisSettings(args.episodes, args.solver_tolerance, args.mlp_probe_steps, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_summary(analyze_run(args.run_dir, settings, device))
    print(f"report: {args.run_dir / 'analysis' / 'report.json'}")


if __name__ == "__main__":
    main()
