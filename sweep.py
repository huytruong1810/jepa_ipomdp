# ABSOLUTE PATH: sweep.py
# ==============================================================================
# MULTI-SEED EXPERIMENT: TRAIN, ANALYSE AND AGGREGATE OVER TRAINING SEEDS
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Usage:
#      uv run sweep.py <name> --seeds 0,1,2,3,4 [hydra overrides ...]
#    e.g. `uv run sweep.py baseline --seeds 0,1,2,3,4` or
#         `uv run sweep.py short --seeds 0,1,2 training.total_episodes=1280`.
#    Every seed k is trained by main.py into runs/sweeps/<name>/seed<k>/ (the overrides are
#    passed through unchanged), analysed by the same analyze_run as analyze.py, and all seeds
#    are aggregated into runs/sweeps/<name>/aggregate.json with Student-t 95% intervals
#    (ipomdp/experiments/aggregate.py explains why the seed is the unit of replication).
#
# 2. Sequential, One Process per Training Run:
#    - The single GPU is the bottleneck (search is Python-bound, section "Known open issues" of
#      HANDOFF.md), so seeds run one after another. Each training run is its own main.py
#      process, so it is launched exactly as a hand-started run would be (same Hydra config
#      composition, same run-directory contents) and its GPU memory is released before the
#      next seed.
#    - main.py exits with 0 only when every collection completed (main.py, section 4). Any
#      other status stops the sweep.
#
# 3. Continuing an Interrupted Sweep:
#    - runs/sweeps/<name>/sweep.json records the overrides and analysis settings. Re-running
#      the command with the SAME overrides skips every seed whose analysis/report.json exists,
#      analyses seeds marked as trained but not yet analysed, and trains the rest; more seeds
#      may be added this way. Different overrides or settings raise: one sweep directory is one
#      experimental condition. A seed directory without the trained marker is an interrupted
#      training run and raises too: delete it, or finish it by hand with
#      `main.py resume=<seed dir>/checkpoints/latest.pt hydra.run.dir=<seed dir>` and create the
#      marker. Resuming is bit-exact only from the state at which latest.pt was written, so
#      that choice is left to the user.
# ==============================================================================

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from ipomdp.experiments import AnalysisSettings, aggregate_reports, analyze_run

REPO = Path(__file__).resolve().parent
TRAINED_MARKER = "TRAINED"


def train(seed_dir: Path, seed: int, overrides: list[str]) -> None:
    command = [sys.executable, str(REPO / "main.py"), f"seed={seed}", f"hydra.run.dir={seed_dir}", *overrides]
    print(f"[sweep] training: {' '.join(command[1:])}", flush=True)
    start = time.monotonic()
    subprocess.run(command, cwd=REPO, check=True)
    (seed_dir / TRAINED_MARKER).write_text(f"{time.monotonic() - start:.0f} s\n")


def record_condition(sweep_dir: Path, overrides: list[str], settings: AnalysisSettings) -> None:
    """Section 3: one sweep directory is one experimental condition."""
    condition = {"overrides": overrides, "analysis": asdict(settings)}
    path = sweep_dir / "sweep.json"
    if path.exists():
        recorded = json.loads(path.read_text())
        if recorded != condition:
            raise ValueError(f"{sweep_dir} was started with {recorded}, not {condition}. Use a new sweep name.")
    else:
        sweep_dir.mkdir(parents=True)
        path.write_text(json.dumps(condition, indent=2))


def main() -> None:
    defaults = AnalysisSettings()
    parser = argparse.ArgumentParser(description="Train, analyse and aggregate over seeds (see module header).")
    parser.add_argument("name", help="Sweep directory name under runs/sweeps/.")
    parser.add_argument("--seeds", required=True, type=lambda s: [int(k) for k in s.split(",")],
                        help="Comma-separated training seeds, e.g. 0,1,2,3,4.")
    parser.add_argument("--episodes", type=int, default=defaults.episodes)
    parser.add_argument("--solver-tolerance", type=float, default=defaults.solver_tolerance)
    parser.add_argument("--mlp-probe-steps", type=int, default=defaults.mlp_probe_steps)
    parser.add_argument("--analysis-seed", type=int, default=defaults.seed)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides passed to every main.py run.")
    args = parser.parse_intermixed_args()
    for override in args.overrides:
        if override.split("=")[0].lstrip("+~") in ("seed", "hydra.run.dir", "resume"):
            raise ValueError(f"{override!r} is set by the sweep itself.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError(f"Duplicate seeds in {args.seeds}.")

    settings = AnalysisSettings(args.episodes, args.solver_tolerance, args.mlp_probe_steps, args.analysis_seed)
    sweep_dir = REPO / "runs" / "sweeps" / args.name
    record_condition(sweep_dir, args.overrides, settings)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reports = []
    for seed in args.seeds:
        seed_dir = sweep_dir / f"seed{seed}"
        report_path = seed_dir / "analysis" / "report.json"
        if not report_path.exists():
            if seed_dir.exists() and not (seed_dir / TRAINED_MARKER).exists():
                raise RuntimeError(f"{seed_dir} holds an interrupted training run (module header, section 3).")
            if not seed_dir.exists():
                train(seed_dir, seed, args.overrides)
            print(f"[sweep] analysing seed {seed}", flush=True)
            analyze_run(seed_dir, settings, device)
        reports.append(json.loads(report_path.read_text()))

    statistics = aggregate_reports(reports)
    aggregate = {"seeds": args.seeds, "optimal_value_b0": reports[0]["optimal_value_b0"],
                 "metrics": {name: asdict(stat) for name, stat in statistics.items()}}
    (sweep_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2))
    print(f"\n{args.name}: {len(args.seeds)} seeds {args.seeds}, V*(b0) = {reports[0]['optimal_value_b0']:.3f}")
    print(f"  {'metric':<36} {'mean':>10} {'95% CI':>10} {'std':>10}")
    for name, stat in statistics.items():
        print(f"  {name:<36} {stat.mean:10.4f} {stat.ci95:>10.4f} {stat.std:10.4f}")
    print(f"aggregate: {sweep_dir / 'aggregate.json'}")


if __name__ == "__main__":
    main()
