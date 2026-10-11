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
#
# 4. Both Domains (2026-10-10):
#    - The domain is the last `env=<name>` override, else conf/config.yaml's default (tiger).
#      Tiger seeds are analysed by analyze_run (exact solver), Wumpus seeds by analyze_wumpus_run,
#      and aggregate_reports uses that domain's metric table (experiments/aggregate.py, section 5).
#      --solver-tolerance is Tiger-only and raises for Wumpus.
#    - Runs trained elsewhere join a sweep by symlinking their directory as seed<k> with the
#      trained marker: a seed whose analysis/report.json exists is aggregated as is. The recorded
#      overrides must then be the ones those runs were trained with.
# ==============================================================================

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from omegaconf import OmegaConf

from ipomdp.experiments import AnalysisSettings, WumpusAnalysisSettings, aggregate_reports, analyze_run, analyze_wumpus_run

REPO = Path(__file__).resolve().parent
TRAINED_MARKER = "TRAINED"


def train(seed_dir: Path, seed: int, overrides: list[str]) -> None:
    command = [sys.executable, str(REPO / "main.py"), f"seed={seed}", f"hydra.run.dir={seed_dir}", *overrides]
    print(f"[sweep] training: {' '.join(command[1:])}", flush=True)
    start = time.monotonic()
    subprocess.run(command, cwd=REPO, check=True)
    (seed_dir / TRAINED_MARKER).write_text(f"{time.monotonic() - start:.0f} s\n")


def domain_of(overrides: list[str]) -> str:
    """Section 4: the last env=<name> override, else the default env of conf/config.yaml."""
    chosen = [o.split("=", 1)[1] for o in overrides if o.split("=", 1)[0] == "env"]
    if chosen:
        return chosen[-1]
    (default,) = [entry.env for entry in OmegaConf.load(REPO / "conf" / "config.yaml").defaults
                  if OmegaConf.is_dict(entry) and "env" in entry]
    return default


def record_condition(sweep_dir: Path, overrides: list[str], settings: AnalysisSettings | WumpusAnalysisSettings) -> None:
    """Section 3: one sweep directory is one experimental condition."""
    condition = {"overrides": overrides, "analysis": asdict(settings)}
    path = sweep_dir / "sweep.json"
    if path.exists():
        recorded = json.loads(path.read_text())
        if recorded != condition:
            raise ValueError(f"{sweep_dir} was started with {recorded}, not {condition}. Use a new sweep name.")
    else:
        sweep_dir.mkdir(parents=True, exist_ok=True)   # may already hold symlinked seeds (section 4)
        path.write_text(json.dumps(condition, indent=2))


def main() -> None:
    defaults = AnalysisSettings()
    parser = argparse.ArgumentParser(description="Train, analyse and aggregate over seeds (see module header).")
    parser.add_argument("name", help="Sweep directory name under runs/sweeps/.")
    parser.add_argument("--seeds", required=True, type=lambda s: [int(k) for k in s.split(",")],
                        help="Comma-separated training seeds, e.g. 0,1,2,3,4.")
    parser.add_argument("--episodes", type=int, default=defaults.episodes)
    parser.add_argument("--solver-tolerance", type=float, default=None,
                        help=f"Tiger only (default {defaults.solver_tolerance}).")
    parser.add_argument("--mlp-probe-steps", type=int, default=defaults.mlp_probe_steps)
    parser.add_argument("--analysis-seed", type=int, default=defaults.seed)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides passed to every main.py run.")
    args = parser.parse_intermixed_args()
    for override in args.overrides:
        if override.split("=")[0].lstrip("+~") in ("seed", "hydra.run.dir", "resume"):
            raise ValueError(f"{override!r} is set by the sweep itself.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError(f"Duplicate seeds in {args.seeds}.")

    domain = domain_of(args.overrides)
    if domain == "wumpus":
        if args.solver_tolerance is not None:
            raise ValueError("--solver-tolerance applies to Tiger only: Wumpus World has no exact solver.")
        settings = WumpusAnalysisSettings(args.episodes, args.mlp_probe_steps, args.analysis_seed)
        analyze = analyze_wumpus_run
    else:
        tolerance = defaults.solver_tolerance if args.solver_tolerance is None else args.solver_tolerance
        settings = AnalysisSettings(args.episodes, tolerance, args.mlp_probe_steps, args.analysis_seed)
        analyze = analyze_run
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
            analyze(seed_dir, settings, device)
        reports.append(json.loads(report_path.read_text()))

    statistics = aggregate_reports(reports, domain)
    reference = ({"optimal_value_b0": reports[0]["optimal_value_b0"]} if domain == "tiger" else {})
    aggregate = {"domain": domain, "seeds": args.seeds, **reference,
                 "metrics": {name: asdict(stat) for name, stat in statistics.items()}}
    (sweep_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2))
    print(f"\n{args.name} ({domain}): {len(args.seeds)} seeds {args.seeds}"
          + (f", V*(b0) = {reference['optimal_value_b0']:.3f}" if reference else ""))
    print(f"  {'metric':<36} {'mean':>10} {'95% CI':>10} {'std':>10}")
    for name, stat in statistics.items():
        print(f"  {name:<36} {stat.mean:10.4f} {stat.ci95:>10.4f} {stat.std:10.4f}")
    print(f"aggregate: {sweep_dir / 'aggregate.json'}")


if __name__ == "__main__":
    main()
