# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/overfit_summary.py
# ==============================================================================
# SUMMARY OF THE OVERFITTING A/B (objective_ab.py LOGS WITH LEARNING CURVES)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Statistics (user guidance: several seeds, paired comparisons, 95% intervals):
#    - Logs are named obj_<arm>_seed<n>.log and hold "RESULT@<updates> name value" lines at the
#      --score-at counts plus final "RESULT name value" lines (scored at --updates).
#    - Per arm and update count: mean over seeds with a Student-t 95% half-width. Every arm is also
#      paired against the baseline arm (the first in sorted order, e.g. "a_base") at the same update
#      count and seed: mean difference with its own t interval.
#    - Usage: uv run python <this> <final update count> runs/wumpus/overfit_ab/obj_*.log
# ==============================================================================

import re
import sys
from collections import defaultdict
from pathlib import Path

from scipy import stats

METRICS = ("head_safe_miss", "head_danger_error", "head_safe", "head_certain_death", "head_nll_train", "head_nll_held",
           "probe_danger_error", "probe_cell_acc", "probe_heading_acc")


def interval(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, float("nan")
    sd = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5
    return mean, stats.t.ppf(0.975, n - 1) * sd / n ** 0.5


final = int(sys.argv[1])
data: dict[tuple[str, int, str], dict[int, float]] = defaultdict(dict)       # (arm, updates, metric) -> seed -> value
for path in map(Path, sys.argv[2:]):
    match = re.fullmatch(r"obj_(.+)_seed(\d+)\.log", path.name)
    arm, seed = match.group(1), int(match.group(2))
    for line in path.read_text().splitlines():
        hit = re.fullmatch(r"RESULT(?:@(\d+))? (\w+) ([-\d.naN]+)", line.strip())
        if hit:
            updates = int(hit.group(1)) if hit.group(1) else final
            data[(arm, updates, hit.group(2))][seed] = float(hit.group(3))

arms = sorted({a for a, _, _ in data})
updates_list = sorted({u for _, u, _ in data})
baseline = arms[0]
for metric in METRICS:
    print(f"== {metric}")
    for updates in updates_list:
        cells = []
        for arm in arms:
            values = data.get((arm, updates, metric), {})
            if not values:
                cells.append(f"{arm} -")
                continue
            mean, half = interval(list(values.values()))
            cell = f"{arm} {mean:.4f}±{half:.4f}(n={len(values)})"
            base = data.get((baseline, updates, metric), {})
            paired = [values[s] - base[s] for s in values if s in base]
            if arm != baseline and len(paired) >= 2:
                d, h = interval(paired)
                cell += f" Δ{d:+.4f}±{h:.4f}"
            cells.append(cell)
        print(f"  @{updates:6d}: " + " | ".join(cells))
