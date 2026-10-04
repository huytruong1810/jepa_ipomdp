# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/ab_summary.py
# ==============================================================================
# A/B SUMMARY OF supervised_filter.py LOGS: PER-ARM MEANS WITH STUDENT-T 95% INTERVALS
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Statistics (user guidance, 2026-10-03: A/B with good statistics):
#    - One log per (arch, seed); seeds differ only in initialisation and minibatch order, and
#      the data pools are identical, so the arms are compared per seed (paired difference
#      B - A with its own interval) as well as by arm means.
#    - Usage: uv run python <this> runs/wumpus/supervised_filter/ab_{arch}_seed{n}.log ...
#      or objective_ab.py logs named obj_{arm}_seed{n}.log, whose "RESULT name value" lines are
#      read instead. With more than two arms every arm is paired against the first one in
#      sorted order (name the baseline arm so that it sorts first, e.g. "a_jepa_k0").
# ==============================================================================

import re
import sys
from collections import defaultdict

from scipy import stats

METRICS = {
    "danger_error": r"danger ahead mean \|error\| ([\d.]+)",
    "certain_death_decoded": r"exact \[0\.6,1\) n=\s*\d+: exact [\d.]+\s+supervised ([\d.]+)",
    "safe_decoded": r"exact \[0,1e-09\) n=\s*\d+: exact [\d.]+\s+supervised ([\d.]+)",
    "pit_error": r"pit ([\d.]+)/",
    "wumpus_error": r"wumpus ([\d.]+)/",
    "gold_error": r"gold ([\d.]+)/",
    "cell_acc": r"cell acc ([\d.]+)",
    "heading_acc": r"heading acc ([\d.]+)",
}


def interval(values: list[float]) -> str:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return f"{mean:.4f}"
    sd = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5
    return f"{mean:.4f} ± {stats.t.ppf(0.975, n - 1) * sd / n ** 0.5:.4f}"


runs: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
for path in sys.argv[1:]:
    arm, seed = re.search(r"(?:ab|obj)_(\w+?)_seed(\d+)\.log", path).groups()
    text = open(path).read()
    results = dict(re.findall(r"^RESULT (\w+) ([-\d.]+)$", text, re.MULTILINE))
    runs[arm][int(seed)] = ({k: float(v) for k, v in results.items()} if results
                            else {k: float(re.search(p, text).group(1)) for k, p in METRICS.items()})
arms = sorted(runs)
metrics = list(next(iter(runs[arms[0]].values())))
base = arms[0]
print(f"{'metric':<22}" + "".join(f"{a + f' (n={len(runs[a])})':<22}" for a in arms))
for metric in metrics:
    print(f"{metric:<22}" + "".join(f"{interval([r[metric] for r in runs[a].values()]):<22}" for a in arms))
for arm in arms[1:]:
    common = sorted(set(runs[base]) & set(runs[arm]))
    print(f"paired {arm} - {base} (seeds {common}):")
    for metric in metrics:
        print(f"  {metric:<22}{interval([runs[arm][s][metric] - runs[base][s][metric] for s in common])}")
