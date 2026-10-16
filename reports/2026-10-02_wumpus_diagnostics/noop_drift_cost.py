# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/noop_drift_cost.py
# ==============================================================================
# DOES THE NO-OP DRIFT COST CELL / HEADING INFORMATION? (C4 FOLLOW-UP)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-15, after noop_drift_diag.py found the current recipe's latent moving 4-7x more on
#    exact no-ops than the oracle recipe's): is that drift where the agent's cell/heading information goes,
#    which would make a no-op consistency term an R1 arm?
# 2. Method (usage: uv run python <this> <run dir>; GPU, ~25 s): the cached pool runs/wumpus/supervised_filter/
#    pools.pt; best.pt's acting filter replayed (i) as trained and (ii) with its update skipped on exact
#    no-ops (history-decidable, not privileged). MLP probes (objective_ab.py's recipe, cell + heading) fitted
#    on 2048 training games, scored on the held-out random half, overall and binned by the no-ops seen so far
#    at matched steps 15-25 (unmatched bins confound no-ops with the number of moves).
# 3. Result 2026-10-15 (six double-value runs): NO. At matched steps accuracy RISES with the no-ops seen
#    (current seed 0: cell 0.34 / 0.46 / 0.50 / 0.60 for 0-2 / 3-5 / 6-9 / 10+ no-ops); skipping no-op updates
#    gains +1 to +4 points overall on the current seeds (cell 0.588 -> 0.631, 0.588 -> 0.611, 0.613 -> 0.627)
#    and nothing on the oracle seeds (95-100%). The gap is in tracking moves and turns.
# ==============================================================================
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ipomdp.domain.wumpus import Percept, WumpusAction
from ipomdp.experiments.runs import load_trained_run

run_dir, dev = Path(sys.argv[1]), torch.device("cuda")
trained = load_trained_run(run_dir, dev)
model = trained.run.eval_agent.model
pools = torch.load("runs/wumpus/supervised_filter/pools.pt", map_location=dev)
GRAB, SHOOT = int(WumpusAction.GRAB), int(WumpusAction.SHOOT)


def replay(data, rows, skip: bool):
    a_all, o_all, init = data["actions"][rows], data["observations"][rows], data["initial"][rows]
    n, T = a_all.shape
    state = model.initial_states(n, init)
    percept, arrow = init.clone(), torch.ones(n, dtype=torch.bool, device=dev)
    count = torch.zeros(n, device=dev)
    zs, counts = [state.clone()], [count.clone()]
    steps = torch.arange(T + 1, device=dev).expand(n, T + 1)
    with torch.no_grad():
        for t in range(T):
            a, o = a_all[:, t], o_all[:, t]
            noop = ((a == GRAB) & ((percept & int(Percept.GLITTER)) == 0)) | ((a == SHOOT) & ~arrow)
            nxt = model.update(state, a, o)
            state = torch.where(noop[:, None], state, nxt) if skip else nxt
            count = count + noop.float()
            arrow &= a != SHOOT
            percept = o
            zs.append(state.clone()); counts.append(count.clone())
    z, c = torch.stack(zs, 1), torch.stack(counts, 1)
    alive = data["alive"][rows]
    return z[alive].float(), torch.stack([c[alive], steps[alive].float()], 1), data["cell"][rows][alive], data["heading"][rows][alive]


train, held = pools["train"], pools["held"]
half_t, half_h = train["actions"].shape[0] // 2, held["actions"].shape[0] // 2
fit_rows = torch.cat([torch.arange(0, 1024, device=dev), torch.arange(half_t, half_t + 1024, device=dev)])
held_rows = torch.arange(half_h, held["actions"].shape[0], device=dev)            # the random half
print(f"== {run_dir.name} best.pt (collection {trained.checkpoint_collection})")
for skip in (False, True):
    fz, _, fc, fh = replay(train, fit_rows, skip)
    hz, hn, hc, hh = replay(held, held_rows, skip)
    mean, std = fz.mean(0), fz.std(0) + 1e-6
    torch.manual_seed(0)
    probe = nn.Sequential(nn.Linear(fz.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 20)).to(dev)
    opt = torch.optim.Adam(probe.parameters(), 1e-3)
    g = torch.Generator(device=dev).manual_seed(0)
    for _ in range(3000):
        r = torch.randint(0, len(fz), (512,), device=dev, generator=g)
        out = probe((fz[r] - mean) / std)
        loss = F.cross_entropy(out[:, :16], fc[r]) + F.cross_entropy(out[:, 16:], fh[r])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        out = probe((hz - mean) / std)
    cell_ok, head_ok = (out[:, :16].argmax(1) == hc).float(), (out[:, 16:].argmax(1) == hh).float()
    rows = [f"all n={len(hn)} cell {cell_ok.mean():.3f} heading {head_ok.mean():.3f}"]
    hs, hn = hn[:, 1], hn[:, 0]
    mid = (hs >= 15) & (hs <= 25)
    for lo, hi in ((0, 3), (3, 6), (6, 10), (10, 99)):
        m = mid & (hn >= lo) & (hn < hi)
        if m.sum() > 50:
            rows.append(f"no-ops [{lo},{hi}) n={int(m.sum())} cell {cell_ok[m].mean():.3f} heading {head_ok[m].mean():.3f}")
    print(f"  {'skip' if skip else 'as trained'}: " + " | ".join(rows), flush=True)
