"""Collects every number shown on the reviewer page into page/data.json (real data only)."""
import json
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from ipomdp.agents import UniformRandomAgent
from ipomdp.domain import BatchedPOMDPEnv
from ipomdp.experiments import load_trained_run
from ipomdp.interpretability import build_probe_dataset, q_values
from ipomdp.training import play_episodes, stream_seed

S = Path(sys.argv[1])
(S / "page").mkdir(exist_ok=True)
ref = torch.load(S / "qstar.pt", weights_only=False)
V, Q = ref["v"], ref["q"]
d = torch.device("cuda")
out = {}

# 1. Sweeps: per-seed paired gaps, aggregates and in-training evaluation curves.
out["sweeps"] = {}
for name in ("default", "polyak"):
    agg = json.load(open(f"runs/sweeps/{name}/aggregate.json"))
    out["sweeps"][name] = {"optimal_value_b0": agg["optimal_value_b0"],
                           "metrics": {k: {kk: v[kk] for kk in ("mean", "ci95", "std", "values")}
                                       for k, v in agg["metrics"].items()}}
    curves = {}
    for seed in agg["seeds"]:
        log = Path(f"runs/sweeps/{name}/seed{seed}/main.log").read_text()
        pts = re.findall(r"collection (\d+): greedy eval discounted return ([-\d.]+) \+- ([\d.]+)", log)
        curves[seed] = [[int(c), float(r), float(e)] for c, r, e in pts]
    out["sweeps"][name]["curves"] = curves

# 2. Exact value and action-value functions over the belief segment (p = P(tiger left)).
p = torch.linspace(0, 1, 201, dtype=torch.float64)
b = torch.stack([p, 1 - p], 1)
out["exact"] = {"p": p.tolist(), "v": V.value(b).tolist(), "q": q_values(Q, b).T.tolist()}

# 3. Learned acting model (polyak) vs exact, per exact posterior, on random-policy histories.
learned, geometry, probes, bounds = [], None, [], []
for seed in range(5):
    run_dir = Path(f"runs/sweeps/polyak/seed{seed}")
    t = load_trained_run(run_dir, d)
    run = t.run
    ep, _ = play_episodes(BatchedPOMDPEnv(t.pomdp, 256, 100, stream_seed(4242, 1), d),
                          UniformRandomAgent(3, 256, stream_seed(4242, 2), d))
    ds = build_probe_dataset(t.pomdp, run.acting_filter, ep)
    z, post = ds.latents, ds.posteriors
    key = post[:, 0].round(decimals=4)
    with torch.no_grad():
        v_hat = run.codec.mean(run.acting_value_head(z)).double()
        r_hat = torch.stack([run.codec.mean(run.acting_reward_head(
            z, F.one_hot(torch.full((len(z),), a, device=d), 3).float())) for a in range(3)], 1).double()
    rows = {}
    for k in key.unique().tolist():
        m = key == k
        if int(m.sum()) < 30:
            continue
        rows[f"{k:.4f}"] = {"n": int(m.sum()), "v": float(v_hat[m].mean()), "r": r_hat[m].mean(0).tolist()}
    learned.append(rows)
    if seed == 0:
        g = torch.Generator(device="cpu").manual_seed(0)
        idx = torch.randperm(len(z), generator=g)[:2400]
        zc = z.double().cpu()
        zc = zc - zc.mean(0)
        _, sv, vh = torch.linalg.svd(zc, full_matrices=False)
        coords = zc[idx] @ vh[:2].T
        var = (sv ** 2 / (sv ** 2).sum())[:2].tolist()
        geometry = {"xy": [[round(float(x), 4), round(float(y), 4)] for x, y in coords],
                    "p": [round(float(v), 4) for v in post[idx, 0].cpu()], "explained": var}
    report = json.load(open(run_dir / "analysis" / "report.json"))
    probes.append({k: {"mean_kl": v["mean_kl"], "mean_l1": v["mean_l1"], "by_posterior": v["by_posterior"]}
                   for k, v in report["probes"].items()})
    bounds.append(report["bounds"])
    print("seed", seed, "done", flush=True)
out["learned"] = learned
out["geometry"] = geometry
out["probes"] = probes
out["bounds"] = bounds
json.dump(out, open(S / "page" / "data.json", "w"))
print("written", (S / "page" / "data.json").stat().st_size)
