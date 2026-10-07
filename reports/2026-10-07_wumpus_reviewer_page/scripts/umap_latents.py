# ABSOLUTE PATH: reports/2026-10-07_wumpus_reviewer_page/scripts/umap_latents.py
# ==============================================================================
# UMAP OF THE AGENT'S LATENTS, COLOURED BY THE EXACT BELIEF (step 2 of 2)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Embedding:
#    - UMAP (n_neighbors 30, min_dist 0.1, euclidean, random_state 0) fitted on ALL alive latents
#      of dump_latents.py (agent and random histories together, so both share one map), after
#      per-feature standardisation. UMAP preserves neighbourhoods, not distances or densities:
#      clusters and colour gradients are meaningful, inter-cluster distances are not.
#    - The page gets a fixed random subsample of 3,000 points per source (seed 0) to stay small.
#    - Runs in a separate environment with umap-learn (the project does not depend on it):
#        uv venv <env>; VIRTUAL_ENV=<env> uv pip install umap-learn
#        <env>/bin/python <this> <latents .npz> <out .json>
#    - Neighbourhood agreement: for each history source, how often a point's 10 nearest neighbours in
#      the 2D map share its label (percept just received, last action, heading, cell), against
#      the chance rate sum_k p_k^2. Result 2026-10-07 (seed 0): on random histories the map is
#      organised first by the percept (93% vs 36%), weakly by heading (44% vs 25%) and cell
#      (43% vs 24%); on the agent's own histories heading and cell agree more (74%, 71%).
#    - Optional third argument: export.py's games_latents.npz. Those latents are standardised with the
#      same statistics and placed by the fitted map's transform() (not refitted), so the example
#      games' paths sit on the same map as the cloud.
#    - A trustworthiness score (sklearn, 15 neighbours, on a 4,000-point subsample) says how far
#      the 2D neighbourhoods reflect the 128-D ones (1 = perfectly).
# ==============================================================================

import json
import sys

import numpy as np
import umap
from sklearn.manifold import trustworthiness

data = np.load(sys.argv[1])
sources = ("agent", "random")
latents = np.concatenate([data[f"latents_{s}"] for s in sources])
standard = (latents - latents.mean(0)) / (latents.std(0) + 1e-6)
reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="euclidean", random_state=0)
embedding = reducer.fit_transform(standard)
games = []
if len(sys.argv) > 3:
    game_data = np.load(sys.argv[3])
    for key in sorted(game_data.files, key=lambda k: int(k[4:])):
        placed = reducer.transform((game_data[key] - latents.mean(0)) / (latents.std(0) + 1e-6))
        games.append([[round(float(a), 3), round(float(b), 3)] for a, b in placed])
rng = np.random.default_rng(0)
check = rng.choice(len(standard), 4000, replace=False)
trust = float(trustworthiness(standard[check], embedding[check], n_neighbors=15))

points, offset = [], 0
for si, s in enumerate(sources):
    count = len(data[f"latents_{s}"])
    keep = np.sort(rng.choice(count, min(3000, count), replace=False))
    for i in keep:
        e = embedding[offset + i]
        points.append([round(float(e[0]), 3), round(float(e[1]), 3), si, int(data[f"cell_{s}"][i]), int(data[f"heading_{s}"][i]),
                       round(float(data[f"danger_{s}"][i]), 3), round(float(data[f"log_layouts_{s}"][i]), 3),
                       int(data[f"step_{s}"][i]), int(data[f"last_action_{s}"][i]), int(data[f"percept_{s}"][i])])
    offset += count
from collections import Counter
from sklearn.neighbors import NearestNeighbors
arr = np.array(points)
agreement = []
for si, s in enumerate(sources):
    rows = arr[arr[:, 2] == si]
    idx = NearestNeighbors(n_neighbors=11).fit(rows[:, :2]).kneighbors(rows[:, :2], return_distance=False)[:, 1:]
    for col, label in ((9, "percept just received"), (8, "last action"), (4, "heading"), (3, "cell")):
        agree = float((rows[idx, col] == rows[:, col][:, None]).mean())
        chance = float(sum((c / len(rows)) ** 2 for c in Counter(rows[:, col]).values()))
        agreement.append({"source": "agent's own games" if s == "agent" else "random actions", "label": label,
                          "agree": agree, "chance": chance})
out = {"games": games, "agreement": agreement, "fields": ["x", "y", "source", "cell", "heading", "danger", "log_layouts", "step", "last_action", "percept"], "sources": list(sources),
       "points": points, "trustworthiness": trust, "total": {s: int(len(data[f"latents_{s}"])) for s in sources}}
with open(sys.argv[2], "w") as f:
    json.dump(out, f, separators=(",", ":"))
print(f"{len(points)} points; trustworthiness {trust:.3f}")
