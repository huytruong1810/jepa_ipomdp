# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/full_info_vi.py
# ==============================================================================
# FULL-INFORMATION UPPER BOUND ON V*(b0)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Scratch diagnostic of 2026-10-02, kept so the HANDOFF numbers are reproducible:
#    - Value iteration over the hunter's 256 states for each of the 16,380 layouts with the
#      domain's own step_rules; the mean over layouts of each known layout's value bounds V*(b0)
#      from above. Also reports the fraction of layouts winnable within 50 / 100 steps.
# ==============================================================================

import torch
from ipomdp.domain.wumpus import WumpusWorld, WumpusState, DISCOUNT
dev = torch.device("cuda")
w = WumpusWorld(dev)
L = w.num_layouts
# hunter index h = cell*16 + heading*4 + arrow*2 + alive (256); index 256 = done (absorbing, value 0)
h = torch.arange(256, device=dev)
lay = torch.arange(L, device=dev)[:, None].expand(L, 256)
hh = h[None].expand(L, 256)
s = WumpusState(lay, hh // 16, (hh // 4) % 4, ((hh // 2) % 2).bool(), (hh % 2).bool(), torch.zeros_like(hh, dtype=torch.bool))
nxt, R = [], []
for a in range(5):
    ns, _, r = w.step_rules(s, torch.full_like(hh, a))
    idx = ns.cell * 16 + ns.heading * 4 + ns.arrow.long() * 2 + ns.wumpus_alive.long()
    nxt.append(torch.where(ns.done, 256, idx)); R.append(r.double())
nxt, R = torch.stack(nxt, -1), torch.stack(R, -1)           # (L,256,5)
V = torch.zeros(L, 257, dtype=torch.float64, device=dev)
from ipomdp.domain.wumpus import START_HEADING
start = 0 * 16 + int(START_HEADING) * 4 + 2 + 1
for k in range(1, 2001):
    Q = R + DISCOUNT * V.gather(1, nxt.view(L, -1)).view(L, 256, 5)
    newV = torch.cat([Q.max(-1).values, V[:, 256:]], 1)
    if k == 50:
        win50 = newV[:, start] > 0
        print(f'50-step window: mean V(b0) {newV[:, start].mean():.1f}; winnable within 50 {win50.double().mean():.4f}')
    if k == 100:
        print(f"100-step window: mean V(b0) {newV[:, start].mean():.1f}; winnable layouts (V>0) {(newV[:, start] > 0).double().mean():.3f}")
    if (newV - V).abs().max() < 1e-9: break
    V = newV
print("winnable ever (inf):", (V[:, start] > 0).double().mean().item())
print(f"infinite horizon ({k} iters): full-information mean V(b0) {V[:, start].mean():.1f}")
print(f"value of waiting forever: {-1/(1-DISCOUNT):.1f}; 100-step window: {-(1-DISCOUNT**100)/(1-DISCOUNT):.1f}")
