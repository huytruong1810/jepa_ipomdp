# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/hoist_forward_check.py
# ==============================================================================
# FORWARD EQUIVALENCE AND SPEED OF THE HOISTED UNROLL
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Diagnostic of 2026-10-04/05, kept so the HANDOFF numbers are reproducible:
#    - Compares BeliefFilter.unroll's per-step loop with input_net hoisted out of the loop, on
#      Tiger and Wumpus sizes, with and without bf16 autocast. Written before the hoist was
#      committed: today's unroll() IS the hoisted version, so rerun it at commit 62deb68 to
#      reproduce (bit-identical under autocast; 4.5-7.6x faster).
# ==============================================================================

import time, torch, torch.nn.functional as F
from ipomdp.models import BeliefFilter
dev = torch.device("cuda")
for (A, O, D, H, B, T) in [(3, 2, 32, 64, 16, 100), (5, 33, 128, 256, 16, 50)]:
    torch.manual_seed(0)
    f = BeliefFilter(A, O, D, H, 1).to(dev)
    a = F.one_hot(torch.randint(0, A, (B, T), device=dev), A).float()
    o = F.one_hot(torch.randint(0, O, (B, T), device=dev), O).float()
    o0 = None if O == 2 else F.one_hot(torch.randint(0, O, (B,), device=dev), O).float()
    def hoisted():
        z = f.start(B, o0); zs = [z]
        x = f.input_net(torch.cat([a, o], -1))           # (B, T, H) once
        for t in range(T):
            z = f.cell(x[:, t], z); zs.append(z)
        return torch.stack(zs, 1)
    for autocast in (False, True):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            ref, new = f.unroll(a, o, o0), hoisted()
            same = torch.equal(ref, new)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(20): f.unroll(a, o, o0)
            torch.cuda.synchronize(); t1 = time.perf_counter()
            for _ in range(20): hoisted()
            torch.cuda.synchronize(); t2 = time.perf_counter()
        print(f"A={A} O={O} D={D} autocast={autocast}: bit-identical {same}, max diff {(ref-new).abs().max().item():.2e}; "
              f"unroll {(t1-t0)/20*1000:.1f} ms vs hoisted {(t2-t1)/20*1000:.1f} ms")
