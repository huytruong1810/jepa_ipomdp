# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/tiger_acceptance_seed.py
# ==============================================================================
# TIGER ACCEPTANCE PROTOCOL AT ANOTHER NETWORK-INIT SEED
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Diagnostic of 2026-10-04/05, kept so the HANDOFF numbers are reproducible:
#    - Usage (from the repo root, or a git worktree with PYTHONPATH=<worktree>/src):
#      python reports/2026-10-02_wumpus_diagnostics/tiger_acceptance_seed.py <seed> [updates]. Runs
#      tests/test_world_model_acceptance.train_and_measure with its torch.manual_seed(0) replaced
#      by <seed> (data seeds unchanged) and prints the value error, planning return and probe KL.
#    - The optional [updates] replaces the test's UPDATES (3000) to check whether the value head's
#      fitted value iteration converges with more updates (2026-10-05, user decision).
#    - About 1.8 GB RAM and 40 min per run; run at most 2-3 at a time while ipomcp is busy.
#    - Logs of 2026-10-05: runs/tiger_acceptance_seeds/ (pre = commit 62deb68, post = with the hoist).
# ==============================================================================

import os, sys, torch
sys.path.insert(0, os.path.join(os.getcwd(), "tests"))
seed = int(sys.argv[1])
original = torch.manual_seed
torch.manual_seed = lambda s: original(seed if s == 0 else s)
import test_world_model_acceptance as acceptance
if len(sys.argv) > 2:
    acceptance.UPDATES = int(sys.argv[2])
m = acceptance.train_and_measure()
torch.manual_seed = original
error = float((m["value"] - m["optimal_value"]).abs().mean())
print(f"RESULT value_error {error:.4f}")
print(f"RESULT planning_return {m['planning_return'][0]:.4f} +- {m['planning_return'][1]:.4f}")
print(f"RESULT linear_kl {m['linear'].mean_kl:.6f}")
