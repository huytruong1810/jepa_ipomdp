# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/noop_drift_diag.py
# ==============================================================================
# DOES THE FILTER MOVE ON EXACT NO-OPS? (CONJECTURE C4, MEASURED ON STORED GREEDY GAMES)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-15, representation plan R0/R1, HANDOFF 6.2): the exact belief is unchanged by a GRAB
#    without GLITTER or a SHOOT without the arrow (both decidable from the history alone). Does the acting
#    filter's latent stay put there, and does it differ between the current and the oracle recipe?
# 2. Method (usage: uv run python <this> <run dir>; CPU, ~7 s): the 128 greedy games of best.pt stored in the
#    buffer (phantom_reward_diag.py's slot logic, simplified to the greedy-flag check), replayed through the
#    acting filter. Relative drift |z_{t+1} - z_t| / |z_t| on live no-op steps vs other live steps; plus the
#    acting head's P(GAME_OVER | z, FORWARD) < 0.02 bin and its deaths (these are the head's own training
#    games, so this bin is expected to be calibrated: head_shift.py item 3).
# 3. Result 2026-10-15 (the six double-value runs): drift on no-ops mean 0.217 / 0.117 / 0.098 (current
#    seeds 0-2) vs 0.030 / 0.023 / 0.021 (oracle), against 0.47-0.81 on other actions in both; 0-1 confident
#    "safe" deaths per network. Whether the drift costs information: noop_drift_cost.py (it does not).
# ==============================================================================
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from ipomdp.domain.wumpus import GAME_OVER, Percept, WumpusAction
from ipomdp.experiments.runs import load_trained_run
from ipomdp.training.checkpointing import load_checkpoint

torch.set_num_threads(2)
run_dir, dev = Path(sys.argv[1]), torch.device("cpu")
trained = load_trained_run(run_dir, dev)
run, cfg = trained.run, trained.cfg
model = run.eval_agent.model
GRAB, SHOOT, FORWARD = int(WumpusAction.GRAB), int(WumpusAction.SHOOT), int(WumpusAction.FORWARD)
NUM_A = len(WumpusAction)

buf = load_checkpoint(run_dir / "checkpoints" / "latest.pt")["buffer"]
t_cfg, c = cfg.training, trained.checkpoint_collection
start = c * t_cfg.env_batch_size + (c // t_cfg.eval_every - 1) * t_cfg.eval_episodes
games = slice(start, start + t_cfg.eval_episodes)
if "greedy" in buf and (not bool(buf["greedy"][games].all()) or bool(buf["greedy"][start - 1])):
    raise RuntimeError("not the greedy block")
actions, observations = buf["actions"][games], buf["observations"][games]
initial, rewards = buf["initial_observations"][games], buf["rewards"][games].double()
del buf
n, T = actions.shape

state = model.initial_states(n, initial)
percept, arrow = initial.clone(), torch.ones(n, dtype=torch.bool)
alive = torch.ones(n, dtype=torch.bool)
noops_so_far = torch.zeros(n)
drift_noop, drift_other, fwd = [], [], []                    # fwd: (p_death, died, noops so far)
head = run.acting_observation_head
with torch.no_grad():
    for t in range(T):
        a, o = actions[:, t], observations[:, t]
        noop = ((a == GRAB) & ((percept & Percept.GLITTER) == 0)) | ((a == SHOOT) & ~arrow)
        is_f = alive & (a == FORWARD)
        if is_f.any():
            p = F.softmax(head(state[is_f], F.one_hot(a[is_f], NUM_A).float()).float(), -1)[:, GAME_OVER]
            fwd.append(torch.stack([p.double(), (rewards[is_f, t] <= -1000).double(), noops_so_far[is_f].double()], 1))
        nxt = model.update(state, a, o)
        rel = (nxt - state).norm(dim=-1) / state.norm(dim=-1).clamp_min(1e-8)
        drift_noop.append(rel[alive & noop]); drift_other.append(rel[alive & ~noop])
        noops_so_far += (alive & noop).float()
        arrow &= a != SHOOT
        percept = o
        alive &= ~((rewards[:, t] >= 998) | (rewards[:, t] <= -1000)) & (o != GAME_OVER)
        state = nxt

dn, do = torch.cat(drift_noop), torch.cat(drift_other)
f = torch.cat(fwd)
conf = f[:, 0] < 0.02
miss = conf & (f[:, 1] > 0)
print(f"== {run_dir.name} (collection {c}, {n} greedy games; deaths {int((rewards <= -1000).any(1).sum())}, FORWARD deaths {int(f[:, 1].sum())})")
print(f"  rel drift |dz|/|z|: no-op n={len(dn)} mean {dn.mean():.3f} median {dn.median():.3f} | "
      f"other n={len(do)} mean {do.mean():.3f} median {do.median():.3f}")
print(f"  FORWARD p<0.02: n={int(conf.sum())} died {f[conf, 1].mean():.4f}; deaths among them {int(miss.sum())}, "
      f"no-ops so far: misses {f[miss, 2].mean() if miss.any() else float('nan'):.1f} vs "
      f"survived {f[conf & ~miss, 2].mean():.1f}")
