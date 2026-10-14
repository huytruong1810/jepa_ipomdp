# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/phantom_reward_diag.py
# ==============================================================================
# DO THE LEARNED HEADS PAY A PHANTOM REWARD FOR NO-OP GRAB / SHOOT? (CONJECTURE C5)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-14, prompted by self_loop_diag.py's result on the oracle-latent networks):
#    - Valuing a self-loop child at r_hat(z, a) / (1 - gamma) made the planner choose GRAB MORE often
#      (seed 0: 39% -> 63% of live steps, 72% of chosen actions flagged as loops). Under the old values
#      the GRAB branch was worth ~gamma V_hat ~ 740; choosing it more under the loop value means
#      r_hat(z, GRAB) / (1 - gamma) is of the same order, i.e. r_hat(z, GRAB) ~ +7, where the exact
#      reward of a GRAB without GLITTER is -1.
#    - The exact reward is a function of the observation history alone (wumpus.py step_rules): GRAB pays
#      999 iff the current percept has GLITTER, else -1; SHOOT costs -10 with the arrow and -1 without
#      (the arrow is spent by the first SHOOT, visible in the action history). A miscalibrated reward head
#      (e.g. a two-hot mixture keeping ~1% mass on the 999 bin) would lift every no-op loop by
#      (r_hat + 1) / (1 - gamma): +100 per unit of phantom reward. That alone could hold the idle value
#      ~950 above the realised return, and it would explain why the oracle latent idles too (a perfect
#      latent does not fix a head that is not sharp on it).
#
# 2. Method (usage: uv run python <this> <run dir> [cpu|cuda]; CPU by default):
#    - States: the 128 greedy evaluation games stored in the buffer (training.store_eval_episodes) at the
#      checkpoint's own collection, i.e. games played by exactly the best.pt networks (greedy flags
#      checked where the buffer stores them, i.e. runs from 52ac3fc on; grounding_screen.py's slot logic), replayed through the acting
#      filter. Live steps only (before the game ended).
#    - Per live step, from the acting heads: r_hat(z, a) for every a, P_hat(o | z, a), and the exact reward
#      of every action given the history (GRAB: 999 if the last percept has GLITTER else -1; SHOOT: -10
#      with the arrow, -1 without; FORWARD -1 except death, so FORWARD/turns are reported but not judged).
#    - Reported, split into idle (unfinished game, after the last new cell; cells from dead reckoning by
#      the exact filter) and progress steps:
#        GRAB  without GLITTER: mean / median r_hat, P_hat(GAME_OVER), and the implied loop lift
#              (r_hat + 1) / (1 - gamma);
#        GRAB  with GLITTER: mean r_hat (should be ~999);
#        SHOOT without the arrow: mean r_hat (exact -1), P_hat(SCREAM bit);
#        the share of the idle state's V_hat that a phantom-free loop would remove:
#              V_hat - max over the no-op actions of [r_exact + gamma V_hat(loop child)].
#
# 2b. Pre-registered readout (recorded 2026-10-14 before any run), per network (oracle-latent final
#    networks seeds 0-2; current agent seeds 0-2):
#    - "Phantom reward": on idle steps, mean r_hat(GRAB | no GLITTER) >= +2 (a loop lift >= +300 over the
#      exact -1 loop) on 2+ of 3 seeds of a population. Then the idle value is propped by a head error,
#      not (only) by the backup: next, a planning-time diagnostic with the exact no-op rewards substituted
#      (privileged, a bound), then a non-privileged fix of the reward head's calibration.
#    - "No phantom": |mean r_hat(GRAB | no GLITTER) + 1| < 0.5 on 2+ of 3 seeds of both populations. Then
#      the loop result is read as pre-registered (a symptom) and fvi_screen.py (C2) is next.
#    - Otherwise partial: report per seed with the SHOOT numbers.
#
# 3. Result 2026-10-14 (logs <run>.phantom_reward_diag.log; CPU, ~45 s per network):
#    - Pre-registered readout: "no phantom" in both populations. On idle steps r_hat(GRAB | no GLITTER) is
#      -1.00 / -1.00 / -1.01 (oracle-latent final networks) and -0.99 / -1.00 / -1.00 (current agent seeds 0-2);
#      r_hat(SHOOT | no arrow) -1.00 everywhere, P_hat(GAME_OVER | GRAB, no GLITTER) and P_hat(SCREAM | no arrow)
#      <= 1e-4. The reward and observation heads are exact on the no-ops. My inference from the loop result
#      (r_hat(GRAB) ~ +7) was wrong.
#    - Found instead (section-3 addition, not pre-registered), the value overshoots what any game can pay:
#                              V_hat > 999 (live)   saw GLITTER / of them not won   one-step Q(GRAB) vs best other at GLITTER
#        oracle seed 0         0.358                13 / 13                         1067 vs 1281 (GRAB argmax 0%)
#        oracle seed 1         0.584                26 / 0                          1329 vs 1319 (92%)
#        oracle seed 2         0.463                25 / 25                         1033 vs 1292 (0%)
#        current seed 0        0.008                39 / 0                          1024 vs 1014 (100%)
#        current seed 1        0.008                38 / 0                          1029 vs 1022 (97%)
#        current seed 2        0.009                46 / 0                          1046 vs 1038 (91%)
#      On oracle seeds 0 and 2 (win rate 0) EVERY game that reached the gold stood on it, perceiving GLITTER,
#      and did not grab: idling was valued ~1290, winning 999 + gamma V(GAME_OVER child). Q(GRAB) > 999 also
#      shows a positive value (+25 to +70) on the absorbing GAME_OVER latent, whose true value is 0.
#      The current agent's idle V_hat is 626-663 (realised idle returns ~ -50): below the ceiling but just as
#      flat; its margin for grabbing is only 7-10.
#    - Reading: an impossible value (> 999) cannot come from any policy's returns; it is produced by the
#      backup itself. This is direct evidence for C2 (maximisation bias through the learned model) and makes
#      fvi_screen.py the next test, with "V > 999" as an extra measure.
# ==============================================================================

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ipomdp.agents import WumpusReferenceAgent
from ipomdp.domain import game_outcomes
from ipomdp.domain.wumpus import GAME_OVER, Percept, WumpusAction
from ipomdp.experiments.runs import load_trained_run
from ipomdp.training.checkpointing import load_checkpoint

run_dir = Path(sys.argv[1])
dev = torch.device(sys.argv[2] if len(sys.argv) > 2 else "cpu")
trained = load_trained_run(run_dir, dev)
world, run, cfg = trained.setup.domain, trained.run, trained.cfg
model = run.eval_agent.model
if model.horizon is not None:
    raise ValueError("Time-aware runs carry steps-left in the state; this diagnostic reads plain latents only.")
GAMMA, GRAB, SHOOT = world.discount, int(WumpusAction.GRAB), int(WumpusAction.SHOOT)
NUM_A = len(WumpusAction)

# --- The checkpoint's own greedy games (grounding_screen.py's slot reconstruction) ---
buf = load_checkpoint(run_dir / "checkpoints" / "latest.pt")["buffer"]
size, t_cfg = buf["size"], cfg.training
if size != buf["actions"].shape[0] and buf["next"] != size:
    raise RuntimeError("the buffer wrapped around; insertion-order reconstruction does not apply.")
if not t_cfg.store_eval_episodes:
    raise ValueError("this run did not store its greedy evaluation games.")
c = trained.checkpoint_collection
if c % t_cfg.eval_every != 0:
    raise ValueError(f"checkpoint collection {c} is not an evaluation collection.")
start = c * t_cfg.env_batch_size + (c // t_cfg.eval_every - 1) * t_cfg.eval_episodes
games = slice(start, start + t_cfg.eval_episodes)
if start + t_cfg.eval_episodes > size or ("greedy" in buf and (not bool(buf["greedy"][games].all())
                                                               or bool(buf["greedy"][start - 1]))):
    raise RuntimeError(f"slot {start}..{start + t_cfg.eval_episodes} is not collection {c}'s greedy block.")
actions = buf["actions"][games].to(dev)
observations = buf["observations"][games].to(dev)
initial = buf["initial_observations"][games].to(dev)
rewards = buf["rewards"][games].double()
n, T = actions.shape
del buf

# Replay: acting latents, the exact filter's cell (for idle), the arrow, the current percept.
shadow = WumpusReferenceAgent(world, n, 0.4)
shadow.reset(initial)
state = model.initial_states(n, initial)
percept, arrow = initial.clone(), torch.ones(n, dtype=torch.bool, device=dev)
rows = []
with torch.no_grad():
    for t in range(T):
        m = shadow._tracker.marginals(shadow._consistent, shadow._own)
        rows.append((state.clone(), percept.clone(), arrow.clone(), m.cell.clone(), m.done.clone()))
        shadow.update(actions[:, t], observations[:, t])
        state = model.update(state, actions[:, t], observations[:, t])
        arrow &= actions[:, t] != SHOOT
        percept = observations[:, t]

unfinished = ~((rewards >= 998).any(1) | (rewards <= -1000).any(1))
items = []                                                   # (z, percept, arrow, idle)
saw_glitter = torch.zeros(n, dtype=torch.bool)
for b in range(n):
    live = [(t, rows[t]) for t in range(T) if not bool(rows[t][4][b])]
    seen, last_new = set(), 0
    for t, r in live:
        if int(r[3][b]) not in seen:
            seen.add(int(r[3][b])); last_new = t
    saw_glitter[b] = any(bool(int(r[1][b]) & Percept.GLITTER) for _, r in live)
    for t, r in live:
        items.append((r[0][b], int(r[1][b]), bool(r[2][b]), bool(unfinished[b]) and t > last_new))
z = torch.stack([i[0] for i in items]).float()
glitter = torch.tensor([bool(i[1] & Percept.GLITTER) for i in items])
arrow = torch.tensor([i[2] for i in items])
idle = torch.tensor([i[3] for i in items])

with torch.no_grad():
    e = model.expand(z)
    r_hat, p_hat = e.rewards.double().cpu(), e.observation_probs.double().cpu()   # (N, A), (N, A, O)
    v_hat = model.values(z).double().cpu()
    # the GRAB / SHOOT child on the "nothing happened" branch: same percept, no bump, no scream
    same = torch.tensor([i[1] for i in items], device=dev)
    loop_v = {}
    for a in (GRAB, SHOOT):
        kid = model.update(z, torch.full((len(z),), a, device=dev), same)
        loop_v[a] = model.values(kid).double().cpu()

print(f"== {run_dir.name} best.pt (collection {c}); its {n} stored greedy games {start}..{start + n}: "
      f"{ {k: round(float(v), 4) for k, v in game_outcomes(rewards).items()} }; live steps {len(items)}, idle {int(idle.sum())}")


def stats(name: str, x: torch.Tensor) -> str:
    return f"{name} n={len(x)}" + (f" mean {x.mean():8.2f} median {x.median():8.2f}" if len(x) else "")


for label, where in (("idle", idle), ("progress", ~idle)):
    g0 = where & ~glitter
    print(f"  [{label}]")
    print(f"    {stats('r_hat(GRAB | no GLITTER)', r_hat[g0, GRAB])}  (exact -1); "
          f"P_hat(GAME_OVER) mean {p_hat[g0, GRAB, GAME_OVER].mean():.4f}; "
          f"implied loop lift (r_hat + 1)/(1 - gamma) mean {((r_hat[g0, GRAB] + 1) / (1 - GAMMA)).mean():8.1f}")
    print(f"    {stats('r_hat(GRAB | GLITTER)', r_hat[where & glitter, GRAB])}  (exact 999)")
    s0 = where & ~arrow
    scream = sum(p_hat[s0, SHOOT, o] for o in range(GAME_OVER) if o & Percept.SCREAM) if s0.any() else torch.zeros(0)
    print(f"    {stats('r_hat(SHOOT | no arrow)', r_hat[s0, SHOOT])}  (exact -1); "
          f"P_hat(SCREAM) mean {scream.mean() if len(scream) else float('nan'):.4f}")
    s1 = where & arrow
    print(f"    {stats('r_hat(SHOOT | arrow)', r_hat[s1, SHOOT])}  (exact -10)")
    for a, name in ((GRAB, "GRAB"), (SHOOT, "SHOOT")):
        noop = g0 if a == GRAB else s0
        exact_loop = -1.0 + GAMMA * loop_v[a][noop]
        learned_q = r_hat[noop, a] + GAMMA * (p_hat[noop, a] * e.next_values.double().cpu()[noop, a]).sum(-1)
        print(f"    no-op {name}: V_hat mean {v_hat[noop].mean():7.1f}; learned Q mean {learned_q.mean():7.1f}; "
              f"exact-reward loop Q (-1 + gamma V_hat(loop child)) mean {exact_loop.mean():7.1f}")

# Added after the first run (2026-10-14, not pre-registered): the idle V_hat exceeded the most any game can
# pay (999: the gold ends the game; every other reward is <= -1), and the agent stood on the gold without
# grabbing. Count it: games that perceived GLITTER while live and did not win, and at GLITTER steps the
# root's one-step Q(GRAB) against the best other action.
won = (rewards >= 998).any(1)
print(f"  games that perceived GLITTER while live: {int(saw_glitter.sum())}; of them not won: "
      f"{int((saw_glitter & ~won).sum())}; V_hat > 999 on {(v_hat > 999).double().mean():.3f} of live steps "
      f"(idle {(v_hat[idle] > 999).double().mean():.3f})")
if glitter.any():
    q = r_hat[glitter] + GAMMA * (p_hat[glitter] * e.next_values.double().cpu()[glitter]).sum(-1)
    other = torch.cat([q[:, :GRAB], q[:, GRAB + 1:]], 1).max(1).values
    print(f"  GLITTER steps {int(glitter.sum())}: one-step Q(GRAB) mean {q[:, GRAB].mean():7.1f} vs best other "
          f"{other.mean():7.1f}; GRAB the one-step argmax on {(q[:, GRAB] >= other).double().mean():.3f}")
