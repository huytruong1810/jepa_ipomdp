# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/q_localise_diag.py
# ==============================================================================
# WHICH PART OF THE LEARNED Q RANKS "TURN, THEN STEP INTO A SAFE CELL" TOO LOW: THE MODEL OR THE VALUE?
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-17, user: "go ahead" after macro_reach_diag.py read VOID):
#    - macro_reach_diag.py showed that at the planner's declined near-turn states (the exact-belief heuristic
#      turns towards a provably safe unvisited cell, or known gold, 2 actions away) the search already holds
#      TURN -> FORWARD and values it 29-45 below the chosen action. The learned Q has three learned parts:
#      the reward head R, the observation head P(o' | z, a) and the value V(z) at the frontier latents.
#    - Wumpus has an exact Bayes filter (layout enumeration), so the exact one-step model (R(b, a), P(o' | b, a),
#      successor beliefs, terminal outcomes) is available at the same states. Swapping exact parts in, one
#      at a time, while keeping the learned value at the filter's latents, localises the ranking error. The
#      branch picks the theory for the next method (HANDOFF 6.2):
#        MODEL  -> decision-aware / value-equivalent model learning (Grimm et al. 2020): the observation head
#                  is wrong exactly where it changes decisions (e.g. the value of information of a new cell).
#        VALUE  -> the value of the stepped-into latent is too low even under the exact model: a value-learning
#                  / representation-sufficiency problem (self-predictive representations, Ni et al. 2024).
#
# 2. Method:
#    - The 512 analysis games of best.pt of each runs/sweeps/wumpus_retrain seed, replayed exactly (return
#      must equal report.json); the declined distance-2 near-turn states are selected exactly as in
#      macro_reach_diag.py (and their count and leaf-only q1_chosen must equal its saved macro_reach.npz).
#    - At each state, a depth-2 expectimax with frontier values V^ (learned, at the filter's latents
#      z'' = filter(filter(z, a1, o1), a2, o2), the latents the planner itself builds):
#          Q2(a1)  = R(a1) + gamma * sum_o1 P(o1 | a1) * max_a2 [R(a2) + gamma * sum_o2 P(o2 | a1, o1, a2) * V(z'')]
#          Q_mac   = the same with a1 = the heuristic's turn and a2 fixed to FORWARD.
#      The readout statistic is the share of states with Q_mac > Q2(chosen) (the planner's action).
#    - Arms (which parts are exact; the value is always learned):
#        M    every part learned, as the search uses them (pruned at observation_prune and renormalised).
#        EP   exact P(o1), P(o2) (and the exact successor's terminal flag: an exact game-over child is worth 0),
#             learned rewards.
#        EPR  exact P and exact R: the exact one-step model with the learned value. (Learned P with exact R is
#             not defined: the learned branches include observations the exact belief rules out.)
#    - Descriptive extras: at the turned child (the exact turn outcome), learned vs exact P(o2 | FORWARD):
#      total variation, P(GAME_OVER), P(glitter); learned V at the stepped-into latents vs at the chosen child.
#    - Usage: uv run python <this> <run dir> 512; results to <run>/analysis/q_localise.npz.
#
# 3. Pre-registered readout (written 2026-10-17 before any run; 5 seeds, distance-2 states, share of
#    Q_mac > Q2(chosen)):
#    - MODEL: EPR - M >= +30 points on >= 3 of 5 seeds. (EP - M carrying most of it -> the observation head.)
#    - VALUE: EPR - M < +10 points on >= 4 of 5 seeds and EPR < 25% on >= 4 of 5 seeds.
#    - Otherwise MIXED: report the per-seed shares and gaps and decide with the user.
#    - Sanity (asserted): exact P sums to 1; exact P(GAME_OVER | turn, FORWARD) = 0 when the target is a
#      provably safe cell (the selection guarantees it); the replay and the macro_reach.npz cross-checks.
#
# 4. Result 2026-10-17 (runs/sweeps/wumpus_retrain seeds 0-4, best.pt, 512 games; replays = report.json and the
#    state selection = macro_reach.npz on every seed; logs runs/wumpus/q_localise_seed{0..4}.log, chain
#    runs/wumpus/q_localise_chain.sh; ~10 min per seed). **Readout: MODEL on 5 of 5 seeds.**
#    | Seed | States | M     | EP    | EPR   | EPR - M | EP - M | TV(P_fwd) | P(glitter) learned / exact | R(FWD) err mean |
#    | 0    | 1407   | 10.0% | 43.0% | 69.4% | +59     | +33    | 0.40      | 0.056 / 0.126              | -95             |
#    | 1    | 1446   | 14.4% | 41.5% | 47.9% | +34     | +27    | 0.47      | 0.034 / 0.118              | -39             |
#    | 2    | 1576   | 10.9% | 46.3% | 64.1% | +53     | +35    | 0.51      | 0.035 / 0.116              | -98             |
#    | 3    | 1865   | 11.2% | 62.2% | 80.9% | +70     | +51    | 0.43      | 0.028 / 0.126              | -107            |
#    | 4    | 2520   | 7.6%  | 57.7% | 68.3% | +61     | +50    | 0.50      | 0.042 / 0.123              | -42             |
#    (Shares of Q_mac > Q2(chosen); unfinished games alone give the same picture.)
#    - The observation head carries most of the error (EP - M is 56-85% of EPR - M). At the turned latent its
#      P(o' | FORWARD) is far from exact (TV median 0.40-0.51) and it predicts glitter (the gold, the reason to
#      enter a new cell) 2.3-4.5x too rarely: the information value of the step is missing from the backup.
#    - The rest is the reward head after the imagined turn: R(FORWARD) is exact at the median but its mean is
#      -39 to -107 (a minority of states carry a phantom death risk the exact belief rules out; learned
#      P(death) mean 0.05-0.11 vs exact 0). Root rewards are exact (median error 0.00 for every action).
#    - The learned value at the stepped-into latents is already 13-21 ABOVE the chosen children's on 4 of 5
#      seeds (seed 1: -2): the value is not the blocker.
#    - Caveat: a depth-2 offline ranking, not play (principle C3: only play counts). The next test is in play.
# ==============================================================================

import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from ipomdp.agents import PlanningAgent, WumpusReferenceAgent
from ipomdp.agents.wumpus_reference import _ZERO, _ahead, _first_actions
from ipomdp.domain import WumpusSimulator, game_outcomes
from ipomdp.domain.wumpus import GAME_OVER, NUM_CELLS, Percept, WumpusAction, WumpusState
from ipomdp.experiments.runs import load_trained_run
from ipomdp.experiments.wumpus import WumpusAnalysisSettings
from ipomdp.planning import BeliefTreeSearch
from ipomdp.training import discounted_returns, play_episodes, stream_seed

run_dir, n = Path(sys.argv[1]), int(sys.argv[2])
dev = torch.device("cuda")
settings = WumpusAnalysisSettings()
trained = load_trained_run(run_dir, dev)
world, run, cfg = trained.setup.domain, trained.run, trained.cfg
model = run.eval_agent.model
FWD, TURNS = int(WumpusAction.FORWARD), (int(WumpusAction.TURN_LEFT), int(WumpusAction.TURN_RIGHT))
NUM_A, NUM_O = world.num_actions, world.num_observations
PRUNE, GAMMA, CHUNK = cfg.mcts.observation_prune, model.discount, 128
GLITTER = np.array([bool(o & Percept.GLITTER) and o != GAME_OVER for o in range(NUM_O)])
planner = PlanningAgent(
    model, BeliefTreeSearch(model, cfg.mcts.num_simulations, cfg.mcts.c_puct, cfg.mcts.dirichlet_alpha,
                            dirichlet_epsilon=0.0, seed=stream_seed(settings.seed, 2),
                            observation_prune=cfg.mcts.observation_prune),
    n, temperature=0.0, seed=stream_seed(settings.seed, 3), device=dev)


def target_distance(cell: int, heading: int, safe: list[bool], is_target) -> int:
    """Fewest actions (FORWARD through safe cells, turns) until the hunter stands on a target cell (override_diag.py)."""
    if is_target(cell):
        return 0
    queue, seen = deque([(cell, heading, 0)]), {(cell, heading)}
    while queue:
        c, h, d = queue.popleft()
        ahead = _ahead(c, h)
        for nxt in ((ahead, h) if ahead is not None and safe[ahead] else None, (c, (h + 1) % 4), (c, (h + 3) % 4)):
            if nxt is None or nxt in seen:
                continue
            if nxt[0] != c and is_target(nxt[0]):
                return d + 1
            seen.add(nxt)
            queue.append((*nxt, d + 1))
    return 10 ** 6


def pruned(probs: np.ndarray) -> np.ndarray:
    """The search's branch weights over all |O| (planning/mcts.py section 6), along the last axis."""
    if PRUNE == 0.0:
        return probs
    kept = np.where(probs >= PRUNE, probs, 0.0)
    return kept / kept.sum(-1, keepdims=True)


class Recorder:
    """The greedy planner plus a shadow exact-belief heuristic; keeps (z, exact belief) at declined distance-2 near-turn states."""

    def __init__(self):
        self.batch_size = n
        self.shadow = WumpusReferenceAgent(world, n, 0.4)
        self.rows: list[dict] = []
        self.t, self.start = 0, time.time()

    def reset(self, initial_observation):
        planner.reset(initial_observation)
        self.shadow.reset(initial_observation)

    def update(self, action, observation):
        planner.update(action, observation)
        self.shadow.update(action, observation)

    def act(self):
        s = self.shadow
        m = s._tracker.marginals(s._consistent, s._own)
        heuristic = s.act().tolist()
        state = planner.state.clone()
        action = planner.act()
        chosen = action.tolist()
        pit, wumpus, gold = m.pit.cpu().tolist(), m.wumpus.cpu().tolist(), m.gold.cpu().tolist()
        alive, done = m.wumpus_alive.tolist(), m.done.tolist()
        for b in range(n):
            if done[b] or heuristic[b] not in TURNS or chosen[b] == heuristic[b]:
                continue
            cell, heading = int(m.cell[b]), int(m.heading[b])
            if gold[b][cell] > 1.0 - _ZERO:
                continue
            danger = [pit[b][c] + (wumpus[b][c] if alive[b] else 0.0) for c in range(NUM_CELLS)]
            safe = [d < _ZERO for d in danger]
            reachable = {c for c, _ in _first_actions(cell, heading, safe)} | {cell}
            known = [c for c in reachable if gold[b][c] > 1.0 - _ZERO]               # rule (b) prefers known gold
            goal = (lambda c: c in known) if known else (lambda c: safe[c] and not s._visited[b][c] and c in reachable)
            if not (known or any(goal(c) for c in range(NUM_CELLS))):
                continue
            if target_distance(cell, heading, safe, goal) != 2:
                continue
            root = planner.planner.roots[b]
            q1_chosen = root.edges[chosen[b]].reward + GAMMA * sum(
                p * c.leaf_value for p, c in zip(root.edges[chosen[b]].observation_probs, root.edges[chosen[b]].children))
            own = s._own
            self.rows.append({"game": b, "t": self.t, "chosen": chosen[b], "turn": heuristic[b], "z": state[b],
                              "layouts": torch.nonzero(s._consistent[b]).squeeze(1),
                              "own": (int(own.cell[b]), int(own.heading[b]), bool(own.arrow[b]),
                                      bool(own.wumpus_alive[b])),
                              "q1_chosen": q1_chosen})
        self.t += 1
        if self.t % 10 == 0:
            print(f"   [{time.strftime('%H:%M:%S')}] decision {self.t}/{cfg.env.max_steps}, "
                  f"{len(self.rows)} declined distance-2 states, {time.time() - self.start:.0f}s", flush=True)
        return action


def exact_outcomes(layouts: torch.Tensor, own: tuple, action: int) -> tuple[float, dict]:
    """Exact one-step model at a belief (uniform over `layouts`, hunter state `own`): E[r], {o: (P, layouts', own', done')}."""
    k = layouts.shape[0]
    full = lambda v, dtype: torch.full((k,), v, dtype=dtype, device=dev)  # noqa: E731
    state = WumpusState(layouts, full(own[0], torch.int64), full(own[1], torch.int64), full(own[2], torch.bool),
                        full(own[3], torch.bool), full(False, torch.bool))
    nxt, obs, reward = world.step_rules(state, full(action, torch.int64))
    out = {}
    for o in torch.unique(obs).tolist():
        sel = obs == o
        i = int(torch.nonzero(sel)[0])
        out[o] = (float(sel.double().mean()), layouts[sel],
                  (int(nxt.cell[i]), int(nxt.heading[i]), bool(nxt.arrow[i]), bool(nxt.wumpus_alive[i])), bool(nxt.done[i]))
    if abs(sum(v[0] for v in out.values()) - 1.0) > 1e-9:
        raise RuntimeError("Exact observation probabilities do not sum to 1.")
    return float(reward.double().mean()), out


def expand_np(latents: torch.Tensor):
    """model.expand, chunked; numpy rewards (N, A), probs (N, A, O), values (N, A, O) and the child latents on the GPU."""
    rs, ps, vs, zs = [], [], [], []
    for i in range(0, latents.shape[0], CHUNK):
        e = model.expand(latents[i:i + CHUNK])
        rs.append(e.rewards.double().cpu().numpy())
        ps.append(e.observation_probs.double().cpu().numpy())
        vs.append(e.next_values.double().cpu().numpy())
        zs.append(e.next_states)
    return np.concatenate(rs), np.concatenate(ps), np.concatenate(vs), torch.cat(zs)


def analyse(rows: list[dict]) -> dict[str, np.ndarray]:
    """Arms M, EP, EPR of section 2 for every row; returns per-row arrays."""
    out = {k: [] for k in ("q_mac_M", "q_cho_M", "q_mac_EP", "q_cho_EP", "q_mac_EPR", "q_cho_EPR", "q1_chosen_check",
                           "tv_fwd", "p_death_fwd_model", "p_glitter_fwd_model", "p_glitter_fwd_exact",
                           "v_step", "v_chosen", "dr_chosen", "dr_turn", "chosen", "dr_fwd2")}
    for start in range(0, len(rows), CHUNK):
        batch = rows[start:start + CHUNK]
        r1, p1, v1, z1 = expand_np(torch.stack([r["z"] for r in batch]))
        # Children needed: for a1 in {turn, chosen}, every o1 kept by the model or possible under the exact belief.
        jobs, child_z = [], []
        exact1 = []
        for i, row in enumerate(batch):
            ex = {a1: exact_outcomes(row["layouts"], row["own"], a1) for a1 in (row["turn"], row["chosen"])}
            exact1.append(ex)
            for a1 in ex:
                kept = set(np.flatnonzero(p1[i, a1] >= PRUNE).tolist()) | set(ex[a1][1])
                for o1 in sorted(kept):
                    jobs.append((i, a1, o1))
                    child_z.append(z1[i, a1, o1])
        r2, p2, v2, _ = expand_np(torch.stack(child_z))
        child = {job: j for j, job in enumerate(jobs)}
        for i, row in enumerate(batch):
            a_t, a_c = row["turn"], row["chosen"]
            ex = exact1[i]
            res = {}
            # Learned arm M: the search's own one-step quantities, two levels deep.
            for a1, macro in ((a_t, True), (a_c, False)):
                w = pruned(p1[i, a1])
                total = 0.0
                for o1 in np.flatnonzero(w > 0):
                    j = child[(i, a1, int(o1))]
                    inner = r2[j] + GAMMA * (pruned(p2[j]) * v2[j]).sum(-1)              # (A,)
                    total += w[o1] * (inner[FWD] if macro else inner.max())
                res[("M", macro)] = r1[i, a1] + GAMMA * total
            # Exact-P arms (EP: learned rewards, EPR: exact rewards).
            for a1, macro in ((a_t, True), (a_c, False)):
                exact_r1, branches = ex[a1]
                total = {"EP": 0.0, "EPR": 0.0}
                for o1, (p, lay, own1, done1) in branches.items():
                    if done1:
                        continue                                                         # exact terminal child: worth 0
                    j = child[(i, a1, o1)]
                    inner = {"EP": np.full(NUM_A, -np.inf), "EPR": np.full(NUM_A, -np.inf)}
                    for a2 in ([FWD] if macro else range(NUM_A)):
                        exact_r2, b2 = exact_outcomes(lay, own1, a2)
                        pv = sum(pp * (0.0 if d2 else v2[j, a2, o2]) for o2, (pp, _, _, d2) in b2.items())
                        inner["EP"][a2] = r2[j, a2] + GAMMA * pv
                        inner["EPR"][a2] = exact_r2 + GAMMA * pv
                    for arm in total:
                        total[arm] += p * (inner[arm][FWD] if macro else inner[arm].max())
                res[("EP", macro)] = r1[i, a1] + GAMMA * total["EP"]
                res[("EPR", macro)] = exact_r1 + GAMMA * total["EPR"]
            for arm in ("M", "EP", "EPR"):
                out[f"q_mac_{arm}"].append(res[(arm, True)])
                out[f"q_cho_{arm}"].append(res[(arm, False)])
            out["q1_chosen_check"].append(r1[i, a_c] + GAMMA * (pruned(p1[i, a_c]) * v1[i, a_c]).sum())
            # Descriptive: the turned child of the (deterministic) exact turn outcome, FORWARD from there.
            (o_t, (_, lay_t, own_t, _)), = ex[a_t][1].items()
            j = child[(i, a_t, o_t)]
            exact_r_fwd2, b2 = exact_outcomes(lay_t, own_t, FWD)
            out["dr_fwd2"].append(r2[j, FWD] - exact_r_fwd2)
            pe = np.zeros(NUM_O)
            for o2, (pp, _, _, _) in b2.items():
                pe[o2] = pp
            if pe[GAME_OVER] > 0.0:
                raise RuntimeError("Exact P(GAME_OVER | turn, FORWARD) > 0 at a provably safe distance-2 target.")
            pm = p2[j, FWD]
            out["tv_fwd"].append(0.5 * np.abs(pm - pe).sum())
            out["p_death_fwd_model"].append(pm[GAME_OVER])
            out["p_glitter_fwd_model"].append(pm[GLITTER].sum())
            out["p_glitter_fwd_exact"].append(pe[GLITTER].sum())
            out["v_step"].append(float((pe * v2[j, FWD]).sum()))
            out["v_chosen"].append(float((pruned(p1[i, a_c]) * v1[i, a_c]).sum()))
            out["dr_chosen"].append(r1[i, a_c] - ex[a_c][0])                         # learned - exact R(b, a)
            out["dr_turn"].append(r1[i, a_t] - ex[a_t][0])
            out["chosen"].append(a_c)
        print(f"   [{time.strftime('%H:%M:%S')}] analysed {min(start + CHUNK, len(rows))}/{len(rows)}", flush=True)
    return {k: np.array(v) for k, v in out.items()}


print(f"== {run_dir.name} best.pt (collection {trained.checkpoint_collection}), {n} games", flush=True)
rec = Recorder()
episodes, _ = play_episodes(WumpusSimulator(world, n, cfg.env.max_steps, stream_seed(settings.seed, 1), dev), rec)
returns = discounted_returns(episodes.rewards, world.discount).cpu().numpy()
print(f"  planner return {returns.mean():.1f}; outcomes "
      + str({k: round(float(v), 4) for k, v in game_outcomes(episodes.rewards).items()}), flush=True)
if n == settings.episodes:
    reported = json.loads((run_dir / "analysis" / "report.json").read_text())["returns"]["learned_planner"]
    if abs(reported["discounted_return"] - returns.mean()) > 1e-6:
        raise RuntimeError(f"Replay is not faithful: {returns.mean()} vs report.json {reported['discounted_return']}.")
    saved = np.load(run_dir / "analysis" / "macro_reach.npz")
    ref = saved["q1_chosen"][saved["distance"] == 2]
    mine = np.array([r["q1_chosen"] for r in rec.rows])
    if ref.shape != mine.shape or np.abs(ref - mine).max() > 1e-6:
        raise RuntimeError(f"State selection differs from macro_reach.npz: {ref.shape} vs {mine.shape}.")

res = analyse(rec.rows)
if np.abs(res["q1_chosen_check"] - np.array([r["q1_chosen"] for r in rec.rows])).max() > 1e-2:
    raise RuntimeError("Re-expansion does not reproduce the tree's leaf-only q1_chosen.")
rewards = episodes.rewards.double().cpu().numpy()
won, died = (rewards >= 998).any(1), (rewards <= -1000).any(1)
outcome = np.where(won, "win", np.where(died, "death", "unfinished"))[[r["game"] for r in rec.rows]]
for kind in ("all", "unfinished", "win"):
    sel = np.ones(len(outcome), bool) if kind == "all" else outcome == kind
    if not sel.any():
        continue
    parts = []
    for arm in ("M", "EP", "EPR"):
        gap = res[f"q_mac_{arm}"][sel] - res[f"q_cho_{arm}"][sel]
        parts.append(f"{arm} {(gap > 0).mean():.3f} (median gap {np.median(gap):+.1f})")
    print(f"  {kind} ({sel.sum()} states): Q_mac > Q2(chosen): " + "; ".join(parts), flush=True)
print(f"  turned child, FORWARD: TV(learned, exact) median {np.median(res['tv_fwd']):.3f}; learned P(death) median "
      f"{np.median(res['p_death_fwd_model']):.4f} mean {res['p_death_fwd_model'].mean():.4f} (exact 0); P(glitter) "
      f"learned mean {res['p_glitter_fwd_model'].mean():.3f} vs exact {res['p_glitter_fwd_exact'].mean():.3f}; "
      f"V at stepped latents - V at chosen children median {np.median(res['v_step'] - res['v_chosen']):+.1f}; "
      f"learned - exact R(FORWARD) there: mean {res['dr_fwd2'].mean():+.1f}, median {np.median(res['dr_fwd2']):+.2f}")
for a in range(NUM_A):
    sel = res["chosen"] == a
    if sel.any():
        print(f"  chosen {world.action_names[a]} ({sel.sum()}): learned - exact R(b, a): mean {res['dr_chosen'][sel].mean():+.2f}, "
              f"median {np.median(res['dr_chosen'][sel]):+.2f}; the turn's: mean {res['dr_turn'][sel].mean():+.2f}")
np.savez(run_dir / "analysis" / "q_localise.npz", outcome=outcome, **res)
