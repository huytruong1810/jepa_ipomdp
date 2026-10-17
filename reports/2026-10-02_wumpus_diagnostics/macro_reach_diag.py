# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/macro_reach_diag.py
# ==============================================================================
# WOULD A TURN-THEN-FORWARD MACRO ACTION REACH THE PLANNER'S DECISION?
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-17, HANDOFF 6.2.1, user's choice "macro actions"):
#    - On the retrain + double recipe (runs/sweeps/wumpus_retrain, 200 +- 44) about 67% of games end
#      unfinished. The exact-belief `near` override (override_diag.py) is worth +224 of the +300
#      navigation prize: turn towards an adjacent provably safe cell and step in. The candidate remedy is a
#      macro edge TURN_x + FORWARD in the search.
#    - Why it may be void (section 1b, "check that an intervention reaches the decision"): BeliefTreeSearch
#      (planning/mcts.py) is an expectimax tree. Every selected child is expanded, and an expanded node's
#      value is max_a Q over its own edges. Once the TURN child is expanded,
#          Q(TURN) = r_turn + gamma * sum_o P(o) * max_a Q(child_o, a)  >=  Q_mac = r_turn + gamma * sum_o P(o) * Q(child_o, FORWARD),
#      so a macro scored by the same heads and value can only change the root argmax where the TURN
#      child's observation branches are still unexpanded at decision time (their leaf V^ stands in for
#      the max), or through PUCT spending simulations differently. This script measures, on the frozen
#      networks and the analysis games, whether Q_mac would beat the planner's choice where it matters.
#
# 2. Method:
#    - best.pt of a run; the greedy planner exactly as analyze_wumpus_run plays it (its return must
#      equal report.json); a shadow exact-belief heuristic (agents/wumpus_reference.py, risk 0.4).
#    - A declined near-turn state: the heuristic's rule (a)/(b) applies with its target at most 3
#      actions away (override_diag.py's `near` condition), the heuristic's action is TURN_LEFT/RIGHT,
#      and the planner chose another action. Primary set: target distance exactly 2, where TURN + FORWARD
#      is precisely the heuristic's path into the target cell (the macro's case). Distance-3 states
#      (two turns, then FORWARD) are reported separately.
#    - At each such state, from the planner's root tree (planner.planner.roots[b]):
#        q_chosen, q_turn         root Q of the chosen action and of the heuristic's turn;
#        expanded                 P(o)-mass of the turn edge's children that are expanded;
#        q_mac                    r_turn + gamma * sum_o P(o) Q_o(FORWARD), with Q_o read from the tree
#                                 when the child is expanded, else from one model.expand of the child
#                                 latent (same observation pruning and renormalisation as the search);
#        q1_chosen, q1_mac        the same with only the leaf V^ of the first/second-level children (no
#                                 deeper search), to show whether the leaf value already ranks the step below.
#    - Sanity (asserted): q_mac <= q_turn + 1e-6 wherever every turn child is expanded (section 1's identity).
#    - Usage: uv run python <this> <run dir> <games>; rows saved to <run>/analysis/macro_reach.npz.
#
# 3. Pre-registered readout (written 2026-10-17, before any run; 5 seeds of runs/sweeps/wumpus_retrain):
#    - VOID: q_mac > q_chosen at < 10% of declined distance-2 near-turn states on >= 4 of 5 seeds. The
#      macro cannot reach the decision, so it is not built; the blocker is the model's value of the turned-
#      and-stepped latent, not search coverage. Return to the user for the next direction.
#    - REACHES: q_mac > q_chosen at >= 25% on >= 3 seeds -> the macro play screen (512 paired games per
#      seed; pass >= +50 on 3 of 5 with unfinished down and deaths up by at most 2 points).
#    - In between: the macro play screen on seeds 0-1 only, as a smoke.
#
# 4. Result 2026-10-17 (runs/sweeps/wumpus_retrain seeds 0-4, best.pt, 512 games each; every planner replay
#    reproduces report.json; logs runs/wumpus/macro_reach_seed{0..4}.log, chain runs/wumpus/macro_reach_chain.sh;
#    ~9 min per seed with ipomcp at load ~10): **VOID on 5 of 5 seeds.**
#    | Seed | Declined d=2 states | q_mac > q_chosen | gap median | leaf-only q1_mac > q1_chosen | q_turn - q_chosen median | turn children expanded |
#    | 0    | 1407                | 0.000            | -32.5      | 0.109                        | -10.0                    | 99.7%                  |
#    | 1    | 1446                | 0.000            | -44.6      | 0.153                        | -13.4                    | 100%                   |
#    | 2    | 1576                | 0.000            | -39.1      | 0.119                        | -8.4                     | 100%                   |
#    | 3    | 1865                | 0.000            | -38.1      | 0.136                        | -8.3                     | 100%                   |
#    | 4    | 2520                | 0.000            | -28.7      | 0.089                        | -14.7                    | 100%                   |
#    - Distance-3 states are the same (0.000 on every seed). Unfinished and won games do not differ.
#    - The turn child is expanded at essentially every declined state, so the tree already holds the
#      TURN -> FORWARD path and values it 29-45 below the chosen action (median). Even the leaf-only
#      one-step values prefer the macro only 9-15% of the time. The search does not miss the safe step:
#      the model's value of the turned-and-stepped latent ranks it below the no-op/turn alternatives
#      (the chosen actions are mostly SHOOT/GRAB no-ops and FORWARD elsewhere).
#    - Readout as pre-registered: the macro is not built and the play screen is not run. C8 sharpens: with a
#      good latent and a double value, the idling is the VALUE's ranking, not search coverage.
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
from ipomdp.domain.wumpus import NUM_CELLS, WumpusAction
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
PRUNE, GAMMA, CHUNK = cfg.mcts.observation_prune, model.discount, 2048
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
    """The search's branch weights over all |O| (planning/mcts.py section 6): keep P >= prune, renormalise."""
    if PRUNE == 0.0:
        return probs
    kept = np.where(probs >= PRUNE, probs, 0.0)
    return kept / kept.sum()


def forward_q(states: list[torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    """Q(s, FORWARD) by one expansion of unexpanded nodes, chunked; returns (Q, its leaf-only Q), both (N,)."""
    out = []
    for i in range(0, len(states), CHUNK):
        e = model.expand(torch.stack(states[i:i + CHUNK]))
        r = e.rewards[:, FWD].double().cpu().numpy()
        p = np.stack([pruned(row) for row in e.observation_probs[:, FWD].double().cpu().numpy()])
        out.append(r + GAMMA * (p * e.next_values[:, FWD].double().cpu().numpy()).sum(-1))
    q = np.concatenate(out) if out else np.zeros(0)
    return q, q                                   # an unexpanded node's FORWARD Q is already leaf-only


def leaf_q(edge) -> float:
    """r + gamma * sum_o P(o) * leaf V^ of the edge's children (no deeper search)."""
    return edge.reward + GAMMA * sum(p * c.leaf_value for p, c in zip(edge.observation_probs, edge.children))


class Recorder:
    """Drives the planner and a shadow exact-belief heuristic on the same history; records declined near-turn states."""

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
        action = planner.act()
        chosen = action.tolist()
        pit, wumpus, gold = m.pit.cpu().tolist(), m.wumpus.cpu().tolist(), m.gold.cpu().tolist()
        alive, done = m.wumpus_alive.tolist(), m.done.tolist()
        pending, pending_states, first_new = [], [], len(self.rows)
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
            distance = target_distance(cell, heading, safe, goal)
            if distance > 3:
                continue
            root = planner.planner.roots[b]
            turn = root.edges[heuristic[b]]
            row = {"game": b, "t": self.t, "distance": distance, "chosen": chosen[b], "turn": heuristic[b],
                   "q_chosen": root.edges[chosen[b]].q, "q_turn": turn.q,
                   "q1_chosen": leaf_q(root.edges[chosen[b]]),
                   "expanded": float(sum(p for p, c in zip(turn.observation_probs, turn.children) if c.edges is not None)),
                   "probs": turn.observation_probs, "q_o": [None] * len(turn.children), "q1_o": [None] * len(turn.children)}
            for k, child in enumerate(turn.children):
                if child.edges is not None:
                    row["q_o"][k] = child.edges[FWD].q
                    row["q1_o"][k] = leaf_q(child.edges[FWD])
                else:
                    pending.append((row, k))
                    pending_states.append(child.state)
            self.rows.append(row)
        q, q1 = forward_q(pending_states)
        for (row, k), qk, q1k in zip(pending, q, q1):
            row["q_o"][k], row["q1_o"][k] = float(qk), float(q1k)
        for row in self.rows[first_new:]:
            r_turn = planner.planner.roots[row["game"]].edges[row["turn"]].reward
            row["q_mac"] = r_turn + GAMMA * float(np.dot(row["probs"], row["q_o"]))
            row["q1_mac"] = r_turn + GAMMA * float(np.dot(row["probs"], row["q1_o"]))
            if row["expanded"] > 1.0 - 1e-9 and row["q_mac"] > row["q_turn"] + 1e-6:
                raise RuntimeError(f"Expectimax identity violated: q_mac {row['q_mac']} > q_turn {row['q_turn']}.")
            del row["probs"], row["q_o"], row["q1_o"]
        self.t += 1
        if self.t % 5 == 0:
            print(f"   [{time.strftime('%H:%M:%S')}] decision {self.t}/{cfg.env.max_steps}, "
                  f"{len(self.rows)} declined near-turn states, {time.time() - self.start:.0f}s", flush=True)
        return action


print(f"== {run_dir.name} best.pt (collection {trained.checkpoint_collection}), {n} games", flush=True)
rec = Recorder()
episodes, _ = play_episodes(WumpusSimulator(world, n, cfg.env.max_steps, stream_seed(settings.seed, 1), dev), rec)
returns = discounted_returns(episodes.rewards, world.discount).cpu().numpy()
print(f"  planner return {returns.mean():.1f}; outcomes "
      + str({k: round(float(v), 4) for k, v in game_outcomes(episodes.rewards).items()}))
if n == settings.episodes:
    reported = json.loads((run_dir / "analysis" / "report.json").read_text())["returns"]["learned_planner"]
    if abs(reported["discounted_return"] - returns.mean()) > 1e-6:
        raise RuntimeError(f"Replay is not faithful: {returns.mean()} vs report.json {reported['discounted_return']}.")

rewards = episodes.rewards.double().cpu().numpy()
won, died = (rewards >= 998).any(1), (rewards <= -1000).any(1)
outcome = np.where(won, "win", np.where(died, "death", "unfinished"))
names = world.action_names
cols = {k: np.array([r[k] for r in rec.rows]) for k in rec.rows[0]} if rec.rows else {}
for label, mask_fn in (("distance 2 (primary)", lambda c: c["distance"] == 2),
                       ("distance 3", lambda c: c["distance"] == 3)):
    if not cols:
        break
    mask = mask_fn(cols)
    for kind in ("all", "unfinished", "win"):
        sel = mask & ((outcome[cols["game"]] == kind) if kind != "all" else True)
        if not sel.any():
            continue
        gap, gap1 = cols["q_mac"][sel] - cols["q_chosen"][sel], cols["q1_mac"][sel] - cols["q1_chosen"][sel]
        mix = {names[a]: round(float((cols["chosen"][sel] == a).mean()), 3) for a in range(len(names))}
        print(f"  {label}, {kind} games: {sel.sum()} states; q_mac > q_chosen {(gap > 0).mean():.3f} "
              f"(gap median {np.median(gap):+.2f}); leaf-only q1_mac > q1_chosen {(gap1 > 0).mean():.3f} "
              f"(median {np.median(gap1):+.2f}); q_turn - q_chosen median {np.median(cols['q_turn'][sel] - cols['q_chosen'][sel]):+.2f}; "
              f"turn children fully expanded {(cols['expanded'][sel] > 1 - 1e-9).mean():.3f}; chosen {mix}", flush=True)
np.savez(run_dir / "analysis" / "macro_reach.npz", outcome=outcome, **cols)
