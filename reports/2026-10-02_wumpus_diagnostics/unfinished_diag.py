# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/unfinished_diag.py
# ==============================================================================
# WHY THE GREEDY PLANNER'S GAMES END UNFINISHED
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-07, current agent u64_store_eval_eps_decoder_k5_seed{0,1,2}):
#    - About 70% of the learned agent's games are still running after 50 steps, against 29% for
#      the exact-belief heuristic (agents/wumpus_reference.py, risk 0.4). Deaths are rare. Is the
#      agent (i) idle or looping while the exact belief still offers a provably safe unvisited
#      cell (the heuristic's rule b would walk there), or (ii) stuck behind risk, with no safe
#      cell left (where the heuristic also waits or gambles)? And does a flat or overestimated
#      learned Q make idling look as good as exploring?
#
# 2. Method:
#    - The greedy planner is built exactly as in experiments/wumpus.py (analyze_wumpus_run) with
#      the same analysis seed, so with <games> = 512 its games and outcomes are the ones in
#      <run>/analysis/report.json (a built-in check that the replay is faithful).
#    - A shadow WumpusReferenceAgent receives the same actions and percepts. Its exact
#      enumeration filter gives the hunter's true cell and heading and the hazard marginals at
#      every step. Rule b's target set (provably safe, unvisited, reachable through safe cells)
#      is recomputed from them, and the heuristic's own choice is recorded for agreement.
#    - At every live step: the planner's root Q (search statistics), V_hat(z) (acting value head,
#      codec mean) and the realised discounted return-to-go.
#    - Mechanism check: if V_hat is flat over live states, Q(FORWARD) - Q(no-op) is about
#      -P(death | z, FORWARD) * (1000 + gamma * V_hat), so ANY nonzero predicted risk makes waiting
#      better, even in front of a provably safe cell. At the steps of unfinished games where the
#      cell ahead is provably safe and unvisited, the script reports the head's P(GAME_OVER | z,
#      FORWARD) and Q(FORWARD) - max Q, and splits the one-step learned Q(FORWARD) - Q(chosen
#      action) (SearchModel.expand) into its reward and gamma * E[V_hat(next)] parts.
#    - Usage: uv run python <this> <run dir> <games>. Use a small <games> as a smoke check.
#
# 3. Result 2026-10-07 (best.pt of seeds 0/1/2, 512 games each; outcomes reproduce report.json exactly):
#    - Unfinished games (67-71%): the hunter visits 3.2-3.6 cells, finds its last new cell around step
#      11-14 and then idles for ~35-38 steps. The idle steps are mostly no-ops: SHOOT without the arrow
#      (-1, nothing happens; 72% / 51% / 32% of those steps), GRAB on an empty cell and turns.
#    - It is not only stuck behind risk: at the last step a provably safe, unvisited, reachable cell
#      still existed in 47% / 47% / 50% of unfinished games (the heuristic's rule b would walk there).
#    - Facing such a cell (2,217-2,712 steps per seed), the planner steps in only 29% / 31% / 42% of the
#      time. When it does not, the head's P(death | FORWARD) is negligible (median 0.0008-0.0019), the
#      reward part of one-step Q(FORWARD) - Q(chosen) is ~0, and the whole deficit is gamma *
#      E[V_hat(next)]: median -15.6 / -17.3 / -15.2 (means -35 / -90 / -42). The value head rates the
#      latent after stepping into a new safe cell BELOW the latent after a no-op. A minority of these
#      steps also show a head that wrongly fears death in a safe cell (mean P(death) 0.07-0.17).
#    - V_hat at live states of unfinished games is 630-690 against a realised return-to-go of -22.
#      The window is a truncation and values bootstrap through it (domain/wumpus.py, section 1), so a
#      wasted step costs only about 1 + (1 - gamma) * V_hat ~ 7.5 under the training objective, less
#      than the V_hat errors between neighbouring latents. Exploration progress is not valued.
# ==============================================================================

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ipomdp.agents import PlanningAgent, WumpusReferenceAgent
import torch.nn.functional as F

from ipomdp.agents.wumpus_reference import _ZERO, _ahead, _first_actions
from ipomdp.domain import WumpusSimulator, game_outcomes
from ipomdp.domain.wumpus import GAME_OVER, NUM_CELLS, WumpusAction
from ipomdp.experiments.runs import load_trained_run
from ipomdp.experiments.wumpus import WumpusAnalysisSettings
from ipomdp.planning import BeliefTreeSearch
from ipomdp.training import play_episodes, stream_seed

run_dir, n = Path(sys.argv[1]), int(sys.argv[2])
dev = torch.device("cuda")
settings = WumpusAnalysisSettings()
trained = load_trained_run(run_dir, dev)
world, run, cfg = trained.setup.domain, trained.run, trained.cfg
model = run.eval_agent.model
planner = PlanningAgent(
    model, BeliefTreeSearch(model, cfg.mcts.num_simulations, cfg.mcts.c_puct, cfg.mcts.dirichlet_alpha,
                            dirichlet_epsilon=0.0, seed=stream_seed(settings.seed, 2),
                            observation_prune=cfg.mcts.observation_prune),
    n, temperature=0.0, seed=stream_seed(settings.seed, 3), device=dev)


class Recorder:
    """Drives the planner and a shadow exact-belief heuristic on the same history; records each step."""

    def __init__(self):
        self.batch_size = n
        self.shadow = WumpusReferenceAgent(world, n, 0.4)
        self.rows: list[dict] = []

    def reset(self, initial_observation):
        planner.reset(initial_observation)
        self.shadow.reset(initial_observation)

    def update(self, action, observation):
        planner.update(action, observation)
        self.shadow.update(action, observation)

    def act(self):
        with torch.no_grad():
            v_hat = run.codec.mean(run.acting_value_head(planner.state).float()).cpu().numpy()
            forward = F.one_hot(torch.full((n,), int(WumpusAction.FORWARD), device=dev), world.num_actions).float()
            p_death = F.softmax(model.observation_head(planner.state, forward).float(), -1)[:, GAME_OVER].cpu().numpy()
            expansion = model.expand(planner.state)
            one_step_r = expansion.rewards.double().cpu().numpy()                                   # (B, A)
            one_step_v = (world.discount * (expansion.observation_probs * expansion.next_values).sum(-1)).double().cpu().numpy()
        s = self.shadow
        m = s._tracker.marginals(s._consistent, s._own)
        heuristic = s.act().tolist()
        action = planner.act()
        q = planner.planner.statistics.q_values
        pit, wumpus, alive = m.pit.cpu().tolist(), m.wumpus.cpu().tolist(), m.wumpus_alive.tolist()
        for b in range(n):
            if m.done[b]:
                self.rows.append(None)
                continue
            danger = [pit[b][c] + (wumpus[b][c] if alive[b] else 0.0) for c in range(NUM_CELLS)]
            safe = [d < _ZERO for d in danger]
            first = _first_actions(int(m.cell[b]), int(m.heading[b]), safe)
            reachable = {c for c, _ in first} | {int(m.cell[b])}
            target = any(safe[c] and not s._visited[b][c] and c in reachable for c in range(NUM_CELLS))
            qs = np.sort(q[b])[::-1]
            ahead = _ahead(int(m.cell[b]), int(m.heading[b]))
            safe_ahead = ahead is not None and safe[ahead] and not s._visited[b][ahead]
            self.rows.append({"cell": int(m.cell[b]), "target": target, "heuristic": heuristic[b],
                              "q_gap": float(qs[0] - qs[1]), "q_max": float(qs[0]), "v_hat": float(v_hat[b]),
                              "safe_ahead": safe_ahead, "p_death": float(p_death[b]),
                              "r1": one_step_r[b], "v1": one_step_v[b],
                              "q_fwd": float(q[b][int(WumpusAction.FORWARD)] - qs[0])})
        return action


rec = Recorder()
episodes, _ = play_episodes(WumpusSimulator(world, n, cfg.env.max_steps, stream_seed(settings.seed, 1), dev), rec)
T = episodes.actions.shape[1]
steps = [[rec.rows[t * n + b] for t in range(T)] for b in range(n)]
actions = episodes.actions.cpu().numpy()
rewards = episodes.rewards.double().cpu().numpy()
to_go = np.zeros_like(rewards)
acc = np.zeros(n)
for t in reversed(range(T)):
    acc = rewards[:, t] + world.discount * acc
    to_go[:, t] = acc
won, died = (rewards >= 998).any(1), (rewards <= -1000).any(1)
outcome = np.where(won, "win", np.where(died, "death", "unfinished"))
print(f"== {run_dir.name} best.pt (collection {trained.checkpoint_collection}), {n} games")
print("  outcomes:", {k: round(float(v), 4) for k, v in game_outcomes(episodes.rewards).items()})

names = world.action_names
for kind in ("win", "death", "unfinished"):
    games = np.flatnonzero(outcome == kind)
    if len(games) == 0:
        continue
    live = [(b, t) for b in games for t in range(T) if steps[b][t] is not None]
    mix = Counter(int(actions[b, t]) for b, t in live)
    cells = [len({r["cell"] for r in steps[b] if r is not None}) for b in games]
    last_new = []
    for b in games:
        seen, last = set(), 0
        for t, r in enumerate(steps[b]):
            if r is not None and r["cell"] not in seen:
                seen.add(r["cell"]); last = t
        last_new.append(last)
    print(f"-- {kind}: {len(games)} games, {len(live)} live steps")
    print("   action mix: " + ", ".join(f"{names[a]} {mix[a] / len(live):.3f}" for a in range(len(names))))
    print(f"   distinct cells visited: mean {np.mean(cells):.2f}; step of last new cell: mean {np.mean(last_new):.1f}")
    target = np.array([steps[b][t]["target"] for b, t in live])
    agree = np.array([steps[b][t]["heuristic"] == actions[b, t] for b, t in live])
    gap = np.array([steps[b][t]["q_gap"] for b, t in live])
    v = np.array([steps[b][t]["v_hat"] for b, t in live])
    g = np.array([to_go[b, t] for b, t in live])
    print(f"   steps with a safe unvisited reachable cell (exact): {target.mean():.3f}; "
          f"agrees with heuristic there {agree[target].mean() if target.any() else float('nan'):.3f}, "
          f"elsewhere {agree[~target].mean() if (~target).any() else float('nan'):.3f}")
    print(f"   root Q gap (best - 2nd): median {np.median(gap):.2f}, <1: {(gap < 1).mean():.3f}, <5: {(gap < 5).mean():.3f}")
    print(f"   V_hat mean {v.mean():.1f} vs realised return-to-go {g.mean():.1f}")
    if kind == "unfinished":
        final_target = np.array([steps[b][T - 1]["target"] for b in games])
        print(f"   at the last step a safe unvisited reachable cell still existed in {final_target.mean():.3f} of games")
        idle = np.array([T - 1 - lt for lt in last_new])
        print(f"   steps since the last new cell at the end: mean {idle.mean():.1f}, median {np.median(idle):.0f}")
        tail = [(b, t) for b, t in live if t > last_new[list(games).index(b)]]
        tmix = Counter(int(actions[b, t]) for b, t in tail)
        if tail:
            print("   action mix after the last new cell: "
                  + ", ".join(f"{names[a]} {tmix[a] / len(tail):.3f}" for a in range(len(names))))
            tt = np.array([steps[b][t]["target"] for b, t in tail])
            print(f"   ... of those steps, a safe target existed at {tt.mean():.3f}")

    sa = [(b, t) for b, t in live if steps[b][t]["safe_ahead"]]
    if sa:
        took = np.array([actions[b, t] == int(WumpusAction.FORWARD) for b, t in sa])
        pd = np.array([steps[b][t]["p_death"] for b, t in sa])
        qf = np.array([steps[b][t]["q_fwd"] for b, t in sa])
        print(f"   facing a provably safe unvisited cell: {len(sa)} steps, FORWARD taken {took.mean():.3f}")
        for label, mask in (("taken", took), ("not taken", ~took)):
            if mask.any():
                print(f"     FORWARD {label}: head P(death|FORWARD) median {np.median(pd[mask]):.4f} mean {pd[mask].mean():.4f}; "
                      f"Q(FORWARD) - max Q median {np.median(qf[mask]):.2f}")
        nt = [(b, t) for (b, t), k in zip(sa, took) if not k]
        if nt:
            fw = int(WumpusAction.FORWARD)
            dr = np.array([steps[b][t]["r1"][fw] - steps[b][t]["r1"][actions[b, t]] for b, t in nt])
            dv = np.array([steps[b][t]["v1"][fw] - steps[b][t]["v1"][actions[b, t]] for b, t in nt])
            print(f"     one-step Q(FORWARD) - Q(chosen) when not taken: reward part median {np.median(dr):.2f}, "
                  f"gamma*E[V_hat next] part median {np.median(dv):.2f} (mean {dv.mean():.2f})")
