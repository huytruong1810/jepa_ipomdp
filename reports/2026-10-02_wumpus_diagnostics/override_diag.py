# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/override_diag.py
# ==============================================================================
# HOW MUCH OF THE GAP TO THE HEURISTIC IS IDLING? PERSISTENT EXACT-BELIEF OVERRIDES
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-11, after counterfactual_diag.py):
#    - One forced FORWARD at a declined safe step pays off on average in unfinished games, but the
#      planner re-declines within 5 steps about half the time, so a one-step fix is not enough. What is
#      the prize of removing the idling *persistently*, and does it close the 391-point gap to the
#      exact-belief heuristic? That bounds what a commitment or option-level remedy could gain.
#
# 2. Method:
#    - The analysis games of analyze_wumpus_run (same streams; arm "planner" must equal report.json),
#      played by three policies on the acting copies of best.pt. A shadow exact-belief heuristic
#      (agents/wumpus_reference.py, risk 0.4) tracks every game:
#        planner   the greedy planner alone;
#        forward   the planner, but FORWARD whenever the cell ahead is provably safe and unvisited
#                  (unfinished_diag.py's `safe_ahead`);
#        navigate  the planner, but the heuristic's action whenever its rule (a) or (b) applies: grab
#                  certain gold, or walk to known gold / the nearest provably safe unvisited reachable
#                  cell. Elsewhere (shooting, gambling on risk, waiting) the planner decides.
#        near      as navigate, but only when that target is at most 3 actions away (at most two turns
#                  and one step into it): the part a short macro-action could cover. navigate - near is
#                  the part that needs walking back through visited cells.
#    - Overrides use privileged exact beliefs; they are diagnostics, never agents to report as results.
#    - Pre-registered readout:
#      - `navigate` within about 100 of the heuristic (>= ~470): the gap is idling and navigation, and an
#        option-level / commitment remedy is worth building.
#      - `navigate` far below the heuristic: the rest of the gap is in the risk decisions (rules c-e).
#      - `forward` - `planner` < +50: a "step into safe cells ahead" remedy is not worth building.
#    - Usage: uv run python <this> <run dir> <games> [comma-separated modes]; arms merge into
#      <run>/analysis/override.npz (the planner arm is required for the paired differences).
#
# 3. Result 2026-10-11 (current agent, best.pt of seeds 0-4, 512 games each; the planner arm reproduces
#    report.json; logs <run>.override_diag.log and <run>.override_near_diag.log; runs/wumpus/
#    override_chain.sh, override_near_chain.sh). Means over 5 seeds, Student-t 95% intervals, heuristic 573:
#    | Arm      | Return       | Paired with planner |
#    | planner  | 182 +- 62    |                     |
#    | forward  | 190 +- 74    | +8 +- 24 (n.s.)     |
#    | near     | 406 +- 89    | +224 +- 75          |
#    | navigate | 482 +- 39    | +300 +- 42          |
#    - Readout: navigate lies within 91 of the heuristic, so idling and navigation explain about 77% of
#      the 391-point gap; the rest is risk decisions (seed 3 keeps its 13.5% deaths under navigate).
#    - Three quarters of the prize (near, +224 of +300; navigate - near +76 +- 55) is a target at most 3
#      actions away: turn towards an adjacent provably safe cell and step in. Forcing FORWARD only when
#      already facing one gains nothing (its counterfactual gain in unfinished games is cancelled by
#      won games where the planner was heading for the gold, counterfactual_diag.py). The missing skill is
#      the turn that sets up a safe step, a short multi-step credit problem, not a one-step action gap.
# ==============================================================================

import json
import sys
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
FWD = int(WumpusAction.FORWARD)


def target_distance(cell: int, heading: int, safe: list[bool], is_target) -> int:
    """Fewest actions (FORWARD through safe cells, turns) until the hunter stands on a target cell."""
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


class OverrideAgent:
    """The greedy planner of analyze_wumpus_run, overridden on an exact-belief condition (section 2)."""

    def __init__(self, mode: str):
        self.batch_size, self.mode = n, mode
        self.planner = PlanningAgent(
            model, BeliefTreeSearch(model, cfg.mcts.num_simulations, cfg.mcts.c_puct, cfg.mcts.dirichlet_alpha,
                                    dirichlet_epsilon=0.0, seed=stream_seed(settings.seed, 2),
                                    observation_prune=cfg.mcts.observation_prune),
            n, temperature=0.0, seed=stream_seed(settings.seed, 3), device=dev)
        self.shadow = WumpusReferenceAgent(world, n, 0.4)
        self.overridden = 0

    def reset(self, initial_observation):
        self.planner.reset(initial_observation)
        self.shadow.reset(initial_observation)

    def update(self, action, observation):
        self.planner.update(action, observation)
        self.shadow.update(action, observation)

    def act(self):
        action = self.planner.act()
        if self.mode == "planner":
            return action
        s = self.shadow
        m = s._tracker.marginals(s._consistent, s._own)
        pit, wumpus, gold = m.pit.cpu().tolist(), m.wumpus.cpu().tolist(), m.gold.cpu().tolist()
        alive, done = m.wumpus_alive.tolist(), m.done.tolist()
        heuristic = s.act().tolist()
        override = action.clone()
        for b in range(n):
            if done[b]:
                continue
            cell, heading = int(m.cell[b]), int(m.heading[b])
            danger = [pit[b][c] + (wumpus[b][c] if alive[b] else 0.0) for c in range(NUM_CELLS)]
            safe = [d < _ZERO for d in danger]
            if self.mode == "forward":
                ahead = _ahead(cell, heading)
                hit = ahead is not None and safe[ahead] and not s._visited[b][ahead]
                choice = FWD
            else:
                first = _first_actions(cell, heading, safe)
                reachable = {c for c, _ in first} | {cell}
                hit = gold[b][cell] > 1.0 - _ZERO or any(
                    c in reachable and (gold[b][c] > 1.0 - _ZERO or (safe[c] and not s._visited[b][c]))
                    for c in range(NUM_CELLS))
                if hit and self.mode == "near" and gold[b][cell] <= 1.0 - _ZERO:
                    known = [c for c in reachable if gold[b][c] > 1.0 - _ZERO]   # rule (b) prefers known gold
                    goal = (lambda c: c in known) if known else (lambda c: safe[c] and not s._visited[b][c])
                    hit = target_distance(cell, heading, safe, goal) <= 3
                choice = heuristic[b]
            if hit:
                override[b] = choice
                self.overridden += int(choice != int(action[b]))
        return override


print(f"== {run_dir.name} best.pt (collection {trained.checkpoint_collection}), {n} games")
results = {}
modes = sys.argv[3].split(",") if len(sys.argv) > 3 else ["planner", "forward", "navigate", "near"]
for mode in modes:
    agent = OverrideAgent(mode)
    episodes, _ = play_episodes(WumpusSimulator(world, n, cfg.env.max_steps, stream_seed(settings.seed, 1), dev), agent)
    returns = discounted_returns(episodes.rewards, world.discount).cpu().numpy()
    out = {k: float(v) for k, v in game_outcomes(episodes.rewards).items()}
    results[mode] = returns
    print(f"  {mode:8s}: return {returns.mean():7.1f} ± {returns.std(ddof=1) / np.sqrt(n):5.1f}  win {out['win_rate']:.3f} "
          f"death {out['death_rate']:.3f} unfinished {out['unfinished_rate']:.3f}  overridden actions {agent.overridden}",
          flush=True)
    if mode == "planner" and n == settings.episodes:
        reported = json.loads((run_dir / "analysis" / "report.json").read_text())["returns"]["learned_planner"]
        if abs(reported["discounted_return"] - returns.mean()) > 1e-6:
            raise RuntimeError(f"Replay is not faithful: {returns.mean()} vs report.json {reported['discounted_return']}.")
for mode in [m for m in modes if m != "planner"]:
    d = results[mode] - results["planner"]
    print(f"  {mode} - planner (paired games): {d.mean():+.1f} ± {1.96 * d.std(ddof=1) / np.sqrt(n):.1f}")
out_file = run_dir / "analysis" / "override.npz"
np.savez(out_file, **({**dict(np.load(out_file)), **results} if out_file.exists() else results))
