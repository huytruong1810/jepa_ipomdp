# ABSOLUTE PATH: reports/2026-10-19_eval_protocol/exact_planner_smoke.py
# ==============================================================================
# SMOKE AND BUDGET SCREEN OF THE EXACT-BELIEF ONLINE PLANNER (DEVELOPMENT GAMES ONLY)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question:
#    - What return does the exact-belief planner (planning/wumpus_exact_model.py: the agent's
#      own belief-tree search with the true rules, exact layout-mask beliefs and rollout
#      leaves) reach per simulation count and rollout count, and at what cost? The answer
#      fixes its configuration BEFORE any held-out play (PROTOCOL.md, "Frozen choices").
#
# 2. Usage:
#      uv run reports/2026-10-19_eval_protocol/exact_planner_smoke.py <sims> <leaf> <rollouts> [games]
#    Plays `games` (default 32) of the DEVELOPMENT set: the first games of analysis seed
#    12345's simulator stream (experiments/wumpus.py), the stream every development number
#    used, so the result is directly comparable to the heuristic and the learned planner on
#    the same layouts. Search settings follow conf/env/wumpus.yaml (c_puct 1.25, observation
#    prune 1e-3, greedy argmax Q). <leaf> is full_info, zero or rollout (planning/wumpus_exact_model.py,
#    section 4). Writes runs/wumpus/exact_planner/smoke_s<sims>_<leaf>_k<rollouts>_n<games>.npz.
#
# 3. Readout (no pass threshold; this is a budget screen):
#    - Mean discounted return, win / death / unfinished rates, action mix, seconds per
#      decision. The heuristic (risk 0.4) on the same games is printed beside it.
#    - Choice rule, written before the runs: take the cheapest configuration within one
#      standard error of the best return seen, subject to a 512-game play taking at most
#      ~12 h (it plays once per analysis seed, not per training seed). Record the choice and
#      the screen's numbers in planning/wumpus_exact_model.py's header.
# ==============================================================================

from pathlib import Path
import sys
import time

import numpy as np
import torch

from ipomdp.agents import PlanningAgent, WumpusReferenceAgent
from ipomdp.domain import WumpusSimulator, WumpusWorld, game_outcomes
from ipomdp.experiments.wumpus import WumpusAnalysisSettings
from ipomdp.planning import BeliefTreeSearch
from ipomdp.planning.wumpus_exact_model import ExactWumpusSearchModel
from ipomdp.training import discounted_returns, play_episodes, stream_seed

sims, leaf, rollouts = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
games = int(sys.argv[4]) if len(sys.argv) > 4 else 32
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
settings = WumpusAnalysisSettings()
MAX_STEPS, C_PUCT, PRUNE = 50, 1.25, 1e-3
world = WumpusWorld(dev)


class Progress:
    """Pass-through agent: a decision counter with timestamps, and every action recorded."""

    def __init__(self, actor):
        self.actor, self.batch_size, self.started, self.actions = actor, actor.batch_size, time.time(), []

    def reset(self, initial_observation):
        self.actor.reset(initial_observation)

    def update(self, action, observation):
        self.actor.update(action, observation)

    def act(self):
        action = self.actor.act()
        self.actions.append(action.clone())
        if len(self.actions) % 5 == 0:
            print(f"  [{time.strftime('%H:%M:%S')}] decision {len(self.actions)}; {time.time() - self.started:.0f} s",
                  flush=True)
        return action


def simulator() -> WumpusSimulator:
    # The analysis stream draws the layouts of all its rows at reset, so the first `games` rows of a
    # 512-row simulator are the development games; a `games`-row simulator would draw others.
    return WumpusSimulator(world, settings.episodes, MAX_STEPS, stream_seed(settings.seed, 1), dev)


class FirstRows:
    """Plays only the first `games` rows of the 512-row development stream; the rest take a fixed no-op."""

    def __init__(self, actor, games: int, total: int):
        self.actor, self.games, self.batch_size = actor, games, total

    def reset(self, initial_observation):
        self.actor.reset(initial_observation[:self.games])

    def update(self, action, observation):
        self.actor.update(action[:self.games], observation[:self.games])

    def act(self):
        action = torch.full((self.batch_size,), int(1), dtype=torch.int64, device=dev)    # TURN_LEFT elsewhere
        action[:self.games] = self.actor.act()
        return action


def summary(name: str, rewards: torch.Tensor, seconds: float | None = None) -> np.ndarray:
    returns = discounted_returns(rewards, world.discount).cpu().numpy()
    out = game_outcomes(rewards)
    cost = "" if seconds is None else f" | {seconds / MAX_STEPS:.1f} s per decision"
    print(f"{name}: return {returns.mean():8.2f} +- {returns.std() / len(returns) ** 0.5:6.2f} | win {out['win_rate']:.3f} "
          f"death {out['death_rate']:.3f} unfinished {out['unfinished_rate']:.3f}{cost}", flush=True)
    return returns


print(f"[{time.strftime('%H:%M:%S')}] exact planner: {sims} simulations, {leaf} leaves, {rollouts} rollouts, "
      f"{games} development games")
model = ExactWumpusSearchModel(world, games, sims, MAX_STEPS, leaf, rollouts, seed=stream_seed(settings.seed, 5))
planner = PlanningAgent(model, BeliefTreeSearch(model, sims, C_PUCT, 0.3, dirichlet_epsilon=0.0,
                                                seed=stream_seed(settings.seed, 2), observation_prune=PRUNE),
                        games, temperature=0.0, seed=stream_seed(settings.seed, 3), device=dev)
actor = Progress(planner)
episodes, _ = play_episodes(simulator(), FirstRows(actor, games, settings.episodes))
seconds = time.time() - actor.started
planner_returns = summary("exact planner", episodes.rewards[:games], seconds)
actions = torch.stack(actor.actions, 1)
live = episodes.rewards[:games] != 0.0
mix = torch.bincount(actions[live].flatten(), minlength=world.num_actions).double()
print("  action mix (live steps):", {a: round(float(m), 3) for a, m in zip(world.action_names, mix / mix.sum())})

reference, _ = play_episodes(simulator(), WumpusReferenceAgent(world, settings.episodes, 0.4))
reference_returns = summary("heuristic (risk 0.4), same games", reference.rewards[:games])
gap = planner_returns - reference_returns
print(f"paired gap planner - heuristic: {gap.mean():.2f} +- {gap.std() / len(gap) ** 0.5:.2f}")

out = Path("runs/wumpus/exact_planner")
out.mkdir(parents=True, exist_ok=True)
np.savez(out / f"smoke_s{sims}_{leaf}_k{rollouts}_n{games}.npz", returns=planner_returns, reference=reference_returns,
         actions=actions.cpu().numpy(), rewards=episodes.rewards[:games].cpu().numpy(), seconds=seconds)
