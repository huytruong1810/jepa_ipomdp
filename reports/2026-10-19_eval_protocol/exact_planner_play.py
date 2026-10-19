# ABSOLUTE PATH: reports/2026-10-19_eval_protocol/exact_planner_play.py
# ==============================================================================
# ONE PLAY OF THE FROZEN EXACT-BELIEF PLANNER ON AN ANALYSIS SEED'S 512 GAMES
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question:
#    - The exact-belief planner is a reference (PROTOCOL.md, "References"): it plays once per
#      ANALYSIS seed, not per training seed, because nothing in it is trained. Its per-game
#      rows are paired game by game with every training seed's games.npz of the same analysis
#      seed, which is what sweep.py --exact-planner-games aggregates (experiments/aggregate.py,
#      section 6).
#
# 2. Usage:
#      uv run reports/2026-10-19_eval_protocol/exact_planner_play.py <analysis_seed>
#    Writes runs/references/wumpus_exact_planner/seed<analysis_seed>/chunk<k>.npz per 128 games
#    and, once all four exist, games.npz with keys "exact_planner.return" / ".won" / ".died" and
#    "reference_risk0.4.return" (the heuristic on the same rows: the pairing check that
#    aggregate_games runs against every training seed's games.npz), plus "analysis_seed".
#    Re-running skips finished chunks, so an interrupted play resumes per chunk.
#
# 3. Configuration (frozen on development games; planning/wumpus_exact_model.py, header 4):
#    full_info leaves, 1000 simulations, no rollouts, c_puct 1.25, observation prune 1e-3,
#    greedy argmax Q, the analysis stream of experiments/wumpus.py (stream_seed(seed, 1) for the
#    512-row simulator). Planner seeds follow the smoke script (streams 5, 2, 3 of the analysis
#    seed); the full_info leaf and greedy choice make them nearly irrelevant.
#    - 128 roots per chunk, the size the smoke measured (193 s per decision, ~2.7 h per chunk,
#      memory known). The other rows of the 512-row simulator take a fixed TURN_LEFT: layouts
#      are drawn for every row at reset and the rules are deterministic per row, so row i's game
#      does not depend on what the other rows do (the smoke's FirstRows argument).
#
# 4. Blindness:
#    - On a held-out analysis seed this script prints progress only, never a return: the number
#      is read once, in the registered aggregate of claim C-a (PROTOCOL.md).
# ==============================================================================

from pathlib import Path
import sys
import time

import numpy as np
import torch

from ipomdp.agents import PlanningAgent, WumpusReferenceAgent
from ipomdp.domain import WumpusSimulator, WumpusWorld, game_outcome_flags
from ipomdp.experiments.wumpus import WumpusAnalysisSettings
from ipomdp.planning import BeliefTreeSearch
from ipomdp.planning.wumpus_exact_model import ExactWumpusSearchModel
from ipomdp.training import discounted_returns, play_episodes, stream_seed

SIMS, LEAF, ROLLOUTS, C_PUCT, PRUNE = 1000, "full_info", 0, 1.25, 1e-3
MAX_STEPS, CHUNK_ROWS, RISK = 50, 128, 0.4

seed = int(sys.argv[1])
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
settings = WumpusAnalysisSettings(seed=seed)
world = WumpusWorld(dev)
out = Path("runs/references/wumpus_exact_planner") / f"seed{seed}"
out.mkdir(parents=True, exist_ok=True)
if settings.episodes % CHUNK_ROWS:
    raise ValueError(f"{settings.episodes} games do not split into chunks of {CHUNK_ROWS}.")


def simulator() -> WumpusSimulator:
    return WumpusSimulator(world, settings.episodes, MAX_STEPS, stream_seed(settings.seed, 1), dev)


class Rows:
    """Plays rows [start, start + count) of the 512-row stream; the others take a fixed TURN_LEFT."""

    def __init__(self, actor, start: int, count: int, total: int):
        self.actor, self.rows, self.batch_size = actor, slice(start, start + count), total
        self.started, self.decisions = time.time(), 0

    def reset(self, initial_observation):
        self.actor.reset(initial_observation[self.rows])

    def update(self, action, observation):
        self.actor.update(action[self.rows], observation[self.rows])

    def act(self):
        action = torch.full((self.batch_size,), int(1), dtype=torch.int64, device=dev)
        action[self.rows] = self.actor.act()
        self.decisions += 1
        if self.decisions % 5 == 0:
            print(f"  [{time.strftime('%H:%M:%S')}] decision {self.decisions}; {time.time() - self.started:.0f} s",
                  flush=True)
        return action


for chunk in range(settings.episodes // CHUNK_ROWS):
    path = out / f"chunk{chunk}.npz"
    if path.exists():
        print(f"chunk {chunk}: done ({path})", flush=True)
        continue
    print(f"[{time.strftime('%H:%M:%S')}] chunk {chunk}: rows {chunk * CHUNK_ROWS}-{(chunk + 1) * CHUNK_ROWS - 1}",
          flush=True)
    model = ExactWumpusSearchModel(world, CHUNK_ROWS, SIMS, MAX_STEPS, LEAF, ROLLOUTS, seed=stream_seed(seed, 5))
    planner = PlanningAgent(model, BeliefTreeSearch(model, SIMS, C_PUCT, 0.3, dirichlet_epsilon=0.0,
                                                    seed=stream_seed(seed, 2), observation_prune=PRUNE),
                            CHUNK_ROWS, temperature=0.0, seed=stream_seed(seed, 3), device=dev)
    actor = Rows(planner, chunk * CHUNK_ROWS, CHUNK_ROWS, settings.episodes)
    episodes, _ = play_episodes(simulator(), actor)
    rewards = episodes.rewards[actor.rows]
    won, died = game_outcome_flags(rewards)
    np.savez(path, returns=discounted_returns(rewards, world.discount).cpu().numpy(), won=won.cpu().numpy(),
             died=died.cpu().numpy(), seconds=time.time() - actor.started)
    print(f"[{time.strftime('%H:%M:%S')}] chunk {chunk}: saved {path}", flush=True)
    del model, planner, actor, episodes
    torch.cuda.empty_cache()

chunks = [np.load(out / f"chunk{chunk}.npz") for chunk in range(settings.episodes // CHUNK_ROWS)]
reference, _ = play_episodes(simulator(), WumpusReferenceAgent(world, settings.episodes, RISK))
np.savez(out / "games.npz",
         **{"exact_planner.return": np.concatenate([c["returns"] for c in chunks]),
            "exact_planner.won": np.concatenate([c["won"] for c in chunks]),
            "exact_planner.died": np.concatenate([c["died"] for c in chunks]),
            f"reference_risk{RISK}.return": discounted_returns(reference.rewards, world.discount).cpu().numpy(),
            "analysis_seed": np.int64(seed)})
print(f"games: {out / 'games.npz'}", flush=True)
