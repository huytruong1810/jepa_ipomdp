# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/sims_diag.py
# ==============================================================================
# GREEDY EVALUATION WITH A DIFFERENT NUMBER OF SIMULATIONS
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Scratch diagnostic of 2026-10-02, kept so the HANDOFF numbers are reproducible:
#    - Usage: uv run python <this> <run dir> <games> <simulations>. Same analysis simulator
#      stream as analyze.py. Source of 'search depth is not the bottleneck' (HANDOFF.md, 4c).
# ==============================================================================

import sys, json
from pathlib import Path
import torch
from ipomdp.experiments.runs import load_trained_run
from ipomdp.domain import WumpusSimulator, game_outcomes
from ipomdp.agents import PlanningAgent
from ipomdp.planning import BeliefTreeSearch
from ipomdp.training import play_episodes, stream_seed, discounted_returns
run_dir, n, sims = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
dev = torch.device("cuda")
tr = load_trained_run(run_dir, dev)
world, run, cfg = tr.setup.domain, tr.run, tr.cfg
model = run.eval_agent.model
planner = PlanningAgent(model, BeliefTreeSearch(model, sims, cfg.mcts.c_puct, cfg.mcts.dirichlet_alpha, dirichlet_epsilon=0.0,
                        seed=stream_seed(12345, 2), observation_prune=cfg.mcts.observation_prune),
                        n, temperature=0.0, seed=stream_seed(12345, 3), device=dev)
ep, _ = play_episodes(WumpusSimulator(world, n, cfg.env.max_steps, stream_seed(12345, 1), dev), planner)
g = discounted_returns(ep.rewards, world.discount)
acts = torch.bincount(ep.actions.flatten(), minlength=5).double()
print(json.dumps({"sims": sims, "n": n, "return": float(g.mean()), "stderr": float(g.std()) / n ** 0.5,
                  **game_outcomes(ep.rewards), "action_fraction": (acts / acts.sum()).tolist()}))
