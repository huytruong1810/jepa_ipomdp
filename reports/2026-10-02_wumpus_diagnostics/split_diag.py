# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/split_diag.py
# ==============================================================================
# PLANNER VS REFERENCE AGENT, SPLIT BY THE START PERCEPT
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Scratch diagnostic of 2026-10-02, kept so the HANDOFF numbers are reproducible:
#    - Usage: uv run python <this> <run dir> <games>. A quiet start (o_0 = 0) makes the first step
#      provably safe; the split separates idling under real danger from failing to explore.
# ==============================================================================

import sys, torch
from pathlib import Path
from ipomdp.experiments.runs import load_trained_run
from ipomdp.domain import WumpusSimulator, game_outcomes
from ipomdp.agents import PlanningAgent, WumpusReferenceAgent
from ipomdp.planning import BeliefTreeSearch
from ipomdp.training import play_episodes, stream_seed, discounted_returns
run_dir, n = Path(sys.argv[1]), int(sys.argv[2]); dev = torch.device("cuda")
tr = load_trained_run(run_dir, dev); world, cfg = tr.setup.domain, tr.cfg; model = tr.run.eval_agent.model
sim = lambda: WumpusSimulator(world, n, cfg.env.max_steps, stream_seed(12345, 1), dev)
planner = PlanningAgent(model, BeliefTreeSearch(model, cfg.mcts.num_simulations, cfg.mcts.c_puct, cfg.mcts.dirichlet_alpha,
          dirichlet_epsilon=0.0, seed=stream_seed(12345, 2), observation_prune=cfg.mcts.observation_prune),
          n, temperature=0.0, seed=stream_seed(12345, 3), device=dev)
for name, agent in (("planner", planner), ("reference0.4", WumpusReferenceAgent(world, n, 0.4))):
    ep, _ = play_episodes(sim(), agent)
    quiet = ep.initial_observations == 0
    for label, rows in (("quiet start", quiet), ("percept at start", ~quiet)):
        r = ep.rewards[rows]; g = discounted_returns(r, world.discount)
        o = game_outcomes(r)
        print(f"{name:13s} {label:17s} n={int(rows.sum()):3d} return {g.mean():7.1f}  win {o['win_rate']:.2f} death {o['death_rate']:.2f} unfinished {o['unfinished_rate']:.2f}")
    if name == "planner":
        moved = (ep.actions == 0).any(1)
        print(f"planner: games with no FORWARD at all: {(~moved).double().mean():.2f} (quiet start {(~moved[quiet]).double().mean():.2f})")
        acts = torch.bincount(ep.actions.flatten(), minlength=5).double(); print("planner action fractions", (acts / acts.sum()).tolist())
