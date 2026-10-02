# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/reward_diag.py
# ==============================================================================
# REWARD HEAD VS EXACT EXPECTED REWARD
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Scratch diagnostic of 2026-10-02, kept so the HANDOFF numbers are reproducible:
#    - Usage: uv run python <this> <run dir> <games> [greedy]. Compares the reward head's GRAB
#      prediction with the exact 1000 P(gold at own cell) - 1 from the enumeration filter.
#    - Source of 'the reward head is accurate' in HANDOFF.md, section 4c.
# ==============================================================================

import sys
from pathlib import Path
import torch
from ipomdp.experiments.runs import load_trained_run
from ipomdp.domain import WumpusSimulator
from ipomdp.agents import PlanningAgent, UniformRandomAgent
from ipomdp.planning import BeliefTreeSearch
from ipomdp.training import play_episodes, stream_seed
from ipomdp.interpretability import build_wumpus_dataset

run_dir, n = Path(sys.argv[1]), int(sys.argv[2])
dev = torch.device("cuda")
tr = load_trained_run(run_dir, dev)
world, run, cfg = tr.setup.domain, tr.run, tr.cfg
model = run.eval_agent.model
A = world.num_actions
sim = lambda: WumpusSimulator(world, n, cfg.env.max_steps, stream_seed(778, 1), dev)
sets = {"random": play_episodes(sim(), UniformRandomAgent(A, n, stream_seed(778, 4), dev))[0]}
if sys.argv[3:] == ["greedy"]:
    planner = PlanningAgent(model, BeliefTreeSearch(model, cfg.mcts.num_simulations, cfg.mcts.c_puct, cfg.mcts.dirichlet_alpha,
                            dirichlet_epsilon=0.0, seed=stream_seed(778, 2), observation_prune=cfg.mcts.observation_prune),
                            n, temperature=0.0, seed=stream_seed(778, 3), device=dev)
    sets["greedy"] = play_episodes(sim(), planner)[0]
DX, DY = [1, 0, -1, 0], [0, 1, 0, -1]   # placeholder; heading convention checked below
with torch.no_grad():
    for name, ep in sets.items():
        d = build_wumpus_dataset(world, model.belief_filter, ep)
        z = d.latents
        rh = torch.cat([model.expand(z[i:i+4096]).rewards for i in range(0, len(z), 4096)]).double()
        pit, wum, gold = d.hazards[:, :16], d.hazards[:, 16:32], d.hazards[:, 32:48]
        pg = gold.gather(1, d.cell[:, None]).squeeze(1)
        exact_grab = 1000 * pg - 1
        print(f"== {name}: {len(z)} alive states")
        print(f"  GRAB: R_hat mean {rh[:,3].mean():.2f}  exact mean {exact_grab.mean():.2f}  "
              f"| where P(gold here)=0 ({(pg<1e-9).double().mean():.2f} of states): R_hat mean {rh[pg<1e-9,3].mean():.2f}, "
              f"frac R_hat>0 {(rh[pg<1e-9,3]>0).double().mean():.3f}")
        hi = pg > 0.5
        if hi.any(): print(f"  GRAB where P(gold here)>0.5 ({int(hi.sum())}): R_hat {rh[hi,3].mean():.1f} exact {exact_grab[hi].mean():.1f}")
        for a, nm in enumerate(["FORWARD", "LEFT", "RIGHT", "GRAB", "SHOOT"]):
            print(f"  {nm:8s} R_hat mean {rh[:,a].mean():8.2f}  std {rh[:,a].std():7.2f}")
