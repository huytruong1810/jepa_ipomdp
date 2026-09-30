"""Search depth of the learned and exact-model planners (50 simulations), per decision."""
import sys
from pathlib import Path

import numpy as np
import torch

from ipomdp.agents import PlanningAgent
from ipomdp.domain import BatchedPOMDPEnv
from ipomdp.experiments import load_trained_run
from ipomdp.planning import BeliefTreeSearch, ExactSearchModel
from ipomdp.training import stream_seed

d = torch.device("cuda")
ref = torch.load(Path(sys.argv[1]) / "qstar.pt", weights_only=False)
t = load_trained_run(Path("runs/sweeps/polyak/seed0"), d)
cfg = t.cfg
for name, model in (("learned", t.run.eval_agent.model), ("exact", ExactSearchModel(t.pomdp, ref["v"], d))):
    B = 128
    planner = BeliefTreeSearch(model, cfg.mcts.num_simulations, cfg.mcts.c_puct, cfg.mcts.dirichlet_alpha, 0.0, 1)
    agent = PlanningAgent(model, planner, B, 0.0, 2, d)
    env = BatchedPOMDPEnv(t.pomdp, B, 100, stream_seed(5, 1), d)
    env.reset(); agent.reset()
    means, maxes = [], []
    for _ in range(30):
        a = agent.act()
        means.append(planner.statistics.mean_depth); maxes.append(planner.statistics.max_depth)
        agent.update(a, env.step(a).observation)
    print(f"{name}: selection depth mean {np.mean(means):.2f}, max {max(maxes)} "
          f"(tree depth incl. the final expansion = depth + 1)")
