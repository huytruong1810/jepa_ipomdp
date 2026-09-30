"""Greedy action of each trained (polyak) agent at each exact belief, vs the exact optimal action."""
import sys
from collections import defaultdict
from pathlib import Path

import torch

from ipomdp.agents import PlanningAgent
from ipomdp.domain import BatchedPOMDPEnv, belief_update, initial_beliefs
from ipomdp.experiments import load_trained_run
from ipomdp.interpretability import q_values
from ipomdp.planning import BeliefTreeSearch
from ipomdp.training import stream_seed

S = Path(sys.argv[1])
ref = torch.load(S / "qstar.pt", weights_only=False)
Q = ref["q"]
d = torch.device("cuda")
names = ["LISTEN", "OPEN_LEFT", "OPEN_RIGHT"]

# Exact optimal action and Q-margins at every net growl count n (P(left) after n net left growls).
print("exact optimal policy (net growls = #left - #right since the last door opening):")
for n in range(-4, 5):
    odds = (0.85 / 0.15) ** n
    p = odds / (1 + odds)
    q = q_values(Q, torch.tensor([[p, 1 - p]], dtype=torch.float64))[0]
    print(f"  net {n:+d}  P(left)={p:.4f}  Q*={[round(float(v), 2) for v in q]}  -> {names[int(q.argmax())]}")

for seed in range(5):
    t = load_trained_run(Path(f"runs/sweeps/polyak/seed{seed}"), d)
    cfg, model = t.cfg, t.run.eval_agent.model
    B = 256
    agent = PlanningAgent(model, BeliefTreeSearch(model, cfg.mcts.num_simulations, cfg.mcts.c_puct,
                                                  cfg.mcts.dirichlet_alpha, 0.0, stream_seed(31337, 1)),
                          B, 0.0, stream_seed(31337, 2), d)
    env = BatchedPOMDPEnv(t.pomdp, B, cfg.env.max_steps, stream_seed(31337, 3), d)
    env.reset(); agent.reset()
    b = initial_beliefs(t.pomdp, B, d)
    counts = defaultdict(lambda: [0, 0, 0])
    optimal = {}
    for _ in range(cfg.env.max_steps):
        a = agent.act()
        p = b[:, 0].cpu()
        opt = q_values(Q, b.cpu()).argmax(-1)
        for pi, ai, oi in zip(p.tolist(), a.cpu().tolist(), opt.tolist()):
            key = round(pi, 4)
            counts[key][ai] += 1
            optimal[key] = oi
        out = env.step(a)
        agent.update(a, out.observation)
        b = belief_update(t.pomdp, b, a, out.observation)
    total = sum(sum(c) for c in counts.values())
    agree = sum(c[optimal[k]] for k, c in counts.items())
    print(f"seed {seed}: agrees with the exact policy on {agree}/{total} decisions ({100 * agree / total:.2f}%)")
    for k in sorted(counts):
        c = counts[k]
        if sum(c) >= 20:
            print(f"   P(left)={k:.4f}  n={sum(c):5d}  listen/openL/openR = {c}  exact -> {names[optimal[k]]}")
