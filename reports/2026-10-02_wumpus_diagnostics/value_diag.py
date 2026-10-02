# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/value_diag.py
# ==============================================================================
# VALUE HEAD VS REALISED RETURN-TO-GO
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Scratch diagnostic of 2026-10-02, kept so the HANDOFF numbers are reproducible:
#    - Usage: uv run python <this> <run dir> <games> [randomonly]. Compares V_hat(z_t) of the acting
#      value head with realised discounted return-to-go, on random (and greedy) games, plus the
#      one-step Bellman backup through the learned model and the mean Q per action.
#    - Source of the 'flat value' finding in HANDOFF.md, section 4c (full_seed0, old setup).
# ==============================================================================

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from ipomdp.experiments.runs import load_trained_run
from ipomdp.domain.wumpus import WumpusSimulator
from ipomdp.agents import PlanningAgent, UniformRandomAgent
from ipomdp.planning import BeliefTreeSearch
from ipomdp.training import play_episodes
from ipomdp.training import stream_seed

run_dir, n = Path(sys.argv[1]), int(sys.argv[2])
dev = torch.device("cuda")
tr = load_trained_run(run_dir, dev)
world, run, cfg = tr.setup.domain, tr.run, tr.cfg
model = run.eval_agent.model
A, O, g = world.num_actions, world.num_observations, world.discount
sim = lambda k: WumpusSimulator(world, n, cfg.env.max_steps, stream_seed(777, k), dev)
planner = PlanningAgent(model, BeliefTreeSearch(model, cfg.mcts.num_simulations, cfg.mcts.c_puct, cfg.mcts.dirichlet_alpha,
                        dirichlet_epsilon=0.0, seed=stream_seed(777, 2), observation_prune=cfg.mcts.observation_prune),
                        n, temperature=0.0, seed=stream_seed(777, 3), device=dev)
sets = {"random": play_episodes(sim(1), UniformRandomAgent(A, n, stream_seed(777, 4), dev))[0]}
if len(sys.argv) < 4:
    sets["greedy"] = play_episodes(sim(1), planner)[0]

def onehot(x, k):
    return x.float() if x.dim() == 3 or (x.dim() == 2 and x.shape[-1] == k and x.dtype.is_floating_point) else F.one_hot(x.long(), k).float()

with torch.no_grad():
    for name, ep in sets.items():
        a, o = onehot(ep.actions, A), onehot(ep.observations, O)
        o0 = None if ep.initial_observations is None else onehot(ep.initial_observations, O)
        z = model.belief_filter.unroll(a, o, o0)                      # (B, T+1, D)
        B, T1, D = z.shape
        v = model.codec.mean(model.value_head(z.reshape(-1, D))).view(B, T1).double()
        r = ep.rewards.double()
        T = r.shape[1]
        G = torch.zeros(B, T + 1, dtype=torch.float64, device=dev)
        for t in reversed(range(T)):
            G[:, t] = r[:, t] + g * G[:, t + 1]
        over = torch.zeros(B, T + 1, dtype=torch.bool, device=dev)
        over[:, 1:] = (r.abs() > 500).cummax(dim=1).values
        alive = ~over
        early = torch.zeros_like(alive); early[:, :50] = True   # truncation biases G late in the window
        unfinished = ~(r.abs() > 500).any(1)
        print(f"== {name} ({n} games): return {G[:,0].mean():.1f}, unfinished {unfinished.double().mean():.2f}")
        print(f"  V_hat(z_0) mean {v[:,0].mean():.1f}   G_0 mean {G[:,0].mean():.1f}")
        m = alive & early
        print(f"  alive t<50: V_hat {v[m].mean():.1f}  G {G[m].mean():.1f}  corr {torch.corrcoef(torch.stack([v[m], G[m]]))[0,1]:.3f}")
        mu = m & unfinished[:, None]
        if mu.any():
            print(f"  alive t<50, unfinished games: V_hat {v[mu].mean():.1f}  G {G[mu].mean():.1f}  (wait-forever value -1/(1-g) truncated)")
        print(f"  game over: V_hat mean {v[over].mean():.2f} |max| {v[over].abs().max():.1f}" if over.any() else "  no game over")
        # one-step Bellman backup through the model at alive states vs V_hat
        zz = z[m][:2048]
        ex = model.expand(zz)
        q = ex.rewards + g * (ex.observation_probs * ex.next_values).sum(-1)     # (N, A)
        vv = model.codec.mean(model.value_head(zz))
        print(f"  backup max_a Q {q.max(1).values.mean():.1f} vs V_hat {vv.mean():.1f}; mean Q per action " +
              " ".join(f"{x:.1f}" for x in q.mean(0).tolist()) + f"; argmax freq {torch.bincount(q.argmax(1), minlength=A).tolist()}")
        print(f"  reward head mean per action " + " ".join(f"{x:.2f}" for x in ex.rewards.mean(0).tolist()))
