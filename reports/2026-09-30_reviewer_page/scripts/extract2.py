"""Runtime, latent trajectories, observation-prediction accuracy and the policy table -> page/data.json."""
import json
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ipomdp.agents import PlanningAgent, UniformRandomAgent
from ipomdp.domain import (BatchedPOMDPEnv, belief_update, build_tiger_pomdp, initial_beliefs,
                           observation_distribution, solve_infinite_horizon)
from ipomdp.experiments import load_trained_run
from ipomdp.interpretability import build_probe_dataset, fit_mlp_probe, q_values
from ipomdp.planning import BeliefTreeSearch, ExactSearchModel
from ipomdp.training import play_episodes, stream_seed

S = Path(sys.argv[1])
data_path = S / "page" / "data.json"
D = json.load(open(data_path))
ref = torch.load(S / "qstar.pt", weights_only=False)
V, Q = ref["v"], ref["q"]
d = torch.device("cuda")
pomdp = build_tiger_pomdp()
sync = lambda: torch.cuda.synchronize()  # noqa: E731


# ---------------------------------------------------------------- 1. runtime
def timed(fn, repeats=1):
    best = float("inf")
    for _ in range(repeats):
        sync(); t0 = time.perf_counter(); fn(); sync()
        best = min(best, time.perf_counter() - t0)
    return best


CACHE = S / "runtime_cache.json"
runtime = json.load(open(CACHE)) if CACHE.exists() else {"solve": []}
for tol in (() if CACHE.exists() else (1.0, 0.1, 0.01)):
    t0 = time.perf_counter()
    sol = solve_infinite_horizon(pomdp, tolerance=tol, prune_epsilon=1e-6)
    runtime["solve"].append({"tolerance": tol, "seconds": time.perf_counter() - t0, "iterations": sol.iterations,
                             "vectors": int(sol.value_function.vectors.shape[0]), "bound": sol.error_bound})
    print("solve", runtime["solve"][-1], flush=True)

B, STEPS = 256, 20
t = load_trained_run(Path("runs/sweeps/polyak/seed0"), d)
cfg = t.cfg
learned = t.run.eval_agent.model


def run_agent(make_agent):
    env = BatchedPOMDPEnv(pomdp, B, STEPS, stream_seed(55, 1), d)
    agent = make_agent()
    return lambda: play_episodes(env, agent)


class AlphaVectorAgent:
    """Exact belief + argmax over the precomputed per-action Q* vectors (the solver's policy)."""
    def __init__(self):
        self.batch_size = B
        self.reset()
    def reset(self):
        self.b = initial_beliefs(pomdp, B, d)
    def act(self):
        return q_values(Q, self.b.cpu()).argmax(-1).to(d)
    def update(self, a, o):
        self.b = belief_update(pomdp, self.b, a, o)


exact_model = ExactSearchModel(pomdp, V, d)
configs = {
    "exact_alpha": lambda: AlphaVectorAgent(),
    "exact_search": lambda: PlanningAgent(exact_model, BeliefTreeSearch(exact_model, cfg.mcts.num_simulations, cfg.mcts.c_puct,
                                                                         cfg.mcts.dirichlet_alpha, 0.0, 1), B, 0.0, 2, d),
    "learned_search": lambda: PlanningAgent(learned, BeliefTreeSearch(learned, cfg.mcts.num_simulations, cfg.mcts.c_puct,
                                                                      cfg.mcts.dirichlet_alpha, 0.0, 1), B, 0.0, 2, d),
}
runtime.setdefault("per_decision", {})
for name, make in ([] if CACHE.exists() else configs.items()):
    seconds = timed(run_agent(make), repeats=2)
    runtime["per_decision"][name] = {"batch": B, "steps": STEPS, "seconds": seconds,
                                     "ms_per_decision": 1000 * seconds / (B * STEPS)}
    print(name, runtime["per_decision"][name], flush=True)
# learned filter step alone (the belief update the agent performs each step)
z = learned.initial_states(B)
a = torch.zeros(B, dtype=torch.int64, device=d); o = torch.zeros(B, dtype=torch.int64, device=d)
secs = timed(lambda: [learned.update(z, a, o) for _ in range(1000)], repeats=3)
bb = initial_beliefs(pomdp, B, d)
secs_b = timed(lambda: [belief_update(pomdp, bb, a, o) for _ in range(1000)], repeats=3)
runtime["belief_update_us"] = {"learned_filter": 1e6 * secs / 1000, "exact_bayes": 1e6 * secs_b / 1000, "batch": B}
runtime["training_seconds"] = [float(Path(f"runs/sweeps/polyak/seed{s}/TRAINED").read_text().split()[0]) for s in range(5)]
D["runtime"] = runtime

# ---------------------------------------------------------------- 2. latent trajectories (seed 0)
run = t.run
ep, _ = play_episodes(BatchedPOMDPEnv(pomdp, 256, 100, stream_seed(4242, 1), d),
                      UniformRandomAgent(3, 256, stream_seed(4242, 2), d))
ds = build_probe_dataset(pomdp, run.acting_filter, ep)
mean = ds.latents.double().mean(0)
_, _, vh = torch.linalg.svd((ds.latents.double() - mean).cpu(), full_matrices=False)
basis = vh[:2].T.to(d)
probe = fit_mlp_probe(ds, steps=3000)
NE = 8
env = BatchedPOMDPEnv(pomdp, NE, 100, stream_seed(9001, 1), d)
agent = PlanningAgent(learned, BeliefTreeSearch(learned, cfg.mcts.num_simulations, cfg.mcts.c_puct,
                                                cfg.mcts.dirichlet_alpha, 0.0, stream_seed(9001, 2)), NE, 0.0,
                      stream_seed(9001, 3), d)
env.reset(); agent.reset()
b = initial_beliefs(pomdp, NE, d)
zs, bs, acts, obs, rews, states = [agent.state.clone()], [b[:, 0].clone()], [], [], [], [env.state]
for _ in range(100):
    act = agent.act(); out = env.step(act)
    agent.update(act, out.observation); b = belief_update(pomdp, b, act, out.observation)
    zs.append(agent.state.clone()); bs.append(b[:, 0].clone()); acts.append(act); obs.append(out.observation)
    rews.append(out.reward); states.append(env.state)
Z = torch.stack(zs, 1)                                                   # (NE, 101, D)
xy = ((Z.double() - mean.to(d)) @ basis).cpu()
decoded = probe(Z.reshape(-1, Z.shape[-1])).reshape(NE, 101, -1)[..., 0].cpu()
D["trajectories"] = [{
    "xy": [[round(float(p[0]), 4), round(float(p[1]), 4)] for p in xy[e]],
    "exact": [round(float(v), 4) for v in torch.stack(bs, 1)[e].cpu()],
    "decoded": [round(float(v), 4) for v in decoded[e]],
    "actions": [int(v) for v in torch.stack(acts, 1)[e].cpu()],
    "observations": [int(v) for v in torch.stack(obs, 1)[e].cpu()],
    "rewards": [float(v) for v in torch.stack(rews, 1)[e].cpu()],
    "tiger": [int(v) for v in torch.stack(states, 1)[e].cpu()],
} for e in range(NE)]
# overwrite the geometry with the same basis so trajectories and cloud share axes
g = torch.Generator().manual_seed(0)
idx = torch.randperm(len(ds.latents), generator=g)[:2400]
cloud = ((ds.latents.double()[idx.to(d)] - mean) @ basis).cpu().tolist()
D["geometry"]["xy"] = [[round(p[0], 4), round(p[1], 4)] for p in cloud]
D["geometry"]["p"] = [round(float(v), 4) for v in ds.posteriors[idx, 0].cpu()]
print("trajectories done", flush=True)

# ---------------------------------------------------------------- 3. observation prediction vs exact
obs_acc = []
for seed in range(5):
    tr = load_trained_run(Path(f"runs/sweeps/polyak/seed{seed}"), d).run
    ep, _ = play_episodes(BatchedPOMDPEnv(pomdp, 256, 100, stream_seed(777, 1), d),
                          UniformRandomAgent(3, 256, stream_seed(777, 2), d))
    A = F.one_hot(ep.actions, 3).float(); O = F.one_hot(ep.observations, 2).float()
    with torch.no_grad():
        lat = tr.acting_filter.unroll(A, O)[:, :-1]                     # z_t before a_t
        p_hat = F.softmax(tr.acting_observation_head(lat.reshape(-1, lat.shape[-1]), A.reshape(-1, 3)).float(), -1).double()
    bel = initial_beliefs(pomdp, 256, d); ex = []
    for s_ in range(100):
        ex.append(observation_distribution(pomdp, bel, ep.actions[:, s_]))
        bel = belief_update(pomdp, bel, ep.actions[:, s_], ep.observations[:, s_])
    post = [initial_beliefs(pomdp, 256, d)[:, 0]]
    bel = initial_beliefs(pomdp, 256, d)
    for s_ in range(99):
        bel = belief_update(pomdp, bel, ep.actions[:, s_], ep.observations[:, s_]); post.append(bel[:, 0])
    p_ex = torch.stack(ex, 1).reshape(-1, 2)
    pl = torch.stack(post, 1).reshape(-1)
    act = ep.actions.reshape(-1); o = ep.observations.reshape(-1)
    row = {"per_action": {}}
    for ai, nm in enumerate(pomdp.action_names):
        m = act == ai
        ph, pe, oo = p_hat[m], p_ex[m], o[m]
        row["per_action"][nm] = {
            "n": int(m.sum()),
            "kl": float((pe * (pe.clamp(min=1e-12).log() - ph.clamp(min=1e-12).log())).sum(-1).mean()),
            "logloss_learned": float(-ph.gather(1, oo[:, None]).log().mean()),
            "logloss_exact": float(-pe.gather(1, oo[:, None]).log().mean()),
            "acc_learned": float((ph.argmax(-1) == oo).double().mean()),
            "acc_exact": float((pe.argmax(-1) == oo).double().mean()),
        }
    m = act == 0
    key = pl[m].round(decimals=4)
    cal = {}
    for k in key.unique().tolist():
        mm = key == k
        if int(mm.sum()) >= 30:
            cal[f"{k:.4f}"] = {"n": int(mm.sum()), "learned": float(p_hat[m][mm][:, 0].mean()),
                               "exact": float(p_ex[m][mm][:, 0].mean()), "empirical": float((o[m][mm] == 0).double().mean())}
    row["listen_calibration"] = cal
    obs_acc.append(row)
    print("obs seed", seed, {k: round(v["kl"], 5) for k, v in row["per_action"].items()}, flush=True)
D["observation"] = obs_acc

# ---------------------------------------------------------------- 4. policy table (policy_check.py output)
txt = (S / "policy_check_output.txt").read_text()
pol = {"agree": [], "rows": []}
for s_, a_, t_ in re.findall(r"seed (\d): agrees with the exact policy on (\d+)/(\d+)", txt):
    pol["agree"].append([int(s_), int(a_), int(t_)])
cur = None
for line in txt.splitlines():
    m1 = re.match(r"seed (\d):", line)
    if m1: cur = int(m1.group(1))
    m2 = re.search(r"P\(left\)=([\d.]+)\s+n=\s*(\d+)\s+listen/openL/openR = \[(\d+), (\d+), (\d+)\]\s+exact -> (\w+)", line)
    if m2 and cur is not None:
        pol["rows"].append([cur, float(m2.group(1)), int(m2.group(3)), int(m2.group(4)), int(m2.group(5)), m2.group(6)])
D["policy"] = pol
json.dump(D, open(data_path, "w"))
print("written", data_path.stat().st_size)
