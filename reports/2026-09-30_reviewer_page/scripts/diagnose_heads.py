"""Where does the learned planner lose return? Heads vs exact model on the agent's own beliefs."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ipomdp.agents import PlanningAgent
from ipomdp.domain import (BatchedPOMDPEnv, action_value_functions, belief_update, initial_beliefs,
                           observation_distribution, solve_infinite_horizon)
from ipomdp.experiments import load_trained_run
from ipomdp.interpretability import q_values
from ipomdp.planning import BeliefTreeSearch
from ipomdp.training import stream_seed

device = torch.device("cuda")
EPISODES, BASE = 128, 777
cache = Path(sys.argv[1])
seeds = [int(s) for s in sys.argv[2].split(",")]

pomdp = None
results = {}
for seed in seeds:
    trained = load_trained_run(Path(f"runs/sweeps/default/seed{seed}"), device)
    pomdp, run, cfg = trained.pomdp, trained.run, trained.cfg
    if not cache.exists():
        sol = solve_infinite_horizon(pomdp, tolerance=0.01, prune_epsilon=1e-6)
        torch.save({"v": sol.value_function, "q": action_value_functions(pomdp, sol.value_function, 1e-6)}, cache)
    ref = torch.load(cache, weights_only=False)
    V, Q = ref["v"], ref["q"]
    model = run.eval_agent.model
    agent = PlanningAgent(model, BeliefTreeSearch(model, cfg.mcts.num_simulations, cfg.mcts.c_puct,
                                                  cfg.mcts.dirichlet_alpha, 0.0, stream_seed(BASE, 1)),
                          EPISODES, 0.0, stream_seed(BASE, 2), device)
    env = BatchedPOMDPEnv(pomdp, EPISODES, cfg.env.max_steps, stream_seed(BASE, 3), device)
    env.reset(); agent.reset()
    b = initial_beliefs(pomdp, EPISODES, device)
    rec = {k: [] for k in ["b", "qstar", "root_q", "act", "r_err", "o_kl", "v_err", "onestep_q"]}
    rewards = []
    for t in range(cfg.env.max_steps):
        z = agent.state
        a = agent.act()
        root_q = torch.as_tensor(agent.planner.statistics.q_values)
        exp = model.expand(z)                                            # learned one-step
        onestep = exp.rewards + pomdp.discount * (exp.observation_probs * exp.next_values).sum(-1)
        bc = b.cpu()
        qstar = q_values(Q, bc)
        true_r = bc @ pomdp.reward.T                                     # (B, A)
        eye = torch.eye(pomdp.num_actions, device=device)
        p_true = torch.stack([observation_distribution(pomdp, b, torch.full((EPISODES,), k, device=device))
                              for k in range(pomdp.num_actions)], 1).cpu()           # (B, A, O)
        p_hat = exp.observation_probs.double().cpu()
        rec["b"].append(bc[:, 0]); rec["qstar"].append(qstar); rec["root_q"].append(root_q)
        rec["act"].append(a.cpu()); rec["onestep_q"].append(onestep.double().cpu())
        rec["r_err"].append(exp.rewards.double().cpu() - true_r)
        rec["o_kl"].append((p_true * (p_true.clamp(min=1e-12).log() - p_hat.clamp(min=1e-12).log())).sum(-1))
        from ipomdp.models.distributions import TwoHotSymlog  # noqa
        v_hat = run.codec.mean(run.value_head(z)).double().cpu()
        rec["v_err"].append(v_hat - V.value(bc))
        out = env.step(a)
        rewards.append(out.reward)
        agent.update(a, out.observation)
        b = belief_update(pomdp, b, a, out.observation)
    cat = {k: torch.stack(v, 1).flatten(0, 1) for k, v in rec.items()}      # (B*T, ...)
    ret = (torch.stack(rewards, 1).double().cpu() @ (pomdp.discount ** torch.arange(cfg.env.max_steps, dtype=torch.float64)))
    best = cat["qstar"].max(-1).values
    regret = best - cat["qstar"].gather(1, cat["act"].unsqueeze(1)).squeeze(1)
    wrong = regret > 1e-6
    p = cat["b"]
    conf = torch.minimum(p, 1 - p)                                          # 0.5 = uninformed
    bucket = lambda m: {f"{k:.4f}": int(((conf.round(decimals=4) == k) & m).sum())
                        for k in conf.round(decimals=4).unique().tolist() if int(((conf.round(decimals=4) == k) & m).sum())}
    res = {
        "return": float(ret.mean()), "return_se": float(ret.std() / np.sqrt(len(ret))),
        "wrong_rate": float(wrong.double().mean()), "regret_mean": float(regret.mean()),
        "regret_total_per_step": float(regret.sum() / len(regret)),
        "wrong_by_min_posterior": bucket(wrong), "visits_by_min_posterior": bucket(torch.ones_like(wrong)),
        "wrong_actions": {n: int((wrong & (cat["act"] == i)).sum()) for i, n in enumerate(pomdp.action_names)},
        "reward_abs_err_by_action": cat["r_err"].abs().mean(0).tolist(),
        "obs_kl_by_action": cat["o_kl"].mean(0).tolist(),
        "value_err_mean": float(cat["v_err"].mean()), "value_abs_err": float(cat["v_err"].abs().mean()),
        "rootq_minus_qstar_by_action": (cat["root_q"] - cat["qstar"]).mean(0).tolist(),
        "onestep_minus_qstar_by_action": (cat["onestep_q"] - cat["qstar"]).mean(0).tolist(),
    }
    key = conf.round(decimals=4)
    res["value_err_by_min_posterior"] = {f"{k:.4f}": round(float(cat["v_err"][key == k].mean()), 3) for k in key.unique().tolist()}
    # Decision margin at the 0.97 belief: optimal door vs LISTEN, exact vs searched vs one-step.
    m = key == 0.0302
    if m.any():
        opt = cat["qstar"][m].argmax(-1)
        pick = lambda q: q[m].gather(1, opt.unsqueeze(1)).squeeze(1) - q[m][:, 0]
        res["margin_at_0.97"] = {"exact": float(pick(cat["qstar"]).mean()), "search": float(pick(cat["root_q"]).mean()),
                                 "one_step": float(pick(cat["onestep_q"]).mean()),
                                 "door_reward_err": float(cat["r_err"][m].gather(1, opt.unsqueeze(1)).mean()),
                                 "search_q_err_listen": float((cat["root_q"] - cat["qstar"])[m][:, 0].mean()),
                                 "search_q_err_door": float((cat["root_q"] - cat["qstar"])[m].gather(1, opt.unsqueeze(1)).mean())}
    # On wrong decisions: which component explains it?
    if wrong.any():
        res["on_wrong"] = {
            "rootq_minus_qstar": (cat["root_q"] - cat["qstar"])[wrong].mean(0).tolist(),
            "reward_err": cat["r_err"][wrong].mean(0).tolist(),
            "value_err": float(cat["v_err"][wrong].mean()),
        }
    results[seed] = res
    print(seed, json.dumps(res, indent=1), flush=True)
