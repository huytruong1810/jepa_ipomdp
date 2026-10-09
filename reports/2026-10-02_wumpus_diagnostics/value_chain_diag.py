# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/value_chain_diag.py
# ==============================================================================
# WHY THE LEARNED VALUE PREFERS IDLING: UNANCHORED CHAIN, MISSING INFORMATION, OR LEARNING?
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-08):
#    - unfinished_diag.py: the value head rates the latent after stepping into a provably safe new
#      cell ~15 below the latent after a no-op, idle states get V_hat ~650 against a realised -22,
#      and the time-aware value (training/run.py, section 9) did not change this. Three hypotheses:
#        H1 unanchored self-consistent chain: every no-op moves the GRU latent, and the Bellman
#           target max_a [R + gamma V-bar(z')] is satisfied along the idle chain at a high value
#           that real reward never corrects (small Bellman residual, large gap to the return);
#        H2 the latent lacks progress information (safe cell ahead, cells explored);
#        H3 the information is there but value learning does not use it.
#
# 2. Method (usage: uv run python <this> <run dir> <games>; 512 reproduces report.json):
#    - Two sets of games on the analysis simulator seed of experiments/wumpus.py: the greedy
#      planner exactly as analyze_wumpus_run builds it, and the exact-belief heuristic
#      (agents/wumpus_reference.py, risk 0.4). A recorder tracks the acting filter's state beside
#      either actor (no random numbers drawn) and a shadow exact tracker gives, per live step,
#      whether a provably safe unvisited cell is directly ahead and how many cells were visited.
#    - Steps are split into IDLE (after the game's last new cell, in an unfinished game) and
#      PROGRESS (all other live steps).
#    - M1 Bellman residual V_hat(z) - max_a Q1(z) (one-step backup through the acting model, the
#      trainer's own target form) and V_hat(z) - realised discounted return-to-go.
#      Also which action attains max Q1, and whether the search (50 simulations) chose it.
#    - M2 drift: ||z_{t+1} - z_t|| and V_hat(z_{t+1}) - V_hat(z_t) on idle vs progress steps.
#    - M3 progress probes: MLPs on frozen states (fit on half of the games, scored on the other
#      half) for "safe unvisited cell ahead" (accuracy vs majority), cells visited and steps
#      since the last new cell (MAE vs the constant mean), on agent and heuristic histories.
#    - M4 grounded value refit: a fresh MLP V_g fitted on the heuristic's states to their Monte
#      Carlo return-to-go (a real policy's value, which explores). At the agent's steps facing a
#      safe unvisited cell where it did NOT step forward, the one-step Q_g = R + gamma sum_o P(o)
#      V_g(child) through the acting model compares FORWARD with the chosen action.
#
# 3. Result 2026-10-08 (best.pt, decoder seeds 0/1/2 and time-aware seed 0; 512 games, outcomes = report.json):
#    - H2 rejected: progress information IS in the latent. Held-out probes: safe unvisited cell ahead
#      93-97% (majority 79-84%), cells visited MAE 0.15-0.37 (constant 1.6-2.3), steps since the last
#      new cell MAE 1.9 (constant 12) on the agent's games and 0.2-0.4 (constant 14) on the heuristic's.
#    - H1 holds as a mechanism: on idle steps V_hat is 618-664 against a realised -22, with a small
#      Bellman residual (median 1.5-6.5, about the 1 + (1 - gamma) V ~ 7 a no-op backup can remove per
#      iteration). On seeds 0/1 the latent barely moves under idle actions (0.17-0.18 against 4-6 on
#      progress steps); seed 2 drifts more (3.5) and idles all the same, so drift is not required.
#    - H3 confirmed: on the SAME frozen latents, a value fitted to real returns (the heuristic's games,
#      held-out MAE 59-65 against 450 for a constant) prefers FORWARD at the agent's declined
#      safe-cell steps: median Q_g(FORWARD) - Q_g(chosen) +76 / +159 / +225 / +146, FORWARD preferred
#      66-73%; the learned value prefers the no-op (median -9 / -18 / -19 / -27).
#    - Conclusion: the representation is not the bottleneck. The bootstrapped value is
#      self-consistent but ungrounded on idle chains, and real returns would correct its ranking.
# ==============================================================================

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ipomdp.agents import PlanningAgent, WumpusReferenceAgent
from ipomdp.agents.wumpus_reference import _ZERO, _ahead
from ipomdp.domain import WumpusSimulator, game_outcomes
from ipomdp.domain.wumpus import NUM_CELLS, WumpusAction
from ipomdp.experiments.runs import load_trained_run
from ipomdp.experiments.wumpus import WumpusAnalysisSettings
from ipomdp.planning import BeliefTreeSearch
from ipomdp.training import play_episodes, stream_seed

run_dir, n = Path(sys.argv[1]), int(sys.argv[2])
dev = torch.device("cuda")
settings = WumpusAnalysisSettings()
trained = load_trained_run(run_dir, dev)
world, run, cfg = trained.setup.domain, trained.run, trained.cfg
model = run.eval_agent.model
FORWARD = int(WumpusAction.FORWARD)
planner = PlanningAgent(
    model, BeliefTreeSearch(model, cfg.mcts.num_simulations, cfg.mcts.c_puct, cfg.mcts.dirichlet_alpha,
                            dirichlet_epsilon=0.0, seed=stream_seed(settings.seed, 2),
                            observation_prune=cfg.mcts.observation_prune),
    n, temperature=0.0, seed=stream_seed(settings.seed, 3), device=dev)


class Recorder:
    """Plays `actor`, tracking the acting model's state and the exact belief beside it (section 2)."""

    def __init__(self, actor):
        self.actor, self.batch_size = actor, n
        self.shadow = WumpusReferenceAgent(world, n, 0.4)
        self.rows: list[list] = []                               # per step: per game (state, safe_ahead, visited) or None

    def reset(self, initial_observation):
        self.actor.reset(initial_observation)
        self.shadow.reset(initial_observation)
        self.state = model.initial_states(n, initial_observation)

    def update(self, action, observation):
        self.actor.update(action, observation)
        self.shadow.update(action, observation)
        self.state = model.update(self.state, action, observation)

    def act(self):
        s = self.shadow
        m = s._tracker.marginals(s._consistent, s._own)
        pit, wumpus, alive = m.pit.cpu().tolist(), m.wumpus.cpu().tolist(), m.wumpus_alive.tolist()
        states = self.state.clone()
        step = []
        for b in range(n):
            if m.done[b]:
                step.append(None)
                continue
            ahead = _ahead(int(m.cell[b]), int(m.heading[b]))
            safe_ahead = (ahead is not None and not s._visited[b][ahead]
                          and pit[b][ahead] + (wumpus[b][ahead] if alive[b] else 0.0) < _ZERO)
            step.append((states[b], safe_ahead, sum(s._visited[b]), int(m.cell[b])))
        self.rows.append(step)
        return self.actor.act()


def play(actor):
    rec = Recorder(actor)
    episodes, _ = play_episodes(WumpusSimulator(world, n, cfg.env.max_steps, stream_seed(settings.seed, 1), dev), rec)
    rewards = episodes.rewards.double().cpu().numpy()
    T = rewards.shape[1]
    to_go, acc = np.zeros_like(rewards), np.zeros(n)
    for t in reversed(range(T)):
        acc = rewards[:, t] + world.discount * acc
        to_go[:, t] = acc
    unfinished = ~((rewards >= 998).any(1) | (rewards <= -1000).any(1))
    items = []                                                   # live steps: dict per step
    for b in range(n):
        live = [(t, rec.rows[t][b]) for t in range(T) if rec.rows[t][b] is not None]
        cells_seen, last_new = set(), 0
        for t, (_, _, _, cell) in live:
            if cell not in cells_seen:
                cells_seen.add(cell); last_new = t
        since = 0
        cells_seen = set()
        for i, (t, (state, safe_ahead, visited, cell)) in enumerate(live):
            since = 0 if cell not in cells_seen else since + 1
            cells_seen.add(cell)
            items.append({"game": b, "t": t, "state": state, "safe_ahead": safe_ahead, "visited": visited,
                          "since_new": since, "idle": bool(unfinished[b] and t > last_new),
                          "action": int(episodes.actions[b, t]), "to_go": to_go[b, t],
                          "next": live[i + 1][1][0] if i + 1 < len(live) else None})
    return episodes, items


@torch.no_grad()
def one_step(states: torch.Tensor, value_fn) -> torch.Tensor:
    """(N, A) one-step Q = R + gamma sum_o P(o) value_fn(child), in chunks (N x A x O children)."""
    out = []
    for i in range(0, len(states), 512):
        e = model.expand(states[i:i + 512])
        values = value_fn(e.next_states.flatten(0, 2)).view_as(e.observation_probs)
        out.append(e.rewards + world.discount * (e.observation_probs * values).sum(-1))
    return torch.cat(out).double()


def fit_mlp(x, y, kind, steps=2000):
    """Small MLP on frozen inputs; kind 'bce' or 'mse' (targets standardised)."""
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(x.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    mu, sd = (0.0, 1.0) if kind == "bce" else (float(y.mean()), float(y.std()) + 1e-8)
    for _ in range(steps):
        out = net(x).squeeze(-1)
        loss = (F.binary_cross_entropy_with_logits(out, y) if kind == "bce" else F.mse_loss(out, (y - mu) / sd))
        opt.zero_grad(); loss.backward(); opt.step()
    return lambda z: net(z).squeeze(-1) * sd + mu


def tensors(items, key):
    return torch.tensor([float(i[key]) for i in items], device=dev)


def split(items):
    fit = [i for i in items if i["game"] < n // 2]
    return fit, [i for i in items if i["game"] >= n // 2]


greedy_episodes, agent = play(planner)
heur_episodes, heur = play(WumpusReferenceAgent(world, n, 0.4))
print(f"== {run_dir.name} best.pt (collection {trained.checkpoint_collection}), {n} games each")
print("  greedy outcomes:", {k: round(float(v), 4) for k, v in game_outcomes(greedy_episodes.rewards).items()})
print("  heuristic outcomes:", {k: round(float(v), 4) for k, v in game_outcomes(heur_episodes.rewards).items()})

# M1 and M2 on the agent's own games.
states = torch.stack([i["state"] for i in agent])
v = torch.cat([model.values(states[i:i + 8192]) for i in range(0, len(states), 8192)]).double()
q1 = one_step(states, model.values)
residual = (v - q1.max(-1).values).cpu().numpy()
gap = v.cpu().numpy() - np.array([i["to_go"] for i in agent])
idle = np.array([i["idle"] for i in agent])
has_next = np.array([i["next"] is not None for i in agent])
latent = lambda s: s if model.horizon is None else s[..., :-1]  # noqa: E731
nxt = torch.stack([i["next"] if i["next"] is not None else i["state"] for i in agent])
drift = (latent(nxt) - latent(states)).norm(dim=-1).cpu().numpy()
dv = (torch.cat([model.values(nxt[i:i + 8192]) for i in range(0, len(nxt), 8192)]).double() - v).cpu().numpy()
scale = latent(states).norm(dim=-1).mean().item()
for label, mask in (("idle", idle), ("progress", ~idle)):
    if not mask.any():
        continue
    mn = mask & has_next
    print(f"-- {label}: {int(mask.sum())} steps")
    print(f"   M1 Bellman residual V - max Q1: median {np.median(residual[mask]):.1f}, mean |.| {np.abs(residual[mask]).mean():.1f}; "
          f"V - return-to-go: mean {gap[mask].mean():.1f}; V mean {v.cpu().numpy()[mask].mean():.1f}")
    acts = np.array([i["action"] for i in agent])
    q1n = q1.cpu().numpy()
    best = q1n.argmax(-1)
    names = world.action_names
    print(f"   M1 argmax Q1 per action: " + ", ".join(f"{names[a]} {(best[mask] == a).mean():.3f}" for a in range(len(names)))
          + f"; search chose argmax Q1 {(best[mask] == acts[mask]).mean():.3f}; "
          f"Q1(chosen) - max Q1 median {np.median(q1n[mask, acts[mask]] - q1n[mask].max(-1)):.1f}")
    print(f"   M2 latent step ||z' - z||: median {np.median(drift[mn]):.3f} (mean |z| {scale:.2f}); "
          f"V(z') - V(z): mean {dv[mn].mean():.2f}, median {np.median(dv[mn]):.2f}")

# M3 progress probes.
for source, items in (("agent", agent), ("heuristic", heur)):
    fit, held = split(items)
    xf, xh = latent(torch.stack([i["state"] for i in fit])), latent(torch.stack([i["state"] for i in held]))
    yf, yh = tensors(fit, "safe_ahead"), tensors(held, "safe_ahead")
    p = fit_mlp(xf, yf, "bce")
    with torch.no_grad():
        acc = ((p(xh) > 0).float() == yh).float().mean().item()
    base = max(yh.mean().item(), 1 - yh.mean().item())
    line = f"   M3 {source}: safe-ahead accuracy {acc:.3f} (majority {base:.3f}, positives {yh.mean().item():.3f})"
    for key in ("visited", "since_new"):
        yf, yh = tensors(fit, key), tensors(held, key)
        p = fit_mlp(xf, yf, "mse")
        with torch.no_grad():
            mae = (p(xh) - yh).abs().mean().item()
        line += f"; {key} MAE {mae:.2f} (constant {(yh - yf.mean()).abs().mean().item():.2f})"
    print(line)

# M4 grounded value refit on the heuristic's states (full state: includes k on time-aware runs).
fit, held = split(heur)
vg = fit_mlp(torch.stack([i["state"] for i in fit]), tensors(fit, "to_go"), "mse", steps=3000)
with torch.no_grad():
    yh = tensors(held, "to_go")
    err = (vg(torch.stack([i["state"] for i in held])) - yh).abs().mean().item()
print(f"   M4 V_g on held-out heuristic states: MAE {err:.1f} (constant {(yh - tensors(fit, 'to_go').mean()).abs().mean().item():.1f})")
declined = [k for k, i in enumerate(agent) if i["safe_ahead"] and i["action"] != FORWARD]
if declined:
    s = states[declined]
    with torch.no_grad():
        qg = one_step(s, vg)
    qa = q1[declined]
    chosen = torch.tensor([agent[k]["action"] for k in declined], device=dev)
    rows = torch.arange(len(declined), device=dev)
    d_g = (qg[:, FORWARD] - qg[rows, chosen]).cpu().numpy()
    d_a = (qa[:, FORWARD] - qa[rows, chosen]).cpu().numpy()
    print(f"   M4 agent steps facing a safe unvisited cell without FORWARD: {len(declined)}")
    print(f"      learned value: Q1(FORWARD) - Q1(chosen) median {np.median(d_a):.1f}, FORWARD preferred {(d_a > 0).mean():.3f}")
    print(f"      grounded V_g : Q_g(FORWARD) - Q_g(chosen) median {np.median(d_g):.1f}, FORWARD preferred {(d_g > 0).mean():.3f}")
