# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/grounding_screen.py
# ==============================================================================
# OFFLINE SCREEN OF VALUE TARGETS: WHICH RETURNS SEPARATE "STEP INTO A SAFE CELL" FROM IDLING?
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-09, HANDOFF section 6 step 1; user choice "screen offline, then train"):
#    - value_chain_diag.py (M4): on the current agent's FROZEN latents, a fresh value fitted to the
#      exact-belief heuristic's Monte Carlo returns prefers FORWARD at the safe unvisited cells the
#      agent declines (66-73%), while the learned value prefers the no-op. Two in-training remedies
#      (time-aware value, n-step returns of the agent's own greedy games) failed at ~9 GPU-hours each.
#    - Before training another remedy, screen candidate targets the AGENT ITSELF could use, on the
#      same frozen latents and with the same M4 test. Only a target that passes is worth a training run.
#    - Caveat on the optimistic target (corrects HANDOFF section 6): a single realised return G is NOT
#      <= V*(b); only its expectation over the hidden layout is. max(V, G) is therefore biased upward
#      by lucky layouts, the known weakness of self-imitation learning in stochastic environments.
#      This screen measures that bias (V_fit - realised return on the greedy games).
#
# 2. Method (usage: uv run python <this> <run dir> <games>; 512 reproduces report.json):
#    - Greedy analysis games and the heuristic's games exactly as value_chain_diag.py plays them. The
#      Recorder, play, one_step, fit_mlp and split helpers are copied from that script: report
#      scripts are standalone module-level programs, and importing one would run it.
#    - The agent's own replay buffer from checkpoints/latest.pt (all 8,960 games of a 100-collection
#      run, nothing overwritten at capacity 10,000), replayed through the best.pt acting filter.
#      Checkpoints before 52ac3fc store no greedy flags, so each game's source is reconstructed from
#      the insertion order of TrainingRun (collect_and_train adds env_batch_size games per collection,
#      evaluate adds eval_episodes greedy games after every eval_every-th collection; the first
#      warmup_episodes are uniform random). The script checks the reconstruction against the stored
#      flags when the checkpoint has them.
#    - Return-to-go G_t of every live step, within the 50-step window and without a bootstrap.
#    - Fresh MLP values fitted on frozen live-step states (games split by index mod 4: fit on 0-1,
#      early stopping on 2, reported on 3; early stopping because the MLP otherwise memorises the
#      per-game luck of Monte Carlo returns):
#        A    max(V_hat(z), G) on all buffer games: the optimistic (self-imitation) candidate;
#        A0   G on the buffer's exploring collection games only (epsilon/temperature policy MC);
#        Aall G on all buffer games (random, exploring and greedy);
#        B    G on the heuristic's games: the control that passed M4 before;
#      plus the learned V_hat as the baseline.
#    - Per value: at the agent's steps facing a provably safe unvisited cell where it did NOT step
#      forward, the one-step Q = R + gamma sum_o P(o) V(child) through the acting model gives the median
#      Q(FORWARD) - Q(chosen) and the fraction preferring FORWARD; and on the greedy games' idle and
#      progress states the mean V - realised return-to-go (the optimism bias).
#    - Pre-registered pass for A: median gap > 0 and FORWARD preferred > 50% on all 3 seeds.
#
# 3. Result 2026-10-09 (best.pt + latest.pt buffer, decoder seeds 0/1/2; 512 games, outcomes = report.json;
#    logs runs/wumpus/u64_store_eval_eps_decoder_k5_seed*.grounding_screen.log):
#    Median Q(FORWARD) - Q(chosen) at the declined safe-cell steps / FORWARD preferred, per seed:
#      learned V_hat            -9 / 23%     -18 / 13%     -19 / 34%
#      A    max(V_hat, G)       -20 / 23%    -28 / 16%     -14 / 35%    FAILS the pre-registered test
#      A0   G, exploring games  -8 / 40%     -49 / 27%     +33 / 65%
#      Aall G, all buffer       +7 / 53%     -51 / 25%     +10 / 54%
#      B    G, heuristic games  +36 / 69%    +69 / 62%     -24 / 39%
#    - A is the learned value in disguise: G exceeds V_hat on only 5% of the buffer's live steps (mean
#      lift 9-11), because V_hat already overestimates (V - return +480 to +690). The max cannot lower
#      the idle values, so the optimistic target has nothing to work with.
#    - The agent's own Monte Carlo returns explain only 9-13% of the return variance (layout luck) and
#      rank inconsistently across seeds.
#    - Even the control B is weaker than in value_chain_diag.py M4 (66-73%) under this early-stopped fit,
#      and fails on seed 2. M4's evidence for teacher returns is therefore sensitive to the fit protocol.
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
from ipomdp.domain.wumpus import WumpusAction
from ipomdp.experiments.runs import load_trained_run
from ipomdp.experiments.wumpus import WumpusAnalysisSettings
from ipomdp.planning import BeliefTreeSearch
from ipomdp.training import play_episodes, stream_seed
from ipomdp.training.checkpointing import load_checkpoint

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
    """Plays `actor`, tracking the acting model's state and the exact belief beside it (value_chain_diag.py)."""

    def __init__(self, actor):
        self.actor, self.batch_size = actor, n
        self.shadow = WumpusReferenceAgent(world, n, 0.4)
        self.rows: list[list] = []

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
            step.append((states[b], safe_ahead, int(m.cell[b])))
        self.rows.append(step)
        return self.actor.act()


def returns_to_go(rewards: np.ndarray) -> np.ndarray:
    """(B, T) discounted in-window return-to-go, no bootstrap."""
    to_go, acc = np.zeros_like(rewards), np.zeros(rewards.shape[0])
    for t in reversed(range(rewards.shape[1])):
        acc = rewards[:, t] + world.discount * acc
        to_go[:, t] = acc
    return to_go


def play(actor):
    rec = Recorder(actor)
    episodes, _ = play_episodes(WumpusSimulator(world, n, cfg.env.max_steps, stream_seed(settings.seed, 1), dev), rec)
    rewards = episodes.rewards.double().cpu().numpy()
    to_go, T = returns_to_go(rewards), rewards.shape[1]
    unfinished = ~((rewards >= 998).any(1) | (rewards <= -1000).any(1))
    items = []
    for b in range(n):
        live = [(t, rec.rows[t][b]) for t in range(T) if rec.rows[t][b] is not None]
        cells_seen, last_new = set(), 0
        for t, (_, _, cell) in live:
            if cell not in cells_seen:
                cells_seen.add(cell); last_new = t
        for t, (state, safe_ahead, _) in live:
            items.append({"game": b, "state": state, "safe_ahead": safe_ahead,
                          "idle": bool(unfinished[b] and t > last_new),
                          "action": int(episodes.actions[b, t]), "to_go": to_go[b, t]})
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


@torch.no_grad()
def batched(fn, x: torch.Tensor) -> torch.Tensor:
    return torch.cat([fn(x[i:i + 8192]) for i in range(0, len(x), 8192)])


def fit_mlp(x, y, vx, vy, steps=3000):
    """
    Small MLP regression on frozen inputs (targets standardised, full batch as value_chain_diag.py),
    early-stopped on held-out games (vx, vy). Monte Carlo returns of the agent's games are dominated by
    layout luck (+-1000), and without early stopping the MLP memorises per-game outcomes (the 32-game
    smoke check: held-out MAE above the constant's).
    """
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(x.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    mu, sd = float(y.mean()), float(y.std()) + 1e-8
    best, best_state = float("inf"), None
    for step in range(steps):
        loss = F.mse_loss(net(x).squeeze(-1), (y - mu) / sd)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 49:
            with torch.no_grad():
                val = F.mse_loss(net(vx).squeeze(-1), (vy - mu) / sd).item()
            if val < best:
                best, best_state = val, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return lambda z: net(z).squeeze(-1) * sd + mu


def held_out_note(v, x, y, constant):
    """Held-out MAE and the fraction of the constant predictor's MSE explained."""
    with torch.no_grad():
        p = v(x)
    mse, mse0 = F.mse_loss(p, y).item(), F.mse_loss(torch.full_like(y, constant), y).item()
    return (f"held-out MAE {(p - y).abs().mean().item():.1f} (constant {(y - constant).abs().mean().item():.1f}), "
            f"MSE explained {1 - mse / mse0:.3f}")


# --- The agent's own buffer (section 2) --------------------------------------
buf = load_checkpoint(run_dir / "checkpoints" / "latest.pt")["buffer"]
size, t_cfg = buf["size"], cfg.training
if size != buf["actions"].shape[0] and buf["next"] != size:
    raise RuntimeError("the buffer wrapped around; insertion-order reconstruction does not apply.")
source = []                                                      # 'random' | 'explore' | 'greedy' per slot
collection = 0
while len(source) < size:
    source += ["random" if collection * t_cfg.env_batch_size < t_cfg.warmup_episodes else "explore"] * t_cfg.env_batch_size
    collection += 1
    if collection % t_cfg.eval_every == 0 and t_cfg.store_eval_episodes:
        source += ["greedy"] * t_cfg.eval_episodes
if len(source) != size:
    raise RuntimeError(f"reconstructed {len(source)} sources for {size} stored games.")
source = np.array(source)
if "greedy" in buf and not np.array_equal(buf["greedy"][:size].cpu().numpy(), source == "greedy"):
    raise RuntimeError("reconstructed sources disagree with the stored greedy flags.")

actions = buf["actions"][:size].to(dev)
observations = buf["observations"][:size].to(dev)
rewards = buf["rewards"][:size].double().cpu().numpy()
buf_to_go = returns_to_go(rewards)
over = np.zeros_like(rewards, dtype=bool)                        # step t is after the game ended
ended = np.zeros(size, dtype=bool)
for t in range(rewards.shape[1]):
    over[:, t] = ended
    ended |= (rewards[:, t] >= 998) | (rewards[:, t] <= -1000)
buf_states = []
with torch.no_grad():
    state = model.initial_states(size, buf["initial_observations"][:size].to(dev))
    for t in range(rewards.shape[1]):
        buf_states.append(state)
        state = model.update(state, actions[:, t], observations[:, t])
buf_states = torch.stack(buf_states, 1)                          # (size, T, state)
live = torch.from_numpy(~over)
buf_x = buf_states[live.to(dev)]
buf_g = torch.from_numpy(buf_to_go[~over]).to(dev).float()
buf_game = np.broadcast_to(np.arange(size)[:, None], over.shape)[~over]
buf_src = source[buf_game]
buf_v = batched(model.values, buf_x).float()
outcomes = {s: {k: round(float(v), 4) for k, v in game_outcomes(torch.from_numpy(rewards[source == s])).items()}
            for s in ("random", "explore", "greedy")}

greedy_episodes, agent = play(planner)
heur_episodes, heur = play(WumpusReferenceAgent(world, n, 0.4))
print(f"== {run_dir.name} best.pt (collection {trained.checkpoint_collection}), {n} games each; buffer {size} games")
print("  greedy outcomes:", {k: round(float(v), 4) for k, v in game_outcomes(greedy_episodes.rewards).items()})
print("  heuristic outcomes:", {k: round(float(v), 4) for k, v in game_outcomes(heur_episodes.rewards).items()})
for s, o in outcomes.items():
    print(f"  buffer {s} games ({int((source == s).sum())}): {o}")
print(f"  buffer live steps: {len(buf_g)}; max(V_hat, G) = G on {(buf_g > buf_v).float().mean().item():.3f} of them, "
      f"mean lift {(torch.clamp(buf_g - buf_v, min=0)).mean().item():.1f}")

# --- Fits ------------------------------------------------------------------
# Games split by index: fit on 0, 1 mod 4; early stopping on 2 mod 4; reported on 3 mod 4.
part = torch.from_numpy(buf_game % 4).to(dev)
explore = torch.from_numpy(buf_src == "explore").to(dev)


def fit_buffer(target, mask):
    f, s, h = mask & (part < 2), mask & (part == 2), mask & (part == 3)
    v = fit_mlp(buf_x[f], target[f], buf_x[s], target[s])
    return v, held_out_note(v, buf_x[h], target[h], target[f].mean().item())


every = torch.ones_like(explore)
values = {"learned V_hat": (model.values, "")}
values["A    max(V_hat, G), all buffer"] = fit_buffer(torch.maximum(buf_v, buf_g), every)
values["A0   G, exploring games"] = fit_buffer(buf_g, explore)
values["Aall G, all buffer"] = fit_buffer(buf_g, every)
hx = torch.stack([i["state"] for i in heur])
hy = torch.tensor([i["to_go"] for i in heur], device=dev).float()
hpart = torch.tensor([i["game"] for i in heur], device=dev) % 4
hf, hs, hh = hpart < 2, hpart == 2, hpart == 3
vb = fit_mlp(hx[hf], hy[hf], hx[hs], hy[hs])
values["B    G, heuristic games"] = (vb, held_out_note(vb, hx[hh], hy[hh], hy[hf].mean().item()))

# --- The M4 test and the optimism bias --------------------------------------
states = torch.stack([i["state"] for i in agent])
to_go = np.array([i["to_go"] for i in agent])
idle = np.array([i["idle"] for i in agent])
declined = [k for k, i in enumerate(agent) if i["safe_ahead"] and i["action"] != FORWARD]
chosen = torch.tensor([agent[k]["action"] for k in declined], device=dev)
rows = torch.arange(len(declined), device=dev)
print(f"  agent steps facing a safe unvisited cell without FORWARD: {len(declined)}; idle {int(idle.sum())}, progress {int((~idle).sum())}")
for name, (v, note) in values.items():
    q = one_step(states[declined], v)
    d = (q[:, FORWARD] - q[rows, chosen]).cpu().numpy()
    bias = batched(v, states).double().cpu().numpy() - to_go
    print(f"  {name:32s} Q(FORWARD) - Q(chosen) median {np.median(d):7.1f}, FORWARD preferred {(d > 0).mean():.3f}; "
          f"V - return idle {bias[idle].mean():6.1f}, progress {bias[~idle].mean():6.1f}" + (f"; {note}" if note else ""))
