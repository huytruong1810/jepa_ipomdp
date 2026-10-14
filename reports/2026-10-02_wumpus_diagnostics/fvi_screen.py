# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/fvi_screen.py
# ==============================================================================
# IS THE FLAT, OPTIMISTIC IDLE VALUE A MAXIMISATION BIAS OF THE BELLMAN BACKUP? (CONJECTURE C2)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-14; HANDOFF sections 1b C2 and 6.2 B):
#    - The agent's value is ~950 above the realised return on idle states, nearly flat, with a Bellman
#      residual of ~5: self-consistent, not under-trained. A flat V is self-consistent under
#      V = max_a [R + gamma E V(child)] only if the max over 5 actions x observation branches of noisy
#      child values lifts it by about (1 - gamma) V + 1 ~ 8 per step. That is the classic maximisation bias
#      (van Hasselt 2010) acting through the learned model.
#    - Does the bias come from the backup itself? Then a double or clipped-double estimator, which removes
#      the upward bias of max over noisy estimates, should give a value with a gradient towards progress and
#      a planner that stops idling. Or does it come from the training dynamics (online filter, EMA target,
#      two-hot loss, few updates per state)? Then the standard max, iterated to convergence on frozen latents,
#      will NOT reproduce the flat optimism.
#
# 2. Method (usage: uv run python <this> <run dir> <games> [<refreshes>]; 512, default 400):
#    - States: the agent's own replay buffer from checkpoints/latest.pt (grounding_screen.py's
#      reconstruction, wrap-around check included), replayed through the ACTING filter, every step 0..T
#      including post-terminal and the window-end latent, exactly the rows the trainer's value loss sees
#      (trainer.py: value_targets over all_latents). A random subset of STATE_GAMES games keeps a target
#      refresh to ~1 s; the subset is fixed by seed 0.
#    - Targets go through the agent's own LearnedSearchModel.expand (learned reward head, observation head,
#      filter successors; no pruning, as in the trainer), with the fitted value in place of the value head.
#      Fitted value iteration: a target network is refreshed every STEPS_PER_REFRESH Adam steps (minibatch
#      4096, MSE on values / SCALE), <refreshes> times from a fresh MLP 256-256 (the other screens' net).
#      gamma^400 ~ 0.02, so 400 refreshes reach the fixed point from 0 within ~2% of the scale.
#      Deviations from training, on purpose: frozen latents, MSE instead of two-hot, a hard target refresh
#      instead of EMA 0.9 per update, and many more updates per state.
#    - Arms:
#        (i)   max:        y = max_a [R + gamma sum_o P(o) V'(child)], one net;
#        (ii)  double:     nets A and B fitted on disjoint game halves (even / odd game index), each target
#                          picks a* with its own target net and evaluates a* with the other's
#                          (y_A = Q_B(s, argmax_a Q_A(s, a))), as in double Q-learning; acting value (A + B) / 2;
#        (iii) clipped:    nets A and B (disjoint halves) share y = max_a [R + gamma sum_o P(o) min(V'_A, V'_B)(child)]
#                          (TD3-style clipping, applied to the child values); acting value min(A, B).
#    - Play: the analysis games of report.json (stream_seed(settings.seed, 1), planner streams 2 and 3,
#      Dirichlet off, temperature 0), with LearnedSearchModel.values replaced on the acting instance by
#      each arm's value (teacher_plugin_diag.py's plug-in), at 1 and at cfg.mcts.num_simulations simulations.
#      The learned-value play at the configured count must reproduce report.json (raises otherwise).
#    - Measures on the agent's own decisions in the learned-value play (grounding_screen.py's exact-belief
#      shadow): V - realised return on idle (unfinished game, after the last new cell) and progress steps;
#      at the steps facing a safe unvisited cell where the agent did not go FORWARD, the fraction where the
#      one-step Q under the arm's value prefers FORWARD over the chosen action; the one-step root Q spread
#      (max - min and top-1 - top-2) at idle steps; the share of live steps with V > 999, a value no policy can
#      reach (the gold pays 999 and ends the game; every other reward is <= -1; phantom_reward_diag.py).
#
# 2b. Pre-registered readout (recorded 2026-10-14 before any run), per network, gains paired per game over
#    the learned value at the same simulation count:
#    - Sanity gate for arm (i): its idle V - return is at least half the learned value's. If it is not, the
#      backup does not create the flat optimism on frozen latents; the training dynamics do. Then arm (i)'s own
#      play is the result to read (it is the "same target, converged" agent), and (ii)/(iii) are secondary.
#    - "The maximisation bias is the mechanism": (ii) or (iii) gains >= +100 at the configured simulation
#      count on 2+ of 3 seeds of a population (oracle-latent final networks; current agent seeds 0-2),
#      with unfinished down. Then: a default-off double/clipped value option in the trainer (Tiger
#      fingerprint), small check, 3 seeds paired, in the population(s) where it passed.
#    - "Not the mechanism": (ii) and (iii) gain < +30 on 2+ of 3 seeds in both populations. Then choose
#      among HANDOFF 6.3 by which part of the target is flat (the measures above).
#    - Otherwise partial: report per seed, with the 1-simulation plays (closest to one-step greedy) as the
#      tie-breaker.
#    - Added before any run (2026-10-14, after phantom_reward_diag.py): the oracle-latent networks hold
#      V_hat > 999 on 36-58% of live steps and seeds 0 and 2 refuse the gold. If (i) also exceeds 999 and
#      (ii)/(iii) do not, the overshoot is the backup's maximisation bias; report it per arm either way.
#
# 3. Smoke check (16 games, 3 refreshes, oracle seed 0; orientation only): runs end to end.
#
# 4. Result: pending.
# ==============================================================================

import copy
import json
import sys
import time
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
from ipomdp.training import discounted_returns, play_episodes, stream_seed
from ipomdp.training.checkpointing import load_checkpoint

run_dir, n = Path(sys.argv[1]), int(sys.argv[2])
refreshes = int(sys.argv[3]) if len(sys.argv) > 3 else 400
STATE_GAMES, STEPS_PER_REFRESH, MINIBATCH, SCALE, CHUNK = 2560, 250, 4096, 1000.0, 2048
dev = torch.device("cuda")
settings = WumpusAnalysisSettings()
trained = load_trained_run(run_dir, dev)
world, run, cfg = trained.setup.domain, trained.run, trained.cfg
model = run.eval_agent.model
if model.horizon is not None:
    raise ValueError("Time-aware runs carry steps-left in the state; this screen fits plain-latent values only.")
GAMMA, FORWARD, NAMES = world.discount, int(WumpusAction.FORWARD), [a.name for a in WumpusAction]
SIMS = [1, cfg.mcts.num_simulations]


# --- Fitted value iteration (section 2) --------------------------------------
def mlp(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    d = cfg.model.latent_dim
    return nn.Sequential(nn.Linear(d, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 1)).to(dev)


def as_value(net: nn.Module):
    return lambda z: net(z.float()).squeeze(-1) * SCALE


@torch.no_grad()
def child_q(states: torch.Tensor, values) -> list[torch.Tensor]:
    """One-step Q (N, A) per value function: R + gamma sum_o P(o) v(child), the children computed once per chunk."""
    out = [[] for _ in values]
    for i in range(0, len(states), CHUNK):
        e = model.expand(states[i:i + CHUNK])
        kids = e.next_states.flatten(0, 2)
        for k, v in enumerate(values):
            out[k].append(e.rewards + GAMMA * (e.observation_probs * v(kids).view_as(e.observation_probs)).sum(-1))
    return [torch.cat(o) for o in out]


@torch.no_grad()
def clipped_q(states: torch.Tensor, va, vb) -> torch.Tensor:
    return child_q(states, [lambda z: torch.minimum(va(z), vb(z))])[0]


def fit(states: torch.Tensor, nets: list[nn.Module], halves: list[torch.Tensor], targets_fn, name: str) -> None:
    """Refresh frozen copies, compute targets for every net from them, then STEPS_PER_REFRESH Adam steps each."""
    opts = [torch.optim.Adam(net.parameters(), lr=1e-3) for net in nets]
    gen = torch.Generator(device=dev).manual_seed(0)
    started = time.time()
    for r in range(refreshes):
        frozen = [as_value(_copy(net)) for net in nets]
        targets = targets_fn(states, frozen)                       # one (N,) target per net
        for net, opt, rows, y in zip(nets, opts, halves, targets):
            for _ in range(STEPS_PER_REFRESH):
                idx = rows[torch.randint(len(rows), (MINIBATCH,), generator=gen, device=dev)]
                loss = F.mse_loss(net(states[idx]).squeeze(-1), y[idx] / SCALE)
                opt.zero_grad(); loss.backward(); opt.step()
        if r % 25 == 24 or r == refreshes - 1:
            with torch.no_grad():
                residual = [(as_value(net)(states[rows]) - y[rows]).abs().mean().item()
                            for net, rows, y in zip(nets, halves, targets)]
            print(f"    {name} refresh {r + 1}: mean target {[round(y.mean().item(), 1) for y in targets]}, "
                  f"|V - target| {[round(x, 1) for x in residual]}; {time.time() - started:.0f} s", flush=True)
    for net in nets:
        net.eval()


def _copy(net: nn.Module) -> nn.Module:
    return copy.deepcopy(net).eval()


def max_targets(states, frozen):
    return [child_q(states, frozen)[0].max(-1).values]


def double_targets(states, frozen):
    qa, qb = child_q(states, frozen)
    pick_a, pick_b = qa.argmax(-1, keepdim=True), qb.argmax(-1, keepdim=True)
    return [qb.gather(1, pick_a).squeeze(1), qa.gather(1, pick_b).squeeze(1)]


def clipped_targets(states, frozen):
    y = clipped_q(states, *frozen).max(-1).values
    return [y, y]


# --- The agent's own buffer (grounding_screen.py's reconstruction) -------------
buf = load_checkpoint(run_dir / "checkpoints" / "latest.pt")["buffer"]
size, t_cfg = buf["size"], cfg.training
if size != buf["actions"].shape[0] and buf["next"] != size:
    raise RuntimeError("the buffer wrapped around; insertion-order reconstruction does not apply.")
pick = torch.randperm(size, generator=torch.Generator().manual_seed(0))[:min(STATE_GAMES, size)].sort().values
with torch.no_grad():
    rows = [model.initial_states(len(pick), buf["initial_observations"][pick].to(dev))]
    actions, observations = buf["actions"][pick].to(dev), buf["observations"][pick].to(dev)
    for t in range(actions.shape[1]):
        rows.append(model.update(rows[-1], actions[:, t], observations[:, t]))
fvi_states = torch.stack(rows, 1).flatten(0, 1).float()            # (games * (T + 1), D), all steps as the trainer
game_index = torch.arange(len(pick), device=dev).repeat_interleave(actions.shape[1] + 1)
every = torch.arange(len(fvi_states), device=dev)
even, odd = every[game_index % 2 == 0], every[game_index % 2 == 1]
del buf


# --- Play (section 2) ----------------------------------------------------------
class Recorder:
    """Drives the planner; per live decision the acting latent, the exact-belief 'safe cell ahead' flag and the cell."""

    def __init__(self, actor, shadow: bool):
        self.actor, self.batch_size, self.rows = actor, n, []
        self.shadow = WumpusReferenceAgent(world, n, 0.4) if shadow else None

    def reset(self, initial_observation):
        self.actor.reset(initial_observation)
        if self.shadow is not None:
            self.shadow.reset(initial_observation)
            self.state = model.initial_states(n, initial_observation)

    def update(self, action, observation):
        self.actor.update(action, observation)
        if self.shadow is not None:
            self.shadow.update(action, observation)
            self.state = model.update(self.state, action, observation)

    def act(self):
        if self.shadow is not None:
            s = self.shadow
            m = s._tracker.marginals(s._consistent, s._own)
            pit, wumpus, alive = m.pit.cpu().tolist(), m.wumpus.cpu().tolist(), m.wumpus_alive.tolist()
            step = []
            for b in range(n):
                if m.done[b]:
                    step.append(None)
                    continue
                ahead = _ahead(int(m.cell[b]), int(m.heading[b]))
                safe = (ahead is not None and not s._visited[b][ahead]
                        and pit[b][ahead] + (wumpus[b][ahead] if alive[b] else 0.0) < _ZERO)
                step.append((self.state[b].clone(), safe, int(m.cell[b])))
            self.rows.append(step)
        return self.actor.act()


def play(sims: int, shadow: bool = False):
    planner = PlanningAgent(
        model, BeliefTreeSearch(model, sims, cfg.mcts.c_puct, cfg.mcts.dirichlet_alpha,
                                dirichlet_epsilon=0.0, seed=stream_seed(settings.seed, 2),
                                observation_prune=cfg.mcts.observation_prune),
        n, temperature=0.0, seed=stream_seed(settings.seed, 3), device=dev)
    rec = Recorder(planner, shadow)
    episodes, _ = play_episodes(WumpusSimulator(world, n, cfg.env.max_steps, stream_seed(settings.seed, 1), dev), rec)
    return episodes, rec


def report(name: str, episodes, base: torch.Tensor | None) -> torch.Tensor:
    returns = discounted_returns(episodes.rewards, GAMMA).double()
    out = {k: round(float(v), 4) for k, v in game_outcomes(episodes.rewards).items()}
    ended = (episodes.rewards.abs() >= 998).long().cumsum(1)
    live = ((ended - (episodes.rewards.abs() >= 998).long()) == 0)
    acts = episodes.actions[live]
    mix = ", ".join(f"{NAMES[a]} {(acts == a).float().mean():.3f}" for a in range(len(NAMES)))
    line = f"  {name:16s}: return {returns.mean():7.1f} +- {returns.std() / np.sqrt(n):5.1f}  {out}"
    if base is not None:
        d = returns - base
        line += f"  paired gain {d.mean():+7.1f} +- {1.96 * d.std() / np.sqrt(n):5.1f}"
    print(line + f"\n      live action mix {mix}", flush=True)
    return returns


print(f"== {run_dir.name} best.pt (collection {trained.checkpoint_collection}), {n} games, simulations {SIMS}; "
      f"FVI on {len(fvi_states)} buffer states of {len(pick)} games, {refreshes} refreshes", flush=True)
base = {}
for sims in SIMS:
    episodes, rec = play(sims, shadow=sims == cfg.mcts.num_simulations)
    base[sims] = report(f"learned V s{sims}", episodes, None)
    if sims == cfg.mcts.num_simulations:
        agent_episodes, agent_rec = episodes, rec
if n == settings.episodes:
    reported = json.loads((run_dir / "analysis" / "report.json").read_text())["returns"]["learned_planner"]
    out = game_outcomes(agent_episodes.rewards)
    for key in ("win_rate", "death_rate", "unfinished_rate"):
        if abs(reported[key] - float(out[key])) > 1e-9:
            raise RuntimeError(f"Replay is not faithful: {key} {float(out[key])} vs report.json {reported[key]}.")
    print("  replay reproduces report.json", flush=True)

# The agent's decisions: idle / progress, declined safe steps, realised return-to-go.
rewards = agent_episodes.rewards.double().cpu().numpy()
to_go, acc = np.zeros_like(rewards), np.zeros(n)
for t in reversed(range(rewards.shape[1])):
    acc = rewards[:, t] + GAMMA * acc
    to_go[:, t] = acc
unfinished = ~((rewards >= 998).any(1) | (rewards <= -1000).any(1))
items = []
for b in range(n):
    live = [(t, agent_rec.rows[t][b]) for t in range(len(agent_rec.rows)) if agent_rec.rows[t][b] is not None]
    seen, last_new = set(), 0
    for t, (_, _, cell) in live:
        if cell not in seen:
            seen.add(cell); last_new = t
    for t, (z, safe, _) in live:
        items.append((z, safe, bool(unfinished[b] and t > last_new), int(agent_episodes.actions[b, t]), to_go[b, t]))
agent_states = torch.stack([i[0] for i in items]).float()
safe = np.array([i[1] for i in items])
idle = np.array([i[2] for i in items])
chosen = torch.tensor([i[3] for i in items], device=dev)
agent_to_go = np.array([i[4] for i in items])
declined = torch.from_numpy(safe & (chosen.cpu().numpy() != FORWARD)).to(dev)
print(f"  agent live steps {len(items)}: idle {int(idle.sum())}, progress {int((~idle).sum())}, "
      f"declined safe steps {int(declined.sum())}", flush=True)


def measures(name: str, value) -> float:
    with torch.no_grad():
        v = torch.cat([value(agent_states[i:i + 8192]) for i in range(0, len(agent_states), 8192)]).double().cpu().numpy()
        q = child_q(agent_states, [value])[0].double()
    bias = v - agent_to_go
    d = q[declined, FORWARD] - q[declined].gather(1, chosen[declined, None]).squeeze(1)
    top = q[torch.from_numpy(idle).to(dev)].sort(-1, descending=True).values
    print(f"  {name:16s} V - return idle {bias[idle].mean():7.1f}, progress {bias[~idle].mean():7.1f}; V > 999 on {(v > 999).mean():.3f}; "
          f"FORWARD preferred at declined safe steps {(d > 0).float().mean().item():.3f} "
          f"(median Q gap {d.median().item():+.1f}); idle root Q spread median {(top[:, 0] - top[:, -1]).median().item():.1f}, "
          f"top-1 - top-2 {(top[:, 0] - top[:, 1]).median().item():.1f}", flush=True)
    return float(bias[idle].mean())


learned_idle_bias = measures("learned V", model.values)

arms = {}
net = mlp(0)
fit(fvi_states, [net], [every], max_targets, "(i) max")
arms["(i) max"] = as_value(net)
a, b = mlp(1), mlp(2)
fit(fvi_states, [a, b], [even, odd], double_targets, "(ii) double")
arms["(ii) double"] = (lambda va, vb: lambda z: 0.5 * (va(z) + vb(z)))(as_value(a), as_value(b))
a, b = mlp(3), mlp(4)
fit(fvi_states, [a, b], [even, odd], clipped_targets, "(iii) clipped")
arms["(iii) clipped"] = (lambda va, vb: lambda z: torch.minimum(va(z), vb(z)))(as_value(a), as_value(b))

for name, value in arms.items():
    idle_bias = measures(name, value)
    if name == "(i) max":
        print(f"    sanity gate (2b): (i) idle bias {idle_bias:.1f} vs half the learned {0.5 * learned_idle_bias:.1f}: "
              f"{'reproduces the flat optimism' if idle_bias >= 0.5 * learned_idle_bias else 'does NOT reproduce it'}", flush=True)
    model.values = value                          # instance attribute: expand() now uses the fitted value
    try:
        for sims in SIMS:
            episodes, _ = play(sims)
            report(f"{name} s{sims}", episodes, base[sims])
    finally:
        del model.values
