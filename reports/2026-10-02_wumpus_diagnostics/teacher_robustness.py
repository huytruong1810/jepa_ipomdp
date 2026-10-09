# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/teacher_robustness.py
# ==============================================================================
# IS THE "TEACHER RETURNS RANK FORWARD FIRST" RESULT ROBUST TO THE FIT PROTOCOL AND DATA SIZE?
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-09; user choice "check B's robustness first"):
#    - value_chain_diag.py M4: a value fitted on frozen latents to the exact-belief heuristic's Monte
#      Carlo returns (256 games, 3000 full-batch Adam steps, no early stopping) prefers FORWARD at the
#      agent's declined safe-cell steps on 66-73% of them.
#    - grounding_screen.py arm B: the same target, but early-stopped on a quarter of the games, gives
#      69% / 62% / 39% (seed 2 fails). Teacher returns are the only remaining value-grounding
#      candidate, so before a training option is built: is the ranking a property of the target or of
#      the fit?
#
# 2. Method (usage: uv run python <this> <run dir> <greedy games> <teacher games>; 512 2048):
#    - The agent's greedy analysis games exactly as value_chain_diag.py / grounding_screen.py play them
#      (outcomes = report.json); their declined steps facing a provably safe unvisited cell are the test.
#    - Heuristic games (agents/wumpus_reference.py, risk 0.4): the 512 on the analysis simulator seed
#      (the same layouts as the greedy games, as in M4), and <teacher games> fresh ones on a separate
#      stream (stream_seed(settings.seed, 20)).
#    - Arms (all: an MLP 256-256 on frozen states, standardised MSE, full-batch Adam 1e-3):
#        M4      512 analysis-seed games, fit on the first half, 3000 steps, no early stopping;
#        ES      the same games, fit on 0-1 mod 4, early stopping on 2 mod 4 (grounding_screen.py B);
#        big     <teacher games> fresh games, fit on 0-2 mod 4, 3000 steps, no early stopping;
#        big-ES  the same games, fit on 0-1 mod 4, early stopping on 2 mod 4;
#      each with network initialisation seeds 0, 1 and 2 (fit noise). Held-out error on 3 mod 4 of the
#      fresh games for every arm (the same held-out states for all arms).
#    - Copied helpers (Recorder, returns_to_go, one_step) as in grounding_screen.py: report scripts are
#      standalone and importing one runs it.
#    - Robust pass: median Q(FORWARD) - Q(chosen) > 0 and FORWARD preferred > 50% for every big-data arm
#      and initialisation seed on all 3 agent seeds.
#
# 3. Result 2026-10-09 (best.pt, decoder seeds 0/1/2; 512 greedy games = report.json, 2048 fresh heuristic games;
#    logs runs/wumpus/u64_store_eval_eps_decoder_k5_seed*.teacher_robustness.log). FORWARD preferred (min-max
#    over the 3 initialisations) / median gap range:
#               seed 0                 seed 1                 seed 2
#      V_hat    23%  -9                13%  -18               34%  -19
#      M4       59-69%  +71..+144      56-76%  +38..+232      66-76%  +84..+453
#      ES       65-69%  +36..+51       62-67%  +65..+69       39-47%  -24..-7
#      big      57-63%  +72..+148      56-79%  +78..+459      63-70%  +97..+179
#      big-ES   67-73%  +42..+54       52-58%  +17..+41       42-45%  -32..-12
#    - NOT robust by the pre-registered test: big-ES fails on seed 2 with 4x the data.
#    - Held-out fit quality hardly differs between arms (MSE explained 0.79-0.86; early stopping at
#      150-550 steps is slightly better), yet the ranking flips. Long fits pass everywhere but their
#      median gaps vary 3-6x across initialisations.
#    - Reading: the test evaluates V_g on the IMAGINED children of the agent's idle-region states,
#      which the heuristic's games barely cover (it rarely idles). The FORWARD preference there is
#      extrapolation, set by how sharply the fit is allowed to extrapolate, not a property of the
#      teacher's returns. M4's 66-73% is one draw from that spread. Teacher returns have no robust
#      offline support; neither does any candidate screened so far (grounding_screen.py).
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

run_dir, n, n_teacher = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
dev = torch.device("cuda")
settings = WumpusAnalysisSettings()
trained = load_trained_run(run_dir, dev)
world, run, cfg = trained.setup.domain, trained.run, trained.cfg
model = run.eval_agent.model
FORWARD = int(WumpusAction.FORWARD)


class Recorder:
    """Plays `actor` on `games` windows, tracking the acting model's state and the exact belief beside it."""

    def __init__(self, actor, games):
        self.actor, self.batch_size = actor, games
        self.shadow = WumpusReferenceAgent(world, games, 0.4)
        self.rows: list[list] = []

    def reset(self, initial_observation):
        self.actor.reset(initial_observation)
        self.shadow.reset(initial_observation)
        self.state = model.initial_states(self.batch_size, initial_observation)

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
        for b in range(self.batch_size):
            if m.done[b]:
                step.append(None)
                continue
            ahead = _ahead(int(m.cell[b]), int(m.heading[b]))
            safe_ahead = (ahead is not None and not s._visited[b][ahead]
                          and pit[b][ahead] + (wumpus[b][ahead] if alive[b] else 0.0) < _ZERO)
            step.append((states[b], safe_ahead))
        self.rows.append(step)
        return self.actor.act()


def returns_to_go(rewards: np.ndarray) -> np.ndarray:
    """(B, T) discounted in-window return-to-go, no bootstrap."""
    to_go, acc = np.zeros_like(rewards), np.zeros(rewards.shape[0])
    for t in reversed(range(rewards.shape[1])):
        acc = rewards[:, t] + world.discount * acc
        to_go[:, t] = acc
    return to_go


def play(actor, games, sim_seed):
    """Live steps of `games` windows: (states, safe_ahead, action, return-to-go, game index), outcomes."""
    rec = Recorder(actor, games)
    episodes, _ = play_episodes(WumpusSimulator(world, games, cfg.env.max_steps, sim_seed, dev), rec)
    rewards = episodes.rewards.double().cpu().numpy()
    to_go = returns_to_go(rewards)
    cols = {"state": [], "safe": [], "action": [], "to_go": [], "game": []}
    for t, step in enumerate(rec.rows):
        for b, row in enumerate(step):
            if row is None:
                continue
            cols["state"].append(row[0]); cols["safe"].append(row[1]); cols["action"].append(int(episodes.actions[b, t]))
            cols["to_go"].append(to_go[b, t]); cols["game"].append(b)
    out = {"state": torch.stack(cols["state"]),
           **{k: torch.tensor(cols[k], device=dev) for k in ("safe", "action", "game")},
           "to_go": torch.tensor(cols["to_go"], device=dev).float()}
    return out, {k: round(float(v), 4) for k, v in game_outcomes(episodes.rewards).items()}


@torch.no_grad()
def one_step(states: torch.Tensor, value_fn) -> torch.Tensor:
    """(N, A) one-step Q = R + gamma sum_o P(o) value_fn(child), in chunks (N x A x O children)."""
    out = []
    for i in range(0, len(states), 512):
        e = model.expand(states[i:i + 512])
        values = value_fn(e.next_states.flatten(0, 2)).view_as(e.observation_probs)
        out.append(e.rewards + world.discount * (e.observation_probs * values).sum(-1))
    return torch.cat(out).double()


def fit_mlp(x, y, init_seed, vx=None, vy=None, steps=3000):
    """MLP regression on frozen inputs (standardised targets); early-stopped on (vx, vy) when given."""
    torch.manual_seed(init_seed)
    net = nn.Sequential(nn.Linear(x.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    mu, sd = float(y.mean()), float(y.std()) + 1e-8
    best, best_state, stopped = float("inf"), None, steps
    for step in range(steps):
        loss = F.mse_loss(net(x).squeeze(-1), (y - mu) / sd)
        opt.zero_grad(); loss.backward(); opt.step()
        if vx is not None and step % 50 == 49:
            with torch.no_grad():
                val = F.mse_loss(net(vx).squeeze(-1), (vy - mu) / sd).item()
            if val < best:
                best, best_state, stopped = val, {k: v.clone() for k, v in net.state_dict().items()}, step + 1
    if best_state is not None:
        net.load_state_dict(best_state)
    return (lambda z: net(z).squeeze(-1) * sd + mu), stopped


planner = PlanningAgent(
    model, BeliefTreeSearch(model, cfg.mcts.num_simulations, cfg.mcts.c_puct, cfg.mcts.dirichlet_alpha,
                            dirichlet_epsilon=0.0, seed=stream_seed(settings.seed, 2),
                            observation_prune=cfg.mcts.observation_prune),
    n, temperature=0.0, seed=stream_seed(settings.seed, 3), device=dev)
agent, agent_outcomes = play(planner, n, stream_seed(settings.seed, 1))
small, small_outcomes = play(WumpusReferenceAgent(world, n, 0.4), n, stream_seed(settings.seed, 1))
big, big_outcomes = play(WumpusReferenceAgent(world, n_teacher, 0.4), n_teacher, stream_seed(settings.seed, 20))
print(f"== {run_dir.name} best.pt (collection {trained.checkpoint_collection})")
print(f"  greedy ({n}): {agent_outcomes}")
print(f"  heuristic, analysis seed ({n}): {small_outcomes}; live steps {len(small['to_go'])}")
print(f"  heuristic, fresh ({n_teacher}): {big_outcomes}; live steps {len(big['to_go'])}")

declined = (agent["safe"] & (agent["action"] != FORWARD)).nonzero().squeeze(-1)
s, chosen = agent["state"][declined], agent["action"][declined]
rows = torch.arange(len(declined), device=dev)
print(f"  agent steps facing a safe unvisited cell without FORWARD: {len(declined)}")
q = one_step(s, model.values)
d = (q[:, FORWARD] - q[rows, chosen]).cpu().numpy()
print(f"  {'learned V_hat':14s}        median {np.median(d):7.1f}, FORWARD preferred {(d > 0).mean():.3f}")

bp = big["game"] % 4
hx, hy = big["state"][bp == 3], big["to_go"][bp == 3]
sp = small["game"]
arms = {
    "M4":     lambda seed: fit_mlp(small["state"][sp < n // 2], small["to_go"][sp < n // 2], seed),
    "ES":     lambda seed: fit_mlp(small["state"][sp % 4 < 2], small["to_go"][sp % 4 < 2], seed,
                                   small["state"][sp % 4 == 2], small["to_go"][sp % 4 == 2]),
    "big":    lambda seed: fit_mlp(big["state"][bp < 3], big["to_go"][bp < 3], seed),
    "big-ES": lambda seed: fit_mlp(big["state"][bp < 2], big["to_go"][bp < 2], seed,
                                   big["state"][bp == 2], big["to_go"][bp == 2]),
}
mse0 = F.mse_loss(torch.full_like(hy, float(big["to_go"][bp < 3].mean())), hy).item()
for name, fit in arms.items():
    for seed in (0, 1, 2):
        v, stopped = fit(seed)
        with torch.no_grad():
            mse = F.mse_loss(v(hx), hy).item()
        q = one_step(s, v)
        d = (q[:, FORWARD] - q[rows, chosen]).cpu().numpy()
        print(f"  {name:8s} init {seed}: median {np.median(d):7.1f}, FORWARD preferred {(d > 0).mean():.3f}; "
              f"fresh held-out MSE explained {1 - mse / mse0:.3f}; steps {stopped}")
