# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/teacher_plugin_diag.py
# ==============================================================================
# DOES A TEACHER-RETURN VALUE, PLUGGED INTO THE EXISTING SEARCH, STOP THE IDLING?
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-12; user choice "plug-in test", HANDOFF sections 5.9 and 6.2 C):
#    - With a near-exact latent (the oracle-latent arm, model.privileged_belief_loss) the agent still
#      idles, and its own value is MORE over-optimistic on idle states (V_hat - realised return ~950).
#      On those latents a value fitted to the exact-belief heuristic's returns ranks FORWARD first at the
#      agent's declined safe steps under all 18 early-stopped fits (teacher_robustness.py).
#    - Ranking one-step children is not playing. Does the unchanged search (same learned reward and
#      observation heads, same filter, same 50 simulations, expectimax) PLAY well when only its leaf
#      value is replaced by such a teacher value? If yes, the value target is the whole problem, and
#      the next step is a non-privileged way to learn a value like it. If no, the learned model or the
#      search also blocks progress and must be diagnosed before any value work.
#
# 2. Method (usage: uv run python <this> <run dir> <greedy games> <teacher games>; 512 2048):
#    - Teacher data: <teacher games> fresh heuristic games (agents/wumpus_reference.py, risk 0.4) on
#      stream_seed(settings.seed, 20), exactly teacher_robustness.py's "big" pool, replayed through the
#      acting filter. Unlike teacher_robustness.py, steps AFTER the game ended are kept (return-to-go 0):
#      the search imagines post-terminal latents (death, climbing out), and a fit that never saw them
#      would extrapolate an arbitrary value there.
#    - Fits (teacher_robustness.py's MLP 256-256, standardised MSE, full-batch Adam 1e-3):
#        big-ES  fit on games 0-1 mod 4, early-stopped on 2 mod 4 (robust in the screen: 18/18);
#        big     fit on games 0-2 mod 4, 3000 steps (less robust there: 13/18);
#      each with initialisation seeds 0, 1, 2.
#    - Play: the analysis games of report.json (stream_seed(settings.seed, 1), planner streams 2 and 3,
#      Dirichlet off, temperature 0), first with the learned value (must reproduce report.json), then
#      with LearnedSearchModel.values replaced on the acting model's instance by each fit. Reported:
#      discounted return +- s.e., win / death / unfinished, and paired-per-game gain over the learned value.
#    - Privileged diagnostic, never an agent: the teacher acts on the exact belief.
#
# 2b. Pre-registered readout (recorded before the runs, 2026-10-12), on the oracle-latent runs' final
#    networks (<run>_final), median over the 3 initialisations of big-ES, per seed:
#    - "The value is the whole problem": on all 3 seeds the return rises by >= +150 paired and unfinished
#      falls by >= 15 points. Then: find a non-privileged target that learns such a value.
#    - "The model/search also blocks": on 2+ seeds the gain is < +50. Then: diagnose the learned model
#      under the teacher value (which imagined outcomes keep it idling) before any value work.
#    - Otherwise: partial; report with the per-seed numbers and the big (long-fit) arm.
# ==============================================================================

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ipomdp.agents import PlanningAgent, WumpusReferenceAgent
from ipomdp.domain import WumpusSimulator, game_outcomes
from ipomdp.experiments.runs import load_trained_run
from ipomdp.experiments.wumpus import WumpusAnalysisSettings
from ipomdp.planning import BeliefTreeSearch
from ipomdp.training import discounted_returns, play_episodes, stream_seed

run_dir, n, n_teacher = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
dev = torch.device("cuda")
settings = WumpusAnalysisSettings()
trained = load_trained_run(run_dir, dev)
world, run, cfg = trained.setup.domain, trained.run, trained.cfg
model = run.eval_agent.model


class Recorder:
    """Plays `actor`, tracking the acting model's latent beside it at every step (live or not)."""

    def __init__(self, actor, games):
        self.actor, self.batch_size, self.rows = actor, games, []

    def reset(self, initial_observation):
        self.actor.reset(initial_observation)
        self.state = model.initial_states(self.batch_size, initial_observation)

    def update(self, action, observation):
        self.actor.update(action, observation)
        self.state = model.update(self.state, action, observation)

    def act(self):
        self.rows.append(self.state.clone())
        return self.actor.act()


def teacher_pool():
    """(states (G, T, D), return-to-go (G, T)) of the fresh heuristic games, post-terminal steps included."""
    rec = Recorder(WumpusReferenceAgent(world, n_teacher, 0.4), n_teacher)
    episodes, _ = play_episodes(WumpusSimulator(world, n_teacher, cfg.env.max_steps, stream_seed(settings.seed, 20), dev), rec)
    rewards = episodes.rewards.double()
    to_go, acc = torch.zeros_like(rewards), torch.zeros(n_teacher, dtype=torch.float64, device=dev)
    for t in reversed(range(rewards.shape[1])):
        acc = rewards[:, t] + world.discount * acc
        to_go[:, t] = acc
    print(f"  heuristic, fresh ({n_teacher}): {_outcomes(episodes)}")
    return torch.stack(rec.rows, 1), to_go.float()


def fit_mlp(x, y, init_seed, vx=None, vy=None, steps=3000):
    """teacher_robustness.py's fit: MLP on frozen inputs, standardised targets, optional early stopping."""
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
    net.eval()
    return (lambda z: net(z.float()).squeeze(-1) * sd + mu), stopped


def _outcomes(episodes) -> dict:
    return {k: round(float(v), 4) for k, v in game_outcomes(episodes.rewards).items()}


def play_planner() -> tuple[torch.Tensor, dict]:
    """The analysis games with the acting model as it currently is (report.json's greedy planner)."""
    planner = PlanningAgent(
        model, BeliefTreeSearch(model, cfg.mcts.num_simulations, cfg.mcts.c_puct, cfg.mcts.dirichlet_alpha,
                                dirichlet_epsilon=0.0, seed=stream_seed(settings.seed, 2),
                                observation_prune=cfg.mcts.observation_prune),
        n, temperature=0.0, seed=stream_seed(settings.seed, 3), device=dev)
    episodes, _ = play_episodes(WumpusSimulator(world, n, cfg.env.max_steps, stream_seed(settings.seed, 1), dev), planner)
    return discounted_returns(episodes.rewards, world.discount).double(), _outcomes(episodes)


def report(name: str, returns: torch.Tensor, outcomes: dict, base: torch.Tensor | None) -> None:
    line = f"  {name:16s}: return {returns.mean():7.1f} +- {returns.std() / np.sqrt(n):5.1f}  {outcomes}"
    if base is not None:
        d = returns - base
        line += f"  paired gain {d.mean():+7.1f} +- {1.96 * d.std() / np.sqrt(n):5.1f}"
    print(line, flush=True)


print(f"== {run_dir.name} best.pt (collection {trained.checkpoint_collection}), {n} games")
base, base_outcomes = play_planner()
report("learned value", base, base_outcomes, None)

states, to_go = teacher_pool()
game = torch.arange(n_teacher, device=dev) % 4
pick = lambda mask: (states[mask].flatten(0, 1), to_go[mask].flatten())  # noqa: E731
(fx, fy), (ex, ey), (lx, ly), (hx, hy) = pick(game < 2), pick(game == 2), pick(game < 3), pick(game == 3)
mse0 = F.mse_loss(torch.full_like(hy, float(ly.mean())), hy).item()
arms = {"big-ES": lambda seed: fit_mlp(fx, fy, seed, ex, ey), "big": lambda seed: fit_mlp(lx, ly, seed)}
for name, fit in arms.items():
    for seed in (0, 1, 2):
        value, stopped = fit(seed)
        with torch.no_grad():
            explained = 1 - F.mse_loss(value(hx), hy).item() / mse0
        model.values = value                         # instance attribute: expand() now uses the teacher value
        try:
            returns, outcomes = play_planner()
        finally:
            del model.values
        report(f"{name} init {seed}", returns, outcomes, base)
        print(f"    fit: steps {stopped}, held-out MSE explained {explained:.3f}", flush=True)
