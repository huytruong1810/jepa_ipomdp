# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/q_head_screen.py
# ==============================================================================
# CAN AN ACTION-VALUE HEAD RESOLVE THE ACTION GAP THAT DIFFERENCES OF A STATE VALUE CANNOT?
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-12; HANDOFF sections 5.9 and 6.2 C):
#    - teacher_plugin_diag.py: a state value V fitted to the exact-belief heuristic's returns on the
#      oracle latents (model.privileged_belief_loss runs) does not stop the idling, at 1, 10 or 50
#      simulations. Under the heuristic an idle step costs about 1 + (1 - gamma) * V ~ 7, while the
#      fit's errors are of order 100. Ranking actions by V on the CHILDREN of a state needs sibling
#      errors to cancel, and a state-value fit gives no reason for them to.
#    - An action-value head Q(z, .) on the PARENT latent compares actions inside one forward pass, and
#      its targets can share the layout: from one state, branch every action and continue with the
#      heuristic in the same (deterministic) world. The common part of those returns (the layout's
#      luck) cancels in their differences. Does such a head play where V on children does not?
#    - Privileged diagnostic, never an agent: the teacher acts on the exact belief and the branches
#      replay the true layout.
#
# 2. Method (usage: uv run python <this> <run dir> <teacher games> <states per game>; 2048 8):
#    - Pass 1: <teacher games> fresh heuristic games (agents/wumpus_reference.py, risk 0.4) on
#      stream_seed(settings.seed, 20), teacher_plugin_diag.py's pool, replayed through the acting filter.
#    - Pass 2: <states per game> live decision steps per game, sampled uniformly (fixed seed). From each,
#      5 rows: replay the recorded prefix exactly (every percept is checked), take action a at t, then
#      the heuristic to the window end. G(s, a) is the discounted return-to-go from t. The row whose a
#      is the heuristic's own action must reproduce the recorded return-to-go (checked).
#    - Fits (MLP 256-256 on frozen latents, standardised MSE, full-batch Adam 1e-3, fit on games 0-1 mod
#      4, early-stopped on 2 mod 4, tested on 3 mod 4; initialisation seeds 0, 1, 2):
#        V      teacher_plugin_diag.py's big-ES state value (every step of pass 1, post-terminal steps
#               included); acts by the one-step lookahead through the learned reward and observation
#               heads, argmax_a r_hat + gamma * sum_o P_hat(o) V(tau(z, a, o)) (no search);
#        Q      5 outputs regressing G(s, .);
#        A      5 outputs regressing G(s, .) - mean_a G(s, a) (the common-layout advantage).
#    - Offline (test states of pass 2): mean over states of G(s, chosen) - G(s, heuristic's action),
#      for V (learned children), V (exact children: the true reward and next latent), Q and A. The
#      chosen action depends on the latent only, so the mean is unbiased for the difference of the
#      heuristic-continuation action values at the belief, though each G is a single-layout sample.
#    - Play: the 512 analysis games (stream_seed(settings.seed, 1)), each fit acting greedily on the
#      acting filter's latent (no search). Reported with outcomes and paired gain over V.
#
# 2b. Pre-registered readout (recorded before the runs, 2026-10-12), on the oracle-latent runs' final
#    networks (<run>_final), median over the 3 initialisations, per seed:
#    - "Action-level values resolve the gap": the A head plays >= +150 above V (paired) and leaves
#      <= 60% unfinished on all 3 seeds. Then: design a non-privileged action-level value for the agent
#      (a change of method; ask the user).
#    - "The gap is not resolvable from this latent and data": the A head gains < +50 on 2+ seeds. Then:
#      read the offline metric (does it rank well on teacher states but drift in play?) before more.
#    - Otherwise: partial; report the per-seed numbers and the Q head beside A.
# ==============================================================================

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ipomdp.agents import WumpusReferenceAgent
from ipomdp.domain import WumpusSimulator, game_outcomes
from ipomdp.domain.wumpus import WumpusState
from ipomdp.experiments.runs import load_trained_run
from ipomdp.experiments.wumpus import WumpusAnalysisSettings
from ipomdp.training import discounted_returns, play_episodes, stream_seed

run_dir, n_teacher, per_game = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
dev = torch.device("cuda")
settings = WumpusAnalysisSettings()
trained = load_trained_run(run_dir, dev)
world, run, cfg = trained.setup.domain, trained.run, trained.cfg
model = run.eval_agent.model
if model.horizon is not None:
    raise ValueError("Time-aware runs carry steps-left in the state; this screen replays plain latents only.")
GAMMA, NUM_A, CHUNK = world.discount, model.num_actions, 2048   # pass 1 runs the tracker at 2048 rows


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


def returns_to_go(rewards: torch.Tensor) -> torch.Tensor:
    rewards = rewards.double()
    out, acc = torch.zeros_like(rewards), torch.zeros(rewards.shape[0], dtype=torch.float64, device=rewards.device)
    for t in reversed(range(rewards.shape[1])):
        acc = rewards[:, t] + GAMMA * acc
        out[:, t] = acc
    return out


def outcomes(rewards: torch.Tensor) -> dict:
    return {k: round(float(v), 4) for k, v in game_outcomes(rewards).items()}


# ---- Pass 1: the teacher's games ------------------------------------------------------------------------
rec = Recorder(WumpusReferenceAgent(world, n_teacher, 0.4), n_teacher)
episodes, env_states = play_episodes(WumpusSimulator(world, n_teacher, cfg.env.max_steps, stream_seed(settings.seed, 20), dev), rec)
T = episodes.actions.shape[1]
latents = torch.stack(rec.rows, 1)                                     # (G, T, D)
to_go = returns_to_go(episodes.rewards)                                # (G, T)
print(f"== {run_dir.name} best.pt (collection {trained.checkpoint_collection}); heuristic, fresh ({n_teacher}): "
      f"{outcomes(episodes.rewards)}", flush=True)

# Live decision steps (the game has not ended before t): the packed state's last entry is `done`.
live = ~WumpusState.from_tensor(env_states[:, :-1].flatten(0, 1)).done.view(n_teacher, T)
rng = np.random.default_rng(0)
picked = []
for g in range(n_teacher):
    steps = torch.nonzero(live[g]).squeeze(-1).cpu().numpy()
    picked += [(g, int(t)) for t in np.sort(rng.choice(steps, min(per_game, len(steps)), replace=False))]
picked = np.array(picked)
S = len(picked)

# ---- Pass 2: every action from every picked step, then the heuristic ------------------------------------
row_game = np.repeat(picked[:, 0], NUM_A)
row_t = np.repeat(picked[:, 1], NUM_A)
row_first = np.tile(np.arange(NUM_A), S)
G = torch.zeros(S * NUM_A, dtype=torch.float64, device=dev)
r_first = torch.zeros(S * NUM_A, dtype=torch.float64, device=dev)
z_next = torch.zeros(S * NUM_A, latents.shape[-1], device=dev)
for lo in range(0, S * NUM_A, CHUNK):
    sl = slice(lo, min(lo + CHUNK, S * NUM_A))
    games = torch.as_tensor(row_game[sl], device=dev)
    t_row = torch.as_tensor(row_t[sl], device=dev)
    first = torch.as_tensor(row_first[sl], device=dev)
    R = len(games)
    rec_actions, rec_obs = episodes.actions[games], episodes.observations[games]
    state = WumpusState.from_tensor(env_states[games, 0])
    z = model.initial_states(R, episodes.initial_observations[games])
    teacher = WumpusReferenceAgent(world, R, 0.4)
    teacher.reset(episodes.initial_observations[games])
    acc = torch.zeros(R, dtype=torch.float64, device=dev)
    for k in range(T):
        action = torch.where(k < t_row, rec_actions[:, k], torch.where(k == t_row, first, teacher.act()))
        state, obs, reward = world.step_rules(state, action)
        prefix = k < t_row
        if bool((obs[prefix] != rec_obs[prefix, k]).any()):
            raise RuntimeError(f"Replay diverged from the recorded game at step {k}.")
        z = model.update(z, action, obs)
        teacher.update(action, obs)
        at = k == t_row
        r_first[sl] = torch.where(at, reward.double(), r_first[sl])
        z_next[sl] = torch.where(at.unsqueeze(-1), z, z_next[sl])
        acc += torch.where(k >= t_row, GAMMA ** (k - t_row).clamp(min=0).double() * reward.double(), 0.0)
    G[sl] = acc
    print(f"  branched rows {sl.stop}/{S * NUM_A}", flush=True)

G = G.view(S, NUM_A)
z_t = latents[picked[:, 0], picked[:, 1]]                              # (S, D)
teacher_action = episodes.actions[picked[:, 0], picked[:, 1]]          # (S,)
own = G.gather(1, teacher_action[:, None]).squeeze(1)
recorded = to_go[picked[:, 0], picked[:, 1]]
if not torch.allclose(own, recorded, atol=1e-6):
    raise RuntimeError(f"Teacher branch does not reproduce its game: max error {(own - recorded).abs().max():.3g}.")
gap = (G.max(1).values - own)
print(f"  {S} states, teacher branches reproduce their games; clairvoyant max_a G - G(teacher): "
      f"mean {gap.mean():.1f}, median {gap.median():.1f}; sd of G {G.std():.1f}; "
      f"sd of G - mean_a G {(G - G.mean(1, keepdim=True)).std():.1f}", flush=True)


# ---- Fits ----------------------------------------------------------------------------------------------
def fit_mlp(x, y, vx, vy, init_seed, steps=3000):
    """teacher_plugin_diag.py's fit, with a vector output: standardised MSE, early stopping every 50 steps."""
    torch.manual_seed(init_seed)
    out = 1 if y.dim() == 1 else y.shape[1]
    net = nn.Sequential(nn.Linear(x.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, out)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    y, vy = y.float().view(len(y), out), vy.float().view(len(vy), out)
    mu, sd = float(y.mean()), float(y.std()) + 1e-8
    best, best_state, stopped = float("inf"), None, steps
    for step in range(steps):
        loss = F.mse_loss(net(x), (y - mu) / sd)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 49:
            with torch.no_grad():
                val = F.mse_loss(net(vx), (vy - mu) / sd).item()
            if val < best:
                best, best_state, stopped = val, {k: v.clone() for k, v in net.state_dict().items()}, step + 1
    net.load_state_dict(best_state)
    net.eval()
    return (lambda z: (net(z.float()) * sd + mu).squeeze(-1)), stopped


games_all = torch.arange(n_teacher, device=dev) % 4
games_s = torch.as_tensor(picked[:, 0], device=dev) % 4
v_pick = lambda m: (latents[m].flatten(0, 1), to_go[m].flatten())  # noqa: E731
(vx, vy), (vex, vey) = v_pick(games_all < 2), v_pick(games_all == 2)
fit_s, es_s, test_s = games_s < 2, games_s == 2, games_s == 3
advantage = G - G.mean(1, keepdim=True)


def one_step_learned(value, z):
    with torch.no_grad():
        e = model.expand(z)
        nv = value(e.next_states.flatten(0, 2)).view(e.next_values.shape)
        return e.rewards.double() + GAMMA * (e.observation_probs * nv).sum(-1).double()


def one_step_exact(value, rows):
    """True reward and next latent of each branch (pass 2), then the fitted V."""
    with torch.no_grad():
        nv = value(z_next.view(S, NUM_A, -1)[rows].flatten(0, 1)).view(-1, NUM_A).double()
    return r_first.view(S, NUM_A)[rows] + GAMMA * nv


def play(policy) -> tuple[torch.Tensor, dict]:
    class Greedy:
        batch_size = settings.episodes

        def reset(self, initial_observation):
            self.state = model.initial_states(self.batch_size, initial_observation)

        def update(self, action, observation):
            self.state = model.update(self.state, action, observation)

        def act(self):
            with torch.no_grad():
                return policy(self.state).argmax(-1)

    eps, _ = play_episodes(WumpusSimulator(world, settings.episodes, cfg.env.max_steps, stream_seed(settings.seed, 1), dev), Greedy())
    return discounted_returns(eps.rewards, GAMMA).double(), outcomes(eps.rewards)


def offline(scores: torch.Tensor) -> str:
    """Mean of G(s, chosen) - G(s, teacher) over the test states, +- 1.96 s.e."""
    d = G[test_s].gather(1, scores.argmax(-1, keepdim=True)).squeeze(1) - own[test_s]
    return f"{d.mean():+7.1f} +- {1.96 * d.std() / np.sqrt(len(d)):5.1f}"


n_games = settings.episodes
for seed in (0, 1, 2):
    v, v_steps = fit_mlp(vx, vy, vex, vey, seed)
    q, q_steps = fit_mlp(z_t[fit_s], G[fit_s], z_t[es_s], G[es_s], seed)
    a, a_steps = fit_mlp(z_t[fit_s], advantage[fit_s], z_t[es_s], advantage[es_s], seed)
    print(f"-- init {seed}: stopped at V {v_steps}, Q {q_steps}, A {a_steps}")
    print(f"   offline G(chosen) - G(teacher) on {int(test_s.sum())} test states: "
          f"V learned children {offline(one_step_learned(v, z_t[test_s]))}, V exact children "
          f"{offline(one_step_exact(v, test_s))}, Q {offline(q(z_t[test_s]))}, A {offline(a(z_t[test_s]))}")
    base, base_out = play(lambda z: one_step_learned(v, z))
    print(f"   play V (one step) : return {base.mean():7.1f} +- {base.std() / np.sqrt(n_games):5.1f}  {base_out}")
    for name, head in (("Q", q), ("A", a)):
        ret, out = play(head)
        d = ret - base
        print(f"   play {name} head      : return {ret.mean():7.1f} +- {ret.std() / np.sqrt(n_games):5.1f}  {out}"
              f"  paired gain {d.mean():+7.1f} +- {1.96 * d.std() / np.sqrt(n_games):5.1f}", flush=True)
