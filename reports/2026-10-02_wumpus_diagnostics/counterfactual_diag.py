# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/counterfactual_diag.py
# ==============================================================================
# IS DECLINING A PROVABLY SAFE STEP PROCRASTINATION? EXACT COUNTERFACTUAL ROLLOUTS
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (HANDOFF section 6, step 1; agreed with the user on 2026-10-11):
#    - unfinished_diag.py found that the greedy planner, facing a provably safe unvisited cell, steps in
#      only 29-42% of the time, and that its one-step learned preference for the no-op is about -15.
#      One idle step costs only about 1 + (1 - gamma) * V_hat ~ 7 under the self-consistent backup, less
#      than the value's errors between neighbouring latents. Hypothesis ("procrastination under a small
#      action gap"): noise in V_hat, not a real preference, decides those steps, and the backup lets the
#      planner postpone forever.
#    - At those declined steps, is FORWARD actually better *under the agent's own continuation policy*,
#      by how much, and how does that true advantage compare with the model's estimate?
#
# 2. Method:
#    - Wumpus is deterministic given the layout (WumpusWorld.step_rules), so a game can be replayed and
#      branched exactly from its packed initial state (privileged, simulator.state at t = 0).
#    - Pass 1 plays the analysis games exactly as analyze_wumpus_run (experiments/wumpus.py) does:
#      same streams, greedy planner on the acting copies of best.pt; with <games> = 512 its outcomes
#      must equal <run>/analysis/report.json (checked). A shadow exact-belief tracker gives the declined
#      safe steps with unfinished_diag.py's `safe_ahead` test (the cell ahead has P(pit) = P(live
#      wumpus) = 0 and is unvisited), and at each one the model's one-step estimate Q1(FORWARD) -
#      Q1(chosen) (SearchModel.expand: R + gamma * E[V_hat(next)]) and the search's root Q gap.
#    - Up to <max steps> declined steps are sampled uniformly (fixed seed). Each gets three rows that
#      replay its history up to t exactly (the replay must reproduce every recorded percept), then:
#        A  the agent's own action at t, then the greedy planner to the window end;
#        A2 the same, a replicate (the search samples observation branches, so its RNG matters);
#        F  FORWARD at t, then the greedy planner.
#      All rows share one batched search with its own seed, so common random numbers between A and F are
#      not available; A2 - A measures the noise floor of a single-rollout difference instead.
#    - Measures per sampled step: Delta_true = G_F - G_A (discounted gamma = 0.99 return-to-go from t),
#      the noise floor G_A2 - G_A, the outcome of each branch, the new cells each branch reaches, and in
#      F the steps until the planner declines a safe step again. Intervals are percentile bootstraps
#      that resample games (steps of one game are correlated).
#    - Pre-registered readout (HANDOFF section 6):
#      (i)  median Delta_true > +15 and > 0 on most steps while the model's estimate is negative: the
#           hypothesis holds; the fix belongs at the action-gap level (advantage learning, deeper search
#           along FORWARD).
#      (ii) Delta_true ~ 0 because F re-idles right after the forced step: idling is persistent and
#           one-step fixes cannot help (commitment / option-level planning, or accept the plateau).
#      (iii) Delta_true <= 0, forced steps lead to deaths or worse: declining is justified by risk the
#           exact safe_ahead test misses downstream; revisit the "idling" diagnosis.
#    - Usage: uv run python <this> <run dir> <games> <max steps>. Writes <run>/analysis/counterfactual.npz.
#      Smoke check with small numbers (the report.json check needs <games> = 512).
#
# 3. Result 2026-10-11 (current agent, best.pt of seeds 0-4, 512 games each, 300 sampled declined steps
#    per seed; every replay reproduced report.json; logs <run>.counterfactual_diag.log; runs/wumpus/
#    counterfactual_chain.sh). Means over seeds with Student-t 95% intervals (5 seeds):
#    - Mostly readout (ii), not (i). In unfinished games the median Delta_true is exactly 0 on every seed:
#      most forced steps change nothing, because the planner declines a later safe step within 5 steps
#      49 +- 9% of the time and keeps declining 59-72% of later safe steps.
#    - But the forced step has a real mean payoff there: +90 +- 38 (all 5 seeds positive, +69 to +143),
#      from win +10.9 +- 4.9 points (deaths +2.0 +- 2.4, n.s.). Delta > 0 at 24% of steps.
#    - The model does not see it: its one-step estimate is negative (median -18 +- 10) and uncorrelated
#      with Delta_true (Spearman -0.07 +- 0.20). Procrastination under noise is right about the sign
#      error; it is wrong that one nudge restarts exploration.
#    - Not readout (iii): forced steps rarely kill (deaths only in F 0.5-5%).
#    - In won games declining was usually right (mean Delta -178 +- 122; win -17 points): the planner
#      was heading for the gold, so "a safe unvisited cell ahead" is not always the best target. The
#      declined-step set mixes procrastination with sensible priorities.
#    - Noise floor: A2 - A has median 0 and sd 140-440, against a Delta sd of 300-580; branch A repeats
#      the original game exactly at 69-89% of steps (the search samples observation branches).
#    - Follow-up override_diag.py: removing the idling persistently (exact-belief navigation) is worth
#      +300 +- 42 per game, three quarters of it from targets at most 3 actions away (turn, then step).
# ==============================================================================

import json
import sys
from pathlib import Path

import numpy as np
import torch

from ipomdp.agents import PlanningAgent, WumpusReferenceAgent
from ipomdp.agents.wumpus_reference import _ZERO, _ahead
from ipomdp.domain import WumpusSimulator, game_outcomes
from ipomdp.domain.wumpus import NUM_CELLS, WumpusAction, WumpusState
from ipomdp.experiments.runs import load_trained_run
from ipomdp.experiments.wumpus import WumpusAnalysisSettings
from ipomdp.planning import BeliefTreeSearch
from ipomdp.training import play_episodes, stream_seed

run_dir, n, max_steps = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
dev = torch.device("cuda")
settings = WumpusAnalysisSettings()
trained = load_trained_run(run_dir, dev)
world, run, cfg = trained.setup.domain, trained.run, trained.cfg
model = run.eval_agent.model
if model.horizon is not None:
    raise ValueError("Time-aware runs carry steps-left in the state; this diagnostic replays plain latents only.")
FWD = int(WumpusAction.FORWARD)
GAMMA = world.discount


def make_search(seed: int) -> BeliefTreeSearch:
    return BeliefTreeSearch(model, cfg.mcts.num_simulations, cfg.mcts.c_puct, cfg.mcts.dirichlet_alpha,
                            dirichlet_epsilon=0.0, seed=seed, observation_prune=cfg.mcts.observation_prune)


def safe_ahead(shadow: WumpusReferenceAgent) -> np.ndarray:
    """unfinished_diag.py's test per row: the cell ahead is provably safe and unvisited (False when done)."""
    m = shadow._tracker.marginals(shadow._consistent, shadow._own)
    pit, wumpus = m.pit.cpu().numpy(), m.wumpus.cpu().numpy()
    alive, done = m.wumpus_alive.cpu().numpy(), m.done.cpu().numpy()
    cells, headings = m.cell.tolist(), m.heading.tolist()
    out = np.zeros(len(cells), dtype=bool)
    for b in range(len(cells)):
        if done[b]:
            continue
        ahead = _ahead(cells[b], headings[b])
        if ahead is not None and not shadow._visited[b][ahead]:
            out[b] = pit[b, ahead] + (wumpus[b, ahead] if alive[b] else 0.0) < _ZERO
    return out


# ---- Pass 1: the analysis games, exactly as analyze_wumpus_run plays them -------------------------------
planner = PlanningAgent(model, make_search(stream_seed(settings.seed, 2)), n, temperature=0.0,
                        seed=stream_seed(settings.seed, 3), device=dev)


class Recorder:
    """Drives the planner and a shadow exact-belief tracker on the same history; records each decision."""

    def __init__(self):
        self.batch_size = n
        self.shadow = WumpusReferenceAgent(world, n, 0.4)
        self.safe, self.one_step, self.root_q = [], [], []

    def reset(self, initial_observation):
        planner.reset(initial_observation)
        self.shadow.reset(initial_observation)

    def update(self, action, observation):
        planner.update(action, observation)
        self.shadow.update(action, observation)

    def act(self):
        with torch.no_grad():
            e = model.expand(planner.state)
            q1 = e.rewards.double() + GAMMA * (e.observation_probs * e.next_values).sum(-1).double()   # (B, A)
        self.one_step.append(q1.cpu().numpy())
        self.safe.append(safe_ahead(self.shadow))
        action = planner.act()
        self.root_q.append(planner.planner.statistics.q_values.copy())
        return action


rec = Recorder()
episodes, states = play_episodes(WumpusSimulator(world, n, cfg.env.max_steps, stream_seed(settings.seed, 1), dev), rec)
T = episodes.actions.shape[1]
outcomes = game_outcomes(episodes.rewards)
print(f"== {run_dir.name} best.pt (collection {trained.checkpoint_collection}), {n} games")
print("  outcomes:", {k: round(float(v), 4) for k, v in outcomes.items()})
if n == settings.episodes:
    reported = json.loads((run_dir / "analysis" / "report.json").read_text())["returns"]["learned_planner"]
    for key in ("win_rate", "death_rate", "unfinished_rate"):
        if abs(reported[key] - float(outcomes[key])) > 1e-9:
            raise RuntimeError(f"Replay is not faithful: {key} {float(outcomes[key])} vs report.json {reported[key]}.")
    print("  replay reproduces report.json")

actions = episodes.actions.cpu().numpy()
observations = episodes.observations.cpu()
rewards = episodes.rewards.double().cpu().numpy()
to_go = np.zeros_like(rewards)
acc = np.zeros(n)
for t in reversed(range(T)):
    acc = rewards[:, t] + GAMMA * acc
    to_go[:, t] = acc
won, died = (rewards >= 998).any(1), (rewards <= -1000).any(1)
game_kind = np.where(won, 0, np.where(died, 1, 2))           # 0 win, 1 death, 2 unfinished
safe = np.stack(rec.safe, 1)                                  # (n, T)
one_step = np.stack(rec.one_step, 1)                          # (n, T, A)
root_q = np.stack(rec.root_q, 1)                              # (n, T, A)

declined = np.argwhere(safe & (actions != FWD))
print(f"  declined safe steps: {len(declined)} in {len(np.unique(declined[:, 0]))} games "
      f"(FORWARD taken at {(safe & (actions == FWD)).sum()} safe steps)")
pick = np.random.default_rng(0).permutation(len(declined))[:max_steps]
sample = declined[np.sort(pick)]
K = len(sample)
g_idx, t_idx = sample[:, 0], sample[:, 1]
chosen = actions[g_idx, t_idx]

# ---- Pass 2: three branches per sampled step ------------------------------------------------------------
R = 3 * K
branch = np.repeat(np.arange(3)[None], K, 0).reshape(-1)      # 0 = A, 1 = A2, 2 = F; row = 3 k + branch
row_game, row_t = np.repeat(g_idx, 3), np.repeat(t_idx, 3)
first = np.where(branch == 2, FWD, np.repeat(chosen, 3))
row_game_t = torch.as_tensor(row_game, device=dev)
row_t_t = torch.as_tensor(row_t, device=dev)
first_t = torch.as_tensor(first, device=dev)
recorded_actions = episodes.actions[row_game_t]               # (R, T)
recorded_obs = episodes.observations[row_game_t]

env_state = WumpusState.from_tensor(states[row_game_t, 0])
z = model.initial_states(R, episodes.initial_observations[row_game_t])
shadow = WumpusReferenceAgent(world, R, 0.4)
shadow.reset(episodes.initial_observations[row_game_t])
search = make_search(stream_seed(settings.seed, 10))

G = np.zeros(R)
reward_rows = np.zeros((R, T))
row_safe = np.zeros((R, T), dtype=bool)                       # safe_ahead at decision k (rows with k >= t)
row_act = np.full((R, T), -1)
for k in range(T):
    s_now = safe_ahead(shadow)
    row_safe[:, k] = s_now & (k >= row_t)
    plan = (k > row_t) & ~env_state.done.cpu().numpy()
    planned = torch.full((R,), int(WumpusAction.TURN_LEFT), dtype=torch.int64, device=dev)  # done rows pay 0
    if plan.any():
        rows = torch.as_tensor(np.flatnonzero(plan), device=dev)
        policy = search.search(z[rows], 0.0)
        planned[rows] = policy.argmax(-1)
    action = torch.where(k < row_t_t, recorded_actions[:, k], torch.where(k == row_t_t, first_t, planned))
    env_state, obs, reward = world.step_rules(env_state, action)
    prefix = k < row_t_t
    if bool((obs[prefix] != recorded_obs[prefix, k]).any()):
        raise RuntimeError(f"Replay diverged from the recorded game at step {k}.")
    z = model.update(z, action, obs)
    shadow.update(action, obs)
    act_np = action.cpu().numpy()
    row_act[:, k] = np.where(k >= row_t, act_np, -1)
    live = k >= row_t
    reward_rows[:, k] = np.where(live, reward.double().cpu().numpy(), 0.0)
    G += np.where(live, GAMMA ** np.clip(k - row_t, 0, None) * reward_rows[:, k], 0.0)
    if k % 10 == 9:
        print(f"  branch step {k + 1}/{T}", flush=True)

# Per-row outcome and exploration after t.
row_won = (reward_rows >= 998).any(1)
row_died = (reward_rows <= -1000).any(1)
row_kind = np.where(row_won, 0, np.where(row_died, 1, 2))
# New cells after t: replay each row's actions (recorded prefix, then its branch) from its initial state.
st = WumpusState.from_tensor(states[row_game_t, 0])
seen_before = [set([0]) for _ in range(R)]
seen_after = [set() for _ in range(R)]
all_actions = torch.as_tensor(np.where(row_act >= 0, row_act, recorded_actions.cpu().numpy()), device=dev)
for k in range(T):
    st, _, _ = world.step_rules(st, all_actions[:, k])
    for r, c in enumerate(st.cell.tolist()):
        (seen_after[r] if k >= row_t[r] else seen_before[r]).add(c)
new_cells = np.array([len(seen_after[r] - seen_before[r]) for r in range(R)])
# Steps after t until the planner declines a safe step again (inf if never) and the decline rate after t.
reidle = np.full(R, np.inf)
decline_rate = np.full(R, np.nan)
for r in range(R):
    ks = [k for k in range(row_t[r] + 1, T) if row_safe[r, k]]
    if ks:
        dec = [k for k in ks if row_act[r, k] != FWD]
        decline_rate[r] = len(dec) / len(ks)
        if dec:
            reidle[r] = dec[0] - row_t[r]

by = lambda x, b: x[branch == b]  # noqa: E731
GA, GA2, GF = by(G, 0), by(G, 1), by(G, 2)
delta, noise = GF - GA, GA2 - GA
delta_avg = GF - 0.5 * (GA + GA2)
model_est = one_step[g_idx, t_idx, FWD] - one_step[g_idx, t_idx, chosen]
search_est = root_q[g_idx, t_idx, FWD] - root_q[g_idx, t_idx, chosen]
orig = to_go[g_idx, t_idx]


def boot(x: np.ndarray, stat, mask: np.ndarray | None = None, reps: int = 2000) -> tuple[float, float, float]:
    """stat(x) with a 95% percentile interval from a bootstrap over games."""
    mask = np.ones(K, dtype=bool) if mask is None else mask
    games = np.unique(g_idx[mask])
    groups = {g: np.flatnonzero(mask & (g_idx == g)) for g in games}
    rng = np.random.default_rng(1)
    reps_out = [stat(x[np.concatenate([groups[g] for g in rng.choice(games, len(games))])]) for _ in range(reps)]
    return float(stat(x[mask])), *np.percentile(reps_out, [2.5, 97.5])


fmt = lambda v: f"{v[0]:+.1f} [{v[1]:+.1f}, {v[2]:+.1f}]"  # noqa: E731
print(f"\n== {K} sampled declined safe steps in {len(np.unique(g_idx))} games")
print(f"  faithfulness: branch A reproduces the original return-to-go exactly at "
      f"{np.mean(np.isclose(GA, orig)):.3f} of steps; median |G_A - original| {np.median(np.abs(GA - orig)):.2f}")
for label, mask in (("all", np.ones(K, dtype=bool)), ("unfinished games", game_kind[g_idx] == 2),
                    ("won games", game_kind[g_idx] == 0)):
    if mask.sum() < 5:
        continue
    print(f"-- {label}: {mask.sum()} steps")
    print(f"   Delta_true = G_F - G_A: median {fmt(boot(delta, np.median, mask))}, mean {fmt(boot(delta, np.mean, mask))}, "
          f"> 0 at {np.mean(delta[mask] > 0):.3f}, < 0 at {np.mean(delta[mask] < 0):.3f}")
    print(f"   G_F - mean(G_A, G_A2): median {fmt(boot(delta_avg, np.median, mask))}, mean {fmt(boot(delta_avg, np.mean, mask))}")
    print(f"   noise floor G_A2 - G_A: median {np.median(noise[mask]):+.1f}, mean {noise[mask].mean():+.1f}, "
          f"|.| > 1 at {np.mean(np.abs(noise[mask]) > 1):.3f}, sd {noise[mask].std():.1f} (Delta sd {delta[mask].std():.1f})")
    print(f"   model one-step Q1(F) - Q1(chosen): median {np.median(model_est[mask]):+.1f}; search root Q gap median "
          f"{np.median(search_est[mask]):+.1f}; Spearman(model, Delta) "
          f"{np.corrcoef(np.argsort(np.argsort(model_est[mask])), np.argsort(np.argsort(delta[mask])))[0, 1]:+.2f}")
    for b, name in ((0, "A"), (1, "A2"), (2, "F")):
        rk = by(row_kind, b)[mask]
        print(f"   branch {name:2s}: win {np.mean(rk == 0):.3f} death {np.mean(rk == 1):.3f} unfinished {np.mean(rk == 2):.3f}; "
              f"new cells after t {by(new_cells, b)[mask].mean():.2f}; G median {np.median(by(G, b)[mask]):+.1f}; "
              f"declines later safe steps {np.nanmean(by(decline_rate, b)[mask]):.3f}")
    ri = by(reidle, 2)[mask]
    print(f"   F re-declines a safe step within 1/3/5/10 steps: "
          + " / ".join(f"{np.mean(ri <= d):.3f}" for d in (1, 3, 5, 10)) + f"; never {np.mean(np.isinf(ri)):.3f}")
    died_f = by(row_kind, 2)[mask] == 1
    died_a = by(row_kind, 0)[mask] == 1
    print(f"   deaths only in F: {np.mean(died_f & ~died_a):.3f}; only in A: {np.mean(died_a & ~died_f):.3f}")

np.savez(run_dir / "analysis" / "counterfactual.npz", game=g_idx, t=t_idx, chosen=chosen, game_kind=game_kind[g_idx],
         G_A=GA, G_A2=GA2, G_F=GF, original=orig, model_est=model_est, search_est=search_est,
         kind_A=by(row_kind, 0), kind_A2=by(row_kind, 1), kind_F=by(row_kind, 2),
         new_A=by(new_cells, 0), new_A2=by(new_cells, 1), new_F=by(new_cells, 2), reidle_F=by(reidle, 2),
         decline_A=by(decline_rate, 0), decline_F=by(decline_rate, 2))
print(f"  wrote {run_dir / 'analysis' / 'counterfactual.npz'}")
