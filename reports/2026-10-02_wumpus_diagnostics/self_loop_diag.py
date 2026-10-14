# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/self_loop_diag.py
# ==============================================================================
# ARE THE IDLE STEPS SELF-LOOPS THAT PROP UP THEIR OWN VALUE? AN EXACT SELF-LOOP BACKUP AT PLANNING TIME
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-13; HANDOFF section 5.9, after q_head_screen.py):
#    - After the last new cell, 50-90% of the greedy agent's actions are GRAB without gold or SHOOT
#      without the arrow (unfinished_diag.py, all 8 current and oracle-latent runs). Both are exact
#      no-ops: the world, the exact belief and (nearly) the latent stay put; value_chain_diag.py
#      measures a latent step of 0.08 on idle steps against 5.4 on progress steps (|z| ~ 7.5).
#    - A no-op child is its own parent, so the planner scores it Q(z, noop) = r + gamma * V_hat(z):
#      the parent's own, optimistic value. In the value target the same branch feeds V_hat(z) back to
#      itself with factor gamma = 0.99, so an overestimate on an idle state shrinks by only 1% per
#      backup instead of being corrected by the states it leads to. That fits the idle-state
#      optimism (V_hat - realised return ~ 950), its flatness, and why a sharper latent (oracle arm,
#      smaller idle steps) idles more.
#    - Exact identity for a deterministic self-loop with reward r: V(z) = max(M, r / (1 - gamma)), M the
#      best non-loop action value. Playing the loop once and then optimally is r + gamma * V(z) < V(z)
#      whenever V(z) > r / (1 - gamma), so the loop is never strictly optimal unless every other action
#      is worse than looping forever, and valuing the loop's child at r / (1 - gamma) changes no
#      fixed point. That is not a bias towards moving: when every move is worse than idling forever
#      (no safe cell left), the identity still picks the loop.
#    - Does applying the identity at planning time alone (no retraining) stop the idling?
#
# 2. Method (usage: uv run python <this> <run dir> <games> <tolerances>; 512 0.05,0.1):
#    - LearnedSearchModel.expand is wrapped on the acting model's instance: a child (a, o') whose latent
#      satisfies ||z' - z|| <= tau * ||z|| gets the loop value r_hat(z, a) / (1 - gamma) instead of
#      V_hat(z'). The planner's backups (root and deeper nodes) all go through expand.
#    - tau: the idle and progress steps differ ~70x (0.08 / 5.4). On the oracle-latent seed 0 smoke
#      check (32 games) GRAB's children move 0.003 / 0.026 / 0.11 of ||z|| (10/50/90%), the turns
#      0.6-0.96 and FORWARD 0.3-0.87, so 0.05 and 0.1 are used (0.02, planned first, misses half of
#      the GRABs). The baseline game prints the per-action distribution of
#      ||z' - z|| / ||z|| over the expanded children at the states it visits.
#    - Play: report.json's analysis games (stream_seed(settings.seed, 1), planner streams 2 and 3,
#      Dirichlet off, temperature 0); the unwrapped play must reproduce report.json. Reported per
#      tolerance: return +- s.e., outcomes, paired gain, the action mix, and the share of decisions whose
#      chosen action was flagged as a loop (probability-weighted over o').
#
# 2b. Pre-registered readout (recorded before the runs, 2026-10-13), current agent seeds 0-4
#    (runs/sweeps/wumpus_current) and the oracle-latent final networks, per tolerance:
#    - "Self-loops are a primary mechanism": on the current agent (seeds 0-2 only, see below) the paired
#      gain is >= +100 on 2+ of 3 seeds and unfinished falls by >= 15 points on average, at both tolerances. Then: build the
#      identity into LearnedSearchModel.expand as a default-off option (it then also shapes the value
#      targets, which go through expand), Tiger fingerprint, and train 3 seeds.
#    - "A symptom": gain < +30 on 2+ of 3 seeds of the oracle-latent networks. The idling moves elsewhere (read the action mix: turn
#      cycles are 2-step loops this identity does not cover).
#    - Added after the 32-game smoke check and before the runs: on the current agent a no-op's child is
#      NOT near its parent (GRAB: median ||z' - z|| / ||z|| 0.46; the filter drifts on no-ops), so the
#      identity can only bite where the latent is belief-consistent. The oracle-latent networks are
#      therefore the sharp test: there, "primary mechanism" means a paired gain >= +100 on 2+ of 3 seeds
#      at both tolerances.
#    - Otherwise partial: report, and look at deaths (the planner may now gamble where the loop's
#      true value, -100, is the best option, which is correct play) and at the action mix.
#
# 2c. Version 2, frozen loop leaves (2026-10-14; this file's method since then; version 1 is at 3924a9d):
#    - Version 1 wrapped LearnedSearchModel.expand, so a loop child got r_hat / (1 - gamma) only as its
#      LEAF value. BeliefTreeSearch expands every child it selects, and an expanded node's value is
#      max_a Q(child, a) over its own children with the learned V_hat (mcts.py, DecisionNode.value). At 50
#      simulations with a uniform prior nearly every root action is selected, so the root's GRAB child was
#      expanded and the identity erased exactly where it should act. Worse, the expanded loop child is a
#      near-copy of the root explored one level deeper, and the max over more noisy estimates lifts it:
#      GRAB rose from 39% to 63% of live steps on oracle seed 0. Version 1 therefore did not test C1.
#      (Checked before this change: the loop value itself was right, r_hat(GRAB) = -1 on 99.9% of the
#      flagged children at the real states and at imagined depths 1 and 2.)
#    - Version 2: FrozenLoopSearch (below) subclasses BeliefTreeSearch. After each expansion, every child
#      with ||z' - z|| <= tau ||z|| is frozen: it is never expanded and keeps the value r_hat(z, a) / (1 - gamma).
#      By the identity of section 1 this changes no decision: if the best non-loop value M exceeds
#      r / (1 - gamma) the loop is never chosen with either value, and otherwise looping forever is optimal.
#      The model, the heads and src/ are untouched.
#    - The readout of section 2b applies unchanged. Logs: <run>.self_loop_frozen.log.
#
# 3. Results.
#    - Version 1 (leaf values only; logs <run>.self_loop_diag.log; baselines reproduce report.json):
#      oracle-latent final networks, paired gain at tau 0.05 / 0.1: seed 0 +26 / +25, seed 1 +6 / -19,
#      seed 2 +27 / +69; GRAB rose on seeds 0 and 2 (e.g. 39% -> 63%). Read by 2b as "a symptom", but void
#      as a test of C1 (section 2c).
#    - Version 2: pending.
# ==============================================================================

import json
import sys
from pathlib import Path

import numpy as np
import torch

from ipomdp.agents import PlanningAgent
from ipomdp.domain import WumpusSimulator, game_outcomes
from ipomdp.domain.wumpus import WumpusAction
from ipomdp.experiments.runs import load_trained_run
from ipomdp.experiments.wumpus import WumpusAnalysisSettings
from ipomdp.planning import BeliefTreeSearch
from ipomdp.planning.search_model import Expansion
from ipomdp.training import discounted_returns, play_episodes, stream_seed

run_dir, n = Path(sys.argv[1]), int(sys.argv[2])
tolerances = [float(t) for t in sys.argv[3].split(",")]
dev = torch.device("cuda")
settings = WumpusAnalysisSettings()
trained = load_trained_run(run_dir, dev)
world, run, cfg = trained.setup.domain, trained.run, trained.cfg
model = run.eval_agent.model
if model.horizon is not None:
    raise ValueError("Time-aware runs carry steps-left in the state; this diagnostic wraps plain latents only.")
GAMMA, NAMES = world.discount, [a.name for a in WumpusAction]
unwrapped = model.expand


def relative_steps(states: torch.Tensor, e: Expansion) -> torch.Tensor:
    """||z' - z|| / ||z|| per child, (N, A, O)."""
    return (e.next_states - states[:, None, None, :]).norm(dim=-1) / states.norm(dim=-1)[:, None, None]


class FrozenLoopSearch(BeliefTreeSearch):
    """BeliefTreeSearch whose self-loop children are frozen leaves worth r_hat / (1 - gamma) (section 2c)."""

    def __init__(self, *args, tau: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau

    def _expand(self, nodes, stats) -> None:
        live = [(node, st) for node, st in zip(nodes, stats) if not getattr(node, "frozen", False)]
        if not live:
            return
        super()._expand([n for n, _ in live], [st for _, st in live])
        for node, node_stats in live:
            changed = False
            for edge in node.edges:
                kids = torch.stack([child.state for child in edge.children])
                rel = ((kids - node.state).norm(dim=-1) / node.state.norm()).tolist()
                for child, step in zip(edge.children, rel):
                    if step <= self.tau:
                        child.frozen, child.leaf_value, changed = True, edge.reward / (1.0 - GAMMA), True
            if changed:
                for edge in node.edges:
                    node_stats.update(edge.refresh(self.model.discount))
                node.refresh()


class Recorder:
    """Drives the planner; per decision: the chosen action's loop mass and (baseline) the children's steps."""

    def __init__(self, planner, tau):
        self.planner, self.batch_size, self.tau = planner, n, tau
        self.loop_mass, self.steps = [], []

    def reset(self, initial_observation):
        self.planner.reset(initial_observation)

    def update(self, action, observation):
        self.planner.update(action, observation)

    def act(self):
        z = self.planner.state
        with torch.no_grad():
            e = unwrapped(z)
            rel = relative_steps(z, e)
        action = self.planner.act()
        if len(self.loop_mass) % 10 == 9:
            print(f"    decision {len(self.loop_mass) + 1}", flush=True)
        mass = (e.observation_probs * (rel <= (self.tau if self.tau is not None else 0.0))).sum(-1)   # (B, A)
        self.loop_mass.append(mass.gather(1, action[:, None]).squeeze(1).cpu())
        if self.tau is None:
            self.steps.append(((e.observation_probs * rel).sum(-1)).cpu())     # prob-weighted, (B, A)
        return action


def play(tau: float | None):
    args = (model, cfg.mcts.num_simulations, cfg.mcts.c_puct, cfg.mcts.dirichlet_alpha)
    kwargs = dict(dirichlet_epsilon=0.0, seed=stream_seed(settings.seed, 2), observation_prune=cfg.mcts.observation_prune)
    search = BeliefTreeSearch(*args, **kwargs) if tau is None else FrozenLoopSearch(*args, tau=tau, **kwargs)
    planner = PlanningAgent(model, search, n, temperature=0.0, seed=stream_seed(settings.seed, 3), device=dev)
    rec = Recorder(planner, tau)
    episodes, _ = play_episodes(WumpusSimulator(world, n, cfg.env.max_steps, stream_seed(settings.seed, 1), dev), rec)
    return episodes, rec


def live_steps(rewards: torch.Tensor) -> torch.Tensor:
    """(B, T) bool: decisions taken before the game ended (a win or a death pays |r| >= 998)."""
    ended = (rewards.abs() >= 998).long().cumsum(1)
    return ((ended - (rewards.abs() >= 998).long()) == 0).cpu()


def summary(name: str, episodes, rec, base) -> torch.Tensor:
    returns = discounted_returns(episodes.rewards, GAMMA).double()
    out = {k: round(float(v), 4) for k, v in game_outcomes(episodes.rewards).items()}
    live = live_steps(episodes.rewards)
    acts = episodes.actions.cpu()[live]
    mix = ", ".join(f"{NAMES[a]} {(acts == a).float().mean():.3f}" for a in range(len(NAMES)))
    line = f"  {name:12s}: return {returns.mean():7.1f} +- {returns.std() / np.sqrt(n):5.1f}  {out}"
    if base is not None:
        d = returns - base
        line += f"  paired gain {d.mean():+7.1f} +- {1.96 * d.std() / np.sqrt(n):5.1f}"
    print(line)
    print(f"      live steps: action mix {mix}; chosen action flagged as a loop "
          f"{torch.stack(rec.loop_mass, 1)[live].mean():.3f}", flush=True)
    return returns


print(f"== {run_dir.name} best.pt (collection {trained.checkpoint_collection}), {n} games, tolerances {tolerances}")
episodes, rec = play(None)
base = summary("unwrapped", episodes, rec, None)
if n == settings.episodes:
    reported = json.loads((run_dir / "analysis" / "report.json").read_text())["returns"]["learned_planner"]
    out = game_outcomes(episodes.rewards)
    for key in ("win_rate", "death_rate", "unfinished_rate"):
        if abs(reported[key] - float(out[key])) > 1e-9:
            raise RuntimeError(f"Replay is not faithful: {key} {float(out[key])} vs report.json {reported[key]}.")
    print("  replay reproduces report.json")
steps = torch.stack(rec.steps, 1)[live_steps(episodes.rewards)].numpy()       # (live decisions, A)
print("  ||z' - z|| / ||z|| of each action's children (prob-weighted), quantiles 10/50/90% over live decisions:")
for a, name in enumerate(NAMES):
    q = np.percentile(steps[:, a], [10, 50, 90])
    print(f"    {name:10s} {q[0]:.4f} / {q[1]:.4f} / {q[2]:.4f}")
for tau in tolerances:
    episodes, rec = play(tau)
    summary(f"tau {tau}", episodes, rec, base)
