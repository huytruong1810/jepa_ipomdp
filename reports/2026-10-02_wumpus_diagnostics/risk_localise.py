# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/risk_localise.py
# ==============================================================================
# WHY DO THE COVERAGE-RETRAINED NETWORKS DIE: AN UNDER-RATED RISK, OR A KNOWINGLY TAKEN ONE?
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-19 night, coverage_games.py readout, user away; my call under research autonomy):
#    - On seed 0 the cover-arm networks repair the turned-latent model (glitter ratio 0.45 -> 0.77, decision share
#      M 26 -> 58%) but die in 28% of games (control 4.5%; return -190 paired vs control).
#    - The forced steps themselves are safe (forward_risk check on games_cover_seed1.pt: +1791 after-turn
#      FORWARDs, +15 risky ones; every death in both arms is a FORWARD with exact danger > 0). But the privileged
#      data are SELECTED: ~2700 per 2048 games of "turn, step into a new cell, survive", every one provably
#      safe. With a latent that separates "provably safe new cell" from "risky new cell" only partly (Step D:
#      a visited-tracking deficit), the death predictions at new cells are pulled down: a sample-selection bias
#      (Heckman 1979) in model learning, which the planner then exploits (the optimizer's curse, Smith & Winkler
#      2006). The rival reading: the risk is predicted correctly and the planner takes it knowingly, because the
#      repaired gold/glitter predictions raised the value of stepping in (a value / risk-attitude problem).
#    - The branch decides the non-privileged coverage mechanism: SELECTION -> coverage must come from the agent's
#      own choices (its own, unfiltered turn-then-step trials, so the deaths it risks are in its data: model-based
#      DAgger, Ross & Bagnell 2012), never from a safety-filtered teacher; VALUE -> coverage repairs the model,
#      and the planner needs pessimism about the remaining risk error (e.g. the double value or a risk margin).
#
# 2. Method (usage: uv run python <this> <run dir> <saved .pt> [<saved .pt> ...] [<games>, smoke only]):
#    - Each saved network set plays the run's 512 analysis games exactly as retrain_play.py does (same streams,
#      greedy, Dirichlet off); its per-game returns must equal retrain_play.py's <saved>.returns.npy.
#    - At every live decision: the planner's own root latent z, the head's P(GAME_OVER | z, FORWARD) (p_O), the
#      reward head's implied risk p_R = -(R(z, FORWARD) + 1) / 1000 (FORWARD pays -1, a death -1001), the root
#      Q-values, the chosen action, and from the exact filter (shadow tracker) the exact danger of the cell ahead
#      (P(pit) + P(live wumpus); 0 for a wall) and whether the cell ahead is unvisited.
#    - Measures per set: calibration ratios mean(p) / mean(exact danger) over live states with danger > 0
#      (all, and those with the cell ahead unvisited); the same over the risky FORWARDs the planner chose; share
#      of fatal FORWARDs where max(p_O, p_R) < 0.5 x exact danger ("under-rated"); exact danger at the fatal steps.
#    - Per-row arrays to <saved>.risk.npz.
#
# 3. Pre-registered readout (2026-10-19 night, before any run; coverage arms of runs/wumpus/coverage, seeds where
#    the cover arm's death rate is >= 2 points above control's; others reported only):
#    - SELECTION: on >= 4 of 5 such seeds (or all, if fewer), the cover set's calibration ratio max(p_O, p_R) on
#      risky unvisited-ahead live states is <= 0.67 x the control set's, AND >= 50% of the cover set's fatal
#      FORWARDs are under-rated.
#    - VALUE: the cover ratio is >= 0.85 x control's and < 30% of fatal FORWARDs are under-rated.
#    - Otherwise MIXED: report per seed; act on the larger share.
# 4. Result 2026-10-20 (runs/wumpus/coverage/risk_seed*.log; every replay = retrain_play.py's returns; all 5 seeds
#    qualify, cover deaths >= control + 2 points). **Readout: MIXED by the rule (SELECTION 2 of 5, VALUE 0 of 5);
#    acted on as SELECTION, the larger share.**
#    | Seed | ratio risky-new ctl -> cov | cov / ctl | fatal under-rated ctl / cov | phantom risk safe-new ctl -> cov | risky FWD chosen ctl -> cov | ratio at chosen ctl / cov |
#    | 0    | 0.995 -> 0.712             | 0.72      | 100 / 100%                  | 0.044 -> 0.030                   | 33 -> 279                   | 0.155 / 0.032             |
#    | 1    | 0.820 -> 0.549             | 0.67      | 100 / 100%                  | 0.060 -> 0.040                   | 19 -> 306                   | 0.034 / 0.175             |
#    | 2    | 0.828 -> 0.533             | 0.64      | 100 / 100%                  | 0.021 -> 0.006                   | 33 -> 272                   | 0.027 / 0.005             |
#    | 3    | 0.704 -> 0.619             | 0.88      | 36 / 83%                    | 0.021 -> 0.007                   | 162 -> 376                  | 0.464 / 0.139             |
#    | 4    | 1.007 -> 0.730             | 0.72      | 100 / 100%                  | 0.038 -> 0.018                   | 19 -> 270                   | 0.046 / 0.018             |
#    - VALUE is refuted: the deaths are not knowingly taken risks; 83-100% of fatal FORWARDs had the risk rated at
#      less than half of the exact danger (control too: every death is an under-rated step).
#    - The selection effect is real on 5 of 5 seeds but moderate: the cover data lower the calibration on risky
#      new cells by 12-36% and the phantom risk at provably safe new cells by 30-70%. The bar (<= 0.67x) is met on
#      2 seeds only.
#    - The amplifier is the planner: it takes 2-15x more risky FORWARDs, and at exactly those its estimate is
#      0.5-17% of the exact danger (the optimizer's curse: the argmax selects the under-estimates). A moderate,
#      broad calibration loss becomes a +28-point death rate.
#    - Implication: (i) coverage data must not be safety-filtered (the own arm, coverage_games.py section 5); (ii)
#      independently of coverage, the planner needs robustness to the risk head's error where it chooses
#      (pessimism / an uncertainty margin on P(GAME_OVER)), the same coupling as C6.
# ==============================================================================

import sys
import time
from pathlib import Path

import numpy as np
import torch

from ipomdp.agents import PlanningAgent
from ipomdp.domain import WumpusSimulator, game_outcomes
from ipomdp.domain.wumpus import GAME_OVER, GRID, NUM_CELLS, WumpusAction, WumpusBeliefTracker
from ipomdp.experiments.runs import load_trained_run
from ipomdp.experiments.wumpus import WumpusAnalysisSettings
from ipomdp.planning import BeliefTreeSearch
from ipomdp.training import discounted_returns, play_episodes, stream_seed

run_dir, saved_sets = Path(sys.argv[1]), [Path(a) for a in sys.argv[2:] if a.endswith(".pt")]
rest = [a for a in sys.argv[2:] if not a.endswith(".pt")]
if not saved_sets or len(rest) > 1:
    raise ValueError("usage: <run dir> <saved .pt> [<saved .pt> ...] [<games> (smoke only)]")
dev = torch.device("cuda")
settings = WumpusAnalysisSettings()
n = int(rest[0]) if rest else settings.episodes
trained = load_trained_run(run_dir, dev)
world, run, cfg = trained.setup.domain, trained.run, trained.cfg
model = run.eval_agent.model
FWD = int(WumpusAction.FORWARD)


class Recorder:
    """The greedy planner plus a shadow exact filter; records the risk quantities at every decision."""

    def __init__(self, planner: PlanningAgent):
        self.planner, self.batch_size = planner, n
        self.tracker = WumpusBeliefTracker(world)
        self.rows, self.t, self.start = [], 0, time.time()

    def reset(self, initial_observation):
        self.planner.reset(initial_observation)
        self.consistent, self.own = self.tracker.start(initial_observation)
        self.visited = torch.nn.functional.one_hot(self.own.cell, NUM_CELLS).bool()

    def update(self, action, observation):
        self.planner.update(action, observation)
        self.consistent, self.own = self.tracker.update(self.consistent, self.own, action, observation)
        self.visited |= torch.nn.functional.one_hot(self.own.cell, NUM_CELLS).bool()

    def act(self):
        m = self.tracker.marginals(self.consistent, self.own)
        x = m.cell % GRID + world._delta[m.heading, 0]
        y = m.cell // GRID + world._delta[m.heading, 1]
        inside = (x >= 0) & (x < GRID) & (y >= 0) & (y < GRID)
        ahead = (x + GRID * y).clamp(0, NUM_CELLS - 1).unsqueeze(1)
        danger = torch.where(inside, m.pit.gather(1, ahead).squeeze(1)
                             + m.wumpus_alive.double() * m.wumpus.gather(1, ahead).squeeze(1), 0.0)
        unvisited = inside & ~self.visited.gather(1, ahead).squeeze(1)
        with torch.no_grad():
            expansion = model.expand(self.planner.state)
        action = self.planner.act()
        self.rows.append({"live": ~m.done, "danger": danger, "unvisited": unvisited, "action": action,
                          "p_o": expansion.observation_probs[:, FWD, GAME_OVER].double(),
                          "p_r": (-(expansion.rewards[:, FWD].double() + 1.0) / 1000.0),
                          "q": torch.as_tensor(self.planner.planner.statistics.q_values, device=dev)})
        self.t += 1
        if self.t % 10 == 0:
            print(f"    decision {self.t}; {time.time() - self.start:.0f} s", flush=True)
        return action


def ratio(p: torch.Tensor, d: torch.Tensor) -> float:
    return float(p.mean() / d.mean()) if d.numel() and float(d.mean()) > 0 else float("nan")


print(f"== {run_dir.name} best.pt (collection {trained.checkpoint_collection}), sets {[str(p) for p in saved_sets]}, "
      f"{n} games", flush=True)
for saved in saved_sets:
    states = torch.load(saved, map_location=dev)
    if set(states) != set(run.networks):
        raise ValueError(f"saved networks {sorted(states)} do not match the run's {sorted(run.networks)}.")
    for name, net in run.networks.items():
        net.load_state_dict(states[name])
    planner = PlanningAgent(
        model, BeliefTreeSearch(model, cfg.mcts.num_simulations, cfg.mcts.c_puct, cfg.mcts.dirichlet_alpha,
                                dirichlet_epsilon=0.0, seed=stream_seed(settings.seed, 2),
                                observation_prune=cfg.mcts.observation_prune),
        n, temperature=0.0, seed=stream_seed(settings.seed, 3), device=dev)
    rec = Recorder(planner)
    episodes, _ = play_episodes(WumpusSimulator(world, n, cfg.env.max_steps, stream_seed(settings.seed, 1), dev), rec)
    returns = discounted_returns(episodes.rewards, world.discount).double().cpu().numpy()
    reference = np.load(saved.with_suffix(".returns.npy"))
    if n == settings.episodes and np.abs(returns - reference).max() > 1e-6:
        raise RuntimeError(f"{saved}: the replay differs from retrain_play.py's returns.")
    rows = {k: torch.stack([r[k] for r in rec.rows], 1).cpu() for k in rec.rows[0]}   # (G, T) / (G, T, A)
    fatal = torch.zeros_like(rows["live"])
    died = episodes.rewards.cpu() < -500
    fatal[died] = True
    fatal &= rows["live"] & (rows["action"] == FWD)
    live, d, p = rows["live"], rows["danger"], torch.maximum(rows["p_o"], rows["p_r"])
    risky = live & (d > 1e-9)
    risky_new = risky & rows["unvisited"]
    chosen = risky & (rows["action"] == FWD)
    under = fatal & (p < 0.5 * d)
    out = {
        "deaths": int(died.any(1).sum()), "fatal_forward": int(fatal.sum()),
        "ratio_risky": ratio(p[risky], d[risky]), "ratio_risky_new": ratio(p[risky_new], d[risky_new]),
        "ratio_o_risky_new": ratio(rows["p_o"][risky_new], d[risky_new]),
        "ratio_r_risky_new": ratio(rows["p_r"][risky_new], d[risky_new]),
        "ratio_chosen": ratio(p[chosen], d[chosen]), "risky_new_states": int(risky_new.sum()),
        "chosen_risky_forward": int(chosen.sum()),
        "under_rated_fatal": float(under.sum() / max(int(fatal.sum()), 1)),
        "fatal_danger_median": float(d[fatal].median()) if fatal.any() else float("nan"),
        "fatal_danger_over_0.4": float((d[fatal] > 0.4).double().mean()) if fatal.any() else float("nan"),
        "phantom_risk_safe_new": float(p[live & (d <= 1e-9) & rows["unvisited"]].mean()),
    }
    print(f"  {saved.stem}: return {returns.mean():.1f}, {game_outcomes(episodes.rewards)}\n    "
          + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in out.items()), flush=True)
    np.savez(saved.with_suffix(".risk.npz" if n == settings.episodes else f".risk_smoke{n}.npz"), fatal=fatal.numpy(), **{k: v.numpy() for k, v in rows.items()},
             summary=np.array(str(out)))
    del planner, rec
    torch.cuda.empty_cache()
print("DONE", flush=True)
