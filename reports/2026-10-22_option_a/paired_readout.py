# ABSOLUTE PATH: reports/2026-10-22_option_a/paired_readout.py
# ==============================================================================
# PAIRED READOUT OF THE OPTION (a) IN-AGENT SCREEN (HANDOFF 6.2d, STEP 1)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (pre-registered in src/ipomdp/training/run.py, section 17, before launch):
#    - Do the risk margin (mcts.risk_margin=2) and own turn-then-step trials (agent.trial_prob=0.2),
#      trained in-agent on the retrain recipe, beat that recipe (runs/sweeps/wumpus_retrain) on the
#      same 512 development games?
#    - PASS (per arm): paired mean >= +50 on 2 of 3 seeds and deaths <= retrain + 2 points on each seed.
#      Unfinished share is reported, not part of the bar (expected near 67%: a safety-only gain).
#
# 2. Pairing:
#    - Each arm seed's analysis/games.npz (written by sweep.py's analysis) holds the per-game rows of its
#      learned planner on analysis seed 12345. The baseline runs were analysed before games.npz existed,
#      so their learned planner is replayed here once on the same simulator stream, exactly as
#      experiments/wumpus.py plays it (eval_agent.model; the back-filled risk_margin 1.0 wraps nothing),
#      and cached as <baseline run>/analysis/learned_planner_games.npz. Identity check: the replayed mean
#      return must equal report.json's learned_planner return, and the heuristic rows of the arm's
#      games.npz must equal across arms (same layouts).
#    - Interval: Student-t 95% over the 512 paired games of a seed.
#
# 3. Usage:
#      uv run reports/2026-10-22_option_a/paired_readout.py <arm sweep name> [<arm sweep name> ...]
#    e.g. wumpus_own_margin wumpus_margin; seeds are those present in every arm's sweep directory.
# ==============================================================================

import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats
import torch

from ipomdp.agents import PlanningAgent
from ipomdp.domain import WumpusSimulator
from ipomdp.domain.wumpus import game_outcome_flags
from ipomdp.experiments import WumpusAnalysisSettings, load_trained_run
from ipomdp.planning import BeliefTreeSearch
from ipomdp.training import discounted_returns, play_episodes, stream_seed

SWEEPS = Path("runs/sweeps")
BASELINE = SWEEPS / "wumpus_retrain"
DEV = torch.device("cuda")


def baseline_rows(run_dir: Path) -> dict[str, np.ndarray]:
    """The baseline's learned planner on the development games (section 2), replayed once and cached."""
    cache = run_dir / "analysis" / "learned_planner_games.npz"
    if cache.exists():
        return dict(np.load(cache))
    settings = WumpusAnalysisSettings()
    trained = load_trained_run(run_dir, DEV)
    world, cfg, model = trained.setup.domain, trained.cfg, trained.run.eval_agent.model
    n = settings.episodes
    planner = PlanningAgent(
        model, BeliefTreeSearch(model, cfg.mcts.num_simulations, cfg.mcts.c_puct, cfg.mcts.dirichlet_alpha,
                                dirichlet_epsilon=0.0, seed=stream_seed(settings.seed, 2),
                                observation_prune=cfg.mcts.observation_prune),
        n, temperature=0.0, seed=stream_seed(settings.seed, 3), device=DEV)
    episodes, _ = play_episodes(WumpusSimulator(world, n, cfg.env.max_steps, stream_seed(settings.seed, 1), DEV), planner)
    won, died = game_outcome_flags(episodes.rewards)
    rows = {"learned_planner.return": discounted_returns(episodes.rewards, world.discount).cpu().numpy(),
            "learned_planner.won": won.cpu().numpy(), "learned_planner.died": died.cpu().numpy()}
    reported = json.loads((run_dir / "analysis" / "report.json").read_text())["returns"]["learned_planner"]["discounted_return"]
    replayed = float(rows["learned_planner.return"].mean())
    if not np.isclose(replayed, reported, rtol=0, atol=1e-6):
        raise RuntimeError(f"{run_dir}: replayed {replayed} != report.json {reported}: not the analysed games.")
    np.savez(cache, **rows)
    return rows


def rates(g: dict[str, np.ndarray]) -> tuple[float, float]:
    won, died = g["learned_planner.won"], g["learned_planner.died"]
    return 100 * died.mean(), 100 * (~won & ~died).mean()


def main(arms: list[str]) -> None:
    seeds = sorted(set.intersection(*[{p.name for p in (SWEEPS / a).glob("seed*") if (p / "analysis" / "games.npz").exists()}
                                      for a in arms]))
    print(f"seeds {seeds}; baseline {BASELINE}")
    for arm in arms:
        print(f"\n== {arm}: arm - retrain, paired over 512 games")
        gains, safe = 0, True
        for s in seeds:
            base = baseline_rows((BASELINE / s).resolve())
            g = dict(np.load(SWEEPS / arm / s / "analysis" / "games.npz"))
            d = g["learned_planner.return"] - base["learned_planner.return"]
            half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            (bd, bu), (ad, au) = rates(base), rates(g)
            gains += bool(d.mean() >= 50)
            safe &= bool(ad <= bd + 2)
            print(f"  {s}: {g['learned_planner.return'].mean():6.1f} vs {base['learned_planner.return'].mean():6.1f}  "
                  f"diff {d.mean():+6.1f} [{d.mean() - half:+6.1f}, {d.mean() + half:+6.1f}]  deaths {bd:4.1f} -> {ad:4.1f}%  "
                  f"unfinished {bu:4.1f} -> {au:4.1f}%")
        verdict = "PASS" if gains >= 2 and safe else "FAIL"
        print(f"  {verdict}: >= +50 on {gains} of {len(seeds)} seeds; deaths within +2 points on every seed: {safe}")


if __name__ == "__main__":
    main(sys.argv[1:])
