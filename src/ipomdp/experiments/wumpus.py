# ABSOLUTE PATH: src/ipomdp/experiments/wumpus.py
# ==============================================================================
# WUMPUS WORLD RUN ANALYSIS (NO SOLVER: RETURNS, OUTCOMES, LATENT PROBES, BELIEF MAPS)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. What Can Be Measured Without an Exact Solver (user decision: no solver for Wumpus):
#    - Performance: discounted return, win, death and unfinished rates of the greedy learned
#      planner on fresh simulator seeds (runs.py, section 3: the training-time best.pt score
#      is a biased maximum). A uniformly random hunter on the SAME layouts (same seed) is the
#      floor. There is no optimum; the exact-belief reference agent (agents/wumpus_reference.py)
#      on the same layouts is the yardstick from below (any policy's return <= V*), one entry
#      per risk threshold in the settings.
#    - Representation: probes from the agent's acting latents to the exact enumeration filter
#      (interpretability/wumpus_latents.py), fitted and evaluated on disjoint episodes, on
#      random-policy histories (broad coverage, but short: a random hunter dies quickly) and on
#      the agent's own histories (the beliefs its decisions depend on).
#
# 2. Outputs (under <run>/analysis/):
#      report.json       returns, outcomes, probe reports
#      belief_maps.png   exact vs decoded hazard maps along the longest held-out agent game
#      latent_pca.png    agent latents coloured by hunter cell and by remaining uncertainty
# ==============================================================================

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from ..agents import PlanningAgent, UniformRandomAgent, WumpusReferenceAgent
from ..domain import WumpusSimulator, WumpusWorld, game_outcomes
from ..interpretability import (build_wumpus_dataset, evaluate_wumpus_probe, fit_linear_wumpus_probe,
                                fit_mlp_wumpus_probe, plot_belief_maps, plot_latent_pca)
from ..planning import BeliefTreeSearch
from ..training import EpisodeBatch, discounted_returns, play_episodes, stream_seed
from .runs import load_trained_run


@dataclass(frozen=True)
class WumpusAnalysisSettings:
    """
    Attributes:
        episodes: Games per return estimate and per probe data set (split in halves for fit / held-out).
        mlp_probe_steps: Adam steps of each MLP probe fit.
        seed: Base seed of the analysis streams (training/seeding.py); never a training stream.
        reference_risk_thresholds: Risk thresholds of the reference agents (agents/wumpus_reference.py,
            section 2d): 0 never gambles, 1 always does.
    """

    episodes: int = 512
    mlp_probe_steps: int = 3000
    seed: int = 12345
    reference_risk_thresholds: tuple[float, ...] = (0.0, 0.4, 1.0)


def _halves(episodes: EpisodeBatch) -> tuple[EpisodeBatch, EpisodeBatch]:
    """Two disjoint episode sets (fit, held-out)."""
    half = episodes.actions.shape[0] // 2
    pick = lambda rows: EpisodeBatch(episodes.actions[rows], episodes.observations[rows],  # noqa: E731
                                     episodes.rewards[rows], episodes.initial_observations[rows])
    return pick(slice(0, half)), pick(slice(half, None))


def _summary(episodes: EpisodeBatch, discount: float) -> dict[str, float]:
    returns = discounted_returns(episodes.rewards, discount)
    return {"discounted_return": float(returns.mean()),
            "discounted_return_stderr": float(returns.std()) / len(returns) ** 0.5,
            **game_outcomes(episodes.rewards)}


def analyze_wumpus_run(run_dir: Path, settings: WumpusAnalysisSettings, device: torch.device) -> dict:
    """Section 1 on a run's best.pt; writes <run>/analysis/ (section 2) and returns the report."""
    trained = load_trained_run(run_dir, device)
    world, run, cfg = trained.setup.domain, trained.run, trained.cfg
    if not isinstance(world, WumpusWorld):
        raise TypeError(f"analyze_wumpus_run needs a Wumpus World run, got {cfg.env.name}.")
    n, length = settings.episodes, cfg.env.max_steps
    simulator = lambda: WumpusSimulator(world, n, length, stream_seed(settings.seed, 1), device)  # noqa: E731

    model = run.eval_agent.model
    planner = PlanningAgent(
        model, BeliefTreeSearch(model, cfg.mcts.num_simulations, cfg.mcts.c_puct, cfg.mcts.dirichlet_alpha,
                                dirichlet_epsilon=0.0, seed=stream_seed(settings.seed, 2),
                                observation_prune=cfg.mcts.observation_prune),
        n, temperature=0.0, seed=stream_seed(settings.seed, 3), device=device)
    agent_episodes, _ = play_episodes(simulator(), planner)
    random_episodes, _ = play_episodes(simulator(), UniformRandomAgent(world.num_actions, n,
                                                                       stream_seed(settings.seed, 4), device))
    report: dict = {"checkpoint_collection": trained.checkpoint_collection,
                    "checkpoint_eval_return": trained.checkpoint_eval_return,
                    "settings": asdict(settings),
                    "returns": {"learned_planner": _summary(agent_episodes, world.discount),
                                "random": _summary(random_episodes, world.discount)},
                    "probes": {}}
    for threshold in settings.reference_risk_thresholds:
        reference_episodes, _ = play_episodes(simulator(), WumpusReferenceAgent(world, n, threshold))
        report["returns"][f"reference_risk{threshold:g}"] = _summary(reference_episodes, world.discount)

    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)
    for source, episodes in (("random", random_episodes), ("agent", agent_episodes)):
        fit_episodes, held_episodes = _halves(episodes)
        fit = build_wumpus_dataset(world, run.acting_filter, fit_episodes)
        held = build_wumpus_dataset(world, run.acting_filter, held_episodes)
        mlp = fit_mlp_wumpus_probe(fit, steps=settings.mlp_probe_steps)
        report["probes"][f"linear_{source}"] = asdict(evaluate_wumpus_probe(fit_linear_wumpus_probe(fit), fit, held))
        report["probes"][f"mlp_{source}"] = asdict(evaluate_wumpus_probe(mlp, fit, held))
        if source == "agent":
            lengths = torch.bincount(held.episode)
            figure = plot_belief_maps(held, mlp, episode=int(lengths.argmax()))
            figure.savefig(out / "belief_maps.png")
            plt.close(figure)
            figure = plot_latent_pca(held)
            figure.savefig(out / "latent_pca.png")
            plt.close(figure)

    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return report
