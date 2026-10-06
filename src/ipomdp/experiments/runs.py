# ABSOLUTE PATH: src/ipomdp/experiments/runs.py
# ==============================================================================
# RUN DIRECTORIES: BUILDING, RELOADING AND ANALYSING ONE TRAINING RUN
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. The Run Directory Is the Unit of Every Experiment:
#    - main.py (Hydra) writes a run to runs/<env>/<timestamp>_seed<seed>/ (or to an explicit
#      hydra.run.dir, as the seed sweep does). Everything downstream -- analysis, seed
#      aggregation -- takes only that directory: the resolved config is <run>/.hydra/config.yaml
#      and the networks are <run>/checkpoints/best.pt. Nothing needs the original command line.
#
# 2. One Place Turns a Config Into a Run:
#    - DOMAIN_BUILDERS and build_run_config were private to main.py, and analyze.py imported the
#      training script to reach them. They live here so that every entry point (train, analyse,
#      sweep) rebuilds a run identically and no script imports another script.
#    - DOMAIN_BUILDERS maps conf/env/<name>.yaml to a DomainSetup (training/run.py, section 6):
#      canonical Tiger is its exact FinitePOMDP sampled by BatchedPOMDPEnv; Wumpus World is
#      the generative WumpusSimulator, summarised by its win/death rates. analyze_run is the
#      Tiger analysis (it needs the exact solver); Wumpus runs are analysed by
#      experiments/wumpus.py.
#
# 3. Analysis Uses Fresh Seeds, Not the Training Evaluation:
#    - best.pt is the checkpoint with the highest greedy evaluation during training. That score
#      is the maximum of several noisy estimates and is therefore biased upwards (winner's
#      curse). analyze_run re-estimates every return on simulator seeds that training never
#      used (interpretability/analysis.py), so its numbers are unbiased for the chosen
#      checkpoint. The training score is still recorded, labelled as the selection score.
#
# 4. Outputs (all under <run>/analysis/):
#      report.json     every probe report, bound, return estimate and the geometry summary
#      geometry.png    latents of held-out random-policy histories coloured by the exact posterior
# ==============================================================================

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
import torch

from ..agents import PlanningAgent, UniformRandomAgent
from ..domain import (BatchedPOMDPEnv, FinitePOMDP, WumpusSimulator, WumpusWorld, action_value_functions,
                      build_tiger_pomdp, game_outcomes, solve_infinite_horizon)
from ..interpretability import BeliefAnalysis, analyze_beliefs, build_probe_dataset, build_wumpus_dataset, plot_latent_pca
from ..models import BeliefFilter
from ..planning import BeliefTreeSearch
from ..telemetry import BeliefGeometryVisualizer
from ..training import (Domain, DomainSetup, EpisodeBatch, Representation, RunConfig, TrainerConfig, TrainingRun,
                        load_checkpoint, play_episodes, stream_seed)


def tiger_setup(device: torch.device) -> DomainSetup:
    """Canonical Tiger: the exact FinitePOMDP and its sampler; no extra episode summary."""
    pomdp = build_tiger_pomdp()
    return DomainSetup(pomdp, lambda batch, steps, seed, dev: BatchedPOMDPEnv(pomdp, batch, steps, seed, dev),
                       lambda episodes: {})


def wumpus_setup(device: torch.device) -> DomainSetup:
    """4x4 Wumpus World: generative simulator (tables on `device`), win/death rates per window."""
    world = WumpusWorld(device)
    return DomainSetup(world, lambda batch, steps, seed, dev: WumpusSimulator(world, batch, steps, seed, dev),
                       lambda episodes: game_outcomes(episodes.rewards))


# Domains selectable through conf/env/<name>.yaml (section 2).
DOMAIN_BUILDERS = {"tiger": tiger_setup, "wumpus": wumpus_setup}


def build_run_config(cfg: DictConfig) -> RunConfig:
    """Flattens the Hydra config (conf/config.yaml) into the typed RunConfig."""
    t, m, a, s = cfg.training, cfg.model, cfg.agent, cfg.mcts
    return RunConfig(
        seed=cfg.seed, episode_length=cfg.env.max_steps, env_batch_size=t.env_batch_size,
        warmup_episodes=t.warmup_episodes, buffer_capacity=t.buffer_capacity, batch_size=t.batch_size,
        updates_per_collection=t.updates_per_collection, eval_episodes=t.eval_episodes,
        latent_dim=m.latent_dim, hidden_dim=m.hidden_dim, num_blocks=m.num_blocks,
        representation=Representation(m.representation), observation_horizon=m.observation_horizon,
        ema_momentum=m.ema_momentum,
        acting_ema_momentum=m.acting_ema_momentum,
        num_bins=m.num_bins, trainer=TrainerConfig(**cfg.trainer), num_simulations=s.num_simulations,
        observation_prune=s.observation_prune,
        c_puct=s.c_puct, dirichlet_alpha=s.dirichlet_alpha, dirichlet_epsilon=s.dirichlet_epsilon,
        temperature=a.temperature, temperature_min=a.temperature_min, temperature_decay=a.temperature_decay,
        epsilon=a.epsilon, epsilon_min=a.epsilon_min,
        store_eval_episodes=t.store_eval_episodes)


def build_training_run(cfg: DictConfig, device: torch.device) -> tuple[DomainSetup, TrainingRun]:
    """The domain named by cfg.env.name and a freshly initialised TrainingRun over it."""
    setup = DOMAIN_BUILDERS[cfg.env.name](device)
    return setup, TrainingRun(setup, build_run_config(cfg), device)


def latent_figure(domain: Domain, belief_filter: BeliefFilter, episodes: EpisodeBatch, save_dir: Path,
                  filename: str) -> plt.Figure:
    """
    The domain's latent-geometry figure, saved as <save_dir>/<filename>.png (for main.py's visualisation).

    Tiger: one PCA plane coloured by the exact posterior of the first state. Wumpus World: one PCA
    plane coloured by the hunter's cell and by the remaining layout uncertainty.
    """
    if isinstance(domain, FinitePOMDP):
        dataset = build_probe_dataset(domain, belief_filter, episodes)
        return BeliefGeometryVisualizer(str(save_dir)).plot(dataset.latents, dataset.posteriors[:, 0],
                                                            domain.state_names[0], filename=filename)
    if isinstance(domain, WumpusWorld):
        save_dir.mkdir(parents=True, exist_ok=True)
        figure = plot_latent_pca(build_wumpus_dataset(domain, belief_filter, episodes))
        figure.savefig(save_dir / f"{filename}.png")
        return figure
    raise TypeError(f"No latent figure for domain {type(domain).__name__}.")


@dataclass(frozen=True)
class TrainedRun:
    """A run directory reloaded with the networks of its best.pt (section 1)."""

    run_dir: Path
    cfg: DictConfig
    setup: DomainSetup
    run: TrainingRun
    checkpoint_collection: int
    checkpoint_eval_return: float     # training-time selection score (biased upwards, section 3)


def load_trained_run(run_dir: Path, device: torch.device) -> TrainedRun:
    """Rebuilds the run from <run>/.hydra/config.yaml and loads <run>/checkpoints/best.pt."""
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    setup, run = build_training_run(cfg, device)
    best = load_checkpoint(run_dir / "checkpoints" / "best.pt")
    for name, net in run.networks.items():
        net.load_state_dict(best["networks"][name])
    run.belief_filter.eval()
    return TrainedRun(run_dir, cfg, setup, run, best["collection"], best["eval_return"])


@dataclass(frozen=True)
class AnalysisSettings:
    """
    Attributes:
        episodes: Episodes per probe data set and per return estimate.
        solver_tolerance: Certified ||V* - V_n||_inf of the reference solution.
        mlp_probe_steps: Adam steps of each MLP probe fit.
        seed: Base seed of the analysis streams (training/seeding.py): 1-6 inside
            analyze_beliefs, 7-8 the greedy planner and its agent, 9-10 the geometry figure.
            Hashed streams never coincide with a training run's streams.
    """

    episodes: int = 512
    solver_tolerance: float = 0.01
    mlp_probe_steps: int = 3000
    seed: int = 12345


def analyze_run(run_dir: Path, settings: AnalysisSettings, device: torch.device) -> BeliefAnalysis:
    """Runs interpretability.analyze_beliefs on a run's best.pt and writes <run>/analysis/ (section 4)."""
    trained = load_trained_run(run_dir, device)
    pomdp, run, cfg = trained.setup.domain, trained.run, trained.cfg
    if not isinstance(pomdp, FinitePOMDP):
        raise TypeError(f"analyze_run needs an exact FinitePOMDP; {cfg.env.name} runs use experiments/wumpus.py.")
    solution = solve_infinite_horizon(pomdp, tolerance=settings.solver_tolerance, prune_epsilon=1e-6)
    action_values = action_value_functions(pomdp, solution.value_function, 1e-6)
    model = run.eval_agent.model
    planner = PlanningAgent(
        model, BeliefTreeSearch(model, cfg.mcts.num_simulations, cfg.mcts.c_puct, cfg.mcts.dirichlet_alpha,
                                dirichlet_epsilon=0.0, seed=stream_seed(settings.seed, 7)),
        settings.episodes, temperature=0.0, seed=stream_seed(settings.seed, 8), device=device)
    analysis = analyze_beliefs(pomdp, run.acting_filter, planner, solution.value_function, action_values,
                               solution.error_bound, cfg.env.max_steps, settings.episodes,
                               settings.mlp_probe_steps, settings.seed, device)

    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)
    report = asdict(analysis) | {"checkpoint_collection": trained.checkpoint_collection,
                                 "checkpoint_eval_return": trained.checkpoint_eval_return,
                                 "settings": asdict(settings)}
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    episodes, _ = play_episodes(
        BatchedPOMDPEnv(pomdp, settings.episodes, cfg.env.max_steps, stream_seed(settings.seed, 9), device),
        UniformRandomAgent(pomdp.num_actions, settings.episodes, stream_seed(settings.seed, 10), device))
    dataset = build_probe_dataset(pomdp, run.acting_filter, episodes)
    plt.close(BeliefGeometryVisualizer(str(out)).plot(dataset.latents, dataset.posteriors[:, 0],
                                                      pomdp.state_names[0], filename="geometry"))
    return analysis
