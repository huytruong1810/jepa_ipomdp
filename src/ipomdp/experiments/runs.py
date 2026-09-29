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
from ..domain import BatchedPOMDPEnv, FinitePOMDP, action_value_functions, build_tiger_pomdp, solve_infinite_horizon
from ..interpretability import BeliefAnalysis, analyze_beliefs, build_probe_dataset
from ..planning import BeliefTreeSearch
from ..telemetry import BeliefGeometryVisualizer
from ..training import RunConfig, TrainerConfig, TrainingRun, load_checkpoint, play_episodes

# Domains selectable through conf/env/<name>.yaml. Each builder returns the exact model.
DOMAIN_BUILDERS = {"tiger": build_tiger_pomdp}


def build_run_config(cfg: DictConfig) -> RunConfig:
    """Flattens the Hydra config (conf/config.yaml) into the typed RunConfig."""
    t, m, a, s = cfg.training, cfg.model, cfg.agent, cfg.mcts
    return RunConfig(
        seed=cfg.seed, episode_length=cfg.env.max_steps, env_batch_size=t.env_batch_size,
        warmup_episodes=t.warmup_episodes, buffer_capacity=t.buffer_capacity, batch_size=t.batch_size,
        updates_per_collection=t.updates_per_collection, eval_episodes=t.eval_episodes,
        latent_dim=m.latent_dim, hidden_dim=m.hidden_dim, num_blocks=m.num_blocks, ema_momentum=m.ema_momentum,
        num_bins=m.num_bins, trainer=TrainerConfig(**cfg.trainer), num_simulations=s.num_simulations,
        c_puct=s.c_puct, dirichlet_alpha=s.dirichlet_alpha, dirichlet_epsilon=s.dirichlet_epsilon,
        temperature=a.temperature, temperature_min=a.temperature_min, temperature_decay=a.temperature_decay)


def build_training_run(cfg: DictConfig, device: torch.device) -> tuple[FinitePOMDP, TrainingRun]:
    """The exact domain named by cfg.env.name and a freshly initialised TrainingRun over it."""
    pomdp = DOMAIN_BUILDERS[cfg.env.name]()
    return pomdp, TrainingRun(pomdp, build_run_config(cfg), device)


@dataclass(frozen=True)
class TrainedRun:
    """A run directory reloaded with the networks of its best.pt (section 1)."""

    run_dir: Path
    cfg: DictConfig
    pomdp: FinitePOMDP
    run: TrainingRun
    checkpoint_collection: int
    checkpoint_eval_return: float     # training-time selection score (biased upwards, section 3)


def load_trained_run(run_dir: Path, device: torch.device) -> TrainedRun:
    """Rebuilds the run from <run>/.hydra/config.yaml and loads <run>/checkpoints/best.pt."""
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    pomdp, run = build_training_run(cfg, device)
    best = load_checkpoint(run_dir / "checkpoints" / "best.pt")
    for name, net in run.networks.items():
        net.load_state_dict(best["networks"][name])
    run.world_model.eval()
    return TrainedRun(run_dir, cfg, pomdp, run, best["collection"], best["eval_return"])


@dataclass(frozen=True)
class AnalysisSettings:
    """
    Attributes:
        episodes: Episodes per probe data set and per return estimate.
        solver_tolerance: Certified ||V* - V_n||_inf of the reference solution.
        mlp_probe_steps: Adam steps of each MLP probe fit.
        seed: Base simulator seed; disjoint from every stream training uses (training/run.py).
    """

    episodes: int = 512
    solver_tolerance: float = 0.01
    mlp_probe_steps: int = 3000
    seed: int = 12345


def analyze_run(run_dir: Path, settings: AnalysisSettings, device: torch.device) -> BeliefAnalysis:
    """Runs interpretability.analyze_beliefs on a run's best.pt and writes <run>/analysis/ (section 4)."""
    trained = load_trained_run(run_dir, device)
    pomdp, run, cfg = trained.pomdp, trained.run, trained.cfg
    solution = solve_infinite_horizon(pomdp, tolerance=settings.solver_tolerance, prune_epsilon=1e-6)
    action_values = action_value_functions(pomdp, solution.value_function, 1e-6)
    model = run.eval_agent.model
    planner = PlanningAgent(
        model, BeliefTreeSearch(model, cfg.mcts.num_simulations, cfg.mcts.c_puct, cfg.mcts.dirichlet_alpha,
                                dirichlet_epsilon=0.0, seed=settings.seed),
        settings.episodes, temperature=0.0, seed=settings.seed, device=device)
    analysis = analyze_beliefs(pomdp, run.world_model.belief_filter, planner, solution.value_function, action_values,
                               solution.error_bound, cfg.env.max_steps, settings.episodes,
                               settings.mlp_probe_steps, settings.seed, device)

    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)
    report = asdict(analysis) | {"checkpoint_collection": trained.checkpoint_collection,
                                 "checkpoint_eval_return": trained.checkpoint_eval_return,
                                 "settings": asdict(settings)}
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    episodes, _ = play_episodes(
        BatchedPOMDPEnv(pomdp, settings.episodes, cfg.env.max_steps, settings.seed + 99, device),
        UniformRandomAgent(pomdp.num_actions, settings.episodes, settings.seed + 99, device))
    dataset = build_probe_dataset(pomdp, run.world_model.belief_filter, episodes)
    plt.close(BeliefGeometryVisualizer(str(out)).plot(dataset.latents, dataset.posteriors[:, 0],
                                                      pomdp.state_names[0], filename="geometry"))
    return analysis
