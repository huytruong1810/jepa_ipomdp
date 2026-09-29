# ABSOLUTE PATH: src/ipomdp/interpretability/analysis.py
# ==============================================================================
# BELIEF ANALYSIS OF A TRAINED AGENT: PROBES, GUARANTEES, BAYES GAP, GEOMETRY
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Three Levels of Evidence, Each Against an Exact Reference:
#    (1) Representation: probes psi from latents to exact posteriors, fitted on one episode set
#        and scored on another (belief_probe.py), on two data distributions:
#          random   uniformly random behaviour -- broad coverage of belief space;
#          agent    the trained greedy planner's own histories -- the beliefs its decisions use.
#    (2) Guarantees: value-error, one-step-regret and discounted-policy-loss bounds derived from
#        the measured decoding error (error_bounds.py), each next to its measured counterpart.
#    (3) Behaviour: discounted returns on the SAME simulator seeds of
#          optimal          exact beliefs, argmax Q*          (the Bayes-optimal reference)
#          decoded-belief   learned filter + probe, argmax Q*(psi(z))  (is the latent
#                           Bayes-sufficient for optimal DECISIONS?)
#          learned planner  the trained agent itself (greedy search over the learned model)
#        The decoded-belief agent isolates the representation: it uses the exact Q*, so any
#        return gap to the optimum is caused by the latent (and the probe), never by the learned
#        reward/value/observation heads or the search.
#
# 2. Geometry:
#    - Principal components of the latents (explained-variance spectrum) and, for two-state
#      domains, the |Spearman| rank correlation of each of the top components with the exact
#      posterior log-odds (the "side") and with |log-odds| (the confidence).
#    - Minimality ratio E_b[tr Var(z | b*)] / tr Var(z): the share of latent variance NOT
#      explained by the exact posterior. 0 means z is a function of b* alone (a minimal
#      sufficient statistic); > 0 means the latent also separates histories with identical
#      posteriors.
#    - Phase-6 observation (canonical Tiger, trained agent): PC1 (68% of variance) is not the
#      side (|Spearman| 0.07 with log-odds); PC2 is. PC1 separates the reset state (b = 0.5
#      after a door opening) from listening histories, including b = 0.5 after balanced
#      listening -- the latent is sufficient but not minimal. A Bayes filter for Tiger only
#      needs the one-dimensional log-odds, so this measures how much extra history the learned
#      filter retains.
#
# 3. Exact References Come From the Solver:
#    - V* and Q* are the certified infinite-horizon solution (domain/solver.py) passed in by the
#      caller; its certified error is reported with the results.
# ==============================================================================

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ..agents import PlanningAgent, UniformRandomAgent
from ..domain import AlphaVectorSet, BatchedPOMDPEnv, FinitePOMDP, belief_update, initial_beliefs
from ..models.world_model import BeliefFilter
from ..training.rollout import discounted_returns, play_episodes
from ..training.seeding import stream_seed
from .belief_probe import (BeliefProbe, ProbeDataset, ProbeReport, build_probe_dataset, evaluate_probe,
                           fit_linear_probe, fit_mlp_probe)
from .error_bounds import ErrorBoundReport, error_bounds, q_values


class ExactBeliefAgent:
    """Tracks the exact posterior and acts greedily on Q*(b, a) (the Bayes-optimal reference)."""

    def __init__(self, model: FinitePOMDP, action_values: list[AlphaVectorSet], batch_size: int,
                 device: torch.device):
        self.model, self.action_values, self.batch_size, self.device = model, action_values, batch_size, device
        self.belief = initial_beliefs(model, batch_size, device)

    def reset(self) -> None:
        self.belief = initial_beliefs(self.model, self.batch_size, self.device)

    def act(self) -> Tensor:
        return q_values(self.action_values, self.belief.cpu()).argmax(-1).to(self.device)

    def update(self, action: Tensor, observation: Tensor) -> None:
        self.belief = belief_update(self.model, self.belief, action, observation)


class DecodedBeliefAgent:
    """Tracks the LEARNED latent, decodes it with a probe and acts greedily on Q*(psi(z), a)."""

    def __init__(self, belief_filter: BeliefFilter, probe: BeliefProbe, action_values: list[AlphaVectorSet],
                 num_actions: int, num_observations: int, batch_size: int):
        self.belief_filter, self.probe, self.action_values = belief_filter, probe, action_values
        self.num_actions, self.num_observations, self.batch_size = num_actions, num_observations, batch_size
        self.latent = belief_filter.initial(batch_size)

    def reset(self) -> None:
        self.latent = self.belief_filter.initial(self.batch_size)

    @torch.no_grad()
    def act(self) -> Tensor:
        decoded = self.probe(self.latent).cpu()
        return q_values(self.action_values, decoded).argmax(-1).to(self.latent.device)

    @torch.no_grad()
    def update(self, action: Tensor, observation: Tensor) -> None:
        self.latent = self.belief_filter.step(self.latent, F.one_hot(action, self.num_actions).float(),
                                              F.one_hot(observation, self.num_observations).float())


@dataclass(frozen=True)
class GeometryReport:
    """Section 2: spectrum, per-component rank correlations (two-state domains) and minimality ratio."""

    explained_variance_ratio: list[float]
    log_odds_spearman: list[float] | None       # |Spearman(PC_k, log-odds)| for the top components
    confidence_spearman: list[float] | None     # |Spearman(PC_k, |log-odds|)|
    minimality_ratio: float


@dataclass(frozen=True)
class ReturnEstimate:
    """Mean discounted return and its standard error over episodes."""

    mean: float
    stderr: float


@dataclass(frozen=True)
class BeliefAnalysis:
    """Everything the analysis measures (module header)."""

    optimal_value_b0: float
    solver_error_bound: float
    probes: dict[str, ProbeReport]            # keys "<linear|mlp>/<random|agent>"
    bounds: dict[str, ErrorBoundReport]       # keys "<linear|mlp>/agent"
    returns: dict[str, ReturnEstimate]        # keys "optimal", "decoded_<linear|mlp>", "learned_planner"
    geometry: GeometryReport


def _average_ranks(values: Tensor) -> Tensor:
    """Ranks with ties sharing their average rank (exact posteriors take few distinct values)."""
    unique, inverse, counts = torch.unique(values, return_inverse=True, return_counts=True)
    end = counts.cumsum(0).double()
    return ((end - counts.double()) + (end - 1)).div(2)[inverse]


def _spearman(x: Tensor, y: Tensor) -> float:
    """Spearman rank correlation (Pearson correlation of tie-averaged ranks)."""
    rx, ry = _average_ranks(x), _average_ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm()))


def latent_geometry(dataset: ProbeDataset, components: int = 5) -> GeometryReport:
    """Section 2 (computed on the CPU in float64)."""
    latents = dataset.latents.double().cpu()
    centred = latents - latents.mean(0)
    _, singular_values, vh = torch.linalg.svd(centred, full_matrices=False)
    variance = singular_values ** 2
    ratios = (variance / variance.sum())[:components].tolist()
    side = confidence = None
    if dataset.posteriors.shape[1] == 2:
        p = dataset.posteriors[:, 0].cpu().clamp(1e-12, 1 - 1e-12)
        log_odds = (p / (1 - p)).log()
        projections = centred @ vh[:3].T
        side = [abs(_spearman(projections[:, k], log_odds)) for k in range(projections.shape[1])]
        confidence = [abs(_spearman(projections[:, k], log_odds.abs())) for k in range(projections.shape[1])]
    key = dataset.posteriors.cpu().round(decimals=6)
    _, group = torch.unique(key, dim=0, return_inverse=True)
    within = torch.zeros(int(group.max()) + 1, dtype=torch.float64).index_add_(
        0, group, torch.ones(len(group), dtype=torch.float64))
    group_mean = torch.zeros(len(within), latents.shape[1], dtype=torch.float64).index_add_(0, group, latents)
    group_mean = group_mean / within.unsqueeze(1)
    residual = ((latents - group_mean[group]) ** 2).sum()
    return GeometryReport(ratios, side, confidence, float(residual / (centred ** 2).sum()))


def _returns(env: BatchedPOMDPEnv, agent, discount: float) -> ReturnEstimate:
    returns = discounted_returns(play_episodes(env, agent)[0].rewards, discount)
    return ReturnEstimate(float(returns.mean()), float(returns.std()) / float(np.sqrt(len(returns))))


def analyze_beliefs(
    model: FinitePOMDP,
    belief_filter: BeliefFilter,
    planner_agent: PlanningAgent,
    optimal_value: AlphaVectorSet,
    optimal_action_values: list[AlphaVectorSet],
    solver_error_bound: float,
    episode_length: int,
    num_episodes: int,
    mlp_probe_steps: int,
    seed: int,
    device: torch.device,
) -> BeliefAnalysis:
    """
    Runs the full analysis (module header).

    Args:
        model: Exact domain.
        belief_filter: The trained agent's filter.
        planner_agent: The trained greedy planning agent (batch size = num_episodes).
        optimal_value, optimal_action_values: Certified V* and per-action Q* vector sets.
        solver_error_bound: Certified ||V* - V_n||_inf of those vectors.
        episode_length: Steps per episode.
        num_episodes: Episodes per data set and per return estimate.
        mlp_probe_steps: Adam steps of each MLP probe fit.
        seed: Base seed. Streams 1-4 seed the probe data sets, 5 the common simulator of the return
            estimates, 6 the uniform agent (training/seeding.py); the caller seeds planner_agent
            from other streams of the same base.
        device: Device of the networks.
    """
    if planner_agent.batch_size != num_episodes:
        raise ValueError("planner_agent.batch_size must equal num_episodes.")
    env = lambda stream: BatchedPOMDPEnv(model, num_episodes, episode_length, stream_seed(seed, stream), device)  # noqa: E731
    uniform = UniformRandomAgent(model.num_actions, num_episodes, stream_seed(seed, 6), device)
    data = {
        "random": (build_probe_dataset(model, belief_filter, play_episodes(env(1), uniform)[0]),
                   build_probe_dataset(model, belief_filter, play_episodes(env(2), uniform)[0])),
        "agent": (build_probe_dataset(model, belief_filter, play_episodes(env(3), planner_agent)[0]),
                  build_probe_dataset(model, belief_filter, play_episodes(env(4), planner_agent)[0])),
    }
    probes: dict[str, ProbeReport] = {}
    fitted: dict[str, BeliefProbe] = {}
    fitters = (("linear", fit_linear_probe), ("mlp", lambda d: fit_mlp_probe(d, steps=mlp_probe_steps)))
    for kind, fit in fitters:
        for source, (fit_set, eval_set) in data.items():
            probe = fit(fit_set)
            probes[f"{kind}/{source}"] = evaluate_probe(probe, eval_set)
            fitted[f"{kind}/{source}"] = probe

    bounds = {}
    for kind in ("linear", "mlp"):
        eval_set = data["agent"][1]
        bounds[f"{kind}/agent"] = error_bounds(optimal_value, optimal_action_values, model.discount,
                                               eval_set.posteriors, fitted[f"{kind}/agent"](eval_set.latents))

    returns = {"optimal": _returns(env(5), ExactBeliefAgent(model, optimal_action_values, num_episodes, device),
                                   model.discount)}
    # The decoded-belief agent uses the probe fitted on RANDOM data: the agent-data probe need not
    # cover the beliefs this different policy visits. All return estimates share one simulator seed
    # (common random numbers), which reduces the variance of their differences.
    for kind in ("linear", "mlp"):
        agent = DecodedBeliefAgent(belief_filter, fitted[f"{kind}/random"], optimal_action_values,
                                   model.num_actions, model.num_observations, num_episodes)
        returns[f"decoded_{kind}"] = _returns(env(5), agent, model.discount)
    returns["learned_planner"] = _returns(env(5), planner_agent, model.discount)

    return BeliefAnalysis(
        optimal_value_b0=float(optimal_value.value(model.initial_belief.unsqueeze(0))),
        solver_error_bound=solver_error_bound,
        probes=probes, bounds=bounds, returns=returns,
        geometry=latent_geometry(data["random"][1]),
    )
