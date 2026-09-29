# ABSOLUTE PATH: src/ipomdp/interpretability/belief_probe.py
# ==============================================================================
# BELIEF PROBES: DECODING THE EXACT BAYES POSTERIOR FROM THE LEARNED LATENT
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. The Question a Probe Answers:
#    - The filter's latent z_t = phi(h_t) is a Bayes-sufficient belief iff there is a map psi with
#      psi(z_t) = b*(h_t) for the exact posterior b* (src/ipomdp/domain/belief.py). A probe is a
#      fitted psi; its error on HELD-OUT histories is what the error bounds of error_bounds.py
#      turn into guarantees on values and decisions.
#
# 2. Data From Any Episodes:
#    - A ProbeDataset is built from recorded episodes (training/rollout.py play_episodes), so the
#      same probe can be fitted/evaluated on random-policy histories (broad coverage) or on the
#      trained agent's own histories (the beliefs its decisions actually depend on). Exact
#      posteriors are replayed with the domain's Bayes filter; latents with the online
#      BeliefFilter from its learned z_0 -- exactly as the agent computes them.
#    - Fit and evaluation use DIFFERENT episode sets supplied by the caller (no leakage through
#      shared histories).
#
# 3. Probes Are Objects:
#    - fit_linear_probe / fit_mlp_probe return a BeliefProbe that decodes new latents into
#      beliefs (softmax outputs, rows on the simplex). Decoding is needed by the policy-level
#      analysis (an agent that acts on psi(z)), not only by scoring.
#    - Linear-softmax probe (multinomial logistic regression, full-batch L-BFGS): the claim "the
#      belief is linearly readable" used for interpretability. MLP probe (two layers, Adam): an
#      upper bound on the posterior information present in z.
#
# 4. Metrics:
#    - KL(b* || psi(z)) in nats (overall mean, worst single case, mean per distinct exact
#      posterior) -- penalises confident errors. R^2 is not used: an untrained GRU over one-hot
#      growls reaches R^2 = 0.93 on canonical Tiger.
#    - L1 distance ||b* - psi(z)||_1 (mean, 99th percentile, max): the quantity the Hoelder-type
#      value and regret bounds are linear in (error_bounds.py).
# ==============================================================================

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..domain import FinitePOMDP, belief_update, initial_beliefs
from ..models.world_model import BeliefFilter
from ..training.episode_buffer import EpisodeBatch


@dataclass(frozen=True)
class ProbeDataset:
    """Latents z_0..z_T and exact posteriors b*_0..b*_T of B episodes, flattened to N = B (T + 1) rows."""

    latents: Tensor       # (N, D) float32
    posteriors: Tensor    # (N, S) float64


@dataclass(frozen=True)
class ProbeReport:
    """
    Divergence of decoded beliefs from exact posteriors on held-out histories.

    Attributes:
        mean_kl, max_kl: KL(b* || psi(z)) in nats.
        mean_l1, p99_l1, max_l1: ||b* - psi(z)||_1.
        by_posterior: {rounded exact P(s = 0 | h): (count, mean KL, mean L1)}.
    """

    mean_kl: float
    max_kl: float
    mean_l1: float
    p99_l1: float
    max_l1: float
    by_posterior: dict[float, tuple[int, float, float]]


@torch.no_grad()
def build_probe_dataset(model: FinitePOMDP, belief_filter: BeliefFilter, episodes: EpisodeBatch) -> ProbeDataset:
    """Replays `episodes` through the exact Bayes filter and the learned filter (section 2)."""
    batch, steps = episodes.actions.shape
    device = episodes.actions.device
    exact = initial_beliefs(model, batch, device)
    posteriors = [exact]
    for t in range(steps):
        exact = belief_update(model, exact, episodes.actions[:, t], episodes.observations[:, t])
        posteriors.append(exact)
    latents = belief_filter.unroll(F.one_hot(episodes.actions, model.num_actions).float(),
                                   F.one_hot(episodes.observations, model.num_observations).float())
    return ProbeDataset(latents.reshape(batch * (steps + 1), -1).float(),
                        torch.stack(posteriors, 1).reshape(batch * (steps + 1), -1))


class BeliefProbe(nn.Module):
    """A fitted map psi from latents to beliefs (standardisation + network + softmax)."""

    def __init__(self, network: nn.Module, mean: Tensor, std: Tensor):
        super().__init__()
        self.network = network
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def log_probs(self, latents: Tensor) -> Tensor:
        """log psi(z), shape (N, S)."""
        return F.log_softmax(self.network((latents.float() - self.mean) / self.std), dim=-1)

    @torch.no_grad()
    def forward(self, latents: Tensor) -> Tensor:
        """psi(z) as float64 beliefs on the simplex, shape (N, S)."""
        return self.log_probs(latents).double().exp()


def _standardisation(latents: Tensor) -> tuple[Tensor, Tensor]:
    return latents.mean(0), latents.std(0).clamp(min=1e-6)


def _cross_entropy(probe: BeliefProbe, dataset: ProbeDataset) -> Tensor:
    return -(dataset.posteriors.float() * probe.log_probs(dataset.latents)).sum(-1).mean()


def fit_linear_probe(dataset: ProbeDataset) -> BeliefProbe:
    """Multinomial logistic regression from latents to exact posteriors (full-batch L-BFGS)."""
    mean, std = _standardisation(dataset.latents)
    probe = BeliefProbe(nn.Linear(dataset.latents.shape[1], dataset.posteriors.shape[1]), mean, std)
    probe.to(dataset.latents.device)
    optimizer = torch.optim.LBFGS(probe.parameters(), max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = _cross_entropy(probe, dataset)
        loss.backward()
        return loss

    optimizer.step(closure)
    return probe


def fit_mlp_probe(dataset: ProbeDataset, steps: int = 3000, hidden: int = 64, seed: int = 0) -> BeliefProbe:
    """Two-layer MLP probe (upper bound on the posterior information present in the latent)."""
    torch.manual_seed(seed)
    mean, std = _standardisation(dataset.latents)
    network = nn.Sequential(nn.Linear(dataset.latents.shape[1], hidden), nn.ELU(),
                            nn.Linear(hidden, dataset.posteriors.shape[1]))
    probe = BeliefProbe(network, mean, std).to(dataset.latents.device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=3e-3)
    for _ in range(steps):
        optimizer.zero_grad()
        _cross_entropy(probe, dataset).backward()
        optimizer.step()
    return probe


@torch.no_grad()
def evaluate_probe(probe: BeliefProbe, dataset: ProbeDataset) -> ProbeReport:
    """Scores a probe on (held-out) data (section 4)."""
    exact = dataset.posteriors
    log_model = probe.log_probs(dataset.latents).double()
    kl = (exact * (exact.clamp(min=1e-12).log() - log_model)).sum(-1)
    l1 = (exact - log_model.exp()).abs().sum(-1)
    key = exact[:, 0].round(decimals=4)
    by_posterior = {}
    for value in key.unique().tolist():
        rows = key == value
        by_posterior[value] = (int(rows.sum()), float(kl[rows].mean()), float(l1[rows].mean()))
    return ProbeReport(float(kl.mean()), float(kl.max()), float(l1.mean()), float(torch.quantile(l1, 0.99)),
                       float(l1.max()), by_posterior)
