# ABSOLUTE PATH: src/ipomdp/interpretability/belief_probe.py
# ==============================================================================
# BELIEF PROBES: HOW CLOSE IS THE LEARNED LATENT TO THE EXACT BAYES POSTERIOR?
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. The Question a Probe Answers:
#    - The filter's latent z_t = phi(h_t) is a Bayes-sufficient belief iff there is a map
#      psi with psi(z_t) = b*(h_t) for the exact posterior b* (src/ipomdp/domain/belief.py).
#      A probe fits psi on held-in histories and reports the divergence on held-out ones.
#
# 2. Metric: KL, Not R^2:
#    - The divergence reported is KL(b*(h) || psi(z)) in nats. R^2 of a linear regression on
#      probabilities was used before and is not discriminative: an UNTRAINED GRU over one-hot
#      growls already reaches R^2 = 0.93 on canonical Tiger because it roughly integrates
#      the growl count. KL penalises exactly the confident-but-wrong beliefs that matter for
#      decisions (opening a door at b = 0.97 vs 0.99).
#
# 3. Two Probe Classes:
#    - Linear-softmax probe (multinomial logistic regression, full-batch L-BFGS): the claim
#      "the belief is linearly readable from the latent" used for interpretability.
#    - MLP probe (two layers, Adam): an upper bound on the information present in z.
#    - Each reports the overall mean KL, the worst single held-out KL, and the mean KL per
#      distinct exact posterior value, so errors concentrated on rare, extreme beliefs are
#      visible instead of being averaged away.
#
# 4. Data:
#    - Histories come from a caller-chosen behaviour policy run in the exact simulator;
#      exact posteriors are computed alongside with the batched Bayes filter. Latents are
#      computed by the online BeliefFilter from its learned z_0, exactly as the agent runs it.
# ==============================================================================

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..domain import BatchedPOMDPEnv, FinitePOMDP, belief_update, initial_beliefs
from ..models.world_model import BeliefFilter

# A behaviour policy maps (step index, exact beliefs (B, S)) to int64 actions (B,).
BehaviourPolicy = Callable[[int, Tensor], Tensor]


@dataclass(frozen=True)
class ProbeReport:
    """
    Divergence of a probe's decoded beliefs from the exact posterior on held-out histories.

    Attributes:
        mean_kl: Mean KL(b* || probe) in nats over all held-out latents.
        max_kl: Largest single held-out KL.
        mean_kl_by_posterior: {rounded exact P(s = 0 | h): (count, mean KL)}.
    """

    mean_kl: float
    max_kl: float
    mean_kl_by_posterior: dict[float, tuple[int, float]]


@dataclass(frozen=True)
class ProbeDataset:
    """Latents z_0..z_T and exact posteriors b*_0..b*_T, flattened over episodes and time."""

    latents: Tensor       # (N, D) float32
    posteriors: Tensor    # (N, S) float64


def uniform_random_policy(num_actions: int, seed: int, device: torch.device) -> BehaviourPolicy:
    """Behaviour policy choosing actions uniformly at random (seeded)."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return lambda t, beliefs: torch.randint(0, num_actions, (beliefs.shape[0],), generator=generator, device=device)


@torch.no_grad()
def collect_probe_dataset(
    model: FinitePOMDP,
    belief_filter: BeliefFilter,
    policy: BehaviourPolicy,
    num_episodes: int,
    episode_length: int,
    seed: int,
    device: torch.device,
) -> ProbeDataset:
    """Runs `policy` in the simulator and records (z_t, b*_t) for t = 0..T of every episode."""
    env = BatchedPOMDPEnv(model, num_episodes, episode_length, seed, device)
    exact = initial_beliefs(model, num_episodes, device)
    latent = belief_filter.initial(num_episodes)
    latents, posteriors = [latent.float()], [exact]
    for t in range(episode_length):
        action = policy(t, exact)
        out = env.step(action)
        exact = belief_update(model, exact, action, out.observation)
        latent = belief_filter.step(latent, F.one_hot(action, model.num_actions).float(),
                                    F.one_hot(out.observation, model.num_observations).float())
        latents.append(latent.float())
        posteriors.append(exact)
    return ProbeDataset(torch.cat(latents), torch.cat(posteriors))


def _report(log_probs: Tensor, posteriors: Tensor) -> ProbeReport:
    exact = posteriors.clamp(min=1e-12)
    kl = (posteriors * (exact.log() - log_probs.double())).sum(dim=-1)
    key = posteriors[:, 0].round(decimals=4)
    by_posterior = {}
    for value in key.unique().tolist():
        rows = key == value
        by_posterior[value] = (int(rows.sum()), float(kl[rows].mean()))
    return ProbeReport(float(kl.mean()), float(kl.max()), by_posterior)


def _split(dataset: ProbeDataset) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Even/odd interleaved split so both halves cover every episode length and belief."""
    x, y = dataset.latents, dataset.posteriors
    mean, std = x[0::2].mean(0), x[0::2].std(0).clamp(min=1e-6)
    x = (x - mean) / std
    return x[0::2], y[0::2].float(), x[1::2], y[1::2]


def linear_probe(dataset: ProbeDataset) -> ProbeReport:
    """Multinomial logistic regression from the latent to the exact posterior (L-BFGS)."""
    x_fit, y_fit, x_eval, y_eval = _split(dataset)
    probe = nn.Linear(x_fit.shape[1], y_fit.shape[1]).to(x_fit.device)
    optimizer = torch.optim.LBFGS(probe.parameters(), max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = -(y_fit * F.log_softmax(probe(x_fit), dim=-1)).sum(dim=-1).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return _report(F.log_softmax(probe(x_eval), dim=-1), y_eval)


def mlp_probe(dataset: ProbeDataset, steps: int = 3000, hidden: int = 64, seed: int = 0) -> ProbeReport:
    """Two-layer MLP probe (upper bound on the posterior information present in the latent)."""
    x_fit, y_fit, x_eval, y_eval = _split(dataset)
    torch.manual_seed(seed)
    probe = nn.Sequential(nn.Linear(x_fit.shape[1], hidden), nn.ELU(), nn.Linear(hidden, y_fit.shape[1])).to(x_fit.device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=3e-3)
    for _ in range(steps):
        optimizer.zero_grad()
        (-(y_fit * F.log_softmax(probe(x_fit), dim=-1)).sum(dim=-1).mean()).backward()
        optimizer.step()
    with torch.no_grad():
        return _report(F.log_softmax(probe(x_eval), dim=-1), y_eval)
