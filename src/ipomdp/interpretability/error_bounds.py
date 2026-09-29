# ABSOLUTE PATH: src/ipomdp/interpretability/error_bounds.py
# ==============================================================================
# FROM BELIEF-DECODING ERROR TO GUARANTEES ON VALUES AND DECISIONS
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. The Lemma (span-Hoelder inequality):
#    - For any vector alpha in R^S and beliefs b, b' in the simplex, sum_s (b - b')_s = 0, hence
#         |alpha . (b - b')| <= span(alpha) / 2 * ||b - b'||_1,   span(alpha) = max_s alpha_s - min_s alpha_s.
#      (Shift alpha by its mid-range c: alpha . (b - b') = (alpha - c) . (b - b') and
#       ||alpha - c||_inf = span(alpha) / 2.)
#    - A maximum of such linear functions (a PWLC value function V(b) = max_k alpha_k . b) is
#      Lipschitz in ||.||_1 with constant L = max_k span(alpha_k) / 2.
#
# 2. The Guarantees (psi = belief probe of belief_probe.py, eps = ||b* - psi(z)||_1):
#    (a) Value error:           |V*(b*) - V*(psi(z))| <= L_V eps
#                                with L_V from the alpha vectors of V*.
#    (b) One-step decision regret of acting greedily on the decoded belief,
#                                a^ = argmax_a Q*(psi(z), a):
#          Q*(b*, a*) - Q*(b*, a^) <= 2 L_Q eps,
#                                with L_Q = max_a Lipschitz(Q*(., a)) from the per-action vector
#                                sets (domain/solver.py action_value_functions). Proof:
#          Q*(b*, a*) - Q*(b*, a^) <= [Q*(psi, a*) + L_Q eps] - [Q*(psi, a^) - L_Q eps] <= 2 L_Q eps.
#    (c) Discounted loss of the policy "act greedily on psi(z_t)" forever, if eps_t <= eps_max:
#          V*(b0) - V^psi(b0) <= 2 L_Q eps_max / (1 - gamma)
#        (performance-difference lemma: the loss is the discounted sum of per-step regrets).
#    - The solver's certified error delta = ||V* - V_n||_inf adds 2 delta (a, b) and
#      2 delta / (1 - gamma) (c) to these bounds when V_n is used in place of V*.
#    - Expected-error form: (a) and (b) hold pointwise, so taking expectations gives
#         E[|V*(b*) - V*(psi)|] <= L_V E[eps],   E[regret] <= 2 L_Q E[eps],
#      and (c) with the per-step expectation under the policy's state distribution,
#         V*(b0) - V^psi(b0) <= 2 L_Q sum_t gamma^t E[eps_t] <= 2 L_Q eps-bar / (1 - gamma).
#      Phase-6 measurement (canonical Tiger, trained agent): the worst-case eps is set by a few
#      rare latents (max L1 0.3-0.9) and makes the worst-case bounds 50-100x loose, while the
#      expected-error bounds are within a small factor of the measurements. Both are reported;
#      eps-bar in (c) is measured on the evaluation data, not on the psi-policy's own
#      distribution, so that form is an estimate rather than a certificate.
#
# 3. Bounds Next to Measurements:
#    - Every bound is reported together with its empirical counterpart on the same held-out
#      beliefs (value errors and one-step regrets computed exactly with V*, Q*), and the policy
#      bound together with the measured return of the decoded-belief policy in the simulator
#      (analysis.py). A bound that is far above its measurement is honest but loose; a
#      measurement above its bound would indicate a bug.
# ==============================================================================

from dataclasses import dataclass

import torch
from torch import Tensor

from ..domain import AlphaVectorSet


def lipschitz_constant(value_function: AlphaVectorSet) -> float:
    """L = max_k span(alpha_k) / 2: the ||.||_1-Lipschitz constant of V(b) = max_k alpha_k . b (section 1)."""
    vectors = value_function.vectors
    return float((vectors.max(dim=1).values - vectors.min(dim=1).values).max()) / 2.0


def q_values(action_values: list[AlphaVectorSet], beliefs: Tensor) -> Tensor:
    """Q(b, a) for beliefs (N, S) from per-action vector sets; shape (N, |A|), float64."""
    return torch.stack([q.value(beliefs) for q in action_values], dim=-1)


@dataclass(frozen=True)
class ErrorBoundReport:
    """
    Guarantees and measurements for one probe on one set of held-out beliefs (module header).

    Attributes:
        lipschitz_value, lipschitz_q: L_V and L_Q.
        belief_l1_mean, belief_l1_max: Measured eps.
        value_error_bound_max, value_error_max, value_error_mean: bound (a) at eps_max vs measured.
        value_error_bound_mean: Expected-error form of (a), L_V E[eps].
        regret_bound_max, regret_max, regret_mean: bound (b) at eps_max vs measured one-step regret.
        regret_bound_mean: Expected-error form of (b), 2 L_Q E[eps].
        suboptimal_decision_rate: Fraction of beliefs where the decoded-belief action is not optimal.
        policy_loss_bound: Bound (c) with eps_max.
        policy_loss_bound_mean: Expected-error form of (c) with the measured E[eps] (an estimate).
    """

    lipschitz_value: float
    lipschitz_q: float
    belief_l1_mean: float
    belief_l1_max: float
    value_error_bound_max: float
    value_error_max: float
    value_error_mean: float
    value_error_bound_mean: float
    regret_bound_max: float
    regret_max: float
    regret_mean: float
    regret_bound_mean: float
    suboptimal_decision_rate: float
    policy_loss_bound: float
    policy_loss_bound_mean: float


def error_bounds(optimal_value: AlphaVectorSet, optimal_action_values: list[AlphaVectorSet], discount: float,
                 true_beliefs: Tensor, decoded_beliefs: Tensor) -> ErrorBoundReport:
    """
    Evaluates guarantees (a)-(c) and their empirical counterparts.

    Args:
        optimal_value: Alpha vectors of V* (or a certified approximation).
        optimal_action_values: Per-action alpha vectors of Q*.
        discount: gamma.
        true_beliefs: Exact posteriors b*, shape (N, S), float64.
        decoded_beliefs: Probe outputs psi(z), shape (N, S), float64.
    """
    true_beliefs, decoded_beliefs = true_beliefs.double().cpu(), decoded_beliefs.double().cpu()
    l_v, l_q = lipschitz_constant(optimal_value), max(lipschitz_constant(q) for q in optimal_action_values)
    eps = (true_beliefs - decoded_beliefs).abs().sum(-1)
    value_error = (optimal_value.value(true_beliefs) - optimal_value.value(decoded_beliefs)).abs()
    q_true = q_values(optimal_action_values, true_beliefs)
    chosen = q_values(optimal_action_values, decoded_beliefs).argmax(-1)
    regret = q_true.max(-1).values - q_true.gather(1, chosen.unsqueeze(1)).squeeze(1)
    eps_max, eps_mean = float(eps.max()), float(eps.mean())
    return ErrorBoundReport(
        lipschitz_value=l_v, lipschitz_q=l_q,
        belief_l1_mean=eps_mean, belief_l1_max=eps_max,
        value_error_bound_max=l_v * eps_max, value_error_max=float(value_error.max()),
        value_error_mean=float(value_error.mean()), value_error_bound_mean=l_v * eps_mean,
        regret_bound_max=2 * l_q * eps_max, regret_max=float(regret.max()), regret_mean=float(regret.mean()),
        regret_bound_mean=2 * l_q * eps_mean,
        suboptimal_decision_rate=float((regret > 1e-9).double().mean()),
        policy_loss_bound=2 * l_q * eps_max / (1.0 - discount),
        policy_loss_bound_mean=2 * l_q * eps_mean / (1.0 - discount),
    )
