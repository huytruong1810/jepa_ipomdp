# ABSOLUTE PATH: src/ipomdp/domain/solver.py
# ==============================================================================
# EXACT POMDP VALUE ITERATION (ALPHA VECTORS + INCREMENTAL PRUNING)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Role in the Project:
#    - This is the exact benchmark. The learned JEPA + MCTS agent must reproduce its
#      values and actions on the canonical Tiger before any larger experiment is run.
#      Through the S x M_j reduction documented in pomdp.py it also serves as the exact
#      benchmark for finitely nested I-POMDPs with finite opponent model sets.
#
# 2. Representation:
#    - The optimal h-step value function is piecewise-linear and convex (Sondik, 1971):
#         V_h(b) = max_{alpha in Gamma_h} alpha . b
#      Each alpha vector is tagged with the first action of the conditional plan it
#      represents, so argmax over Gamma_h is also the optimal first action.
#    - Gamma_1 = {R(., a) : a in A}. There is no Gamma_0 object: V_0 = 0 needs no policy,
#      and starting at Gamma_1 avoids a sentinel "no action" tag.
#
# 3. Backup via Incremental Pruning (Cassandra, Littman & Zhang, 1997):
#      Gamma_{a,o}   = { R(., a)/|O| + gamma * M_{a,o} alpha : alpha in Gamma_h },
#                      M_{a,o}[s, s'] = T(s'|s,a) O(o|s',a)
#      Gamma_a       = prune( ... prune( prune(Gamma_{a,o1} (+) Gamma_{a,o2}) (+) Gamma_{a,o3}) ... )
#      Gamma_{h+1}   = prune( union_a Gamma_a )
#    - (+) is the cross-sum. Pruning after every cross-sum keeps intermediate sets small.
#
# 4. epsilon-Pruning With a Proven Error (Lark's filter, as in White 1991 / Cassandra 1998):
#    - prune(Gamma, eps) returns W subset of Gamma with  V_Gamma - eps <= V_W <= V_Gamma
#      everywhere on the simplex. Steps:
#         a) remove exact duplicates and pointwise-dominated vectors (error 0);
#         b) seed W with the argmax vector at a fixed sample of beliefs (each is a true
#            maximiser of V_Gamma at that belief, so it belongs to the envelope);
#         c) for each remaining candidate alpha solve the witness LP against W only:
#               max delta  s.t.  (alpha - w) . b >= delta  for all w in W,  b in simplex.
#            If delta <= eps, discard alpha: V_W >= alpha - eps now, and W only grows, so
#            the inequality still holds at the end. Otherwise the LP returns a witness
#            belief b; the best remaining candidate at b is moved into W and alpha is
#            re-examined later.
#      Testing against W (not against "all other survivors") matters: sequential removal
#      against survivors lets errors chain (a vector removed for being within eps of a
#      vector that is itself later removed), which breaks the eps guarantee.
#    - An exact backup therefore uses eps = EXACT_PRUNE_EPSILON (float64 round-off scale).
#      A larger eps trades a PROVEN value error for speed: one backup performs |O|
#      projection prunes, |O| - 1 cross-sum prunes and one union prune per action set, so
#         || B~V - BV ||_inf <= 2 |O| eps     (B = exact Bellman operator).
#
# 5. Certified Infinite-Horizon Error Bound:
#    - With V~_n = B~ V~_{n-1} and e = 2|O| eps, the triangle inequality and gamma-contraction
#      of B give
#         || V* - V~_n ||_inf <= ( gamma * || V~_n - V~_{n-1} ||_inf + e ) / (1 - gamma).
#      The sup-norm distance between two PWLC functions is computed EXACTLY by LPs
#      (max over the simplex of each one minus the other), so the reported bound is a
#      proof, not an estimate. Iteration stops once the bound is below `tolerance`.
#    - On the canonical Tiger, eps = 1e-6 certifies V* to 1e-4 in well under a minute,
#      whereas eps = 1e-9 keeps many vectors with negligible advantage mid-horizon.
# ==============================================================================

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linprog
from torch import Tensor

from .pomdp import FinitePOMDP

# Pruning epsilon for "exact" solves. Tiger values are O(10^2); 1e-9 is ~1e-11 relative,
# above float64 round-off in the backups and far below anything that affects decisions.
# Accumulated over any horizon it contributes at most 2|O| * 1e-9 / (1 - gamma) = 8e-8.
EXACT_PRUNE_EPSILON = 1e-9

# Size and seed of the belief sample that seeds W in the pruning pre-pass. Correctness never
# depends on them; they only reduce the number of LPs. Fixed seed => deterministic solves.
_NUM_SEED_BELIEFS = 256
_SEED_BELIEF_RNG_SEED = 0


@dataclass(frozen=True)
class AlphaVectorSet:
    """
    Piecewise-linear convex value function V(b) = max_k vectors[k] . b.

    Attributes:
        vectors: float64 tensor of shape (K, S).
        actions: int64 tensor of shape (K,), first action of each vector's conditional plan.
    """

    vectors: Tensor
    actions: Tensor

    def value(self, belief: Tensor) -> Tensor:
        """V(b) for beliefs of shape (B, S); returns shape (B,)."""
        return (belief.to(torch.float64) @ self.vectors.to(belief.device).T).max(dim=-1).values

    def greedy_action(self, belief: Tensor) -> Tensor:
        """Optimal first action for beliefs of shape (B, S); returns int64 shape (B,)."""
        best = (belief.to(torch.float64) @ self.vectors.to(belief.device).T).argmax(dim=-1)
        return self.actions.to(belief.device)[best]


@dataclass(frozen=True)
class InfiniteHorizonSolution:
    """
    Result of value iteration run to a certified tolerance.

    Attributes:
        value_function: Gamma_n, the final alpha-vector set.
        iterations: n, the number of backups performed (horizon of V~_n).
        error_bound: Proven upper bound on || V* - V~_n ||_inf.
    """

    value_function: AlphaVectorSet
    iterations: int
    error_bound: float


def _seed_beliefs(num_states: int) -> np.ndarray:
    """Simplex vertices plus uniformly distributed interior beliefs from a fixed seed."""
    rng = np.random.default_rng(_SEED_BELIEF_RNG_SEED)
    return np.vstack([np.eye(num_states), rng.dirichlet(np.ones(num_states), size=_NUM_SEED_BELIEFS)])


def _simplex_lp(cost: np.ndarray, a_ub: np.ndarray) -> np.ndarray:
    """
    Solves min cost . x  s.t.  a_ub x <= 0,  x = [b, t],  b in the simplex,  t free.

    Both LPs used by the solver (witness margin and PWLC sup-norm) have this form.
    Returns the optimal x.
    """
    num_states = cost.shape[0] - 1
    a_eq = np.concatenate([np.ones(num_states), [0.0]])[None, :]
    bounds = [(0.0, 1.0)] * num_states + [(None, None)]
    result = linprog(cost, A_ub=a_ub, b_ub=np.zeros(a_ub.shape[0]), A_eq=a_eq, b_eq=[1.0],
                     bounds=bounds, method="highs")
    if result.status != 0:
        raise RuntimeError(f"Simplex LP failed: {result.message}")
    return result.x


def _witness(candidate: np.ndarray, envelope: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Largest margin by which `candidate` beats every vector of `envelope` at a single belief.

    Returns:
        (delta, b) with delta = max_b min_w (candidate - w) . b and b its maximiser.
    """
    # x = [b, delta]; minimise -delta s.t. (w - candidate) . b + delta <= 0 for all w.
    cost = np.zeros(candidate.shape[0] + 1)
    cost[-1] = -1.0
    x = _simplex_lp(cost, np.hstack([envelope - candidate, np.ones((envelope.shape[0], 1))]))
    return float(x[-1]), x[:-1]


def prune(vectors: Tensor, actions: Tensor, epsilon: float) -> tuple[Tensor, Tensor]:
    """
    epsilon-prunes an alpha-vector set (Lark's filter; see module header, section 4).

    Args:
        vectors: float64 tensor of shape (K, S).
        actions: int64 tensor of shape (K,).
        epsilon: Maximum allowed loss of value anywhere on the simplex (>= 0).

    Returns:
        (vectors, actions) of the kept subset W, in original order, with
        V_input - epsilon <= V_W <= V_input.
    """
    vecs = vectors.cpu().numpy()
    acts = actions.cpu().numpy()

    # a) Exact duplicates keep their first occurrence.
    _, first_idx = np.unique(vecs, axis=0, return_index=True)
    order = np.sort(first_idx)
    vecs, acts = vecs[order], acts[order]

    # a) Pointwise domination with index tie-break, so no two vectors remove each other:
    #    j removes i  iff  v_j >= v_i everywhere and (v_j > v_i somewhere or j < i).
    ge = (vecs[None, :, :] >= vecs[:, None, :]).all(axis=-1)   # ge[i, j]: v_j >= v_i
    gt = (vecs[None, :, :] > vecs[:, None, :]).any(axis=-1)    # gt[i, j]: v_j > v_i somewhere
    earlier = np.arange(len(vecs))[None, :] < np.arange(len(vecs))[:, None]
    dominates = ge & (gt | earlier)
    np.fill_diagonal(dominates, False)
    keep = ~dominates.any(axis=1)
    vecs, acts = vecs[keep], acts[keep]

    # b) Seed W with the maximiser at each sample belief.
    in_envelope = np.zeros(len(vecs), dtype=bool)
    in_envelope[np.unique((_seed_beliefs(vecs.shape[1]) @ vecs.T).argmax(axis=1))] = True

    # c) Lark's filter over the remaining candidates.
    candidates = list(np.flatnonzero(~in_envelope))
    while candidates:
        idx = candidates.pop()
        delta, belief = _witness(vecs[idx], vecs[in_envelope])
        if delta <= epsilon:
            continue
        pool = candidates + [idx]
        best = pool[int(np.argmax(vecs[pool] @ belief))]
        in_envelope[best] = True
        if best != idx:
            candidates.remove(best)
            candidates.append(idx)

    return torch.from_numpy(vecs[in_envelope].copy()), torch.from_numpy(acts[in_envelope].copy())


def _initial_value_function(model: FinitePOMDP, epsilon: float) -> AlphaVectorSet:
    """Gamma_1 = { R(., a) : a in A }, pruned."""
    return AlphaVectorSet(*prune(model.reward.clone(), torch.arange(model.num_actions), epsilon))


def backup(model: FinitePOMDP, value_function: AlphaVectorSet, epsilon: float) -> AlphaVectorSet:
    """
    One dynamic-programming backup Gamma_h -> Gamma_{h+1} by incremental pruning.

    Args:
        model: Exact POMDP specification (CPU tensors).
        value_function: Gamma_h.
        epsilon: Pruning epsilon; the result is within 2|O| epsilon of the exact backup.

    Returns:
        Gamma_{h+1}.
    """
    gamma = model.discount
    num_o = model.num_observations
    per_action_vectors, per_action_actions = [], []

    for a in range(model.num_actions):
        tag = lambda n: torch.full((n,), a)  # noqa: E731 - every vector here starts with action a
        cross_sum = None
        for o in range(num_o):
            # M_{a,o}[s, s'] = T(s'|s,a) O(o|s',a); projected vectors are R/|O| + gamma * M alpha.
            projection = model.transition[a] * model.observation[a, :, o].unsqueeze(0)
            projected = model.reward[a].unsqueeze(0) / num_o + gamma * value_function.vectors @ projection.T
            projected, _ = prune(projected, tag(projected.shape[0]), epsilon)
            if cross_sum is None:
                cross_sum = projected
            else:
                combined = (cross_sum.unsqueeze(1) + projected.unsqueeze(0)).reshape(-1, model.num_states)
                cross_sum, _ = prune(combined, tag(combined.shape[0]), epsilon)
        per_action_vectors.append(cross_sum)
        per_action_actions.append(tag(cross_sum.shape[0]))

    return AlphaVectorSet(*prune(torch.cat(per_action_vectors), torch.cat(per_action_actions), epsilon))


def solve_finite_horizon(model: FinitePOMDP, horizon: int) -> list[AlphaVectorSet]:
    """
    Exact optimal value functions for horizons 1..horizon (pruning at EXACT_PRUNE_EPSILON).

    Args:
        model: Exact POMDP specification.
        horizon: Largest horizon to solve (>= 1).

    Returns:
        [Gamma_1, ..., Gamma_horizon]. With t steps remaining, act greedily w.r.t. Gamma_t.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}.")
    value_functions = [_initial_value_function(model, EXACT_PRUNE_EPSILON)]
    while len(value_functions) < horizon:
        value_functions.append(backup(model, value_functions[-1], EXACT_PRUNE_EPSILON))
    return value_functions


def _max_difference(upper: AlphaVectorSet, lower: AlphaVectorSet) -> float:
    """Exact max over the simplex of V_upper(b) - V_lower(b), via one LP per upper vector."""
    lower_vecs = lower.vectors.numpy()
    a_ub = np.hstack([lower_vecs, -np.ones((lower_vecs.shape[0], 1))])  # beta . b - t <= 0
    best = -np.inf
    for alpha in upper.vectors.numpy():
        # x = [b, t]; maximise alpha . b - t, i.e. minimise -alpha . b + t.
        x = _simplex_lp(np.concatenate([-alpha, [1.0]]), a_ub)
        best = max(best, float(alpha @ x[:-1] - x[-1]))
    return best


def sup_norm_distance(first: AlphaVectorSet, second: AlphaVectorSet) -> float:
    """Exact || V_first - V_second ||_inf over the belief simplex."""
    return max(_max_difference(first, second), _max_difference(second, first))


def solve_infinite_horizon(
    model: FinitePOMDP,
    tolerance: float,
    prune_epsilon: float,
    max_iterations: int = 10_000,
) -> InfiniteHorizonSolution:
    """
    Value iteration until the certified bound || V* - V~_n ||_inf <= tolerance.

    Args:
        model: Exact POMDP specification.
        tolerance: Required bound on the sup-norm error to V*.
        prune_epsilon: Pruning epsilon. Its contribution 2|O| eps / (1 - gamma) to the bound
            must be below `tolerance`, otherwise the tolerance is unreachable.
        max_iterations: Hard cap; exceeding it raises instead of returning an uncertified result.

    Returns:
        InfiniteHorizonSolution with the final Gamma_n and its proven error bound.
    """
    gamma = model.discount
    backup_error = 2 * model.num_observations * prune_epsilon
    if backup_error / (1.0 - gamma) >= tolerance:
        raise ValueError(
            f"prune_epsilon={prune_epsilon} alone contributes {backup_error / (1.0 - gamma):.3g} "
            f"to the bound, which cannot meet tolerance={tolerance}."
        )
    previous = _initial_value_function(model, prune_epsilon)
    for iteration in range(2, max_iterations + 1):
        current = backup(model, previous, prune_epsilon)
        bound = (gamma * sup_norm_distance(current, previous) + backup_error) / (1.0 - gamma)
        if bound <= tolerance:
            return InfiniteHorizonSolution(current, iteration, bound)
        previous = current
    raise RuntimeError(f"Value iteration did not reach tolerance {tolerance} within {max_iterations} iterations.")
