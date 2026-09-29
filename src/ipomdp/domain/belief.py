# ABSOLUTE PATH: src/ipomdp/domain/belief.py
# ==============================================================================
# EXACT BATCHED BAYES FILTER OVER A FinitePOMDP
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. The Belief Update tau(b, a, o):
#      b'(s') = O(o | s', a) * sum_s T(s' | s, a) b(s)  /  P(o | b, a)
#      P(o | b, a) = sum_s' O(o | s', a) * sum_s T(s' | s, a) b(s)
#    - This is the ground truth the learned JEPA belief filter is measured against:
#      probes decode the latent into b(s) (interpretability/belief_probe.py), and the
#      learned observation head is scored against P(o | b, a) by KL
#      (tests/test_world_model_acceptance.py).
#
# 2. Batched, Stateless Functions:
#    - Plain functions over (B, S) tensors, not a stateful "oracle" object. Callers keep
#      their own belief tensor, which removes a class of bugs where an oracle's internal
#      state silently diverged from the trajectory being evaluated (and lets the same code
#      run on thousands of trajectories at once).
#
# 3. Impossible Observations Raise:
#    - If P(o | b, a) = 0 the observation contradicts the model; renormalising would hide a
#      simulator or bookkeeping bug. There is no epsilon clamp.
# ==============================================================================

import torch
from torch import Tensor

from .pomdp import FinitePOMDP


def predict_state(model: FinitePOMDP, belief: Tensor, action: Tensor) -> Tensor:
    """
    Pushes a belief through the transition model: b^a(s') = sum_s T(s' | s, a) b(s).

    Args:
        model: Exact POMDP specification.
        belief: Beliefs of shape (B, S), float64, rows summing to 1.
        action: Action indices of shape (B,), int64.

    Returns:
        Predicted state distributions of shape (B, S).
    """
    transition = model.transition.to(belief.device)[action]  # (B, S, S')
    return torch.einsum("bs,bst->bt", belief, transition)


def observation_distribution(model: FinitePOMDP, belief: Tensor, action: Tensor) -> Tensor:
    """
    Exact predictive observation distribution P(o | b, a).

    Args:
        model: Exact POMDP specification.
        belief: Beliefs of shape (B, S), float64.
        action: Action indices of shape (B,), int64.

    Returns:
        Probabilities of shape (B, O), rows summing to 1.
    """
    emission = model.observation.to(belief.device)[action]  # (B, S', O)
    return torch.einsum("bt,bto->bo", predict_state(model, belief, action), emission)


def belief_update(model: FinitePOMDP, belief: Tensor, action: Tensor, observation: Tensor) -> Tensor:
    """
    Exact Bayes filter step b' = tau(b, a, o).

    Args:
        model: Exact POMDP specification.
        belief: Beliefs of shape (B, S), float64.
        action: Action indices a_t of shape (B,), int64.
        observation: Observation indices o_{t+1} of shape (B,), int64.

    Returns:
        Posterior beliefs of shape (B, S).

    Raises:
        ValueError: If any observation has zero probability under the model.
    """
    emission = model.observation.to(belief.device)[action, :, observation]  # (B, S')
    unnormalised = predict_state(model, belief, action) * emission
    evidence = unnormalised.sum(dim=-1, keepdim=True)
    if (evidence <= 0).any():
        raise ValueError("Observation has zero probability under the model; the trajectory is inconsistent.")
    return unnormalised / evidence


def initial_beliefs(model: FinitePOMDP, batch_size: int, device: torch.device) -> Tensor:
    """Returns b0 replicated to shape (batch_size, S), float64."""
    return model.initial_belief.to(device).expand(batch_size, -1).clone()
