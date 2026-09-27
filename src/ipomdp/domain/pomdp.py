# ABSOLUTE PATH: src/ipomdp/domain/pomdp.py
# ==============================================================================
# FINITE POMDP SPECIFICATION: THE SINGLE SOURCE OF TRUTH FOR A DOMAIN
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. One Specification, Many Consumers:
#    - A domain is described exactly once, as dense probability/reward tensors.
#      Every component that needs the true model reads these same tensors:
#         * BatchedPOMDPEnv   (src/ipomdp/domain/env.py)    samples from T, O, R, b0.
#         * belief_update     (src/ipomdp/domain/belief.py) computes the exact posterior.
#         * solve_*           (src/ipomdp/domain/solver.py) computes exact optimal values.
#    - Rationale: an earlier iteration hard-coded the Tiger dynamics separately in the
#      simulator, in an analytical "oracle", and in an observation-class table. The copies
#      drifted from the canonical specification (an informative growl after door openings
#      and a free observation at t=0) and, because the oracle shared the simulator's error,
#      no test could detect it. A single specification makes that class of bug impossible.
#
# 2. Tensor Conventions (action-major, float64):
#    - transition[a, s, s']  = T(s' | s, a)
#    - observation[a, s', o] = O(o | s', a)     (observation emitted from the ARRIVAL state)
#    - reward[a, s]          = R(s, a)          (expected immediate reward before transition)
#    - initial_belief[s]     = b0(s)
#    - Action-major layout makes the per-action slices used by the Bayes filter and the
#      value-iteration backup contiguous (T[a], O[a]) and matches the Cassandra .POMDP
#      file format (per-action "T:" and "O:" blocks), easing audits against the reference.
#    - float64 is mandatory: exact solvers and Bayes filters are used as ground truth for
#      learned components, so their numerical error must be negligible relative to any
#      error we want to measure (float32 epsilon ~1e-7 would dominate KL values ~1e-6).
#
# 3. Validation Is Eager and Strict:
#    - Every invariant (shapes, non-negativity, row-stochasticity within 1e-9, discount in
#      (0, 1), unique names) is checked in __post_init__ and raises ValueError. There is no
#      silent renormalisation: a malformed model is a bug in the domain definition.
#
# 4. Extension Path to I-POMDPs:
#    - A finitely nested I-POMDP_{i,l} whose opponent model set M_j is finite (e.g. a
#      fixed level-0 policy, or a finite set of level-(l-1) solved models) is equivalent
#      to a POMDP over the interactive state space IS_i = S x M_j. Folding the opponent's
#      policy into T and O yields a FinitePOMDP, so the same env, filter, and exact solver
#      serve as benchmark infrastructure from single-agent Tiger up through nested levels.
#      The single-agent canonical Tiger is the degenerate case where M_j is empty.
# ==============================================================================

from dataclasses import dataclass

import torch
from torch import Tensor

# Tolerance for checking that probability tables are row-stochastic. Tables are built from
# exact decimal literals (0.85, 0.15, 0.5) so their rows sum to 1 within ~1e-16 in float64;
# 1e-9 leaves ample room for tables assembled by products (I-POMDP folding) while still
# catching any genuine specification error.
_STOCHASTIC_ATOL = 1e-9


@dataclass(frozen=True)
class FinitePOMDP:
    """
    Exact specification of a discrete, discounted, infinite-horizon POMDP.

    Attributes:
        transition: T(s' | s, a), float64 tensor of shape (A, S, S).
        observation: O(o | s', a), float64 tensor of shape (A, S, O).
        reward: R(s, a), float64 tensor of shape (A, S).
        initial_belief: b0(s), float64 tensor of shape (S,).
        discount: Discount factor gamma in (0, 1).
        state_names: Human-readable name per state, length S.
        action_names: Human-readable name per action, length A.
        observation_names: Human-readable name per observation, length O.
    """

    transition: Tensor
    observation: Tensor
    reward: Tensor
    initial_belief: Tensor
    discount: float
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    observation_names: tuple[str, ...]

    def __post_init__(self) -> None:
        num_s, num_a, num_o = len(self.state_names), len(self.action_names), len(self.observation_names)

        for label, names in (("state", self.state_names), ("action", self.action_names),
                             ("observation", self.observation_names)):
            if len(names) == 0:
                raise ValueError(f"FinitePOMDP requires at least one {label}.")
            if len(set(names)) != len(names):
                raise ValueError(f"Duplicate {label} names: {names}")

        expected_shapes = {
            "transition": (num_a, num_s, num_s),
            "observation": (num_a, num_s, num_o),
            "reward": (num_a, num_s),
            "initial_belief": (num_s,),
        }
        for field_name, shape in expected_shapes.items():
            tensor = getattr(self, field_name)
            if not isinstance(tensor, Tensor):
                raise ValueError(f"{field_name} must be a torch.Tensor, got {type(tensor).__name__}.")
            if tensor.dtype != torch.float64:
                raise ValueError(f"{field_name} must be float64, got {tensor.dtype}.")
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{field_name} must have shape {shape}, got {tuple(tensor.shape)}.")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{field_name} contains non-finite entries.")

        for field_name in ("transition", "observation", "initial_belief"):
            tensor = getattr(self, field_name)
            if (tensor < 0).any():
                raise ValueError(f"{field_name} contains negative probabilities.")
            row_sums = tensor.sum(dim=-1)
            if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=_STOCHASTIC_ATOL, rtol=0.0):
                raise ValueError(f"{field_name} rows must sum to 1; got sums {row_sums.tolist()}.")

        if not 0.0 < self.discount < 1.0:
            raise ValueError(f"discount must lie in (0, 1) for an infinite-horizon POMDP, got {self.discount}.")

    @property
    def num_states(self) -> int:
        """Number of hidden states |S|."""
        return len(self.state_names)

    @property
    def num_actions(self) -> int:
        """Number of actions |A|."""
        return len(self.action_names)

    @property
    def num_observations(self) -> int:
        """Number of observations |O|."""
        return len(self.observation_names)

    @property
    def value_bound(self) -> float:
        """
        Bound on |r| and on |V^pi(b)| for every policy and belief: max|R| / (1 - gamma).

        Used to size the two-hot value/reward bins (src/ipomdp/models/distributions.py).
        """
        return float(self.reward.abs().max()) / (1.0 - self.discount)

    @property
    def reward_range(self) -> tuple[float, float]:
        """(min, max) immediate reward over all (s, a); used by value-error bounds."""
        return float(self.reward.min()), float(self.reward.max())
