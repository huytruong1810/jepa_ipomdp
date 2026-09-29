"""
Exact domain layer: POMDP specifications, the canonical Tiger, the exact Bayes filter,
the exact value-iteration solver, and a batched simulator. Every other layer treats the
FinitePOMDP here as the single source of truth for the environment's dynamics.
"""

from .belief import belief_update, initial_beliefs, observation_distribution, predict_state
from .env import BatchedPOMDPEnv, StepOutput
from .pomdp import FinitePOMDP
from .solver import (
    EXACT_PRUNE_EPSILON,
    AlphaVectorSet,
    InfiniteHorizonSolution,
    action_value_functions,
    backup,
    prune,
    solve_finite_horizon,
    solve_infinite_horizon,
    sup_norm_distance,
)
from .tiger import TigerAction, TigerObservation, TigerState, build_tiger_pomdp

__all__ = [
    "EXACT_PRUNE_EPSILON",
    "AlphaVectorSet",
    "BatchedPOMDPEnv",
    "FinitePOMDP",
    "InfiniteHorizonSolution",
    "StepOutput",
    "TigerAction",
    "TigerObservation",
    "TigerState",
    "action_value_functions",
    "backup",
    "belief_update",
    "build_tiger_pomdp",
    "initial_beliefs",
    "observation_distribution",
    "predict_state",
    "prune",
    "solve_finite_horizon",
    "solve_infinite_horizon",
    "sup_norm_distance",
]
