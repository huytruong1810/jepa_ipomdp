"""
Domain layer: POMDP specifications, the canonical Tiger, the exact Bayes filter, the exact
value-iteration solver and a batched simulator; and Wumpus World (generative simulator and exact
enumeration filter, no solver). Every other layer treats the specifications here (FinitePOMDP,
WumpusWorld.step_rules) as the single source of truth for each environment's dynamics.
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
from .wumpus import (WumpusAction, WumpusBeliefTracker, WumpusMarginals, WumpusSimulator, WumpusState, WumpusWorld,
                     game_outcomes)

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
    "WumpusAction",
    "WumpusBeliefTracker",
    "WumpusMarginals",
    "WumpusSimulator",
    "WumpusState",
    "WumpusWorld",
    "action_value_functions",
    "backup",
    "belief_update",
    "build_tiger_pomdp",
    "game_outcomes",
    "initial_beliefs",
    "observation_distribution",
    "predict_state",
    "prune",
    "solve_finite_horizon",
    "solve_infinite_horizon",
    "sup_norm_distance",
]
