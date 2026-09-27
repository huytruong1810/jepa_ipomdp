"""JEPA world model (belief filter, EMA target, stochastic transition), prediction heads and building blocks."""

from .distributions import TwoHotSymlog, symexp, symlog
from .heads import ObservationProbeHead, OpponentPolicyHead, RewardHead, ValueHead
from .layers import RMSNorm, SwiGLUResidualBlock, build_residual_stack
from .world_model import BeliefFilter, LatentTransition, RecurrentJEPA

__all__ = [
    "BeliefFilter",
    "LatentTransition",
    "ObservationProbeHead",
    "OpponentPolicyHead",
    "RMSNorm",
    "RecurrentJEPA",
    "RewardHead",
    "SwiGLUResidualBlock",
    "TwoHotSymlog",
    "ValueHead",
    "build_residual_stack",
    "symexp",
    "symlog",
]
