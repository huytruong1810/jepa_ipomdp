# ABSOLUTE PATH: src/ipomdp/models/__init__.py
"""Neural network architectures, layers, distributions, extractors, and world models for JEPA-IPOMDP."""

from .layers import (
    RMSNorm,
    SwiGLUResidualBlock,
    build_residual_stack,
    LearnedPositionalEncoding2D,
    AttentionPooler,
    SwarmActionEncoder,
)
from .distributions import TwoHotSymlog, symlog, symexp
from .extractors import (
    FeatureExtractor,
    SlotAttention,
    MLPFeatureExtractor,
    CNNFeatureExtractor,
)
from .heads import (
    ValueHead,
    RewardHead,
    DiscretePolicyHead,
    ObservationProbeHead,
)
from .world_model import (
    RecurrentContextEncoder,
    CausalRelationalPredictor,
    RecurrentJEPABase,
)

__all__ = [
    "RMSNorm",
    "SwiGLUResidualBlock",
    "build_residual_stack",
    "LearnedPositionalEncoding2D",
    "AttentionPooler",
    "SwarmActionEncoder",
    "TwoHotSymlog",
    "symlog",
    "symexp",
    "FeatureExtractor",
    "SlotAttention",
    "MLPFeatureExtractor",
    "CNNFeatureExtractor",
    "ValueHead",
    "RewardHead",
    "DiscretePolicyHead",
    "ObservationProbeHead",
    "RecurrentContextEncoder",
    "CausalRelationalPredictor",
    "RecurrentJEPABase",
]
