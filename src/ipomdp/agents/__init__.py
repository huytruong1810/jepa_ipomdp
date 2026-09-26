"""Batched agent that filters latent beliefs with the JEPA encoder and acts by latent MCTS."""

from .jepa_agent import DiscreteJEPAAgent

__all__ = ["DiscreteJEPAAgent"]
