# ABSOLUTE PATH: src/ipomdp/interfaces.py
# ==============================================================================
# PLANNER INTERFACE BOUNDARY
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Tensor Contract:
#    - Beliefs exchanged with a planner are latent tensors of shape (B, D); search returns
#      batched action distributions of shape (B, |A|).
#
# 2. Separation of Filtering and Planning:
#    - Belief filtering belongs to the world model (models/world_model.py, BeliefFilter);
#      a planner only searches from given latents.
#
# 3. Scope:
#    - Environments are FinitePOMDP simulators (src/ipomdp/domain) and the agent is a
#      single concrete class, so neither needs an abstract base. This interface remains
#      because the agent is written against planners, not a specific search algorithm.
# ==============================================================================

from abc import ABC, abstractmethod

from torch import Tensor


class AbstractPlanner(ABC):
    """
    Abstract interface for planning algorithms operating over latent belief representations.
    Operates over belief latents of shape (B, D).
    """

    @abstractmethod
    def search(self, root_state: Tensor, temperature: float = 1.0) -> Tensor:
        """
        Performs open-loop MCTS lookahead search in latent space and returns action distributions.

        Args:
            root_state: Root belief latents of shape (B, D).
            temperature: Action selection temperature (0.0 for argmax, 1.0 for proportional).

        Returns:
            Batched policy action probabilities of shape (B, action_dim_i).
        """
        pass
