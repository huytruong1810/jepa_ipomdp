# ABSOLUTE PATH: src/ipomdp/interfaces.py
# ==============================================================================
# PLANNER INTERFACE BOUNDARY
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Structural Multi-Object Tensor Contract:
#    - Beliefs exchanged with a planner are latent tensors of shape (B, N_obj, D_latent).
#
# 2. Causal Belief Filter Boundary:
#    - encode_context implements the information-state recurrence
#         b_t = Filter(b_{t-1}, a_{t-1}, o_t)
#      and search returns batched action distributions of shape (B, |A|).
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
    Operates over multi-object latent belief state tensors of shape (B, N_obj, D_latent).
    """

    @abstractmethod
    def encode_context(self, obs: Tensor, action: Tensor, prev_belief: Tensor) -> Tensor:
        """
        Advances latent belief state through the recurrent context filter.

        Args:
            obs: Raw observation tensor of shape (B, *obs_shape).
            action: Action tensor taken at step t-1 of shape (B, action_dim_i).
            prev_belief: Prior recurrent belief state tensor of shape (B, N_obj, D_latent).

        Returns:
            Updated recurrent belief state tensor of shape (B, N_obj, D_latent).
        """
        pass

    @abstractmethod
    def search(self, root_state: Tensor, temperature: float = 1.0) -> Tensor:
        """
        Performs open-loop MCTS lookahead search in latent space and returns action distributions.

        Args:
            root_state: Initial root belief state tensor of shape (B, N_obj, D_latent).
            temperature: Action selection temperature (0.0 for argmax, 1.0 for proportional).

        Returns:
            Batched policy action probabilities of shape (B, action_dim_i).
        """
        pass
