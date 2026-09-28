# ABSOLUTE PATH: src/ipomdp/planning/search_model.py
# ==============================================================================
# SEARCH MODELS: WHAT THE BELIEF-TREE SEARCH NEEDS FROM A MODEL
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. One Interface, Two Implementations:
#    - The planner (planning/mcts.py) is written against SearchModel only. Two models
#      implement it:
#         ExactSearchModel    states are exact beliefs b in the simplex; rewards, observation
#                             probabilities and successors come from the FinitePOMDP.
#         LearnedSearchModel  states are learned latents z; rewards/observations/values come
#                             from the heads, successors from the BeliefFilter.
#    - The exact model lets the SEARCH be validated against the exact solver with no learning
#      involved (Phase-4 small-scale check); swapping in the learned model then isolates the
#      cost of the learned representation. A search bug can no longer hide behind a learning
#      bug or vice versa.
#
# 2. Expansion Returns Every Child at Once:
#      expand(states (N, ...)) -> rewards (N, A)          E[r | s, a]
#                                 observation_probs (N, A, O)  P(o' | s, a)
#                                 next_states (N, A, O, ...)   tau(s, a, o')
#                                 next_values (N, A, O)        leaf estimate V(tau(s, a, o'))
#    - Branching over observations is exact (|O| is small in discrete POMDPs), so every action
#      of an expanded node has a one-step lookahead Q immediately:
#         Q(s, a) = R(s, a) + gamma * sum_o P(o | s, a) V(tau(s, a, o))
#
# 3. Belief Tracking Uses the Same Model:
#    - initial_states(B) and update(states, a, o) are the model's filter: exact Bayes for the
#      exact model, BeliefFilter for the learned one. The agent therefore plans from exactly the
#      state representation it tracks.
#
# 4. Precision:
#    - The exact model computes in float64 (ground truth). The learned model runs its networks
#      in float32 without autocast: planning evaluates few, small batches, and bfloat16 logits
#      would perturb the two-hot means the search compares.
# ==============================================================================

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor

from ..domain import AlphaVectorSet, FinitePOMDP, belief_update, initial_beliefs, observation_distribution
from ..models.distributions import TwoHotSymlog
from ..models.heads import ObservationHead, RewardHead, ValueHead
from ..models.world_model import BeliefFilter


@dataclass(frozen=True)
class Expansion:
    """All one-step successors of N states (shapes in the module header, section 2)."""

    rewards: Tensor
    observation_probs: Tensor
    next_states: Tensor
    next_values: Tensor


class SearchModel(Protocol):
    """What the belief-tree search and the planning agent need from a model."""

    num_actions: int
    num_observations: int
    discount: float

    def initial_states(self, batch_size: int) -> Tensor:
        """States representing the prior, shape (B, ...)."""
        ...

    def update(self, states: Tensor, actions: Tensor, observations: Tensor) -> Tensor:
        """Filter step s' = tau(s, a, o'), int64 actions/observations of shape (B,)."""
        ...

    def expand(self, states: Tensor) -> Expansion:
        """All children of each state (module header, section 2)."""
        ...


class ExactSearchModel:
    """Exact beliefs over a FinitePOMDP; leaf values from an optional alpha-vector set."""

    def __init__(self, model: FinitePOMDP, leaf_values: AlphaVectorSet | None, device: torch.device):
        """
        Args:
            model: Exact POMDP.
            leaf_values: Leaf evaluation V(b) (e.g. the solver's V*); None means V = 0.
            device: Device of the belief tensors.
        """
        self.model = model
        self.leaf_values = leaf_values
        self.device = device
        self.num_actions = model.num_actions
        self.num_observations = model.num_observations
        self.discount = model.discount
        self._reward = model.reward.to(device)
        self._observation = model.observation.to(device)
        self._transition = model.transition.to(device)

    def initial_states(self, batch_size: int) -> Tensor:
        return initial_beliefs(self.model, batch_size, self.device)

    def update(self, states: Tensor, actions: Tensor, observations: Tensor) -> Tensor:
        return belief_update(self.model, states, actions, observations)

    def _values(self, beliefs: Tensor) -> Tensor:
        if self.leaf_values is None:
            return torch.zeros(beliefs.shape[:-1], dtype=torch.float64, device=self.device)
        return self.leaf_values.value(beliefs.reshape(-1, beliefs.shape[-1])).view(beliefs.shape[:-1])

    def expand(self, states: Tensor) -> Expansion:
        n, num_a, num_o = states.shape[0], self.num_actions, self.num_observations
        rewards = states @ self._reward.T                                              # (N, A)
        predicted = torch.einsum("ns,ast->nat", states, self._transition)              # (N, A, S')
        joint = predicted.unsqueeze(2) * self._observation.transpose(1, 2).unsqueeze(0)  # (N, A, O, S')
        observation_probs = joint.sum(-1)                                              # (N, A, O)
        # Zero-probability observations have weight 0 in every expectation; their successor
        # is set to the predicted state distribution so that every child is a valid belief.
        evidence = observation_probs.unsqueeze(-1)
        next_states = torch.where(evidence > 0, joint / evidence.clamp(min=torch.finfo(joint.dtype).tiny),
                                  predicted.unsqueeze(2).expand(-1, -1, num_o, -1))
        return Expansion(rewards, observation_probs, next_states, self._values(next_states))


class LearnedSearchModel:
    """Learned latents: BeliefFilter successors, two-hot reward/value heads, observation head."""

    def __init__(
        self,
        belief_filter: BeliefFilter,
        reward_head: RewardHead,
        observation_head: ObservationHead,
        value_head: ValueHead,
        codec: TwoHotSymlog,
        num_actions: int,
        num_observations: int,
        discount: float,
    ):
        self.belief_filter = belief_filter
        self.reward_head = reward_head
        self.observation_head = observation_head
        self.value_head = value_head
        self.codec = codec
        self.num_actions = num_actions
        self.num_observations = num_observations
        self.discount = discount

    @torch.no_grad()
    def initial_states(self, batch_size: int) -> Tensor:
        return self.belief_filter.initial(batch_size).clone()

    @torch.no_grad()
    def update(self, states: Tensor, actions: Tensor, observations: Tensor) -> Tensor:
        return self.belief_filter.step(states, F.one_hot(actions, self.num_actions).float(),
                                       F.one_hot(observations, self.num_observations).float())

    @torch.no_grad()
    def expand(self, states: Tensor) -> Expansion:
        n, num_a, num_o = states.shape[0], self.num_actions, self.num_observations
        eye_a = torch.eye(num_a, device=states.device)
        latents = states.repeat_interleave(num_a, dim=0)                                 # (N*A, D)
        actions = eye_a.repeat(n, 1)                                                     # (N*A, A)
        rewards = self.codec.mean(self.reward_head(latents, actions)).view(n, num_a)
        observation_probs = F.softmax(self.observation_head(latents, actions).float(), dim=-1).view(n, num_a, num_o)
        next_states = self.belief_filter.step(
            latents.repeat_interleave(num_o, dim=0),
            actions.repeat_interleave(num_o, dim=0),
            torch.eye(num_o, device=states.device).repeat(n * num_a, 1),
        )                                                                                # (N*A*O, D)
        next_values = self.codec.mean(self.value_head(next_states)).view(n, num_a, num_o)
        return Expansion(rewards, observation_probs, next_states.view(n, num_a, num_o, -1), next_values)
