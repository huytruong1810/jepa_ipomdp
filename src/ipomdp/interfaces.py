# ABSOLUTE PATH: src/ipomdp/interfaces.py
# ==============================================================================
# ABSTRACT INTERFACE BOUNDARIES & TEMPORAL CAUSALITY CONTRACTS
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Structural Multi-Object Tensor Contracts:
#    - Mandates structured multi-object belief tensor dimensions (B x N_obj x D_latent)
#      across all belief filtering and planning interfaces.
#
# 2. Strict Causal Belief Filter Boundary:
#    - IPOMDPAgent.update_belief mandates non-optional current observation o_t and
#      past action a_{t-1}, enforcing the Bayesian information state recurrence:
#         b_t = Filter(b_{t-1}, a_{t-1}, o_t)
#
# 3. Batched Policy Contract:
#    - AbstractPlanner.search returns policy probability tensors of shape (B, action_dim_i),
#      supporting GPU-parallelized MCTS simulation across synchronized batch channels.
# ==============================================================================

import torch
from torch import Tensor
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
from .types import AgentID, Observation, Action, State, StepResult



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


class IPOMDPAgent(ABC):
    """
    Abstract interface for an agent operating within a multi-agent POMDP / I-POMDP environment.
    Enforces temporal causality and belief encapsulation across parallel environment channels.
    """

    @property
    @abstractmethod
    def agent_id(self) -> AgentID:
        """Unique string identifier for the agent (e.g., 'agent_0')."""
        pass

    @abstractmethod
    def act(self, obs: Observation) -> Action:
        """
        Selects actions for current observations by querying internal belief and planner.

        Args:
            obs: Current environment observation container (batched across B parallel envs).

        Returns:
            Action container wrapping chosen action tensor of shape (B, 1) or (B, action_dim).
        """
        pass

    @abstractmethod
    def update_belief(self, obs: Observation, prev_action: Action) -> None:
        """
        Updates internal recurrent belief state using current observation and past action.
        Mathematically enforces: b_t = Filter(b_{t-1}, a_{t-1}, o_t)

        Args:
            obs: Observation received at current step t of shape (B, *obs_shape).
            prev_action: Action taken at step t-1 of shape (B, action_dim) or (B, 1).
                         At t=0, a zero-initialized dummy action token must be supplied.
        """
        pass

    @abstractmethod
    def reset(self, batch_size: int = 1) -> None:
        """
        Clears recurrent hidden belief states across all parallel batch channels.

        Args:
            batch_size: Number of parallel environment instances (B).
        """
        pass

    @abstractmethod
    def reset_index(self, env_idx: int) -> None:
        """
        Resets recurrent hidden states for a single parallel environment index without affecting others.

        Args:
            env_idx: Parallel batch channel index to reset (0 <= env_idx < B).
        """
        pass


class IPOMDPEnv(ABC):
    """
    Abstract interface for a multi-agent partially observable environment.
    All inputs and outputs conform to batched tensor contracts across B parallel channels.
    """

    @abstractmethod
    def reset(self) -> Tuple[Dict[AgentID, Observation], Dict[AgentID, dict]]:
        """
        Initializes environment dynamics and returns initial observations.

        Returns:
            Tuple of (observation_dict, info_dict) mapped by AgentID.
        """
        pass

    @abstractmethod
    def step(self, actions: Dict[AgentID, Action]) -> StepResult:
        """
        Advances environment simulation by one discrete timestep using joint agent actions.

        Args:
            actions: Dictionary mapping AgentID to Action containers.

        Returns:
            StepResult containing observations, rewards, terminations, truncations, and infos.
        """
        pass

    @abstractmethod
    def render(self, actions: Dict[AgentID, Action], step_results: StepResult) -> str:
        """Renders current environment state and joint action diagnostics to text/string format."""
        pass

    @abstractmethod
    def _transition_dynamics(self, state: State, actions: Dict[AgentID, Action]) -> State:
        """The mathematical state transition distribution: T(s, a, s')"""
        pass

    @abstractmethod
    def _get_observation(self, state: State, actions: Dict[AgentID, Action]) -> Dict[AgentID, Observation]:
        """The mathematical observation emission function: O(s', a, o)"""
        pass

    @abstractmethod
    def _get_reward(
        self,
        state: State,
        actions: Dict[AgentID, Action],
        next_state: Optional[State] = None
    ) -> Dict[AgentID, float]:
        """The mathematical reward function: R(s, a) or R(s, a, s')"""
        pass

