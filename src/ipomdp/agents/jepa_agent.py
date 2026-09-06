# ABSOLUTE PATH: src/ipomdp/agents/jepa_agent.py
# ==============================================================================
# STATEFUL & STATELESS MULTI-AGENT POMDP INTERFACE ENCAPSULATIONS
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Recurrent Information State Encapsulation:
#    - DiscreteJEPAAgent maintains internal recurrent belief tensors b_t in
#      R^(B x N_obj x D_latent) and past action tokens a_{t-1}.
#    - Supports single-channel isolated resets (reset_index(env_idx)) to maintain
#      belief integrity across asynchronous vectorized environment terminations.
#
# 2. Dynamic Rank-Safe Action Conversion:
#    - Automatically detects and converts 1D/2D integer action indices into one-hot
#      tensors inside update_belief via Action.to_one_hot, preserving leading
#      batch dimensions (B >= 1) even when B = 1.
#
# 3. Scalar Telemetry Aggregation:
#    - Extracts scalar float mean values across parallel batch channels
#      (np.mean([r.value for r in self.planner.roots])) for TensorBoard logging.
# ==============================================================================

from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F

from ..interfaces import IPOMDPAgent, AbstractPlanner
from ..types import AgentID, Observation, Action


class StatelessAgent(IPOMDPAgent):
    """
    Stateless Agent executing fixed or uniform random policy distributions.
    Maintains interface compliance without recurrent belief tracking.
    """

    def __init__(self, agent_id: AgentID, action_dim: int, action_idx: Optional[int] = None):
        """
        Initializes Stateless Agent.

        Args:
            agent_id: Unique string identifier (e.g., 'agent_1').
            action_dim: Total discrete action choices (|A_j|).
            action_idx: Optional deterministic action index (e.g., 0 for constant LISTEN).
        """
        self._agent_id = str(agent_id)
        self.action_dim = int(action_dim)
        self._deterministic_action = action_idx

    @property
    def agent_id(self) -> AgentID:
        return self._agent_id

    def act(self, obs: Observation) -> Action:
        """Returns batched actions across parallel environment channels."""
        b = obs.data.size(0)
        if self._deterministic_action is not None:
            act_idx = torch.full((b, 1), self._deterministic_action, dtype=torch.float32)
        else:
            act_idx = torch.randint(0, self.action_dim, (b, 1), dtype=torch.float32)
        return Action(data=act_idx)

    def update_belief(self, obs: Observation, prev_action: Action) -> None:
        """No-op for stateless baseline agents."""
        pass

    def reset(self, batch_size: int = 1) -> None:
        """No-op for stateless baseline agents."""
        pass

    def reset_index(self, env_idx: int) -> None:
        """No-op for stateless baseline agents."""
        pass


class DiscreteJEPAAgent(IPOMDPAgent):
    """
    Stateful I-POMDP Agent maintaining recurrent belief state updates and latent MCTS planning.
    """

    def __init__(
        self,
        agent_id: AgentID,
        planner: AbstractPlanner,
        latent_dim: int,
        action_dim: int,
        device: torch.device,
        num_objects: int = 2,
        temperature: float = 1.0,
        temperature_min: float = 0.1,
        temperature_decay: float = 0.999
    ):
        """
        Initializes Discrete JEPA Agent.

        Args:
            agent_id: Unique string identifier (e.g., 'agent_0').
            planner: Planner instance implementing AbstractPlanner.
            latent_dim: Latent representation dimension per object slot.
            action_dim: Total discrete action choices (|A_i|).
            device: Hardware torch.device.
            num_objects: Number of structured object slots.
            temperature: Initial policy sampling temperature.
            temperature_min: Minimum temperature floor.
            temperature_decay: Multiplicative decay factor per decision step.
        """
        self._agent_id = str(agent_id)
        self.planner = planner
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.num_objects = int(num_objects)
        self.device = device

        self.belief: Optional[torch.Tensor] = None
        self.prev_action: Optional[torch.Tensor] = None

        self.temperature = float(temperature)
        self.temperature_min = float(temperature_min)
        self.temperature_decay = float(temperature_decay)
        self.last_value: float = 0.0

        self.reset(batch_size=1)

    @property
    def agent_id(self) -> AgentID:
        return self._agent_id

    def reset(self, batch_size: int = 1) -> None:
        """Clears hidden recurrent belief states and past action memory across batch channels."""
        self.belief = torch.zeros(batch_size, self.num_objects, self.latent_dim, device=self.device)
        self.prev_action = torch.zeros(batch_size, self.action_dim, device=self.device)

    def reset_index(self, env_idx: int) -> None:
        """Resets hidden recurrent belief for a single environment batch channel."""
        if self.belief is not None:
            self.belief[env_idx].zero_()
        if self.prev_action is not None:
            self.prev_action[env_idx].zero_()

    def update_belief(self, obs: Observation, prev_action: Action) -> None:
        """
        Updates recurrent hidden belief state: b_t = Filter(b_{t-1}, a_{t-1}, o_t).
        Converts integer indices rank-safely without losing leading batch dimensions when B = 1.
        """
        prev_a_tensor = Action.to_one_hot(prev_action.data, self.action_dim, self.device)

        obs_tensor = obs.data.to(self.device, non_blocking=True)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)


        with torch.no_grad():
            self.belief = self.planner.encode_context(obs_tensor, prev_a_tensor, self.belief)

    def act(self, obs: Observation) -> Action:
        """Determines action distribution via latent space MCTS lookahead search."""
        policy_dist = self.planner.search(root_state=self.belief, temperature=self.temperature)

        # Scalar telemetry extraction across batch channels
        if hasattr(self.planner, 'roots') and self.planner.roots:
            self.last_value = float(np.mean([r.value for r in self.planner.roots]))
        elif hasattr(self.planner, 'root') and self.planner.root is not None:
            self.last_value = float(self.planner.root.value)

        action_idx = torch.multinomial(policy_dist, num_samples=1)
        action = Action(data=action_idx.float())

        self.prev_action = F.one_hot(
            action_idx.squeeze(-1).long(), num_classes=self.action_dim
        ).float()

        return action

    def anneal_temperature(self) -> float:
        """Anneals sampling temperature towards minimum threshold."""
        self.temperature = max(self.temperature_min, self.temperature * self.temperature_decay)
        return self.temperature
