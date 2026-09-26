# ABSOLUTE PATH: src/ipomdp/agents/jepa_agent.py
# ==============================================================================
# BATCHED JEPA + LATENT-MCTS AGENT
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Recurrent Information State:
#    - The agent keeps, per parallel episode, a latent belief b_t in R^(N_obj x D_latent)
#      and the one-hot previous action a_{t-1}. observe(o_t) advances the filter
#         b_t = Filter(b_{t-1}, a_{t-1}, o_t)
#      and act() plans from b_t and records a_t for the next observe().
#
# 2. Episode Start and the Empty History:
#    - The canonical POMDP emits no observation before the first decision (see
#      src/ipomdp/domain/env.py). The agent therefore starts every episode with
#      b_{-1} = 0, a_{-1} = 0 and is shown o_0 = 0 (the all-zero one-hot, i.e. "no
#      observation yet"). Because this input is identical in every episode it carries no
#      information about the hidden state: the filter's first output is a learned
#      constant, playing the role of the prior b0. It is an agent-side encoding of the
#      empty history, not a domain observation.
#
# 3. Opponent Lives in the Environment:
#    - The agent acts in the (possibly opponent-folded) POMDP presented by the simulator;
#      it never receives the opponent's action as an input. Multi-agent reasoning enters
#      only through the learned opponent head inside the planner.
#
# 4. Plain Tensors at the Boundary:
#    - Actions are int64 tensors of shape (B,), observations are float one-hot tensors of
#      shape (B, |O|). No wrapper classes or AgentID dictionaries.
# ==============================================================================

import torch
import torch.nn.functional as F
from torch import Tensor

from ..interfaces import AbstractPlanner


class DiscreteJEPAAgent:
    """Batched agent that filters beliefs with the JEPA encoder and acts by latent MCTS."""

    def __init__(
        self,
        planner: AbstractPlanner,
        batch_size: int,
        num_actions: int,
        num_objects: int,
        latent_dim: int,
        device: torch.device,
        temperature: float,
        temperature_min: float,
        temperature_decay: float,
    ):
        """
        Args:
            planner: Latent planner providing encode_context() and search().
            batch_size: Number of parallel episodes B.
            num_actions: |A|.
            num_objects: Number of latent object slots N_obj.
            latent_dim: Dimension D_latent of each slot.
            device: Device of the belief tensors.
            temperature: Initial MCTS visit-count sampling temperature.
            temperature_min: Floor of the annealed temperature.
            temperature_decay: Multiplicative decay applied by anneal_temperature().
        """
        self.planner = planner
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.device = device
        self.temperature = temperature
        self.temperature_min = temperature_min
        self.temperature_decay = temperature_decay
        self.belief = torch.zeros(batch_size, num_objects, latent_dim, device=device)
        self.prev_action = torch.zeros(batch_size, num_actions, device=device)
        self.last_root_value = 0.0

    def reset_rows(self, mask: Tensor) -> None:
        """Clears the recurrent state of the episodes where `mask` (bool, shape (B,)) is True."""
        self.belief[mask] = 0.0
        self.prev_action[mask] = 0.0

    def observe(self, observation: Tensor) -> None:
        """
        Advances the belief filter with o_t.

        Args:
            observation: One-hot o_t of shape (B, |O|); all zeros at the start of an episode.
        """
        with torch.no_grad():
            self.belief = self.planner.encode_context(observation, self.prev_action, self.belief)

    def _record(self, action: Tensor) -> Tensor:
        self.prev_action = F.one_hot(action, num_classes=self.num_actions).float()
        return action

    def act(self) -> Tensor:
        """Plans from the current belief and samples a_t; returns int64 shape (B,)."""
        policy = self.planner.search(root_state=self.belief, temperature=self.temperature)
        self.last_root_value = float(sum(root.value for root in self.planner.roots) / len(self.planner.roots))
        return self._record(torch.multinomial(policy, num_samples=1).squeeze(-1))

    def act_uniformly(self) -> Tensor:
        """Samples a_t uniformly at random (buffer warm-up); returns int64 shape (B,)."""
        return self._record(torch.randint(0, self.num_actions, (self.batch_size,), device=self.device))

    def anneal_temperature(self) -> float:
        """Decays the sampling temperature towards its floor and returns the new value."""
        self.temperature = max(self.temperature_min, self.temperature * self.temperature_decay)
        return self.temperature
