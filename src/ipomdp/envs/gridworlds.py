# ABSOLUTE PATH: src/ipomdp/envs/gridworlds.py
# ==============================================================================
# DISCRETE MULTI-AGENT GRIDWORLD ENVIRONMENT (UAV TARGET PURSUIT)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Instance-Level Execution with Vectorized Batching:
#    - Python grid environments operate over discrete spatial cell coordinates.
#    - Parallel environment batching is achieved by wrapping individual UAVEnv
#      instances inside SyncVectorEnv.
#
# 2. Strict State and Observation Contracts:
#    - State encapsulates exact spatial coordinates for all participating agents:
#      (UAV=0, TARGET=1) in discrete grid cell space [0, grid_size-1].
#    - Observations provide noisy directional/listening signals with probability
#      (1 - noise_prob) of perfect localization, adhering to POMDP partial observability.
# ==============================================================================

import random
from typing import Dict, Tuple, Optional
import torch

from ..interfaces import IPOMDPEnv
from ..types import AgentID, State, Observation, Action, StepResult
from ..telemetry.registry import register_env

# UAV Discrete Action Constants
UP: int = 0
DOWN: int = 1
LEFT: int = 2
RIGHT: int = 3
LISTEN: int = 4

# Agent Spatial Index Constants in Global State
UAV: int = 0
TARGET: int = 1


@register_env("uav")
class UAVEnv(IPOMDPEnv):
    """
    Multi-Agent Grid World with Noisy Spatial Observations and Pursuit Dynamics.
    Agent 0 (UAV) pursues Agent 1 (Target) on an N x N discrete lattice.
    """

    def __init__(self, grid_size: int = 3, noise_prob: float = 0.15, batch_size: int = 1):
        """
        Initializes UAV gridworld dimensions and observation noise parameters.

        Args:
            grid_size: Lattice dimension N for N x N grid (default: 3).
            noise_prob: Probability of receiving random uniform observation when listening.
            batch_size: Instance batch size (must be 1; use SyncVectorEnv for parallel batching).
        """
        super().__init__()
        assert batch_size == 1, "UAVEnv operates at instance level. Use SyncVectorEnv for parallel batching."
        self.n = int(grid_size)
        self.noise = float(noise_prob)
        self.agents = ["agent_0", "agent_1"]
        self._current_state = State(torch.zeros(2, 2, dtype=torch.float32))

        self._move_map = {
            UP: (-1, 0),
            DOWN: (1, 0),
            LEFT: (0, -1),
            RIGHT: (0, 1),
            LISTEN: (0, 0),
        }

    def reset(self) -> Tuple[Dict[AgentID, Observation], Dict[AgentID, dict]]:
        """
        Initializes agent positions uniformly at distinct random grid coordinates.

        Returns:
            Tuple of (initial_observations_dict, initial_infos_dict).
        """
        self._current_state = State(torch.zeros(2, 2, dtype=torch.float32))
        row1 = torch.randint(0, self.n, (1,)).item()
        col1 = torch.randint(0, self.n, (1,)).item()
        row2 = torch.randint(0, self.n, (1,)).item()
        col2 = torch.randint(0, self.n, (1,)).item()

        while (row1, col1) == (row2, col2):
            row2 = torch.randint(0, self.n, (1,)).item()
            col2 = torch.randint(0, self.n, (1,)).item()

        self._current_state.data[UAV, 0] = float(row1)
        self._current_state.data[UAV, 1] = float(col1)
        self._current_state.data[TARGET, 0] = float(row2)
        self._current_state.data[TARGET, 1] = float(col2)

        obs = {agent: self._uniform_obs() for agent in self.agents}
        infos = {agent: {"true_state": self._current_state} for agent in self.agents}
        return obs, infos

    def step(self, actions: Dict[AgentID, Action]) -> StepResult:
        """
        Advances environment simulation by one discrete timestep.

        Args:
            actions: Dictionary mapping AgentID to Action containers.

        Returns:
            StepResult containing observations, rewards, terminations, truncations, and infos.
        """
        current_state = self._current_state
        next_state = self._transition_dynamics(current_state, actions)
        rewards = self._get_reward(current_state, actions, next_state)
        observations = self._get_observation(next_state, actions)

        self._current_state = next_state
        capture = self._check_capture(next_state)
        terminations = {agent: torch.tensor([capture], dtype=torch.bool) for agent in self.agents}
        truncations = {agent: torch.tensor([False], dtype=torch.bool) for agent in self.agents}
        infos = {agent: {"true_state": self._current_state} for agent in self.agents}
        rewards_t = {agent: torch.tensor([r], dtype=torch.float32) for agent, r in rewards.items()}

        return StepResult(observations, rewards_t, terminations, truncations, infos)

    def _transition_dynamics(self, state: State, actions: Dict[AgentID, Action]) -> State:
        """Applies spatial translation vectors bounded by grid boundaries [0, N-1]."""
        new_state = State(state.data.clone())
        for agent_id in self.agents:
            action = int(actions[agent_id].data.view(-1)[0].item())
            agent_idx = UAV if agent_id == "agent_0" else TARGET
            if action in self._move_map:
                old_pos = state.data[agent_idx, :].tolist()
                new_pos = tuple(
                    max(0, min(self.n - 1, int(a + b)))
                    for a, b in zip(old_pos, self._move_map[action])
                )
                new_state.data[agent_idx, 0] = float(new_pos[0])
                new_state.data[agent_idx, 1] = float(new_pos[1])
        return new_state

    def _check_capture(self, state: State) -> bool:
        """Evaluates whether UAV and Target occupy identical spatial grid coordinates."""
        return torch.equal(state.data[UAV, :], state.data[TARGET, :])

    def _listen(self, state: State, subject: AgentID) -> Observation:
        """Emits noisy spatial observation of target coordinates."""
        other_agent_loc = state.data[UAV if subject == "agent_0" else TARGET, :].tolist()
        if self.n <= 1:
            return Observation(data=torch.tensor(other_agent_loc, dtype=torch.float32))

        if random.random() < self.noise:
            excluded_row, excluded_col = other_agent_loc
            weights = torch.ones(self.n ** 2).float()
            weights[int(excluded_row) * self.n + int(excluded_col)] = 0.0
            sampled_flat_index = torch.multinomial(weights, num_samples=1, replacement=False)
            obs_tensor = torch.tensor(
                [sampled_flat_index.item() // self.n, sampled_flat_index.item() % self.n],
                dtype=torch.float32
            )
        else:
            obs_tensor = torch.tensor(other_agent_loc, dtype=torch.float32)
        return Observation(data=obs_tensor)

    def _get_observation(self, state: State, actions: Dict[AgentID, Action]) -> Dict[AgentID, Observation]:
        """Emits listening observation if action is LISTEN, else uniform spatial noise."""
        observations = {}
        for agent_id in self.agents:
            action = int(actions[agent_id].data.view(-1)[0].item())
            other_id = [aid for aid in actions if aid != agent_id][0]
            observations[agent_id] = self._listen(state, other_id) if action == LISTEN else self._uniform_obs()
        return observations

    def _uniform_obs(self) -> Observation:
        """Samples uniform spatial coordinates across grid lattice."""
        row, col = torch.randint(0, self.n, size=(2,)).tolist()
        return Observation(data=torch.tensor([float(row), float(col)], dtype=torch.float32))

    def _get_reward(
        self,
        state: State,
        actions: Dict[AgentID, Action],
        next_state: Optional[State] = None
    ) -> Dict[AgentID, float]:
        """Calculates step cost (-0.01) and capture bonus (+1.0 / -1.0)."""
        rewards = {"agent_0": -0.01, "agent_1": 0.01}
        if next_state is not None and self._check_capture(next_state):
            rewards["agent_0"] += 1.0
            rewards["agent_1"] -= 1.0
        return rewards

    def render(self, actions: Dict[AgentID, Action], step_results: StepResult) -> str:
        """Renders gridworld state to ASCII string representation."""
        uav_pos = self._current_state.data[UAV, :].long().tolist()
        tgt_pos = self._current_state.data[TARGET, :].long().tolist()
        grid_str = ""
        for r in range(self.n):
            row_chars = []
            for c in range(self.n):
                if [r, c] == uav_pos and [r, c] == tgt_pos:
                    row_chars.append("X")  # Capture
                elif [r, c] == uav_pos:
                    row_chars.append("U")
                elif [r, c] == tgt_pos:
                    row_chars.append("T")
                else:
                    row_chars.append(".")
            grid_str += " ".join(row_chars) + "\n"
        return grid_str
