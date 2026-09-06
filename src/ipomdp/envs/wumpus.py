# ABSOLUTE PATH: src/ipomdp/envs/wumpus.py
# ==============================================================================
# INTERACTIVE MULTI-AGENT WUMPUS WORLD POMDP ENVIRONMENT
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Canonical Knowledge-Base Foundation with Adversarial Multi-Agent Extension:
#    - Extends the canonical Wumpus World (Russell & Norvig, AIMA) into a partially
#      observable game between Hunter (Agent 0) and an intelligent Wumpus (Agent 1).
#
# 2. Heading and Movement Symmetry:
#    - Both agents track spatial positions (r, c) on an N x N discrete lattice and
#      discrete orientation: NORTH (0), EAST (1), SOUTH (2), WEST (3).
#    - Agent 0 (Hunter) Actions: FORWARD (0), TURN_LEFT (1), TURN_RIGHT (2), GRAB (3), SHOOT (4).
#    - Agent 1 (Wumpus) Actions: FORWARD (0), TURN_LEFT (1), TURN_RIGHT (2).
#
# 3. Asymmetric Environmental Physics & Immunity:
#    - Wumpus is immune to pits (does not fall) and ignores gold (cannot grab).
#    - Wumpus perceives BREEZE, GLITTER, and BUMP to infer cave geometry and hunter intentions.
#    - Hunter perceives STENCH, BREEZE, GLITTER, BUMP, and SCREAM.
#
# 4. Arrow Dynamics & One-Time Death Penalty:
#    - Hunter possesses 1 arrow. Executing SHOOT casts an instant ray in current heading.
#    - If arrow intersects a live Wumpus, Wumpus dies (alive=False), Hunter receives SCREAM,
#      and Wumpus incurs a one-time death penalty (-1000). While dead, Wumpus receives 0 reward
#      and cannot move or hunt.
#
# 5. Victory & Termination Rules:
#    - Hunter wins immediately upon GRAB on the gold tile (+1000 reward, terminated=True).
#    - Hunter dies if entering a pit (-1000) or occupying the same tile as a live Wumpus (-1000).
#    - Live Wumpus receives +1000 upon eating the Hunter.
# ==============================================================================

import random
from typing import Dict, Tuple, Optional, List
import torch

from ..interfaces import IPOMDPEnv
from ..types import AgentID, State, Observation, Action, StepResult
from ..telemetry.registry import register_env

# Discrete Actions for Hunter (Agent 0)
FORWARD: int = 0
TURN_LEFT: int = 1
TURN_RIGHT: int = 2
GRAB: int = 3
SHOOT: int = 4

# Discrete Actions for Wumpus (Agent 1)
WUMPUS_STILL: int = 0
WUMPUS_FORWARD: int = 1
WUMPUS_TURN_LEFT: int = 2
WUMPUS_TURN_RIGHT: int = 3

# Headings / Orientations
NORTH: int = 0
EAST: int = 1
SOUTH: int = 2
WEST: int = 3

# Heading Coordinate Deltas (row, col)
HEADING_DELTAS: Dict[int, Tuple[int, int]] = {
    NORTH: (-1, 0),
    EAST: (0, 1),
    SOUTH: (1, 0),
    WEST: (0, -1),
}

# Observation Feature Indices (5-dimensional percept vector)
# Percept: [Stench, Breeze, Glitter, Bump, Scream]
IDX_STENCH: int = 0
IDX_BREEZE: int = 1
IDX_GLITTER: int = 2
IDX_BUMP: int = 3
IDX_SCREAM: int = 4
NUM_PERCEPTS: int = 5


@register_env("wumpus")
class MultiAgentWumpusEnv(IPOMDPEnv):
    """
    Interactive Multi-Agent Wumpus World Environment.
    Agent 0: Hunter exploring cave for gold, armed with 1 arrow.
    Agent 1: Active Wumpus hunting the agent, moving and rotating.
    """

    def __init__(
        self,
        grid_size: int = 4,
        pit_prob: float = 0.2,
        max_steps: int = 50,
        batch_size: int = 1
    ):
        """
        Initializes cave dimensions, pit generation rate, and episode horizon.

        Args:
            grid_size: Lattice dimension N for N x N cave (default: 4).
            pit_prob: Independent Bernoulli probability of pit spawn per cell (default: 0.2).
            max_steps: Maximum step horizon before episode truncation (default: 50).
            batch_size: Instance batch size (must be 1; use SyncVectorEnv for parallel batching).
        """
        super().__init__()
        assert batch_size == 1, "MultiAgentWumpusEnv operates at instance level. Use SyncVectorEnv for parallel batching."
        self.n: int = int(grid_size)
        self.pit_prob: float = float(pit_prob)
        self.max_steps: int = int(max_steps)
        self.agents: List[AgentID] = ["agent_0", "agent_1"]

        # Environment State Variables
        self._step_count: int = 0
        self._hunter_pos: List[int] = [0, 0]
        self._hunter_heading: int = EAST
        self._hunter_has_arrow: bool = True
        self._hunter_alive: bool = True
        self._hunter_has_gold: bool = False
        self._hunter_bump: bool = False

        self._wumpus_pos: List[int] = [0, 0]
        self._wumpus_heading: int = EAST
        self._wumpus_alive: bool = True
        self._wumpus_died_this_step: bool = False
        self._wumpus_bump: bool = False

        self._gold_pos: List[int] = [0, 0]
        self._pits: List[List[bool]] = [[False] * self.n for _ in range(self.n)]
        self._scream_heard: bool = False

        self._current_state: State = State(torch.zeros(self.n, self.n, dtype=torch.float32))

    def reset(self) -> Tuple[Dict[AgentID, Observation], Dict[AgentID, dict]]:
        """
        Generates a randomized cave layout with guaranteed safe start at (0, 0).

        Returns:
            Tuple of (initial_observations_dict, initial_infos_dict).
        """
        self._step_count = 0
        self._hunter_pos = [0, 0]
        self._hunter_heading = EAST
        self._hunter_has_arrow = True
        self._hunter_alive = True
        self._hunter_has_gold = False
        self._hunter_bump = False

        self._wumpus_alive = True
        self._wumpus_died_this_step = False
        self._wumpus_bump = False
        self._wumpus_heading = random.choice([NORTH, EAST, SOUTH, WEST])
        self._scream_heard = False

        # Safe start guarantee: (0,0), (0,1), and (1,0) contain no pits
        self._pits = [[False] * self.n for _ in range(self.n)]
        for r in range(self.n):
            for c in range(self.n):
                if (r, c) not in [(0, 0), (0, 1), (1, 0)]:
                    if random.random() < self.pit_prob:
                        self._pits[r][c] = True

        # Place Gold at a random tile other than (0, 0) and not in a pit
        available_gold_cells = [
            (r, c) for r in range(self.n) for c in range(self.n)
            if (r, c) != (0, 0) and not self._pits[r][c]
        ]
        if not available_gold_cells:
            # Fallback if pits saturated all non-origin cells
            self._pits[1][1] = False
            available_gold_cells = [(1, 1)]
        self._gold_pos = list(random.choice(available_gold_cells))

        # Place Wumpus at a random tile other than (0, 0)
        available_cells = [(r, c) for r in range(self.n) for c in range(self.n) if (r, c) != (0, 0)]
        self._wumpus_pos = list(random.choice(available_cells))


        self._update_global_state_tensor()

        obs = self._get_observations()
        infos = {
            agent_id: {
                "true_state": self._current_state,
                "hunter_pos": list(self._hunter_pos),
                "wumpus_pos": list(self._wumpus_pos),
                "gold_pos": list(self._gold_pos),
                "wumpus_alive": self._wumpus_alive,
            }
            for agent_id in self.agents
        }
        return obs, infos

    def step(self, actions: Dict[AgentID, Action]) -> StepResult:
        """
        Advances the multi-agent Wumpus World simulation by one timestep.

        Execution Order:
        1. Action interpretation and orientation updates.
        2. Hunter arrow shooting and ray casting (if executed).
        3. Spatial movement with boundary collision detection.
        4. Interaction resolution: Pit falls, Wumpus predation, Gold grab.
        5. Percept emission and payoff calculation.
        """
        self._step_count += 1
        self._wumpus_died_this_step = False
        self._scream_heard = False
        self._hunter_bump = False
        self._wumpus_bump = False

        act_hunter = int(actions["agent_0"].data.view(-1)[0].item())
        act_wumpus = int(actions["agent_1"].data.view(-1)[0].item()) if "agent_1" in actions else FORWARD

        # ---------------------------------------------------------------------
        # 1. Hunter Action Execution
        # ---------------------------------------------------------------------
        shot_arrow_this_step = False
        if self._hunter_alive:
            if act_hunter == TURN_LEFT:
                self._hunter_heading = (self._hunter_heading - 1) % 4
            elif act_hunter == TURN_RIGHT:
                self._hunter_heading = (self._hunter_heading + 1) % 4
            elif act_hunter == FORWARD:
                dr, dc = HEADING_DELTAS[self._hunter_heading]
                nr, nc = self._hunter_pos[0] + dr, self._hunter_pos[1] + dc
                if 0 <= nr < self.n and 0 <= nc < self.n:
                    self._hunter_pos = [nr, nc]
                else:
                    self._hunter_bump = True
            elif act_hunter == GRAB:
                if self._hunter_pos == self._gold_pos and not self._hunter_has_gold:
                    self._hunter_has_gold = True
            elif act_hunter == SHOOT:
                if self._hunter_has_arrow:
                    self._hunter_has_arrow = False
                    shot_arrow_this_step = True
                    # Cast arrow ray along current heading
                    if self._wumpus_alive:
                        dr, dc = HEADING_DELTAS[self._hunter_heading]
                        curr_r, curr_c = self._hunter_pos[0] + dr, self._hunter_pos[1] + dc
                        while 0 <= curr_r < self.n and 0 <= curr_c < self.n:
                            if [curr_r, curr_c] == self._wumpus_pos:
                                self._wumpus_alive = False
                                self._wumpus_died_this_step = True
                                self._scream_heard = True
                                break
                            curr_r += dr
                            curr_c += dc

        # ---------------------------------------------------------------------
        # 2. Wumpus Action Execution (if alive)
        # ---------------------------------------------------------------------
        if self._wumpus_alive:
            if act_wumpus == WUMPUS_TURN_LEFT:
                self._wumpus_heading = (self._wumpus_heading - 1) % 4
            elif act_wumpus == WUMPUS_TURN_RIGHT:
                self._wumpus_heading = (self._wumpus_heading + 1) % 4
            elif act_wumpus == WUMPUS_FORWARD:
                dr, dc = HEADING_DELTAS[self._wumpus_heading]
                nr, nc = self._wumpus_pos[0] + dr, self._wumpus_pos[1] + dc
                if 0 <= nr < self.n and 0 <= nc < self.n:
                    self._wumpus_pos = [nr, nc]
                else:
                    self._wumpus_bump = True
            # WUMPUS_STILL (0): Wumpus stays in place (canonical stationary mode)

        # ---------------------------------------------------------------------
        # 3. Post-Movement Interactions & Deaths
        # ---------------------------------------------------------------------
        hunter_killed_by_pit = False
        hunter_killed_by_wumpus = False

        if self._hunter_alive:
            # Pit check
            hr, hc = self._hunter_pos
            if self._pits[hr][hc]:
                self._hunter_alive = False
                hunter_killed_by_pit = True

            # Live Wumpus check
            if self._wumpus_alive and self._hunter_pos == self._wumpus_pos:
                self._hunter_alive = False
                hunter_killed_by_wumpus = True

        # ---------------------------------------------------------------------
        # 4. Payoffs & Rewards Calculation
        # ---------------------------------------------------------------------
        reward_hunter = -1.0  # Base step cost
        if shot_arrow_this_step:
            reward_hunter -= 9.0  # Total -10.0 for shooting arrow (-1 base - 9 extra)

        if self._hunter_has_gold:
            reward_hunter += 1000.0

        if hunter_killed_by_pit or hunter_killed_by_wumpus:
            reward_hunter -= 1000.0

        # Wumpus Payoff
        if self._wumpus_died_this_step:
            reward_wumpus = -1000.0  # One-time penalty upon death
        elif not self._wumpus_alive:
            reward_wumpus = 0.0      # Zero reward while dead
        else:
            reward_wumpus = -1.0     # Base active step cost
            if hunter_killed_by_wumpus:
                reward_wumpus += 1000.0

        # ---------------------------------------------------------------------
        # 5. Terminations & Truncations
        # ---------------------------------------------------------------------
        terminated = bool(self._hunter_has_gold or not self._hunter_alive)
        truncated = bool(self._step_count >= self.max_steps and not terminated)

        self._update_global_state_tensor()

        observations = self._get_observations()
        rewards_t = {
            "agent_0": torch.tensor([reward_hunter], dtype=torch.float32),
            "agent_1": torch.tensor([reward_wumpus], dtype=torch.float32),
        }
        terminations = {
            "agent_0": torch.tensor([terminated], dtype=torch.bool),
            "agent_1": torch.tensor([terminated], dtype=torch.bool),
        }
        truncations = {
            "agent_0": torch.tensor([truncated], dtype=torch.bool),
            "agent_1": torch.tensor([truncated], dtype=torch.bool),
        }
        infos = {
            agent_id: {
                "true_state": self._current_state,
                "hunter_pos": list(self._hunter_pos),
                "wumpus_pos": list(self._wumpus_pos),
                "gold_pos": list(self._gold_pos),
                "wumpus_alive": self._wumpus_alive,
                "hunter_alive": self._hunter_alive,
                "gold_grabbed": self._hunter_has_gold,
            }
            for agent_id in self.agents
        }

        return StepResult(observations, rewards_t, terminations, truncations, infos)

    def _get_observations(self) -> Dict[AgentID, Observation]:
        """Constructs 5-dimensional local percept vectors for both agents."""
        hr, hc = self._hunter_pos
        wr, wc = self._wumpus_pos

        # Stench: adjacent to live Wumpus
        stench = 0.0
        if self._wumpus_alive and (abs(hr - wr) + abs(hc - wc) <= 1):
            stench = 1.0

        # Breeze for Hunter: adjacent to any pit
        breeze_hunter = 0.0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = hr + dr, hc + dc
            if 0 <= nr < self.n and 0 <= nc < self.n and self._pits[nr][nc]:
                breeze_hunter = 1.0
                break

        # Glitter for Hunter: on gold tile and gold not yet grabbed
        glitter_hunter = 1.0 if (self._hunter_pos == self._gold_pos and not self._hunter_has_gold) else 0.0
        bump_hunter = 1.0 if self._hunter_bump else 0.0
        scream_hunter = 1.0 if self._scream_heard else 0.0

        obs_hunter = torch.tensor(
            [stench, breeze_hunter, glitter_hunter, bump_hunter, scream_hunter],
            dtype=torch.float32
        )

        # Breeze for Wumpus: adjacent to any pit
        breeze_wumpus = 0.0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = wr + dr, wc + dc
            if 0 <= nr < self.n and 0 <= nc < self.n and self._pits[nr][nc]:
                breeze_wumpus = 1.0
                break

        glitter_wumpus = 1.0 if (self._wumpus_pos == self._gold_pos and not self._hunter_has_gold) else 0.0
        bump_wumpus = 1.0 if self._wumpus_bump else 0.0

        obs_wumpus = torch.tensor(
            [0.0, breeze_wumpus, glitter_wumpus, bump_wumpus, 0.0],
            dtype=torch.float32
        )

        return {
            "agent_0": Observation(data=obs_hunter),
            "agent_1": Observation(data=obs_wumpus),
        }

    def _update_global_state_tensor(self) -> None:
        """Serializes global state into a spatial tensor representation."""
        # 6 spatial channels: [Hunter, Wumpus, Pits, Gold, HunterHeading, WumpusHeading]
        state_tensor = torch.zeros(6, self.n, self.n, dtype=torch.float32)
        if self._hunter_alive:
            state_tensor[0, self._hunter_pos[0], self._hunter_pos[1]] = 1.0
        if self._wumpus_alive:
            state_tensor[1, self._wumpus_pos[0], self._wumpus_pos[1]] = 1.0
        for r in range(self.n):
            for c in range(self.n):
                if self._pits[r][c]:
                    state_tensor[2, r, c] = 1.0
        if not self._hunter_has_gold:
            state_tensor[3, self._gold_pos[0], self._gold_pos[1]] = 1.0
        state_tensor[4, self._hunter_pos[0], self._hunter_pos[1]] = float(self._hunter_heading)
        state_tensor[5, self._wumpus_pos[0], self._wumpus_pos[1]] = float(self._wumpus_heading)

        self._current_state = State(data=state_tensor)

    def render(self, actions: Dict[AgentID, Action], step_results: StepResult) -> str:
        """Renders the Wumpus cave lattice in human-readable ASCII format."""
        heading_arrows = {NORTH: "^", EAST: ">", SOUTH: "v", WEST: "<"}
        h_arrow = heading_arrows.get(self._hunter_heading, "H")
        w_arrow = heading_arrows.get(self._wumpus_heading, "W")

        grid_lines = []
        grid_lines.append(f"+{'---+' * self.n}")
        for r in range(self.n):
            cells = []
            for c in range(self.n):
                char = "   "
                is_hunter = ([r, c] == self._hunter_pos) and self._hunter_alive
                is_wumpus = ([r, c] == self._wumpus_pos) and self._wumpus_alive
                is_pit = self._pits[r][c]
                is_gold = ([r, c] == self._gold_pos) and not self._hunter_has_gold

                if is_hunter and is_gold:
                    char = f"H{h_arrow}G"
                elif is_hunter:
                    char = f" H{h_arrow}"
                elif is_wumpus:
                    char = f" W{w_arrow}"
                elif is_pit:
                    char = " P "
                elif is_gold:
                    char = " G "
                cells.append(char)
            grid_lines.append("|" + "|".join(cells) + "|")
            grid_lines.append(f"+{'---+' * self.n}")

        status = (
            f"Step: {self._step_count} | Hunter: {'Alive' if self._hunter_alive else 'Dead'} "
            f"| Wumpus: {'Alive' if self._wumpus_alive else 'Dead'} | Gold: {'Grabbed' if self._hunter_has_gold else 'In Cave'}"
        )
        return "\n".join(grid_lines) + "\n" + status

    def _transition_dynamics(self, state: State, actions: Dict[AgentID, Action]) -> State:
        return self._current_state

    def _get_observation(self, state: State, actions: Dict[AgentID, Action]) -> Dict[AgentID, Observation]:
        return self._get_observations()

    def _get_reward(
        self,
        state: State,
        actions: Dict[AgentID, Action],
        next_state: Optional[State] = None
    ) -> Dict[AgentID, float]:
        return {"agent_0": 0.0, "agent_1": 0.0}

