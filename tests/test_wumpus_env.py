# ABSOLUTE PATH: tests/test_wumpus_env.py
"""Unit tests for the Interactive Multi-Agent Wumpus World POMDP domain."""

import pytest
import torch
from ipomdp.types import Action, Observation, State
from ipomdp.envs.wumpus import (
    MultiAgentWumpusEnv,
    FORWARD,
    TURN_LEFT,
    TURN_RIGHT,
    GRAB,
    SHOOT,
    WUMPUS_STILL,
    WUMPUS_FORWARD,
    WUMPUS_TURN_LEFT,
    WUMPUS_TURN_RIGHT,
    NORTH,
    EAST,
    SOUTH,
    WEST,
    IDX_STENCH,
    IDX_BREEZE,
    IDX_GLITTER,
    IDX_BUMP,
    IDX_SCREAM,
)
from ipomdp.envs.vector import SyncVectorEnv


class TestMultiAgentWumpusEnv:
    """Exhaustive unit test suite for MultiAgentWumpusEnv."""

    def test_wumpus_reset_and_safe_spawn(self):
        env = MultiAgentWumpusEnv(grid_size=4, pit_prob=0.3)
        obs, infos = env.reset()

        assert "agent_0" in obs and "agent_1" in obs
        assert obs["agent_0"].data.shape == torch.Size([5])
        assert obs["agent_1"].data.shape == torch.Size([5])

        # Hunter starts at (0,0) facing EAST with arrow
        assert env._hunter_pos == [0, 0]
        assert env._hunter_heading == EAST
        assert env._hunter_has_arrow is True
        assert env._hunter_alive is True
        assert env._hunter_has_gold is False

        # Safe start guarantee: (0,0), (0,1), (1,0) have no pits
        assert not env._pits[0][0]
        assert not env._pits[0][1]
        assert not env._pits[1][0]

    def test_wumpus_stationary_action(self):
        env = MultiAgentWumpusEnv(grid_size=4)
        env.reset()
        env._wumpus_pos = [2, 2]
        env._wumpus_heading = NORTH

        actions = {
            "agent_0": Action(torch.tensor([TURN_LEFT], dtype=torch.float32)),
            "agent_1": Action(torch.tensor([WUMPUS_STILL], dtype=torch.float32)),
        }
        res = env.step(actions)
        assert env._wumpus_pos == [2, 2]
        assert env._wumpus_heading == NORTH

    def test_hunter_and_wumpus_rotation_and_movement(self):
        env = MultiAgentWumpusEnv(grid_size=4)
        env.reset()
        # Clear pits to isolate kinematics
        env._pits = [[False] * 4 for _ in range(4)]
        # Force positions in middle of cave
        env._hunter_pos = [1, 1]
        env._hunter_heading = EAST
        env._wumpus_pos = [2, 2]
        env._wumpus_heading = NORTH

        # Hunter turns right (SOUTH), Wumpus turns left (WEST)
        actions = {
            "agent_0": Action(torch.tensor([TURN_RIGHT], dtype=torch.float32)),
            "agent_1": Action(torch.tensor([WUMPUS_TURN_LEFT], dtype=torch.float32)),
        }
        res = env.step(actions)
        assert env._hunter_heading == SOUTH
        assert env._wumpus_heading == WEST
        assert env._hunter_pos == [1, 1]
        assert env._wumpus_pos == [2, 2]

        # Both move forward
        actions = {
            "agent_0": Action(torch.tensor([FORWARD], dtype=torch.float32)),
            "agent_1": Action(torch.tensor([WUMPUS_FORWARD], dtype=torch.float32)),
        }
        res = env.step(actions)
        assert env._hunter_pos == [2, 1]  # Moved SOUTH
        assert env._wumpus_pos == [2, 1]  # Moved WEST -> [2, 1]

    def test_boundary_bump_detection(self):
        env = MultiAgentWumpusEnv(grid_size=4)
        env.reset()
        env._hunter_pos = [0, 0]
        env._hunter_heading = NORTH  # Facing top wall

        actions = {
            "agent_0": Action(torch.tensor([FORWARD], dtype=torch.float32)),
            "agent_1": Action(torch.tensor([WUMPUS_STILL], dtype=torch.float32)),
        }
        res = env.step(actions)
        assert env._hunter_pos == [0, 0]
        assert env._hunter_bump is True
        assert res.observations["agent_0"].data[IDX_BUMP].item() == 1.0

    def test_percept_emissions(self):
        env = MultiAgentWumpusEnv(grid_size=4)
        env.reset()
        env._hunter_pos = [1, 1]
        env._wumpus_pos = [1, 2]  # Adjacent to hunter (distance 1)
        env._pits[2][1] = True    # Pit below hunter
        env._gold_pos = [1, 1]    # Gold at hunter tile

        obs = env._get_observations()
        # Hunter perceives Stench, Breeze, Glitter
        assert obs["agent_0"].data[IDX_STENCH].item() == 1.0
        assert obs["agent_0"].data[IDX_BREEZE].item() == 1.0
        assert obs["agent_0"].data[IDX_GLITTER].item() == 1.0
        assert obs["agent_0"].data[IDX_BUMP].item() == 0.0
        assert obs["agent_0"].data[IDX_SCREAM].item() == 0.0

        # Wumpus is at (1,2) -> adjacent to pit at (2,1)? dist = |1-2|+|2-1| = 2 (not adjacent)
        assert obs["agent_1"].data[IDX_GLITTER].item() == 0.0

    def test_hunter_shoot_arrow_kills_wumpus_and_single_penalty(self):
        env = MultiAgentWumpusEnv(grid_size=4)
        env.reset()
        env._hunter_pos = [1, 0]
        env._hunter_heading = EAST
        env._wumpus_pos = [1, 3]  # In line of sight
        env._wumpus_alive = True

        actions = {
            "agent_0": Action(torch.tensor([SHOOT], dtype=torch.float32)),
            "agent_1": Action(torch.tensor([FORWARD], dtype=torch.float32)),
        }
        res = env.step(actions)

        # Wumpus is dead
        assert env._wumpus_alive is False
        assert env._hunter_has_arrow is False
        assert res.observations["agent_0"].data[IDX_SCREAM].item() == 1.0
        assert res.rewards["agent_0"].item() == -10.0
        assert res.rewards["agent_1"].item() == -1000.0  # One-time death penalty

        # Step 2: Next step while Wumpus is dead
        actions = {
            "agent_0": Action(torch.tensor([TURN_RIGHT], dtype=torch.float32)),
            "agent_1": Action(torch.tensor([WUMPUS_STILL], dtype=torch.float32)),
        }
        res2 = env.step(actions)
        # Dead Wumpus receives 0 reward (not penalized continuously)
        assert res2.rewards["agent_1"].item() == 0.0
        assert res2.observations["agent_0"].data[IDX_SCREAM].item() == 0.0
        assert res2.observations["agent_0"].data[IDX_STENCH].item() == 0.0  # Dead wumpus emits no stench

    def test_hunter_gold_grab_win(self):
        env = MultiAgentWumpusEnv(grid_size=4)
        env.reset()
        env._hunter_pos = [2, 2]
        env._gold_pos = [2, 2]
        env._pits[2][2] = False
        env._wumpus_pos = [3, 3]


        actions = {
            "agent_0": Action(torch.tensor([GRAB], dtype=torch.float32)),
            "agent_1": Action(torch.tensor([WUMPUS_STILL], dtype=torch.float32)),
        }
        res = env.step(actions)
        assert env._hunter_has_gold is True
        assert res.rewards["agent_0"].item() == 999.0  # +1000 - 1 step cost
        assert res.terminations["agent_0"].item() is True
        assert res.terminations["agent_1"].item() is True

    def test_hunter_pit_fall_death(self):
        env = MultiAgentWumpusEnv(grid_size=4)
        env.reset()
        env._hunter_pos = [1, 1]
        env._hunter_heading = EAST
        env._pits[1][2] = True  # Pit at (1, 2)
        env._wumpus_pos = [3, 3]

        actions = {
            "agent_0": Action(torch.tensor([FORWARD], dtype=torch.float32)),
            "agent_1": Action(torch.tensor([WUMPUS_STILL], dtype=torch.float32)),
        }
        res = env.step(actions)
        assert env._hunter_alive is False
        assert res.rewards["agent_0"].item() == -1001.0  # -1000 death - 1 step cost
        assert res.terminations["agent_0"].item() is True

    def test_wumpus_eats_hunter(self):
        env = MultiAgentWumpusEnv(grid_size=4)
        env.reset()
        env._hunter_pos = [1, 1]
        env._hunter_heading = EAST
        env._wumpus_pos = [1, 2]
        env._wumpus_heading = WEST
        env._wumpus_alive = True

        # Hunter moves EAST to (1,2) onto Wumpus tile
        actions = {
            "agent_0": Action(torch.tensor([FORWARD], dtype=torch.float32)),
            "agent_1": Action(torch.tensor([WUMPUS_STILL], dtype=torch.float32)),
        }
        res = env.step(actions)
        assert env._hunter_pos == [1, 2]
        assert env._wumpus_pos == [1, 2]
        assert env._hunter_alive is False
        assert res.rewards["agent_0"].item() == -1001.0  # Eaten
        assert res.rewards["agent_1"].item() == 999.0    # +1000 eat hunter - 1 step cost
        assert res.terminations["agent_0"].item() is True

    def test_wumpus_immune_to_pits(self):
        env = MultiAgentWumpusEnv(grid_size=4)
        env.reset()
        env._wumpus_pos = [1, 1]
        env._wumpus_heading = EAST
        env._pits[1][2] = True  # Pit in front of Wumpus

        actions = {
            "agent_0": Action(torch.tensor([TURN_LEFT], dtype=torch.float32)),
            "agent_1": Action(torch.tensor([WUMPUS_FORWARD], dtype=torch.float32)),
        }
        res = env.step(actions)
        assert env._wumpus_pos == [1, 2]
        assert env._wumpus_alive is True  # Immune to pits!

    def test_wumpus_render(self):
        env = MultiAgentWumpusEnv(grid_size=4)
        env.reset()
        render_str = env.render({}, None)
        assert isinstance(render_str, str)
        assert "Step: 0" in render_str
        assert "+---+" in render_str

    def test_wumpus_vector_env_execution(self):
        num_envs = 4
        env_fn = lambda: MultiAgentWumpusEnv(grid_size=4, max_steps=5)
        vec_env = SyncVectorEnv(env_fn, num_envs=num_envs)

        obs, infos = vec_env.reset()
        assert obs["agent_0"].data.shape == torch.Size([num_envs, 5])
        assert obs["agent_1"].data.shape == torch.Size([num_envs, 5])

        actions = {
            "agent_0": Action(torch.zeros(num_envs, 1, dtype=torch.float32)),
            "agent_1": Action(torch.zeros(num_envs, 1, dtype=torch.float32)),
        }
        next_obs, rews, terms, truncs, infos = vec_env.step(actions)
        assert rews["agent_0"].shape == torch.Size([num_envs, 1])
        assert rews["agent_1"].shape == torch.Size([num_envs, 1])
        assert next_obs["agent_0"].data.shape == torch.Size([num_envs, 5])
