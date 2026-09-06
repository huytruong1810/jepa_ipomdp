# ABSOLUTE PATH: tests/test_envs.py
"""Unit tests for multi-agent environments, Bayesian oracles, and vectorization."""

import pytest
import torch
from ipomdp.types import Action, Observation, State
from ipomdp.envs.tiger import (
    MultiAgentTigerEnv,
    TigerBayesianOracle,
    encode_observation_to_index,
    decode_index_to_observation,
    OBS_CATEGORIES,
    NUM_OBS_CLASSES,
    LISTEN,
    OPEN_LEFT,
    OPEN_RIGHT,
    TIGER_LEFT,
    TIGER_RIGHT,
    GROWL_LEFT,
    GROWL_RIGHT,
    SILENCE,
    CREAK_LEFT,
    CREAK_RIGHT,
)
from ipomdp.envs.vector import SyncVectorEnv
from ipomdp.envs.gridworlds import UAVEnv, UP, DOWN, LEFT, RIGHT, UAV, TARGET



class TestMultiAgentTigerEnv:
    """Rigorous tests for canonical Multi-Agent Tiger environment."""

    def test_reset_and_canonical_emissions(self):
        env = MultiAgentTigerEnv(growl_accuracy=0.85, max_steps=20)
        obs, infos = env.reset()

        assert "agent_0" in obs and "agent_1" in obs
        assert isinstance(obs["agent_0"], Observation)
        assert obs["agent_0"].data.shape == torch.Size([2])
        assert obs["agent_1"].data.shape == torch.Size([2])

        # State should be discrete 0 or 1
        true_s = int(infos["agent_0"]["true_state"].data.item())
        assert true_s in [TIGER_LEFT, TIGER_RIGHT]

    def test_state_persistence_on_joint_listen(self):
        env = MultiAgentTigerEnv(growl_accuracy=1.0, max_steps=20)
        env.reset()
        initial_state = int(env._current_state.data.item())

        listen_actions = {
            "agent_0": Action(torch.tensor([LISTEN], dtype=torch.float32)),
            "agent_1": Action(torch.tensor([LISTEN], dtype=torch.float32))
        }

        for _ in range(5):
            res = env.step(listen_actions)
            curr_state = int(env._current_state.data.item())
            assert curr_state == initial_state
            assert float(res.rewards["agent_0"].item()) == -1.0
            assert float(res.rewards["agent_1"].item()) == -1.0

    def test_state_stochastic_reset_on_door_open(self):
        env = MultiAgentTigerEnv(growl_accuracy=1.0, max_steps=20)
        env.reset()

        open_actions = {
            "agent_0": Action(torch.tensor([OPEN_LEFT], dtype=torch.float32)),
            "agent_1": Action(torch.tensor([LISTEN], dtype=torch.float32))
        }

        # Verify open action executes and assigns valid payoffs
        res = env.step(open_actions)
        reward_0 = float(res.rewards["agent_0"].item())
        assert reward_0 in [10.0, -100.0]

    def test_truncation_horizon(self):
        max_steps = 5
        env = MultiAgentTigerEnv(max_steps=max_steps)
        env.reset()

        listen_actions = {
            "agent_0": Action(torch.tensor([LISTEN], dtype=torch.float32)),
            "agent_1": Action(torch.tensor([LISTEN], dtype=torch.float32))
        }

        for step in range(1, max_steps + 1):
            res = env.step(listen_actions)
            is_trunc = res.truncations["agent_0"].item()
            if step < max_steps:
                assert not is_trunc
            else:
                assert is_trunc


class TestTigerBayesianOracle:
    """Rigorous tests for exact analytical Bayesian oracle."""

    def test_oracle_initial_prior(self):
        oracle = TigerBayesianOracle(growl_accuracy=0.85)
        assert oracle.belief_tiger_left == 0.5
        oracle.reset()
        assert oracle.belief_tiger_left == 0.5

    def test_oracle_bayesian_evidence_accumulation(self):
        oracle = TigerBayesianOracle(growl_accuracy=0.85)
        oracle.reset()

        growl_left_obs = torch.tensor([GROWL_LEFT, SILENCE], dtype=torch.float32)

        # After hearing growl from left, probability of Tiger-Left must increase monotonically
        prev_b = oracle.belief_tiger_left
        for _ in range(4):
            new_b = oracle.update(LISTEN, growl_left_obs)
            assert new_b > prev_b
            prev_b = new_b

        # High confidence in Tiger-Left (> 0.95)
        assert oracle.belief_tiger_left > 0.95

    def test_oracle_amnesia_reset_on_open(self):
        oracle = TigerBayesianOracle(growl_accuracy=0.85)
        oracle.reset()

        growl_left_obs = torch.tensor([GROWL_LEFT, SILENCE], dtype=torch.float32)
        for _ in range(5):
            oracle.update(LISTEN, growl_left_obs)
        assert oracle.belief_tiger_left > 0.95

        # Opening door resets prior to 0.5
        # If observation is growl left, b = (0.85 * 0.5) / ((0.85 * 0.5) + (0.15 * 0.5)) = 0.85
        post_open_b = oracle.update(OPEN_LEFT, growl_left_obs)
        assert pytest.approx(post_open_b, abs=1e-5) == 0.85

    def test_observation_distribution_and_roundtrips(self):
        oracle = TigerBayesianOracle(growl_accuracy=0.85, creak_accuracy=1.0)
        oracle.reset()

        dist = oracle.get_exact_observation_distribution(action=LISTEN, opponent_action=LISTEN)
        assert dist.shape == torch.Size([NUM_OBS_CLASSES])
        assert pytest.approx(float(dist.sum().item()), abs=1e-6) == 1.0
        assert dist.dtype == torch.float64

        for idx, (g, c) in enumerate(OBS_CATEGORIES):
            t = decode_index_to_observation(idx)
            assert pytest.approx(t[0].item(), abs=1e-3) == g
            assert pytest.approx(t[1].item(), abs=1e-3) == c
            encoded = encode_observation_to_index(t)
            assert encoded == idx


class TestSyncVectorEnv:
    """Rigorous tests for synchronous vectorized multi-agent environment wrapper."""

    def test_vector_env_shapes_and_terminal_preservation(self):
        num_envs = 4
        env_fn = lambda: MultiAgentTigerEnv(max_steps=3)
        vec_env = SyncVectorEnv(env_fn, num_envs=num_envs)

        obs, infos = vec_env.reset()
        assert obs["agent_0"].data.shape == torch.Size([num_envs, 2])
        assert infos["agent_0"]["true_state"].shape == torch.Size([num_envs, 1])

        actions = {
            "agent_0": Action(torch.zeros(num_envs, 1, dtype=torch.float32)),
            "agent_1": Action(torch.zeros(num_envs, 1, dtype=torch.float32))
        }

        # Step 1
        obs, rews, terms, truncs, infos = vec_env.step(actions)
        assert rews["agent_0"].shape == torch.Size([num_envs, 1])
        assert not truncs["agent_0"].any().item()

        # Step 2
        obs, rews, terms, truncs, infos = vec_env.step(actions)
        assert not truncs["agent_0"].any().item()

        # Step 3 (Truncation step)
        obs, rews, terms, truncs, infos = vec_env.step(actions)
        assert truncs["agent_0"].all().item()

        # Check terminal observations and states are saved in infos
        assert "terminal_obs" in infos["agent_0"]
        assert "terminal_state" in infos["agent_0"]
        assert infos["agent_0"]["terminal_obs"].shape == torch.Size([num_envs, 2])
        assert infos["agent_0"]["terminal_state"].shape == torch.Size([num_envs, 1])


class TestUAVEnv:
    """Rigorous tests for discrete multi-agent UAV gridworld."""

    def test_uav_reset_and_step(self):
        env = UAVEnv(grid_size=4, noise_prob=0.0)
        obs, infos = env.reset()

        assert "agent_0" in obs and "agent_1" in obs
        assert obs["agent_0"].data.shape == torch.Size([2])
        assert obs["agent_1"].data.shape == torch.Size([2])
        assert infos["agent_0"]["true_state"].data.shape == torch.Size([2, 2])

        # Step with movements
        actions = {
            "agent_0": Action(torch.tensor([UP], dtype=torch.float32)),
            "agent_1": Action(torch.tensor([LISTEN], dtype=torch.float32)),
        }
        res = env.step(actions)
        assert res.observations["agent_0"].data.shape == torch.Size([2])
        assert res.rewards["agent_0"].shape == torch.Size([1])
        assert not res.truncations["agent_0"].item()

    def test_uav_render(self):
        env = UAVEnv(grid_size=3)
        obs, infos = env.reset()
        actions = {
            "agent_0": Action(torch.tensor([LISTEN], dtype=torch.float32)),
            "agent_1": Action(torch.tensor([LISTEN], dtype=torch.float32)),
        }
        res = env.step(actions)
        render_str = env.render(actions, res)
        assert isinstance(render_str, str)
        assert len(render_str) > 0

