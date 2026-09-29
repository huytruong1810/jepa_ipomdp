# ABSOLUTE PATH: src/ipomdp/training/rollout.py
# ==============================================================================
# PLAYING WHOLE EPISODES AND SCORING THEM
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. One Rollout Routine for Collection, Evaluation and Analysis:
#    - play_episodes runs B lock-step episodes from reset to truncation with canonical timing
#      (agent.reset(); a_t = act(); o_{t+1}, r_t = env.step(a_t); agent.update(a_t, o_{t+1})).
#      Training collection, greedy evaluation and the interpretability tools all use it, so
#      they cannot drift apart.
#
# 2. Privileged States Are Returned Separately:
#    - The hidden states s_0..s_T are returned next to the EpisodeBatch (which never contains
#      them) for diagnostics such as latent-trajectory plots and probes. They never reach the
#      agent or the trainer.
#
# 3. The Metric Is the Discounted Return:
#    - discounted_returns computes sum_t gamma^t r_t from t = 0, the quantity V*(b0) = 19.37 of
#      the canonical Tiger refers to. With 100-step episodes the truncated tail is at most
#      0.95^100 * 2000 = 11.8 in the worst case and ~0.1 for near-optimal play. The undiscounted
#      episode sum logged by an earlier version is not comparable to V*.
# ==============================================================================

from typing import Protocol

import torch
from torch import Tensor

from ..domain import BatchedPOMDPEnv
from .episode_buffer import EpisodeBatch


class Agent(Protocol):
    """What play_episodes needs: PlanningAgent, UniformRandomAgent and analysis agents implement it."""

    batch_size: int

    def reset(self) -> None: ...

    def act(self) -> Tensor: ...

    def update(self, action: Tensor, observation: Tensor) -> None: ...


def play_episodes(env: BatchedPOMDPEnv, agent: Agent) -> tuple[EpisodeBatch, Tensor]:
    """
    Plays env.batch_size complete episodes.

    Args:
        env: Simulator (its batch size must equal the agent's).
        agent: Any Agent (its own settings decide exploration).

    Returns:
        (EpisodeBatch of shape (B, T), hidden states s_0..s_T of shape (B, T + 1)).
    """
    if env.batch_size != agent.batch_size:
        raise ValueError(f"env batch {env.batch_size} != agent batch {agent.batch_size}.")
    env.reset()
    agent.reset()
    actions, observations, rewards, states = [], [], [], [env.state]
    for _ in range(env.max_steps):
        action = agent.act()
        out = env.step(action)
        agent.update(action, out.observation)
        actions.append(action)
        observations.append(out.observation)
        rewards.append(out.reward)
        states.append(env.state)
    if not bool(out.truncated.all()):
        raise RuntimeError("Episodes must truncate exactly at env.max_steps.")
    return (EpisodeBatch(torch.stack(actions, 1), torch.stack(observations, 1), torch.stack(rewards, 1)),
            torch.stack(states, 1))


def discounted_returns(rewards: Tensor, discount: float) -> Tensor:
    """sum_t gamma^t r_t for rewards of shape (B, T); float64, shape (B,)."""
    weights = discount ** torch.arange(rewards.shape[1], dtype=torch.float64, device=rewards.device)
    return rewards.double() @ weights
