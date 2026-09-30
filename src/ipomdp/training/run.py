# ABSOLUTE PATH: src/ipomdp/training/run.py
# ==============================================================================
# TRAINING RUN: THE COMPLETE, CHECKPOINTABLE STATE OF ONE EXPERIMENT
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. One Object Owns Everything That Evolves:
#    - Networks, optimiser, replay buffer, simulators, planners, agents, the collection
#      counter and the best evaluation score live in TrainingRun. state_dict() captures all of
#      it INCLUDING every random-generator state, so a run resumed from a checkpoint continues
#      bit-for-bit as if it had never stopped (tested in tests/test_training_run.py).
#    - Phase-5 finding: the earlier loop saved only network weights and the optimiser. After a
#      restart the replay buffer was empty while the warm-up phase was already over, so the
#      first update raised; and resumed runs were not reproducible.
#
# 2. Collection vs Evaluation:
#    - Collection: env_batch_size episodes with the exploring planner (root Dirichlet noise,
#      annealed visit-count temperature), or uniformly random actions during warm-up, stored
#      whole in the EpisodeBuffer and followed by updates_per_collection gradient steps.
#    - Evaluation: a separate simulator (its own seed stream) and a separate greedy planner
#      over the SAME learned model (no noise, temperature 0, argmax Q). Its mean discounted
#      return is the number compared with the exact optimum V*(b0) (training/rollout.py).
#
# 3. Seeds:
#    - Every random consumer (network initialisation, collection env, evaluation env, replay
#      sampling, planners, agents) is seeded with stream_seed(cfg.seed, RunStream.<consumer>),
#      a hash of the pair (training/seeding.py). No two consumers share a generator, within a
#      run or across the runs of a seed sweep.
#
# 4. The Agent Acts With Polyak-Averaged Weights (Phase-8 decision):
#    - The planner (collection and evaluation alike) uses EMA copies of the belief filter and the
#      reward, observation and value heads, updated after every optimiser step with
#      acting_ema_momentum (models/ema.py). Training continues on the online weights.
#    - Phase-8 finding: with a constant learning rate on Tiger's high-variance door rewards
#      (-100/+10) and bootstrapped values, the online heads jitter by 1-3 between updates
#      (door-reward error at b = 0.97 moved from -1.76 to +0.19 between two checkpoints of one
#      run), while the exact margin between opening and listening at that belief is only 0.70.
#      Whichever way the last update left the heads decided the action: one of five seeds
#      listened too long (gap -1.97) and late evaluations dipped (to 11.4 and 14.6). Averaging
#      the weights averages that jitter out without changing the objective.
#    - Momentum 0.99 averages over ~100 updates (about 1.5 collections at 64 updates each).
#      Learning escapes always-listen after ~1000 updates, so a much slower average (0.999,
#      ~16 collections) would delay the agent behind what the online model has learned.
#    - The acting copies are checkpointed with every other network and are what analysis and
#      probes read (ipomdp.experiments), since they are the model the agent actually runs.
#
# 5. Representation (training/trainer.py, sections 2 and 2b):
#    - cfg.representation selects the JEPA agent (the filter is wrapped in RecurrentJEPA with
#      its predictor and EMA target, checkpointed as "world_model") or the decoder baseline
#      (the bare filter, checkpointed as "belief_filter"; no JEPA parts are built). Heads,
#      acting copies, planners and seed streams are identical.
#    - With the same seed both start from the same filter weights. The heads are drawn after
#      the JEPA predictor (keeping the JEPA runs' initialisation, and thus the cited results,
#      reproducible), so the decoder baseline's initial head weights are a different draw --
#      no more than the difference between two seeds.
# ==============================================================================

from dataclasses import dataclass

import numpy as np
import torch

from ..agents import PlanningAgent, UniformRandomAgent
from ..domain import BatchedPOMDPEnv, FinitePOMDP
from ..models import (BeliefFilter, LatentPredictor, ObservationHead, RecurrentJEPA, RewardHead, TwoHotSymlog, ValueHead,
                      ema_update, frozen_copy)
from ..planning import BeliefTreeSearch, LearnedSearchModel
from .episode_buffer import EpisodeBuffer
from .rollout import discounted_returns, play_episodes
from .seeding import RunStream, stream_seed
from .trainer import Representation, TrainerConfig, WorldModelTrainer



@dataclass(frozen=True)
class RunConfig:
    """Everything that defines a run (built from conf/config.yaml by ipomdp.experiments.build_run_config)."""

    seed: int
    episode_length: int
    env_batch_size: int
    warmup_episodes: int
    buffer_capacity: int
    batch_size: int
    updates_per_collection: int
    eval_episodes: int
    latent_dim: int
    hidden_dim: int
    num_blocks: int
    representation: Representation
    ema_momentum: float               # JEPA target filter; unused by the decoder baseline
    acting_ema_momentum: float
    num_bins: int
    trainer: TrainerConfig
    num_simulations: int
    c_puct: float
    dirichlet_alpha: float
    dirichlet_epsilon: float
    temperature: float
    temperature_min: float
    temperature_decay: float


class TrainingRun:
    """Builds and advances one experiment (module header)."""

    def __init__(self, pomdp: FinitePOMDP, cfg: RunConfig, device: torch.device):
        self.pomdp, self.cfg, self.device = pomdp, cfg, device
        seed = lambda stream: stream_seed(cfg.seed, stream)  # noqa: E731
        torch.manual_seed(seed(RunStream.NETWORK_INIT))
        num_a, num_o = pomdp.num_actions, pomdp.num_observations

        self.belief_filter = BeliefFilter(num_a, num_o, cfg.latent_dim, cfg.hidden_dim, cfg.num_blocks).to(device)
        self.jepa = (RecurrentJEPA(self.belief_filter,
                                   LatentPredictor(cfg.latent_dim, num_a, cfg.hidden_dim, cfg.num_blocks),
                                   cfg.ema_momentum).to(device)
                     if cfg.representation is Representation.JEPA else None)
        self.value_head = ValueHead(cfg.latent_dim, cfg.hidden_dim, cfg.num_blocks, cfg.num_bins).to(device)
        self.reward_head = RewardHead(cfg.latent_dim, num_a, cfg.hidden_dim, cfg.num_blocks, cfg.num_bins).to(device)
        self.observation_head = ObservationHead(cfg.latent_dim, num_a, num_o, cfg.hidden_dim, cfg.num_blocks).to(device)
        self.codec = TwoHotSymlog(cfg.num_bins, pomdp.value_bound).to(device)
        self.trainer = WorldModelTrainer(self.belief_filter, self.jepa, self.value_head, self.reward_head, self.observation_head,
                                         self.codec, num_a, num_o, pomdp.discount, cfg.trainer, device)

        # Polyak-averaged copies the agent acts with (section 4).
        self.acting_filter = frozen_copy(self.belief_filter)
        self.acting_reward_head = frozen_copy(self.reward_head)
        self.acting_observation_head = frozen_copy(self.observation_head)
        self.acting_value_head = frozen_copy(self.value_head)
        model = LearnedSearchModel(self.acting_filter, self.acting_reward_head, self.acting_observation_head,
                                   self.acting_value_head, self.codec, num_a, num_o, pomdp.discount)
        self.train_agent = PlanningAgent(
            model, BeliefTreeSearch(model, cfg.num_simulations, cfg.c_puct, cfg.dirichlet_alpha, cfg.dirichlet_epsilon,
                                    seed=seed(RunStream.TRAIN_PLANNER)),
            cfg.env_batch_size, cfg.temperature, seed(RunStream.TRAIN_AGENT), device)
        self.eval_agent = PlanningAgent(
            model, BeliefTreeSearch(model, cfg.num_simulations, cfg.c_puct, cfg.dirichlet_alpha, dirichlet_epsilon=0.0,
                                    seed=seed(RunStream.EVAL_PLANNER)),
            cfg.eval_episodes, temperature=0.0, seed=seed(RunStream.EVAL_AGENT), device=device)
        self.warmup_agent = UniformRandomAgent(num_a, cfg.env_batch_size, seed(RunStream.WARMUP_AGENT), device)
        self.env = BatchedPOMDPEnv(pomdp, cfg.env_batch_size, cfg.episode_length, seed(RunStream.COLLECTION_ENV), device)
        self.eval_env = BatchedPOMDPEnv(pomdp, cfg.eval_episodes, cfg.episode_length, seed(RunStream.EVAL_ENV), device)
        self.buffer = EpisodeBuffer(cfg.buffer_capacity, cfg.episode_length, device, seed(RunStream.BUFFER))
        self.collection = 0
        self.best_eval_return = -float("inf")

    @property
    def networks(self) -> dict[str, torch.nn.Module]:
        representation = {"world_model": self.jepa} if self.jepa is not None else {"belief_filter": self.belief_filter}
        return {**representation, "value": self.value_head, "reward": self.reward_head,
                "observation": self.observation_head, "target_value": self.trainer.target_value_head,
                "acting_filter": self.acting_filter, "acting_reward": self.acting_reward_head,
                "acting_observation": self.acting_observation_head, "acting_value": self.acting_value_head}

    def _update_acting(self) -> None:
        """Polyak step of the acting copies towards the online weights (section 4)."""
        momentum = self.cfg.acting_ema_momentum
        for acting, online in ((self.acting_filter, self.belief_filter),
                               (self.acting_reward_head, self.reward_head),
                               (self.acting_observation_head, self.observation_head),
                               (self.acting_value_head, self.value_head)):
            ema_update(acting, online, momentum)

    @property
    def in_warmup(self) -> bool:
        return self.collection * self.cfg.env_batch_size < self.cfg.warmup_episodes

    def collect_and_train(self) -> dict[str, float]:
        """One collection of env_batch_size episodes followed by the gradient updates."""
        warmup = self.in_warmup
        episodes, _ = play_episodes(self.env, self.warmup_agent if warmup else self.train_agent)
        self.buffer.add(episodes)
        metrics: dict[str, float] = {}
        for _ in range(self.cfg.updates_per_collection):
            metrics = self.trainer.train_step(self.buffer.sample(self.cfg.batch_size))
            self._update_acting()
        if not warmup:
            agent = self.train_agent
            agent.temperature = max(self.cfg.temperature_min, agent.temperature * self.cfg.temperature_decay)
        returns = discounted_returns(episodes.rewards, self.pomdp.discount)
        metrics["collect_discounted_return"] = float(returns.mean())
        for a, name in enumerate(self.pomdp.action_names):
            metrics[f"collect_action_fraction/{name}"] = float((episodes.actions == a).double().mean())
        metrics["warmup"] = float(warmup)
        metrics["temperature"] = self.train_agent.temperature
        self.collection += 1
        return metrics

    def evaluate(self) -> dict[str, float]:
        """Greedy evaluation (section 2); updates best_eval_return."""
        episodes, _ = play_episodes(self.eval_env, self.eval_agent)
        returns = discounted_returns(episodes.rewards, self.pomdp.discount)
        mean, stderr = float(returns.mean()), float(returns.std()) / np.sqrt(len(returns))
        self.best_eval_return = max(self.best_eval_return, mean)
        metrics = {"eval_discounted_return": mean, "eval_discounted_return_stderr": stderr}
        for a, name in enumerate(self.pomdp.action_names):
            metrics[f"eval_action_fraction/{name}"] = float((episodes.actions == a).double().mean())
        return metrics

    def state_dict(self) -> dict:
        """The complete state of the run, including all random-generator states (section 1)."""
        return {
            "collection": self.collection,
            "best_eval_return": self.best_eval_return,
            "networks": {name: net.state_dict() for name, net in self.networks.items()},
            "optimizer": self.trainer.optimizer.state_dict(),
            "buffer": self.buffer.state_dict(),
            "env": self.env.state_dict(),
            "eval_env": self.eval_env.state_dict(),
            "warmup_agent": self.warmup_agent.state_dict(),
            "train_agent": self.train_agent.state_dict(),
            "eval_agent": self.eval_agent.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
        }

    def load_state_dict(self, state: dict) -> None:
        """Restores a state produced by state_dict()."""
        self.collection = state["collection"]
        self.best_eval_return = state["best_eval_return"]
        for name, net in self.networks.items():
            net.load_state_dict(state["networks"][name])
        self.trainer.optimizer.load_state_dict(state["optimizer"])
        self.buffer.load_state_dict(state["buffer"])
        self.env.load_state_dict(state["env"])
        self.eval_env.load_state_dict(state["eval_env"])
        self.warmup_agent.load_state_dict(state["warmup_agent"])
        self.train_agent.load_state_dict(state["train_agent"])
        self.eval_agent.load_state_dict(state["eval_agent"])
        torch.set_rng_state(state["torch_rng"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
