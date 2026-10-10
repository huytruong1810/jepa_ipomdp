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
#    - cfg.observation_horizon = k > 0 adds the multi-step observation term (trainer.py,
#      section 2c). Its predictor is drawn after every existing network, just before the
#      trainer, so runs with k = 0 (canonical Tiger) initialise exactly as before; it is
#      checkpointed as "multistep_observation" and never read by the planner.
#
# 6. Any Domain, Through a DomainSetup:
#    - The run needs only the Domain protocol below (sizes, names, discount, value bound,
#      whether o_0 exists), a simulator factory and a per-domain episode summary. Canonical
#      Tiger supplies its FinitePOMDP and BatchedPOMDPEnv; Wumpus World its WumpusWorld,
#      WumpusSimulator and win/death rates (ipomdp.experiments builds both). Nothing in the
#      training loop reads exact model tensors, so domains without an exact model (and
#      without a solver) train identically.
#
# 7. Epsilon-Uniform Exploration (user decision, 2026-10-05; agents/planning_agent.py, section 3):
#    - After warm-up, each collected action is replaced with probability epsilon by a uniformly
#      random one. epsilon starts at cfg.epsilon and decays with the temperature
#      (temperature_decay per collection) to cfg.epsilon_min.
#    - Motivation (Wumpus diagnosis, HANDOFF section 5.4, item 14): the observation head's
#      death prediction is miscalibrated because dangerous FORWARD outcomes are rare in the
#      agent's data; one extra forced FORWARD step from every live reference state halved the
#      head's danger error. Uniform actions in the states the agent actually reaches are the
#      domain-free version of that augmentation.
#    - The value targets are Bellman optimality backups through the learned model (trainer.py),
#      so off-policy actions do not bias them; greedy evaluation never uses epsilon.
#    - cfg.epsilon = 0 (the default, canonical Tiger) draws no extra random numbers.
#
# 8. Greedy Evaluation Games as Training Data (user decision, 2026-10-05):
#    - cfg.store_eval_episodes adds every greedy evaluation's episodes to the replay buffer.
#    - Motivation (reports/2026-10-02_wumpus_diagnostics/head_shift.py): on Wumpus the observation
#      head is calibrated on its training data (buffer FORWARD steps: predicted death 0.071,
#      realised 0.071) but not on the greedy planner's own histories (0.002 vs 0.097). The
#      exploring policy does not reach the states the greedy planner reaches, and the planner
#      steps exactly where the head is wrong. A fresh head trained with greedy games added read
#      0.53 at held-out dangerous steps (realised 0.67) against 0.17 from the buffer alone.
#    - Value targets are Bellman optimality backups through the model, so mixing in greedy data
#      biases nothing. Evaluation keeps its own simulator stream, and every evaluation plays fresh
#      layouts, so storing the games does not leak evaluation layouts into later evaluations.
#    - Off by default (canonical Tiger): the buffer and every random stream are then untouched.
#
# 9. Time-Aware (Finite-Horizon) Value (user decision, 2026-10-07):
#    - cfg.time_aware_value replaces the value head by HorizonValueHead V(z, k) over the steps
#      left in the episode_length window. The trainer's targets and the planner's search model
#      count k down to a terminal 0 (models/heads.py, 2b; trainer.py, 3b; search_model.py, 5).
#    - Motivation: on Wumpus World the greedy planner idled in front of provably safe cells,
#      because a stationary value bootstraps through the window's truncation and a wasted step
#      looks almost free (reports/2026-10-02_wumpus_diagnostics/unfinished_diag.py).
#    - The value head is built in the same place in the initialisation order. With the flag off
#      (the default, canonical Tiger) it is the same ValueHead, so the run is unchanged.
#    - Result 2026-10-08 (Wumpus, current recipe + this flag, 3 seeds, paired): returns 200 / 108 / 143
#      against 186 / 166 / 215; still ~69% unfinished (the agent idles with GRAB instead of SHOOT).
#      The truncation was not the cause. Kept as an option, not used.
#
# 10. n-Step Greedy-Return Value Grounding (user decision, 2026-10-08; trainer.py, section 3c):
#    - cfg.value_nstep = n > 0 mixes n-step returns of the stored greedy evaluation games into the
#      value targets with weight cfg.value_nstep_mix. It requires store_eval_episodes (section 8),
#      the only source of greedy games; the buffer records each game's source (episode_buffer.py, 4).
#    - Motivation: reports/2026-10-02_wumpus_diagnostics/value_chain_diag.py (the bootstrapped value
#      is ungrounded on the greedy planner's idle chains).
#    - n = 0 (the default, canonical Tiger) builds and draws nothing new.
#    - Result 2026-10-09 (Wumpus, current recipe + n = 10, mix 0.5, 3 seeds, paired): returns 88 / 93 / 126
#      against 186 / 166 / 215; unfinished 68-79%. Idle V-hat fell (650 -> 290-380) but the value still
#      ranks a no-op above stepping into a safe cell (median gap -30 to -37, was -15): the greedy
#      policy's own returns are uniformly poor and do not separate exploring from idling.
#      Kept as an option, not used.
#
# 11. Detached Planning Head (user decision, 2026-10-10; trainer.py, section 3d):
#    - cfg.detached_planning_head builds a second ObservationHead trained on detached latents. The
#      planners and the value targets use it (through its Polyak copy for acting); the joint
#      observation head only shapes the decoder's filter. Decoder representation only.
#    - Motivation: reports/2026-10-02_wumpus_diagnostics/objective_ab.py, section 2d (the latent keeps
#      improving with more updates while the joint head overfits). Offline, an NLL-early-stopped
#      fresh head was not better on confident "safe" misses, but the offline pools hold no greedy
#      games, which the agent stores (section 8) and which fixed most deaths; hence the agent test.
#    - Built after every other network and only when enabled, so the default (canonical Tiger)
#      draws and builds nothing new.
# ==============================================================================

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from ..agents import PlanningAgent, UniformRandomAgent
from ..models import (BeliefFilter, HorizonValueHead, LatentPredictor, MultiStepObservationPredictor, ObservationHead, RecurrentJEPA, RewardHead, TwoHotSymlog, ValueHead,
                      ema_update, frozen_copy)
from ..planning import BeliefTreeSearch, LearnedSearchModel
from .episode_buffer import EpisodeBatch, EpisodeBuffer
from .rollout import Simulator, discounted_returns, play_episodes
from .seeding import RunStream, stream_seed
from .trainer import Representation, TrainerConfig, WorldModelTrainer



class Domain(Protocol):
    """What a training run needs to know about a domain (section 6)."""

    action_names: tuple[str, ...]
    observation_names: tuple[str, ...]
    discount: float

    @property
    def num_actions(self) -> int: ...

    @property
    def num_observations(self) -> int: ...

    @property
    def value_bound(self) -> float: ...

    @property
    def has_initial_observation(self) -> bool: ...


@dataclass(frozen=True)
class DomainSetup:
    """
    A domain wired for training (section 6).

    Attributes:
        domain: Sizes, names, discount, value bound and o_0 flag.
        make_simulator: (batch_size, max_steps, seed, device) -> a seeded Simulator.
        episode_metrics: Domain-specific summary of played episodes (e.g. win rate); may be empty.
    """

    domain: Domain
    make_simulator: Callable[[int, int, int, torch.device], Simulator]
    episode_metrics: Callable[[EpisodeBatch], dict[str, float]]


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
    observation_horizon: int          # multi-step observation term (trainer.py, 2c); 0 = off
    ema_momentum: float               # JEPA target filter; unused by the decoder baseline
    acting_ema_momentum: float
    num_bins: int
    trainer: TrainerConfig
    num_simulations: int
    observation_prune: float          # planning/mcts.py, section 6
    c_puct: float
    dirichlet_alpha: float
    dirichlet_epsilon: float
    temperature: float
    temperature_min: float
    temperature_decay: float
    epsilon: float                    # epsilon-uniform exploration in collection (section 7); 0 = off
    epsilon_min: float
    store_eval_episodes: bool         # add greedy evaluation games to the replay buffer (section 8)
    time_aware_value: bool            # finite-horizon value V(z, steps left) (section 9)
    value_nstep: int                  # n-step greedy-return value grounding (section 10); 0 = off
    value_nstep_mix: float
    detached_planning_head: bool      # planner reads a head on detached latents (section 11)


class TrainingRun:
    """Builds and advances one experiment (module header)."""

    def __init__(self, setup: DomainSetup, cfg: RunConfig, device: torch.device):
        self.setup, self.cfg, self.device = setup, cfg, device
        if cfg.value_nstep > 0 and not cfg.store_eval_episodes:
            raise ValueError("value_nstep > 0 needs store_eval_episodes: greedy games are its only data (section 10).")
        domain = self.domain = setup.domain
        seed = lambda stream: stream_seed(cfg.seed, stream)  # noqa: E731
        torch.manual_seed(seed(RunStream.NETWORK_INIT))
        num_a, num_o = domain.num_actions, domain.num_observations

        self.belief_filter = BeliefFilter(num_a, num_o, cfg.latent_dim, cfg.hidden_dim, cfg.num_blocks).to(device)
        self.jepa = (RecurrentJEPA(self.belief_filter,
                                   LatentPredictor(cfg.latent_dim, num_a, cfg.hidden_dim, cfg.num_blocks),
                                   cfg.ema_momentum).to(device)
                     if cfg.representation is Representation.JEPA else None)
        horizon = cfg.episode_length if cfg.time_aware_value else None
        self.value_head = (HorizonValueHead(cfg.latent_dim, cfg.hidden_dim, cfg.num_blocks, cfg.num_bins, horizon)
                           if cfg.time_aware_value
                           else ValueHead(cfg.latent_dim, cfg.hidden_dim, cfg.num_blocks, cfg.num_bins)).to(device)
        self.reward_head = RewardHead(cfg.latent_dim, num_a, cfg.hidden_dim, cfg.num_blocks, cfg.num_bins).to(device)
        self.observation_head = ObservationHead(cfg.latent_dim, num_a, num_o, cfg.hidden_dim, cfg.num_blocks).to(device)
        self.codec = TwoHotSymlog(cfg.num_bins, domain.value_bound).to(device)
        self.multistep_predictor = (MultiStepObservationPredictor(cfg.latent_dim, num_a, num_o, cfg.hidden_dim,
                                                                  cfg.num_blocks, cfg.observation_horizon).to(device)
                                    if cfg.observation_horizon > 0 else None)
        self.planning_observation_head = (ObservationHead(cfg.latent_dim, num_a, num_o, cfg.hidden_dim, cfg.num_blocks).to(device)
                                          if cfg.detached_planning_head else None)                 # section 11
        self.trainer = WorldModelTrainer(self.belief_filter, self.jepa, self.value_head, self.reward_head, self.observation_head,
                                         self.codec, num_a, num_o, domain.discount, cfg.trainer, device,
                                         self.multistep_predictor, horizon, cfg.value_nstep, cfg.value_nstep_mix,
                                         self.planning_observation_head)

        # Polyak-averaged copies the agent acts with (section 4).
        self.acting_filter = frozen_copy(self.belief_filter)
        self.acting_reward_head = frozen_copy(self.reward_head)
        self.acting_observation_head = frozen_copy(self.observation_head)
        self.acting_value_head = frozen_copy(self.value_head)
        self.acting_planning_observation_head = (frozen_copy(self.planning_observation_head)
                                                 if self.planning_observation_head is not None else None)
        planner_observation = (self.acting_observation_head if self.acting_planning_observation_head is None
                               else self.acting_planning_observation_head)
        model = LearnedSearchModel(self.acting_filter, self.acting_reward_head, planner_observation,
                                   self.acting_value_head, self.codec, num_a, num_o, domain.discount, horizon)
        self.train_agent = PlanningAgent(
            model, BeliefTreeSearch(model, cfg.num_simulations, cfg.c_puct, cfg.dirichlet_alpha, cfg.dirichlet_epsilon,
                                    seed=seed(RunStream.TRAIN_PLANNER), observation_prune=cfg.observation_prune),
            cfg.env_batch_size, cfg.temperature, seed(RunStream.TRAIN_AGENT), device, epsilon=cfg.epsilon)
        self.eval_agent = PlanningAgent(
            model, BeliefTreeSearch(model, cfg.num_simulations, cfg.c_puct, cfg.dirichlet_alpha, dirichlet_epsilon=0.0,
                                    seed=seed(RunStream.EVAL_PLANNER), observation_prune=cfg.observation_prune),
            cfg.eval_episodes, temperature=0.0, seed=seed(RunStream.EVAL_AGENT), device=device)
        self.warmup_agent = UniformRandomAgent(num_a, cfg.env_batch_size, seed(RunStream.WARMUP_AGENT), device)
        self.env = setup.make_simulator(cfg.env_batch_size, cfg.episode_length, seed(RunStream.COLLECTION_ENV), device)
        self.eval_env = setup.make_simulator(cfg.eval_episodes, cfg.episode_length, seed(RunStream.EVAL_ENV), device)
        self.buffer = EpisodeBuffer(cfg.buffer_capacity, cfg.episode_length, domain.has_initial_observation, device,
                                    seed(RunStream.BUFFER))
        self.collection = 0
        self.best_eval_return = -float("inf")
        self.domain_metric_names: tuple[str, ...] = ()     # keys of setup.episode_metrics, set by evaluate()

    @property
    def networks(self) -> dict[str, torch.nn.Module]:
        representation = {"world_model": self.jepa} if self.jepa is not None else {"belief_filter": self.belief_filter}
        multistep = {"multistep_observation": self.multistep_predictor} if self.multistep_predictor is not None else {}
        planning = ({"planning_observation": self.planning_observation_head,
                     "acting_planning_observation": self.acting_planning_observation_head}
                    if self.planning_observation_head is not None else {})
        return {**representation, **multistep, "value": self.value_head, "reward": self.reward_head,
                "observation": self.observation_head, "target_value": self.trainer.target_value_head,
                "acting_filter": self.acting_filter, "acting_reward": self.acting_reward_head,
                "acting_observation": self.acting_observation_head, "acting_value": self.acting_value_head, **planning}

    def _update_acting(self) -> None:
        """Polyak step of the acting copies towards the online weights (section 4)."""
        momentum = self.cfg.acting_ema_momentum
        for acting, online in ((self.acting_filter, self.belief_filter),
                               (self.acting_reward_head, self.reward_head),
                               (self.acting_observation_head, self.observation_head),
                               (self.acting_value_head, self.value_head)):
            ema_update(acting, online, momentum)
        if self.planning_observation_head is not None:                                  # section 11
            ema_update(self.acting_planning_observation_head, self.planning_observation_head, momentum)

    @property
    def in_warmup(self) -> bool:
        return self.collection * self.cfg.env_batch_size < self.cfg.warmup_episodes

    def collect_and_train(self) -> dict[str, float]:
        """One collection of env_batch_size episodes followed by the gradient updates."""
        warmup = self.in_warmup
        episodes, _ = play_episodes(self.env, self.warmup_agent if warmup else self.train_agent)
        self.buffer.add(episodes, greedy=False)
        metrics: dict[str, float] = {}
        for _ in range(self.cfg.updates_per_collection):
            metrics = self.trainer.train_step(self.buffer.sample(self.cfg.batch_size))
            self._update_acting()
        if not warmup:
            agent = self.train_agent
            agent.temperature = max(self.cfg.temperature_min, agent.temperature * self.cfg.temperature_decay)
            agent.epsilon = max(self.cfg.epsilon_min, agent.epsilon * self.cfg.temperature_decay)
        returns = discounted_returns(episodes.rewards, self.domain.discount)
        metrics["collect_discounted_return"] = float(returns.mean())
        metrics |= {f"collect_{k}": v for k, v in self.setup.episode_metrics(episodes).items()}
        for a, name in enumerate(self.domain.action_names):
            metrics[f"collect_action_fraction/{name}"] = float((episodes.actions == a).double().mean())
        metrics["warmup"] = float(warmup)
        metrics["temperature"] = self.train_agent.temperature
        metrics["epsilon"] = self.train_agent.epsilon
        self.collection += 1
        return metrics

    def evaluate(self) -> dict[str, float]:
        """Greedy evaluation (section 2); updates best_eval_return."""
        episodes, _ = play_episodes(self.eval_env, self.eval_agent)
        if self.cfg.store_eval_episodes:
            self.buffer.add(episodes, greedy=True)
        returns = discounted_returns(episodes.rewards, self.domain.discount)
        mean, stderr = float(returns.mean()), float(returns.std()) / np.sqrt(len(returns))
        self.best_eval_return = max(self.best_eval_return, mean)
        metrics = {"eval_discounted_return": mean, "eval_discounted_return_stderr": stderr}
        domain_metrics = self.setup.episode_metrics(episodes)
        self.domain_metric_names = tuple(domain_metrics)
        metrics |= {f"eval_{k}": v for k, v in domain_metrics.items()}
        for a, name in enumerate(self.domain.action_names):
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
