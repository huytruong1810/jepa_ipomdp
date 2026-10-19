# ABSOLUTE PATH: src/ipomdp/baselines/recurrent_dqn.py
# ==============================================================================
# MODEL-FREE RECURRENT BASELINE: R2D2-LITE (RECURRENT DOUBLE DQN, N-STEP, VALUE RESCALING)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Why This Baseline (HANDOFF 1c, item 3; reports/2026-10-19_eval_protocol/PROTOCOL.md, C-c):
#    - The paper's agent is "learned model + belief-tree search over a recurrent latent". The
#      question a reviewer asks first is what the model and the search buy over a recurrent
#      value learner trained on the same data. R2D2 (Kapturowski et al. 2019) is the standard
#      model-free answer for partially observable control: a recurrent Q-network over the
#      action-observation history, double Q-learning, n-step targets, and the invertible value
#      rescaling h of Pohlen et al. (2018). It is off-policy and value-based like our agent, so
#      the comparison isolates the learned model and the search.
#    - "Lite" = the parts that matter at this scale, each stated with its reason:
#      * GRU instead of LSTM: our belief filter is a GRU (models/world_model.py), so the
#        recurrent core is matched; the cell type is not the variable under test.
#      * Whole-episode replay from the reset, no stored recurrent state and no burn-in: Wumpus
#        windows are 50 steps, so every training sequence starts from the true initial hidden
#        state, which removes the stale-state problem burn-in exists to mitigate (R2D2 section
#        3). It is the same replay the agent's trainer uses (training/episode_buffer.py).
#      * Uniform replay instead of prioritised: the agent uses uniform replay (conf/config.yaml,
#        section 1b), so the replay scheme is matched; prioritisation is a candidate if the
#        baseline is the weaker side, stated here so it is not silently missing.
#      * One process with batched lock-step actors instead of distributed actors: per-row
#        epsilon follows the Ape-X ladder eps_i = 0.4^(1 + 7 i / (B - 1)) (Horgan et al. 2018),
#        which is what R2D2's distributed actors use.
#
# 2. Matched Budget (the comparison is at equal environment interaction):
#    - Same data stream as the agent's recipe (PROTOCOL.md "Frozen recipe"): 64 games per
#      collection, 6400 collected games (the first 256 uniformly random), a greedy evaluation
#      of 128 games every 5 collections that is ALSO stored in the replay buffer
#      (training.store_eval_episodes=true), best checkpoint = best evaluation. That is 8960
#      games = 448,000 environment steps, exactly the agent's.
#    - The collection and evaluation layouts use the agent's RunStream ids (training/seeding.py)
#      of the same training seed, so seed k of the baseline sees the same layout sequence as
#      seed k of the agent (its own actions differ, so its trajectories do).
#    - Gradient updates: 192 per collection = 19,200, the agent's count (64 per collection plus
#      two whole-buffer retrains of 6400). Each update trains on `batch_size` whole episodes.
#
# 3. Learning Rule:
#    - Input at step t: one-hot o_t (33 observations, GAME_OVER included) and one-hot a_{t-1}
#      (a "none" slot at t = 0), through a ReLU MLP into the GRU; Q(h_t, .) from a dueling
#      head (Wang et al. 2016). Zero initial hidden state (the reset is known).
#    - Target for (h_t, a_t): y_t = h( sum_{k<n'} gamma^k r_{t+k} + gamma^{n'} Q_tgt(h_{t+n'}, a*) )
#      with a* = argmax_a Q_online(h_{t+n'}, a) (double Q), n' = min(n, T - t) and the
#      bootstrap at h_T when the window ends. Bootstrapping at truncation matches the agent,
#      whose value has no time input either (partial-episode bootstrapping, Pardo et al. 2018);
#      after a win or death the absorbing GAME_OVER percept lets Q learn 0.
#    - h(x) = sign(x)(sqrt(|x| + 1) - 1) + eps x, eps = 1e-3, with its closed-form inverse: Wumpus
#      returns span about +-1000, and h is what lets one network fit +999 wins next to -1 steps.
#    - Loss: mean squared error between Q_online(h_t, a_t) (rescaled space) and y_t over all
#      B x T steps. Target network: Polyak averaging, tau per update (the agent's acting and
#      value-target copies are Polyak averages too).
#
# 4. Hyperparameters (development-set screen, PROTOCOL.md C-c; fixed before any held-out run):
#    - Defaults follow R2D2 where it applies: Adam lr 1e-4 with eps 1e-3, gradient clip 40,
#      n = 5, gamma = the domain's 0.99. The screen varies lr and n on development seeds and
#      records the chosen setting here.
# ==============================================================================

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..agents import UniformRandomAgent
from ..domain import WumpusSimulator, WumpusWorld
from ..training import EpisodeBatch, EpisodeBuffer, RunStream, discounted_returns, play_episodes, stream_seed

RESCALE_EPS = 1e-3


def value_rescale(x: Tensor) -> Tensor:
    """h(x) of Pohlen et al. (2018), section 3."""
    return torch.sign(x) * (torch.sqrt(x.abs() + 1.0) - 1.0) + RESCALE_EPS * x


def value_rescale_inverse(y: Tensor) -> Tensor:
    """Closed-form h^{-1}(y) (Pohlen et al. 2018, Proposition A.2)."""
    eps = RESCALE_EPS
    return torch.sign(y) * (((torch.sqrt(1.0 + 4.0 * eps * (y.abs() + 1.0 + eps)) - 1.0) / (2.0 * eps)) ** 2 - 1.0)


def apex_epsilons(rows: int, base: float = 0.4, alpha: float = 7.0) -> Tensor:
    """Per-row exploration eps_i = base^(1 + alpha i / (rows - 1)) (Ape-X, section 1)."""
    if rows < 2:
        raise ValueError(f"The epsilon ladder needs at least 2 rows, got {rows}.")
    return base ** (1.0 + alpha * torch.arange(rows, dtype=torch.float64) / (rows - 1))


@dataclass(frozen=True)
class RecurrentDQNConfig:
    """Section 2-4. Every field is part of a baseline run's recorded condition."""

    hidden_dim: int = 256
    learning_rate: float = 1e-4
    adam_eps: float = 1e-3
    grad_clip_norm: float = 40.0
    nstep: int = 5
    target_tau: float = 0.005
    batch_size: int = 32
    env_batch_size: int = 64
    total_episodes: int = 6400
    warmup_episodes: int = 256
    updates_per_collection: int = 192
    eval_every: int = 5
    eval_episodes: int = 128
    buffer_capacity: int = 10000


class RecurrentQNetwork(nn.Module):
    """MLP embedding of (o_t, a_{t-1}) -> GRU -> dueling Q head (section 3)."""

    def __init__(self, num_observations: int, num_actions: int, hidden_dim: int):
        super().__init__()
        self.num_observations, self.num_actions = num_observations, num_actions
        self.embed = nn.Sequential(nn.Linear(num_observations + num_actions + 1, hidden_dim), nn.ReLU())
        self.core = nn.GRU(hidden_dim, hidden_dim, batch_first=True)    # one cuDNN call per unroll (section 3)
        self.value = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.advantage = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_actions))

    def initial_state(self, batch: int, device: torch.device) -> Tensor:
        return torch.zeros(batch, self.core.hidden_size, device=device)

    def _inputs(self, previous_action: Tensor, observation: Tensor) -> Tensor:
        x = torch.cat([F.one_hot(observation, self.num_observations), F.one_hot(previous_action, self.num_actions + 1)],
                      dim=-1).float()
        return self.embed(x)

    def step(self, hidden: Tensor, previous_action: Tensor, observation: Tensor) -> Tensor:
        """h_t from h_{t-1}, a_{t-1} (num_actions = none, at t = 0) and o_t."""
        _, hidden = self.core(self._inputs(previous_action, observation).unsqueeze(1), hidden.unsqueeze(0))
        return hidden.squeeze(0)

    def q_values(self, hidden: Tensor) -> Tensor:
        """Dueling Q (rescaled space): V + A - mean_a A."""
        advantage = self.advantage(hidden)
        return self.value(hidden) + advantage - advantage.mean(dim=-1, keepdim=True)

    def unroll(self, episodes: EpisodeBatch) -> Tensor:
        """Q(h_t, .) for t = 0..T (T + 1 states: h_T is the bootstrap state), shape (B, T + 1, |A|)."""
        batch = episodes.actions.shape[0]
        device = episodes.actions.device
        none = torch.full((batch, 1), self.num_actions, dtype=torch.int64, device=device)
        previous = torch.cat([none, episodes.actions], dim=1)                                  # a_{t-1}, t = 0..T
        observations = torch.cat([episodes.initial_observations.unsqueeze(1), episodes.observations], dim=1)
        states, _ = self.core(self._inputs(previous, observations), self.initial_state(batch, device).unsqueeze(0))
        return self.q_values(states)


def nstep_targets(rewards: Tensor, bootstrap: Tensor, discount: float, nstep: int) -> Tensor:
    """
    Raw-space n-step targets, section 3: rewards (B, T); bootstrap (B, T + 1) = Q_tgt(h_s, a*_s)
    in RAW space for every s. Returns (B, T) with n' = min(n, T - t) and the bootstrap at h_{t+n'}.
    """
    batch, length = rewards.shape
    if bootstrap.shape != (batch, length + 1):
        raise ValueError(f"bootstrap must have shape ({batch}, {length + 1}), got {tuple(bootstrap.shape)}.")
    padded = torch.cat([rewards, rewards.new_zeros(batch, nstep)], dim=1)
    target = torch.zeros_like(rewards)
    for k in range(nstep):
        target = target + discount ** k * padded[:, k:k + length]
    horizon = torch.clamp(length - torch.arange(length, device=rewards.device), max=nstep)    # n' per t
    end = torch.arange(length, device=rewards.device) + horizon
    # Rewards past the window end were zero-padded above; the bootstrap is discounted by n'.
    return target + (discount ** horizon.to(rewards.dtype)) * bootstrap[:, end]


class RecurrentDQNTrainer:
    """Online network, Polyak target network and Adam (section 3)."""

    def __init__(self, network: RecurrentQNetwork, cfg: RecurrentDQNConfig, discount: float):
        self.network, self.cfg, self.discount = network, cfg, discount
        self.target = RecurrentQNetwork(network.num_observations, network.num_actions, cfg.hidden_dim).to(
            next(network.parameters()).device)
        self.target.load_state_dict(network.state_dict())
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(network.parameters(), lr=cfg.learning_rate, eps=cfg.adam_eps)

    def train_step(self, episodes: EpisodeBatch) -> Tensor:
        """One update; returns the detached loss tensor (no host sync here: the caller checks finiteness per collection)."""
        q = self.network.unroll(episodes)                                          # (B, T + 1, A)
        with torch.no_grad():
            greedy = q.argmax(dim=-1, keepdim=True)
            bootstrap = value_rescale_inverse(self.target.unroll(episodes).gather(-1, greedy).squeeze(-1))
            target = value_rescale(nstep_targets(episodes.rewards, bootstrap, self.discount, self.cfg.nstep))
        taken = q[:, :-1].gather(-1, episodes.actions.unsqueeze(-1)).squeeze(-1)
        loss = F.mse_loss(taken, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.cfg.grad_clip_norm)
        self.optimizer.step()
        with torch.no_grad():
            for online, tracked in zip(self.network.parameters(), self.target.parameters()):
                tracked.lerp_(online, self.cfg.target_tau)
        return loss.detach()


class RecurrentDQNAgent:
    """The rollout Agent protocol (training/rollout.py): epsilon-greedy per row; epsilon 0 = greedy."""

    def __init__(self, network: RecurrentQNetwork, epsilon: Tensor, seed: int, device: torch.device):
        self.network, self.batch_size, self.device = network, epsilon.shape[0], device
        self.epsilon = epsilon.to(device=device, dtype=torch.float32)
        self._generator = torch.Generator(device=device)
        self._generator.manual_seed(seed)
        self._hidden = None

    @torch.no_grad()
    def reset(self, initial_observation: Tensor | None) -> None:
        if initial_observation is None:
            raise ValueError("The recurrent DQN baseline needs the domain's initial observation (Wumpus World).")
        none = torch.full((self.batch_size,), self.network.num_actions, dtype=torch.int64, device=self.device)
        self._hidden = self.network.step(self.network.initial_state(self.batch_size, self.device), none,
                                         initial_observation)

    @torch.no_grad()
    def act(self) -> Tensor:
        greedy = self.network.q_values(self._hidden).argmax(dim=-1)
        explore = torch.rand(self.batch_size, generator=self._generator, device=self.device) < self.epsilon
        uniform = torch.randint(0, self.network.num_actions, (self.batch_size,), generator=self._generator,
                                device=self.device)
        return torch.where(explore, uniform, greedy)

    @torch.no_grad()
    def update(self, action: Tensor, observation: Tensor) -> None:
        self._hidden = self.network.step(self._hidden, action, observation)


@dataclass
class RecurrentDQNResult:
    """What train_recurrent_dqn returns: the best and final networks' weights and the evaluation curve."""

    best_state: dict
    final_state: dict
    best_collection: int
    evaluations: list[tuple[int, float]]


def train_recurrent_dqn(cfg: RecurrentDQNConfig, world: WumpusWorld, max_steps: int, seed: int,
                        device: torch.device, log=print) -> RecurrentDQNResult:
    """
    Section 2: the matched data stream (warm-up, epsilon-ladder collection, stored greedy
    evaluations, best = best evaluation) of training seed `seed`; `world` lives on `device`.
    """
    if cfg.total_episodes % cfg.env_batch_size or cfg.warmup_episodes % cfg.env_batch_size:
        raise ValueError("total_episodes and warmup_episodes must be multiples of env_batch_size.")
    stream = lambda s: stream_seed(seed, s)  # noqa: E731
    length = max_steps
    torch.manual_seed(stream(RunStream.NETWORK_INIT))
    network = RecurrentQNetwork(world.num_observations, world.num_actions, cfg.hidden_dim).to(device)
    trainer = RecurrentDQNTrainer(network, cfg, world.discount)
    buffer = EpisodeBuffer(cfg.buffer_capacity, length, True, device, stream(RunStream.BUFFER))
    collect_env = WumpusSimulator(world, cfg.env_batch_size, length, stream(RunStream.COLLECTION_ENV), device)
    eval_env = WumpusSimulator(world, cfg.eval_episodes, length, stream(RunStream.EVAL_ENV), device)
    warmup = UniformRandomAgent(world.num_actions, cfg.env_batch_size, stream(RunStream.WARMUP_AGENT), device)
    actor = RecurrentDQNAgent(network, apex_epsilons(cfg.env_batch_size), stream(RunStream.TRAIN_AGENT), device)
    evaluator = RecurrentDQNAgent(network, torch.zeros(cfg.eval_episodes), stream(RunStream.EVAL_AGENT), device)

    best_return, best_state, best_collection, evaluations = -float("inf"), None, -1, []
    collections = cfg.total_episodes // cfg.env_batch_size
    for collection in range(1, collections + 1):
        in_warmup = (collection - 1) * cfg.env_batch_size < cfg.warmup_episodes
        episodes, _ = play_episodes(collect_env, warmup if in_warmup else actor)
        buffer.add(episodes, greedy=False)
        loss = float("nan")
        if not in_warmup:
            losses = torch.stack([trainer.train_step(buffer.sample(cfg.batch_size))
                                  for _ in range(cfg.updates_per_collection)])
            if not bool(torch.isfinite(losses).all()):    # one host sync per collection, never skipped
                raise FloatingPointError(f"Non-finite recurrent DQN loss in collection {collection}.")
            loss = float(losses.mean())
        if collection % cfg.eval_every == 0:
            greedy, _ = play_episodes(eval_env, evaluator)
            buffer.add(greedy, greedy=True)
            mean = float(discounted_returns(greedy.rewards, world.discount).mean())
            evaluations.append((collection, mean))
            if mean >= best_return:
                best_return, best_collection = mean, collection
                best_state = {k: v.detach().clone() for k, v in network.state_dict().items()}
            log(f"collection {collection}: greedy eval {mean:8.2f} (best {best_return:8.2f} at {best_collection}), "
                f"mean loss {loss:.4f}")
    return RecurrentDQNResult(best_state, {k: v.detach().clone() for k, v in network.state_dict().items()},
                              best_collection, evaluations)
