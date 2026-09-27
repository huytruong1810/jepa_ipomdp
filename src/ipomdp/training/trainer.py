# ABSOLUTE PATH: src/ipomdp/training/trainer.py
# ==============================================================================
# WHOLE-EPISODE WORLD-MODEL TRAINER (JEPA SELF-PREDICTION + REWARD/VALUE GROUNDING)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Whole-Episode Unrolls:
#    - Each update filters complete episodes from the learned z_0 (see episode_buffer.py),
#      so training sees exactly the latents the online agent computes. There is no burn-in,
#      no mask, and no zero-initialised mid-episode state.
#
# 2. Loss Terms (all averaged over the B x T transitions of the batch):
#      JEPA        SmoothL1( g(z_t, a_t, a^j_t, z~q), sg(z-bar_{t+1}) ) + kl_scale * max(KL, free_nats)
#                  z-bar is the EMA target filter's latent; q is the posterior over the
#                  discrete latent z; KL is the balanced prior/posterior KL (world_model.py).
#      Reward      TwoHot( R(z_t, a_t, a^j_t), r_t )
#      Value       TwoHot( V(z_t), G^lambda_t )
#      Opponent    CE( pi_j(z_t), a^j_t )
#      Imagination consistency_scale * TwoHot( V(d_k), sg(G^imag_k) ), see section 4.
#    - Reward and value are the grounding that makes z_t a belief; JEPA self-prediction
#      alone does not (Phase-2 study, documented in world_model.py section 3). VICReg was
#      removed for the reasons given there.
#
# 3. TD(lambda) Targets With Truncation Bootstrapping:
#      G_{T} = V(z_T),   G_t = r_t + gamma * ((1 - lambda) V(z_{t+1}) + lambda G_{t+1})
#    - Every episode ends by TRUNCATION (continuing task), so the final latent is always
#      bootstrapped with V(z_T); there are no terminal states and no (1 - done) factors.
#    - gamma is the domain's discount (0.95 for canonical Tiger), passed in by main.py.
#
# 4. Imagination Consistency (reviewed in Phase 3):
#    - From every real latent d_0 = z_t, the prior transition imagines d_{h+1} =
#      g(d_h, a_{t+h}, a^j_{t+h}) along the actions actually taken, for h < L_t =
#      min(H, T - t). Imagined rewards r^_h = R(d_h, a_{t+h}, a^j_{t+h}) use the latent the
#      action is taken FROM (an earlier version paired a_{t+h} with d_{h+1}, which is off by
#      one step). Targets G^imag_L = V(d_L), G^imag_k = r^_k + gamma G^imag_{k+1}; the value
#      head is trained on d_1..d_{L-1} so MCTS values imagined latents consistently.
#
# 5. Failure Is Loud:
#    - A non-finite loss (or a non-finite two-hot target, see distributions.py) raises
#      FloatingPointError instead of silently skipping or sanitising the update; silent
#      skips and NaN-to-zero substitutions hid numerical bugs in an earlier iteration.
#
# 6. Mixed Precision:
#    - The forward pass runs under bfloat16 autocast on CUDA; KL, two-hot and SmoothL1
#      losses are computed in float32.
# ==============================================================================

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ..models.distributions import TwoHotSymlog
from ..models.heads import OpponentPolicyHead, RewardHead, ValueHead
from ..models.world_model import RecurrentJEPA
from .episode_buffer import EpisodeBatch


@dataclass(frozen=True)
class TrainerConfig:
    """Optimisation and loss hyper-parameters (conf/config.yaml, section `trainer`)."""

    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    lambda_return: float
    kl_free_nats: float
    kl_scale: float
    imagination_horizon: int
    consistency_scale: float


class WorldModelTrainer:
    """One gradient step of the JEPA world model and its heads on a batch of whole episodes."""

    def __init__(
        self,
        world_model: RecurrentJEPA,
        value_head: ValueHead,
        reward_head: RewardHead,
        opponent_head: OpponentPolicyHead,
        num_actions: int,
        num_observations: int,
        num_opponent_actions: int,
        discount: float,
        config: TrainerConfig,
        device: torch.device,
    ):
        self.world_model = world_model
        self.value_head = value_head
        self.reward_head = reward_head
        self.opponent_head = opponent_head
        self.num_actions = num_actions
        self.num_observations = num_observations
        self.num_opponent_actions = num_opponent_actions
        self.discount = discount
        self.config = config
        self.device = device
        self.twohot = TwoHotSymlog().to(device)
        self.parameters = [
            *world_model.belief_filter.parameters(),
            *world_model.transition.parameters(),
            *value_head.parameters(),
            *reward_head.parameters(),
            *opponent_head.parameters(),
        ]
        self.optimizer = torch.optim.AdamW(self.parameters, lr=config.learning_rate, weight_decay=config.weight_decay)

    def lambda_returns(self, rewards: Tensor, values: Tensor) -> Tensor:
        """
        TD(lambda) targets for truncated episodes.

        Args:
            rewards: r_0..r_{T-1}, shape (B, T).
            values: V(z_0)..V(z_T), shape (B, T + 1).

        Returns:
            G_0..G_{T-1}, shape (B, T).
        """
        lam = self.config.lambda_return
        returns = torch.empty_like(rewards)
        next_return = values[:, -1]
        for t in reversed(range(rewards.shape[1])):
            next_return = rewards[:, t] + self.discount * ((1.0 - lam) * values[:, t + 1] + lam * next_return)
            returns[:, t] = next_return
        return returns

    def _imagination_loss(self, latents: Tensor, actions: Tensor, opponent_actions: Tensor) -> Tensor:
        """
        Value consistency on prior-imagined latents (module header, section 4).

        Args:
            latents: Real z_0..z_{T-1}, shape (B, T, D).
            actions: One-hot a_0..a_{T-1}, shape (B, T, |A|).
            opponent_actions: One-hot a^j_0..a^j_{T-1}, shape (B, T, |A_j|).

        Returns:
            Scalar loss averaged over all trained imagined latents.
        """
        batch, steps, dim = latents.shape
        horizon = self.config.imagination_horizon
        # L_t = min(H, T - t): imagination never runs past the actions actually recorded.
        rollout_length = torch.clamp(steps - torch.arange(steps, device=self.device), max=horizon)
        rollout_length = rollout_length.expand(batch, -1).reshape(-1)

        def shifted(x: Tensor, h: int) -> Tensor:
            """x_{t+h} for every t, zero-padded past the episode end; shape (B*T, width)."""
            return F.pad(x[:, h:], (0, 0, 0, h)).reshape(batch * steps, -1)

        imagined = [latents.reshape(-1, dim)]
        imagined_rewards = []
        for h in range(horizon):
            action, opponent_action = shifted(actions, h), shifted(opponent_actions, h)
            with torch.no_grad():
                imagined_rewards.append(self.twohot.decode(
                    self.reward_head(imagined[-1], action, opponent_action).float(), real_scale=True).squeeze(-1))
            imagined.append(self.world_model.predict_next_belief(imagined[-1], action, opponent_action))

        # Backward recursion over h = H..1. Entries with h > L_t are never read: every d_h
        # with h <= L_t sits at the end of the rollout (h == L_t) or recurses into h + 1 <= L_t.
        losses, counts = [], []
        target = torch.zeros(batch * steps, device=self.device)
        for h in range(horizon, 0, -1):
            with torch.no_grad():
                bootstrap = self.twohot.decode(self.value_head(imagined[h]).float(), real_scale=True).squeeze(-1)
                recursion = imagined_rewards[h] + self.discount * target if h < horizon else bootstrap
                target = torch.where(rollout_length == h, bootstrap, recursion)
            trained = h < rollout_length  # d_h with h < L_t has a genuine multi-step target
            if trained.any():
                loss = self.twohot(self.value_head(imagined[h][trained]), target[trained]).float()
                losses.append(loss.sum())
                counts.append(int(trained.sum()))
        if not losses:
            return torch.zeros((), device=self.device)
        return torch.stack(losses).sum() / sum(counts)

    def train_step(self, episodes: EpisodeBatch) -> dict[str, float]:
        """
        One optimiser step on a batch of complete episodes.

        Returns:
            Scalar diagnostics (losses and value explained variance).
        """
        self.world_model.train()
        cfg = self.config
        actions = F.one_hot(episodes.actions, self.num_actions).float()
        observations = F.one_hot(episodes.observations, self.num_observations).float()
        opponent_actions = F.one_hot(episodes.opponent_actions, self.num_opponent_actions).float()
        batch, steps = episodes.actions.shape

        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            latents = self.world_model.belief_filter.unroll(actions, observations)          # (B, T+1, D)
            with torch.no_grad():
                targets = self.world_model.target_filter.unroll(actions, observations)    # (B, T+1, D)

            flat = lambda x: x.reshape(batch * steps, -1)  # noqa: E731
            current = flat(latents[:, :-1])
            action_flat, opponent_flat = flat(actions), flat(opponent_actions)

            predicted, kl = self.world_model.transition.predict_train(
                current, action_flat, opponent_flat, flat(targets[:, 1:]))
            loss_prediction = F.smooth_l1_loss(predicted.float(), flat(targets[:, 1:]).float())
            loss_kl = cfg.kl_scale * kl.clamp(min=cfg.kl_free_nats).mean()

            loss_reward = self.twohot(self.reward_head(current, action_flat, opponent_flat),
                                      episodes.rewards.reshape(-1)).float().mean()

            with torch.no_grad():
                values = self.twohot.decode(
                    self.value_head(latents.reshape(batch * (steps + 1), -1)).float(), real_scale=True)
                returns = self.lambda_returns(episodes.rewards, values.view(batch, steps + 1))
            value_logits = self.value_head(current)
            loss_value = self.twohot(value_logits, returns.reshape(-1)).float().mean()

            loss_opponent = F.cross_entropy(self.opponent_head(current).float(), episodes.opponent_actions.reshape(-1))

            loss_consistency = cfg.consistency_scale * self._imagination_loss(
                latents[:, :-1], actions, opponent_actions)

            total = loss_prediction + loss_kl + loss_reward + loss_value + loss_opponent + loss_consistency

        if not torch.isfinite(total):
            raise FloatingPointError(
                f"Non-finite loss: prediction={loss_prediction.item()}, kl={loss_kl.item()}, "
                f"reward={loss_reward.item()}, value={loss_value.item()}, opponent={loss_opponent.item()}, "
                f"consistency={loss_consistency.item()}")

        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, cfg.grad_clip_norm)
        self.optimizer.step()
        self.world_model.update_target()

        with torch.no_grad():
            predicted_values = self.twohot.decode(value_logits.float(), real_scale=True).squeeze(-1)
            target_values = returns.reshape(-1)
            explained_variance = 1.0 - (target_values - predicted_values).var() / target_values.var().clamp(min=1e-8)

        return {
            "loss_total": total.item(),
            "loss_prediction": loss_prediction.item(),
            "loss_kl": loss_kl.item(),
            "loss_reward": loss_reward.item(),
            "loss_value": loss_value.item(),
            "loss_opponent": loss_opponent.item(),
            "loss_consistency": loss_consistency.item(),
            "value_explained_variance": explained_variance.item(),
        }
