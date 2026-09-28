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
#      JEPA         MSE( g(z_t, a_t), sg(z-bar_{t+1}) )      z-bar = EMA target filter
#      Reward       TwoHot( R(z_t, a_t), r_t )
#      Value        TwoHot( V(z_t), G^lambda_t )
#      Observation  CE( P(o | sg(z_t), a_t), o_{t+1} )       planning model, detached input
#    - No opponent terms: the model describes the POMDP the agent faces, with the opponent
#      folded into the environment (models/heads.py, section 1).
#    - Reward and value are the grounding that makes z_t a belief; JEPA self-prediction
#      alone does not (Phase-2 study, models/world_model.py section 3). The observation head
#      sees detached latents so it cannot change the representation (world_model.py section 6).
#    - Two-hot means are unbiased (models/distributions.py), so V and R decode to expected
#      returns/rewards even for multimodal targets such as Tiger's -100/+10 door reward.
#
# 3. TD(lambda) Targets With Truncation Bootstrapping:
#      G_{T} = V(z_T),   G_t = r_t + gamma * ((1 - lambda) V(z_{t+1}) + lambda G_{t+1})
#    - Every episode ends by TRUNCATION (continuing task), so the final latent is always
#      bootstrapped with V(z_T); there are no terminal states and no (1 - done) factors.
#    - gamma is the domain's discount (0.95 for canonical Tiger), passed in by main.py.
#
# 4. No Imagination Losses:
#    - An earlier version also trained V on latents imagined by a stochastic latent
#      transition ("consistency"). Planning now imagines through the real filter
#      (observation branching), so imagined latents lie on the manifold V is trained on and
#      that loss (and the transition it depended on) was removed.
#
# 5. Failure Is Loud:
#    - A non-finite loss (or a non-finite / out-of-bound two-hot target, see distributions.py)
#      raises FloatingPointError instead of silently skipping or sanitising the update.
#
# 6. Mixed Precision:
#    - The forward pass runs under bfloat16 autocast on CUDA; losses are computed in float32.
# ==============================================================================

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ..models.distributions import TwoHotSymlog
from ..models.heads import ObservationHead, RewardHead, ValueHead
from ..models.world_model import RecurrentJEPA
from .episode_buffer import EpisodeBatch


@dataclass(frozen=True)
class TrainerConfig:
    """Optimisation hyper-parameters (conf/config.yaml, section `trainer`)."""

    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    lambda_return: float


class WorldModelTrainer:
    """One gradient step of the JEPA world model and its heads on a batch of whole episodes."""

    def __init__(
        self,
        world_model: RecurrentJEPA,
        value_head: ValueHead,
        reward_head: RewardHead,
        observation_head: ObservationHead,
        codec: TwoHotSymlog,
        num_actions: int,
        num_observations: int,
        discount: float,
        config: TrainerConfig,
        device: torch.device,
    ):
        self.world_model = world_model
        self.value_head = value_head
        self.reward_head = reward_head
        self.observation_head = observation_head
        self.twohot = codec
        self.num_actions = num_actions
        self.num_observations = num_observations
        self.discount = discount
        self.config = config
        self.device = device
        self.parameters = [
            *world_model.belief_filter.parameters(),
            *world_model.predictor.parameters(),
            *value_head.parameters(),
            *reward_head.parameters(),
            *observation_head.parameters(),
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

    def train_step(self, episodes: EpisodeBatch) -> dict[str, float]:
        """
        One optimiser step on a batch of complete episodes.

        Returns:
            Scalar diagnostics (losses and value explained variance).
        """
        self.world_model.train()
        actions = F.one_hot(episodes.actions, self.num_actions).float()
        observations = F.one_hot(episodes.observations, self.num_observations).float()
        batch, steps = episodes.actions.shape

        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            latents = self.world_model.belief_filter.unroll(actions, observations)          # (B, T+1, D)
            with torch.no_grad():
                targets = self.world_model.target_filter.unroll(actions, observations)    # (B, T+1, D)

            flat = lambda x: x.reshape(batch * steps, -1)  # noqa: E731
            current = flat(latents[:, :-1])
            action_flat = flat(actions)

            predicted = self.world_model.predictor(current, action_flat)
            loss_prediction = F.mse_loss(predicted.float(), flat(targets[:, 1:]).float())

            loss_reward = self.twohot.loss(self.reward_head(current, action_flat),
                                           episodes.rewards.reshape(-1)).mean()

            with torch.no_grad():
                values = self.twohot.mean(self.value_head(latents.reshape(batch * (steps + 1), -1)))
                returns = self.lambda_returns(episodes.rewards, values.view(batch, steps + 1))
            value_logits = self.value_head(current)
            loss_value = self.twohot.loss(value_logits, returns.reshape(-1)).mean()

            loss_observation = F.cross_entropy(
                self.observation_head(current.detach(), action_flat).float(),
                episodes.observations.reshape(-1))

            total = loss_prediction + loss_reward + loss_value + loss_observation

        if not torch.isfinite(total):
            raise FloatingPointError(
                f"Non-finite loss: prediction={loss_prediction.item()}, reward={loss_reward.item()}, "
                f"value={loss_value.item()}, observation={loss_observation.item()}")

        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, self.config.grad_clip_norm)
        self.optimizer.step()
        self.world_model.update_target()

        with torch.no_grad():
            predicted_values = self.twohot.mean(value_logits)
            target_values = returns.reshape(-1)
            explained_variance = 1.0 - (target_values - predicted_values).var() / target_values.var().clamp(min=1e-8)

        return {
            "loss_total": total.item(),
            "loss_prediction": loss_prediction.item(),
            "loss_reward": loss_reward.item(),
            "loss_value": loss_value.item(),
            "loss_observation": loss_observation.item(),
            "value_explained_variance": explained_variance.item(),
        }
