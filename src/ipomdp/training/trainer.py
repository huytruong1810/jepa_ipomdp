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
#      Value        TwoHot( V(sg(z_t)), sg(max_a [R(z_t,a) + gamma sum_o P(o|z_t,a) V-bar(tau(z_t,a,o))]) )
#                   planning model, detached input, all latents z_0..z_T
#      Observation  CE( P(o | sg(z_t), a_t), o_{t+1} )       planning model, detached input
#    - No opponent terms: the model describes the POMDP the agent faces, with the opponent
#      folded into the environment (models/heads.py, section 1).
#    - Only the JEPA and reward terms shape the representation; they are the terms grounded in
#      real data. JEPA self-prediction alone does not make z_t a belief (Phase-2 study,
#      models/world_model.py section 3); reward grounding does.
#    - The value and observation heads read DETACHED latents. For the value head this is
#      essential (Phase-5 finding): its Bellman target is computed from the model itself, so it
#      carries no information from real rewards, and letting it shape the encoder is a
#      representation-collapse pressure (making latents alike satisfies self-consistent
#      targets). With gradients into the filter the belief probe degraded from KL 0.002 to
#      0.045, the reward head lost its belief dependence (door-reward error 13) and the greedy
#      agent listened forever (return -19.88). Detached, on the same random-policy data: probe
#      KL 0.0008, door-reward error 1-2, and the learned-model greedy agent earns 17.8 +- 1.6
#      (optimal 19.28).
#    - Two-hot means are unbiased (models/distributions.py), so V and R decode to expected
#      returns/rewards even for multimodal targets such as Tiger's -100/+10 door reward.
#
# 3. Value Target: the Bellman Optimality Backup Through the Learned Model (Phase-5 decision):
#      V_target(z) = max_a [ R(z, a) + gamma * sum_o P(o | z, a) * V-bar(tau(z, a, o)) ]
#    - R and P(o | z, a) are the reward and observation heads, tau is the belief filter, and
#      V-bar is a slow EMA copy of the value head (target network). The target is computed with
#      LearnedSearchModel.expand -- literally the one-step backup the planner performs -- for
#      EVERY latent z_0..z_T of the batch (no future data is needed), without gradients.
#    - This is fitted value iteration over visited beliefs: off-policy, aiming at V* directly.
#      On an exact model it is exact value iteration (checked against the solver in the tests).
#    - Phase-5 finding that motivated it: TD(lambda) targets estimate the value of the policy
#      that COLLECTED the data. The exploring collection policy (root Dirichlet noise and a
#      visit-count temperature) opened doors at random ~9% of the time, so under it a confident
#      belief was worth -215 and the post-opening prior -482; with such leaves listening always
#      beat opening and the greedy agent listened forever (evaluation return -19.88, exactly the
#      always-listen return, at every evaluation).
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

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ..models.distributions import TwoHotSymlog
from ..models.heads import ObservationHead, RewardHead, ValueHead
from ..models.world_model import RecurrentJEPA
from ..planning.search_model import LearnedSearchModel
from .episode_buffer import EpisodeBatch


@dataclass(frozen=True)
class TrainerConfig:
    """Optimisation hyper-parameters (conf/config.yaml, section `trainer`)."""

    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    value_target_momentum: float


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
        self.target_value_head = copy.deepcopy(value_head).requires_grad_(False)
        self.target_model = LearnedSearchModel(world_model.belief_filter, reward_head, observation_head,
                                               self.target_value_head, codec, num_actions, num_observations, discount)

    @torch.no_grad()
    def value_targets(self, latents: Tensor) -> Tensor:
        """
        Bellman optimality backup through the learned model (module header, section 3).

        Args:
            latents: Belief latents of shape (N, D).

        Returns:
            max_a [R + gamma sum_o P(o) V-bar(tau)], shape (N,), float32.
        """
        with torch.autocast(self.device.type, enabled=False):
            expansion = self.target_model.expand(latents.float())
            q = expansion.rewards + self.discount * (expansion.observation_probs * expansion.next_values).sum(-1)
        return q.max(dim=-1).values

    @torch.no_grad()
    def _update_target_value_head(self) -> None:
        momentum = self.config.value_target_momentum
        for target, online in zip(self.target_value_head.parameters(), self.value_head.parameters()):
            target.mul_(momentum).add_(online, alpha=1.0 - momentum)

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

            all_latents = latents.reshape(batch * (steps + 1), -1)
            targets_value = self.value_targets(all_latents.detach())
            value_logits = self.value_head(all_latents.detach())
            loss_value = self.twohot.loss(value_logits, targets_value).mean()

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
        self._update_target_value_head()

        with torch.no_grad():
            value_error = (self.twohot.mean(value_logits) - targets_value).abs().mean()

        return {
            "loss_total": total.item(),
            "loss_prediction": loss_prediction.item(),
            "loss_reward": loss_reward.item(),
            "loss_value": loss_value.item(),
            "loss_observation": loss_observation.item(),
            "value_target_mean": targets_value.mean().item(),
            "value_abs_error": value_error.item(),
        }
