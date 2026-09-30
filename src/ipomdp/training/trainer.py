# ABSOLUTE PATH: src/ipomdp/training/trainer.py
# ==============================================================================
# WHOLE-EPISODE WORLD-MODEL TRAINER (JEPA OR DECODER REPRESENTATION + REWARD/VALUE GROUNDING)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Whole-Episode Unrolls:
#    - Each update filters complete episodes from the learned z_0 (see episode_buffer.py),
#      so training sees exactly the latents the online agent computes. There is no burn-in,
#      no mask, and no zero-initialised mid-episode state.
#
# 2. Loss Terms of the JEPA Agent (Representation.JEPA; averaged over the B x T transitions):
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
# 2b. The Decoder Baseline (Representation.DECODER, the baseline chosen after the Tiger gate):
#      Reward       TwoHot( R(z_t, a_t), r_t )                   (unchanged)
#      Value        as above                                      (unchanged, detached input)
#      Observation  CE( P(o | z_t, a_t), o_{t+1} )                gradients INTO the filter
#    - The JEPA term, its predictor and its EMA target filter are absent (jepa=None). The
#      representation is instead shaped by the next-observation likelihood: the observation
#      head IS the decoder. This is the one change against the JEPA agent, so a comparison
#      of the two isolates what replacing observation reconstruction by latent
#      self-prediction buys; the heads, value targets, Polyak acting copies and search are
#      identical.
#    - The decoder predicts o_{t+1} from (z_t, a_t) rather than reconstructing o_t from z_t
#      (DreamerV3's posterior decoder): the filter has just read o_t, so reconstructing it can
#      be satisfied by copying the last input and carries no belief pressure.
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
#    - gamma is the domain's discount (0.95 for canonical Tiger), passed in by TrainingRun.
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
#      The value targets are computed with autocast disabled, in float32, exactly as the planner
#      evaluates the same backup (planning/search_model.py, section 4).
# ==============================================================================

from dataclasses import dataclass
from enum import StrEnum

import torch
import torch.nn.functional as F
from torch import Tensor

from ..models.distributions import TwoHotSymlog
from ..models.ema import ema_update, frozen_copy
from ..models.heads import ObservationHead, RewardHead, ValueHead
from ..models.world_model import BeliefFilter, RecurrentJEPA
from ..planning.search_model import LearnedSearchModel
from .episode_buffer import EpisodeBatch


class Representation(StrEnum):
    """What shapes the belief filter besides reward grounding (sections 2 and 2b)."""

    JEPA = "jepa"          # latent self-prediction against an EMA target; observation head detached
    DECODER = "decoder"    # next-observation likelihood of the observation head; no JEPA parts


@dataclass(frozen=True)
class TrainerConfig:
    """Optimisation hyper-parameters (conf/config.yaml, section `trainer`)."""

    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    value_target_momentum: float


class WorldModelTrainer:
    """One gradient step of the belief filter and its heads on a batch of whole episodes."""

    def __init__(
        self,
        belief_filter: BeliefFilter,
        jepa: RecurrentJEPA | None,
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
        """
        Args:
            belief_filter: The online filter z_{t+1} = f(z_t, a_t, o_{t+1}).
            jepa: The JEPA parts built around belief_filter (Representation.JEPA), or None for
                the decoder baseline (Representation.DECODER, section 2b).
        """
        if jepa is not None and jepa.belief_filter is not belief_filter:
            raise ValueError("jepa must wrap the trained belief_filter")
        self.belief_filter = belief_filter
        self.jepa = jepa
        self.representation = Representation.DECODER if jepa is None else Representation.JEPA
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
            *belief_filter.parameters(),
            *(jepa.predictor.parameters() if jepa is not None else ()),
            *value_head.parameters(),
            *reward_head.parameters(),
            *observation_head.parameters(),
        ]
        self.optimizer = torch.optim.AdamW(self.parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
        self.target_value_head = frozen_copy(value_head)
        self.target_model = LearnedSearchModel(belief_filter, reward_head, observation_head,
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

    def train_step(self, episodes: EpisodeBatch) -> dict[str, float]:
        """
        One optimiser step on a batch of complete episodes.

        Returns:
            Scalar diagnostics (losses, mean value target and mean |V - V_target|).
        """
        self.belief_filter.train()
        actions = F.one_hot(episodes.actions, self.num_actions).float()
        observations = F.one_hot(episodes.observations, self.num_observations).float()
        batch, steps = episodes.actions.shape
        losses: dict[str, Tensor] = {}

        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            latents = self.belief_filter.unroll(actions, observations)                     # (B, T+1, D)
            flat = lambda x: x.reshape(batch * steps, -1)  # noqa: E731
            current = flat(latents[:, :-1])
            action_flat = flat(actions)

            if self.jepa is not None:
                with torch.no_grad():
                    targets = self.jepa.target_filter.unroll(actions, observations)       # (B, T+1, D)
                predicted = self.jepa.predictor(current, action_flat)
                losses["prediction"] = F.mse_loss(predicted.float(), flat(targets[:, 1:]).float())

            losses["reward"] = self.twohot.loss(self.reward_head(current, action_flat),
                                                episodes.rewards.reshape(-1)).mean()

            all_latents = latents.reshape(batch * (steps + 1), -1)
            targets_value = self.value_targets(all_latents.detach())
            value_logits = self.value_head(all_latents.detach())
            losses["value"] = self.twohot.loss(value_logits, targets_value).mean()

            # The decoder baseline's representation term; a detached planning-model fit otherwise.
            decoded = current if self.representation is Representation.DECODER else current.detach()
            losses["observation"] = F.cross_entropy(self.observation_head(decoded, action_flat).float(),
                                                    episodes.observations.reshape(-1))

            total = sum(losses.values())

        if not torch.isfinite(total):
            raise FloatingPointError(
                "Non-finite loss: " + ", ".join(f"{name}={loss.item()}" for name, loss in losses.items()))

        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, self.config.grad_clip_norm)
        self.optimizer.step()
        if self.jepa is not None:
            self.jepa.update_target()
        ema_update(self.target_value_head, self.value_head, self.config.value_target_momentum)

        with torch.no_grad():
            value_error = (self.twohot.mean(value_logits) - targets_value).abs().mean()

        return {
            "loss_total": total.item(),
            **{f"loss_{name}": loss.item() for name, loss in losses.items()},
            "value_target_mean": targets_value.mean().item(),
            "value_abs_error": value_error.item(),
        }
