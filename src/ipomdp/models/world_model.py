# ABSOLUTE PATH: src/ipomdp/models/world_model.py
# ==============================================================================
# RECURRENT JEPA WORLD MODEL: BELIEF FILTER, EMA TARGET, STOCHASTIC LATENT TRANSITION
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. The Belief Filter Mirrors the Exact Bayes Filter's Signature:
#    - Exact:   b0 given,        b_{t+1} = tau(b_t, a_t, o_{t+1})   (src/ipomdp/domain/belief.py)
#    - Learned: z_0 = learned,   z_{t+1} = GRU([a_t, o_{t+1}], z_t)
#      The learned initial latent z_0 is the network's representation of the prior b0. There
#      is no observation at t = 0 (canonical Tiger), so no placeholder input is needed.
#      z_t = phi(h_t) is a function of the history h_t = (a_0, o_1, ..., a_{t-1}, o_t), and
#      the Phase-2 acceptance test probes it against the exact posterior b*(h_t).
#
# 2. One Latent Vector (No Object Slots):
#    - Single-agent Tiger has one binary hidden variable. An earlier iteration split the
#      latent into N_obj "object slots" with slot embeddings, inter-slot attention and
#      attention pooling. Nothing gave those slots meaning, and the per-slot offsets let
#      VICReg's variance target be met without encoding any information. Structured slots
#      will be reintroduced only when the interactive state (S x M_j) gives them semantics.
#
# 3. Training Signal (see src/ipomdp/training/trainer.py):
#    - JEPA self-prediction: the transition predicts the EMA target filter's next latent
#      z-bar_{t+1} from (z_t, a_t), in latent space, with no observation reconstruction.
#    - Grounding: reward prediction from (z_t, a_t) (plus the value/TD targets).
#    - Measured on canonical Tiger (Phase-2 isolated study, random-policy data): JEPA
#      self-prediction ALONE leaves the latent no more belief-like than an untrained network
#      (probe KL 0.018 vs 0.024 nats); reward grounding brings it to 0.0015 (linear probe) /
#      0.0002 (MLP probe). This matches the self-predictive RL analysis of Ni et al. (ICLR
#      2024): latent self-prediction has uninformative fixed points unless grounded. VICReg
#      was removed: applied to the encoder it made the latent WORSE (KL 0.076 without reward,
#      0.0026 vs 0.0015 with reward), and its old placement (on predictor outputs, slots
#      pooled into one batch) could not prevent encoder collapse at all.
#
# 4. EMA Target Filter:
#    - A frozen copy of the filter, updated as target <- m * target + (1 - m) * online after
#      every optimiser step, produces the self-prediction targets (BYOL/JEPA stop-gradient).
#
# 5. Stochastic Latent Transition (DreamerV3-style; reviewed in Phase 3):
#    - In a POMDP the next latent is random (it depends on o_{t+1}). The transition therefore
#      samples a discrete latent z ~ Cat(N_cat x N_class): from a posterior q(z | z_t, a, z-bar_{t+1})
#      during training and from a prior p(z | z_t, a) during imagination (MCTS), with
#      straight-through gradients and KL balancing (alpha = 0.8) between the two.
#    - Opponent actions enter as a one-hot; the single-agent POMDP passes a singleton
#      opponent action space until Phases 3-4 settle the opponent model.
# ==============================================================================

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .layers import build_residual_stack


class BeliefFilter(nn.Module):
    """Learned recurrent belief filter z_{t+1} = f(z_t, a_t, o_{t+1}) with a learned z_0."""

    def __init__(self, num_actions: int, num_observations: int, latent_dim: int, hidden_dim: int, num_blocks: int):
        """
        Args:
            num_actions: |A|; actions are fed one-hot.
            num_observations: |O|; observations are fed one-hot.
            latent_dim: Dimension D of the belief latent.
            hidden_dim: Width of the input network.
            num_blocks: SwiGLU residual blocks in the input network.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.initial_latent = nn.Parameter(torch.zeros(latent_dim))
        self.input_net = build_residual_stack(num_actions + num_observations, hidden_dim, hidden_dim, num_blocks)
        self.cell = nn.GRUCell(hidden_dim, latent_dim)

    def initial(self, batch_size: int) -> Tensor:
        """z_0 replicated to shape (batch_size, D)."""
        return self.initial_latent.expand(batch_size, -1)

    def step(self, latent: Tensor, action: Tensor, observation: Tensor) -> Tensor:
        """
        One filter step.

        Args:
            latent: z_t, shape (B, D).
            action: One-hot a_t, shape (B, |A|).
            observation: One-hot o_{t+1}, shape (B, |O|).

        Returns:
            z_{t+1}, shape (B, D).
        """
        return self.cell(self.input_net(torch.cat([action, observation], dim=-1)), latent)

    def unroll(self, actions: Tensor, observations: Tensor) -> Tensor:
        """
        Filters whole episodes from z_0.

        Args:
            actions: One-hot a_0..a_{T-1}, shape (B, T, |A|).
            observations: One-hot o_1..o_T, shape (B, T, |O|).

        Returns:
            z_0..z_T, shape (B, T + 1, D).
        """
        latent = self.initial(actions.shape[0])
        latents = [latent]
        for t in range(actions.shape[1]):
            latent = self.step(latent, actions[:, t], observations[:, t])
            latents.append(latent)
        return torch.stack(latents, dim=1)


class LatentTransition(nn.Module):
    """Stochastic latent dynamics z_{t+1} = g(z_t, a_t, a^j_t, z), z ~ q (train) or p (imagine)."""

    def __init__(
        self,
        latent_dim: int,
        num_actions: int,
        num_opponent_actions: int,
        hidden_dim: int,
        num_blocks: int,
        num_categoricals: int,
        num_classes: int,
        kl_balance: float,
    ):
        """
        Args:
            latent_dim: Dimension D of the belief latent.
            num_actions: |A_i| (ego actions, one-hot).
            num_opponent_actions: |A_j| (opponent actions, one-hot).
            hidden_dim: Width of the prior, posterior and decoder networks.
            num_blocks: SwiGLU residual blocks per network.
            num_categoricals: Number of categorical variables in z.
            num_classes: Classes per categorical variable.
            kl_balance: Weight alpha on KL(sg(q) || p) (trains the prior); 1 - alpha on KL(q || sg(p)).
        """
        super().__init__()
        self.num_categoricals = num_categoricals
        self.num_classes = num_classes
        self.kl_balance = kl_balance
        action_width = num_actions + num_opponent_actions
        z_width = num_categoricals * num_classes
        self.prior_net = build_residual_stack(latent_dim + action_width, hidden_dim, z_width, num_blocks)
        self.posterior_net = build_residual_stack(2 * latent_dim + action_width, hidden_dim, z_width, num_blocks)
        self.decoder = build_residual_stack(latent_dim + action_width + z_width, hidden_dim, latent_dim, num_blocks)

    def _logits(self, net: nn.Module, inputs: Tensor) -> Tensor:
        return net(inputs).view(inputs.shape[0], self.num_categoricals, self.num_classes)

    def _sample(self, logits: Tensor) -> Tensor:
        """Straight-through Gumbel-softmax one-hot sample, flattened to (B, N_cat * N_class)."""
        logits = logits.float()
        uniform = torch.rand_like(logits).clamp(1e-7, 1.0 - 1e-7)
        noisy = logits - torch.log(-torch.log(uniform))
        hard = F.one_hot(noisy.argmax(dim=-1), self.num_classes).float()
        soft = noisy.softmax(dim=-1)
        return (hard - soft.detach() + soft).flatten(1)

    @staticmethod
    def _categorical_kl(p_logits: Tensor, q_logits: Tensor) -> Tensor:
        """KL(p || q) summed over categoricals, float32; shape (B,)."""
        p_log = F.log_softmax(p_logits.float(), dim=-1)
        q_log = F.log_softmax(q_logits.float(), dim=-1)
        return (p_log.exp() * (p_log - q_log)).sum(dim=(-1, -2))

    def imagine(self, latent: Tensor, action: Tensor, opponent_action: Tensor) -> Tensor:
        """Samples z from the prior and returns an imagined z_{t+1}, shape (B, D)."""
        conditioning = torch.cat([latent, action, opponent_action], dim=-1)
        z = self._sample(self._logits(self.prior_net, conditioning)).to(latent.dtype)
        return self.decoder(torch.cat([conditioning, z], dim=-1))

    def predict_train(
        self, latent: Tensor, action: Tensor, opponent_action: Tensor, target_next: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Posterior-sampled prediction of the target latent and the balanced KL.

        Returns:
            (predicted z_{t+1} of shape (B, D), balanced KL of shape (B,)).
        """
        conditioning = torch.cat([latent, action, opponent_action], dim=-1)
        prior_logits = self._logits(self.prior_net, conditioning)
        posterior_logits = self._logits(self.posterior_net, torch.cat([conditioning, target_next], dim=-1))
        z = self._sample(posterior_logits).to(latent.dtype)
        predicted = self.decoder(torch.cat([conditioning, z], dim=-1))
        kl = (self.kl_balance * self._categorical_kl(posterior_logits.detach(), prior_logits)
              + (1.0 - self.kl_balance) * self._categorical_kl(posterior_logits, prior_logits.detach()))
        return predicted, kl


class RecurrentJEPA(nn.Module):
    """Online belief filter, its EMA target copy, and the stochastic latent transition."""

    def __init__(self, belief_filter: BeliefFilter, transition: LatentTransition, ema_momentum: float):
        """
        Args:
            belief_filter: Online filter trained by gradient descent.
            transition: Stochastic latent transition.
            ema_momentum: m in target <- m * target + (1 - m) * online.
        """
        super().__init__()
        self.belief_filter = belief_filter
        self.transition = transition
        self.ema_momentum = ema_momentum
        self.target_filter = copy.deepcopy(belief_filter).requires_grad_(False)

    @torch.no_grad()
    def update_target(self) -> None:
        """EMA update of the target filter (in place, after each optimiser step)."""
        for target, online in zip(self.target_filter.parameters(), self.belief_filter.parameters()):
            target.mul_(self.ema_momentum).add_(online, alpha=1.0 - self.ema_momentum)

    def predict_next_belief(self, latent: Tensor, action: Tensor, opponent_action: Tensor) -> Tensor:
        """Prior imagination step used by MCTS: (B, D) -> (B, D)."""
        return self.transition.imagine(latent, action, opponent_action)
