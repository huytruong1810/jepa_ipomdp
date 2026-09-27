# ABSOLUTE PATH: src/ipomdp/models/world_model.py
# ==============================================================================
# RECURRENT JEPA WORLD MODEL: BELIEF FILTER, EMA TARGET, SELF-PREDICTION
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
#    - JEPA self-prediction: the predictor predicts the EMA target filter's next latent
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
# 5. Deterministic JEPA Predictor (representation learning only):
#    - g(z_t, a_t, a^j_t) predicts the EMA target latent z-bar_{t+1}; the loss is the mean
#      squared error. In a POMDP z-bar_{t+1} is random (it depends on o_{t+1}), so g learns
#      its conditional mean -- the "expected self-prediction" (EZP) objective of Ni et al.,
#      which together with reward grounding is what the Phase-2 study validated.
#    - The predictor is NOT used for planning. Phase-3 study (canonical Tiger, 1000 updates):
#      a DreamerV3-style stochastic latent transition imagined biased beliefs (post-LISTEN
#      mean 0.57 from b = 0.85, never moving towards TL), because (i) the EMA target lives in
#      a different latent space than the online latents the heads read (probe error 0.41 vs
#      0.03 on the same histories), (ii) predicting online latents instead collapses the
#      representation (probe error 0.13), and (iii) straight-through discrete latents learned
#      even the two-outcome growl distribution poorly. The discrete latent, prior/posterior,
#      KL balancing and unimix were therefore removed.
#
# 6. Planning Model = Observation Branching Through the Filter:
#    - MCTS imagines by sampling o' from the learned P(o' | z, a, a^j) (ObservationHead,
#      models/heads.py, trained on detached latents) and stepping the SAME filter,
#      z' = BeliefFilter.step(z, a, o'). Imagined latents are therefore exactly the latents
#      the filter would produce after that history, on the manifold the heads were trained
#      on, and branching mirrors the exact belief tree of the benchmark solver.
# ==============================================================================

import copy

import torch
import torch.nn as nn
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


class LatentPredictor(nn.Module):
    """Deterministic JEPA predictor g(z_t, a_t, a^j_t) of the EMA target latent z-bar_{t+1}."""

    def __init__(self, latent_dim: int, num_actions: int, num_opponent_actions: int, hidden_dim: int,
                 num_blocks: int):
        super().__init__()
        self.net = build_residual_stack(latent_dim + num_actions + num_opponent_actions, hidden_dim, latent_dim,
                                        num_blocks)

    def forward(self, latent: Tensor, action: Tensor, opponent_action: Tensor) -> Tensor:
        """(B, D), one-hot (B, |A_i|), one-hot (B, |A_j|) -> predicted z-bar_{t+1}, (B, D)."""
        return self.net(torch.cat([latent, action, opponent_action], dim=-1))


class RecurrentJEPA(nn.Module):
    """Online belief filter, its EMA target copy, and the JEPA self-prediction predictor."""

    def __init__(self, belief_filter: BeliefFilter, predictor: LatentPredictor, ema_momentum: float):
        """
        Args:
            belief_filter: Online filter trained by gradient descent.
            predictor: Deterministic predictor of the EMA target's next latent.
            ema_momentum: m in target <- m * target + (1 - m) * online.
        """
        super().__init__()
        self.belief_filter = belief_filter
        self.predictor = predictor
        self.ema_momentum = ema_momentum
        self.target_filter = copy.deepcopy(belief_filter).requires_grad_(False)

    @torch.no_grad()
    def update_target(self) -> None:
        """EMA update of the target filter (in place, after each optimiser step)."""
        for target, online in zip(self.target_filter.parameters(), self.belief_filter.parameters()):
            target.mul_(self.ema_momentum).add_(online, alpha=1.0 - self.ema_momentum)
