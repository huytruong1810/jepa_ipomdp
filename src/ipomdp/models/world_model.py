# ABSOLUTE PATH: src/ipomdp/models/world_model.py
# ==============================================================================
# RECURRENT JEPA WORLD MODEL, CONTEXT FILTER & CAUSAL PREDICTOR
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Recurrent Context Encoder (Belief Filter):
#    - Implements the learned Bayesian belief filter b_t = Filter(b_{t-1}, a_{t-1}, o_t)
#      in R^(B x N_obj x D_latent).
#    - Injects learnable slot positional encodings before inter-slot self-attention /
#      spatial cross-attention to guarantee semantic slot identity persistence.
#    - Fuses broadcasted past action with extracted object tokens and advances state
#      via nn.GRUCell.
#
# 2. Causal Relational Dynamics Predictor & bfloat16 KL Stability:
#    - Predicts open-loop stochastic transitions in latent space:
#         b_{t+1} ~ p_phi(b_{t+1} | b_t, a_i, a_j, z_t)
#    - Parameterizes stochasticity using N_cat x N_class discrete categorical latents z_t
#      sampled via Straight-Through Gumbel-Softmax with symmetric probability clamping.
#    - Employs DreamerV3 KL Balancing (alpha = 0.8) to train the prior p_phi to track the
#      posterior q_psi while regularizing posterior drift.
#    - Promotes probability distributions to float32 before logarithmic evaluation in
#      _kl_divergence to prevent mantissa underflow and NaN values under bfloat16 AMP.
#    - Invokes .contiguous() on expanded query mask tokens to ensure memory stride
#      safety in compiled Transformer decoders.
#
# 3. Target EMA Encoder Synchronization:
#    - RecurrentJEPABase manages the online context filter and an Exponential Moving Average
#      (EMA) target encoder. Updates execute in-place under @torch.no_grad() for maximum
#      GPU memory efficiency and PyTorch compile safety.
# ==============================================================================

import copy
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .extractors import FeatureExtractor
from .layers import SwarmActionEncoder, build_residual_stack, RMSNorm
from ..types import Action


class RecurrentContextEncoder(nn.Module):
    """
    Recurrent Context Filter for Interactive POMDP Belief Tracking:
    b_t = Filter(b_{t-1}, a_{t-1}, o_t) in R^(B x N_obj x D_latent).
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        action_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 2
    ):
        """
        Initializes Recurrent Context Encoder.

        Args:
            feature_extractor: Feature extractor module producing object tokens.
            action_dim: Dimensionality of action vector.
            latent_dim: Target latent representation dimension per object slot.
            hidden_dim: Intermediate feature dimension.
            num_blocks: Number of SwiGLU residual blocks in fusion stack.
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.num_objects = feature_extractor.num_objects
        self.is_permutation_invariant = feature_extractor.is_permutation_invariant
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)

        self.feature_proj = nn.Linear(feature_extractor.output_dim, latent_dim)

        # Slot positional embeddings to preserve slot identities across time
        self.slot_pos_embed = nn.Parameter(torch.randn(1, self.num_objects, latent_dim))
        nn.init.normal_(self.slot_pos_embed, std=0.02)

        if self.is_permutation_invariant:
            self.object_tracker = nn.TransformerDecoderLayer(
                d_model=latent_dim, nhead=4, dim_feedforward=hidden_dim * 2,
                activation="gelu", batch_first=True, norm_first=True
            )
        else:
            self.inter_slot_attn = nn.MultiheadAttention(
                embed_dim=latent_dim, num_heads=4, batch_first=True
            )
            self.slot_norm = RMSNorm(latent_dim)

        self.fusion_layer = build_residual_stack(latent_dim + self.action_dim, hidden_dim, hidden_dim, num_blocks)
        self.gru_cell = nn.GRUCell(input_size=hidden_dim, hidden_size=latent_dim)

    def forward(self, obs: torch.Tensor, prev_action: torch.Tensor, prev_belief: torch.Tensor) -> torch.Tensor:
        """
        Updates recurrent belief state tensor given new observation and previous action.

        Args:
            obs: Raw observation tensor of shape (B, *obs_shape).
            prev_action: Action tensor taken at step t-1 of shape (B, action_dim) or (B, 1) integer index.
            prev_belief: Prior recurrent belief state tensor of shape (B, N_obj, D_latent).

        Returns:
            Updated belief state tensor of shape (B, N_obj, D_latent).
        """
        b = obs.size(0)

        # Ensure prev_action is a one-hot float tensor of shape (B, action_dim)
        prev_action_onehot = Action.to_one_hot(prev_action, self.action_dim, device=obs.device)
        if prev_action_onehot.size(0) != b:
            prev_action_onehot = prev_action_onehot.expand(b, -1)


        obs_objects = self.feature_extractor(obs)
        projected_objects = self.feature_proj(obs_objects)

        pos_embed = self.slot_pos_embed.to(dtype=projected_objects.dtype)
        projected_objects_pos = projected_objects + pos_embed
        prev_belief_pos = prev_belief + pos_embed

        if self.is_permutation_invariant:
            tracked_objects = self.object_tracker(tgt=prev_belief_pos, memory=projected_objects_pos)
        else:
            attn_out, _ = self.inter_slot_attn(
                query=projected_objects_pos,
                key=projected_objects_pos,
                value=projected_objects
            )
            tracked_objects = self.slot_norm(projected_objects + attn_out)

        # Broadcast action vector across all object slots
        prev_a_broadcast = prev_action_onehot.unsqueeze(1).expand(-1, self.num_objects, -1)
        x = torch.cat([tracked_objects, prev_a_broadcast], dim=-1)

        fused = self.fusion_layer(x)
        fused_flat = fused.reshape(b * self.num_objects, -1)
        prev_belief_flat = prev_belief.reshape(b * self.num_objects, -1)

        new_belief_flat = self.gru_cell(fused_flat, prev_belief_flat)
        return new_belief_flat.reshape(b, self.num_objects, -1)


class CausalRelationalPredictor(nn.Module):
    """
    Latent Transition World Model: b_{t+1} ~ p_phi(b_{t+1} | b_t, a_i, a_j, z_t).
    Uses Transformer Prior/Posterior towers with Straight-Through Gumbel-Softmax categoricals.
    """

    def __init__(
        self,
        num_objects: int,
        latent_dim: int,
        action_dim_i: int,
        action_dim_j: int,
        hidden_dim: int = 128,
        num_categoricals: int = 4,
        num_classes: int = 4,
        num_blocks: int = 2
    ):
        """
        Initializes Causal Relational Dynamics Predictor.

        Args:
            num_objects: Number of structured object slots.
            latent_dim: Dimension of latent belief state per slot.
            action_dim_i: Dimensionality of ego action vector.
            action_dim_j: Dimensionality of opponent action vector.
            hidden_dim: Transformer feedforward and projection dimension.
            num_categoricals: Number of discrete latent categorical variables (default: 4).
            num_classes: Classes per discrete categorical variable (default: 4).
            num_blocks: Number of Transformer encoder/decoder layers.
        """
        super().__init__()
        self.num_objects = int(num_objects)
        self.latent_dim = int(latent_dim)
        self.num_categoricals = int(num_categoricals)
        self.num_classes = int(num_classes)
        self.tau = 1.0
        z_dim = self.num_categoricals * self.num_classes

        self.swarm_encoder = SwarmActionEncoder(action_dim_i, action_dim_j, hidden_dim=hidden_dim, num_heads=4)
        self.action_proj = nn.Linear(hidden_dim, latent_dim)

        self.mask_token = nn.Parameter(torch.randn(1, 1, latent_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.slot_pos_embed = nn.Parameter(torch.randn(1, num_objects, latent_dim))
        nn.init.normal_(self.slot_pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=4, dim_feedforward=hidden_dim * 2,
            activation="gelu", batch_first=True, norm_first=True
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim, nhead=4, dim_feedforward=hidden_dim * 2,
            activation="gelu", batch_first=True, norm_first=True
        )

        self.prior_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks, enable_nested_tensor=False)
        self.post_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks, enable_nested_tensor=False)
        self.dynamics_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_blocks)

        self.prior_proj = nn.Linear(latent_dim, z_dim)
        self.post_proj = nn.Linear(latent_dim, z_dim)
        self.z_fusion = nn.Linear(latent_dim + z_dim, latent_dim)

    def sample_z_categorical(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Straight-Through Gumbel-Softmax Sampling.
        Executes in float32 with symmetric probability clamping to prevent tail skew.

        Args:
            logits: Unnormalized discrete categorical logits of shape (B, N, num_cat, num_class).

        Returns:
            Continuous straight-through one-hot samples of shape (B, N, num_cat * num_class).
        """
        orig_dtype = logits.dtype
        logits_f32 = logits.float()

        uniform = torch.rand_like(logits_f32)
        u_safe = torch.clamp(uniform, 1e-7, 1.0 - 1e-7)
        gumbel = -torch.log(-torch.log(u_safe))

        noisy_logits = (logits_f32 + gumbel) / self.tau

        hard_sample = F.one_hot(
            torch.argmax(noisy_logits, dim=-1), num_classes=self.num_classes
        ).float()

        soft_sample = F.softmax(noisy_logits, dim=-1)
        z_f32 = hard_sample.detach() - soft_sample.detach() + soft_sample

        z = z_f32.to(dtype=orig_dtype)
        return z.view(z.shape[0], z.shape[1], -1)

    def _kl_divergence(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Computes analytical KL divergence KL(p || q) across categorical distributions.
        Promotes distributions to float32 to prevent bfloat16 mantissa underflow.
        """
        orig_dtype = p.dtype
        p_f32 = p.float()
        q_f32 = q.float()

        p_safe = torch.clamp(p_f32, 1e-7, 1.0)
        q_safe = torch.clamp(q_f32, 1e-7, 1.0)

        kl = torch.sum(p_safe * (torch.log(p_safe) - torch.log(q_safe)), dim=-1)
        return kl.to(dtype=orig_dtype)

    def forward_train(
        self,
        belief: torch.Tensor,
        ego_a: torch.Tensor,
        opp_a: torch.Tensor,
        target_next_belief: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Trains dynamics predictor using Posterior and Prior Transformer towers with KL Balancing.

        Args:
            belief: Current belief state tensor of shape (B, N_obj, D_latent).
            ego_a: Ego action tensor of shape (B, action_dim_i).
            opp_a: Opponent action tensor of shape (B, M, action_dim_j) or (B, action_dim_j).
            target_next_belief: Target EMA next belief state tensor of shape (B, N_obj, D_latent).

        Returns:
            Tuple of (predicted_next_belief, kl_loss).
        """
        b, n, _ = belief.shape
        joint_features = self.swarm_encoder(ego_a, opp_a)
        action_token = self.action_proj(joint_features).unsqueeze(1)

        pos_embed = self.slot_pos_embed.to(dtype=belief.dtype)
        belief_pos = belief + pos_embed
        target_next_pos = target_next_belief + pos_embed

        # Prior Tower: Conditioned strictly on (action, b_t)
        prior_in = torch.cat([action_token, belief_pos], dim=1)
        prior_out = self.prior_transformer(prior_in)[:, 1:, :]
        prior_logits = self.prior_proj(prior_out).view(b, n, self.num_categoricals, self.num_classes)

        # Posterior Tower: Conditioned on (action, b_{t+1}^target)
        post_in = torch.cat([action_token, target_next_pos], dim=1)
        post_out = self.post_transformer(post_in)[:, 1:, :]
        post_logits = self.post_proj(post_out).view(b, n, self.num_categoricals, self.num_classes)

        z = self.sample_z_categorical(post_logits)

        dyn_context = self.z_fusion(torch.cat([belief, z], dim=-1))
        memory = torch.cat([action_token, dyn_context], dim=1)
        queries = (self.mask_token.expand(b, n, -1) + belief_pos).contiguous()

        next_belief = self.dynamics_decoder(tgt=queries, memory=memory)

        # DreamerV3 KL Balancing (alpha = 0.8)
        post_probs = F.softmax(post_logits, dim=-1)
        prior_probs = F.softmax(prior_logits, dim=-1)
        alpha = 0.8

        kl_prior_moves = self._kl_divergence(post_probs.detach(), prior_probs)
        kl_post_moves = self._kl_divergence(post_probs, prior_probs.detach())
        kl_loss = (alpha * kl_prior_moves + (1.0 - alpha) * kl_post_moves).sum(dim=2).mean(dim=1).unsqueeze(-1)

        return next_belief, kl_loss

    def forward(self, belief: torch.Tensor, ego_action: torch.Tensor, opp_actions: torch.Tensor) -> torch.Tensor:
        """
        Executes open-loop prior prediction step without target observations (used in MCTS search).

        Args:
            belief: Current belief state tensor of shape (B, N_obj, D_latent).
            ego_action: Ego action tensor of shape (B, action_dim_i).
            opp_actions: Opponent action tensor of shape (B, M, action_dim_j) or (B, action_dim_j).

        Returns:
            Imagined next belief state tensor of shape (B, N_obj, D_latent).
        """
        b, n, _ = belief.shape
        joint_features = self.swarm_encoder(ego_action, opp_actions)
        action_token = self.action_proj(joint_features).unsqueeze(1)

        pos_embed = self.slot_pos_embed.to(dtype=belief.dtype)
        belief_pos = belief + pos_embed

        prior_in = torch.cat([action_token, belief_pos], dim=1)
        prior_out = self.prior_transformer(prior_in)[:, 1:, :]
        prior_logits = self.prior_proj(prior_out).view(b, n, self.num_categoricals, self.num_classes)

        z = self.sample_z_categorical(prior_logits)

        dyn_context = self.z_fusion(torch.cat([belief, z], dim=-1))
        memory = torch.cat([action_token, dyn_context], dim=1)
        queries = (self.mask_token.expand(b, n, -1) + belief_pos).contiguous()

        return self.dynamics_decoder(tgt=queries, memory=memory)


class RecurrentJEPABase(nn.Module):
    """
    Top-Level Joint-Embedding Predictive Architecture Container.
    Manages online context encoder filter, dynamics predictor, and target EMA encoder.
    """

    def __init__(
        self,
        encoder: RecurrentContextEncoder,
        predictor: CausalRelationalPredictor,
        ema_momentum: float = 0.99
    ):
        """
        Initializes Recurrent JEPA world model container.

        Args:
            encoder: Online recurrent context encoder filter.
            predictor: Causal relational dynamics predictor.
            ema_momentum: Exponential Moving Average momentum coefficient.
        """
        super().__init__()
        self.context_encoder = encoder
        self.predictor = predictor
        self.ema_momentum = float(ema_momentum)

        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self):
        """In-place EMA target encoder update for compile and memory safety."""
        for tgt, ctx in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            tgt.mul_(self.ema_momentum).add_(ctx, alpha=1.0 - self.ema_momentum)

    def encode_context(self, obs: torch.Tensor, prev_action: torch.Tensor, prev_belief: torch.Tensor) -> torch.Tensor:
        """Advances online recurrent belief filter: b_t = Filter(b_{t-1}, a_{t-1}, o_t)."""
        return self.context_encoder(obs, prev_action, prev_belief)

    @torch.no_grad()
    def encode_target(self, next_obs: torch.Tensor, action: torch.Tensor, belief: torch.Tensor) -> torch.Tensor:
        """Computes target belief state b_{t+1}^target using frozen EMA target encoder."""
        return self.target_encoder(next_obs, action, belief)

    def predict_next_belief(self, belief: torch.Tensor, ego_action: torch.Tensor, opp_actions: torch.Tensor) -> torch.Tensor:
        """Executes open-loop prior imagination step for MCTS rollouts."""
        return self.predictor(belief, ego_action, opp_actions)

    def predict_next_belief_train(
        self,
        belief: torch.Tensor,
        ego_action: torch.Tensor,
        opp_actions: torch.Tensor,
        target_next: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Trains dynamics predictor using Posterior and Prior Transformer towers."""
        return self.predictor.forward_train(belief, ego_action, opp_actions, target_next)
