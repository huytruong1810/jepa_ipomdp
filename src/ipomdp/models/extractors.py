# ABSOLUTE PATH: src/ipomdp/models/extractors.py
# ==============================================================================
# PERCEPTUAL FEATURE EXTRACTORS & OBJECT SLOT ATTENTION MODULES
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Arbitrary Tensor Rank Reshaping in MLP Feature Extractor:
#    - In MLPFeatureExtractor.forward, features are reshaped dynamically as:
#         out.view(*obs.shape[:-1], self._num_objects, self._output_dim)
#      This supports single-step observation batches (B, obs_dim) as well as full
#      sequence batches (B, T, obs_dim) without rank-mismatch crashes.
#
# 2. Slot-Identity Role Persistence:
#    - For vector-state environments (e.g., Tiger), MLPs project flat observations into
#      N_obj slots. Injecting learnable self.slot_identity_embed (sigma = 0.02) guarantees
#      that Slot 0 and Slot 1 preserve semantic role distinctions when processed by
#      subsequent permutation-invariant attention poolers.
#
# 3. Competitive Softmax & Key Normalization in Slot Attention:
#    - In SlotAttention, slots compete for spatial keys via softmax over slots (dim=1):
#         attn = softmax(q @ k.T * scale, dim=1)
#      followed by unbiased key normalization across spatial locations (dim=2):
#         attn = attn / (attn.sum(dim=-1, keepdim=True) + eps)
#      Setting eps = 1e-6 guarantees numerical stability under bfloat16/float16 execution.
#
# 4. TorchDynamo Graph-Break Elimination in CNN Normalization:
#    - Replaced data-dependent obs.max() > 1.0 checks with compile-safe dtype inspection
#      (if not obs.is_floating_point() or obs.dtype == torch.uint8: obs = obs.float() / 255.0),
#      eliminating host-device synchronization and CUDA graph capture invalidation.
# ==============================================================================

from abc import ABC, abstractmethod
from typing import Tuple
import torch
import torch.nn as nn

from .layers import build_residual_stack, RMSNorm, LearnedPositionalEncoding2D
from ..telemetry.registry import register_extractor


class FeatureExtractor(nn.Module, ABC):
    """Abstract base class for perceptual feature extractors."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Latent dimension per extracted object token."""
        pass

    @property
    @abstractmethod
    def num_objects(self) -> int:
        """Number of structured object slots."""
        pass

    @property
    @abstractmethod
    def is_permutation_invariant(self) -> bool:
        """Whether extractor outputs are permutation-invariant visual slots."""
        pass

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Extracts structured object tokens from raw observations.

        Args:
            obs: Observation tensor of shape (..., *obs_shape).

        Returns:
            Object tokens tensor of shape (..., num_objects, output_dim).
        """
        pass


class SlotAttention(nn.Module):
    """
    Object Slot Attention Module (Locatello et al., 2020).
    Dynamically binds spatial feature maps into discrete competitive object slots.
    """

    def __init__(self, num_slots: int, dim: int, iters: int = 3, eps: float = 1e-6):
        """
        Initializes Slot Attention parameters.

        Args:
            num_slots: Number of competitive object slots.
            dim: Slot and key/query/value embedding dimension.
            iters: Number of competitive routing refinement iterations.
            eps: Denominator epsilon guard for key normalization (1e-6 for bfloat16 safety).
        """
        super().__init__()
        self.num_slots = int(num_slots)
        self.dim = int(dim)
        self.iters = int(iters)
        self.eps = float(eps)
        self.scale = self.dim ** -0.5

        # Learnable slot distribution initialization parameters
        self.slots_mu = nn.Parameter(torch.randn(1, 1, self.dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, self.dim))
        nn.init.uniform_(self.slots_logsigma, -0.5, 0.5)

        self.to_q = nn.Linear(self.dim, self.dim, bias=False)
        self.to_k = nn.Linear(self.dim, self.dim, bias=False)
        self.to_v = nn.Linear(self.dim, self.dim, bias=False)

        self.gru = nn.GRUCell(self.dim, self.dim)
        self.mlp = build_residual_stack(self.dim, self.dim * 2, self.dim, num_blocks=1)
        self.norm_input = RMSNorm(self.dim)
        self.norm_slots = RMSNorm(self.dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Routes spatial feature map tokens into discrete object slots.

        Args:
            inputs: Spatial key/value feature tokens of shape (B, N_keys, D).

        Returns:
            Extracted object slot tokens of shape (B, num_slots, D).
        """
        b, n, d = inputs.shape
        inputs_norm = self.norm_input(inputs)
        k = self.to_k(inputs_norm)
        v = self.to_v(inputs_norm)

        # Sample initial Gaussian slots per batch instance
        mu = self.slots_mu.expand(b, self.num_slots, -1)
        sigma = self.slots_logsigma.exp().expand(b, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)

        for _ in range(self.iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            q = self.to_q(slots_norm)

            # Dot-product attention: slots compete for spatial key locations
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale

            # Competition across slots (dim=1)
            attn = dots.softmax(dim=1)

            # Weighted normalization across keys (dim=2)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            # Recurrent GRU slot update
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            ).reshape(b, self.num_slots, d)

            slots = slots + self.mlp(slots)

        return slots


@register_extractor("mlp")
class MLPFeatureExtractor(FeatureExtractor):
    """
    Object slot extractor for flat vector observations (e.g., Multi-Agent Tiger).
    Injects learnable slot-identity encodings to preserve semantic role distinctions.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 128, num_objects: int = 2, num_blocks: int = 1):
        """
        Initializes MLP slot extractor.

        Args:
            obs_dim: Dimensionality of raw observation vector.
            hidden_dim: Latent feature dimension per object slot.
            num_objects: Number of distinct object slots.
            num_blocks: Number of SwiGLU residual blocks.
        """
        super().__init__()
        self._output_dim = int(hidden_dim)
        self._num_objects = int(num_objects)

        self.net = build_residual_stack(
            input_dim=obs_dim,
            hidden_dim=hidden_dim * 2,
            output_dim=num_objects * hidden_dim,
            num_blocks=num_blocks
        )

        # Slot-identity embeddings to break slot symmetry for vector inputs
        self.slot_identity_embed = nn.Parameter(torch.randn(1, num_objects, hidden_dim) * 0.02)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def num_objects(self) -> int:
        return self._num_objects

    @property
    def is_permutation_invariant(self) -> bool:
        return False

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Projects flat observation vectors into structured object slots across arbitrary batch ranks.

        Args:
            obs: Observation tensor of shape (..., obs_dim).

        Returns:
            Structured slot tokens of shape (..., num_objects, output_dim).
        """
        batch_shape = obs.shape[:-1]
        out = self.net(obs)
        slots = out.view(*batch_shape, self._num_objects, self._output_dim)

        pos_embed = self.slot_identity_embed.to(dtype=slots.dtype)
        return slots + pos_embed


@register_extractor("cnn")
class CNNFeatureExtractor(FeatureExtractor):
    """Extracts permutation-invariant visual object slots from spatial image frames."""

    def __init__(self, input_shape: Tuple[int, int, int], hidden_dim: int = 128, num_objects: int = 4):
        """
        Initializes visual CNN backbone and Slot Attention tracker.

        Args:
            input_shape: Tuple of (channels, height, width).
            hidden_dim: Output dimension per object slot.
            num_objects: Number of spatial visual slots.
        """
        super().__init__()
        self._output_dim = int(hidden_dim)
        self._num_objects = int(num_objects)

        if input_shape[1] < 64 or input_shape[2] < 64:
            self.convs = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(64, hidden_dim, kernel_size=3, stride=1, padding=1), nn.ReLU()
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                nn.Conv2d(64, hidden_dim, kernel_size=3, stride=1, padding=1), nn.ReLU()
            )

        dummy_in = torch.zeros(1, *input_shape)
        dummy_out = self.convs(dummy_in)
        _, _, h, w = dummy_out.shape

        self.pos_encoding = LearnedPositionalEncoding2D(hidden_dim, h, w)
        self.slot_attention = SlotAttention(num_slots=num_objects, dim=hidden_dim)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def num_objects(self) -> int:
        return self._num_objects

    @property
    def is_permutation_invariant(self) -> bool:
        return True

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Processes image frames through Conv2d backbone and Slot Attention.

        Args:
            obs: Image tensor of shape (B, C, H, W).

        Returns:
            Extracted visual object slots of shape (B, num_objects, hidden_dim).
        """
        # Compile-safe normalization: Avoid data-dependent .max() GPU checks
        if not obs.is_floating_point() or obs.dtype == torch.uint8:
            obs = obs.float() / 255.0

        features = self.convs(obs)
        features_with_pos = self.pos_encoding(features)

        return self.slot_attention(features_with_pos)
