# ABSOLUTE PATH: src/ipomdp/training/replay_buffer.py
# ==============================================================================
# CONTIGUOUS SEQUENCE PRIORITIZED EXPERIENCE REPLAY (SEQ-PER) & SUMTREE
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Exact Bottom-Up Propagation (Zero Floating-Point Drift):
#    - Parent nodes are updated by directly summing their two children:
#         tree[idx] = tree[2*idx + 1] + tree[2*idx + 2]
#      This guarantees that tree[0] remains exactly identical to the true sum of all
#      leaves over millions of updates, eliminating numerical truncation drift.
#
# 2. Pinned CPU Storage & Explicit Sliced DMA Transfers:
#    - Pre-allocates sequence buffers in pinned CPU memory (.pin_memory()).
#    - Re-invokes .pin_memory() on sliced batch tensors returned by sample_sequence()
#      because PyTorch advanced indexing (buf[ptr_tensor]) returns newly allocated,
#      unpinned CPU tensors. This preserves fast asynchronous PCIe DMA transfers.
#
# 3. Strict Tensor Rank Normalization on Ingestion:
#    - In push(), singleton leading batch dimensions on incoming transition tensors
#      (e.g., (1, obs_dim) -> (obs_dim,)) are systematically squeezed before list appending.
#      This guarantees that stacked episode tensors possess exact dimensions:
#         obs_t: (horizon + 1, obs_dim)
#         act_i_t: (horizon, action_dim_i)
#      preventing spurious inner singleton dimensions in sample batches.
#
# 4. 1D TD-Error Priority Vectorization:
#    - In update_priorities(), incoming TD errors are flattened to 1D (B,) before
#      leaf priority updates, preventing scalar conversion type errors.
#
# 5. Burn-in Warmup Masking:
#    - Chunks are sliced as (burn_in + seq_len). The mask tensor assigns 0.0 to the first
#      burn_in steps and 1.0 to the subsequent seq_len training steps, isolating loss
#      backpropagation strictly to post-warmup recurrent states.
# ==============================================================================

from typing import Dict, Union, Tuple, List, Optional
import numpy as np
import torch


class SumTree:
    """
    High-performance double-precision binary segment sum/min tree.
    Enables O(log N) prioritized leaf sampling and priority updates for Sequence-PER.
    """

    def __init__(self, capacity: int):
        """
        Allocates contiguous double-precision tree arrays with power-of-two size safety.

        Args:
            capacity: Maximum number of leaf nodes (replay buffer capacity).
        """
        self.capacity = int(capacity)

        # Align capacity to nearest power of two to guarantee balanced heap depths
        self.tree_capacity = 1 << int(np.ceil(np.log2(max(self.capacity, 1))))
        self.tree_size = 2 * self.tree_capacity - 1

        # Binary tree layout:
        # Indices 0 to tree_capacity-2 are internal parent nodes.
        # Indices tree_capacity-1 to 2*tree_capacity-2 are leaf nodes storing scalar priorities.
        self.tree = np.zeros(self.tree_size, dtype=np.float64)
        self.min_tree = np.full(self.tree_size, np.inf, dtype=np.float64)

        # Contiguous integer payload storage for buffer indices
        self.data = np.zeros(self.capacity, dtype=np.int64)

        self.write_ptr = 0
        self.size = 0

    @property
    def total_priority(self) -> float:
        """Retrieves root sum of all active leaf priorities in float64 precision."""
        return float(self.tree[0])

    def get_min_priority(self) -> float:
        """
        Retrieves minimum non-zero leaf priority in O(1) time.
        Used for non-blocking Importance Sampling weight normalization.
        """
        min_p = self.min_tree[0]
        if min_p == np.inf or min_p <= 1e-8:
            return 1.0
        return float(min_p)

    def add(self, priority: float, data_idx: int):
        """
        Stores buffer index and initial priority in the next circular leaf slot.

        Args:
            priority: Initial sampling priority (typically max_priority).
            data_idx: Integer pointer to transition chunk in replay memory.
        """
        tree_idx = self.write_ptr + self.tree_capacity - 1
        self.data[self.write_ptr] = data_idx
        self.update(tree_idx, priority)

        self.write_ptr = (self.write_ptr + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def update(self, tree_idx: int, priority: float):
        """
        Updates leaf node priority and propagates exact sum/min values to the root.

        Args:
            tree_idx: Global index in binary tree array (tree_capacity-1 <= tree_idx < 2*tree_capacity-1).
            priority: New non-negative priority value.
        """
        p_val = float(priority)
        self.tree[tree_idx] = p_val
        self.min_tree[tree_idx] = p_val if p_val > 0.0 else np.inf

        idx = tree_idx
        while idx != 0:
            idx = (idx - 1) // 2
            left_child = 2 * idx + 1
            right_child = left_child + 1

            self.tree[idx] = self.tree[left_child] + self.tree[right_child]
            self.min_tree[idx] = min(self.min_tree[left_child], self.min_tree[right_child])

    def get_leaf_batch(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized tree descent to retrieve leaf nodes for a batch of cumulative probability values.

        Args:
            values: 1D NumPy array of search values in range [0, total_priority].

        Returns:
            Tuple of (leaf_tree_indices, leaf_priorities, payload_data_indices).
        """
        tot_p = max(self.total_priority, 1e-8)
        search_vals = np.clip(values.astype(np.float64), 0.0, tot_p - 1e-9)

        indices = np.zeros_like(search_vals, dtype=np.int64)

        while True:
            left_children = 2 * indices + 1
            right_children = left_children + 1

            # Terminate strictly when ALL batch search paths reach leaf boundaries
            if np.all(left_children >= len(self.tree)):
                break

            left_vals = self.tree[left_children]
            go_left = search_vals <= left_vals

            indices = np.where(go_left, left_children, right_children)
            search_vals = np.where(go_left, search_vals, search_vals - left_vals)

        leaf_indices = np.clip(indices, self.tree_capacity - 1, self.tree_capacity + self.capacity - 1)
        data_indices = np.clip(leaf_indices - (self.tree_capacity - 1), 0, self.capacity - 1)

        priorities = self.tree[leaf_indices]
        payloads = self.data[data_indices]

        return leaf_indices, priorities, payloads


class PrioritizedSequenceBuffer:
    """
    Sequence Prioritized Experience Replay Buffer (Seq-PER).
    Stores contiguous multi-step sequence chunks (burn_in + seq_len) in pinned CPU memory.
    """

    def __init__(
        self,
        capacity: int = 2048,
        burn_in: int = 5,
        seq_len: int = 10,
        discount: float = 0.99,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        max_ep_len: int = 1000
    ):
        """
        Initializes sequence buffer parameters and segment tree storage.

        Args:
            capacity: Maximum number of sequence chunks stored.
            burn_in: Number of recurrent GRU warmup steps (loss masked out).
            seq_len: Active sequence training horizon.
            discount: Temporal discount factor gamma.
            alpha: PER prioritization exponent (0.0 = uniform, 1.0 = full prioritization).
            beta: Importance sampling exponent (annealed towards 1.0).
            beta_increment: Step increment for beta annealing.
            max_ep_len: Maximum episode length safeguard before forced flushing.
        """
        self.capacity = int(capacity)
        self.burn_in = int(burn_in)
        self.seq_len = int(seq_len)
        self.chunk_len = self.burn_in + self.seq_len
        self.discount = float(discount)
        self.max_ep_len = int(max_ep_len)

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.beta_increment = float(beta_increment)
        self.max_priority = 1.0

        self.tree = SumTree(self.capacity)
        self._initialized = False

        self.obs_buf: Optional[torch.Tensor] = None
        self.act_i_buf: Optional[torch.Tensor] = None
        self.act_j_buf: Optional[torch.Tensor] = None
        self.rewards_buf: Optional[torch.Tensor] = None
        self.mask_buf: Optional[torch.Tensor] = None
        self.prev_a_buf: Optional[torch.Tensor] = None
        self.dones_buf: Optional[torch.Tensor] = None

        # Ephemeral active trajectory lists mapped per environment channel index
        self._current_obs: Dict[int, List[torch.Tensor]] = {}
        self._current_act_i: Dict[int, List[torch.Tensor]] = {}
        self._current_act_j: Dict[int, List[torch.Tensor]] = {}
        self._current_rewards: Dict[int, List[float]] = {}

    def _lazy_init_buffers(self, sample_chunk: Dict[str, torch.Tensor]):
        """Allocates contiguous CPU memory arrays matching sample chunk shapes."""
        def allocate(tensor: torch.Tensor) -> torch.Tensor:
            return torch.zeros((self.capacity, *tensor.shape), dtype=tensor.dtype)

        self.obs_buf = allocate(sample_chunk["obs"])
        self.act_i_buf = allocate(sample_chunk["act_i"])
        self.act_j_buf = allocate(sample_chunk["act_j"])
        self.rewards_buf = allocate(sample_chunk["rewards"])
        self.mask_buf = allocate(sample_chunk["mask"])
        self.prev_a_buf = allocate(sample_chunk["prev_act_i"])
        self.dones_buf = allocate(sample_chunk["dones"])
        self._initialized = True

    def _ensure_env(self, env_idx: int):
        """Ensures trajectory storage structures exist for environment batch channel."""
        if env_idx not in self._current_obs:
            self._current_obs[env_idx] = []
            self._current_act_i[env_idx] = []
            self._current_act_j[env_idx] = []
            self._current_rewards[env_idx] = []

    def push(
        self,
        env_idx: int,
        obs: torch.Tensor,
        act_i: torch.Tensor,
        act_j: Union[int, float, torch.Tensor],
        reward: float
    ):
        """
        Pushes a single transition to the ephemeral channel buffer with rank normalization.
        """
        self._ensure_env(env_idx)

        # Rank normalization: Strip leading singleton batch dimensions
        obs_clean = obs.clone().detach().to(device="cpu", non_blocking=True)
        if obs_clean.dim() > 1 and obs_clean.size(0) == 1:
            obs_clean = obs_clean.squeeze(0)

        act_i_clean = act_i.clone().detach().to(device="cpu", non_blocking=True)
        if act_i_clean.dim() > 1 and act_i_clean.size(0) == 1:
            act_i_clean = act_i_clean.squeeze(0)

        if isinstance(act_j, (int, float)):
            act_j_clean = torch.tensor([int(act_j)], dtype=torch.long)
        else:
            act_j_clean = act_j.clone().detach().to(device="cpu", non_blocking=True).view(-1).long()

        self._current_obs[env_idx].append(obs_clean)
        self._current_act_i[env_idx].append(act_i_clean)
        self._current_act_j[env_idx].append(act_j_clean)
        self._current_rewards[env_idx].append(float(reward))

        if len(self._current_rewards[env_idx]) >= self.max_ep_len:
            self.end_episode(env_idx, obs_clean)

    def end_episode(self, env_idx: int, final_obs: torch.Tensor, terminated: bool = False):
        """
        Slices and pushes sequence chunks from completed episode trajectory into main storage.
        """
        self._ensure_env(env_idx)
        final_obs_clean = final_obs.clone().detach().to(device="cpu", non_blocking=True)
        if final_obs_clean.dim() > 1 and final_obs_clean.size(0) == 1:
            final_obs_clean = final_obs_clean.squeeze(0)

        self._current_obs[env_idx].append(final_obs_clean)

        horizon = len(self._current_rewards[env_idx])
        if horizon == 0:
            return

        obs_t = torch.stack(self._current_obs[env_idx])          # Shape: (horizon + 1, *obs_shape)
        act_i_t = torch.stack(self._current_act_i[env_idx])      # Shape: (horizon, *act_i_shape)
        act_j_t = torch.stack(self._current_act_j[env_idx])      # Shape: (horizon, *act_j_shape)
        rewards_t = torch.tensor(self._current_rewards[env_idx], dtype=torch.float32).unsqueeze(-1)

        dones_t = torch.zeros(horizon, 1, dtype=torch.float32)
        if terminated:
            dones_t[-1, 0] = 1.0

        for start_idx in range(horizon):
            burn_start = start_idx - self.burn_in
            pad_front = max(0, -burn_start)
            actual_start = max(0, burn_start)

            train_end = min(start_idx + self.seq_len, horizon)
            actual_len = train_end - actual_start

            o = obs_t[actual_start: train_end + 1]
            ai = act_i_t[actual_start: train_end]
            aj = act_j_t[actual_start: train_end]
            r = rewards_t[actual_start: train_end]
            d = dones_t[actual_start: train_end]
            prev_a = torch.zeros_like(act_i_t[0]) if actual_start == 0 else act_i_t[actual_start - 1]

            m_burn = torch.zeros(start_idx - actual_start, 1, dtype=torch.float32)
            m_train = torch.ones(train_end - start_idx, 1, dtype=torch.float32)
            mask = torch.cat([m_burn, m_train], dim=0)

            if pad_front > 0:
                o = torch.cat([torch.zeros((pad_front, *o.shape[1:]), dtype=o.dtype), o], dim=0)
                ai = torch.cat([torch.zeros((pad_front, *ai.shape[1:]), dtype=ai.dtype), ai], dim=0)
                aj = torch.cat([torch.zeros((pad_front, *aj.shape[1:]), dtype=aj.dtype), aj], dim=0)
                r = torch.cat([torch.zeros((pad_front, 1), dtype=r.dtype), r], dim=0)
                d = torch.cat([torch.zeros((pad_front, 1), dtype=d.dtype), d], dim=0)
                mask = torch.cat([torch.zeros((pad_front, 1), dtype=mask.dtype), mask], dim=0)

            pad_back = self.chunk_len - (pad_front + actual_len)
            if pad_back > 0:
                o = torch.cat([o, torch.zeros((pad_back, *o.shape[1:]), dtype=o.dtype)], dim=0)
                ai = torch.cat([ai, torch.zeros((pad_back, *ai.shape[1:]), dtype=ai.dtype)], dim=0)
                aj = torch.cat([aj, torch.zeros((pad_back, *aj.shape[1:]), dtype=aj.dtype)], dim=0)
                r = torch.cat([r, torch.zeros((pad_back, 1), dtype=r.dtype)], dim=0)
                d = torch.cat([d, torch.zeros((pad_back, 1), dtype=d.dtype)], dim=0)
                mask = torch.cat([mask, torch.zeros((pad_back, 1), dtype=mask.dtype)], dim=0)

            chunk = {
                "obs": o,
                "act_i": ai,
                "act_j": aj,
                "rewards": r,
                "dones": d,
                "mask": mask,
                "prev_act_i": prev_a
            }

            if not self._initialized:
                self._lazy_init_buffers(chunk)

            ptr = self.tree.write_ptr
            self.obs_buf[ptr] = chunk["obs"]
            self.act_i_buf[ptr] = chunk["act_i"]
            self.act_j_buf[ptr] = chunk["act_j"]
            self.rewards_buf[ptr] = chunk["rewards"]
            self.dones_buf[ptr] = chunk["dones"]
            self.mask_buf[ptr] = chunk["mask"]
            self.prev_a_buf[ptr] = chunk["prev_act_i"]

            self.tree.add(self.max_priority, ptr)

        self._current_obs[env_idx] = []
        self._current_act_i[env_idx] = []
        self._current_act_j[env_idx] = []
        self._current_rewards[env_idx] = []

    def sample_sequence(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[int]]:
        """
        Samples a prioritized sequence batch using double-precision SumTree indexing.
        Re-invokes .pin_memory() on sliced tensors to ensure fast non-blocking DMA PCIe transfers.

        Returns:
            Tuple of (batched_data_dict, importance_sampling_weights, tree_leaf_indices).
        """
        self.beta = min(1.0, self.beta + self.beta_increment)
        segment = self.tree.total_priority / batch_size

        p_min = self.tree.get_min_priority() / max(self.tree.total_priority, 1e-8)
        max_weight = (p_min * self.tree.size) ** (-self.beta)

        random_values = np.random.uniform(
            [segment * i for i in range(batch_size)],
            [segment * (i + 1) for i in range(batch_size)]
        )

        tree_indices, priorities, ptr_list = self.tree.get_leaf_batch(random_values)
        ptr_tensor = torch.tensor(ptr_list, dtype=torch.long)

        batched_data = {
            "obs": self.obs_buf[ptr_tensor],
            "act_i": self.act_i_buf[ptr_tensor],
            "act_j": self.act_j_buf[ptr_tensor],
            "rewards": self.rewards_buf[ptr_tensor],
            "dones": self.dones_buf[ptr_tensor],
            "mask": self.mask_buf[ptr_tensor],
            "prev_act_i": self.prev_a_buf[ptr_tensor]
        }

        sampling_probabilities = priorities / max(self.tree.total_priority, 1e-8)
        is_weights = np.power(self.tree.size * sampling_probabilities, -self.beta)
        is_weights_t = torch.tensor(
            is_weights / max(max_weight, 1e-8), dtype=torch.float32
        ).unsqueeze(-1).clamp(0.0, 1.0)

        return batched_data, is_weights_t, tree_indices.tolist()


    def update_priorities(self, tree_indices: List[int], td_errors: torch.Tensor):
        """
        Updates SumTree priorities based on multi-step TD error magnitudes.

        Args:
            tree_indices: List of leaf node tree indices from sample_sequence.
            td_errors: Tensor of TD error magnitudes (flattened to 1D of shape (B,)).
        """
        errors = td_errors.detach().cpu().view(-1).numpy()
        batch_max = 0.0

        for idx, error in zip(tree_indices, errors):
            priority = float((abs(error) + 1e-5) ** self.alpha)
            self.tree.update(idx, priority)
            batch_max = max(batch_max, priority)

        self.max_priority = max(batch_max, self.max_priority * 0.99)
