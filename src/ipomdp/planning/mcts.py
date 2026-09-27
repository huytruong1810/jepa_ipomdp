# ABSOLUTE PATH: src/ipomdp/planning/mcts.py
# ==============================================================================
# LATENT BELIEF-TREE MONTE CARLO TREE SEARCH (MCTS) PLANNER
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Latent Belief-Tree Search by Observation Branching (Phase 3 decision):
#    - A child of latent z under action a is z' = BeliefFilter.step(z, a, o') for an
#      observation o' sampled from the learned P(o' | z, a, a^j) (models/heads.py,
#      ObservationHead). Imagined latents are thus exactly the filter's latents for that
#      history (models/world_model.py, section 6). num_observation_samples children are
#      drawn per action; Phase 4 reviews the search itself (exact branching over |O|,
#      selection, backup) against the exact solver.
#
# 2. Row-Aligned Opponent Action Batch Sampling (B >= 1 Safety):
#    - In _evaluate_and_expand_batched, uses repeat_interleave(num_branches, dim=0)
#      and multinomial sampling to strictly preserve row alignment matching rep_beliefs
#      for vectorized environment channels (B > 1), preventing cross-channel action leakage.
#
# 3. bfloat16-Safe Multinomial Sampling:
#    - Probabilities are explicitly promoted to float32 before calling torch.multinomial
#      (rep_opp_probs.float().view(-1, act_dim)), eliminating CUDA runtime kernel failures.
#
# 4. Bellman Step-Reward Integration & PUCT Action Selection:
#    - Evaluates expected branch Q-values with immediate transition step-rewards:
#         Q(s, a) = (1 / |C(s, a)|) * sum_{c in C(s, a)} [ r_c + gamma * V_c ]
#      where V_c = c.value if c.visit_count > 0 else c.bootstrap_value.
#    - Normalizes Q-values via MinMaxStats to prevent varying reward scales from
#      overpowering the exploration bonus:
#         score = Q_norm(s, a) + c_puct * P(a) * (sqrt(N(s)) / (1 + N(s, a)))
#
# 5. Static Memory Pool CUDA Graph Safety:
#    - Applies explicit .clone() calls on compiled model outputs inside
#      _evaluate_and_expand_batched to prevent static CUDA Graph address overwrites.
# ==============================================================================

import math
import random
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F

from ..interfaces import AbstractPlanner
from ..models.world_model import RecurrentJEPA
from ..models.heads import ObservationHead, OpponentPolicyHead, RewardHead, ValueHead
from ..models.distributions import TwoHotSymlog


class MinMaxStats:
    """
    Tracks and normalizes scalar value bounds across MCTS search paths.
    Maintains balanced PUCT exploration across dynamic return scales.
    """

    def __init__(self):
        self.maximum = -float('inf')
        self.minimum = float('inf')

    def update(self, value: float):
        """Updates minimum and maximum observed return bounds."""
        if value > self.maximum:
            self.maximum = float(value)
        if value < self.minimum:
            self.minimum = float(value)

    def normalize(self, value: float) -> float:
        """Normalizes an input scalar return into the range [0.0, 1.0]."""
        if self.maximum > self.minimum:
            return float((value - self.minimum) / (self.maximum - self.minimum))
        return 0.5


class LatentSearchNode:
    """
    Belief-latent node of the latent MCTS search tree.
    """

    def __init__(
        self,
        belief: torch.Tensor,
        action_taken: Optional[int] = None,
        reward: float = 0.0,
        prior: float = 1.0
    ):
        """
        Initializes latent search node.

        Args:
            belief: Single-sample belief latent of shape (1, D).
            action_taken: Discrete ego action leading to this node.
            reward: Immediate step reward r_t emitted during arrival transition.
            prior: Action selection prior probability P(s, a).
        """
        self.belief = belief
        self.action_taken = action_taken
        self.reward = float(reward)
        self.prior = float(prior)
        self.children: Dict[int, List['LatentSearchNode']] = {}
        self.visit_count = 0
        self.value = 0.0
        self.bootstrap_value = 0.0


class LatentBeliefTreeSearch(AbstractPlanner):
    """
    Monte Carlo Tree Search over learned belief latents, branching on sampled observations.
    """

    def __init__(
        self,
        world_model: RecurrentJEPA,
        value_head: ValueHead,
        reward_head: RewardHead,
        opponent_head: OpponentPolicyHead,
        observation_head: ObservationHead,
        codec: TwoHotSymlog,
        action_dim_i: int,
        action_dim_j: int,
        num_simulations: int = 50,
        num_observation_samples: int = 1,
        discount: float = 0.99,
        c_puct: float = 1.25,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25
    ):
        """
        Initializes Discrete Latent MCTS Planner.

        Args:
            world_model: Recurrent JEPA world model (its online belief filter steps imagined latents).
            value_head: Value distribution projection head.
            reward_head: Immediate reward distribution projection head.
            opponent_head: Opponent policy prediction head.
            observation_head: Planning model P(o' | z, a, a^j).
            codec: Two-hot codec shared with the trainer (decodes value/reward means).
            action_dim_i: Ego action dimensionality.
            action_dim_j: Opponent action dimensionality.
            num_simulations: Number of search rollouts per decision step.
            num_observation_samples: Observations sampled (children created) per action at each expansion.
            discount: Discount factor gamma.
            c_puct: PUCT exploration constant.
            dirichlet_alpha: Alpha parameter for root Dirichlet noise (default: 0.3).
            dirichlet_epsilon: Mixing weight for root Dirichlet noise (default: 0.25).
        """
        super().__init__()
        self.world_model = world_model
        self.observation_head = observation_head
        self.value_head = value_head
        self.reward_head = reward_head
        self.opponent_head = opponent_head
        self.action_dim_i = int(action_dim_i)
        self.action_dim_j = int(action_dim_j)
        self.num_simulations = int(num_simulations)
        self.num_observation_samples = int(num_observation_samples)
        self.discount = float(discount)
        self.c_puct = float(c_puct)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_epsilon = float(dirichlet_epsilon)

        self.device = next(self.value_head.parameters()).device
        self.twohot = codec

        # Multi-root references across parallel environment channels
        self.roots: List[LatentSearchNode] = []
        self.root: Optional[LatentSearchNode] = None

        # Search telemetry diagnostics
        self.last_avg_depth: float = 0.0
        self.last_max_depth: float = 0.0
        self.last_q_spread: float = 0.0
        self.last_entropy: float = 0.0

    @torch.no_grad()
    def search(self, root_state: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Executes parallel batched latent MCTS simulations across B environment channels.

        Args:
            root_state: Root belief latents of shape (B, D).
            temperature: Action selection sampling temperature.

        Returns:
            Batched policy action probability tensor of shape (B, action_dim_i).
        """
        b_batch = root_state.size(0)

        roots = [LatentSearchNode(belief=root_state[i:i + 1].detach()) for i in range(b_batch)]
        stats = [MinMaxStats() for _ in range(b_batch)]

        self.roots = roots
        self.root = roots[0]

        # Evaluate and expand initial root nodes with Dirichlet exploration noise
        self._evaluate_and_expand_batched(roots, is_root=True)

        all_search_depths = []
        for _ in range(self.num_simulations):
            search_paths = []
            leaf_nodes = []

            for b in range(b_batch):
                node = roots[b]
                path = [node]
                while node.children:
                    action = self._select_action(node, stats[b])
                    children = node.children[action]
                    node = children[0] if len(children) == 1 else random.choice(children)
                    path.append(node)
                search_paths.append(path)
                leaf_nodes.append(node)
                all_search_depths.append(len(path) - 1)

            self._evaluate_and_expand_batched(leaf_nodes, is_root=False)

            for b in range(b_batch):
                self._backpropagate(search_paths[b], leaf_nodes[b].bootstrap_value, stats[b])

        self.last_avg_depth = float(np.mean(all_search_depths)) if all_search_depths else 0.0
        self.last_max_depth = float(np.max(all_search_depths)) if all_search_depths else 0.0

        dists = [self._get_action_distribution(roots[b], temperature) for b in range(b_batch)]
        return torch.tensor(dists, dtype=torch.float32, device=self.device)

    def _select_action(self, node: LatentSearchNode, stats: MinMaxStats) -> int:
        """
        Selects ego action maximizing Bellman-integrated PUCT score:
        Q(s, a) = (1 / |C(s, a)|) * sum_{c in C(s, a)} [ r_c + gamma * V_c ]
        """
        best_score = -float('inf')
        best_action = -1

        for action in range(self.action_dim_i):
            if action not in node.children:
                continue
            children = node.children[action]
            action_visits = sum(c.visit_count for c in children)
            action_prior = children[0].prior if children else (1.0 / self.action_dim_i)

            if action_visits == 0:
                total_boot = sum(c.bootstrap_value for c in children) / max(len(children), 1)
                q_init = stats.normalize(total_boot)
                score = q_init + self.c_puct * action_prior * math.sqrt(node.visit_count + 1)
            else:
                total_q = 0.0
                for c in children:
                    v_next = c.value if c.visit_count > 0 else c.bootstrap_value
                    total_q += (c.reward + self.discount * v_next)
                expected_q = total_q / len(children)

                normalized_q = stats.normalize(expected_q)
                u = self.c_puct * action_prior * math.sqrt(node.visit_count) / (1 + action_visits)
                score = normalized_q + u

            if score > best_score:
                best_score = score
                best_action = action

        return best_action if best_action != -1 else random.randint(0, self.action_dim_i - 1)

    def _evaluate_and_expand_batched(self, nodes: List[LatentSearchNode], is_root: bool = False):
        """
        Evaluates neural heads and expands child branches across all leaf nodes.
        Enforces bfloat16-safe multinomial sampling and CUDA Graph address safety via explicit .clone().
        """
        if not nodes:
            return

        batched_beliefs = torch.cat([n.belief for n in nodes], dim=0)
        b_eval = batched_beliefs.size(0)
        num_branches = self.action_dim_i * self.num_observation_samples

        rep_beliefs = batched_beliefs.repeat_interleave(num_branches, dim=0)

        action_indices = torch.arange(self.action_dim_i, device=self.device).repeat_interleave(self.num_observation_samples)
        rep_ego_a = F.one_hot(action_indices.repeat(b_eval), num_classes=self.action_dim_i).float()

        use_amp = (self.device.type == 'cuda')
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
            opp_logits = self.opponent_head(batched_beliefs).clone()

            # Row-aligned opponent action sampling (float32 for multinomial): one opponent
            # action per expanded branch, drawn from pi_j(. | z) of the branch's parent latent.
            rep_opp_probs = F.softmax(opp_logits.float(), dim=-1).repeat_interleave(num_branches, dim=0)
            opp_a_idx = torch.multinomial(rep_opp_probs, num_samples=1).squeeze(-1)
            rep_opp_a = F.one_hot(opp_a_idx, num_classes=self.action_dim_j).float()

            observation_probs = F.softmax(self.observation_head(rep_beliefs, rep_ego_a, rep_opp_a).float(), dim=-1)
            observation = torch.multinomial(observation_probs, num_samples=1).squeeze(-1)
            next_beliefs = self.world_model.belief_filter.step(
                rep_beliefs, rep_ego_a, F.one_hot(observation, observation_probs.shape[-1]).to(rep_beliefs.dtype)).clone()
            rewards_logits = self.reward_head(rep_beliefs, rep_ego_a, rep_opp_a).clone()
            v_logits = self.value_head(next_beliefs).clone()

        rewards = self.twohot.mean(rewards_logits).tolist()
        next_values = self.twohot.mean(v_logits).tolist()

        base_prior = 1.0 / self.action_dim_i
        for b_idx, node in enumerate(nodes):
            dirichlet_noise = (
                np.random.dirichlet([self.dirichlet_alpha] * self.action_dim_i) if is_root else None
            )
            for a_idx in range(self.action_dim_i):
                node.children[a_idx] = []
                action_prior = (
                    (1.0 - self.dirichlet_epsilon) * base_prior + self.dirichlet_epsilon * float(dirichlet_noise[a_idx])
                ) if is_root else base_prior

                for k_idx in range(self.num_observation_samples):
                    flat_idx = (b_idx * num_branches) + (a_idx * self.num_observation_samples) + k_idx
                    child_node = LatentSearchNode(
                        belief=next_beliefs[flat_idx:flat_idx + 1].detach(),
                        action_taken=a_idx,
                        reward=rewards[flat_idx],
                        prior=action_prior
                    )
                    child_node.bootstrap_value = next_values[flat_idx]
                    node.children[a_idx].append(child_node)

    def _backpropagate(self, search_path: List[LatentSearchNode], value: float, stats: MinMaxStats):
        """Propagates target returns backward up the search path."""
        for node in reversed(search_path):
            node.visit_count += 1
            node.value += (value - node.value) / node.visit_count
            value = node.reward + (self.discount * value)
            stats.update(node.value)
            stats.update(value)

    def _get_action_distribution(self, root: LatentSearchNode, temperature: float = 1.0) -> List[float]:
        """Calculates policy probability distribution from root child visit counts and computes search telemetry."""
        counts = [
            sum(c.visit_count for c in root.children[i]) if i in root.children else 0
            for i in range(self.action_dim_i)
        ]
        q_vals = [
            sum(c.value for c in root.children[i]) / max(len(root.children[i]), 1)
            if (i in root.children and sum(c.visit_count for c in root.children[i]) > 0) else 0.0
            for i in range(self.action_dim_i)
        ]
        self.last_q_spread = float(max(q_vals) - min(q_vals)) if q_vals else 0.0

        if sum(counts) == 0:
            self.last_entropy = float(np.log(self.action_dim_i))
            return [1.0 / self.action_dim_i] * self.action_dim_i

        if temperature == 0.0:
            best_idx = counts.index(max(counts))
            self.last_entropy = 0.0
            return [1.0 if i == best_idx else 0.0 for i in range(len(counts))]

        max_count = max(counts)
        adjusted = [(c / max_count) ** (1.0 / max(temperature, 1e-4)) for c in counts]
        total_adj = sum(adjusted)
        probs = [c / total_adj for c in adjusted]
        self.last_entropy = float(-sum(p * np.log(max(p, 1e-8)) for p in probs))
        return probs
