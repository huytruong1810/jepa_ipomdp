# ABSOLUTE PATH: src/ipomdp/planning/mcts.py
# ==============================================================================
# BELIEF-TREE MONTE CARLO TREE SEARCH WITH EXACT OBSERVATION BRANCHING
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Tree Structure (a belief tree, as in exact POMDP planning):
#      decision node  s          (a belief b or a learned latent z; see search_model.py)
#        -> edge (s, a)          reward R(s, a), observation probabilities P(o | s, a)
#          -> child  tau(s,a,o)  one decision node per observation o (exact branching)
#    - Expanding a node calls SearchModel.expand once and creates ALL |A| x |O| children, each
#      with a leaf estimate V(child). Every action therefore has a one-step lookahead value
#      from the moment its parent is expanded.
#
# 2. Values: Expectimax Backups on the Expanded Tree:
#      V^(s)    = max_a Q(s, a)            if s is expanded, else its leaf estimate V(s)
#      Q(s, a)  = R(s, a) + gamma * sum_o P(o | s, a) * V^(tau(s, a, o))
#    - Decision nodes take the MAX over actions and chance edges the EXACT EXPECTATION over
#      observations, so the expanded tree computes precisely the finite belief-tree values the
#      exact solver computes, with the leaf estimates at its frontier. With exact leaves the
#      search can only improve on them; it never mixes in the values of exploratory actions.
#    - Phase-4 finding: the earlier mean backup (V^ = average of all returns backed up through
#      a node, as in MuZero) made estimates WORSE with more search. At b = (0.97, 0.03) with
#      exact V_8 leaves, root Q(OPEN_RIGHT) fell from the exact 12.80 (1 simulation) to -2.98
#      (1000 simulations) because returns of exploratory door openings (Q = -90) were averaged
#      in; the search then preferred LISTEN over the optimal OPEN_RIGHT. MuZero tolerates mean
#      backups only because a learned policy prior keeps exploration narrow; here the prior is
#      uniform and the branch probabilities are exact, so expectimax is the principled choice.
#    - Only nodes on the simulated path change, so V^ is cached per node and recomputed
#      bottom-up along the path after each expansion.
#
# 3. Selection (PUCT, AlphaZero/MuZero form):
#      a* = argmax_a  Q_norm(s, a) + c_puct * P(a) * sqrt(N(s) + 1) / (1 + N(s, a))
#    - P(a) is uniform (no policy network); at the root it is mixed with Dirichlet(alpha)
#      noise with weight epsilon when exploration is requested (training), and left uniform
#      for evaluation (dirichlet_epsilon = 0).
#    - Q_norm is Q min-max normalised over all Q values computed in the current tree, so
#      c_puct is independent of the reward scale (-100 .. +10 on Tiger).
#    - Descent below the chosen edge samples o ~ P(o | s, a), so simulations concentrate on
#      likely observation branches.
#    - Bugs of the previous search fixed here: unvisited actions were scored by V(child)
#      WITHOUT the immediate reward, so opening a door (r = -45 in expectation at b0) looked
#      as good as listening; Python's `random`/NumPy global generators were unseeded; a
#      silent random-action fallback existed; Q telemetry ignored rewards.
#
# 4. Reproducibility:
#    - All randomness (Dirichlet noise, observation sampling) comes from one seeded
#      numpy Generator owned by the planner.
#
# 5. Output:
#    - temperature = 0 (evaluation): a one-hot on argmax_a Q(root, a), the Bellman-optimal
#      action of the searched tree.
#    - temperature > 0 (training collection): visit counts shaped by counts^(1/T), which
#      follow PUCT's exploration and give a smoothed, exploratory policy.
#    - Root visit counts and Q values are recorded for telemetry and tests.
# ==============================================================================

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from .search_model import SearchModel


class MinMaxStats:
    """Running minimum/maximum of Q values in one tree, for PUCT normalisation."""

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value: float) -> None:
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        """Maps value into [0, 1]; 0.5 until two distinct values have been seen."""
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return 0.5


@dataclass
class DecisionNode:
    """A belief (or latent) node; `edges` is None until the node is expanded."""

    state: Tensor
    leaf_value: float
    visits: int = 0
    edges: list["Edge"] | None = None
    cached_value: float | None = None

    def value(self) -> float:
        """V^: max_a Q(s, a) once expanded (cached, see refresh), else the leaf estimate."""
        return self.leaf_value if self.cached_value is None else self.cached_value

    def refresh(self, discount: float) -> None:
        """Recomputes the cached max_a Q(s, a) from the edges' current children values."""
        self.cached_value = max(edge.q(discount) for edge in self.edges)


@dataclass
class Edge:
    """Action edge (s, a): expected reward, exact observation distribution, one child per o."""

    reward: float
    observation_probs: np.ndarray
    children: list[DecisionNode]
    visits: int = 0

    def q(self, discount: float) -> float:
        """R(s, a) + gamma * sum_o P(o | s, a) V^(child_o)."""
        return self.reward + discount * float(np.dot(self.observation_probs, [c.value() for c in self.children]))


@dataclass
class SearchStatistics:
    """Root-level results of the last search (one row per root)."""

    visit_counts: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    q_values: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    mean_depth: float = 0.0
    max_depth: int = 0


class BeliefTreeSearch:
    """Batched PUCT search over belief trees with exact observation branching."""

    def __init__(
        self,
        model: SearchModel,
        num_simulations: int,
        c_puct: float,
        dirichlet_alpha: float,
        dirichlet_epsilon: float,
        seed: int,
    ):
        """
        Args:
            model: SearchModel providing expansions (exact or learned).
            num_simulations: Simulations per root per search (>= 1).
            c_puct: Exploration constant.
            dirichlet_alpha: Concentration of the root prior noise.
            dirichlet_epsilon: Weight of the root prior noise (0 disables it).
            seed: Seed of the planner's random generator.
        """
        if num_simulations < 1:
            raise ValueError(f"num_simulations must be >= 1, got {num_simulations}.")
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.rng = np.random.default_rng(seed)
        self.roots: list[DecisionNode] = []
        self.statistics = SearchStatistics()

    def _expand(self, nodes: list[DecisionNode], stats: list[MinMaxStats]) -> None:
        """Expands all `nodes` with one batched model call and records their Q values."""
        expansion = self.model.expand(torch.stack([node.state for node in nodes]))
        rewards = expansion.rewards.double().cpu().numpy()
        probs = expansion.observation_probs.double().cpu().numpy()
        values = expansion.next_values.double().cpu().numpy()
        next_states = expansion.next_states
        for i, (node, node_stats) in enumerate(zip(nodes, stats)):
            node.edges = [
                Edge(reward=float(rewards[i, a]), observation_probs=probs[i, a],
                     children=[DecisionNode(next_states[i, a, o], float(values[i, a, o]))
                               for o in range(self.model.num_observations)])
                for a in range(self.model.num_actions)
            ]
            for edge in node.edges:
                node_stats.update(edge.q(self.model.discount))
            node.refresh(self.model.discount)

    def _select(self, node: DecisionNode, prior: np.ndarray, stats: MinMaxStats) -> int:
        """PUCT action selection (module header, section 3)."""
        total_visits = sum(edge.visits for edge in node.edges)
        scores = [
            stats.normalize(edge.q(self.model.discount))
            + self.c_puct * prior[a] * np.sqrt(total_visits + 1) / (1 + edge.visits)
            for a, edge in enumerate(node.edges)
        ]
        return int(np.argmax(scores))

    @torch.no_grad()
    def search(self, root_states: Tensor, temperature: float) -> Tensor:
        """
        Runs num_simulations simulations from every root.

        Args:
            root_states: States of shape (B, ...), as produced by the SearchModel.
            temperature: 0 for a one-hot argmax of visit counts, > 0 for counts^(1/T).

        Returns:
            Action distributions of shape (B, |A|), float32 on the states' device.
        """
        num_roots, num_a = root_states.shape[0], self.model.num_actions
        roots = [DecisionNode(root_states[i], leaf_value=0.0) for i in range(num_roots)]
        stats = [MinMaxStats() for _ in range(num_roots)]
        self._expand(roots, stats)
        uniform = np.full(num_a, 1.0 / num_a)
        priors = [
            (1.0 - self.dirichlet_epsilon) * uniform
            + self.dirichlet_epsilon * self.rng.dirichlet([self.dirichlet_alpha] * num_a)
            if self.dirichlet_epsilon > 0 else uniform
            for _ in range(num_roots)
        ]

        depths = []
        for _ in range(self.num_simulations):
            paths, leaves = [], []
            for root, prior, root_stats in zip(roots, priors, stats):
                node, path = root, []
                while node.edges is not None:
                    edge = node.edges[self._select(node, prior if node is root else uniform, root_stats)]
                    observation = self.rng.choice(len(edge.children),
                                                  p=edge.observation_probs / edge.observation_probs.sum())
                    path.append((node, edge))
                    node = edge.children[observation]
                paths.append(path)
                leaves.append(node)
                depths.append(len(path))
            self._expand(leaves, stats)
            for leaf, path, root_stats in zip(leaves, paths, stats):
                leaf.visits += 1
                for node, edge in reversed(path):
                    root_stats.update(edge.q(self.model.discount))
                    edge.visits += 1
                    node.visits += 1
                    node.refresh(self.model.discount)

        self.roots = roots
        counts = np.array([[edge.visits for edge in root.edges] for root in roots], dtype=np.float64)
        self.statistics = SearchStatistics(
            visit_counts=counts,
            q_values=np.array([[edge.q(self.model.discount) for edge in root.edges] for root in roots]),
            mean_depth=float(np.mean(depths)),
            max_depth=int(np.max(depths)),
        )
        if temperature == 0.0:
            policy = np.zeros_like(counts)
            policy[np.arange(num_roots), self.statistics.q_values.argmax(axis=1)] = 1.0
        else:
            shaped = (counts / counts.max(axis=1, keepdims=True)) ** (1.0 / temperature)
            policy = shaped / shaped.sum(axis=1, keepdims=True)
        return torch.as_tensor(policy, dtype=torch.float32, device=root_states.device)
