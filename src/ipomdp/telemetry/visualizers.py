# ABSOLUTE PATH: src/ipomdp/telemetry/visualizers.py
# ==============================================================================
# BELIEF-GEOMETRY, SEARCH-TREE AND REWARD-TRAJECTORY FIGURES
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Belief Geometry, Not Per-Episode Trajectories:
#    - BeliefGeometryVisualizer projects the latents of MANY histories onto one shared PCA plane
#      and colours them by the exact posterior (interpretability/analysis.py). The earlier
#      latent plot fitted a separate PCA per episode (axes not comparable across plots) and
#      coloured by the hidden state, which the belief cannot know.
#
# 2. Discrete Integer Layer Indexing for MCTS Tree Layouts:
#    - Uses discrete integer layer keys (2*d for state nodes, 2*d + 1 for action nodes)
#      to eliminate NetworkX multipartite layout sorting glitches across versions.
# ==============================================================================

from pathlib import Path
from typing import List, Dict, Optional, Any
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import torch


class BeliefGeometryVisualizer:
    """Latents of many histories in one shared PCA plane, coloured by the exact posterior."""

    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot(self, latents: torch.Tensor, posteriors: torch.Tensor, state_name: str, filename: str) -> plt.Figure:
        """
        Scatter of latents (N, D) in their first two principal components, coloured by the exact
        posterior of state 0 (N,). One PCA is fitted over all N latents, so axes are comparable
        across the whole data set (an earlier version fitted a new PCA per episode).

        Returns:
            The figure (also saved as <filename>.png), for TensorBoard.
        """
        centred = latents.double() - latents.double().mean(0)
        _, singular_values, vh = torch.linalg.svd(centred, full_matrices=False)
        coords = (centred @ vh[:2].T).cpu().numpy()
        ratios = (singular_values ** 2 / (singular_values ** 2).sum())[:2].tolist()
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        points = ax.scatter(coords[:, 0], coords[:, 1], c=posteriors.cpu().numpy(), cmap="coolwarm", vmin=0.0,
                            vmax=1.0, s=6, alpha=0.6)
        fig.colorbar(points, ax=ax, label=f"exact posterior P({state_name} | history)")
        ax.set_xlabel(f"PC1 ({ratios[0]:.1%} of variance)")
        ax.set_ylabel(f"PC2 ({ratios[1]:.1%} of variance)")
        ax.set_title("Belief-filter latents coloured by the exact Bayes posterior")
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()
        fig.savefig(self.save_dir / f"{filename}.png")
        return fig


class MCTSGraphVisualizer:
    """Renders belief-tree searches (planning/mcts.py) into interactive Plotly HTML graphs."""

    def __init__(self, save_dir: str = "plots/trees"):
        """Initializes plot output directory."""
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def visualize(self, root_node, discount: float, action_names: tuple[str, ...],
                  observation_names: tuple[str, ...], filename: str = "mcts_tree", max_depth: int = 4):
        """
        Exports the belief tree below `root_node` (planning/mcts.py DecisionNode) to HTML.

        Decision nodes show visits and V^; action nodes show visits, Q(s, a) and R(s, a);
        edges into children are labelled by the observation and its probability.
        """
        G = nx.DiGraph()
        self._add_nodes_edges(G, root_node, discount, action_names, observation_names, depth=0, max_depth=max_depth)
        pos = nx.multipartite_layout(G, subset_key="depth", align="horizontal")

        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        state_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'state']
        action_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'action']

        traces = [edge_trace]

        if state_nodes:
            s_x = [pos[n][0] for n in state_nodes]
            s_y = [pos[n][1] for n in state_nodes]
            s_text = [G.nodes[n]['text'] for n in state_nodes]
            traces.append(go.Scatter(
                x=s_x, y=s_y, mode='markers',
                hoverinfo='text', text=s_text,
                marker=dict(size=12, color='lightblue', line=dict(width=2, color='DarkSlateGrey'))
            ))

        if action_nodes:
            a_x = [pos[n][0] for n in action_nodes]
            a_y = [pos[n][1] for n in action_nodes]
            a_text = [G.nodes[n]['text'] for n in action_nodes]
            traces.append(go.Scatter(
                x=a_x, y=a_y, mode='markers',
                hoverinfo='text', text=a_text,
                marker=dict(size=10, color='orange', symbol='diamond', line=dict(width=1, color='DarkRed'))
            ))

        fig = go.Figure(
            data=traces,
            layout=go.Layout(
                title='Latent MCTS Search Tree',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        fig.write_html(str(self.save_dir / f"{filename}.html"))

    def _add_nodes_edges(self, G: nx.DiGraph, node, discount: float, action_names: tuple[str, ...],
                         observation_names: tuple[str, ...], depth: int, max_depth: int,
                         parent_id: str | None = None, branch_text: str = ""):
        node_id = f"d{depth}_{id(node)}"
        G.add_node(node_id, depth=2 * depth, type='state',
                   text=f"<b>Belief node</b><br>{branch_text}Visits: {node.visits}<br>V^: {node.value():.3f}")
        if parent_id is not None:
            G.add_edge(parent_id, node_id)
        if node.edges is None or depth >= max_depth:
            return
        for action, edge in enumerate(node.edges):
            action_id = f"{node_id}_a{action}"
            G.add_node(action_id, depth=2 * depth + 1, type='action',
                       text=(f"<b>{action_names[action]}</b><br>Visits: {edge.visits}<br>"
                             f"Q: {edge.q:.3f}<br>R: {edge.reward:.3f}"))
            G.add_edge(node_id, action_id)
            for observation, child in enumerate(edge.children):
                self._add_nodes_edges(
                    G, child, discount, action_names, observation_names, depth + 1, max_depth, action_id,
                    f"o = {observation_names[observation]} (p = {edge.observation_probs[observation]:.3f})<br>")


class RewardTrajectoryVisualizer:
    """
    Visualizes canonical within-episode cumulative reward trajectories,
    confidence bounds, and action-annotated rollout histories.
    """

    def __init__(self, save_dir: str = "plots/rewards"):
        """Initializes plot output directory."""
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_cumulative_rewards(
        self,
        trajectories: List[Dict[str, Any]],
        filename: str = "canonical_episode_cumulative_rewards",
        title_suffix: str = "",
        max_sample_plots: int = 5,
        action_map: Optional[Dict[int, str]] = None
    ) -> plt.Figure:
        """
        Generates and saves the canonical within-episode cumulative reward trajectory figure.

        Args:
            trajectories: List of dicts, each with keys:
                - 'cum_rewards': Sequence of cumulative rewards of length T + 1 (starts at 0.0).
                - 'actions': Sequence of integer actions of length T.
                - 'rewards': Sequence of float step rewards of length T.
            filename: Base name for saved PNG (without .png).
            title_suffix: Optional extra context in plot title (e.g. 'Step 125,000').
            max_sample_plots: Number of individual episode traces to show in the bottom panel.
            action_map: Optional mapping from action index to readable name.

        Returns:
            Matplotlib Figure object for TensorBoard summary logging.
        """
        if not trajectories:
            raise ValueError("plot_cumulative_rewards needs at least one trajectory.")

        # Determine uniform max length across trajectories
        max_t = max(len(t["cum_rewards"]) for t in trajectories)
        n_trajs = len(trajectories)

        # Pad with last value to construct matrix for mean / std calculation
        aligned_matrix = np.zeros((n_trajs, max_t), dtype=np.float32)
        for i, t in enumerate(trajectories):
            series = np.array(t["cum_rewards"], dtype=np.float32)
            aligned_matrix[i, :len(series)] = series
            if len(series) < max_t:
                aligned_matrix[i, len(series):] = series[-1]

        timesteps = np.arange(max_t)
        mean_cum_rew = np.mean(aligned_matrix, axis=0)
        std_cum_rew = np.std(aligned_matrix, axis=0)
        min_cum_rew = np.min(aligned_matrix, axis=0)
        max_cum_rew = np.max(aligned_matrix, axis=0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), dpi=150, sharex=True)

        # -------------------------------------------------------------
        # Panel 1: Aggregate Within-Episode Cumulative Reward Trajectory
        # -------------------------------------------------------------
        ax1.plot(timesteps, mean_cum_rew, color="#1f77b4", linewidth=2.5, label="Mean Cumulative Reward")
        ax1.fill_between(
            timesteps,
            mean_cum_rew - std_cum_rew,
            mean_cum_rew + std_cum_rew,
            color="#1f77b4",
            alpha=0.25,
            label=r"$\pm 1$ Standard Deviation"
        )
        ax1.fill_between(
            timesteps,
            min_cum_rew,
            max_cum_rew,
            color="#1f77b4",
            alpha=0.10,
            label="Min - Max Envelope"
        )
        ax1.axhline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        ax1.set_ylabel("Cumulative Reward", fontsize=12, fontweight="bold")
        sub = f" ({title_suffix})" if title_suffix else ""
        ax1.set_title(
            f"Agent Cumulative Reward Over Time in Episode\n"
            f"(Averaged across N={n_trajs} Episode Rollouts{sub})",
            fontsize=13, fontweight="bold"
        )
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="lower left", framealpha=0.9)

        # -------------------------------------------------------------
        # Panel 2: Individual Sample Episode Trajectories with Action Markers
        # -------------------------------------------------------------
        n_samples = min(max_sample_plots, n_trajs)
        colors = ["#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#17becf"]

        for idx in range(n_samples):
            traj = trajectories[idx]
            c_series = traj["cum_rewards"]
            act_series = traj.get("actions", [])
            rew_series = traj.get("rewards", [])
            ep_timesteps = np.arange(len(c_series))

            color = colors[idx % len(colors)]
            ax2.plot(ep_timesteps, c_series, color=color, linewidth=1.5, alpha=0.8, label=f"Episode {idx + 1}")

            for step_idx, act in enumerate(act_series):
                curr_c = c_series[step_idx + 1]
                rew = rew_series[step_idx] if step_idx < len(rew_series) else 0.0

                if rew > 0:
                    # Positive reward (e.g. Treasure +10) -> Gold star
                    ax2.scatter(step_idx + 1, curr_c, color="gold", edgecolors="black", s=140, marker="*", zorder=5)
                elif rew <= -10:
                    # Heavy penalty (e.g. Tiger -100) -> Red X
                    ax2.scatter(step_idx + 1, curr_c, color="red", edgecolors="darkred", s=80, marker="X", zorder=5)
                else:
                    # Small step cost (e.g. Listen -1.0) -> Small circle
                    ax2.scatter(step_idx + 1, curr_c, color=color, s=25, marker="o", alpha=0.6)

        # Marker legend
        ax2.scatter([], [], color="gray", marker="o", s=30, label="Listen / Step (-1.0)")
        ax2.scatter([], [], color="gold", edgecolors="black", marker="*", s=120, label="Treasure / Goal (+10.0)")
        ax2.scatter([], [], color="red", edgecolors="darkred", marker="X", s=80, label="Penalty / Hazard (-100.0)")

        ax2.axhline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        ax2.set_xlabel("Episode Timestep ($t$)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Cumulative Reward", fontsize=12, fontweight="bold")
        ax2.set_title("Sample Episode Rollouts (with Action Markers)", fontsize=12, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="lower left", framealpha=0.9, fontsize=9)

        fig.tight_layout()
        out_path = self.save_dir / f"{filename}.png"
        fig.savefig(out_path)

        return fig

