# ABSOLUTE PATH: src/ipomdp/telemetry/visualizers.py
# ==============================================================================
# LATENT VISUALIZERS, MCTS GRAPH RENDERERS & SEMANTIC PROBES
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Vector Beliefs:
#    - Belief latents are single vectors (models/world_model.py, section 2); trajectories
#      of shape (T, D) are projected with PCA directly (reviewed in Phase 6).
#
# 2. Discrete Integer Layer Indexing for MCTS Tree Layouts:
#    - Uses discrete integer layer keys (2*d for state nodes, 2*d + 1 for action nodes)
#      to eliminate NetworkX multipartite layout sorting glitches across versions.
# ==============================================================================

from pathlib import Path
from typing import List, Dict, Optional, Any
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import torch


class LatentSpaceVisualizer:
    """Visualizes high-dimensional latent belief trajectories using Permutation-Invariant PCA."""

    def __init__(self, save_dir: str = "plots/latent"):
        """Initializes plot output directory."""
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _pool_belief(self, b_seq: torch.Tensor, value_head: Optional[torch.nn.Module] = None) -> np.ndarray:
        """Belief latents (T, D) as a NumPy matrix for PCA."""
        if b_seq.dim() != 2:
            raise ValueError(f"Expected belief latents of shape (T, D), got {tuple(b_seq.shape)}.")
        return b_seq.detach().float().cpu().numpy()

    def plot_trajectory(
        self,
        beliefs: List[torch.Tensor],
        true_states: List[torch.Tensor],
        actions: List[int],
        filename: str = "latent_trajectory",
        value_head: Optional[torch.nn.Module] = None,
        action_map: Optional[Dict[int, str]] = None,
        n_components: int = 2
    ):
        """Plots a single episodic trajectory in 1D (vs Time) or 2D PCA space."""
        if len(beliefs) < 2:
            return

        b_seq = torch.cat(beliefs, dim=0)
        belief_matrix_np = self._pool_belief(b_seq, value_head)

        n_samples, n_features = belief_matrix_np.shape
        n_comp = min(n_components, n_samples, n_features)
        if n_comp < 1:
            return

        pca = PCA(n_components=n_comp)
        coords = pca.fit_transform(belief_matrix_np)
        labels = [int(s.view(-1)[0].item()) if torch.is_tensor(s) else int(s) for s in true_states]

        cmap = mpl.colormaps['tab10']
        colors = [cmap(l % 10) for l in labels]

        plt.figure(figsize=(12, 9 if n_comp >= 2 else 6))

        if n_comp == 1:
            timesteps = np.arange(len(coords))
            coords_1d = coords[:, 0]

            plt.plot(timesteps, coords_1d, color='gray', alpha=0.5, linewidth=2, zorder=1)
            plt.scatter(timesteps, coords_1d, c=colors, s=120, edgecolors='white', zorder=2)

            for i in range(len(coords_1d)):
                if i < len(actions):
                    act_idx = actions[i]
                    act_str = action_map[act_idx] if (action_map and act_idx in action_map) else f"A:{act_idx}"
                else:
                    act_str = "Terminal"

                plt.annotate(
                    act_str,
                    (timesteps[i], coords_1d[i]),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none")
                )

            plt.scatter(timesteps[0], coords_1d[0], marker='*', color='gold', s=400, edgecolors='black', zorder=3, label='Start')
            plt.scatter(timesteps[-1], coords_1d[-1], marker='X', color='black', s=250, zorder=3, label='End')

            plt.title("Permutation-Invariant Latent Belief Trajectory (1D PCA vs Time)", fontsize=14, pad=15)
            plt.xlabel("Timestep (t)")
            plt.ylabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} Variance)")
        else:
            coords_2d = coords
            plt.plot(coords_2d[:, 0], coords_2d[:, 1], color='gray', alpha=0.5, linewidth=2, zorder=1)
            plt.scatter(coords_2d[:, 0], coords_2d[:, 1], c=colors, s=120, edgecolors='white', zorder=2)

            for i in range(len(coords_2d)):
                if i < len(actions):
                    act_idx = actions[i]
                    act_str = action_map[act_idx] if (action_map and act_idx in action_map) else f"A:{act_idx}"
                else:
                    act_str = "Terminal"

                plt.annotate(
                    act_str,
                    (coords_2d[i, 0], coords_2d[i, 1]),
                    xytext=(8, 8),
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none")
                )

            plt.scatter(coords_2d[0, 0], coords_2d[0, 1], marker='*', color='gold', s=500, edgecolors='black', zorder=3, label='Start')
            plt.scatter(coords_2d[-1, 0], coords_2d[-1, 1], marker='X', color='black', s=300, zorder=3, label='End')

            plt.title("Permutation-Invariant Latent Belief Trajectory (2D PCA)", fontsize=14, pad=15)
            plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} Variance)")
            plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} Variance)")

        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_multiple_trajectories(
        self,
        trajectories: List[Dict[str, Any]],
        filename: str = "holistic_latent_space",
        value_head: Optional[torch.nn.Module] = None,
        n_components: int = 2
    ):
        """Fits a universal PCA space across multiple trajectories in 1D or 2D."""
        if not trajectories:
            return

        all_beliefs = []
        lengths = []
        for traj in trajectories:
            b_seq = torch.cat(traj["beliefs"], dim=0)
            all_beliefs.append(b_seq)
            lengths.append(b_seq.size(0))

        global_b_seq = torch.cat(all_beliefs, dim=0)
        global_matrix_np = self._pool_belief(global_b_seq, value_head)

        n_samples, n_features = global_matrix_np.shape
        n_comp = min(n_components, n_samples, n_features)
        if n_comp < 1:
            return

        pca = PCA(n_components=n_comp)
        global_coords = pca.fit_transform(global_matrix_np)

        plt.figure(figsize=(14, 10 if n_comp >= 2 else 6))
        start_idx = 0

        for idx, traj in enumerate(trajectories):
            end_idx = start_idx + lengths[idx]
            color = traj.get('color', 'gray')
            name = traj.get('name', f'Trajectory {idx}')

            if n_comp == 1:
                traj_y = global_coords[start_idx:end_idx, 0]
                timesteps = np.arange(len(traj_y))

                plt.plot(timesteps, traj_y, color=color, alpha=0.7, linewidth=2.5, marker='o', label=name, zorder=2)
                plt.scatter(timesteps[0], traj_y[0], color=color, marker='*', s=300, edgecolor='black', zorder=4)
                plt.scatter(timesteps[-1], traj_y[-1], color=color, marker='s', s=100, edgecolor='black', zorder=4)
            else:
                traj_x = global_coords[start_idx:end_idx, 0]
                traj_y = global_coords[start_idx:end_idx, 1]

                plt.plot(traj_x, traj_y, color=color, alpha=0.6, linewidth=3, label=name, zorder=2)
                plt.scatter(traj_x[0], traj_y[0], color=color, marker='*', s=400, edgecolor='black', zorder=4)
                plt.scatter(traj_x[-1], traj_y[-1], color=color, marker='o', s=100, edgecolor='black', zorder=4)

            start_idx = end_idx

        if n_comp == 1:
            plt.title("Holistic Latent Space Divergence (1D Universal PCA vs Time)", fontsize=16, pad=15)
            plt.xlabel("Timestep (t)")
            plt.ylabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} Variance)")
        else:
            plt.title("Holistic Latent Space Divergence (2D Universal PCA)", fontsize=16, pad=15)
            plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} Variance)")
            plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} Variance)")

        plt.legend(loc='best', fontsize=10, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_interactive(
        self,
        beliefs: List[torch.Tensor],
        filename: str = "interactive_latent.html",
        value_head: Optional[torch.nn.Module] = None
    ):
        """Generates an interactive Plotly HTML visualizer."""
        if len(beliefs) < 2:
            return

        b_seq = torch.cat(beliefs, dim=0)
        belief_matrix_np = self._pool_belief(b_seq, value_head)

        n_samples, n_features = belief_matrix_np.shape
        n_components = min(3, n_samples, n_features)
        if n_components < 1:
            return

        pca = PCA(n_components=n_components)
        coords = pca.fit_transform(belief_matrix_np)
        time_steps = np.arange(len(coords))

        if n_components >= 3:
            fig = go.Figure(data=[go.Scatter3d(
                x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                mode='lines+markers',
                marker=dict(size=6, color=time_steps, colorscale='Viridis', opacity=0.8, colorbar=dict(title="Time")),
                line=dict(color='gray', width=2)
            )])
            fig.update_layout(title="Interactive 3D Latent Trajectory")
        else:
            fig = go.Figure(data=[go.Scatter(
                x=coords[:, 0], y=coords[:, 1],
                mode='lines+markers',
                marker=dict(size=10, color=time_steps, colorscale='Viridis', showscale=True),
                line=dict(color='gray', width=2)
            )])
            fig.update_layout(title="Interactive 2D Latent Trajectory")

        fig.write_html(str(self.save_dir / filename))


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
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No Trajectories Available", ha="center", va="center")
            return fig

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
        ax2.set_xticks(timesteps)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="lower left", framealpha=0.9, fontsize=9)

        fig.tight_layout()
        out_path = self.save_dir / f"{filename}.png"
        fig.savefig(out_path)

        return fig

