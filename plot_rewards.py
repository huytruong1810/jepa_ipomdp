# ABSOLUTE PATH: plot_rewards.py
# ==============================================================================
# TENSORBOARD REWARD TRAJECTORY EXTRACTION & PLOTTING UTILITY
# ==============================================================================

from pathlib import Path
import glob
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def plot_reward_trajectories(logdir: str = "tiger_tensorboard", save_path: str = "tiger_plots/reward_trajectories.png"):
    runs = sorted(glob.glob(f"{logdir}/*"))
    if not runs:
        print(f"[!] No runs found in {logdir}")
        return

    all_steps = []
    all_rewards = []
    all_mins = []
    all_maxs = []
    r_losses_steps = []
    r_losses_vals = []

    for run in runs:
        try:
            ea = EventAccumulator(run)
            ea.Reload()
            tags = ea.Tags().get("scalars", [])
            
            if "Train/episode_reward_mean" in tags:
                for e in ea.Scalars("Train/episode_reward_mean"):
                    all_steps.append(e.step)
                    all_rewards.append(e.value)
            
            if "Train/episode_reward_min" in tags:
                for e in ea.Scalars("Train/episode_reward_min"):
                    all_mins.append((e.step, e.value))
                    
            if "Train/episode_reward_max" in tags:
                for e in ea.Scalars("Train/episode_reward_max"):
                    all_maxs.append((e.step, e.value))
                    
            if "Train/loss_reward" in tags:
                for e in ea.Scalars("Train/loss_reward"):
                    r_losses_steps.append(e.step)
                    r_losses_vals.append(e.value)
        except Exception as ex:
            print(f"Skipping run {run}: {ex}")

    if not all_steps:
        print(f"[!] No reward points logged yet in {logdir}.")
        return

    # Sort points chronologically by step
    sorted_pairs = sorted(zip(all_steps, all_rewards), key=lambda x: x[0])
    steps, rewards = zip(*sorted_pairs)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=150)

    # Plot Episode Mean Return
    line1 = ax1.plot(steps, rewards, color="#1f77b4", linewidth=2.0, label="Episode Return (Mean)")
    ax1.set_xlabel("Environment Steps", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Episode Return", color="#1f77b4", fontsize=12, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Plot Min/Max bands if available
    if all_mins and all_maxs:
        min_dict = dict(all_mins)
        max_dict = dict(all_maxs)
        common_steps = [s for s in steps if s in min_dict and s in max_dict]
        if common_steps:
            mins = [min_dict[s] for s in common_steps]
            maxs = [max_dict[s] for s in common_steps]
            ax1.fill_between(common_steps, mins, maxs, color="#1f77b4", alpha=0.15, label="Min/Max Return Band")

    # Secondary Axis: Reward Loss if present
    if r_losses_steps:
        ax2 = ax1.twinx()
        sorted_rloss = sorted(zip(r_losses_steps, r_losses_vals), key=lambda x: x[0])
        rl_steps, rl_vals = zip(*sorted_rloss)
        line2 = ax2.plot(rl_steps, rl_vals, color="#ff7f0e", linestyle=":", linewidth=1.5, label="Reward Loss (TwoHot CE)")
        ax2.set_ylabel("Reward Head Loss", color="#ff7f0e", fontsize=12, fontweight="bold")
        ax2.tick_params(axis="y", labelcolor="#ff7f0e")

    plt.title(f"JEPA I-POMDP Tiger Reward Trajectory ({steps[0]:,} -> {steps[-1]:,} Steps)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[✓] Reward trajectory plot successfully saved to: {save_path}")


if __name__ == "__main__":
    plot_reward_trajectories()
