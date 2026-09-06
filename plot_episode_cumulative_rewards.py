# ABSOLUTE PATH: plot_episode_cumulative_rewards.py
# ==============================================================================
# CANONICAL EPISODIC CUMULATIVE REWARD TRAJECTORY GENERATOR
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Canonical Within-Episode Cumulative Return:
#    - Tracks the cumulative reward C_t = sum_{k=1}^t r_k over timestep t in [0, max_steps]
#      emitted by the environment payoff function R(s_t, a_t):
#         * Listen: -1.0
#         * Open Correct Door (Treasure): +10.0
#         * Open Incorrect Door (Tiger): -100.0
#
# 2. Parallel Vectorized Batch Rollout:
#    - Uses SyncVectorEnv to evaluate multiple parallel episode rollouts concurrently,
#      yielding synchronized episodic time series across parallel seeds.
#
# 3. Canonical Trajectory Visualizer:
#    - X-Axis: "Episode Timestep (t)"
#    - Y-Axis: "Cumulative Reward"
#    - Plots individual sample trajectories and the mean trajectory with confidence band.
# ==============================================================================

from pathlib import Path
import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn.functional as F

import sys
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ipomdp.telemetry import setup_logger, ModelCheckpointer
from ipomdp.telemetry.registry import make_extractor
from ipomdp.envs import (
    MultiAgentTigerEnv,
    SyncVectorEnv,
    encode_observation_to_index,
    LISTEN,
    OPEN_LEFT,
    OPEN_RIGHT,
    TIGER_LEFT,
    TIGER_RIGHT,
)
from ipomdp.models import (
    RecurrentContextEncoder,
    CausalRelationalPredictor,
    RecurrentJEPABase,
    ValueHead,
    RewardHead,
    DiscretePolicyHead,
)
from ipomdp.planning import DiscreteLatentOpenLoopSearch
from ipomdp.agents import DiscreteJEPAAgent, StatelessAgent
from ipomdp.types import Action


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("\n" + "=" * 70)
    print("  GENERATING CANONICAL WITHIN-EPISODE CUMULATIVE REWARD TRAJECTORIES ")
    print("=" * 70 + "\n")

    logger = setup_logger("Reward_Trajectory_Plotter")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = int(getattr(cfg, "eval_batch_size", 16))
    max_steps = int(cfg.env.env_kwargs.get("max_steps", 20))
    num_simulations = int(getattr(cfg.mcts, "eval_simulations", 15))

    extractor = make_extractor(
        cfg.env.extractor_name,
        obs_dim=cfg.env.obs_dim,
        hidden_dim=cfg.model.latent_dim,
        num_objects=cfg.model.num_objects,
        num_blocks=cfg.model.num_blocks
    )
    encoder = RecurrentContextEncoder(
        extractor,
        action_dim=cfg.env.action_dim_i,
        latent_dim=cfg.model.latent_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_blocks=cfg.model.num_blocks
    )
    predictor = CausalRelationalPredictor(
        num_objects=cfg.model.num_objects,
        latent_dim=cfg.model.latent_dim,
        action_dim_i=cfg.env.action_dim_i,
        action_dim_j=cfg.env.action_dim_j,
        hidden_dim=cfg.model.hidden_dim,
        num_blocks=cfg.model.num_blocks
    )
    jepa_model = RecurrentJEPABase(encoder, predictor).to(device)

    value_head = ValueHead(
        latent_dim=cfg.model.latent_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_blocks=cfg.model.num_blocks
    ).to(device)

    reward_head = RewardHead(
        latent_dim=cfg.model.latent_dim,
        action_dim_i=cfg.env.action_dim_i,
        action_dim_j=cfg.env.action_dim_j,
        hidden_dim=cfg.model.hidden_dim,
        num_blocks=cfg.model.num_blocks
    ).to(device)

    opponent_head = DiscretePolicyHead(
        latent_dim=cfg.model.latent_dim,
        action_dim=cfg.env.action_dim_j,
        num_opponents=1,
        hidden_dim=cfg.model.hidden_dim,
        num_blocks=cfg.model.num_blocks
    ).to(device)

    models_dict = {
        "jepa": jepa_model,
        "value": value_head,
        "reward": reward_head,
        "opponent": opponent_head
    }

    checkpointer = ModelCheckpointer("tiger_checkpoints", logger)
    ckpt_path = "tiger_checkpoints/latest_checkpoint.pt"
    step_loaded = checkpointer.load(ckpt_path, models_dict, device=device)
    logger.info(f"Loaded model weights from step {step_loaded}")

    jepa_model.eval()
    value_head.eval()
    reward_head.eval()
    opponent_head.eval()

    # Optimize PyTorch CPU threading to prevent core contention
    torch.set_num_threads(min(8, torch.get_num_threads()))

    total_episodes = int(getattr(cfg, "eval_total_episodes", 1000))
    batch_size = min(int(getattr(cfg, "eval_batch_size", 25)), total_episodes)
    num_simulations = int(getattr(cfg.mcts, "eval_simulations", 100))
    max_steps = int(cfg.env.env_kwargs.max_steps)

    logger.info(f"Target Total Episodes: {total_episodes} (Batch Size: {batch_size}, MCTS Sims: {num_simulations})")

    planner = DiscreteLatentOpenLoopSearch(
        jepa_model=jepa_model,
        value_head=value_head,
        reward_head=reward_head,
        opponent_head=opponent_head,
        action_dim_i=cfg.env.action_dim_i,
        action_dim_j=cfg.env.action_dim_j,
        num_simulations=num_simulations,
        num_latent_obs=cfg.mcts.num_latent_obs
    )

    hunter_agent = DiscreteJEPAAgent(
        agent_id="agent_0",
        planner=planner,
        latent_dim=cfg.model.latent_dim,
        action_dim=cfg.env.action_dim_i,
        device=device,
        num_objects=cfg.model.num_objects,
        temperature=0.2  # Low temperature for greedy/eval policy
    )
    opp_agent = StatelessAgent(agent_id="agent_1", action_dim=cfg.env.action_dim_j, action_idx=0)

    env_fn = lambda: MultiAgentTigerEnv(**dict(cfg.env.get("env_kwargs", {})))
    vec_env = SyncVectorEnv(env_fn, num_envs=batch_size)

    from ipomdp.telemetry import RewardTrajectoryVisualizer
    reward_viz = RewardTrajectoryVisualizer(save_dir="tiger_plots")
    out_path = Path("tiger_plots") / "canonical_episode_cumulative_rewards.png"

    all_trajectories = []
    episodes_run = 0

    with torch.inference_mode():
        while episodes_run < total_episodes:
            current_batch_size = min(batch_size, total_episodes - episodes_run)
            if current_batch_size != batch_size:
                vec_env = SyncVectorEnv(env_fn, num_envs=current_batch_size)

            observations, infos = vec_env.reset()
            hunter_agent.reset(batch_size=current_batch_size)
            opp_agent.reset(batch_size=current_batch_size)

            prev_actions = {
                "agent_0": Action(data=torch.zeros(current_batch_size, cfg.env.action_dim_i, device=device)),
                "agent_1": Action(data=torch.zeros(current_batch_size, cfg.env.action_dim_j, device=device))
            }

            batch_cum_rewards = np.zeros((current_batch_size, max_steps + 1), dtype=np.float32)
            batch_step_rewards = np.zeros((current_batch_size, max_steps), dtype=np.float32)
            batch_actions = np.zeros((current_batch_size, max_steps), dtype=np.int32)

            for t in range(max_steps):
                hunter_agent.update_belief(observations["agent_0"], prev_actions["agent_0"])
                act_i = hunter_agent.act(observations["agent_0"])
                act_j = opp_agent.act(observations["agent_1"])

                actions = {"agent_0": act_i, "agent_1": act_j}
                next_obs, rews, terms, truncs, infos = vec_env.step(actions)

                r_t = rews["agent_0"].view(-1).cpu().numpy()
                a_t = act_i.data.view(-1).cpu().numpy().astype(np.int32)

                batch_step_rewards[:, t] = r_t
                batch_cum_rewards[:, t + 1] = batch_cum_rewards[:, t] + r_t
                batch_actions[:, t] = a_t

                prev_actions = actions
                observations = next_obs

            # Store finished batch trajectories
            for i in range(current_batch_size):
                all_trajectories.append({
                    "cum_rewards": batch_cum_rewards[i].tolist(),
                    "actions": batch_actions[i].tolist(),
                    "rewards": batch_step_rewards[i].tolist()
                })

            episodes_run += current_batch_size

            # Progressive Plot & Statistics Update
            fig = reward_viz.plot_cumulative_rewards(
                trajectories=all_trajectories,
                filename="canonical_episode_cumulative_rewards",
                title_suffix=f"N={len(all_trajectories)}/{total_episodes}, 100 Sims, Model Step {step_loaded:,}",
                max_sample_plots=min(5, len(all_trajectories))
            )
            plt.close(fig)

            all_final = [t["cum_rewards"][-1] for t in all_trajectories]
            logger.info(
                f"Progressive Milestone: N={len(all_trajectories)}/{total_episodes} Episodes | "
                f"Mean Return: {np.mean(all_final):.2f} +/- {np.std(all_final):.2f} | "
                f"Min: {np.min(all_final):.1f} | Max: {np.max(all_final):.1f}"
            )

    print(f"\n[✓] Canonical cumulative reward trajectory plot updated: {out_path}")
    all_final = [t["cum_rewards"][-1] for t in all_trajectories]
    print(f"    Final Evaluated Episodes: N={len(all_trajectories)}")
    print(f"    Mean Final Return at t={max_steps}: {np.mean(all_final):.2f} +/- {np.std(all_final):.2f}")
    print(f"    Min Final Return: {np.min(all_final):.2f} | Max Final Return: {np.max(all_final):.2f}\n")


if __name__ == "__main__":
    main()
