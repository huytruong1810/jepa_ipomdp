# ABSOLUTE PATH: eval_tiger.py
# ==============================================================================
# MULTI-AGENT TIGER I-POMDP BENCHMARK & REWARD PERFORMANCE EVALUATOR
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Multi-Metric Domain Evaluation:
#    - Measures Episode Return (Mean, Std, Min, Max, Median), Treasure Success Rate (%),
#      Tiger Penalty Rate (%), Door Choice Accuracy (%), and Average Listen Count
#      over N independent episodes.
#
# 2. Strict Causal Context Filtering & Latent MCTS:
#    - Evaluates the learned RecurrentJEPABase world model and DiscreteLatentOpenLoopSearch
#      planner against the canonical baseline opponent.
#
# 3. Telemetry Integration:
#    - Formats evaluation metrics into clean tables and logs structured JSON telemetry.
# ==============================================================================

from pathlib import Path
import json
import sys
import hydra
import numpy as np
from omegaconf import DictConfig
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ipomdp.telemetry import setup_logger, log_telemetry, ModelCheckpointer
from ipomdp.telemetry.registry import make_extractor
from ipomdp.envs import (
    MultiAgentTigerEnv,
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
from ipomdp.agents import DiscreteJEPAAgent
from ipomdp.types import Action


def evaluate_tiger_performance(
    jepa_model: RecurrentJEPABase,
    value_head: ValueHead,
    reward_head: RewardHead,
    opponent_head: DiscretePolicyHead,
    cfg: DictConfig,
    num_episodes: int = 100,
    device: torch.device = torch.device("cpu")
) -> dict:
    """Evaluates the JEPA agent across num_episodes episodes of Tiger POMDP."""
    obs_dim = cfg.env.obs_dim
    action_dim_i = cfg.env.action_dim_i
    action_dim_j = cfg.env.action_dim_j

    planner = DiscreteLatentOpenLoopSearch(
        jepa_model=jepa_model,
        value_head=value_head,
        reward_head=reward_head,
        opponent_head=opponent_head,
        action_dim_i=action_dim_i,
        action_dim_j=action_dim_j,
        num_simulations=cfg.mcts.num_simulations,
        num_latent_obs=cfg.mcts.num_latent_obs
    )

    agent = DiscreteJEPAAgent(
        agent_id="agent_0",
        planner=planner,
        latent_dim=cfg.model.latent_dim,
        action_dim=action_dim_i,
        device=device,
        num_objects=cfg.model.num_objects
    )

    env_kwargs = dict(cfg.env.get("env_kwargs", {}))
    env = MultiAgentTigerEnv(**env_kwargs)

    returns = []
    episode_lengths = []
    total_listens = 0
    correct_openings = 0
    incorrect_openings = 0
    truncations = 0

    for _ in tqdm(range(num_episodes), desc="Evaluating Tiger Episodes"):
        obs_dict, infos = env.reset()
        agent.reset()

        ep_return = 0.0
        steps = 0
        ep_listens = 0
        prev_action = Action(torch.zeros(1, action_dim_i, dtype=torch.float32, device=device))

        while True:
            steps += 1
            curr_obs = obs_dict["agent_0"]
            with torch.no_grad():
                agent.update_belief(curr_obs, prev_action)

            act_container = agent.act(curr_obs)
            act_idx = int(act_container.data.item()) if act_container.data.numel() == 1 else int(torch.argmax(act_container.data, dim=-1).item())

            current_tiger_state = int(env._current_state.data.view(-1)[0].item())

            if act_idx == LISTEN:
                ep_listens += 1
                total_listens += 1
            elif act_idx == OPEN_LEFT:
                if current_tiger_state == TIGER_RIGHT:
                    correct_openings += 1
                else:
                    incorrect_openings += 1
            elif act_idx == OPEN_RIGHT:
                if current_tiger_state == TIGER_LEFT:
                    correct_openings += 1
                else:
                    incorrect_openings += 1

            # Opponent takes canonical listen action (action 0)
            joint_actions = {
                "agent_0": Action(torch.tensor([act_idx], dtype=torch.float32)),
                "agent_1": Action(torch.tensor([0], dtype=torch.float32))
            }

            res = env.step(joint_actions)
            step_rew = float(res.rewards["agent_0"].item())
            ep_return += step_rew

            term = res.terminations["agent_0"].item()
            trunc = res.truncations["agent_0"].item()

            if term or trunc:
                if trunc and not term:
                    truncations += 1
                returns.append(ep_return)
                episode_lengths.append(steps)
                break

            prev_action = Action(torch.zeros(1, action_dim_i, dtype=torch.float32, device=device))
            prev_action.data[0, act_idx] = 1.0
            obs_dict = res.observations

    total_doors = correct_openings + incorrect_openings
    door_acc = float(correct_openings / max(1, total_doors)) * 100.0

    metrics = {
        "num_episodes": num_episodes,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "median_return": float(np.median(returns)),
        "mean_steps": float(np.mean(episode_lengths)),
        "avg_listens_per_ep": float(total_listens / num_episodes),
        "treasure_openings": correct_openings,
        "tiger_penalties": incorrect_openings,
        "door_accuracy_pct": door_acc,
        "treasure_rate_pct": float(correct_openings / num_episodes) * 100.0,
        "tiger_penalty_rate_pct": float(incorrect_openings / num_episodes) * 100.0,
        "truncation_rate_pct": float(truncations / num_episodes) * 100.0,
    }
    return metrics


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("\n" + "=" * 70)
    print("  JEPA I-POMDP : MULTI-AGENT TIGER WORLD PERFORMANCE BENCHMARK ")
    print("=" * 70 + "\n")

    logger = setup_logger("Tiger_Evaluation_Benchmark")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = cfg.env.obs_dim
    action_dim_i = cfg.env.action_dim_i
    action_dim_j = cfg.env.action_dim_j

    extractor = make_extractor(
        cfg.env.extractor_name,
        obs_dim=obs_dim,
        hidden_dim=cfg.model.latent_dim,
        num_objects=cfg.model.num_objects,
        num_blocks=cfg.model.num_blocks
    )
    encoder = RecurrentContextEncoder(
        extractor,
        action_dim=action_dim_i,
        latent_dim=cfg.model.latent_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_blocks=cfg.model.num_blocks
    )
    predictor = CausalRelationalPredictor(
        num_objects=cfg.model.num_objects,
        latent_dim=cfg.model.latent_dim,
        action_dim_i=action_dim_i,
        action_dim_j=action_dim_j,
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
        action_dim_i=action_dim_i,
        action_dim_j=action_dim_j,
        hidden_dim=cfg.model.hidden_dim,
        num_blocks=cfg.model.num_blocks
    ).to(device)

    opponent_head = DiscretePolicyHead(
        latent_dim=cfg.model.latent_dim,
        action_dim=action_dim_j,
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
    if not Path(ckpt_path).exists():
        logger.error(f"Checkpoint not found at {ckpt_path}. Please verify training checkpoint path.")
        sys.exit(1)

    step_loaded = checkpointer.load(ckpt_path, models_dict, device=device)
    logger.info(f"Loaded checkpoint from step {step_loaded}")

    jepa_model.eval()
    value_head.eval()
    reward_head.eval()
    opponent_head.eval()

    num_eps = int(getattr(cfg, "eval_episodes", 100))
    metrics = evaluate_tiger_performance(
        jepa_model=jepa_model,
        value_head=value_head,
        reward_head=reward_head,
        opponent_head=opponent_head,
        cfg=cfg,
        num_episodes=num_eps,
        device=device
    )

    metrics["step_loaded"] = step_loaded
    log_telemetry(logger, "Tiger_Evaluation_Metrics", metrics)

    out_dir = Path("tiger_plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "eval_tiger_results.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 70)
    print(f"  TIGER PERFORMANCE BENCHMARK RESULTS (Step {step_loaded}, N={metrics['num_episodes']})")
    print("=" * 70)
    print(f"  Mean Return:            {metrics['mean_return']:8.2f} +/- {metrics['std_return']:.2f}")
    print(f"  Return Range:           [{metrics['min_return']:.1f}, {metrics['max_return']:.1f}] (Median: {metrics['median_return']:.1f})")
    print(f"  Door Choice Accuracy:   {metrics['door_accuracy_pct']:8.2f}%")
    print(f"  Treasure Open Rate:     {metrics['treasure_rate_pct']:8.2f}% ({metrics['treasure_openings']} episodes)")
    print(f"  Tiger Penalty Rate:     {metrics['tiger_penalty_rate_pct']:8.2f}% ({metrics['tiger_penalties']} episodes)")
    print(f"  Avg Listens per Ep:     {metrics['avg_listens_per_ep']:8.2f}")
    print(f"  Mean Episode Steps:     {metrics['mean_steps']:8.2f}")
    print(f"  Truncation Rate:        {metrics['truncation_rate_pct']:8.2f}%")
    print("=" * 70)
    print(f"  Detailed metrics saved to: {metrics_path}\n")


if __name__ == "__main__":
    main()
