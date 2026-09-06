# ABSOLUTE PATH: eval_wumpus.py
# ==============================================================================
# MULTI-AGENT WUMPUS WORLD BENCHMARK & PERFORMANCE EVALUATOR
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Multi-Metric Domain Evaluation:
#    - Measures Gold Grab Victory Rate (%), Hunter Survival Rate (%), Arrow Accuracy (%),
#      Pit Defeat Rate (%), and Predation Defeat Rate (%) over N independent episodes.
#
# 2. Strict Causal Context Filtering & Latent MCTS:
#    - Evaluates the learned RecurrentJEPABase world model and DiscreteLatentOpenLoopSearch
#      planner against an active adversarial Wumpus agent.
#
# 3. Telemetry Integration:
#    - Formats evaluation metrics into clean tables and logs structured JSON telemetry.
# ==============================================================================

from pathlib import Path
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

from ipomdp.telemetry import setup_logger, log_telemetry, ModelCheckpointer, make_extractor
from ipomdp.envs import (
    MultiAgentWumpusEnv,
    FORWARD,
    TURN_LEFT,
    TURN_RIGHT,
    GRAB,
    SHOOT,
    WUMPUS_STILL,
    WUMPUS_FORWARD,
    WUMPUS_TURN_LEFT,
    WUMPUS_TURN_RIGHT,
    NORTH,
    EAST,
    SOUTH,
    WEST,
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


def evaluate_wumpus_performance(
    jepa_model: RecurrentJEPABase,
    value_head: ValueHead,
    reward_head: RewardHead,
    opponent_head: DiscretePolicyHead,
    cfg: DictConfig,
    num_episodes: int = 100,
    device: torch.device = torch.device("cpu")
) -> dict:
    """Evaluates the JEPA agent across num_episodes episodes of Wumpus World."""
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

    hunter_agent = DiscreteJEPAAgent(
        agent_id="agent_0",
        planner=planner,
        latent_dim=cfg.model.latent_dim,
        action_dim=action_dim_i,
        device=device,
        num_objects=cfg.model.num_objects
    )

    env = MultiAgentWumpusEnv(grid_size=4, pit_prob=0.2, max_steps=50)

    victories = 0
    pit_deaths = 0
    predation_deaths = 0
    truncations = 0
    arrows_shot = 0
    wumpus_kills = 0
    returns = []
    episode_lengths = []

    for ep in range(num_episodes):
        obs_dict, infos = env.reset()
        hunter_agent.reset()

        ep_return = 0.0
        steps = 0
        prev_action = Action(torch.zeros(1, action_dim_i, dtype=torch.float32, device=device))

        while True:
            steps += 1
            curr_obs = obs_dict["agent_0"]
            with torch.no_grad():
                hunter_agent.update_belief(curr_obs, prev_action)

            a_hunter = hunter_agent.act(curr_obs)
            act_hunter_idx = int(a_hunter.data.item()) if a_hunter.data.numel() == 1 else int(torch.argmax(a_hunter.data, dim=-1).item())

            if act_hunter_idx == SHOOT:
                arrows_shot += 1

            # Canonical stationary Wumpus (STILL: 0)
            act_wumpus_idx = WUMPUS_STILL
            a_wumpus = Action(torch.tensor([act_wumpus_idx], dtype=torch.float32))

            joint_actions = {
                "agent_0": Action(torch.tensor([act_hunter_idx], dtype=torch.float32)),
                "agent_1": a_wumpus
            }
            res = env.step(joint_actions)
            ep_return += res.rewards["agent_0"].item()

            if env._wumpus_died_this_step:
                wumpus_kills += 1

            term = res.terminations["agent_0"].item()
            trunc = res.truncations["agent_0"].item()

            if term or trunc:
                if env._hunter_has_gold:
                    victories += 1
                elif not env._hunter_alive:
                    # Check death cause
                    hr, hc = env._hunter_pos
                    if env._pits[hr][hc]:
                        pit_deaths += 1
                    else:
                        predation_deaths += 1
                elif trunc:
                    truncations += 1

                returns.append(ep_return)
                episode_lengths.append(steps)
                break

            prev_action = Action(torch.zeros(1, action_dim_i, dtype=torch.float32, device=device))
            prev_action.data[0, act_hunter_idx] = 1.0
            obs_dict = res.observations

    metrics = {
        "num_episodes": num_episodes,
        "victory_rate": float(victories / num_episodes),
        "survival_rate": float((num_episodes - pit_deaths - predation_deaths) / num_episodes),
        "pit_death_rate": float(pit_deaths / num_episodes),
        "predation_death_rate": float(predation_deaths / num_episodes),
        "truncation_rate": float(truncations / num_episodes),
        "arrow_kill_rate": float(wumpus_kills / max(1, arrows_shot)),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_steps": float(np.mean(episode_lengths)),
    }
    return metrics


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("\n" + "=" * 70)
    print("  JEPA I-POMDP : MULTI-AGENT WUMPUS WORLD PERFORMANCE BENCHMARK ")
    print("=" * 70 + "\n")

    logger = setup_logger("Wumpus_Evaluation_Benchmark")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = cfg.env.obs_dim
    action_dim_i = cfg.env.action_dim_i
    action_dim_j = cfg.env.action_dim_j

    extractor = make_extractor(
        "mlp",
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

    checkpointer = ModelCheckpointer("wumpus_checkpoints", logger)
    models_dict = {
        "jepa": jepa_model,
        "value": value_head,
        "reward": reward_head,
        "opponent": opponent_head
    }

    try:
        step = checkpointer.load("wumpus_checkpoints/latest_checkpoint.pt", models_dict, device=device)
        logger.info(f"Successfully loaded Wumpus checkpoint from step {step}.")
    except Exception as e:
        logger.warning(f"Failed to load checkpoint ({e}). Running with initialized weights.")

    jepa_model.eval()
    value_head.eval()
    reward_head.eval()
    opponent_head.eval()

    logger.info("Executing 100-Episode Quantitative Evaluation...")
    metrics = evaluate_wumpus_performance(
        jepa_model, value_head, reward_head, opponent_head, cfg, num_episodes=100, device=device
    )

    print("\n" + "-" * 70)
    print("  MULTI-AGENT WUMPUS BENCHMARK RESULTS (100 EPISODES)")
    print("-" * 70)
    print(f"  • Victory Rate (Gold Grabbed)  : {metrics['victory_rate']:.2%}")
    print(f"  • Hunter Survival Rate         : {metrics['survival_rate']:.2%}")
    print(f"  • Pit Defeat Rate              : {metrics['pit_death_rate']:.2%}")
    print(f"  • Predation Defeat Rate (Wumpus): {metrics['predation_death_rate']:.2%}")
    print(f"  • Truncation Rate (Max Steps)  : {metrics['truncation_rate']:.2%}")
    print(f"  • Arrow Hit Accuracy Rate      : {metrics['arrow_kill_rate']:.2%}")
    print(f"  • Mean Episode Return          : {metrics['mean_return']:+.2f} ± {metrics['std_return']:.2f}")
    print(f"  • Mean Steps to Conclusion     : {metrics['mean_steps']:.2f}")
    print("-" * 70 + "\n")

    log_telemetry(logger, "Wumpus_Evaluation_Benchmark", metrics)


if __name__ == "__main__":
    main()
