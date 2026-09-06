# ABSOLUTE PATH: enjoy_wumpus.py
# ==============================================================================
# INTERACTIVE WUMPUS WORLD WALKTHROUGH & VISUAL DIAGNOSTIC CLI
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Canonical Percept Translation:
#    - Decodes 5-channel continuous/binary percept vector [Stench, Breeze, Glitter, Bump, Scream]
#      into human-readable terminal indicators.
#
# 2. Strict Causal Context Filtering & Open-Loop Planning:
#    - Advances recurrent belief state b_t = Filter(b_{t-1}, a_{t-1}, o_t) prior to querying
#      DiscreteLatentOpenLoopSearch for action selection a_t ~ pi(a | b_t).
#
# 3. Comprehensive Multi-Agent Interactive State Visualization:
#    - Renders ASCII grid with spatial coordinates, Hunter heading (^, >, v, <), Wumpus heading,
#      pits, gold, active percepts, transition payoffs, and search tree value predictions.
# ==============================================================================

from pathlib import Path
import sys
import time
import hydra
from omegaconf import DictConfig
import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ipomdp.telemetry import setup_logger, ModelCheckpointer, make_extractor
from ipomdp.envs import (
    MultiAgentWumpusEnv,
    FORWARD,
    TURN_LEFT,
    TURN_RIGHT,
    GRAB,
    SHOOT,
    NORTH,
    EAST,
    SOUTH,
    WEST,
    SyncVectorEnv,
)
from ipomdp.models import (
    RecurrentContextEncoder,
    CausalRelationalPredictor,
    RecurrentJEPABase,
    ValueHead,
    RewardHead,
    DiscretePolicyHead,
    TwoHotSymlog,
)

from ipomdp.planning import DiscreteLatentOpenLoopSearch
from ipomdp.agents import DiscreteJEPAAgent, StatelessAgent
from ipomdp.types import Action


def translate_percepts(obs_tensor: torch.Tensor) -> str:
    """Translates 5-channel observation tensor into formatted percept labels."""
    vals = obs_tensor.view(-1).tolist()
    stench = "👃 STENCH" if vals[0] > 0.5 else "  no-stench"
    breeze = "💨 BREEZE" if vals[1] > 0.5 else "  no-breeze"
    glitter = "✨ GLITTER" if vals[2] > 0.5 else "  no-glitter"
    bump = "💥 BUMP" if vals[3] > 0.5 else "  no-bump"
    scream = "😱 SCREAM" if vals[4] > 0.5 else "  no-scream"
    return f"[{stench} | {breeze} | {glitter} | {bump} | {scream}]"


def translate_hunter_action(action_idx: int) -> str:
    """Translates Hunter action index into text."""
    mapping = {
        0: "FORWARD (🚶)",
        1: "TURN LEFT (↺)",
        2: "TURN RIGHT (↻)",
        3: "GRAB GOLD (🏆)",
        4: "SHOOT ARROW (🏹)",
    }
    return mapping.get(int(action_idx), f"UNKNOWN ({action_idx})")


def translate_wumpus_action(action_idx: int) -> str:
    """Translates Wumpus action index into text."""
    mapping = {
        0: "STILL (🛑)",
        1: "FORWARD (🐾)",
        2: "TURN LEFT (↺)",
        3: "TURN RIGHT (↻)",
    }
    return mapping.get(int(action_idx), f"UNKNOWN ({action_idx})")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Ensure env is wumpus if not overridden
    if cfg.env.name != "wumpus":
        print("[Warning] Overriding default env to 'wumpus' for enjoy_wumpus...")
    
    print("\n" + "=" * 70)
    print("  JEPA I-POMDP : INTERACTIVE MULTI-AGENT WUMPUS WORLD WALKTHROUGH ")
    print("=" * 70 + "\n")

    logger = setup_logger("Wumpus_Inference")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[System] Operating on device: {device}")

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
        print(f"[System] Successfully loaded checkpoint from step {step}.\n")
    except Exception as e:
        print(f"[System] Checkpoint not found ({e}). Running with initialized weights.\n")

    jepa_model.eval()
    value_head.eval()
    reward_head.eval()
    opponent_head.eval()

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

    twohot = TwoHotSymlog().to(device)
    env = MultiAgentWumpusEnv(grid_size=4, pit_prob=0.2, max_steps=40)
    obs_dict, infos = env.reset()
    hunter_agent.reset()

    print("=" * 70)
    print("  EPISODE START : INITIAL CAVE STATE")
    print("=" * 70)
    print(env.render({}, None))
    print(f"Hunter Initial Percepts: {translate_percepts(obs_dict['agent_0'].data)}")
    print("-" * 70)

    cumulative_reward = 0.0
    prev_action = Action(torch.zeros(1, action_dim_i, dtype=torch.float32, device=device))

    for step_num in range(1, 41):
        # 1. Update belief with current observation
        curr_obs = obs_dict["agent_0"]
        with torch.no_grad():
            hunter_agent.update_belief(curr_obs, prev_action)
            b_t = hunter_agent.belief
            val_est = twohot.decode(value_head(b_t), real_scale=True).item()
            opp_logits = opponent_head(b_t)
            opp_probs = torch.softmax(opp_logits, dim=-1).squeeze().tolist()

        # 2. Plan action using Latent MCTS
        a_hunter = hunter_agent.act(curr_obs)
        act_hunter_idx = int(a_hunter.data.item()) if a_hunter.data.numel() == 1 else int(torch.argmax(a_hunter.data, dim=-1).item())

        # Canonical stationary Wumpus action (STILL: 0)
        act_wumpus_idx = WUMPUS_STILL
        a_wumpus = Action(torch.tensor([act_wumpus_idx], dtype=torch.float32))

        # 3. Environment Step
        joint_actions = {
            "agent_0": Action(torch.tensor([act_hunter_idx], dtype=torch.float32)),
            "agent_1": a_wumpus
        }
        step_result = env.step(joint_actions)

        r_hunter = step_result.rewards["agent_0"].item()
        cumulative_reward += r_hunter
        term = step_result.terminations["agent_0"].item()
        trunc = step_result.truncations["agent_0"].item()

        print(f"\n--- STEP {step_num} ---")
        print(f"Hunter Action: {translate_hunter_action(act_hunter_idx)} | Wumpus Action: {translate_wumpus_action(act_wumpus_idx)}")
        print(f"Latent Value Estimate V(b_t): {val_est:+.2f}")
        print(f"Predicted Opponent Policy Prior: Still={opp_probs[0]:.2f}, Fwd={opp_probs[1]:.2f}, Left={opp_probs[2]:.2f}, Right={opp_probs[3]:.2f}")
        print(f"Percepts Emitted: {translate_percepts(step_result.observations['agent_0'].data)}")
        print(f"Step Reward: {r_hunter:+.1f} | Cumulative Return: {cumulative_reward:+.1f}")
        print(env.render(joint_actions, step_result))

        if term or trunc:
            status_str = "🏆 VICTORY (Gold Grabbed!)" if env._hunter_has_gold else "💀 DEFEAT (Hunter Died)"
            if trunc and not term:
                status_str = "⏱️ TRUNCATED (Max Steps Reached)"
            print("\n" + "=" * 70)
            print(f"  EPISODE CONCLUDED: {status_str}")
            print(f"  Final Cumulative Return: {cumulative_reward:+.1f} across {step_num} steps")
            print("=" * 70 + "\n")
            break

        prev_action = Action(Action.to_one_hot(act_hunter_idx, action_dim_i, device=device))
        obs_dict = step_result.observations
        time.sleep(0.5)



if __name__ == "__main__":
    main()
