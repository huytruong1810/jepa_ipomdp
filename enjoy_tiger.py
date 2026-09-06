# ABSOLUTE PATH: enjoy_tiger.py
# ==============================================================================
# INTERACTIVE TIGER WALKTHROUGH & VISUAL DIAGNOSTIC CLI
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Canonical Observation Translation:
#    - Translates numerical observation signals ([growl, creak]) into human-readable text.
#
# 2. Strict Causal Context Filtering:
#    - Advances recurrent belief b_t = Filter(b_{t-1}, a_{t-1}, o_t) before querying MCTS.
#
# 3. Dynamic Hardware Execution:
#    - Automatically detects CUDA GPU availability for real-time model evaluation.
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
    MultiAgentTigerEnv, TIGER_LEFT, TIGER_RIGHT, GROWL_LEFT, GROWL_RIGHT, CREAK_LEFT,
    CREAK_RIGHT, OPEN_LEFT, OPEN_RIGHT, SyncVectorEnv
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


def translate_obs(obs_tensor: torch.Tensor) -> str:
    """Translates numeric observation tensor into human-readable text labels."""
    vals = obs_tensor.view(-1).tolist()
    growl, creak = vals[0], vals[1]

    if abs(growl - GROWL_LEFT) < 1e-3:
        growl_str = "Growl from LEFT door"
    elif abs(growl - GROWL_RIGHT) < 1e-3:
        growl_str = "Growl from RIGHT door"
    else:
        growl_str = f"Growl signal: {growl:.2f}"

    if abs(creak - CREAK_LEFT) < 1e-3:
        creak_str = "Left Door CREAKED"
    elif abs(creak - CREAK_RIGHT) < 1e-3:
        creak_str = "Right Door CREAKED"
    else:
        creak_str = "Doors SILENT"

    return f"{growl_str} | {creak_str}"


def translate_action(action_idx: int) -> str:
    """Translates discrete action index into text description."""
    if action_idx == 0:
        return "LISTEN"
    elif action_idx == 1:
        return "OPEN LEFT DOOR"
    elif action_idx == 2:
        return "OPEN RIGHT DOOR"
    return "UNKNOWN"


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("\n" + "=" * 60)
    print("  JEPA I-POMDP : PERSISTENT TIGER WALKTHROUGH ")
    print("=" * 60 + "\n")

    logger = setup_logger("IPOMDP_Inference")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[System] Operating on device: {device}")

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

    value_head = ValueHead(cfg.model.latent_dim, cfg.model.hidden_dim, cfg.model.num_blocks).to(device)
    reward_head = RewardHead(cfg.model.latent_dim, cfg.env.action_dim_i, cfg.env.action_dim_j, cfg.model.hidden_dim, cfg.model.num_blocks).to(device)
    opponent_head = DiscretePolicyHead(cfg.model.latent_dim, cfg.env.action_dim_j, 1, cfg.model.hidden_dim, cfg.model.num_blocks).to(device)

    checkpointer = ModelCheckpointer(f"{cfg.env.name}_checkpoints", logger)
    models_dict = {"jepa": jepa_model, "value": value_head, "reward": reward_head, "opponent": opponent_head}

    try:
        checkpointer.load(f"{cfg.env.name}_checkpoints/latest_checkpoint.pt", models_dict, device=device)
        print("[System] Successfully loaded trained checkpoint.\n")
    except Exception as e:
        print(f"[System] Failed to load checkpoint: {e}\n")

    planner = DiscreteLatentOpenLoopSearch(
        jepa_model, value_head, reward_head, opponent_head,
        cfg.env.action_dim_i, cfg.env.action_dim_j,
        cfg.mcts.num_simulations, cfg.mcts.num_latent_obs
    )

    agents = {
        "agent_0": DiscreteJEPAAgent(
            agent_id="agent_0",
            planner=planner,
            latent_dim=cfg.model.latent_dim,
            action_dim=cfg.env.action_dim_i,
            device=device,
            num_objects=cfg.model.num_objects,
            temperature=0.1
        ),
        "agent_1": StatelessAgent("agent_1", cfg.env.action_dim_j, action_idx=0)
    }

    env_fn = lambda: MultiAgentTigerEnv(max_steps=20)
    vec_env = SyncVectorEnv(env_fn, num_envs=1)

    observations, infos = vec_env.reset()
    for agent in agents.values():
        agent.reset(batch_size=1)

    prev_actions = {
        aid: Action(data=torch.zeros(1, cfg.env.action_dim_i if aid == "agent_0" else cfg.env.action_dim_j, device=device))
        for aid in agents
    }

    step = 0
    done = False
    total_reward = 0.0

    while not done:
        print(f"--- Step {step} -----------------------------------------")

        true_state = int(infos["agent_0"]["true_state"].data.view(-1)[0].item())
        true_state_str = "LEFT" if true_state == TIGER_LEFT else "RIGHT"
        print(f"  [Omniscient Truth] The Tiger is currently behind the {true_state_str} door.")

        obs_tensor_0 = observations["agent_0"].data.view(-1)
        print(f" [Agent 0 Observes]: {translate_obs(obs_tensor_0)}")

        agents["agent_0"].update_belief(observations["agent_0"], prev_actions["agent_0"])
        agents["agent_1"].update_belief(observations["agent_1"], prev_actions["agent_1"])

        time.sleep(0.5)

        actions = {aid: agent.act(observations[aid]) for aid, agent in agents.items()}
        act_0_idx = int(actions["agent_0"].data.view(-1)[0].item())
        act_1_idx = int(actions["agent_1"].data.view(-1)[0].item())

        print(f" [Agent 0 Action]  : Decides to >> {translate_action(act_0_idx)} <<")
        print(f" [Agent 1 Action]  : Decides to >> {translate_action(act_1_idx)} <<")

        next_obs, rews, terms, truncs, infos = vec_env.step(actions)

        reward_0 = float(rews["agent_0"].view(-1)[0].item())
        total_reward += reward_0
        print(f" [Reward]          : {reward_0:+.2f} (Total: {total_reward:+.2f})")

        if act_0_idx in [OPEN_LEFT, OPEN_RIGHT]:
            print("-" * 60)
            if (act_0_idx == OPEN_LEFT and true_state == TIGER_LEFT) or (
                    act_0_idx == OPEN_RIGHT and true_state == TIGER_RIGHT):
                print(" Ouch! Agent 0 was EATEN by the Tiger!")
            else:
                print(" Success! Agent 0 safely found the GOLD!")
            print(" Environment triggers internal reset. Tiger moves to a new location...")
            print("-" * 60)

        if truncs["agent_0"].view(-1)[0].item():
            done = True
            print("\n" + "=" * 60)
            print(f" TURN LIMIT REACHED. Episode Truncated. Final Score: {total_reward:.2f}")
            print("=" * 60 + "\n")

        observations = next_obs
        prev_actions = actions
        step += 1
        time.sleep(1)


if __name__ == "__main__":
    main()