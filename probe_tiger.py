# ABSOLUTE PATH: probe_tiger.py
# ==============================================================================
# RIGOROUS SEMANTIC PROBE & MECHANISTIC AMNESIA VERIFICATION
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Mechanistic Amnesia Verification:
#    - Probes the recurrent belief state immediately following door reset transitions
#      to verify that prior accumulated listening evidence is completely dumped:
#         H(P_probe(s | b_post_reset)) ≈ ln(2) ≈ 0.693 nats
#
# 2. Geometric Origin Reset Distance (L2 Norm):
#    - Evaluates Euclidean distance from initial uniform belief vector b_0:
#         ||b_t - b_0||_2
#      demonstrating geometric contraction back to the prior origin after opening doors.
#      Post-reset belief resides at beliefs[7] following the OPEN action update (step 6).
#
# 3. Action-Unbiased Linear Probe Dataset:
#    - Samples belief representations across all discrete actions (LISTEN, OPEN_LEFT,
#      OPEN_RIGHT) to prevent dataset distribution shift during probe training.
#
# 4. Dynamic Hardware Device Selection & Truncation Isolation:
#    - Dynamically targets CUDA or CPU hardware based on device availability.
#    - Re-initializes belief and previous action tensors to zero upon episode truncation.
# ==============================================================================

from pathlib import Path
import sys
import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ipomdp.telemetry import (
    setup_logger,
    ModelCheckpointer,
    make_extractor,
    LatentSpaceVisualizer,
)
from ipomdp.envs import (
    MultiAgentTigerEnv,
    LISTEN,
    OPEN_LEFT,
    OPEN_RIGHT,
    GROWL_LEFT,
    GROWL_RIGHT,
    SILENCE,
)
from ipomdp.models import (
    RecurrentContextEncoder,
    CausalRelationalPredictor,
    RecurrentJEPABase,
)
from ipomdp.types import Action, Observation


class TigerLinearProbe(nn.Module):
    """Linear classifier probing latent belief state representations to predict true tiger position."""

    def __init__(self, latent_dim: int, num_objects: int):
        super().__init__()
        self.net = nn.Linear(latent_dim * num_objects, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(x.size(0), -1)
        return self.net(x_flat)


def gather_probe_dataset(env, jepa_model, cfg, num_steps=3000, device=None):
    """Gathers belief representations across all action choices to prevent training dataset shift."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    beliefs, labels = [], []

    obs, infos = env.reset()
    b_curr = torch.zeros(1, cfg.model.num_objects, cfg.model.latent_dim, device=device)
    prev_a = Action(data=torch.zeros(1, cfg.env.action_dim_i, device=device))

    for _ in range(num_steps):
        true_state = int(infos["agent_0"]["true_state"].data.view(-1)[0].item())
        act_idx = torch.randint(0, cfg.env.action_dim_i, (1,)).item()

        actions = {
            "agent_0": Action(data=torch.tensor([act_idx], dtype=torch.float32)),
            "agent_1": Action(data=torch.tensor([LISTEN], dtype=torch.float32))
        }

        a_i_onehot = Action(data=F.one_hot(torch.tensor([act_idx]), num_classes=cfg.env.action_dim_i).float().to(device))
        obs_obj = Observation(data=obs["agent_0"].data.view(1, -1).to(device))

        b_curr = jepa_model.encode_context(obs_obj.data, prev_a.data, b_curr)

        beliefs.append(b_curr.detach())
        labels.append(true_state)

        prev_a = a_i_onehot
        obs, _, terminations, truncations, infos = env.step(actions)

        # Truncation/Termination Guard: Re-initialize recurrent state on episode boundary
        if terminations["agent_0"].item() or truncations["agent_0"].item():
            obs, infos = env.reset()
            b_curr = torch.zeros(1, cfg.model.num_objects, cfg.model.latent_dim, device=device)
            prev_a = Action(data=torch.zeros(1, cfg.env.action_dim_i, device=device))

    return torch.cat(beliefs, dim=0), torch.tensor(labels, dtype=torch.long, device=device)


def generate_forced_trajectory(jepa_model, cfg, actions, observations, name, color, device):
    """Executes a forced sequence of actions and observations to project semantic paths into PCA space."""
    b_curr = torch.zeros(1, cfg.model.num_objects, cfg.model.latent_dim, device=device)
    prev_a = Action(data=torch.zeros(1, cfg.env.action_dim_i, device=device))
    beliefs = [b_curr.detach()]

    for a, o in zip(actions, observations):
        obs_tensor = torch.tensor(o, dtype=torch.float32, device=device).view(1, -1)
        b_curr = jepa_model.encode_context(obs_tensor, prev_a.data, b_curr)
        beliefs.append(b_curr.detach())
        prev_a = Action(data=F.one_hot(torch.tensor([a]), num_classes=cfg.env.action_dim_i).float().to(device))

    return {"name": name, "beliefs": beliefs, "color": color}


def run_semantic_trajectory_analysis(jepa_model, cfg, logger, device):
    """Plots semantic trajectory divergence and analyzes belief reset behavior when opening doors."""
    logger.info("Generating Forced Trajectories for Latent Space PCA...")

    traj_left = generate_forced_trajectory(
        jepa_model, cfg, actions=[LISTEN] * 10, observations=[[GROWL_LEFT, SILENCE]] * 10,
        name="10x Hear Left (Certainty Left)", color="blue", device=device
    )

    traj_right = generate_forced_trajectory(
        jepa_model, cfg, actions=[LISTEN] * 10, observations=[[GROWL_RIGHT, SILENCE]] * 10,
        name="10x Hear Right (Certainty Right)", color="red", device=device
    )

    traj_alt = generate_forced_trajectory(
        jepa_model, cfg, actions=[LISTEN] * 10, observations=[[GROWL_LEFT, SILENCE], [GROWL_RIGHT, SILENCE]] * 5,
        name="Alternating L/R (Ambiguity)", color="purple", device=device
    )

    traj_listen_left_open = generate_forced_trajectory(
        jepa_model, cfg,
        actions=[LISTEN] * 5 + [OPEN_LEFT] + [LISTEN] * 4,
        observations=[[GROWL_LEFT, SILENCE]] * 6 + [[GROWL_RIGHT, SILENCE]] * 4,
        name="5x Hear Left -> OPEN_LEFT -> 4x Hear Right", color="green", device=device
    )

    traj_listen_right_open = generate_forced_trajectory(
        jepa_model, cfg,
        actions=[LISTEN] * 5 + [OPEN_RIGHT] + [LISTEN] * 4,
        observations=[[GROWL_RIGHT, SILENCE]] * 6 + [[GROWL_LEFT, SILENCE]] * 4,
        name="5x Hear Right -> OPEN_RIGHT -> 4x Hear Left", color="orange", device=device
    )

    # Correct Post-Reset Index Alignment: Index 7 is the true post-open belief vector
    b_0 = traj_listen_left_open["beliefs"][0]
    b_pre_open_4 = traj_listen_left_open["beliefs"][5]
    b_post_open_4 = traj_listen_left_open["beliefs"][7]
    dist_pre_4 = float(torch.norm(b_pre_open_4 - b_0).item())
    dist_post_4 = float(torch.norm(b_post_open_4 - b_0).item())

    b_pre_open_5 = traj_listen_right_open["beliefs"][5]
    b_post_open_5 = traj_listen_right_open["beliefs"][7]
    dist_pre_5 = float(torch.norm(b_pre_open_5 - b_0).item())
    dist_post_5 = float(torch.norm(b_post_open_5 - b_0).item())

    logger.info("=== BELIEF AMNESIA ORIGIN RESET DISTANCE ANALYSIS ===")
    logger.info(f"[Traj Left -> Open] Pre-Open L2 Dist: {dist_pre_4:.4f} | Post-Open L2 Dist: {dist_post_4:.4f}")
    logger.info(f"[Traj Right -> Open] Pre-Open L2 Dist: {dist_pre_5:.4f} | Post-Open L2 Dist: {dist_post_5:.4f}")

    visualizer = LatentSpaceVisualizer(save_dir=f"{cfg.env.name}_plots/probe")

    visualizer.plot_multiple_trajectories(
        trajectories=[traj_left, traj_right, traj_alt, traj_listen_left_open, traj_listen_right_open],
        filename="holistic_semantic_trajectories",
        value_head=None,
        n_components=2
    )

    visualizer.plot_multiple_trajectories(
        trajectories=[traj_left, traj_right, traj_alt, traj_listen_left_open, traj_listen_right_open],
        filename="holistic_semantic_trajectories_1d",
        value_head=None,
        n_components=1
    )

    logger.info(f"Holistic Latent Trajectory plots saved to '{cfg.env.name}_plots/probe/'")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("=== CAUSAL-JEPA RIGOROUS SEMANTICS PROBE ===")
    logger = setup_logger("Tiger_Probe")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Probe Execution Device: {device}")

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

    checkpointer = ModelCheckpointer(f"{cfg.env.name}_checkpoints", logger)
    try:
        checkpointer.load(f"{cfg.env.name}_checkpoints/latest_checkpoint.pt", {"jepa": jepa_model}, device=device)
        logger.info("Successfully loaded pre-trained model for probing.")
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")

    run_semantic_trajectory_analysis(jepa_model, cfg, logger, device)

    logger.info("Gathering rollout dataset for Linear Probe training...")
    env = MultiAgentTigerEnv()
    X_train, y_train = gather_probe_dataset(env, jepa_model, cfg, num_steps=3000, device=device)

    probe = TigerLinearProbe(latent_dim=cfg.model.latent_dim, num_objects=cfg.model.num_objects).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    logger.info("Training Linear Classifier Probe on Latent Space...")
    for epoch in range(100):
        optimizer.zero_grad()
        logits = probe(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    logger.info("Executing Mechanistic Amnesia Test (Post-Reset Entropy)")

    obs, infos = env.reset()
    b_curr = torch.zeros(1, cfg.model.num_objects, cfg.model.latent_dim, device=device)
    prev_a = Action(data=torch.zeros(1, cfg.env.action_dim_i, device=device))
    entropies = []

    for _ in range(5):
        a_i = Action(data=F.one_hot(torch.tensor([LISTEN]), num_classes=cfg.env.action_dim_i).float().to(device))
        obs_tensor = obs["agent_0"].data.view(1, -1).to(device)

        b_curr = jepa_model.encode_context(obs_tensor, prev_a.data, b_curr)

        prob = F.softmax(probe(b_curr), dim=-1)
        entropy = -torch.sum(prob * torch.log(prob + 1e-8)).item()
        entropies.append(float(entropy))

        prev_a = a_i
        actions = {
            "agent_0": Action(data=torch.tensor([LISTEN], dtype=torch.float32)),
            "agent_1": Action(data=torch.tensor([LISTEN], dtype=torch.float32))
        }

        obs, _, terminations, truncations, infos = env.step(actions)
        if terminations["agent_0"].item() or truncations["agent_0"].item():
            obs, infos = env.reset()
            b_curr.zero_()
            prev_a = Action(data=torch.zeros(1, cfg.env.action_dim_i, device=device))

    actions = {
        "agent_0": Action(data=torch.tensor([OPEN_LEFT], dtype=torch.float32)),
        "agent_1": Action(data=torch.tensor([LISTEN], dtype=torch.float32))
    }
    a_i = Action(data=F.one_hot(torch.tensor([OPEN_LEFT]), num_classes=cfg.env.action_dim_i).float().to(device))
    obs, _, _, _, infos = env.step(actions)

    obs_tensor = obs["agent_0"].data.view(1, -1).to(device)
    b_curr = jepa_model.encode_context(obs_tensor, a_i.data, b_curr)

    prob = F.softmax(probe(b_curr), dim=-1)
    post_reset_entropy = float(-torch.sum(prob * torch.log(prob + 1e-8)).item())
    entropies.append(post_reset_entropy)

    logger.info(f"Entropy Trajectory (Last step is Post-Reset): {entropies}")


if __name__ == "__main__":
    main()