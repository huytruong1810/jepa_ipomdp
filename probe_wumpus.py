# ABSOLUTE PATH: probe_wumpus.py
# ==============================================================================
# SPATIAL COORDINATE & SEMANTIC LATENT PROBE FOR WUMPUS WORLD
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Non-Intrusive Latent Space Probing:
#    - Trains linear classifier heads on frozen recurrent belief states b_t to determine
#      whether the learned embeddings linearly encode spatial coordinates (r_h, c_h),
#      (r_w, c_w), and gold acquisition state without explicit reconstruction supervision.
#
# 2. PCA Trajectory Projection:
#    - Reduces high-dimensional object-token embeddings into 2D principal components to
#      visualize spatial exploration manifolds and cave boundary interactions.
# ==============================================================================

from pathlib import Path
import sys
import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ipomdp.telemetry import setup_logger, ModelCheckpointer, make_extractor, log_telemetry
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
)
from ipomdp.models import (
    RecurrentContextEncoder,
    CausalRelationalPredictor,
    RecurrentJEPABase,
)
from ipomdp.types import Action, Observation


class SpatialLinearProbe(nn.Module):
    """Linear probe to predict discrete spatial coordinate classes (0..N-1)."""
    def __init__(self, in_features: int, num_classes: int = 4):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def collect_wumpus_probe_data(
    env: MultiAgentWumpusEnv,
    jepa_model: RecurrentJEPABase,
    num_steps: int = 4000,
    device: torch.device = torch.device("cpu")
):
    """Collects paired latent belief states and ground-truth spatial coordinates."""
    b_states = []
    hunter_rows = []
    hunter_cols = []
    wumpus_rows = []
    wumpus_cols = []
    gold_status = []

    obs_dict, infos = env.reset()
    b_t = torch.zeros(1, 2, 32, device=device)
    prev_a = Action(torch.zeros(1, 5, dtype=torch.float32, device=device))

    for _ in range(num_steps):
        curr_obs = obs_dict["agent_0"]
        with torch.no_grad():
            b_t = jepa_model.context_encoder(curr_obs.data.unsqueeze(0).to(device), prev_a.data.to(device), b_t)

        flat_b = b_t.view(1, -1).cpu().squeeze(0)
        b_states.append(flat_b)
        hunter_rows.append(env._hunter_pos[0])
        hunter_cols.append(env._hunter_pos[1])
        wumpus_rows.append(env._wumpus_pos[0])
        wumpus_cols.append(env._wumpus_pos[1])
        gold_status.append(1 if env._hunter_has_gold else 0)

        # Step environment: Hunter random exploration, Wumpus stationary (STILL: 0)
        a_h = int(torch.randint(0, 5, (1,)).item())
        a_w = WUMPUS_STILL
        res = env.step({
            "agent_0": Action(torch.tensor([a_h], dtype=torch.float32)),
            "agent_1": Action(torch.tensor([a_w], dtype=torch.float32)),
        })

        if res.terminations["agent_0"].item() or res.truncations["agent_0"].item():
            obs_dict, infos = env.reset()
            b_t = torch.zeros(1, 2, 32, device=device)
            prev_a = Action(torch.zeros(1, 5, dtype=torch.float32, device=device))
        else:
            prev_a = Action(torch.zeros(1, 5, dtype=torch.float32, device=device))
            prev_a.data[0, a_h] = 1.0
            obs_dict = res.observations

    return (
        torch.stack(b_states),
        torch.tensor(hunter_rows, dtype=torch.long),
        torch.tensor(hunter_cols, dtype=torch.long),
        torch.tensor(wumpus_rows, dtype=torch.long),
        torch.tensor(wumpus_cols, dtype=torch.long),
        torch.tensor(gold_status, dtype=torch.long),
    )


def train_probe(x: torch.Tensor, y: torch.Tensor, num_classes: int = 4, epochs: int = 40) -> float:
    """Trains a linear probe with cross entropy loss and returns top-1 accuracy."""
    split = int(0.8 * len(x))
    x_tr, y_tr = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]

    probe = SpatialLinearProbe(in_features=x.shape[1], num_classes=num_classes)
    optimizer = optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        probe.train()
        optimizer.zero_grad()
        logits = probe(x_tr)
        loss = criterion(logits, y_tr)
        loss.backward()
        optimizer.step()

    probe.eval()
    with torch.no_grad():
        val_logits = probe(x_val)
        preds = torch.argmax(val_logits, dim=-1)
        acc = (preds == y_val).float().mean().item()
    return float(acc)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("\n" + "=" * 70)
    print("  JEPA I-POMDP : WUMPUS WORLD SPATIAL & SEMANTIC LATENT PROBE ")
    print("=" * 70 + "\n")

    logger = setup_logger("Wumpus_Probe")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Probe Execution Device: {device}")

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

    checkpointer = ModelCheckpointer("wumpus_checkpoints", logger)
    try:
        step = checkpointer.load("wumpus_checkpoints/latest_checkpoint.pt", {"jepa": jepa_model}, device=device)
        logger.info(f"Successfully loaded Wumpus checkpoint from step {step}.")
    except Exception as e:
        logger.warning(f"Failed to load checkpoint ({e}). Running with initialized weights.")

    jepa_model.eval()
    env = MultiAgentWumpusEnv(grid_size=4, pit_prob=0.2, max_steps=50)

    logger.info("Collecting latent belief trajectories for spatial probing...")
    b_states, h_r, h_c, w_r, w_c, g_s = collect_wumpus_probe_data(env, jepa_model, num_steps=3000, device=device)

    logger.info("Training linear spatial probes on frozen embeddings...")
    acc_hr = train_probe(b_states, h_r, num_classes=4)
    acc_hc = train_probe(b_states, h_c, num_classes=4)
    acc_wr = train_probe(b_states, w_r, num_classes=4)
    acc_wc = train_probe(b_states, w_c, num_classes=4)
    acc_gold = train_probe(b_states, g_s, num_classes=2)

    print("\n" + "-" * 70)
    print("  WUMPUS WORLD LATENT PROBE DECISION ACCURACIES")
    print("-" * 70)
    print(f"  • Hunter Row Coordinate Decodability : {acc_hr:.2%} (Random Chance: 25.0%)")
    print(f"  • Hunter Col Coordinate Decodability : {acc_hc:.2%} (Random Chance: 25.0%)")
    print(f"  • Wumpus Row Coordinate Decodability : {acc_wr:.2%} (Random Chance: 25.0%)")
    print(f"  • Wumpus Col Coordinate Decodability : {acc_wc:.2%} (Random Chance: 25.0%)")
    print(f"  • Gold Acquisition Decodability      : {acc_gold:.2%} (Random Chance: 50.0%)")
    print("-" * 70)

    # PCA Trajectory Plot
    save_dir = Path("wumpus_plots/probe")
    save_dir.mkdir(parents=True, exist_ok=True)
    pca = PCA(n_components=2)
    b_pca = pca.fit_transform(b_states.numpy()[:500])

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(b_pca[:, 0], b_pca[:, 1], c=h_r.numpy()[:500], cmap="viridis", alpha=0.8)
    plt.colorbar(scatter, label="Hunter Row Coordinate")
    plt.title("Wumpus Latent Belief Space 2D PCA Manifold")
    plt.xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    plt.ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / "wumpus_latent_pca.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved PCA manifold visualization to '{save_dir / 'wumpus_latent_pca.png'}'.")

    log_telemetry(logger, "Wumpus_Spatial_Latent_Probe", {
        "acc_hunter_row": acc_hr,
        "acc_hunter_col": acc_hc,
        "acc_wumpus_row": acc_wr,
        "acc_wumpus_col": acc_wc,
        "acc_gold": acc_gold,
    })


if __name__ == "__main__":
    main()
