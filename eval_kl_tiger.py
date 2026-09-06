# ABSOLUTE PATH: eval_kl_tiger.py
# ==============================================================================
# ORACLE BENCHMARK & INFORMATION-THEORETIC KL DIVERGENCE EVALUATOR
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Non-Intrusive Offline Observation Probe:
#    - Trains ObservationProbeHead P_theta(o_{t+1} | b_t, a_t) directly on frozen
#      JEPA latent belief states b_t and action context a_t without modifying
#      world model representations.
#
# 2. Complete 6-Class Marginalization:
#    - Maps categorical predictions into canonical binary growl probabilities:
#         P(GROWL_LEFT)  = P(class 0) + P(class 2) + P(class 3)
#         P(GROWL_RIGHT) = P(class 1) + P(class 4) + P(class 5)
#      accounting for joint growl-creak signal interactions.
#
# 3. Dual-Regime Evidence Accumulation Trajectories:
#    - Rigorously tracks Bayesian belief convergence across both physical regimes:
#         * Tiger-Left Regime  (s = TL): Evaluates P(GL) convergence (50.0% -> 85.0%)
#         * Tiger-Right Regime (s = TR): Evaluates P(GR) convergence (50.0% -> 85.0%)
# ==============================================================================

from pathlib import Path
import sys
import hydra
import numpy as np
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
    log_telemetry,
    ModelCheckpointer,
    make_extractor,
    compute_distribution_kl_divergence,
    compute_observation_accuracy_metrics,
)
from ipomdp.envs import (
    MultiAgentTigerEnv,
    TigerBayesianOracle,
    encode_observation_to_index,
    NUM_OBS_CLASSES,
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
    ObservationProbeHead,
)
from ipomdp.types import Action, Observation


def collect_probe_training_data(env, jepa_model, cfg, num_steps: int = 8000, device=None):
    """Collects rollout pairs ((b_t, a_t), true_obs_{t+1}) for observation probing."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    beliefs, action_onehots, target_obs_indices = [], [], []

    obs, infos = env.reset()
    b_curr = torch.zeros(1, cfg.model.num_objects, cfg.model.latent_dim, device=device)
    prev_a = Action(data=torch.zeros(1, cfg.env.action_dim_i, device=device))

    for _ in range(num_steps):
        if torch.rand(1).item() < 0.8:
            act_idx = LISTEN
        else:
            act_idx = OPEN_LEFT if torch.rand(1).item() < 0.5 else OPEN_RIGHT

        actions = {
            "agent_0": Action(data=torch.tensor([act_idx], dtype=torch.float32)),
            "agent_1": Action(data=torch.tensor([LISTEN], dtype=torch.float32))
        }

        a_i_onehot = F.one_hot(torch.tensor([act_idx]), num_classes=cfg.env.action_dim_i).float().to(device)
        obs_obj = obs["agent_0"].data.view(1, -1).to(device)

        b_curr = jepa_model.encode_context(obs_obj, prev_a.data, b_curr)

        next_obs, _, terminations, truncations, infos = env.step(actions)
        obs_idx = encode_observation_to_index(next_obs["agent_0"].data)

        beliefs.append(b_curr.detach())
        action_onehots.append(a_i_onehot.detach())
        target_obs_indices.append(obs_idx)

        prev_a = Action(data=a_i_onehot)
        obs = next_obs

        if terminations["agent_0"].item() or truncations["agent_0"].item():
            obs, infos = env.reset()
            b_curr = torch.zeros(1, cfg.model.num_objects, cfg.model.latent_dim, device=device)
            prev_a = Action(data=torch.zeros(1, cfg.env.action_dim_i, device=device))

    return torch.cat(beliefs, dim=0), torch.cat(action_onehots, dim=0), torch.tensor(target_obs_indices, dtype=torch.long, device=device)


def train_observation_probe(probe, b_train, a_train, y_train, epochs: int = 60, batch_size: int = 128):
    """Trains Action-Conditioned ObservationProbeHead on frozen JEPA representations."""
    optimizer = optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    dataset_size = b_train.size(0)

    probe.train()
    for _ in range(epochs):
        perm = torch.randperm(dataset_size)
        for i in range(0, dataset_size, batch_size):
            indices = perm[i:i + batch_size]
            batch_b, batch_a, batch_y = b_train[indices], a_train[indices], y_train[indices]

            optimizer.zero_grad()
            logits = probe(batch_b, batch_a)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        preds = torch.argmax(probe(b_train, a_train), dim=-1)
        acc = float((preds == y_train).float().mean().item())
    return acc


def evaluate_fidelity_and_accuracy(jepa_model, obs_probe, cfg, logger, num_episodes: int = 100, device=None):
    """Executes side-by-side rollouts of JEPA Agent and Analytical Bayesian Oracle."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = MultiAgentTigerEnv()
    oracle = TigerBayesianOracle(growl_accuracy=0.85, creak_accuracy=1.0)

    kl_overall, kl_listen, kl_open = [], [], []
    acc_oracle_overall, acc_jepa_overall = [], []
    acc_oracle_listen, acc_jepa_listen = [], []
    acc_oracle_open, acc_jepa_open = [], []

    for _ in range(num_episodes):
        obs, infos = env.reset()
        oracle.reset()

        b_curr = torch.zeros(1, cfg.model.num_objects, cfg.model.latent_dim, device=device)
        prev_a = Action(data=torch.zeros(1, cfg.env.action_dim_i, device=device))

        for step in range(cfg.env.env_kwargs.max_steps):
            if torch.rand(1).item() < 0.8:
                act_idx = LISTEN
            else:
                act_idx = OPEN_LEFT if torch.rand(1).item() < 0.5 else OPEN_RIGHT

            a_i_onehot = F.one_hot(torch.tensor([act_idx]), num_classes=cfg.env.action_dim_i).float().to(device)

            p_true = oracle.get_exact_observation_distribution(action=act_idx, opponent_action=LISTEN).to(device)
            b_curr = jepa_model.encode_context(obs["agent_0"].data.view(1, -1).to(device), prev_a.data, b_curr)

            with torch.no_grad():
                probe_logits = obs_probe(b_curr, a_i_onehot)
                p_pred = F.softmax(probe_logits, dim=-1).squeeze(0)

            actions = {
                "agent_0": Action(data=torch.tensor([act_idx], dtype=torch.float32)),
                "agent_1": Action(data=torch.tensor([LISTEN], dtype=torch.float32))
            }
            next_obs, _, terminations, truncations, _ = env.step(actions)
            true_obs_idx = torch.tensor([encode_observation_to_index(next_obs["agent_0"].data)], dtype=torch.long, device=device)

            kl_val, _ = compute_distribution_kl_divergence(p_true.unsqueeze(0), p_pred.unsqueeze(0))
            step_kl = float(kl_val.item())

            acc_metrics = compute_observation_accuracy_metrics(p_true.unsqueeze(0), p_pred.unsqueeze(0), true_obs_idx)

            kl_overall.append(step_kl)
            acc_oracle_overall.append(acc_metrics["acc_oracle"])
            acc_jepa_overall.append(acc_metrics["acc_jepa"])

            if act_idx == LISTEN:
                kl_listen.append(step_kl)
                acc_oracle_listen.append(acc_metrics["acc_oracle"])
                acc_jepa_listen.append(acc_metrics["acc_jepa"])
            else:
                kl_open.append(step_kl)
                acc_oracle_open.append(acc_metrics["acc_oracle"])
                acc_jepa_open.append(acc_metrics["acc_jepa"])

            oracle.update(act_idx, next_obs["agent_0"].data)
            prev_a = Action(data=a_i_onehot)
            obs = next_obs

            if terminations["agent_0"].item() or truncations["agent_0"].item():
                break

    mean_acc_oracle = float(np.mean(acc_oracle_overall))
    mean_acc_jepa = float(np.mean(acc_jepa_overall))
    efficiency_overall = float((mean_acc_jepa / mean_acc_oracle * 100.0) if mean_acc_oracle > 0 else 100.0)

    mean_acc_oracle_listen = float(np.mean(acc_oracle_listen)) if acc_oracle_listen else 0.0
    mean_acc_jepa_listen = float(np.mean(acc_jepa_listen)) if acc_jepa_listen else 0.0
    efficiency_listen = float((mean_acc_jepa_listen / mean_acc_oracle_listen * 100.0) if mean_acc_oracle_listen > 0 else 100.0)

    return {
        "kl_overall": float(np.mean(kl_overall)),
        "kl_listen": float(np.mean(kl_listen)) if kl_listen else 0.0,
        "kl_open_reset": float(np.mean(kl_open)) if kl_open else 0.0,
        "kl_max": float(np.max(kl_overall)),
        "acc_oracle_overall": mean_acc_oracle,
        "acc_jepa_overall": mean_acc_jepa,
        "efficiency_overall": efficiency_overall,
        "acc_oracle_listen": mean_acc_oracle_listen,
        "acc_jepa_listen": mean_acc_jepa_listen,
        "efficiency_listen": efficiency_listen,
        "acc_oracle_open": float(np.mean(acc_oracle_open)) if acc_oracle_open else 0.0,
        "acc_jepa_open": float(np.mean(acc_jepa_open)) if acc_jepa_open else 0.0,
    }


def run_interpretability_trajectory_simulation(jepa_model, obs_probe, cfg, device=None):
    """Executes dual-regime monotonic evidence accumulation sequences (Tiger-Left and Tiger-Right)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    a_listen = F.one_hot(torch.tensor([LISTEN]), num_classes=cfg.env.action_dim_i).float().to(device)

    # -------------------------------------------------------------------------
    # Regime 1: Tiger-Left (s = TL) Evidence Accumulation (Hear GROWL_LEFT)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("  REGIME 1: TIGER-LEFT EVIDENCE ACCUMULATION (Hear GROWL_LEFT, b_t -> 1.0)")
    print("-" * 75)

    oracle_l = TigerBayesianOracle(growl_accuracy=0.85, creak_accuracy=1.0)
    oracle_l.reset()

    b_curr_l = torch.zeros(1, cfg.model.num_objects, cfg.model.latent_dim, device=device)
    prev_a_l = Action(data=torch.zeros(1, cfg.env.action_dim_i, device=device))
    growl_left_obs = torch.tensor([GROWL_LEFT, SILENCE], dtype=torch.float32, device=device)

    for step in range(8):
        b_star = oracle_l.belief_tiger_left
        p_oracle_gl = float((oracle_l.growl_acc * b_star + (1.0 - oracle_l.growl_acc) * (1.0 - b_star)) * 100.0)

        with torch.no_grad():
            probe_logits = obs_probe(b_curr_l, a_listen)
            p_jepa = F.softmax(probe_logits, dim=-1).squeeze(0)
            # Full marginalization: GROWL_LEFT corresponds to classes 0, 2, and 3
            p_jepa_gl = float((p_jepa[0] + p_jepa[2] + p_jepa[3]).item() * 100.0)

        tag = " (Initial Prior)" if step == 0 else ""
        print(f"  Step {step} (b* = {b_star:6.4f}) | Oracle P(GL) = {p_oracle_gl:5.2f}% | JEPA P(GL) = {p_jepa_gl:5.2f}%{tag}")

        b_curr_l = jepa_model.encode_context(growl_left_obs.view(1, -1), prev_a_l.data, b_curr_l)
        oracle_l.update(LISTEN, growl_left_obs.cpu())
        prev_a_l = Action(data=a_listen)

    print("-" * 75)

    # -------------------------------------------------------------------------
    # Regime 2: Tiger-Right (s = TR) Evidence Accumulation (Hear GROWL_RIGHT)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("  REGIME 2: TIGER-RIGHT EVIDENCE ACCUMULATION (Hear GROWL_RIGHT, b_t -> 0.0)")
    print("-" * 75)

    oracle_r = TigerBayesianOracle(growl_accuracy=0.85, creak_accuracy=1.0)
    oracle_r.reset()

    b_curr_r = torch.zeros(1, cfg.model.num_objects, cfg.model.latent_dim, device=device)
    prev_a_r = Action(data=torch.zeros(1, cfg.env.action_dim_i, device=device))
    growl_right_obs = torch.tensor([GROWL_RIGHT, SILENCE], dtype=torch.float32, device=device)

    for step in range(8):
        b_star = oracle_r.belief_tiger_left  # P(TL)
        p_oracle_gr = float(((1.0 - oracle_r.growl_acc) * b_star + oracle_r.growl_acc * (1.0 - b_star)) * 100.0)

        with torch.no_grad():
            probe_logits = obs_probe(b_curr_r, a_listen)
            p_jepa = F.softmax(probe_logits, dim=-1).squeeze(0)
            # Full marginalization: GROWL_RIGHT corresponds to classes 1, 4, and 5
            p_jepa_gr = float((p_jepa[1] + p_jepa[4] + p_jepa[5]).item() * 100.0)

        tag = " (Initial Prior)" if step == 0 else ""
        print(f"  Step {step} (b* = {b_star:6.4f}) | Oracle P(GR) = {p_oracle_gr:5.2f}% | JEPA P(GR) = {p_jepa_gr:5.2f}%{tag}")

        b_curr_r = jepa_model.encode_context(growl_right_obs.view(1, -1), prev_a_r.data, b_curr_r)
        oracle_r.update(LISTEN, growl_right_obs.cpu())
        prev_a_r = Action(data=a_listen)

    print("-" * 75 + "\n")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("\n" + "=" * 75)
    print("  JEPA vs BAYESIAN ORACLE : OBSERVATION ACCURACY & INTERPRETABILITY ")
    print("=" * 75 + "\n")

    logger = setup_logger("Tiger_KL_Accuracy_Benchmark")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluation Device: {device}")

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
        logger.info("Successfully restored pre-trained JEPA model checkpoint.")
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}. Running with initialized weights.")

    env = MultiAgentTigerEnv()
    logger.info("Collecting rollout dataset for non-intrusive observation probe training...")
    b_train, a_train, y_train = collect_probe_training_data(env, jepa_model, cfg, num_steps=8000, device=device)

    obs_probe = ObservationProbeHead(
        latent_dim=cfg.model.latent_dim,
        action_dim=cfg.env.action_dim_i,
        num_obs_classes=NUM_OBS_CLASSES
    ).to(device)
    logger.info("Training Action-Conditioned ObservationProbeHead on frozen latent belief states...")
    accuracy = train_observation_probe(obs_probe, b_train, a_train, y_train, epochs=60)
    logger.info(f"Observation Probe Training Complete. Offline Decoder Accuracy: {accuracy:.2%}")

    logger.info("Evaluating KL Divergence & Top-1 Observation Accuracy over 100 episodes...")
    res = evaluate_fidelity_and_accuracy(jepa_model, obs_probe, cfg, logger, num_episodes=100, device=device)

    print("\n" + "-" * 75)
    print("  OBSERVATION DISTRIBUTION FIDELITY (KL DIVERGENCE)")
    print("-" * 75)
    print(f"  • Overall Mean D_KL(P* || P_theta) : {res['kl_overall']:.6f} nats")
    print(f"  • Listen Step D_KL (Filtering)   : {res['kl_listen']:.6f} nats")
    print(f"  • Open Door D_KL (Amnesia Reset)  : {res['kl_open_reset']:.6f} nats")
    print(f"  • Max Single-Step D_KL            : {res['kl_max']:.6f} nats")
    print("-" * 75)

    print("\n" + "-" * 75)
    print("  EMPIRICAL TOP-1 PREDICTION ACCURACY & BAYES-OPTIMAL EFFICIENCY")
    print("-" * 75)
    print(f"  • Overall Top-1 Accuracy          : Oracle = {res['acc_oracle_overall']:.2%} | JEPA = {res['acc_jepa_overall']:.2%}")
    print(f"  • Overall Efficiency Ratio        : {res['efficiency_overall']:.2f}% of Bayes-Optimal Upper Bound")
    print(f"  • Listen Step Top-1 Accuracy      : Oracle = {res['acc_oracle_listen']:.2%} | JEPA = {res['acc_jepa_listen']:.2%}")
    print(f"  • Listen Step Efficiency Ratio    : {res['efficiency_listen']:.2f}% of Bayes-Optimal Upper Bound")
    print(f"  • Open Door (Reset) Accuracy      : Oracle = {res['acc_oracle_open']:.2%} | JEPA = {res['acc_jepa_open']:.2%}  <-- 50% RESET BOUND")
    print("-" * 75)

    run_interpretability_trajectory_simulation(jepa_model, obs_probe, cfg, device=device)
    log_telemetry(logger, "JEPA_Observation_Fidelity_KL_Accuracy", res)


if __name__ == "__main__":
    main()
