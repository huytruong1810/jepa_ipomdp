# ABSOLUTE PATH: src/ipomdp/telemetry/metrics.py
# ==============================================================================
# UNIFIED TELEMETRY TRACKER, KL DIVERGENCE & ACCURACY METRICS
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. High-Precision Evaluation:
#    - Executes all logarithmic and probability operations in torch.float64 to
#      eliminate rounding cancellation noise when measuring small divergences (D_KL < 1e-4).
#
# 2. Bounded Re-Normalization:
#    - Clamps probability inputs to [eps, 1.0] and re-normalizes distributions to guarantee
#      sum(p) == 1.0, ensuring the information-theoretic lower bound D_KL(P* || P_theta) >= 0.
#
# 3. Multi-Metric Distribution Evaluation:
#    - Evaluates Top-1 hard accuracy, Negative Log-Likelihood (NLL) loss, Brier score,
#      and Bayes-Optimal relative efficiency ratio:
#         Efficiency Ratio = (JEPA_Acc / Oracle_Acc) * 100.0
#      quantifying how closely JEPA approaches the theoretical predictability ceiling.
#
# 4. Strict 1D Rank Normalization on Target Indices:
#    - Forces target_idx to a 1D tensor (B,) upon entry via .view(-1) to prevent
#      implicit broadcasting against 1D predictions (B,).
# ==============================================================================

from pathlib import Path
import time
from typing import Dict, Tuple, Optional, Union
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class MetricsLogger:
    """Domain-agnostic time-series metric tracker using TensorBoard."""

    def __init__(self, log_dir: str, experiment_name: str):
        """Creates timestamped experiment directory and initializes SummaryWriter."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(log_dir) / f"{experiment_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.run_dir))

    def log_metrics(self, metrics_dict: Dict[str, Union[float, int]], step: int, prefix: str = ""):
        """Logs a dictionary of scalar metrics with optional UI category prefix."""
        for key, value in metrics_dict.items():
            if value is None:
                continue
            tag = f"{prefix}/{key}" if prefix else key
            self.writer.add_scalar(tag, float(value), int(step))

    def log_figure(self, tag: str, figure, step: int):
        """Logs a matplotlib figure directly into TensorBoard."""
        self.writer.add_figure(tag, figure, global_step=int(step))

    def log_hyperparams(self, hparam_dict: dict, metric_dict: dict):
        """Logs static hyperparameters alongside final evaluation metrics."""
        self.writer.add_hparams(hparam_dict, metric_dict)

    def close(self):
        """Closes SummaryWriter handle."""
        self.writer.close()


def compute_distribution_kl_divergence(
    p_true: torch.Tensor,
    p_pred: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-12
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes KL divergence D_KL(P_true || P_pred) between true and predicted distributions.

    Mathematical Formulation:
        D_KL(P* || P_theta) = sum_{k=1}^K P*(x_k) * [ ln(P*(x_k)) - ln(P_theta(x_k)) ]

    Args:
        p_true: Ground-truth probability distribution tensor of shape (..., K) or logits.
        p_pred: Predicted probability distribution tensor of shape (..., K) or logits.
        mask: Optional mask tensor of shape (...) to filter valid steps.
        eps: Numerical stability threshold.

    Returns:
        Tuple of (per_sample_kl_tensor, summary_metrics_dict).
    """
    p_true_f64 = p_true.detach().to(dtype=torch.float64)
    p_pred_f64 = p_pred.detach().to(dtype=torch.float64)

    if not torch.allclose(p_true_f64.sum(dim=-1), torch.ones_like(p_true_f64.sum(dim=-1)), atol=1e-3):
        p_true_f64 = F.softmax(p_true_f64, dim=-1)

    if not torch.allclose(p_pred_f64.sum(dim=-1), torch.ones_like(p_pred_f64.sum(dim=-1)), atol=1e-3):
        p_pred_f64 = F.softmax(p_pred_f64, dim=-1)

    p_true_clamped = torch.clamp(p_true_f64, min=eps, max=1.0)
    p_pred_clamped = torch.clamp(p_pred_f64, min=eps, max=1.0)

    p_true_norm = p_true_clamped / p_true_clamped.sum(dim=-1, keepdim=True)
    p_pred_norm = p_pred_clamped / p_pred_clamped.sum(dim=-1, keepdim=True)

    kl_elementwise = p_true_norm * (torch.log(p_true_norm) - torch.log(p_pred_norm))
    kl_sample = torch.sum(kl_elementwise, dim=-1)

    if mask is not None:
        mask_f64 = mask.detach().to(dtype=torch.float64).view_as(kl_sample)
        valid_count = mask_f64.sum().item()
        if valid_count > 0:
            mean_kl = (kl_sample * mask_f64).sum().item() / valid_count
            valid_samples = kl_sample[mask_f64.bool()]
            std_kl = valid_samples.std().item() if valid_count > 1 else 0.0
            min_kl = valid_samples.min().item()
            max_kl = valid_samples.max().item()
        else:
            mean_kl, std_kl, min_kl, max_kl = 0.0, 0.0, 0.0, 0.0
    else:
        mean_kl = kl_sample.mean().item()
        std_kl = kl_sample.std().item() if kl_sample.numel() > 1 else 0.0
        min_kl = kl_sample.min().item()
        max_kl = kl_sample.max().item()

    metrics = {
        "kl_mean": float(mean_kl),
        "kl_std": float(std_kl),
        "kl_min": float(min_kl),
        "kl_max": float(max_kl),
    }

    return kl_sample, metrics


def compute_observation_accuracy_metrics(
    p_oracle: torch.Tensor,
    p_jepa: torch.Tensor,
    target_idx: torch.Tensor,
    eps: float = 1e-12
) -> Dict[str, float]:
    """
    Computes Top-1 accuracy, NLL, Brier score, and Bayes-Optimal efficiency ratio.

    Args:
        p_oracle: Analytical oracle probability tensor of shape (B, K) or (K,).
        p_jepa: Model predicted probability tensor of shape (B, K) or (K,).
        target_idx: True discrete category index tensor of shape (B,), (B, 1), or scalar.
        eps: Epsilon threshold for log stability.

    Returns:
        Dictionary containing Oracle and JEPA accuracy, NLL, Brier score, and Efficiency Ratios.
    """
    if p_oracle.dim() == 1:
        p_oracle = p_oracle.unsqueeze(0)
    if p_jepa.dim() == 1:
        p_jepa = p_jepa.unsqueeze(0)

    p_oracle_f64 = p_oracle.detach().to(dtype=torch.float64)
    p_jepa_f64 = p_jepa.detach().to(dtype=torch.float64)

    # Rank Safety Guard: Force target_idx to 1D tensor (B,) to prevent (B, B) implicit broadcasting
    target_f64 = target_idx.detach().to(dtype=torch.long).view(-1)

    p_oracle_norm = p_oracle_f64 / p_oracle_f64.sum(dim=-1, keepdim=True).clamp(min=eps)
    p_jepa_norm = p_jepa_f64 / p_jepa_f64.sum(dim=-1, keepdim=True).clamp(min=eps)

    pred_oracle = torch.argmax(p_oracle_norm, dim=-1)
    pred_jepa = torch.argmax(p_jepa_norm, dim=-1)

    acc_oracle_mask = (pred_oracle == target_f64).to(dtype=torch.float64)
    acc_jepa_mask = (pred_jepa == target_f64).to(dtype=torch.float64)

    acc_oracle = float(acc_oracle_mask.mean().item())
    acc_jepa = float(acc_jepa_mask.mean().item())

    efficiency_ratio = float((acc_jepa / acc_oracle * 100.0) if acc_oracle > 0 else 100.0)

    p_oracle_target = p_oracle_norm.gather(1, target_f64.unsqueeze(-1)).squeeze(-1).clamp(min=eps)
    p_jepa_target = p_jepa_norm.gather(1, target_f64.unsqueeze(-1)).squeeze(-1).clamp(min=eps)

    nll_oracle = float(-torch.log(p_oracle_target).mean().item())
    nll_jepa = float(-torch.log(p_jepa_target).mean().item())

    one_hot_target = F.one_hot(target_f64, num_classes=p_oracle_norm.size(-1)).to(dtype=torch.float64)
    brier_oracle = float(((p_oracle_norm - one_hot_target) ** 2).sum(dim=-1).mean().item())
    brier_jepa = float(((p_jepa_norm - one_hot_target) ** 2).sum(dim=-1).mean().item())

    return {
        "acc_oracle": acc_oracle,
        "acc_jepa": acc_jepa,
        "efficiency_ratio": efficiency_ratio,
        "nll_oracle": nll_oracle,
        "nll_jepa": nll_jepa,
        "brier_oracle": brier_oracle,
        "brier_jepa": brier_jepa,
    }
