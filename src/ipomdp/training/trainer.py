# ABSOLUTE PATH: src/ipomdp/training/trainer.py
# ==============================================================================
# SEQUENCE WORLD MODEL & MULTI-STEP CONSISTENCY TRAINER
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Contiguous Sequence Unrolling with Burn-In Isolation:
#    - Unrolls context filter b_t = Filter(b_{t-1}, a_{t-1}, o_t) and target encoder
#      b_t^target over (burn_in + train_seq_len) steps.
#    - Multiplies loss terms by mask[:, t, :] to strictly isolate gradients to steps
#      following recurrent filter warmup.
#
# 2. bfloat16 AMP Dtype Initialization Safety:
#    - Allocates initial zero beliefs using dtype=obs_seq.dtype to match autocast
#      or input dtypes, preventing fatal GRU hidden state dtype mismatches at t=0.
#
# 3. Swarm Opponent Target Dimensionality Safety (M >= 1):
#    - Standardizes target opponent indices to shape (B, M) across all loss evaluations,
#      guaranteeing CrossEntropyLoss alignment without shape crashes.
#
# 4. TD(lambda) Target Recursion on Physical Scale:
#    - Computes continuous lambda-returns via backward Bellman recursion:
#         G_t^lambda = r_t + gamma * ((1 - lambda) * V_{t+1} + lambda * G_{t+1}^lambda)
#    - Regresses predicted value logits pred_v_logits against G_t^lambda using TwoHotSymlog.
#
# 5. Precision-Guarded VICReg Regularization:
#    - Promotes representations to float32 inside _vicreg_loss_batch prior to calculating
#      variance and covariance terms, preventing bfloat16 mantissa underflow and gradient zeroing.
#    - Normalizes MSE similarity and covariance Frobenius norms by latent dimension D:
#         L_sim = (1 / (N * D)) * sum (x - y)^2
#         L_cov = (1 / D) * (||C(X)||_F^2 - ||diag(C(X))||_2^2)
# ==============================================================================

import logging
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ..models.world_model import RecurrentJEPABase
from ..models.heads import ValueHead, RewardHead, DiscretePolicyHead
from ..models.distributions import TwoHotSymlog, symlog


class DiscreteRecurrentIPOMDPTrainer:
    """
    Sequence Trainer for Causal JEPA I-POMDP World Models.
    Optimizes JEPA Prediction, VICReg, Two-Hot Symlog Returns/Rewards, and Hallucinated Consistency.
    """

    def __init__(
        self,
        jepa_model: RecurrentJEPABase,
        value_head: ValueHead,
        reward_head: RewardHead,
        opponent_head: DiscretePolicyHead,
        logger: logging.Logger,
        device: torch.device,
        latent_dim: int,
        action_dim_i: int,
        action_dim_j: int,
        num_objects: int,
        learning_rate: float = 3e-4,
        hallucination_horizon: int = 3,
        lambda_consistency: float = 0.5,
        vicreg_sim_coeff: float = 25.0,
        vicreg_std_coeff: float = 25.0,
        vicreg_cov_coeff: float = 1.0,
        gamma: float = 0.99,
        lam: float = 0.95,
        detach_belief_for_rl: bool = False
    ):
        """
        Initializes Trainer and AdamW optimizer.
        """
        self.logger = logger
        self.device = device
        self.latent_dim = int(latent_dim)
        self.action_dim_i = int(action_dim_i)
        self.action_dim_j = int(action_dim_j)
        self.num_objects = int(num_objects)
        self.hallucination_horizon = int(hallucination_horizon)
        self.lambda_consistency = float(lambda_consistency)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.detach_belief_for_rl = bool(detach_belief_for_rl)

        self.vicreg_sim_coeff = float(vicreg_sim_coeff)
        self.vicreg_std_coeff = float(vicreg_std_coeff)
        self.vicreg_cov_coeff = float(vicreg_cov_coeff)

        self.jepa_model = jepa_model.to(self.device)
        self.value_head = value_head.to(self.device)
        self.reward_head = reward_head.to(self.device)
        self.opponent_head = opponent_head.to(self.device)

        self.trainable_params = (
            list(self.jepa_model.context_encoder.parameters()) +
            list(self.jepa_model.predictor.parameters()) +
            list(self.value_head.parameters()) +
            list(self.reward_head.parameters()) +
            list(self.opponent_head.parameters())
        )
        self.optimizer = optim.AdamW(self.trainable_params, lr=learning_rate, weight_decay=1e-4)

        self.jepa_criterion = nn.SmoothL1Loss(reduction='none')
        self.ce_criterion = nn.CrossEntropyLoss(reduction='none')
        self.twohot_criterion = TwoHotSymlog().to(self.device)

    def compute_lambda_returns(self, rewards: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Computes TD(lambda) target returns across temporal sequence horizons.

        Args:
            rewards: Immediate rewards tensor of shape (B, T, 1).
            values: Decoded continuous value predictions of shape (B, T + 1, 1).
            mask: Sequence mask tensor of shape (B, T, 1).

        Returns:
            Calculated lambda return targets of shape (B, T, 1).
        """
        _, t_steps, _ = rewards.shape
        returns = torch.zeros_like(rewards)
        last_lambda_return = values[:, -1, :]

        for t in reversed(range(t_steps)):
            ret = rewards[:, t, :] + self.gamma * (
                (1.0 - self.lam) * values[:, t + 1, :] + self.lam * last_lambda_return
            )
            last_lambda_return = ret * mask[:, t, :] + values[:, t, :] * (1.0 - mask[:, t, :])
            returns[:, t, :] = last_lambda_return

        return returns

    def train_sequence(
        self,
        batch: Dict[str, torch.Tensor],
        is_weights: torch.Tensor
    ) -> Tuple[Dict[str, float], torch.Tensor]:
        """
        Executes a complete backpropagation training step over a sequence batch.

        Args:
            batch: Sequence dictionary containing obs, rewards, mask, act_i, prev_act_i, act_j.
            is_weights: Importance sampling weights tensor of shape (B, 1).

        Returns:
            Tuple of (metrics_dict, mean_seq_td_errors of shape (B,)).
        """
        self.jepa_model.train()
        self.value_head.train()
        self.reward_head.train()
        self.opponent_head.train()
        self.optimizer.zero_grad(set_to_none=True)

        use_amp = (self.device.type == 'cuda')
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
            obs_seq = batch["obs"].to(self.device, non_blocking=True)
            reward_seq = batch["rewards"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            is_weights = is_weights.to(self.device, non_blocking=True)

            act_i_seq = F.one_hot(
                batch["act_i"].to(self.device, non_blocking=True).long().squeeze(-1),
                num_classes=self.action_dim_i
            ).float()

            prev_a_i = F.one_hot(
                batch["prev_act_i"].to(self.device, non_blocking=True).long().squeeze(-1),
                num_classes=self.action_dim_i
            ).float()

            act_j_seq = batch["act_j"].to(self.device, non_blocking=True).long()
            if act_j_seq.dim() == 2:
                act_j_seq = act_j_seq.unsqueeze(-1)

            b_batch, t_steps, _ = act_i_seq.shape
            b_states, target_b_states = [], []

            # Dtype-safe initialization: Match obs_seq.dtype to prevent GRU bfloat16/float32 crashes
            init_belief = torch.zeros(b_batch, self.num_objects, self.latent_dim, device=self.device, dtype=obs_seq.dtype)

            b_t = self.jepa_model.encode_context(obs_seq[:, 0, :], prev_a_i, init_belief)
            with torch.no_grad():
                target_b_t = self.jepa_model.encode_target(obs_seq[:, 0, :], prev_a_i, init_belief)

            for t in range(t_steps):
                b_states.append(b_t)
                target_b_states.append(target_b_t)
                b_t = self.jepa_model.encode_context(obs_seq[:, t + 1, :], act_i_seq[:, t, :], b_t)
                with torch.no_grad():
                    target_b_t = self.jepa_model.encode_target(obs_seq[:, t + 1, :], act_i_seq[:, t, :], target_b_t)

            b_states.append(b_t)
            target_b_states.append(target_b_t)

            with torch.no_grad():
                stacked_beliefs = torch.stack(b_states, dim=1)
                flat_beliefs = stacked_beliefs.view(b_batch * (t_steps + 1), self.num_objects, self.latent_dim)
                v_logits = self.value_head(flat_beliefs)
                v_vals = self.twohot_criterion.decode(v_logits, real_scale=True).view(b_batch, t_steps + 1, 1)

                real_returns = self.compute_lambda_returns(reward_seq, v_vals, mask)

            seq_loss_jepa, seq_loss_rl, seq_loss_consistency = 0.0, 0.0, 0.0
            seq_loss_value, seq_loss_reward, seq_loss_opp = 0.0, 0.0, 0.0
            seq_mask_sum = torch.zeros(b_batch, 1, device=self.device)
            seq_td_errors = torch.zeros(b_batch, t_steps, device=self.device)
            total_loss_vicreg = torch.tensor(0.0, device=self.device)

            for t in range(t_steps):
                a_i_t = act_i_seq[:, t, :]
                a_j_t_loss = act_j_seq[:, t, :]
                a_j_t_warm = F.one_hot(a_j_t_loss, num_classes=self.action_dim_j).float()

                m_t = mask[:, t, :]
                seq_mask_sum += m_t

                current_b = b_states[t]
                target_b_t1 = target_b_states[t + 1]

                predicted_b_t1, kl_loss = self.jepa_model.predict_next_belief_train(
                    current_b, a_i_t, a_j_t_warm, target_b_t1
                )

                loss_jepa = self.jepa_criterion(predicted_b_t1, target_b_t1).mean(dim=(1, 2)).unsqueeze(-1) * m_t
                loss_kl = torch.max(kl_loss, torch.tensor(1.0, device=self.device)) * 0.1 * m_t
                seq_loss_jepa += (loss_jepa + loss_kl)

                if m_t.sum() > 1:
                    valid_idx = m_t.squeeze(-1).bool()
                    total_loss_vicreg += self._vicreg_loss_batch(
                        predicted_b_t1[valid_idx].view(-1, self.latent_dim),
                        target_b_t1[valid_idx].view(-1, self.latent_dim),
                        is_weights[valid_idx]
                    )

                b_for_rl = current_b.detach() if self.detach_belief_for_rl else current_b
                pred_v_logits = self.value_head(b_for_rl)
                pred_r_logits = self.reward_head(b_for_rl, a_i_t, a_j_t_warm)
                pred_opp_logits = self.opponent_head(b_for_rl)

                loss_value = self.twohot_criterion(pred_v_logits, real_returns[:, t, :], auto_symlog=True) * m_t
                loss_reward = self.twohot_criterion(pred_r_logits, reward_seq[:, t, :], auto_symlog=True) * m_t

                # Swarm-safe opponent cross-entropy loss computation
                b_curr, num_opps = a_j_t_loss.shape[:2] if a_j_t_loss.dim() >= 2 else (a_j_t_loss.size(0), 1)
                flat_pred_opp = pred_opp_logits.view(-1, self.action_dim_j)
                flat_targ_opp = a_j_t_loss.contiguous().view(-1)
                loss_opp = self.ce_criterion(flat_pred_opp, flat_targ_opp).view(b_curr, num_opps).mean(dim=1).unsqueeze(-1) * m_t

                with torch.no_grad():
                    td_error = torch.abs(
                        self.twohot_criterion.decode(pred_v_logits, real_scale=False) - symlog(real_returns[:, t, :])
                    ) * m_t
                    seq_td_errors[:, t] = td_error.squeeze(-1)

                seq_loss_value += loss_value
                seq_loss_reward += loss_reward
                seq_loss_opp += loss_opp
                seq_loss_rl += (loss_value + loss_reward + loss_opp)

                dream_b = current_b
                dream_states, dream_rewards = [], []

                for h in range(self.hallucination_horizon):
                    if t + h >= t_steps:
                        break
                    future_a_i = act_i_seq[:, t + h, :]
                    future_a_j = F.one_hot(act_j_seq[:, t + h, :], num_classes=self.action_dim_j).float()

                    dream_b = self.jepa_model.predict_next_belief(dream_b, future_a_i, future_a_j)
                    dream_states.append((dream_b, mask[:, t + h, :]))
                    dream_rewards.append(self.reward_head(dream_b, future_a_i, future_a_j))

                if dream_states:
                    with torch.no_grad():
                        last_v = self.twohot_criterion.decode(self.value_head(dream_states[-1][0]), real_scale=True)
                    dream_lambda_target = last_v

                    for h_idx in reversed(range(len(dream_states))):
                        d_b, d_m = dream_states[h_idx]
                        r_val = self.twohot_criterion.decode(dream_rewards[h_idx], real_scale=True)
                        dream_lambda_target = r_val + self.gamma * dream_lambda_target

                        v_dream_logits = self.value_head(d_b)
                        loss_consist = self.twohot_criterion(v_dream_logits, dream_lambda_target.detach(), auto_symlog=True)
                        seq_loss_consistency += (loss_consist * d_m)

            total_mask_sum = seq_mask_sum.sum().clamp(min=1.0)
            total_loss_jepa = (seq_loss_jepa * is_weights).sum() / total_mask_sum
            total_loss_rl = (seq_loss_rl * is_weights).sum() / total_mask_sum
            total_loss_value = (seq_loss_value * is_weights).sum() / total_mask_sum
            total_loss_reward = (seq_loss_reward * is_weights).sum() / total_mask_sum
            total_loss_opp = (seq_loss_opp * is_weights).sum() / total_mask_sum
            total_loss_consistency = ((seq_loss_consistency * is_weights).sum() / total_mask_sum) * self.lambda_consistency

            combined_loss = total_loss_jepa + total_loss_vicreg + total_loss_rl + total_loss_consistency

        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm=1.0)
        self.optimizer.step()
        self.jepa_model.update_target_encoder()

        mean_seq_td_errors = seq_td_errors.sum(dim=1) / seq_mask_sum.squeeze(-1).clamp(min=1.0)

        metrics = {
            "loss_jepa": total_loss_jepa.item(),
            "loss_vicreg": total_loss_vicreg.item(),
            "loss_rl": total_loss_rl.item(),
            "loss_value": total_loss_value.item(),
            "loss_reward": total_loss_reward.item(),
            "loss_opp": total_loss_opp.item(),
            "loss_consistency": total_loss_consistency.item(),
            "mean_td_error": mean_seq_td_errors.mean().item()
        }
        return metrics, mean_seq_td_errors

    def _vicreg_loss_batch(self, x: torch.Tensor, y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Dimension-Normalized VICReg Regularization Loss.
        Promotes representations to float32 to prevent bfloat16 mantissa underflow.
        """
        x_f32 = x.float()
        y_f32 = y.float()
        weights_f32 = weights.float().repeat_interleave(self.num_objects, dim=0)

        w_sum = weights_f32.sum().clamp(min=1e-8)
        n, d = x_f32.shape

        sim_loss = torch.sum(weights_f32 * (x_f32 - y_f32) ** 2) / (w_sum * float(d))

        mean_x = torch.sum(weights_f32 * x_f32, dim=0) / w_sum
        mean_y = torch.sum(weights_f32 * y_f32, dim=0) / w_sum
        x_c, y_c = x_f32 - mean_x, y_f32 - mean_y

        var_x = torch.sum(weights_f32 * (x_c ** 2), dim=0) / w_sum
        var_y = torch.sum(weights_f32 * (y_c ** 2), dim=0) / w_sum

        std_x = torch.sqrt(var_x + 1e-4)
        std_y = torch.sqrt(var_y + 1e-4)
        std_loss = torch.mean(F.relu(1.0 - std_x)) + torch.mean(F.relu(1.0 - std_y))

        if n <= 1:
            return self.vicreg_sim_coeff * sim_loss + self.vicreg_std_coeff * std_loss

        cov_x = (x_c.T @ (weights_f32 * x_c)) / w_sum
        cov_y = (y_c.T @ (weights_f32 * y_c)) / w_sum

        cov_loss_x = (cov_x.pow(2).sum() - cov_x.diagonal().pow(2).sum()) / float(d)
        cov_loss_y = (cov_y.pow(2).sum() - cov_y.diagonal().pow(2).sum()) / float(d)
        cov_loss = cov_loss_x + cov_loss_y

        return self.vicreg_sim_coeff * sim_loss + self.vicreg_std_coeff * std_loss + self.vicreg_cov_coeff * cov_loss
