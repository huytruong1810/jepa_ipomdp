# ABSOLUTE PATH: src/ipomdp/training/trainer.py
# ==============================================================================
# WHOLE-EPISODE WORLD-MODEL TRAINER (JEPA OR DECODER REPRESENTATION + REWARD/VALUE GROUNDING)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Whole-Episode Unrolls:
#    - Each update filters complete episodes from the learned z_0, folding in o_0 for domains
#      that emit one (models/world_model.py section 1; episode_buffer.py), so training sees
#      exactly the latents the online agent computes. There is no burn-in,
#      no mask, and no zero-initialised mid-episode state.
#
# 2. Loss Terms of the JEPA Agent (Representation.JEPA; averaged over the B x T transitions):
#      JEPA         MSE( g(z_t, a_t), sg(z-bar_{t+1}) )      z-bar = EMA target filter
#      Reward       TwoHot( R(z_t, a_t), r_t )
#      Value        TwoHot( V(sg(z_t)), sg(max_a [R(z_t,a) + gamma sum_o P(o|z_t,a) V-bar(tau(z_t,a,o))]) )
#                   planning model, detached input, all latents z_0..z_T
#      Observation  CE( P(o | sg(z_t), a_t), o_{t+1} )       planning model, detached input
#    - No opponent terms: the model describes the POMDP the agent faces, with the opponent
#      folded into the environment (models/heads.py, section 1).
#    - Only the JEPA and reward terms shape the representation; they are the terms grounded in
#      real data. JEPA self-prediction alone does not make z_t a belief (Phase-2 study,
#      models/world_model.py section 3); reward grounding does.
#    - The value and observation heads read DETACHED latents. For the value head this is
#      essential (Phase-5 finding): its Bellman target is computed from the model itself, so it
#      carries no information from real rewards, and letting it shape the encoder is a
#      representation-collapse pressure (making latents alike satisfies self-consistent
#      targets). With gradients into the filter the belief probe degraded from KL 0.002 to
#      0.045, the reward head lost its belief dependence (door-reward error 13) and the greedy
#      agent listened forever (return -19.88). Detached, on the same random-policy data: probe
#      KL 0.0008, door-reward error 1-2, and the learned-model greedy agent earns 17.8 +- 1.6
#      (optimal 19.28).
#    - Two-hot means are unbiased (models/distributions.py), so V and R decode to expected
#      returns/rewards even for multimodal targets such as Tiger's -100/+10 door reward.
#
# 2b. The Decoder Baseline (Representation.DECODER, the baseline chosen after the Tiger gate):
#      Reward       TwoHot( R(z_t, a_t), r_t )                   (unchanged)
#      Value        as above                                      (unchanged, detached input)
#      Observation  CE( P(o | z_t, a_t), o_{t+1} )                gradients INTO the filter
#    - The JEPA term, its predictor and its EMA target filter are absent (jepa=None). The
#      representation is instead shaped by the next-observation likelihood: the observation
#      head IS the decoder. This is the one change against the JEPA agent, so a comparison
#      of the two isolates what replacing observation reconstruction by latent
#      self-prediction buys; the heads, value targets, Polyak acting copies and search are
#      identical.
#    - The decoder predicts o_{t+1} from (z_t, a_t) rather than reconstructing o_t from z_t
#      (DreamerV3's posterior decoder): the filter has just read o_t, so reconstructing it can
#      be satisfied by copying the last input and carries no belief pressure.
#
# 2c. Optional Multi-Step Observation Term (RunConfig.observation_horizon = k > 0; either representation):
#      MultiStep    mean_j CE( P(o_{t+j} | z_t, a_t..a_{t+j-1}), o_{t+j} ),  j = 1..k, t + j <= T
#                   gradients INTO the filter (models/heads.py, section 4)
#    - Added because on Wumpus World neither JEPA nor one-step observation prediction makes the
#      latent track the hunter's position, heading and the danger ahead, while the same filter
#      learns them from privileged supervision (HANDOFF.md, section 4c). Each horizon j is
#      averaged over its valid starts, then the k horizons are averaged, so the term has the
#      scale of one cross-entropy whatever k is.
#    - k = 0 (the default, and canonical Tiger) builds nothing and leaves training unchanged.
#
# 3. Value Target: the Bellman Optimality Backup Through the Learned Model (Phase-5 decision):
#      V_target(z) = max_a [ R(z, a) + gamma * sum_o P(o | z, a) * V-bar(tau(z, a, o)) ]
#    - R and P(o | z, a) are the reward and observation heads, tau is the belief filter, and
#      V-bar is a slow EMA copy of the value head (target network). The target is computed with
#      LearnedSearchModel.expand -- literally the one-step backup the planner performs -- for
#      EVERY latent z_0..z_T of the batch (no future data is needed), without gradients.
#    - This is fitted value iteration over visited beliefs: off-policy, aiming at V* directly.
#      On an exact model it is exact value iteration (checked against the solver in the tests).
#    - Phase-5 finding that motivated it: TD(lambda) targets estimate the value of the policy
#      that COLLECTED the data. The exploring collection policy (root Dirichlet noise and a
#      visit-count temperature) opened doors at random ~9% of the time, so under it a confident
#      belief was worth -215 and the post-opening prior -482; with such leaves listening always
#      beat opening and the greedy agent listened forever (evaluation return -19.88, exactly the
#      always-listen return, at every evaluation).
#    - gamma is the domain's discount (0.95 for canonical Tiger), passed in by TrainingRun.
#
# 3b. Optional Finite-Horizon Value (horizon = H; model.time_aware_value, user decision 2026-10-07):
#      V_target(z_t, k_t) = max_a [ R(z_t, a) + gamma * sum_o P(o | z_t, a) * V-bar(tau(z_t, a, o), k_t - 1) ],
#      k_t = H - t,   V(., 0) = 0
#    - The value head becomes V(z, k) (models/heads.py, section 2b), and the backup counts the
#      steps left down to the end of the window, where the value is exactly 0 (terminal, not
#      bootstrapped). It is still computed by LearnedSearchModel.expand, now on states [z, k]
#      (planning/search_model.py, section 5), so targets and planner stay the same backup.
#    - The buffer stores whole windows (T = H), so latent z_t of every episode has k_t = H - t.
#      The rows with k = 0 (z_T) are dropped from the value loss: their value is defined, not learned.
#    - Motivation: reports/2026-10-02_wumpus_diagnostics/unfinished_diag.py (the stationary value
#      makes a wasted step cost only about 1 + (1 - gamma) * V, so the greedy planner idled).
#    - horizon=None (the default and canonical Tiger) leaves the computation unchanged.
#
# 3c. Optional n-Step Greedy-Return Grounding (nstep = n > 0; model.value_nstep, user decision 2026-10-08):
#      G_t = sum_{i<m} gamma^i r_{t+i} + gamma^m V-bar(z_{t+m}),   m = min(n, T - t)
#      V_target(z_t) = (1 - mix) * Bellman(z_t) + mix * G_t   on rows t < T of GREEDY episodes only
#    - Why (reports/2026-10-02_wumpus_diagnostics/value_chain_diag.py): on Wumpus the bootstrapped value
#      is self-consistent but ungrounded on the greedy planner's idle chains (V-hat ~650 against a
#      realised -22). A one-step backup along a no-op can lower V-hat by only about 1 + (1 - gamma) V
#      per iteration; an n-step return of real rewards lowers it by about (1 - gamma^n) V. A value fitted
#      to real returns on the same latents ranked "step into a safe cell" above a no-op.
#    - Greedy episodes only (EpisodeBuffer, section 4; stored by training/run.py, section 8): they are
#      on-policy for the greedy planner, whose own idling must be priced in. The exploring
#      policy's returns are NOT used, since they valued Tiger's confident beliefs as if doors
#      were opened at random and made the greedy agent listen forever (section 3).
#    - V-bar is the target value head on the batch's own detached latents (with k on finite-horizon
#      runs, where V-bar(z_T, 0) = 0 ends the sum). Rows t = T keep the Bellman target.
#    - nstep = 0 (the default and canonical Tiger) leaves the computation unchanged.
#
# 4. No Imagination Losses:
#    - An earlier version also trained V on latents imagined by a stochastic latent
#      transition ("consistency"). Planning now imagines through the real filter
#      (observation branching), so imagined latents lie on the manifold V is trained on and
#      that loss (and the transition it depended on) was removed.
#
# 5. Failure Is Loud:
#    - A non-finite loss (or a non-finite / out-of-bound two-hot target, see distributions.py)
#      raises FloatingPointError instead of silently skipping or sanitising the update.
#
# 6. Mixed Precision:
#    - The forward pass runs under bfloat16 autocast on CUDA; losses are computed in float32.
#      The value targets are computed with autocast disabled, in float32, exactly as the planner
#      evaluates the same backup (planning/search_model.py, section 4).
# ==============================================================================

from dataclasses import dataclass
from enum import StrEnum

import torch
import torch.nn.functional as F
from torch import Tensor

from ..models.distributions import TwoHotSymlog
from ..models.ema import ema_update, frozen_copy
from ..models.heads import HorizonValueHead, MultiStepObservationPredictor, ObservationHead, RewardHead, ValueHead
from ..models.world_model import BeliefFilter, RecurrentJEPA
from ..planning.search_model import LearnedSearchModel
from .episode_buffer import EpisodeBatch


class Representation(StrEnum):
    """What shapes the belief filter besides reward grounding (sections 2 and 2b)."""

    JEPA = "jepa"          # latent self-prediction against an EMA target; observation head detached
    DECODER = "decoder"    # next-observation likelihood of the observation head; no JEPA parts


@dataclass(frozen=True)
class TrainerConfig:
    """Optimisation hyper-parameters (conf/config.yaml, section `trainer`)."""

    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    value_target_momentum: float


class WorldModelTrainer:
    """One gradient step of the belief filter and its heads on a batch of whole episodes."""

    def __init__(
        self,
        belief_filter: BeliefFilter,
        jepa: RecurrentJEPA | None,
        value_head: ValueHead | HorizonValueHead,
        reward_head: RewardHead,
        observation_head: ObservationHead,
        codec: TwoHotSymlog,
        num_actions: int,
        num_observations: int,
        discount: float,
        config: TrainerConfig,
        device: torch.device,
        multistep_predictor: MultiStepObservationPredictor | None = None,
        horizon: int | None = None,
        nstep: int = 0,
        nstep_mix: float = 0.0,
    ):
        """
        Args:
            belief_filter: The online filter z_{t+1} = f(z_t, a_t, o_{t+1}).
            jepa: The JEPA parts built around belief_filter (Representation.JEPA), or None for
                the decoder baseline (Representation.DECODER, section 2b).
            multistep_predictor: The optional multi-step observation term (section 2c), or None.
            horizon: Steps per window H for the finite-horizon value (section 3b), or None.
            nstep, nstep_mix: n and the mixing weight of the n-step greedy-return target (section 3c); n = 0 is off.
        """
        if nstep < 0 or not 0.0 <= nstep_mix <= 1.0 or (nstep > 0 and nstep_mix == 0.0):
            raise ValueError(f"need nstep >= 0 and nstep_mix in (0, 1] when nstep > 0; got {nstep}, {nstep_mix}.")
        if jepa is not None and jepa.belief_filter is not belief_filter:
            raise ValueError("jepa must wrap the trained belief_filter")
        self.belief_filter = belief_filter
        self.jepa = jepa
        self.representation = Representation.DECODER if jepa is None else Representation.JEPA
        self.value_head = value_head
        self.reward_head = reward_head
        self.observation_head = observation_head
        self.multistep_predictor = multistep_predictor
        self.twohot = codec
        self.num_actions = num_actions
        self.num_observations = num_observations
        self.discount = discount
        self.horizon = horizon
        self.nstep = nstep
        self.nstep_mix = nstep_mix
        self.config = config
        self.device = device
        self.parameters = [
            *belief_filter.parameters(),
            *(jepa.predictor.parameters() if jepa is not None else ()),
            *value_head.parameters(),
            *reward_head.parameters(),
            *observation_head.parameters(),
            *(multistep_predictor.parameters() if multistep_predictor is not None else ()),
        ]
        self.optimizer = torch.optim.AdamW(self.parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
        self.target_value_head = frozen_copy(value_head)
        self.target_model = LearnedSearchModel(belief_filter, reward_head, observation_head,
                                               self.target_value_head, codec, num_actions, num_observations, discount,
                                               horizon)

    @torch.no_grad()
    def value_targets(self, latents: Tensor) -> Tensor:
        """
        Bellman optimality backup through the learned model (module header, sections 3 and 3b).

        Args:
            latents: Search-model states: belief latents (N, D), or [z, k] (N, D + 1) with a horizon.

        Returns:
            max_a [R + gamma sum_o P(o) V-bar(tau)], shape (N,), float32.
        """
        with torch.autocast(self.device.type, enabled=False):
            expansion = self.target_model.expand(latents.float())
            q = expansion.rewards + self.discount * (expansion.observation_probs * expansion.next_values).sum(-1)
        return q.max(dim=-1).values

    @torch.no_grad()
    def nstep_returns(self, states: Tensor, rewards: Tensor) -> Tensor:
        """
        n-step returns bootstrapped with the target value (module header, section 3c).

        Args:
            states: Search-model states of every step, (B, T + 1, D) or (B, T + 1, D + 1) with a horizon.
            rewards: Rewards r_0 .. r_{T-1}, (B, T).

        Returns:
            G_t for t = 0 .. T - 1, shape (B, T), float32.
        """
        batch, steps = rewards.shape
        n = self.nstep
        with torch.autocast(self.device.type, enabled=False):
            boot = self.target_model.values(states.flatten(0, 1).float()).view(batch, steps + 1)
            padded = torch.cat([rewards.float(), rewards.new_zeros(batch, n).float()], dim=1)    # zero past T
            t = torch.arange(steps, device=rewards.device)
            returns = sum(self.discount ** i * padded[:, i:i + steps] for i in range(n))
            m = torch.clamp(steps - t, max=n)
            return returns + (self.discount ** m.float()) * boot[:, t + m]

    def train_step(self, episodes: EpisodeBatch) -> dict[str, float]:
        """
        One optimiser step on a batch of complete episodes.

        Returns:
            Scalar diagnostics (losses, mean value target and mean |V - V_target|).
        """
        self.belief_filter.train()
        actions = F.one_hot(episodes.actions, self.num_actions).float()
        observations = F.one_hot(episodes.observations, self.num_observations).float()
        initial = (None if episodes.initial_observations is None
                   else F.one_hot(episodes.initial_observations, self.num_observations).float())
        batch, steps = episodes.actions.shape
        losses: dict[str, Tensor] = {}

        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            latents = self.belief_filter.unroll(actions, observations, initial)            # (B, T+1, D)
            flat = lambda x: x.reshape(batch * steps, -1)  # noqa: E731
            current = flat(latents[:, :-1])
            action_flat = flat(actions)

            if self.jepa is not None:
                with torch.no_grad():
                    targets = self.jepa.target_filter.unroll(actions, observations, initial)  # (B, T+1, D)
                predicted = self.jepa.predictor(current, action_flat)
                losses["prediction"] = F.mse_loss(predicted.float(), flat(targets[:, 1:]).float())

            losses["reward"] = self.twohot.loss(self.reward_head(current, action_flat),
                                                episodes.rewards.reshape(-1)).mean()

            all_latents = latents.reshape(batch * (steps + 1), -1)
            if self.horizon is None:
                targets_value = self.value_targets(all_latents.detach())
                value_logits = self.value_head(all_latents.detach())
            else:
                if steps != self.horizon:
                    raise ValueError(f"finite-horizon value needs whole windows: T = {steps}, H = {self.horizon}.")
                steps_left = (self.horizon - torch.arange(steps + 1, device=self.device)).repeat(batch).float()
                live = steps_left > 0                                                    # k = 0 rows: V = 0 by definition
                states = torch.cat([all_latents.detach().float(), steps_left.unsqueeze(-1)], dim=-1)[live]
                targets_value = self.value_targets(states)
                value_logits = self.value_head(all_latents.detach()[live], steps_left[live])
            if self.nstep > 0:
                targets_value, nstep_metrics = self._mix_nstep(latents.detach(), episodes, targets_value)
            losses["value"] = self.twohot.loss(value_logits, targets_value).mean()

            # The decoder baseline's representation term; a detached planning-model fit otherwise.
            decoded = current if self.representation is Representation.DECODER else current.detach()
            losses["observation"] = F.cross_entropy(self.observation_head(decoded, action_flat).float(),
                                                    episodes.observations.reshape(-1))

            if self.multistep_predictor is not None:
                horizons = self.multistep_predictor(latents[:, :-1], actions)
                losses["multistep_observation"] = sum(
                    F.cross_entropy(logits.float().reshape(-1, self.num_observations),
                                    episodes.observations[:, j:].reshape(-1))
                    for j, logits in enumerate(horizons)) / len(horizons)

            total = sum(losses.values())

        if not torch.isfinite(total):
            raise FloatingPointError(
                "Non-finite loss: " + ", ".join(f"{name}={loss.item()}" for name, loss in losses.items()))

        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, self.config.grad_clip_norm)
        self.optimizer.step()
        if self.jepa is not None:
            self.jepa.update_target()
        ema_update(self.target_value_head, self.value_head, self.config.value_target_momentum)

        with torch.no_grad():
            value_error = (self.twohot.mean(value_logits) - targets_value).abs().mean()

        return {
            **(nstep_metrics if self.nstep > 0 else {}),
            "loss_total": total.item(),
            **{f"loss_{name}": loss.item() for name, loss in losses.items()},
            "value_target_mean": targets_value.mean().item(),
            "value_abs_error": value_error.item(),
        }

    def _mix_nstep(self, latents: Tensor, episodes: EpisodeBatch,
                   bellman: Tensor) -> tuple[Tensor, dict[str, float]]:
        """Section 3c: the value targets with greedy rows t < T mixed with their n-step returns."""
        if episodes.greedy is None:
            raise ValueError("the n-step greedy-return target needs batches sampled from an EpisodeBuffer.")
        batch, steps = episodes.rewards.shape
        states = latents.float()
        if self.horizon is not None:
            steps_left = (self.horizon - torch.arange(steps + 1, device=self.device)).float()
            states = torch.cat([states, steps_left.expand(batch, -1).unsqueeze(-1)], dim=-1)
        returns = torch.cat([self.nstep_returns(states, episodes.rewards),
                             torch.zeros(batch, 1, device=self.device)], dim=1)          # (B, T + 1); t = T unused
        use = episodes.greedy.unsqueeze(1) & (torch.arange(steps + 1, device=self.device) < steps)
        if self.horizon is not None:                                                     # rows k = 0 are not trained
            returns, use = returns[:, :-1], use[:, :-1]
        returns, use = returns.flatten(), use.flatten()
        mixed = torch.where(use, (1.0 - self.nstep_mix) * bellman + self.nstep_mix * returns, bellman)
        greedy_rows = int(use.sum())
        return mixed, {"value_nstep_rows": float(greedy_rows),
                       "value_nstep_minus_bellman": float((returns - bellman)[use].mean()) if greedy_rows else 0.0}
