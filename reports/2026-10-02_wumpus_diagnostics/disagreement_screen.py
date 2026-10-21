# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/disagreement_screen.py
# ==============================================================================
# IS THERE A NON-PRIVILEGED SIGNAL OF WHERE THE ONE-STEP MODEL IS WRONG? (HANDOFF 6.2b, E1)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-21, HANDOFF 6.2b, user-approved): the proposed method (E2) uses epistemic uncertainty over
#    the one-step model twice: optimism in collection (direct trials to transitions whose outcome the model is
#    unsure about; Pathak et al. 2019, Sekar et al. 2020) and pessimism in acting (a margin scaled by the same
#    uncertainty; MOPO, Yu et al. 2020). Both need a signal that is (V) high at the declined turned latents, where
#    the observation model under-predicts glitter ~2x because the agent's data rarely reach them (feature_localise.py,
#    coverage_games.py), and (R) large where the death-risk prediction is wrong (risk_localise.py). This screen asks
#    whether the cheapest such signal, the disagreement of a bootstrapped ensemble of observation heads on the frozen
#    latent (Osband et al. 2016; Lakshminarayanan et al. 2017), has both properties.
#    - Caveat stated in advance (R1a, HANDOFF 7): head ensembles on one latent failed as PREDICTORS (the planner
#      exploited them). Here they are only a signal. A shared latent error makes all members agree; then the next
#      signal is whole-model ensembles (filter included), which are costlier.
#
# 2. Method (usage: uv run python <this> <run dir> [K]; networks = the run's best.pt acting copies, the ones that
#    selected the declined states in <run>/analysis/declined_states.pt):
#    - Data: the run's buffer up to best.pt's greedy block (feature_localise.py's slots), acting-filter latents
#      z_0..z_{T-1}, transitions (z_t, a_t, o_{t+1}) at live steps (no GAME_OVER before t). Games split 90/10
#      (generator seed 0, as in feature_localise.py); the held-out 10% are the in-distribution reference.
#    - Ensemble: K = 5 fresh MLP observation heads (z ++ onehot(a) -> 2 hidden layers of hidden_dim, SiLU -> |O|
#      logits), member j trained on a bootstrap resample of the training games (multinomial with replacement, seed j)
#      by cross-entropy, Adam 1e-3, batch 4096, early-stopped on held-out NLL (every 250 steps, patience 6).
#    - Signals at a = FORWARD: BALD mutual information MI(z) = H(mean_j p_j) - mean_j H(p_j) (Houlsby et al. 2011),
#      and sd_O(z) = the members' standard deviation of P(GAME_OVER | z, FORWARD).
#    - (V) AUC of MI between the declined TURNED latents (declined_states.pt z_turn; positives) and held-out live
#      buffer states (negatives). Secondary: against held-out states whose cell ahead is unvisited and not a wall
#      (novelty-matched negatives); and at the declined states, the rank correlation of MI with the joint head's
#      glitter error |P_joint(glitter) - P_exact(glitter)|.
#    - (R) On held-out live buffer states whose cell ahead is unvisited and has exact danger > 0 (exact layout
#      enumeration, as feature_localise.py's buffer_factors): Spearman of sd_O with the joint head's error
#      |P_joint(GAME_OVER | z, FORWARD) - P_exact(GAME_OVER | b, FORWARD)|; MI reported beside it.
#
# 3. Pre-registered readout (2026-10-21, before any full run; seeds 0-4 of runs/sweeps/wumpus_retrain):
#    - PASS: (V) AUC >= 0.75 AND (R) Spearman >= 0.3, both on >= 4 of 5 seeds -> the head-ensemble disagreement
#      is the E2 signal (with E0/E0' deciding which halves E2 needs).
#    - (V) passes, (R) fails -> use it for directed collection only; the acting margin needs another signal
#      (whole-model ensemble) or a fixed margin (E0').
#    - (V) fails -> the members share the latent's blind spot at the turned latents: whole-model ensembles next, or
#      a count-based novelty signal on the decodable "cell ahead unvisited" factor (Step D: it decodes at 0.97-0.98).
# ==============================================================================

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

from ipomdp.domain.wumpus import GAME_OVER, NUM_CELLS, Percept, WumpusAction, WumpusBeliefTracker, WumpusState
from ipomdp.agents.wumpus_reference import _ahead
from ipomdp.experiments.runs import load_trained_run
from ipomdp.training.checkpointing import load_checkpoint

run_dir = Path(sys.argv[1])
K = int(sys.argv[2]) if len(sys.argv) > 2 else 5
dev = torch.device("cuda")
trained = load_trained_run(run_dir, dev)
world, run, cfg = trained.setup.domain, trained.run, trained.cfg
model = run.eval_agent.model
FWD, NUM_A, NUM_O = int(WumpusAction.FORWARD), world.num_actions, world.num_observations
GLITTER = torch.tensor([bool(o & Percept.GLITTER) and o != GAME_OVER for o in range(NUM_O)], device=dev)
AHEAD = torch.tensor([[-1 if _ahead(c, h) is None else _ahead(c, h) for h in range(4)] for c in range(NUM_CELLS)],
                     device=dev)
t0 = time.time()

# --- Data: buffer slots as in feature_localise.py section 2 ---
buf = load_checkpoint(run_dir / "checkpoints" / "latest.pt")["buffer"]
size, t_cfg, c = buf["size"], cfg.training, trained.checkpoint_collection
if size != buf["actions"].shape[0] and buf["next"] != size:
    raise RuntimeError("the buffer wrapped around; insertion-order reconstruction does not apply.")
start = c * t_cfg.env_batch_size + (c // t_cfg.eval_every - 1) * t_cfg.eval_episodes
end = start + t_cfg.eval_episodes
if "greedy" in buf and (not bool(buf["greedy"][start:end].all()) or bool(buf["greedy"][start - 1])):
    raise RuntimeError(f"slot {start}..{end} is not collection {c}'s greedy block.")
actions, observations, initial = (buf[k][:end].to(dev) for k in ("actions", "observations", "initial_observations"))
del buf
games, steps = actions.shape

with torch.no_grad():
    z_all = torch.cat([model.belief_filter.unroll(F.one_hot(actions[i:i + 512], NUM_A).float(),
                                                  F.one_hot(observations[i:i + 512], NUM_O).float(),
                                                  F.one_hot(initial[i:i + 512], NUM_O).float())[:, :-1].float()
                       for i in range(0, games, 512)])                                  # (B, T, D)
ended = (observations == GAME_OVER).long().cumsum(1)
live = torch.cat([torch.ones(games, 1, dtype=torch.bool, device=dev), ended[:, :-1] == 0], 1)
split = torch.Generator().manual_seed(0)
held_games = torch.zeros(games, dtype=torch.bool)
held_games[torch.randperm(games, generator=split)[: games // 10]] = True
held_games = held_games.to(dev)
train_rows = (~held_games)[:, None] & live
held_rows = held_games[:, None] & live
print(f"== {run_dir.name}: buffer {games} games at collection {c}, {int(train_rows.sum())} train / "
      f"{int(held_rows.sum())} held-out live transitions; K = {K}; {time.time() - t0:.0f} s", flush=True)


# --- Ensemble ---
def head() -> nn.Module:
    d, h = z_all.shape[-1] + NUM_A, cfg.model.hidden_dim
    return nn.Sequential(nn.Linear(d, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, NUM_O)).to(dev)


def inputs(z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return torch.cat([z, F.one_hot(a, NUM_A).float()], -1)


x_held, y_held = inputs(z_all[held_rows], actions[held_rows]), observations[held_rows]
train_games = torch.nonzero(~held_games).squeeze(1)


def fit(member: int, batch: int = 4096, max_steps: int = 20000, every: int = 250, patience: int = 6) -> nn.Module:
    g = torch.Generator(device=dev).manual_seed(member)
    torch.manual_seed(member)
    picks = train_games[torch.randint(0, len(train_games), (len(train_games),), device=dev, generator=g)]
    rows = live[picks]
    x = inputs(z_all[picks][rows], actions[picks][rows])
    y = observations[picks][rows]
    net, best, best_state, bad = head(), float("inf"), None, 0
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for step in range(1, max_steps + 1):
        idx = torch.randint(0, len(x), (batch,), device=dev, generator=g)
        loss = F.cross_entropy(net(x[idx]), y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % every == 0:
            with torch.no_grad():
                held = float(F.cross_entropy(net(x_held), y_held))
            if held < best - 1e-5:
                best, best_state, bad = held, {k_: v.clone() for k_, v in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= patience:
                    break
    net.load_state_dict(best_state)
    print(f"  member {member}: early-stopped at step {step}, held-out NLL {best:.4f}; {time.time() - t0:.0f} s",
          flush=True)
    return net.eval().requires_grad_(False)


members = [fit(j) for j in range(K)]


def forward_probs(z: torch.Tensor) -> torch.Tensor:
    """Members' P(o' | z, FORWARD), (K, N, O)."""
    x = inputs(z, torch.full((len(z),), FWD, dtype=torch.int64, device=dev))
    with torch.no_grad():
        return torch.stack([F.softmax(m(x), -1) for m in members])


def entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p * p.clamp(min=1e-12).log()).sum(-1)


def signals(z: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    p = forward_probs(z)
    mi = entropy(p.mean(0)) - entropy(p).mean(0)
    return mi.double().cpu().numpy(), p[..., GAME_OVER].std(0).double().cpu().numpy()


def joint_forward(z: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return F.softmax(model.observation_head(z, F.one_hot(torch.full((len(z),), FWD, device=dev), NUM_A).float())
                         .float(), -1)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score(pos) > score(neg)) with ties at 1/2 (Mann-Whitney)."""
    scores = np.concatenate([pos, neg])
    ranks = np.empty(len(scores))
    ranks[np.argsort(scores, kind="mergesort")] = np.arange(1, len(scores) + 1)
    for v in np.unique(scores):                                   # average ranks of ties
        tie = scores == v
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# --- Exact P(o' | b, FORWARD) and the cell ahead at held-out buffer states (layout enumeration) ---
def exact_held(chunk: int = 64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tracker, layouts = WumpusBeliefTracker(world), torch.arange(world.num_layouts, device=dev)
    rows_g = torch.nonzero(held_games).squeeze(1)
    p_out, unvisited_out, wall_out = [], [], []
    for i in range(0, len(rows_g), chunk):
        gi = rows_g[i:i + chunk]
        consistent, own = tracker.start(initial[gi])
        g, num_l = consistent.shape
        visited = F.one_hot(own.cell, NUM_CELLS).bool()
        p_t, u_t, w_t = [], [], []
        for t in range(steps):
            expand = lambda v: v.unsqueeze(1).expand(g, num_l)  # noqa: E731
            hyp = WumpusState(layouts.expand(g, -1), expand(own.cell), expand(own.heading), expand(own.arrow),
                              expand(own.wumpus_alive), expand(own.done))
            weights = consistent.double()
            _, obs, _ = world.step_rules(hyp, torch.full((g, 1), FWD, dtype=torch.int64, device=dev))
            p = torch.zeros(g, NUM_O, dtype=torch.float64, device=dev).scatter_add_(1, obs, weights)
            p_t.append(p / weights.sum(1, keepdim=True))
            ahead = AHEAD[own.cell, own.heading]
            w_t.append(ahead < 0)
            u_t.append((ahead >= 0) & ~visited.gather(1, ahead.clamp(min=0).unsqueeze(1)).squeeze(1))
            consistent, own = tracker.update(consistent, own, actions[gi, t], observations[gi, t])
            visited = visited | F.one_hot(own.cell, NUM_CELLS).bool()
        p_out.append(torch.stack(p_t, 1))
        unvisited_out.append(torch.stack(u_t, 1))
        wall_out.append(torch.stack(w_t, 1))
    return torch.cat(p_out), torch.cat(unvisited_out), torch.cat(wall_out)


p_exact, unvisited, wall = exact_held()                                   # (G_held, T, O), (G_held, T) x 2
held_live = live[held_games]
z_held = z_all[held_games]
print(f"  exact held-out targets; {time.time() - t0:.0f} s", flush=True)

# --- (V) declined turned latents vs held-out buffer states ---
declined = torch.load(run_dir / "analysis" / "declined_states.pt", map_location=dev, weights_only=False)
z_turn = declined["z_turn"].float()
mi_turn, _ = signals(z_turn)
mi_neg, sd_neg = signals(z_held[held_live])
novel = (unvisited & ~wall)[held_live].cpu().numpy()
auc_v, auc_novel = auc(mi_turn, mi_neg), auc(mi_turn, mi_neg[novel])
glitter_err = (joint_forward(z_turn)[:, GLITTER].sum(1).double()
               - declined["p_fwd_turn"].to(dev)[:, GLITTER].sum(1).double()).abs().cpu().numpy()
rho_glitter = float(spearmanr(mi_turn, glitter_err).statistic)
print(f"  (V) MI median: declined turned {np.median(mi_turn):.4f}, held-out buffer {np.median(mi_neg):.4f}, novel "
      f"held-out {np.median(mi_neg[novel]):.4f}; AUC {auc_v:.3f} (novelty-matched {auc_novel:.3f}); Spearman with "
      f"the joint glitter error at the declined states {rho_glitter:.3f} ({len(mi_turn)} states)", flush=True)

# --- (R) risky new cells in held-out buffer states ---
p_exact_live = p_exact[held_live]
danger = p_exact_live[:, GAME_OVER].double()
risky = ((danger > 0) & (unvisited & ~wall)[held_live]).cpu().numpy()
p_joint = joint_forward(z_held[held_live])[:, GAME_OVER].double()
err = (p_joint - danger).abs().cpu().numpy()
rho_sd = float(spearmanr(sd_neg[risky], err[risky]).statistic)
rho_mi = float(spearmanr(mi_neg[risky], err[risky]).statistic)
calib = float(p_joint.cpu().numpy()[risky].mean() / danger.cpu().numpy()[risky].mean())
print(f"  (R) {int(risky.sum())} risky new-cell states: joint calibration {calib:.3f}; Spearman(sd_O, |error|) "
      f"{rho_sd:.3f}, Spearman(MI, |error|) {rho_mi:.3f}", flush=True)

summary = {"collection": c, "K": K, "auc_v": auc_v, "auc_v_novel": auc_novel, "rho_glitter": rho_glitter,
           "declined_states": len(mi_turn), "risky_states": int(risky.sum()), "rho_r_sd": rho_sd, "rho_r_mi": rho_mi,
           "joint_risk_calibration": calib, "pass_v": auc_v >= 0.75, "pass_r": rho_sd >= 0.3}
(run_dir / "analysis" / "disagreement_screen.json").write_text(json.dumps(summary, indent=2))
print(f"  readout: (V) {'pass' if summary['pass_v'] else 'fail'}, (R) {'pass' if summary['pass_r'] else 'fail'}; "
      f"{time.time() - t0:.0f} s\nDONE", flush=True)
