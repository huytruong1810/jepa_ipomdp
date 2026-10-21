# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/ensemble_disagreement.py
# ==============================================================================
# DOES WHOLE-MODEL DISAGREEMENT (FILTER INCLUDED) TRACK WHERE THE ONE-STEP MODEL IS WRONG? (HANDOFF 6.2c, C0 / C1)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-21 evening, user decision: option (c)):
#    - Every idling cure so far (cover / union data, coverage_games.py) brings the deaths back. At the risky
#      FORWARDs the planner chooses, the predicted death risk is 2-5% of the exact one, although the AVERAGE risk
#      calibration is fine (union 0.83-0.94): the optimizer's curse (Smith & Winkler 2006) at the argmax. A uniform
#      x2/x4 margin cannot reach a risk rated at ~3% of exact (pessimism_play.py, E0' FAIL), and a bootstrapped
#      ensemble of observation HEADS on one frozen latent measures novelty, not error (disagreement_screen.py, E1
#      FAIL: the members share the latent's blind spot).
#    - The remaining candidate: disagreement between WHOLE models. Each member filters the same history with its
#      OWN filter and predicts P(o' | h, FORWARD) and R(h, FORWARD) from its own latent. If the latent itself is
#      wrong in a state, independently trained filters should disagree there (deep ensembles / bootstrapped models
#      as epistemic uncertainty: Lakshminarayanan et al. 2017; Osband et al. 2016; PETS, Chua et al. 2018). A signal
#      that tracks the error would scale a state-dependent risk margin (pessimism, MOPO, Yu et al. 2020) and direct
#      collection (optimism, Sekar et al. 2020), the C2 method.
#
# 2. Method (usage: uv run python <this> <run dir> --act <set.pt> --members <a.pt> <b.pt> ... --tag <name>):
#    - Network sets are saved retrains (objective_ab.py --save); each is loaded into the run's networks (acting
#      copies included, as risk_localise.py does) and read with the acting filter and model.expand.
#    - Histories (no planner replay): the Wumpus simulator's only randomness is its layout generator, so replaying
#      recorded actions on the analysis stream (stream_seed(settings.seed, 1)) rebuilds the exact games.
#      (R) the acting set's own 512 analysis games, actions from <act>.risk.npz (risk_localise.py); the replayed
#      returns must equal <act>.returns.npy. (V) best.pt's games (analysis/exact_play_learned.npz, learned arm),
#      whose declined states are in analysis/declined_states.pt.
#    - (R) At every decision t of the acting set's games, member m's risk p_m = max(P_m(GAME_OVER | z_t, FORWARD),
#      p_R), p_R = -(R_m(z_t, FORWARD) + 1) / 1000 (FORWARD pays -1, a death -1001; as risk_localise.py). mean and
#      sd (unbiased) over members; p_act = the acting set's own (must equal its risk.npz p_o / p_r). Exact danger
#      of the cell ahead, unvisited, fatal and the chosen action come from <act>.risk.npz.
#      - rho_sd: Spearman(sd, |p_act - danger|) on risky new cells (live, danger > 0, cell ahead unvisited).
#        Reference (not a gate): Spearman(p_act, |error|), since a spread that only scales with the prediction
#        would be a trivial signal.
#      - cover_fatal(beta): share of fatal FORWARDs where mean + beta x sd >= 0.5 x danger (the margin that would
#        have stopped them; risk_localise.py's "under-rated" bar); also over the chosen risky FORWARDs, and with
#        p_act + beta x sd. beta in {1, 2}.
#      - phantom(beta): mean of (mean + beta x sd) over safe new cells (danger 0, unvisited) / mean of p_act there:
#        the cost, a phantom risk that would re-create the idling.
#      - When the acting set is itself a member (C0), the sd is also reported over the other members only
#        (leave-acting-out): with the acting model inside, part of the sd-error correlation is mechanical.
#    - (V) At the declined states: member m's z_root = its filter on the history up to (game, t); z_turn =
#      expand(z_root).next_states[turn, o_turn] (feature_localise.py); P_m(glitter | z_turn, FORWARD). rho_v:
#      Spearman(sd of P(glitter), |P_act(glitter) - P_exact(glitter)|). For every member with a
#      declined_states_<stem>.pt from feature_localise.py --networks, the recomputed z_turn must equal its.
#    - Outputs: <run>/analysis/ensemble_disagreement_<tag>.json and .npz (per-row arrays).
#
# 3. Pre-registered readout (2026-10-21 evening, before any run).
#    - C0 (exploratory; members = the four coverage sets control / cover / own / union of a seed, acting = union,
#      then cover): the sets differ by their extra data, not only by chance, so their spread mixes data differences
#      with epistemic uncertainty. Same thresholds as C1, read as exploratory only; C1 runs whatever C0 shows.
#    - C1 (registered; members = K = 5 bootstrapped whole-model retrains of the CONTROL data per seed,
#      objective_ab.py section 2k, boot<j>_seed<s>.pt; acting = union (the idling cure, not a member), and
#      secondary acting = control):
#      - (R) per seed: rho_sd >= 0.3 AND for some beta in {1, 2}: cover_fatal(beta) >= 0.5 with phantom(beta) < 2,
#        with the margin mean + beta x sd. Seeds where the acting set has < 5 fatal FORWARDs do not count for the
#        coverage half (reported, pooled over seeds).
#      - (V) per seed: rho_v >= 0.3 (acting = union and control both reported; union decides).
#      - PASS = (R) and (V) each on >= 4 of 5 seeds with acting = union (an interim on seeds 0-2 needs 3 of 3 and is
#        labelled interim) -> propose C2 to the user (a K-model search wrapper with an uncertainty-scaled margin,
#        screened offline on union vs control; optimism in collection with the same signal).
#      - (R) passes, (V) fails -> C2 as pessimism only (the margin); collection needs another signal.
#      - (V) passes, (R) fails -> the signal can direct collection but not the acting margin; report to the user.
#      - Both fail -> whole-model disagreement does not find the latent's blind spots either; back to the user with
#        option (a) (own trials + fixed margin) and the representation branch (HANDOFF 1b candidate 1).
#    - AMENDMENT (2026-10-21 17:05, after C0, BEFORE any C1 measurement; the C1 retrains were still on seed 0):
#      C0 shows that the margin mean + beta x sd covers union's fatal FORWARDs mostly through the MEAN (the other
#      members rate the risk higher than union does), not through the spread: p_act + beta x sd covers only 11-41%.
#      With control-data members acting on union's decisions, the mean is control's prediction: a data
#      difference again, not an uncertainty signal. The clean C1 test is the one where the members share the
#      actor's data, as a C2 ensemble would. So:
#      - (R) PRIMARY = acting control (its own decisions; 14-97 fatal FORWARDs per seed, so coverage counts on every
#        seed), members = the 5 boot sets (control not a member), margin mean + beta x sd, thresholds as above.
#      - Union acting is SECONDARY (reported with both margins; the p_act + beta x sd column says whether the
#        spread alone reaches union's under-rated risks). C2 would then need union-data bootstraps.
#      - (V) is read on both actors; PASS needs it with acting union (the declined states and their glitter error
#        are union-independent; only the error term changes).
#
# 4. C0 result (2026-10-21; runs/wumpus/coverage/c0_*_seed*.log; members = control / cover / own / union, every
#    identity check passes; ~75 s per run). Exploratory by design. **(R) 3 of 5 (union acting), 0 of 5 (cover
#    acting); (V) 0 of 5 for both.**
#    | Seed | act   | fatal | rho_sd (loo) | rho(p_act, err) | cover b1 / phantom b1 | cover b2 / phantom b2 | p_act + sd cover b1 | rho_v |
#    | 0    | union | 84    | 0.65 (0.28)  | -0.35           | 0.62 / 1.65           | 0.76 / 2.17           | 0.30                | +0.08 |
#    | 1    | union | 27    | 0.49 (0.54)  | -0.29           | 0.70 / 1.58           | 0.89 / 2.11           | 0.11                | -0.21 |
#    | 2    | union | 80    | 0.46 (0.49)  | -0.40           | 0.51 / 1.62           | 0.76 / 2.39           | 0.14                | -0.00 |
#    | 3    | union | 56    | 0.52 (0.48)  | -0.14           | 0.68 / 2.33           | 0.71 / 3.49           | 0.41                | -0.07 |
#    | 4    | union | 103   | 0.53 (0.41)  | -0.09           | 0.49 / 2.47           | 0.67 / 3.53           | 0.16                | -0.29 |
#    | 0-4  | cover | 140-220 | -0.30 to 0.68 | -0.39 to -0.79 | 0.72-0.83 / 1.84-5.42 | 0.78-0.98 / 2.45-7.88 | 0.22-0.48         | -0.21 to +0.07 |
#    - (R): the spread ranks the risk error on risky new cells (rho 0.46-0.65 with union acting, 0.28-0.54 leaving
#      union out), unlike E1's head ensemble (<= 0.23). But the coverage comes mostly from the mean shift between
#      differently-trained sets, which is the confound stated in the amendment; with cover acting the phantom risk
#      explodes (the cover networks' risk is low everywhere, so any mean shift is large relative to it).
#    - (V): the spread of P(glitter) at the declined turned latents is unrelated to the glitter error (rho -0.29 to
#      +0.08, leave-acting-out the same) although union's glitter is repaired relative to control's: like E1, no
#      sign that disagreement finds the value-of-information blind spot.
# ==============================================================================

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from ipomdp.domain import WumpusSimulator
from ipomdp.domain.wumpus import GAME_OVER, Percept, WumpusAction
from ipomdp.experiments.runs import load_trained_run
from ipomdp.experiments.wumpus import WumpusAnalysisSettings
from ipomdp.training import discounted_returns, play_episodes, stream_seed

parser = argparse.ArgumentParser()
parser.add_argument("run_dir", type=Path)
parser.add_argument("--act", type=Path, required=True, help="The acting network set (its .risk.npz holds the games).")
parser.add_argument("--members", type=Path, nargs="+", required=True)
parser.add_argument("--tag", required=True)
args = parser.parse_args()
if len(args.members) < 2:
    raise ValueError("an ensemble needs at least two members.")
dev = torch.device("cuda")
settings = WumpusAnalysisSettings()
trained = load_trained_run(args.run_dir, dev)
world, run, cfg = trained.setup.domain, trained.run, trained.cfg
model = run.eval_agent.model
FWD, NUM_A, NUM_O = int(WumpusAction.FORWARD), world.num_actions, world.num_observations
GLITTER = torch.tensor([bool(o & Percept.GLITTER) and o != GAME_OVER for o in range(NUM_O)], device=dev)
CHUNK = 2048
BETAS = (1.0, 2.0)
t0 = time.time()


def log(msg: str) -> None:
    print(f"  [{time.time() - t0:5.0f} s] {msg}", flush=True)


class Replay:
    """Plays recorded actions (the simulator's only randomness is the layout draw at reset)."""

    def __init__(self, actions: torch.Tensor):
        self.actions, self.batch_size, self.t = actions, actions.shape[0], 0

    def reset(self, initial_observation):
        self.t = 0

    def act(self):
        self.t += 1
        return self.actions[:, self.t - 1]

    def update(self, action, observation):
        pass


def replay(actions: np.ndarray, reference_returns: np.ndarray, what: str):
    acts = torch.as_tensor(actions, dtype=torch.int64, device=dev)
    sim = WumpusSimulator(world, acts.shape[0], cfg.env.max_steps, stream_seed(settings.seed, 1), dev)
    episodes, _ = play_episodes(sim, Replay(acts))
    returns = discounted_returns(episodes.rewards, world.discount).double().cpu().numpy()
    if np.abs(returns - reference_returns).max() > 1e-6:
        raise RuntimeError(f"{what}: replayed returns differ from the recorded ones.")
    return episodes


def load_set(path: Path) -> None:
    states = torch.load(path, map_location=dev)
    if set(states) != set(run.networks):
        raise ValueError(f"{path}: networks {sorted(states)} do not match the run's {sorted(run.networks)}.")
    for name, net in run.networks.items():
        net.load_state_dict(states[name])


def latents(episodes) -> torch.Tensor:
    """Acting-filter latents z_0..z_T (n, T + 1, D), float32, no autocast."""
    with torch.no_grad():
        return run.acting_filter.unroll(F.one_hot(episodes.actions, NUM_A).float(),
                                        F.one_hot(episodes.observations, NUM_O).float(),
                                        F.one_hot(episodes.initial_observations, NUM_O).float()).float()


def forward_risk(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(P(GAME_OVER | z, FORWARD), p_R) for flat latents (N, D), float64."""
    p_o, p_r = [], []
    for i in range(0, len(z), CHUNK):
        with torch.no_grad():
            e = model.expand(z[i:i + CHUNK])
        p_o.append(e.observation_probs[:, FWD, GAME_OVER].double())
        p_r.append(-(e.rewards[:, FWD].double() + 1.0) / 1000.0)
    return torch.cat(p_o), torch.cat(p_r)


def turned_glitter(z_root: torch.Tensor, turn: torch.Tensor, o_turn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z_turn, glitter = [], []
    for i in range(0, len(z_root), CHUNK):
        with torch.no_grad():
            e = model.expand(z_root[i:i + CHUNK])
            idx = torch.arange(len(e.next_states), device=dev)
            z_t = e.next_states[idx, turn[i:i + CHUNK], o_turn[i:i + CHUNK]]
            glitter.append(model.expand(z_t).observation_probs[:, FWD][:, GLITTER].sum(1).double())
        z_turn.append(z_t)
    return torch.cat(z_turn), torch.cat(glitter)


# --- Histories ---
risk = np.load(args.act.with_suffix(".risk.npz"))
act_games = replay(risk["action"], np.load(args.act.with_suffix(".returns.npy")), f"{args.act} games")
best_play = np.load(args.run_dir / "analysis" / "exact_play_learned.npz")
best_games = replay(best_play["actions"], best_play["returns"], "best.pt games")
declined = torch.load(args.run_dir / "analysis" / "declined_states.pt", map_location=dev, weights_only=False)
d_game = torch.tensor([m["game"] for m in declined["meta"]], device=dev)
d_t = torch.tensor([m["t"] for m in declined["meta"]], device=dev)
d_turn = torch.tensor([m["turn"] for m in declined["meta"]], device=dev)
d_o = declined["o_turn"].to(dev)
glitter_exact = declined["p_fwd_turn"].to(dev)[:, GLITTER].sum(1).double()
G, T = risk["action"].shape
log(f"{args.run_dir.name}: acting {args.act.name} ({G} games), {len(d_game)} declined states, "
    f"{len(args.members)} members; histories replayed")


def member_quantities(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    load_set(path)
    z = latents(act_games)[:, :-1].reshape(G * T, -1)
    p_o, p_r = forward_risk(z)
    z_turn, glitter = turned_glitter(latents(best_games)[d_game, d_t], d_turn, d_o)
    saved = args.run_dir / "analysis" / f"declined_states_{path.stem}.pt"
    if saved.exists():
        z_saved = torch.load(saved, map_location=dev, weights_only=False)["z_turn"].float()
        if (z_saved - z_turn).abs().max() > 1e-4:
            raise RuntimeError(f"{path}: recomputed turned latents differ from {saved}.")
    return p_o.view(G, T), p_r.view(G, T), glitter


p_o_act, p_r_act, glitter_act = member_quantities(args.act)
live = torch.as_tensor(risk["live"], device=dev)
for name, mine, recorded in (("p_o", p_o_act, risk["p_o"]), ("p_r", p_r_act, risk["p_r"])):
    gap = (mine - torch.as_tensor(recorded, device=dev))[live].abs().max()
    if gap > 1e-4:
        raise RuntimeError(f"acting set: {name} differs from its risk.npz by {float(gap):.2e}.")
p_act = torch.maximum(p_o_act, p_r_act)
log("acting set reproduces its risk.npz")

p_members, g_members = [], []
act_index = None
for j, path in enumerate(args.members):
    if path.resolve() == args.act.resolve():
        act_index = j
        p_members.append(p_act)
        g_members.append(glitter_act)
        continue
    p_o, p_r, glitter = member_quantities(path)
    p_members.append(torch.maximum(p_o, p_r))
    g_members.append(glitter)
    log(f"member {path.name} done")
P, Gl = torch.stack(p_members), torch.stack(g_members)          # (M, G, T), (M, N_declined)

# --- Measures ---
danger = torch.as_tensor(risk["danger"], device=dev)
unvisited = torch.as_tensor(risk["unvisited"], device=dev)
action = torch.as_tensor(risk["action"], device=dev)
fatal = torch.as_tensor(risk["fatal"], device=dev)
risky = live & (danger > 1e-9)
risky_new = risky & unvisited
safe_new = live & (danger <= 1e-9) & unvisited
chosen = risky & (action == FWD)
err = (p_act - danger).abs()


def rho(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(spearmanr(x.cpu().numpy(), y.cpu().numpy()).statistic) if len(x) > 2 else float("nan")


def share(mask: torch.Tensor, margin: torch.Tensor) -> float:
    return float((margin[mask] >= 0.5 * danger[mask]).double().mean()) if bool(mask.any()) else float("nan")


def r_measures(members: torch.Tensor) -> dict:
    mean, sd = members.mean(0), members.std(0)
    out = {"rho_sd": rho(sd[risky_new], err[risky_new]), "rho_mean_err": rho(mean[risky_new], err[risky_new])}
    for b in BETAS:
        margin, margin_act = mean + b * sd, p_act + b * sd
        out[f"cover_fatal_b{b:g}"] = share(fatal, margin)
        out[f"cover_fatal_act_b{b:g}"] = share(fatal, margin_act)
        out[f"cover_chosen_b{b:g}"] = share(chosen, margin)
        out[f"cover_chosen_act_b{b:g}"] = share(chosen, margin_act)
        out[f"phantom_b{b:g}"] = float(margin[safe_new].mean() / p_act[safe_new].mean())
        out[f"phantom_act_b{b:g}"] = float(margin_act[safe_new].mean() / p_act[safe_new].mean())
    return out


summary = {
    "act": str(args.act), "members": [str(p) for p in args.members], "act_is_member": act_index is not None,
    "risky_new_states": int(risky_new.sum()), "safe_new_states": int(safe_new.sum()),
    "fatal_forward": int(fatal.sum()), "chosen_risky_forward": int(chosen.sum()),
    "rho_p_act_err": rho(p_act[risky_new], err[risky_new]),
    "cover_fatal_b0": share(fatal, p_act), "cover_chosen_b0": share(chosen, p_act),
    "phantom_abs_b0": float(p_act[safe_new].mean()),
}
summary.update(r_measures(P))
if act_index is not None:
    others = torch.cat([P[:act_index], P[act_index + 1:]])
    summary.update({f"loo_{k}": v for k, v in r_measures(others).items()} if len(others) >= 2 else {})
sd_g = Gl.std(0)
glitter_err = (glitter_act - glitter_exact).abs()
summary["rho_v"] = rho(sd_g, glitter_err)
summary["rho_v_mean_err"] = rho(sd_g, (Gl.mean(0) - glitter_exact).abs())
summary["glitter_ratio_act"] = float(glitter_act.mean() / glitter_exact.mean())
summary["glitter_ratio_mean"] = float(Gl.mean(0).mean() / glitter_exact.mean())
if act_index is not None:
    others_g = torch.cat([Gl[:act_index], Gl[act_index + 1:]])
    summary["loo_rho_v"] = rho(others_g.std(0), glitter_err) if len(others_g) >= 2 else float("nan")
coverage_counts = summary["fatal_forward"] >= 5
summary["pass_r"] = bool(summary["rho_sd"] >= 0.3 and (not coverage_counts or any(
    summary[f"cover_fatal_b{b:g}"] >= 0.5 and summary[f"phantom_b{b:g}"] < 2 for b in BETAS)))
summary["coverage_counts"] = bool(coverage_counts)
summary["pass_v"] = bool(summary["rho_v"] >= 0.3)

out = args.run_dir / "analysis" / f"ensemble_disagreement_{args.tag}"
out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
np.savez(out.with_suffix(".npz"), p_members=P.cpu().numpy(), glitter_members=Gl.cpu().numpy(),
         glitter_exact=glitter_exact.cpu().numpy(), p_act=p_act.cpu().numpy())
print("  " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in summary.items()
                       if k not in ("act", "members")), flush=True)
print(f"  readout: (R) {'pass' if summary['pass_r'] else 'fail'}, (V) {'pass' if summary['pass_v'] else 'fail'}"
      f"{'' if coverage_counts else ' (coverage half not counted: < 5 fatal FORWARDs)'}\nDONE", flush=True)
