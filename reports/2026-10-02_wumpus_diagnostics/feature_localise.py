# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/feature_localise.py
# ==============================================================================
# WHICH FACTOR OF THE EXACT P(o' | b, FORWARD) IS MISSING FROM THE LATENT, AND IS IT MISSING ONLY AFTER A TURN?
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-19, HANDOFF 6.2 Step D; pre-authorised, algorithm-design phase):
#    - Step A (obs_fit_vs_latent.py) read LATENT: at the planner's declined distance-2 near-turn states no head on
#      the frozen acting latent recovers the exact P(o' | turned latent, FORWARD) (glitter ratio ~0.5, TV ~0.3).
#      Step B (exact_model_play.py) priced the learned one-step model at ~+208 per game.
#    - P(o' | b, FORWARD) factorises over the cell ahead in the hunter's heading: whether it is a wall, whether it
#      is unvisited, and the posterior P(gold), P(pit), P(live wumpus) there; glitter after FORWARD is exactly
#      (1 - wall) * P(gold ahead) + wall * P(gold here) (asserted). Which factor the latent lacks, and where,
#      picks the method branch (HANDOFF 1b candidates):
#        D1 SHIFT        -> the factors decode on ordinary buffer states but not at the turned latents: imagined /
#                           counterfactual latents are off the filter's training support. Remedy: coverage of
#                           decision-relevant counterfactuals, or training the model on its own branches (cand. 4).
#        D2 COMPOSITION  -> the factors decode at the turned latents too, yet P(glitter) does not (Step A): the
#                           information is there and the observation model fails to compose it. Remedy: decision-
#                           aware weighting (VaGraM) or a Q-equivalent term (cand. 2-3).
#        D3 REPRESENTATION -> a glitter-driving factor does not decode even on buffer states: the latent lacks it.
#                           Remedy: a self-predictive latent on the retrain recipe (Ni et al. 2024; cand. 1), then an
#                           architecture A/B.
#
# 2. Method (usage: uv run python <this> <run dir> <cpu|cuda> [games]; then <this> summary <run dir> ...):
#    - Test states: the 512 analysis games of best.pt replayed exactly (return must equal report.json); the declined
#      distance-2 states selected as in q_localise_diag.py (Recorder copied; count and leaf-only q1_chosen must
#      equal macro_reach.npz). The turned latent is the planner's own child z1 = expand(z)[turn, o_turn] with the
#      exact (deterministic) turn outcome; the joint head's TV at it must reproduce q_localise.npz's tv_fwd.
#      With fewer than 512 games (smoke) the cross-checks against saved files are skipped.
#    - Factors (exact layout-enumeration filter; always the cell ahead in the latent's OWN heading, so at the
#      root latent the root heading and at the turned latent the turned heading): wall ahead, unvisited ahead
#      (binary); P(gold), P(pit), P(live wumpus) ahead (0 at a wall); P(glitter), P(breeze), P(stench) in o' after
#      FORWARD from the exact P(o' | b, FORWARD). Visited cells come from the exact own-cell trajectory (start cell
#      visited, as in WumpusReferenceAgent).
#    - Training data: the buffer slots of latest.pt that existed at best.pt's collection (obs_fit_vs_latent.py's
#      slot logic), acting latents z_0..z_{T-1}, live (not done) states only; the same 10% game split (seed 0).
#    - Probes (8 sigmoid outputs, soft binary cross-entropy, Adam 1e-3, minibatch 4096, held-out loss every 250
#      steps, best state kept, stop after 6 evaluations without improvement, at most 12,000 steps):
#        mlp     2 hidden layers of the agent's hidden_dim, SiLU.
#        linear  one linear layer (capacity control).
#        random  the mlp on the latents of a freshly initialised filter of the same architecture (selectivity
#                control, Hewitt & Liang 2019); scored on (a) only.
#    - Evaluation sets: (a) held-out buffer states; (b) the declined ROOT latents (on-trajectory, fresh games);
#      (c) the declined TURNED latents. Binary factors: accuracy (prediction > 0.5) against the majority class of
#      the TRAINING buffer; probability factors: mean abs error against the training-buffer mean as a constant.
#      Also the calibration ratio mean(pred) / mean(exact), and the composed glitter (1 - wall^) * gold^ at (c).
#    - --networks <saved .pt> (coverage_games.py readout (i), 2026-10-19): the states are still selected by the
#      best.pt replay; then every network is replaced by the saved retrained set, the root latents are recomputed
#      from the replayed histories with its acting filter (asserted equal to the planner's own with best.pt), and
#      everything below uses the retrained networks. The probe data stay the same base-buffer slots for every
#      arm. Adds q_localise's decision share M at (c) (the joint arm must reproduce q_localise.npz with best.pt).
#      Outputs carry the suffix _<saved stem> (e.g. _cover_seed0); compare mode: <this> compare control cover <runs...>
#      reads feature_localise_<arm>_<run name>.json.
#    - Outputs in <run>/analysis/: feature_localise.json (summary), feature_localise.npz (per-row predictions and
#      targets at (b) and (c)), declined_states.pt (root and turned latents, o_turn, exact P(o' | FORWARD) at both,
#      factors, row metadata) so later method screens at these states skip the 512-game replay.
#
# 3. Pre-registered readout (written 2026-10-19 before any run; 5 seeds of runs/sweeps/wumpus_retrain, best.pt,
#    the mlp probe). "Decodable" on a set: binary factor accuracy >= 90% and >= 5 points over the training
#    majority; probability factor mean abs error <= 0.05 and <= half the constant's. Glitter-driving factors:
#    wall ahead, unvisited ahead, P(gold ahead).
#    - D3 REPRESENTATION: some glitter-driving factor not decodable on (a) on >= 4 of 5 seeds.
#    - D1 SHIFT: every glitter-driving factor decodable on (a) and some not decodable on (c), on >= 4 of 5 seeds.
#      Sub-reading: also not on (b) -> a fresh-game shift, not turn-specific; decodable on (b) -> the turn step
#      leaves the support.
#    - D2 COMPOSITION/OBJECTIVE: every glitter-driving factor decodable on (a) and (c), the composed glitter
#      ratio at (c) in [0.67, 1.5], and Step A's arm (iv) glitter ratio < 0.67, on >= 4 of 5 seeds.
#    - Otherwise MIXED: report the per-seed table and decide with the user.
#    - Pre-launch correction (2026-10-19, after the 8-game smoke, before any full run): at (b) and (c) the binary
#      targets are near-constant by the selection (the cell ahead after the turn is never a wall), so the training
#      majority is trivially right there and the "+5 points over majority" margin could never pass. The margin is
#      applied on (a) only; on (b) and (c) a binary factor needs accuracy >= 90%.
#    - Sanity: if the random-filter probe is decodable on (a) for every glitter-driving factor, the probe metric
#      does not select for learned information; report it and do not read the branch.
#
# 4. Result 2026-10-19 (runs/sweeps/wumpus_retrain seeds 0-4, best.pt, GPU; replays = report.json, states =
#    macro_reach.npz, joint-head TV = q_localise.npz on every seed; logs runs/wumpus/feature_localise_seed{0..4}.log,
#    readout runs/wumpus/feature_localise_summary.log; 10-15 min per seed). **Readout: MIXED** (D3 on 3 of 5 seeds,
#    D1 on 2 of 5, D2 on 0; the random-filter control is selective on 5 of 5). mlp probe, per seed 0-4:
#    | Factor                 | (a) held-out buffer          | (b) declined roots           | (c) turned latents            |
#    | unvisited ahead (acc)  | .97 / .90 / .98 / .89 / .90  | .93 / .94 / .96 / .91 / .90  | .72 / .77 / .81 / .77 / .65   |
#    | wall ahead (acc)       | .99 / .93 / .99 / .92 / .93  | .97 / .96 / .98 / .95 / .90  | .97 / .96 / .99 / .92 / .96   |
#    | P(gold ahead) MAE      | .009/.014/.009/.014/.014     | .018/.014/.015/.014/.024     | .069/.054/.057/.077/.065      |
#    | P(gold ahead) mean ratio| .99/1.05/.94/.97/1.09       | (exact ~0 there)             | .59 / .59 / .58 / .51 / .55   |
#    (constant baselines: unvisited 51-53% on (a); gold MAE .034-.035 on (a), .088-.097 on (c); random filter on (a):
#    unvisited 70-73%, gold .024-.026; linear probe on (a): unvisited 80-95%.)
#    - D3 fires on seeds 1, 3, 4 by "unvisited" alone at (a), at 0.900 / 0.890 / 0.896 against the 0.90 bar (the
#      seeds whose best.pt is earliest: collections 55 / 85 / 65). D1 fires on seeds 0 and 2.
#    - The pattern that holds on 5 of 5 seeds: P(gold ahead) is decodable and calibrated on buffer states (and on
#      the declined roots), but at the turned latents it is under-rated ~2x (ratio 0.51-0.59), and the turned latent
#      reads the (always unvisited) cell ahead as visited in 19-35% of the states. The composed glitter
#      (1 - wall^) gold^ (0.49-0.58 of exact) equals the direct probe (0.51-0.61): composition is NOT the failure
#      (D2 refuted on 5 of 5); the factors themselves are wrong after the turn. The probe recovers about the same
#      glitter as Step A's arm (iv) (0.49-0.62), so Step A's "not decodable" is this shift.
#    - Reading: mostly a support SHIFT specific to the declined turn (the latent after "turn towards the unvisited
#      safe cell" at states where the agent itself declines is rare in its data), plus a weaker, marginal
#      representation deficit in "visited" tracking on the three weaker seeds. Consistent with the feedback-loop
#      conjecture of HANDOFF 1b (idle -> no such data -> the model under-rates new cells -> idle). Decided with the
#      user per section 3.
# ==============================================================================

import copy
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FACTORS = ("wall", "unvisited", "gold", "pit", "wumpus", "glitter", "breeze", "stench")
BINARY = {"wall", "unvisited"}
DRIVERS = ("wall", "unvisited", "gold")


def decodable(m: dict, name: str, key: str) -> bool:
    """Section 3's per-factor rule on one evaluation set's metrics (the majority margin on set (a) only)."""
    if name in BINARY:
        return m["acc"] >= 0.90 and (key != "a" or m["acc"] >= m["acc_majority"] + 0.05)
    return m["mae"] <= 0.05 and m["mae"] <= 0.5 * m["mae_constant"]


if sys.argv[1] == "compare":
    # coverage_games.py section 3: cover - control at (c), per seed.
    control_stem, cover_stem = sys.argv[2], sys.argv[3]
    for run in map(Path, sys.argv[4:]):
        a = {stem: json.loads((run / "analysis" / f"feature_localise_{stem}_{run.name}.json").read_text())
             for stem in (control_stem, cover_stem)}
        get = lambda stem, k: a[stem][k]  # noqa: E731
        d_glitter = get(cover_stem, "joint_glitter_ratio_c") - get(control_stem, "joint_glitter_ratio_c")
        d_share = get(cover_stem, "decision_share_M") - get(control_stem, "decision_share_M")
        print(f"{run.name}: joint glitter ratio {get(control_stem, 'joint_glitter_ratio_c'):.2f} -> "
              f"{get(cover_stem, 'joint_glitter_ratio_c'):.2f} ({d_glitter:+.2f}); decision share M "
              f"{get(control_stem, 'decision_share_M'):.3f} -> {get(cover_stem, 'decision_share_M'):.3f} "
              f"({100 * d_share:+.1f} pts); TV median {get(control_stem, 'joint_tv_median_c'):.3f} -> "
              f"{get(cover_stem, 'joint_tv_median_c'):.3f}; probe (c) gold ratio "
              f"{a[control_stem]['mlp']['c']['gold']['calibration']:.2f} -> {a[cover_stem]['mlp']['c']['gold']['calibration']:.2f}, "
              f"unvisited {a[control_stem]['mlp']['c']['unvisited']['acc']:.3f} -> "
              f"{a[cover_stem]['mlp']['c']['unvisited']['acc']:.3f}; (a) unvisited "
              f"{a[control_stem]['mlp']['a']['unvisited']['acc']:.3f} -> {a[cover_stem]['mlp']['a']['unvisited']['acc']:.3f}"
              f"; MECHANISM {'pass' if d_glitter >= 0.15 and d_share >= 0.10 else 'fail'}")
    sys.exit(0)

if sys.argv[1] == "summary":
    # Section 3's readout over the seeds' feature_localise.json files.
    rows, verdicts = [], []
    for run in map(Path, sys.argv[2:]):
        s = json.loads((run / "analysis" / "feature_localise.json").read_text())
        ok = {k: {f: decodable(s["mlp"][k][f], f, k) for f in FACTORS} for k in ("a", "b", "c")}
        rnd = all(decodable(s["random"]["a"][f], f, "a") for f in DRIVERS)
        step_a = json.loads((run / "analysis" / "obs_fit_vs_latent.json").read_text())["arms"]["(iv) all-soft"]
        d = {"D3": not all(ok["a"][f] for f in DRIVERS),
             "D1": all(ok["a"][f] for f in DRIVERS) and not all(ok["c"][f] for f in DRIVERS),
             "D2": all(ok["a"][f] and ok["c"][f] for f in DRIVERS)
                   and 0.67 <= s["composed_glitter_ratio_c"] <= 1.5 and step_a["glitter_ratio"] < 0.67,
             "D1_root_ok": all(ok["b"][f] for f in DRIVERS), "random_selective": not rnd}
        verdicts.append(d)
        print(f"{run.name}: " + "; ".join(
            f"({k}) " + " ".join(f"{f}{'+' if ok[k][f] else '-'}" for f in FACTORS) for k in ("a", "b", "c"))
            + f"; composed glitter ratio (c) {s['composed_glitter_ratio_c']:.2f}, Step A (iv) "
              f"{step_a['glitter_ratio']:.2f}; " + ", ".join(f"{k} {v}" for k, v in d.items()))
    count = {k: sum(v[k] for v in verdicts) for k in ("D3", "D1", "D2", "D1_root_ok", "random_selective")}
    need = 4 if len(verdicts) == 5 else len(verdicts)
    branch = next((k for k in ("D3", "D1", "D2") if count[k] >= need), "MIXED")
    if count["random_selective"] < len(verdicts):
        branch += " (random-filter control NOT selective on some seed: do not read the branch)"
    print(f"counts {count}; readout {branch}")
    sys.exit(0)

from ipomdp.agents import PlanningAgent, WumpusReferenceAgent  # noqa: E402
from ipomdp.agents.wumpus_reference import _ZERO, _ahead, _first_actions  # noqa: E402
from ipomdp.domain import WumpusSimulator  # noqa: E402
from ipomdp.domain.wumpus import GAME_OVER, NUM_CELLS, Percept, WumpusAction, WumpusBeliefTracker, WumpusState  # noqa: E402
from ipomdp.experiments.runs import load_trained_run  # noqa: E402
from ipomdp.experiments.wumpus import WumpusAnalysisSettings  # noqa: E402
from ipomdp.planning import BeliefTreeSearch  # noqa: E402
from ipomdp.training import discounted_returns, play_episodes, stream_seed  # noqa: E402
from ipomdp.training.checkpointing import load_checkpoint  # noqa: E402

args = sys.argv[1:]
networks_path = None
if "--networks" in args:
    i = args.index("--networks")
    networks_path = Path(args[i + 1])
    del args[i:i + 2]
run_dir, dev = Path(args[0]), torch.device(args[1])
settings = WumpusAnalysisSettings()
n = int(args[2]) if len(args) > 2 else settings.episodes
full = n == settings.episodes
trained = load_trained_run(run_dir, dev)
world, run, cfg = trained.setup.domain, trained.run, trained.cfg
model = run.eval_agent.model
FWD, TURNS = int(WumpusAction.FORWARD), (int(WumpusAction.TURN_LEFT), int(WumpusAction.TURN_RIGHT))
NUM_A, NUM_O = world.num_actions, world.num_observations
PRUNE, GAMMA, CHUNK = cfg.mcts.observation_prune, model.discount, 128
BITS = {name: np.array([bool(o & bit) and o != GAME_OVER for o in range(NUM_O)])
        for name, bit in (("glitter", Percept.GLITTER), ("breeze", Percept.BREEZE), ("stench", Percept.STENCH))}
AHEAD = torch.tensor([[-1 if _ahead(c, h) is None else _ahead(c, h) for h in range(4)] for c in range(NUM_CELLS)],
                     device=dev)
planner = PlanningAgent(
    model, BeliefTreeSearch(model, cfg.mcts.num_simulations, cfg.mcts.c_puct, cfg.mcts.dirichlet_alpha,
                            dirichlet_epsilon=0.0, seed=stream_seed(settings.seed, 2),
                            observation_prune=cfg.mcts.observation_prune),
    n, temperature=0.0, seed=stream_seed(settings.seed, 3), device=dev)


# --- Test states: q_localise_diag.py's selection, copied (reports scripts do not import each other) ---

def target_distance(cell: int, heading: int, safe: list[bool], is_target) -> int:
    """Fewest actions (FORWARD through safe cells, turns) until the hunter stands on a target cell (override_diag.py)."""
    if is_target(cell):
        return 0
    queue, seen = deque([(cell, heading, 0)]), {(cell, heading)}
    while queue:
        c, h, d = queue.popleft()
        ahead = _ahead(c, h)
        for nxt in ((ahead, h) if ahead is not None and safe[ahead] else None, (c, (h + 1) % 4), (c, (h + 3) % 4)):
            if nxt is None or nxt in seen:
                continue
            if nxt[0] != c and is_target(nxt[0]):
                return d + 1
            seen.add(nxt)
            queue.append((*nxt, d + 1))
    return 10 ** 6


class Recorder:
    """The greedy planner plus a shadow exact-belief heuristic; keeps (z, exact belief, visited) at declined states."""

    def __init__(self):
        self.batch_size = n
        self.shadow = WumpusReferenceAgent(world, n, 0.4)
        self.rows: list[dict] = []
        self.t, self.start = 0, time.time()

    def reset(self, initial_observation):
        planner.reset(initial_observation)
        self.shadow.reset(initial_observation)

    def update(self, action, observation):
        planner.update(action, observation)
        self.shadow.update(action, observation)

    def act(self):
        s = self.shadow
        m = s._tracker.marginals(s._consistent, s._own)
        heuristic = s.act().tolist()
        state = planner.state.clone()
        action = planner.act()
        chosen = action.tolist()
        pit, wumpus, gold = m.pit.cpu().tolist(), m.wumpus.cpu().tolist(), m.gold.cpu().tolist()
        alive, done = m.wumpus_alive.tolist(), m.done.tolist()
        for b in range(n):
            if done[b] or heuristic[b] not in TURNS or chosen[b] == heuristic[b]:
                continue
            cell, heading = int(m.cell[b]), int(m.heading[b])
            if gold[b][cell] > 1.0 - _ZERO:
                continue
            danger = [pit[b][c] + (wumpus[b][c] if alive[b] else 0.0) for c in range(NUM_CELLS)]
            safe = [d < _ZERO for d in danger]
            reachable = {c for c, _ in _first_actions(cell, heading, safe)} | {cell}
            known = [c for c in reachable if gold[b][c] > 1.0 - _ZERO]               # rule (b) prefers known gold
            goal = (lambda c: c in known) if known else (lambda c: safe[c] and not s._visited[b][c] and c in reachable)
            if not (known or any(goal(c) for c in range(NUM_CELLS))):
                continue
            if target_distance(cell, heading, safe, goal) != 2:
                continue
            root = planner.planner.roots[b]
            q1_chosen = root.edges[chosen[b]].reward + GAMMA * sum(
                p * c.leaf_value for p, c in zip(root.edges[chosen[b]].observation_probs, root.edges[chosen[b]].children))
            own = s._own
            self.rows.append({"game": b, "t": self.t, "chosen": chosen[b], "turn": heuristic[b], "z": state[b],
                              "layouts": torch.nonzero(s._consistent[b]).squeeze(1),
                              "own": (int(own.cell[b]), int(own.heading[b]), bool(own.arrow[b]),
                                      bool(own.wumpus_alive[b])),
                              "visited": list(s._visited[b]), "q1_chosen": q1_chosen})
        self.t += 1
        if self.t % 10 == 0:
            print(f"   [{time.strftime('%H:%M:%S')}] decision {self.t}/{cfg.env.max_steps}, "
                  f"{len(self.rows)} declined distance-2 states, {time.time() - self.start:.0f}s", flush=True)
        return action


def exact_outcomes(layouts: torch.Tensor, own: tuple, action: int) -> tuple[float, dict]:
    """Exact one-step model at a belief (uniform over `layouts`, hunter state `own`): E[r], {o: (P, layouts', own', done')}."""
    k = layouts.shape[0]
    full_ = lambda v, dtype: torch.full((k,), v, dtype=dtype, device=dev)  # noqa: E731
    state = WumpusState(layouts, full_(own[0], torch.int64), full_(own[1], torch.int64), full_(own[2], torch.bool),
                        full_(own[3], torch.bool), full_(False, torch.bool))
    nxt, obs, reward = world.step_rules(state, full_(action, torch.int64))
    out = {}
    for o in torch.unique(obs).tolist():
        sel = obs == o
        i = int(torch.nonzero(sel)[0])
        out[o] = (float(sel.double().mean()), layouts[sel],
                  (int(nxt.cell[i]), int(nxt.heading[i]), bool(nxt.arrow[i]), bool(nxt.wumpus_alive[i])), bool(nxt.done[i]))
    if abs(sum(v[0] for v in out.values()) - 1.0) > 1e-9:
        raise RuntimeError("Exact observation probabilities do not sum to 1.")
    return float(reward.double().mean()), out


def factors_from(p_fwd: np.ndarray, pit: np.ndarray, wumpus: np.ndarray, gold: np.ndarray, alive: np.ndarray,
                 cell: np.ndarray, heading: np.ndarray, visited: np.ndarray, live: np.ndarray) -> np.ndarray:
    """Section 2's 8 factors (N, 8) from exact P(o' | FORWARD) (N, O), marginals (N, 16), own state and visited (N, 16).

    The glitter identity is exact only where entering cannot kill (the gold may share a cell with a pit or a live
    wumpus, and then FORWARD shows GAME_OVER, not glitter), so it is asserted at live states whose cell ahead is a
    wall or provably safe; the declined test states are such states by their selection.
    """
    ahead = AHEAD.cpu().numpy()[cell, heading]
    wall = ahead < 0
    idx = np.where(wall, 0, ahead)
    rows = np.arange(len(cell))
    pick = lambda m: np.where(wall, 0.0, m[rows, idx])  # noqa: E731
    out = np.stack([wall.astype(float), np.where(wall, 0.0, ~visited[rows, idx]).astype(float), pick(gold), pick(pit),
                    pick(wumpus * alive[:, None]), *(p_fwd[:, BITS[k]].sum(1) for k in ("glitter", "breeze", "stench"))], 1)
    glitter_identity = (1.0 - out[:, 0]) * out[:, 2] + out[:, 0] * gold[rows, cell]
    check = live & ((out[:, 0] > 0.5) | (out[:, 3] + out[:, 4] < 1e-12))
    if np.abs(glitter_identity - out[:, 5])[check].max(initial=0.0) > 1e-6:
        raise RuntimeError("P(glitter | FORWARD) != (1 - wall) P(gold ahead) + wall P(gold here).")
    return out


def row_factors(layouts: torch.Tensor, own: tuple, visited: list[bool]) -> tuple[np.ndarray, np.ndarray]:
    """Exact P(o' | b, FORWARD) (O,) and the 8 factors (8,) at one recorded belief."""
    p = np.zeros(NUM_O)
    for o, (pp, _, _, _) in exact_outcomes(layouts, own, FWD)[1].items():
        p[o] = pp
    pit = world.pit[layouts].double().mean(0).cpu().numpy()
    wum = F.one_hot(world.wumpus[layouts], NUM_CELLS).double().mean(0).cpu().numpy()
    gold = F.one_hot(world.gold[layouts], NUM_CELLS).double().mean(0).cpu().numpy()
    f = factors_from(p[None], pit[None], wum[None], gold[None], np.array([float(own[3])]), np.array([own[0]]),
                     np.array([own[1]]), np.array([visited]), np.array([True]))
    return p, f[0]


# --- 1. Test states: replay, root and turned latents with their exact factors ---
print(f"== {run_dir.name} best.pt (collection {trained.checkpoint_collection}), {n} games on {dev}", flush=True)
t0 = time.time()
rec = Recorder()
episodes, _ = play_episodes(WumpusSimulator(world, n, cfg.env.max_steps, stream_seed(settings.seed, 1), dev), rec)
returns = discounted_returns(episodes.rewards, world.discount).cpu().numpy()
print(f"  replay return {returns.mean():.4f}, {len(rec.rows)} declined states, {time.time() - t0:.0f} s", flush=True)
if full:
    reported = json.loads((run_dir / "analysis" / "report.json").read_text())["returns"]["learned_planner"]
    if abs(reported["discounted_return"] - returns.mean()) > 1e-6:
        raise RuntimeError(f"Replay is not faithful: {returns.mean()} vs report.json {reported['discounted_return']}.")
    saved_mr = np.load(run_dir / "analysis" / "macro_reach.npz")
    ref = saved_mr["q1_chosen"][saved_mr["distance"] == 2]
    mine = np.array([r["q1_chosen"] for r in rec.rows])
    if ref.shape != mine.shape or np.abs(ref - mine).max() > 1e-6:
        raise RuntimeError(f"State selection differs from macro_reach.npz: {ref.shape} vs {mine.shape}.")
del planner

rows = rec.rows
z_root = torch.stack([r["z"] for r in rows])


def history_latents() -> torch.Tensor:
    """Acting-filter latents z_0..z_T (n, T + 1, D) of the replayed analysis games, float32, no autocast."""
    with torch.no_grad():
        return run.acting_filter.unroll(F.one_hot(episodes.actions, NUM_A).float(),
                                        F.one_hot(episodes.observations, NUM_O).float(),
                                        F.one_hot(episodes.initial_observations, NUM_O).float()).float()


games_idx = torch.tensor([r["game"] for r in rows], device=dev)
steps_idx = torch.tensor([r["t"] for r in rows], device=dev)
if (history_latents()[games_idx, steps_idx] - z_root).abs().max() > 1e-4:
    raise RuntimeError("Unrolled history latents do not reproduce the planner's own at the declined states.")
if networks_path is not None:
    states = torch.load(networks_path, map_location=dev)
    if set(states) != set(run.networks):
        raise ValueError(f"saved networks {sorted(states)} do not match the run's {sorted(run.networks)}.")
    for name, net in run.networks.items():
        net.load_state_dict(states[name])
    z_root = history_latents()[games_idx, steps_idx]
    print(f"  networks {networks_path}: root latents recomputed from the histories", flush=True)
base_mode = networks_path is None
o_turn, z_turn, p_joint = [], [], []
for start in range(0, len(rows), CHUNK):
    batch = rows[start:start + CHUNK]
    with torch.no_grad():
        e = model.expand(z_root[start:start + CHUNK])
    for i, r in enumerate(batch):
        (o_t, _), = exact_outcomes(r["layouts"], r["own"], r["turn"])[1].items()
        o_turn.append(o_t)
        z_turn.append(e.next_states[i, r["turn"], o_t])
z_turn = torch.stack(z_turn)
for start in range(0, len(rows), CHUNK):
    with torch.no_grad():
        p_joint.append(model.expand(z_turn[start:start + CHUNK]).observation_probs[:, FWD].double().cpu().numpy())
p_joint = np.concatenate(p_joint)
root_p, root_f, turn_p, turn_f = [], [], [], []
for r in rows:
    p, f = row_factors(r["layouts"], r["own"], r["visited"])
    root_p.append(p); root_f.append(f)
    (_, (_, lay_t, own_t, _)), = exact_outcomes(r["layouts"], r["own"], r["turn"])[1].items()
    p, f = row_factors(lay_t, own_t, r["visited"])
    turn_p.append(p); turn_f.append(f)
root_p, root_f, turn_p, turn_f = map(np.array, (root_p, root_f, turn_p, turn_f))
tv_joint = 0.5 * np.abs(p_joint - turn_p).sum(1)
saved = np.load(run_dir / "analysis" / "q_localise.npz") if full else None
if full and base_mode:
    if np.abs(tv_joint - saved["tv_fwd"]).max() > 1e-4:
        raise RuntimeError("The joint head at the turned latents does not reproduce q_localise.npz's tv_fwd.")
print(f"  turned latents: joint-head TV median {np.median(tv_joint):.3f}; exact glitter at (c) "
      f"{turn_f[:, 5].mean():.3f}, wall {turn_f[:, 0].mean():.3f}, unvisited {turn_f[:, 1].mean():.3f}", flush=True)
torch.save({"z_root": z_root.cpu(), "z_turn": z_turn.cpu(), "o_turn": torch.tensor(o_turn),
            "p_fwd_root": torch.from_numpy(root_p), "p_fwd_turn": torch.from_numpy(turn_p),
            "factors_root": torch.from_numpy(root_f), "factors_turn": torch.from_numpy(turn_f), "factor_names": FACTORS,
            "meta": [{k: r[k] for k in ("game", "t", "chosen", "turn", "own", "visited")} for r in rows],
            "layouts": [r["layouts"].cpu() for r in rows]},
           run_dir / "analysis" / ("declined_states_smoke.pt" if not full else "declined_states.pt" if base_mode
                                   else f"declined_states_{networks_path.stem}.pt"))

def pruned(probs: np.ndarray) -> np.ndarray:
    """The search's branch weights over all |O| (planning/mcts.py section 6), along the last axis."""
    if PRUNE == 0.0:
        return probs
    kept = np.where(probs >= PRUNE, probs, 0.0)
    return kept / kept.sum(-1, keepdims=True)


def expand_np(latents_: torch.Tensor):
    """model.expand, chunked; numpy rewards (N, A), probs (N, A, O), values (N, A, O) and the child latents."""
    rs, ps, vs, zs = [], [], [], []
    for i in range(0, latents_.shape[0], CHUNK):
        with torch.no_grad():
            e = model.expand(latents_[i:i + CHUNK])
        rs.append(e.rewards.double().cpu().numpy())
        ps.append(e.observation_probs.double().cpu().numpy())
        vs.append(e.next_values.double().cpu().numpy())
        zs.append(e.next_states)
    return np.concatenate(rs), np.concatenate(ps), np.concatenate(vs), torch.cat(zs)


def arm_m() -> tuple[np.ndarray, np.ndarray]:
    """q_localise's arm M (depth-2 expectimax, every part learned) at the root latents: Q_mac, Q2(chosen)."""
    q_mac, q_cho = [], []
    for start_ in range(0, len(rows), CHUNK):
        batch, turns = rows[start_:start_ + CHUNK], o_turn[start_:start_ + CHUNK]
        r1, p1, _, z1 = expand_np(z_root[start_:start_ + CHUNK])
        jobs, child_z = [], []
        for i, row in enumerate(batch):
            for a1 in (row["turn"], row["chosen"]):
                kept = set(np.flatnonzero(p1[i, a1] >= PRUNE).tolist()) | ({turns[i]} if a1 == row["turn"] else set())
                for o1 in sorted(kept):
                    jobs.append((i, a1, o1))
                    child_z.append(z1[i, a1, o1])
        r2, p2, v2, _ = expand_np(torch.stack(child_z))
        child = {job: j for j, job in enumerate(jobs)}
        for i, row in enumerate(batch):
            for a1, macro in ((row["turn"], True), (row["chosen"], False)):
                w = pruned(p1[i, a1])
                total = 0.0
                for o1 in np.flatnonzero(w > 0):
                    j = child[(i, a1, int(o1))]
                    inner = r2[j] + GAMMA * (pruned(p2[j]) * v2[j]).sum(-1)
                    total += w[o1] * (inner[FWD] if macro else inner.max())
                (q_mac if macro else q_cho).append(r1[i, a1] + GAMMA * total)
    return np.array(q_mac), np.array(q_cho)


q_mac, q_cho = arm_m()
if full and base_mode and (np.abs(q_mac - saved["q_mac_M"]).max() > 1e-2 or np.abs(q_cho - saved["q_cho_M"]).max() > 1e-2):
    raise RuntimeError("Arm M does not reproduce q_localise.npz.")
decision_share = float((q_mac > q_cho).mean())
print(f"  decision share M (Q_mac > Q2(chosen)) {decision_share:.3f}", flush=True)

# --- 2. Training data: buffer at best.pt's collection, acting latents, exact factors per live state ---
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


def latents(filt: nn.Module, chunk: int = 512) -> torch.Tensor:
    """Filter latents z_0..z_{T-1} (B, T, D), float32, no autocast."""
    out = []
    with torch.no_grad():
        for i in range(0, games, chunk):
            a = F.one_hot(actions[i:i + chunk], NUM_A).float()
            o = F.one_hot(observations[i:i + chunk], NUM_O).float()
            z0 = F.one_hot(initial[i:i + chunk], NUM_O).float()
            out.append(filt.unroll(a, o, z0)[:, :-1].float())
    return torch.cat(out)


def buffer_factors(chunk: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact factors (B, T, 8) float32 and the live mask (B, T) for every buffer state, by layout enumeration."""
    tracker, out, live = WumpusBeliefTracker(world), [], []
    layouts = torch.arange(world.num_layouts, device=dev)
    for i in range(0, games, chunk):
        a_c, o_c = actions[i:i + chunk], observations[i:i + chunk]
        consistent, own = tracker.start(initial[i:i + chunk])
        g, num_l = consistent.shape
        visited = F.one_hot(own.cell, NUM_CELLS).bool()
        per_t, live_t = [], []
        for t in range(steps):
            expand = lambda v: v.unsqueeze(1).expand(g, num_l)  # noqa: E731
            hyp = WumpusState(layouts.expand(g, -1), expand(own.cell), expand(own.heading), expand(own.arrow),
                              expand(own.wumpus_alive), expand(own.done))
            weights = consistent.double()
            _, obs, _ = world.step_rules(hyp, torch.full((g, 1), FWD, dtype=torch.int64, device=dev))
            p = torch.zeros(g, NUM_O, dtype=torch.float64, device=dev).scatter_add_(1, obs, weights)
            p = p / weights.sum(1, keepdim=True)
            m = tracker.marginals(consistent, own)
            per_t.append(torch.from_numpy(factors_from(
                p.cpu().numpy(), m.pit.cpu().numpy(), m.wumpus.cpu().numpy(), m.gold.cpu().numpy(),
                m.wumpus_alive.cpu().numpy().astype(float), own.cell.cpu().numpy(), own.heading.cpu().numpy(),
                visited.cpu().numpy(), ~own.done.cpu().numpy())).float())
            live_t.append(~own.done.cpu())
            consistent, own = tracker.update(consistent, own, a_c[:, t], o_c[:, t])
            visited = visited | F.one_hot(own.cell, NUM_CELLS).bool()
        out.append(torch.stack(per_t, 1))
        live.append(torch.stack(live_t, 1))
    return torch.cat(out).to(dev), torch.cat(live).to(dev)


t0 = time.time()
z_all = latents(run.acting_filter)
f_all, live = buffer_factors()
torch.manual_seed(0)
random_filter = copy.deepcopy(run.acting_filter)
for module in random_filter.modules():
    if module is not random_filter and hasattr(module, "reset_parameters"):
        module.reset_parameters()
z_random = latents(random_filter)
split = torch.Generator().manual_seed(0)
held = torch.zeros(games, dtype=torch.bool)
held[torch.randperm(games, generator=split)[: games // 10]] = True
held = held.to(dev)[:, None] & live
train = (~held.any(1, keepdim=True)) & live
print(f"  buffer: {games} games x {steps} steps at collection {c}, {int(train.sum())} train / {int(held.sum())} "
      f"held-out live states; factors in {time.time() - t0:.0f} s", flush=True)


def fit_probe(z: torch.Tensor, kind: str, seed: int = 0, batch: int = 4096, max_steps: int = 12000, every: int = 250,
              patience: int = 6) -> tuple[nn.Module, int, float]:
    torch.manual_seed(seed)
    d, h = z.shape[-1], cfg.model.hidden_dim
    probe = (nn.Linear(d, len(FACTORS)) if kind == "linear" else
             nn.Sequential(nn.Linear(d, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, len(FACTORS)))).to(dev)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    g = torch.Generator(device=dev).manual_seed(seed)
    zt, ft, zh, fh = z[train], f_all[train], z[held], f_all[held]
    best, best_state, best_step, bad = float("inf"), None, 0, 0
    for step in range(1, max_steps + 1):
        idx = torch.randint(0, len(zt), (batch,), device=dev, generator=g)
        loss = F.binary_cross_entropy_with_logits(probe(zt[idx]), ft[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % every == 0:
            with torch.no_grad():
                held_loss = float(F.binary_cross_entropy_with_logits(probe(zh), fh))
            if held_loss < best - 1e-5:
                best, best_state, best_step, bad = held_loss, copy.deepcopy(probe.state_dict()), step, 0
            else:
                bad += 1
                if bad >= patience:
                    break
    probe.load_state_dict(best_state)
    return probe.eval().requires_grad_(False), best_step, best


train_mean = f_all[train].mean(0).cpu().numpy()


def metrics(pred: np.ndarray, target: np.ndarray, key: str) -> dict:
    out = {}
    for k, name in enumerate(FACTORS):
        p, y = pred[:, k], target[:, k]
        m = {"exact_mean": float(y.mean()), "pred_mean": float(p.mean()),
             "calibration": float(p.mean() / y.mean()) if y.mean() > 0 else float("nan")}
        if name in BINARY:
            majority = float(train_mean[k] >= 0.5)
            m |= {"acc": float(((p > 0.5) == (y > 0.5)).mean()), "acc_majority": float((y == majority).mean())}
        else:
            m |= {"mae": float(np.abs(p - y).mean()), "mae_constant": float(np.abs(train_mean[k] - y).mean())}
        m["decodable"] = decodable(m, name, key)
        out[name] = m
    return out


def predict(probe: nn.Module, z: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(probe(z.float())).double().cpu().numpy()


# --- 3. Probes and readout ---
summary, per_row = {"states": len(rows), "collection": c, "joint_tv_median_c": float(np.median(tv_joint)),
                    "decision_share_M": decision_share, "networks": None if base_mode else str(networks_path)}, {}
sets = {"a": (None, f_all[held].double().cpu().numpy()), "b": (z_root, root_f), "c": (z_turn, turn_f)}
for name, z in (("mlp", z_all), ("linear", z_all), ("random", z_random)):
    t0 = time.time()
    probe, step, held_loss = fit_probe(z, "linear" if name == "linear" else "mlp")
    summary[name] = {"fit_step": step, "held_loss": held_loss}
    for key, (zs, target) in sets.items():
        if name == "random" and key != "a":
            continue
        pred = predict(probe, z[held] if zs is None else zs)
        summary[name][key] = metrics(pred, target, key)
        if zs is not None:
            per_row[f"{name}_{key}"] = pred
        print(f"  {name:6s} ({key}): " + " ".join(
            f"{f}={'acc' if f in BINARY else 'mae'} {summary[name][key][f]['acc' if f in BINARY else 'mae']:.3f}"
            f"{'+' if summary[name][key][f]['decodable'] else '-'}" for f in FACTORS), flush=True)
    print(f"  {name}: early-stopped at step {step}, held-out loss {held_loss:.4f}, {time.time() - t0:.0f} s", flush=True)
mlp_c = per_row["mlp_c"]
composed = (1.0 - mlp_c[:, 0]) * mlp_c[:, 2]
summary["composed_glitter_ratio_c"] = float(composed.mean() / turn_f[:, 5].mean())
summary["direct_glitter_ratio_c"] = float(mlp_c[:, 5].mean() / turn_f[:, 5].mean())
summary["joint_glitter_ratio_c"] = float(p_joint[:, BITS["glitter"]].sum(1).mean() / turn_f[:, 5].mean())
print(f"  glitter at (c), learned / exact: joint head {summary['joint_glitter_ratio_c']:.2f}, mlp direct "
      f"{summary['direct_glitter_ratio_c']:.2f}, composed (1 - wall^) gold^ {summary['composed_glitter_ratio_c']:.2f}",
      flush=True)
suffix = ("" if base_mode else f"_{networks_path.stem}") + ("" if full else "_smoke")
np.savez(run_dir / "analysis" / f"feature_localise{suffix}.npz", factors_root=root_f, factors_turn=turn_f,
         p_joint_turn=p_joint, **per_row)
(run_dir / "analysis" / f"feature_localise{suffix}.json").write_text(json.dumps(summary, indent=2))
print("DONE", flush=True)
