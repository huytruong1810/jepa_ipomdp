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
# 4. Result: (to be filled after the runs.)
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

run_dir, dev = Path(sys.argv[1]), torch.device(sys.argv[2])
settings = WumpusAnalysisSettings()
n = int(sys.argv[3]) if len(sys.argv) > 3 else settings.episodes
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
if full:
    saved = np.load(run_dir / "analysis" / "q_localise.npz")
    if np.abs(tv_joint - saved["tv_fwd"]).max() > 1e-4:
        raise RuntimeError("The joint head at the turned latents does not reproduce q_localise.npz's tv_fwd.")
print(f"  turned latents: joint-head TV median {np.median(tv_joint):.3f}; exact glitter at (c) "
      f"{turn_f[:, 5].mean():.3f}, wall {turn_f[:, 0].mean():.3f}, unvisited {turn_f[:, 1].mean():.3f}", flush=True)
torch.save({"z_root": z_root.cpu(), "z_turn": z_turn.cpu(), "o_turn": torch.tensor(o_turn),
            "p_fwd_root": torch.from_numpy(root_p), "p_fwd_turn": torch.from_numpy(turn_p),
            "factors_root": torch.from_numpy(root_f), "factors_turn": torch.from_numpy(turn_f), "factor_names": FACTORS,
            "meta": [{k: r[k] for k in ("game", "t", "chosen", "turn", "own", "visited")} for r in rows],
            "layouts": [r["layouts"].cpu() for r in rows]},
           run_dir / "analysis" / ("declined_states.pt" if full else "declined_states_smoke.pt"))

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
summary, per_row = {"states": len(rows), "collection": c, "joint_tv_median_c": float(np.median(tv_joint))}, {}
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
suffix = "" if full else "_smoke"
np.savez(run_dir / "analysis" / f"feature_localise{suffix}.npz", factors_root=root_f, factors_turn=turn_f,
         p_joint_turn=p_joint, **per_row)
(run_dir / "analysis" / f"feature_localise{suffix}.json").write_text(json.dumps(summary, indent=2))
print("DONE", flush=True)
