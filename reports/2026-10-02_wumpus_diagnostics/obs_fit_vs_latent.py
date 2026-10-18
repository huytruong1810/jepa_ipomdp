# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/obs_fit_vs_latent.py
# ==============================================================================
# IS THE OBSERVATION HEAD'S ERROR AT THE TURNED LATENT A FIT/OBJECTIVE PROBLEM, OR MISSING FROM THE LATENT?
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-18, HANDOFF 6.2 Step A; pre-authorised):
#    - q_localise_diag.py read MODEL on 5 of 5 seeds (C9): at the planner's declined distance-2 near-turn
#      states, the learned observation head's P(o' | turned latent, FORWARD) is far from the exact one (TV
#      0.40-0.51) and predicts glitter 2.3-4.5x too rarely, so the backup misses the value of information of
#      entering a new cell.
#    - Two causes predict different methods (HANDOFF 6.2 Step C):
#        FIT/OBJECTIVE -> the latent holds the information, the head's training does not put accuracy where
#                         decisions need it: a value-equivalent observation loss (Grimm et al. 2020/2021) or
#                         decision-aware weighting of the cross-entropy (VaGraM, Voelcker et al. 2022).
#        LATENT        -> no head on this latent can read it: the representation objective must change
#                         (self-predictive / value-equivalent latent term, Ni et al. 2024).
#    - The test is decodability: fresh heads on the FROZEN acting latents, trained on the run's own buffer,
#      scored at the declined states (which come from the 512 analysis games, never in the buffer).
#
# 2. Method (usage: uv run python <this> <run dir>; best.pt of a runs/sweeps/wumpus_retrain seed):
#    - Test states: the 512 analysis games replayed exactly (return must equal report.json); the declined
#      distance-2 states are selected exactly as in q_localise_diag.py (its Recorder, copied; count and leaf-only
#      q1_chosen must equal macro_reach.npz). The turned latent is z1 = filter(z, turn, o_turn), o_turn the
#      exact (deterministic) turn outcome; the target is the exact P(o' | turned belief, FORWARD).
#    - Training data: the buffer slots of latest.pt that existed at best.pt's collection (head_swap_screen.py's
#      slot logic), acting latents z_0..z_{T-1}; a 10% game split (fixed seed) is held out for early stopping.
#      Exact soft targets P(o' | b_t, a) for all 5 actions at every buffer state come from the layout-
#      enumeration filter (WumpusBeliefTracker): step_rules over the consistent layouts, counts by observation.
#    - Arms (the agent's ObservationHead architecture, Adam 3e-4, minibatch 4096, held-out loss every 250 steps,
#      best state kept, stop after 6 evaluations without improvement, at most 12,000 steps):
#        (i)   joint:      best.pt's head (must reproduce q_localise.npz's tv_fwd).
#        (ii)  fresh-hard: cross-entropy on the sampled o_{t+1} at the buffer's own a_t (the agent's objective,
#                          not jointly trained, not on the growing schedule).
#        (iii) fresh-soft: soft cross-entropy on the exact P(o' | b_t, a_t), same inputs as (ii): label noise
#                          removed.
#        (iv)  all-soft:   soft cross-entropy on the exact P for all 5 actions at every buffer state (uniform
#                          action): the decodability bound, i.e. does the latent hold it at all.
#    - Measures at the test states (FORWARD from the turned latent): median TV to exact, mean P(glitter) ratio
#      learned / exact, mean P(GAME_OVER) (exact 0 by the selection), and the DECISION readout: q_localise's
#      depth-2 arm M with each head swapped into model.observation_head (everything else learned), as the
#      share of states with Q_mac > Q2(chosen), next to the saved M and EP shares. (i)'s M must reproduce
#      q_localise.npz's q_mac_M / q_cho_M.
#
# 3. Pre-registered readout (written 2026-10-18 before any run; 5 seeds). "Close": median TV <= 0.15 and
#    glitter ratio in [0.67, 1.5]. "Far": median TV >= 0.30 or glitter ratio < 0.5.
#    - FIT/OBJECTIVE: (iv) close, or its decision share >= the saved EP share - 10 points, while (i) is far,
#      on >= 4 of 5 seeds. Sub-reading: (iii) close and (ii) far -> label noise / sample efficiency of the hard
#      CE; (ii) close and (i) far -> the joint training / schedule.
#    - LATENT: (iv) far on >= 4 of 5 seeds.
#    - Otherwise MIXED: report the per-seed numbers and decide with the user.
#    - Caveat (R1a, HANDOFF 7): a fresh head that looks better offline can play worse (the planner selects its
#      errors). This screen only localises the error; it does not pick a head for play.
#
# 4. Result 2026-10-18 (runs/sweeps/wumpus_retrain seeds 0-4, best.pt; replays = report.json, states =
#    macro_reach.npz, arm (i) = q_localise.npz on every seed; logs runs/wumpus/obs_fit_seed{0..4}.log, chain
#    runs/wumpus/obs_fit_chain.sh; ~10-13 min per seed; results <run>/analysis/obs_fit_vs_latent.json).
#    **Readout: LATENT (arm (iv) far on 4 of 5 seeds; seed 0 between).** Median TV / glitter ratio / decision share:
#    | Seed | (i) joint          | (ii) fresh-hard    | (iii) fresh-soft   | (iv) all-soft      | saved M / EP |
#    | 0    | 0.40 / 0.44 / 10%  | 0.38 / 0.52 / 10%  | 0.37 / 0.55 / 15%  | 0.30 / 0.62 / 19%  | 10 / 43%     |
#    | 1    | 0.47 / 0.29 / 14%  | 0.32 / 0.48 / 17%  | 0.29 / 0.53 / 24%  | 0.27 / 0.49 / 26%  | 14 / 42%     |
#    | 2    | 0.51 / 0.30 / 11%  | 0.55 / 0.56 / 21%  | 0.42 / 0.61 / 25%  | 0.34 / 0.51 / 23%  | 11 / 46%     |
#    | 3    | 0.43 / 0.22 / 11%  | 0.44 / 0.45 / 24%  | 0.41 / 0.51 / 29%  | 0.40 / 0.53 / 34%  | 11 / 62%     |
#    | 4    | 0.50 / 0.34 / 8%   | 0.48 / 0.45 / 21%  | 0.46 / 0.56 / 32%  | 0.31 / 0.53 / 27%  | 8 / 58%      |
#    - Margins are thin on two seeds: seed 1's (iv) is far by the glitter ratio alone (0.49 < 0.5), seed 4's by
#      TV 0.313 (>= 0.30). No arm is close on any seed; the FIT branch (iv close or its share >= EP - 10) fails
#      on 5 of 5.
#    - Not all of the error is in the latent. Each step away from the joint head helps on every seed: a fresh
#      head on the same objective (ii) already lifts the glitter ratio from 0.22-0.44 to 0.45-0.56, exact
#      targets (iii) and all actions (iv) a little more, and the decision share rises from 8-14% to 19-34%.
#      Even so, (iv) recovers well under half of the EP - M gap (EP 42-62%). About half of the glitter
#      under-prediction is fit/schedule (joint vs fresh), and the remainder (ratio ~0.5, TV ~0.3) is not
#      decodable from the frozen latent by this head.
#    - Implication for Step C (HANDOFF 6.2): an observation-head-only objective (value-equivalent loss, VaGraM
#      weighting) cannot close this on the current latent; the latent itself must carry "what the cell ahead can
#      reveal" (a latent-side term: self-predictive / value-equivalent representation learning, candidate 3),
#      possibly with a decision-aware head on top. Caveat: decodability is measured with the agent's head
#      architecture and early stopping on the agent's own buffer, so "not decodable" means "not by this probe
#      from these data".
# ==============================================================================

import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ipomdp.agents import PlanningAgent, WumpusReferenceAgent
from ipomdp.agents.wumpus_reference import _ZERO, _ahead, _first_actions
from ipomdp.domain import WumpusSimulator
from ipomdp.domain.wumpus import GAME_OVER, NUM_CELLS, Percept, WumpusAction, WumpusBeliefTracker, WumpusState
from ipomdp.experiments.runs import load_trained_run
from ipomdp.experiments.wumpus import WumpusAnalysisSettings
from ipomdp.models import ObservationHead
from ipomdp.planning import BeliefTreeSearch
from ipomdp.training import discounted_returns, play_episodes, stream_seed
from ipomdp.training.checkpointing import load_checkpoint

run_dir = Path(sys.argv[1])
dev = torch.device("cuda")
settings = WumpusAnalysisSettings()
n = settings.episodes
trained = load_trained_run(run_dir, dev)
world, run, cfg = trained.setup.domain, trained.run, trained.cfg
model = run.eval_agent.model
FWD, TURNS = int(WumpusAction.FORWARD), (int(WumpusAction.TURN_LEFT), int(WumpusAction.TURN_RIGHT))
NUM_A, NUM_O = world.num_actions, world.num_observations
PRUNE, GAMMA, CHUNK = cfg.mcts.observation_prune, model.discount, 128
GLITTER = np.array([bool(o & Percept.GLITTER) and o != GAME_OVER for o in range(NUM_O)])
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


def pruned(probs: np.ndarray) -> np.ndarray:
    """The search's branch weights over all |O| (planning/mcts.py section 6), along the last axis."""
    if PRUNE == 0.0:
        return probs
    kept = np.where(probs >= PRUNE, probs, 0.0)
    return kept / kept.sum(-1, keepdims=True)


class Recorder:
    """The greedy planner plus a shadow exact-belief heuristic; keeps (z, exact belief) at declined distance-2 near-turn states."""

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
                              "q1_chosen": q1_chosen})
        self.t += 1
        if self.t % 10 == 0:
            print(f"   [{time.strftime('%H:%M:%S')}] decision {self.t}/{cfg.env.max_steps}, "
                  f"{len(self.rows)} declined distance-2 states, {time.time() - self.start:.0f}s", flush=True)
        return action


def exact_outcomes(layouts: torch.Tensor, own: tuple, action: int) -> tuple[float, dict]:
    """Exact one-step model at a belief (uniform over `layouts`, hunter state `own`): E[r], {o: (P, layouts', own', done')}."""
    k = layouts.shape[0]
    full = lambda v, dtype: torch.full((k,), v, dtype=dtype, device=dev)  # noqa: E731
    state = WumpusState(layouts, full(own[0], torch.int64), full(own[1], torch.int64), full(own[2], torch.bool),
                        full(own[3], torch.bool), full(False, torch.bool))
    nxt, obs, reward = world.step_rules(state, full(action, torch.int64))
    out = {}
    for o in torch.unique(obs).tolist():
        sel = obs == o
        i = int(torch.nonzero(sel)[0])
        out[o] = (float(sel.double().mean()), layouts[sel],
                  (int(nxt.cell[i]), int(nxt.heading[i]), bool(nxt.arrow[i]), bool(nxt.wumpus_alive[i])), bool(nxt.done[i]))
    if abs(sum(v[0] for v in out.values()) - 1.0) > 1e-9:
        raise RuntimeError("Exact observation probabilities do not sum to 1.")
    return float(reward.double().mean()), out


def expand_np(latents: torch.Tensor):
    """model.expand, chunked; numpy rewards (N, A), probs (N, A, O), values (N, A, O) and the child latents on the GPU."""
    rs, ps, vs, zs = [], [], [], []
    for i in range(0, latents.shape[0], CHUNK):
        e = model.expand(latents[i:i + CHUNK])
        rs.append(e.rewards.double().cpu().numpy())
        ps.append(e.observation_probs.double().cpu().numpy())
        vs.append(e.next_values.double().cpu().numpy())
        zs.append(e.next_states)
    return np.concatenate(rs), np.concatenate(ps), np.concatenate(vs), torch.cat(zs)


def turned_targets(rows: list[dict]) -> tuple[list[int], np.ndarray]:
    """The exact turn outcome o_turn per row and the exact P(o' | turned belief, FORWARD) (N, O)."""
    o_turn, pe = [], np.zeros((len(rows), NUM_O))
    for i, row in enumerate(rows):
        (o_t, (_, lay_t, own_t, _)), = exact_outcomes(row["layouts"], row["own"], row["turn"])[1].items()
        o_turn.append(o_t)
        for o2, (pp, _, _, _) in exact_outcomes(lay_t, own_t, FWD)[1].items():
            pe[i, o2] = pp
    if (pe[:, GAME_OVER] > 0.0).any():
        raise RuntimeError("Exact P(GAME_OVER | turn, FORWARD) > 0 at a provably safe distance-2 target.")
    return o_turn, pe


def arm_m(rows: list[dict], o_turn: list[int]) -> dict[str, np.ndarray]:
    """q_localise's arm M with the current model.observation_head; plus P(o' | turned latent, FORWARD)."""
    out = {k: [] for k in ("q_mac", "q_cho", "p_fwd")}
    for start in range(0, len(rows), CHUNK):
        batch, turns = rows[start:start + CHUNK], o_turn[start:start + CHUNK]
        r1, p1, _, z1 = expand_np(torch.stack([r["z"] for r in batch]))
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
                out["q_mac" if macro else "q_cho"].append(r1[i, a1] + GAMMA * total)
            out["p_fwd"].append(p2[child[(i, row["turn"], turns[i])], FWD])
    return {k: np.array(v) for k, v in out.items()}


# --- 1. Test states ---
print(f"== {run_dir.name} best.pt (collection {trained.checkpoint_collection}), {n} games", flush=True)
rec = Recorder()
episodes, _ = play_episodes(WumpusSimulator(world, n, cfg.env.max_steps, stream_seed(settings.seed, 1), dev), rec)
returns = discounted_returns(episodes.rewards, world.discount).cpu().numpy()
reported = json.loads((run_dir / "analysis" / "report.json").read_text())["returns"]["learned_planner"]
if abs(reported["discounted_return"] - returns.mean()) > 1e-6:
    raise RuntimeError(f"Replay is not faithful: {returns.mean()} vs report.json {reported['discounted_return']}.")
saved_mr = np.load(run_dir / "analysis" / "macro_reach.npz")
ref = saved_mr["q1_chosen"][saved_mr["distance"] == 2]
mine = np.array([r["q1_chosen"] for r in rec.rows])
if ref.shape != mine.shape or np.abs(ref - mine).max() > 1e-6:
    raise RuntimeError(f"State selection differs from macro_reach.npz: {ref.shape} vs {mine.shape}.")
del planner
torch.cuda.empty_cache()
saved = np.load(run_dir / "analysis" / "q_localise.npz")
o_turn, p_exact = turned_targets(rec.rows)
print(f"  replay = report.json, {len(rec.rows)} states = macro_reach.npz", flush=True)

# --- 2. Training data: buffer at best.pt's collection, acting latents, exact soft targets for every action ---
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


def latents(actions, observations, initial, chunk: int = 512) -> torch.Tensor:
    """Acting-filter latents z_0..z_{T-1} (B, T, D), float32, no autocast."""
    out = []
    with torch.no_grad():
        for i in range(0, actions.shape[0], chunk):
            a = F.one_hot(actions[i:i + chunk], NUM_A).float()
            o = F.one_hot(observations[i:i + chunk], NUM_O).float()
            z0 = F.one_hot(initial[i:i + chunk], NUM_O).float()
            out.append(run.acting_filter.unroll(a, o, z0)[:, :-1].float())
    return torch.cat(out)


def exact_soft_targets(chunk: int = 128) -> torch.Tensor:
    """Exact P(o' | b_t, a) for every buffer state and action, (B, T, A, O) float32, by layout enumeration."""
    tracker, out = WumpusBeliefTracker(world), []
    layouts = torch.arange(world.num_layouts, device=dev)
    for i in range(0, games, chunk):
        a_c, o_c = actions[i:i + chunk], observations[i:i + chunk]
        consistent, own = tracker.start(initial[i:i + chunk])
        g, num_l = consistent.shape
        per_t = []
        for t in range(steps):
            expand = lambda v: v.unsqueeze(1).expand(g, num_l)  # noqa: E731
            hyp = WumpusState(layouts.expand(g, -1), expand(own.cell), expand(own.heading), expand(own.arrow),
                              expand(own.wumpus_alive), expand(own.done))
            weights = consistent.float()
            probs = torch.zeros(g, NUM_A, NUM_O, device=dev)
            for a in range(NUM_A):
                _, obs, _ = world.step_rules(hyp, torch.full((g, 1), a, dtype=torch.int64, device=dev))
                probs[:, a].scatter_add_(1, obs, weights)
            per_t.append(probs / weights.sum(1)[:, None, None])
            consistent, own = tracker.update(consistent, own, a_c[:, t], o_c[:, t])
        out.append(torch.stack(per_t, 1))
    return torch.cat(out)


t0 = time.time()
z_all = latents(actions, observations, initial)                                  # (B, T, D)
soft_all = exact_soft_targets()                                                  # (B, T, A, O)
own_soft = soft_all.gather(2, actions[:, :, None, None].expand(-1, -1, 1, NUM_O)).squeeze(2)
if float((own_soft.gather(2, observations[:, :, None]) <= 0).sum()) > 0:
    raise RuntimeError("A recorded observation has exact probability 0: buffer and filter disagree.")
split = torch.Generator().manual_seed(0)
held = torch.zeros(games, dtype=torch.bool)
held[torch.randperm(games, generator=split)[: games // 10]] = True
held = held.to(dev)
print(f"  buffer: {games} games x {steps} steps at collection {c}; exact targets in {time.time() - t0:.0f} s", flush=True)


def flat(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return x[mask].reshape(-1, *x.shape[2:])


DATA = {split_name: {"z": flat(z_all, m), "a": flat(actions, m), "o": flat(observations, m),
                     "own": flat(own_soft, m), "all": flat(soft_all, m)}
        for split_name, m in (("train", ~held), ("held", held))}


def batch_loss(head: nn.Module, d: dict, idx: torch.Tensor, kind: str, g: torch.Generator | None) -> torch.Tensor:
    z = d["z"][idx]
    if kind == "all":                                                            # uniform action, exact soft target
        a = (torch.randint(0, NUM_A, (len(idx),), device=dev, generator=g) if g is not None
             else torch.arange(len(idx), device=dev) % NUM_A)
        target = d["all"][idx, a]
    else:
        a = d["a"][idx]
        target = d["own"][idx] if kind == "soft" else None
    logits = head(z, F.one_hot(a, NUM_A).float())
    if target is None:
        return F.cross_entropy(logits, d["o"][idx])
    return -(target * F.log_softmax(logits, -1)).sum(-1).mean()


def fit_head(kind: str, seed: int = 0, batch: int = 4096, max_steps: int = 12000, every: int = 250,
             patience: int = 6) -> tuple[nn.Module, int, float]:
    torch.manual_seed(seed)
    head = ObservationHead(cfg.model.latent_dim, NUM_A, NUM_O, cfg.model.hidden_dim, cfg.model.num_blocks).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=3e-4)
    g = torch.Generator(device=dev).manual_seed(seed)
    tr, he = DATA["train"], DATA["held"]
    best, best_state, best_step, bad = float("inf"), None, 0, 0
    for step in range(1, max_steps + 1):
        loss = batch_loss(head, tr, torch.randint(0, len(tr["z"]), (batch,), device=dev, generator=g), kind, g)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % every == 0:
            with torch.no_grad():
                held_loss = float(sum(batch_loss(head, he, torch.arange(i, min(i + 16384, len(he["z"])), device=dev),
                                                 kind, None) * min(16384, len(he["z"]) - i)
                                      for i in range(0, len(he["z"]), 16384)) / len(he["z"]))
            if held_loss < best - 1e-5:
                best, best_state, best_step, bad = held_loss, {k: v.clone() for k, v in head.state_dict().items()}, step, 0
            else:
                bad += 1
                if bad >= patience:
                    break
    head.load_state_dict(best_state)
    return head.eval().requires_grad_(False), best_step, best


# --- 3. Arms and readout ---
joint = model.observation_head
heads = {"(i) joint": (joint, None, None)}
for name, kind in (("(ii) fresh-hard", "hard"), ("(iii) fresh-soft", "soft"), ("(iv) all-soft", "all")):
    t0 = time.time()
    head, step, held_loss = fit_head(kind)
    heads[name] = (head, step, held_loss)
    print(f"  {name}: early-stopped at step {step}, held-out loss {held_loss:.4f}, {time.time() - t0:.0f} s", flush=True)

ep_share = float((saved["q_mac_EP"] > saved["q_cho_EP"]).mean())
m_share = float((saved["q_mac_M"] > saved["q_cho_M"]).mean())
glitter_exact = p_exact[:, GLITTER].sum(1).mean()
print(f"  saved shares Q_mac > Q2(chosen): M {m_share:.3f}, EP {ep_share:.3f}, EPR "
      f"{float((saved['q_mac_EPR'] > saved['q_cho_EPR']).mean()):.3f}; exact P(glitter) {glitter_exact:.3f}", flush=True)
result = {}
for name, (head, _, _) in heads.items():
    model.observation_head = head
    try:
        res = arm_m(rec.rows, o_turn)
    finally:
        model.observation_head = joint
    tv = 0.5 * np.abs(res["p_fwd"] - p_exact).sum(1)
    ratio = res["p_fwd"][:, GLITTER].sum(1).mean() / glitter_exact
    share = float((res["q_mac"] > res["q_cho"]).mean())
    if name == "(i) joint":
        if np.abs(tv - saved["tv_fwd"]).max() > 1e-4 or np.abs(res["q_mac"] - saved["q_mac_M"]).max() > 1e-2 \
                or np.abs(res["q_cho"] - saved["q_cho_M"]).max() > 1e-2:
            raise RuntimeError("The joint arm does not reproduce q_localise.npz.")
    close = np.median(tv) <= 0.15 and 0.67 <= ratio <= 1.5
    far = np.median(tv) >= 0.30 or ratio < 0.5
    result[name] = {"tv_median": float(np.median(tv)), "glitter_ratio": float(ratio),
                    "p_death_mean": float(res["p_fwd"][:, GAME_OVER].mean()), "decision_share": share,
                    "close": bool(close), "far": bool(far)}
    print(f"  {name:16s}: TV median {np.median(tv):.3f}, P(glitter) ratio {ratio:.2f}, P(death) mean "
          f"{res['p_fwd'][:, GAME_OVER].mean():.4f}, Q_mac > Q2(chosen) {share:.3f} "
          f"[{'close' if close else 'far' if far else 'between'}]", flush=True)
    np.save(run_dir / "analysis" / f"obs_fit_{name.split()[0].strip('()')}_pfwd.npy", res["p_fwd"])
summary = {"states": len(rec.rows), "saved_M": m_share, "saved_EP": ep_share, "arms": result,
           "fits": {k: {"step": v[1], "held_loss": v[2]} for k, v in heads.items() if v[1] is not None}}
(run_dir / "analysis" / "obs_fit_vs_latent.json").write_text(json.dumps(summary, indent=2))
print("DONE", flush=True)
