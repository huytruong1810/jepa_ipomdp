# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/exact_model_play.py
# ==============================================================================
# DOES THE EXACT ONE-STEP MODEL, WITH THE LEARNED VALUE AND FILTER, STOP THE IDLING IN PLAY? (PRIVILEGED BOUND)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-18, HANDOFF 6.2 Step B; the user approved the plan that includes it):
#    - q_localise_diag.py (C9) showed OFFLINE that swapping the exact observation probabilities and rewards
#      into a depth-2 expectimax flips the declined turn-then-step decision at 48-81% of states (learned:
#      8-14%). Offline rankings have failed in play before (C3: teacher values, Q/A heads). Before building a
#      decision-aware model objective (Step C), this bounds what a perfect one-step model would be worth IN PLAY
#      with the agent's own value and filter, inside the unchanged 50-simulation belief-tree search.
#    - PASS -> the model is worth fixing (Step C). FAIL -> the offline ranking does not survive play; stop and
#      report before any method.
#
# 2. Method (usage: uv run python <this> <run dir> <arm> [<games>]; arms learned | EP | EPR | EPRt | smoke[:<arm>]):
#    - Networks: best.pt of a runs/sweeps/wumpus_retrain seed (load_trained_run). The 512 analysis games
#      (stream_seed(settings.seed, 1)), planner streams 2 and 3, Dirichlet off, temperature 0, as report.json.
#    - HybridSearchModel (the SearchModel protocol of planning/search_model.py) wraps the run's
#      LearnedSearchModel. A search state is [z (D), node id] in float32 (ids < 2^24 are exact). The id indexes a
#      tensor registry of exact beliefs: the hunter's own state (cell, heading, arrow, wumpus alive, done; the
#      same for every consistent layout, domain/wumpus.py section 5) for every node, and the consistent-layout
#      mask (L = 16,380 bools) only for nodes that were expanded (a "bank" row). A child is stored lazily as
#      (parent id, a, o) and its mask is materialised when it is expanded: mask_parent & (obs(parent, a) == o).
#      Roots are ids 0..B-1, tracked by WumpusBeliefTracker on the real observations; every update() clears
#      the registry back to the roots (BeliefTreeSearch builds a fresh tree at every act, mcts.py search()).
#    - expand() for N nodes (the learned part on the whole leaf batch, as in the stock planner, so the same
#      kernels give the same numbers; the exact part in chunks of 32): the learned expand gives the child latents z' = filter(z, a, o) for
#      every o (the planner's own children), learned rewards, probabilities and child values. For every action,
#      step_rules over the (N, L) layout hypotheses gives the exact quantities:
#        exact R(b, a) = mean reward over the consistent layouts;  exact P(o | b, a) = share of consistent
#        layouts emitting o;  the exact child's own state (done = an exact game over).
#    - Arms: learned = the stock planner (BeliefTreeSearch over the LearnedSearchModel; must reproduce
#      report.json); EP = exact P, learned rewards; EPR = exact P and R. The value is always the learned one at
#      the filter's latents, terminal children included (see 3a). EPRt = EPR with V = 0 at exact game-over
#      children (q_localise_diag.py's convention), kept only as the record of 3a. Pruning (observation_prune) applies to the exact
#      probabilities as to the learned ones; exact zeros are therefore never branched on.
#    - smoke: 32 games; asserts that the hybrid with no exact parts plays the stock planner's actions bit for bit,
#      checks the exact P and R of the hybrid against a per-node reference (q_localise_diag.py's exact_outcomes)
#      at the first expansion of every decision, then plays EPR and prints its action mix and time per decision.
#    - Per arm: return, win / death / unfinished, live action mix, cells visited per game; per-game discounted
#      returns saved to <run>/analysis/exact_play_<arm>.npz; the paired gain over learned when that file exists.
#
# 3. Pre-registered readout (from HANDOFF 6.2, written 2026-10-18 before any run; 5 seeds):
#    - PASS: EPR >= +100 paired (per game, 512 games) on >= 3 of 5 seeds, with the unfinished rate down and the
#      death rate up by at most 2 points on those seeds.
#    - EP vs EPR splits the observation head from the reward head (descriptive).
#    - FAIL: the offline ranking does not survive play (C3); stop and report before any method.
#    - Privileged diagnostic: the exact belief is a bound on what a perfect one-step model could do, never an agent.
# 3a. Protocol change after the smoke (2026-10-18, before any 512-game run; seed 0, 32 games):
#    - As first written, the exact arms set V = 0 at exact game-over children. That arm (now EPRt) won 0 of 32
#      games (learned: 10 of 32), with 0 deaths: at the gold cell the hunter bumped into a wall forever instead of
#      grabbing (log runs/wumpus/exact_play_smoke_seed0.log; trajectories from a scratch debug script). Mechanism:
#      Q(GRAB) = exact 999 + gamma * 0, while Q(bump) = -1 + gamma * V^(glitter latent) and V^ there exceeds
#      ~1,010; the learned planner grabs because its value at the post-GRAB latent is not 0. The learned value is
#      co-adapted with its own terminal latents, so zeroing them is not "the exact model with the learned value"
#      but a value/model mismatch (the head_swap_screen.py caveat).
#    - Changed: the learned value at every child, terminal children included (EPR, EP). Smoke of this arm
#      (logged as smoke:EPRv, runs/wumpus/exact_play_smoke_EPRv_seed0.log): 424.4 vs 232.1, paired +192 +- 123,
#      wins 31 -> 50%, deaths 0. The pass criterion of 3 is unchanged and applies to EPR as now defined.
#    - q_localise_diag.py's EP/EPR used V = 0 at exact terminal children; at its states (safe targets) that only
#      touches GRAB children, so its reading is unaffected in kind but not re-checked.
#
# 4. Result: (to be filled after the runs)
# ==============================================================================

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ipomdp.agents import PlanningAgent
from ipomdp.domain import WumpusSimulator, game_outcomes
from ipomdp.domain.wumpus import GAME_OVER, WumpusAction, WumpusBeliefTracker, WumpusState
from ipomdp.experiments.runs import load_trained_run
from ipomdp.experiments.wumpus import WumpusAnalysisSettings
from ipomdp.planning import BeliefTreeSearch
from ipomdp.planning.search_model import Expansion
from ipomdp.training import discounted_returns, play_episodes, stream_seed

run_dir, arm = Path(sys.argv[1]), sys.argv[2]
ARMS = {"EP": (True, False, False), "EPR": (True, True, False), "EPRt": (True, True, True)}  # exact P, R, terminal
if arm not in ("learned", "smoke", *ARMS) and not (arm.startswith("smoke:") and arm[6:] in ARMS):
    raise ValueError(f"unknown arm {arm}")
dev = torch.device("cuda")
settings = WumpusAnalysisSettings()
n = int(sys.argv[3]) if len(sys.argv) > 3 else (32 if arm.startswith("smoke") else settings.episodes)
trained = load_trained_run(run_dir, dev)
world, run, cfg = trained.setup.domain, trained.run, trained.cfg
model = run.eval_agent.model
if model.horizon is not None:
    raise ValueError("Time-aware runs carry steps-left in the state; this script reads plain latents only.")
if cfg.mcts.observation_prune <= 0.0:
    raise ValueError("Exact zero-probability observations must be pruned (observation_prune > 0).")
NUM_A, NUM_O, NUM_L, D = world.num_actions, world.num_observations, world.num_layouts, cfg.model.latent_dim
NAMES = [a.name for a in WumpusAction]
CHUNK = 32


class HybridSearchModel:
    """The learned search model with the exact one-step model (P, optionally R, terminal flags) swapped in (section 2)."""

    def __init__(self, learned, exact_p: bool, exact_r: bool, exact_terminal: bool, batch_size: int,
                 check: bool = False):
        self.learned, self.exact_p, self.exact_r, self.check = learned, exact_p, exact_r, check
        self.exact_terminal = exact_terminal
        self.num_actions, self.num_observations, self.discount = NUM_A, NUM_O, learned.discount
        self.tracker, self.batch = WumpusBeliefTracker(world), batch_size
        self.layouts = torch.arange(NUM_L, device=dev)
        bank_cap = batch_size * (cfg.mcts.num_simulations + 2)
        cap = bank_cap * NUM_A * NUM_O + batch_size
        if cap >= 2 ** 24:
            raise ValueError("node ids would not be exact in float32.")
        self.bank = torch.zeros(bank_cap, NUM_L, dtype=torch.bool, device=dev)      # masks of expanded nodes
        self.bank_row = torch.full((cap,), -1, dtype=torch.int64, device=dev)
        self.parent = torch.full((cap,), -1, dtype=torch.int64, device=dev)
        self.action = torch.zeros(cap, dtype=torch.int64, device=dev)
        self.obs = torch.zeros(cap, dtype=torch.int64, device=dev)
        self.own = {k: torch.zeros(cap, dtype=torch.int64, device=dev) for k in ("cell", "heading", "arrow", "alive", "done")}
        self.root_consistent = self.root_own = None
        self.next_id = self.next_row = 0
        self.checked_this_decision = False

    # --- the registry ---
    def _reset_registry(self) -> Tensor:
        b = self.batch
        self.bank[:b] = self.root_consistent
        self.bank_row[:b] = torch.arange(b, device=dev)
        self.bank_row[b:max(self.next_id, b)] = -1
        own = self.root_own
        for k, v in (("cell", own.cell), ("heading", own.heading), ("arrow", own.arrow), ("alive", own.wumpus_alive),
                     ("done", own.done)):
            self.own[k][:b] = v.long()
        self.next_id, self.next_row = b, b
        self.checked_this_decision = False
        return torch.arange(b, device=dev, dtype=torch.float32)

    def _hypotheses(self, ids: Tensor) -> WumpusState:
        g = ids.shape[0]
        e = lambda k, dtype=torch.int64: self.own[k][ids].unsqueeze(1).expand(g, NUM_L).to(dtype)  # noqa: E731
        return WumpusState(self.layouts.expand(g, -1), e("cell"), e("heading"), e("arrow", torch.bool),
                           e("alive", torch.bool), e("done", torch.bool))

    def _materialise(self, ids: Tensor) -> Tensor:
        """Bank rows of `ids` (expanding nodes), creating the masks of lazy children from their parents."""
        lazy = ids[self.bank_row[ids] < 0]
        if lazy.numel():
            parents = self.parent[lazy]
            if bool((self.bank_row[parents] < 0).any()):
                raise RuntimeError("a lazy node's parent has no materialised mask.")
            _, obs, _ = world.step_rules(self._hypotheses(parents), self.action[lazy].unsqueeze(1))
            masks = self.bank[self.bank_row[parents]] & (obs == self.obs[lazy].unsqueeze(1))
            if not bool(masks.any(1).all()):
                raise RuntimeError("an expanded node has no consistent layout (an exact-zero branch was expanded).")
            rows = torch.arange(self.next_row, self.next_row + lazy.numel(), device=dev)
            if int(rows[-1]) >= self.bank.shape[0]:
                raise RuntimeError("mask bank full.")
            self.bank[rows] = masks
            self.bank_row[lazy] = rows
            self.next_row += lazy.numel()
        return self.bank_row[ids]

    # --- the SearchModel protocol ---
    def initial_states(self, batch_size: int, initial_observations: Tensor | None) -> Tensor:
        z = self.learned.initial_states(batch_size, initial_observations)
        self.root_consistent, self.root_own = self.tracker.start(initial_observations)
        return torch.cat([z, self._reset_registry().unsqueeze(1)], 1)

    def update(self, states: Tensor, actions: Tensor, observations: Tensor) -> Tensor:
        z = self.learned.update(states[:, :D], actions, observations)
        self.root_consistent, self.root_own = self.tracker.update(self.root_consistent, self.root_own, actions,
                                                                  observations)
        return torch.cat([z, self._reset_registry().unsqueeze(1)], 1)

    @torch.no_grad()
    def expand(self, states: Tensor) -> Expansion:
        # The learned expand sees the whole leaf batch, as in the stock planner (same kernels, same numbers);
        # only the exact part is chunked.
        e = self.learned.expand(states[:, :D])
        num = states.shape[0]
        ids = states[:, D].long()
        child_ids = torch.arange(self.next_id, self.next_id + num * NUM_A * NUM_O, device=dev).view(num, NUM_A, NUM_O)
        self.next_id += num * NUM_A * NUM_O
        if self.next_id > self.parent.shape[0]:
            raise RuntimeError("node registry full.")
        next_states = torch.cat([e.next_states, child_ids.unsqueeze(-1).float()], -1)
        if not (self.exact_p or self.exact_r):
            return Expansion(e.rewards, e.observation_probs, next_states, e.next_values)
        parts = [self._exact(ids[i:i + CHUNK], child_ids[i:i + CHUNK]) for i in range(0, num, CHUNK)]
        rewards, probs, child_done = (torch.cat([p[k] for p in parts]) for k in range(3))
        values = torch.where(child_done & self.exact_terminal, 0.0, e.next_values.double())
        return Expansion(rewards if self.exact_r else e.rewards, probs if self.exact_p else e.observation_probs,
                         next_states, values)

    def _exact(self, ids: Tensor, child_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Exact R(b, a) (g, A), P(o | b, a) (g, A, O) and child game-over flags; registers the children's own states."""
        g = ids.shape[0]
        masks = self.bank[self._materialise(ids)]                                      # (g, L)
        hyp = self._hypotheses(ids)
        count = masks.sum(1).double()
        rewards = torch.zeros(g, NUM_A, dtype=torch.float64, device=dev)
        probs = torch.zeros(g, NUM_A, NUM_O, dtype=torch.float64, device=dev)
        child_done = torch.zeros(g, NUM_A, NUM_O, dtype=torch.bool, device=dev)
        for a in range(NUM_A):
            nxt, obs, reward = world.step_rules(hyp, torch.full((g, 1), a, dtype=torch.int64, device=dev))
            rewards[:, a] = (reward.double() * masks).sum(1) / count
            key = masks.unsqueeze(1) & (obs.unsqueeze(1) == torch.arange(NUM_O, device=dev)[None, :, None])  # (g,O,L)
            probs[:, a] = key.sum(-1).double() / count[:, None]
            first = key.float().argmax(-1)                                              # a consistent layout per o
            pick = lambda t: t.gather(1, first).long()  # noqa: E731
            cid = child_ids[:, a]
            self.parent[cid] = ids.unsqueeze(1).expand(g, NUM_O)
            self.action[cid] = a
            self.obs[cid] = torch.arange(NUM_O, device=dev).expand(g, NUM_O)
            for k, field in (("cell", nxt.cell), ("heading", nxt.heading), ("arrow", nxt.arrow),
                             ("alive", nxt.wumpus_alive), ("done", nxt.done)):
                self.own[k][cid] = pick(field)
            child_done[:, a] = pick(nxt.done).bool()
        if self.check and not self.checked_this_decision:
            self._check(ids[0], rewards[0], probs[0])
            self.checked_this_decision = True
        return rewards, probs, child_done

    def _check(self, node: Tensor, rewards: Tensor, probs: Tensor) -> None:
        """Per-node reference (q_localise_diag.py's exact_outcomes) for the first expansion of a decision."""
        layouts = torch.nonzero(self.bank[self.bank_row[node]]).squeeze(1)
        k = layouts.shape[0]
        full = lambda key, dtype: torch.full((k,), int(self.own[key][node]), dtype=torch.int64, device=dev).to(dtype)  # noqa: E731
        state = WumpusState(layouts, full("cell", torch.int64), full("heading", torch.int64), full("arrow", torch.bool),
                            full("alive", torch.bool), full("done", torch.bool))
        for a in range(NUM_A):
            _, obs, reward = world.step_rules(state, torch.full((k,), a, dtype=torch.int64, device=dev))
            ref = torch.bincount(obs, minlength=NUM_O).double() / k
            if (ref - probs[a]).abs().max() > 1e-12 or abs(float(reward.double().mean()) - float(rewards[a])) > 1e-9:
                raise RuntimeError(f"hybrid exact model differs from the per-node reference at action {a}.")


class Progress:
    """Pass-through agent: records every action and prints a decision counter (a stall shows within minutes)."""

    def __init__(self, actor):
        self.actor, self.batch_size, self.decisions, self.started = actor, actor.batch_size, 0, time.time()
        self.actions: list[Tensor] = []

    def reset(self, initial_observation):
        self.actor.reset(initial_observation)

    def update(self, action, observation):
        self.actor.update(action, observation)

    def act(self):
        self.decisions += 1
        if self.decisions % 10 == 0:
            print(f"    [{time.strftime('%H:%M:%S')}] decision {self.decisions}; {time.time() - self.started:.0f} s",
                  flush=True)
        action = self.actor.act()
        self.actions.append(action.clone())
        return action


def play(search_model):
    planner = PlanningAgent(
        search_model, BeliefTreeSearch(search_model, cfg.mcts.num_simulations, cfg.mcts.c_puct,
                                       cfg.mcts.dirichlet_alpha, dirichlet_epsilon=0.0,
                                       seed=stream_seed(settings.seed, 2), observation_prune=cfg.mcts.observation_prune),
        n, temperature=0.0, seed=stream_seed(settings.seed, 3), device=dev)
    actor = Progress(planner)
    episodes, _ = play_episodes(WumpusSimulator(world, n, cfg.env.max_steps, stream_seed(settings.seed, 1), dev), actor)
    del planner                                    # BeliefTreeSearch keeps its last trees
    torch.cuda.empty_cache()
    return episodes, torch.stack(actor.actions, 1), time.time() - actor.started


def report(name: str, episodes, seconds: float) -> np.ndarray:
    returns = discounted_returns(episodes.rewards, world.discount).double().cpu().numpy()
    out = {k: round(float(v), 4) for k, v in game_outcomes(episodes.rewards).items()}
    over = (episodes.rewards.abs() >= 998)
    live = (over.long().cumsum(1) - over.long()) == 0
    acts = episodes.actions[live]
    mix = ", ".join(f"{NAMES[a]} {(acts == a).float().mean():.3f}" for a in range(NUM_A))
    print(f"  {name:8s}: return {returns.mean():7.1f} +- {returns.std() / np.sqrt(n):5.1f}  {out}  "
          f"({seconds / 60:.1f} min, {seconds / cfg.env.max_steps:.1f} s per decision)\n      live action mix {mix}",
          flush=True)
    return returns


print(f"== {run_dir.name} best.pt (collection {trained.checkpoint_collection}), arm {arm}, {n} games", flush=True)
if arm.startswith("smoke"):
    exact_arm = arm[6:] or "EPR"
    stock, stock_actions, secs = play(model)
    base = report("learned", stock, secs)
    same, same_actions, secs = play(HybridSearchModel(model, False, False, False, n))
    if not torch.equal(stock_actions, same_actions):
        raise RuntimeError("The hybrid with no exact parts does not reproduce the stock planner's actions.")
    print("  hybrid (no exact parts) = stock planner, bit for bit", flush=True)
    epr, _, secs = play(HybridSearchModel(model, *ARMS[exact_arm], n, check=True))
    print("  exact P and R = per-node reference at every decision's first expansion", flush=True)
    gain = report(exact_arm, epr, secs) - base
    print(f"  {exact_arm} paired gain {gain.mean():+.1f} +- {1.96 * gain.std() / np.sqrt(n):.1f} (smoke, {n} games)", flush=True)
else:
    search_model = model if arm == "learned" else HybridSearchModel(model, *ARMS[arm], n)
    episodes, _, secs = play(search_model)
    returns = report(arm, episodes, secs)
    if n == settings.episodes:
        if arm == "learned":
            reported = json.loads((run_dir / "analysis" / "report.json").read_text())["returns"]["learned_planner"]
            if abs(reported["discounted_return"] - returns.mean()) > 1e-6:
                raise RuntimeError(f"Replay is not faithful: {returns.mean()} vs {reported['discounted_return']}.")
            print("  replay reproduces report.json", flush=True)
        out = game_outcomes(episodes.rewards)
        np.savez(run_dir / "analysis" / f"exact_play_{arm}.npz", returns=returns,
                 rewards=episodes.rewards.cpu().numpy(), actions=episodes.actions.cpu().numpy(),
                 **{k: float(v) for k, v in out.items()})
        base_file = run_dir / "analysis" / "exact_play_learned.npz"
        if arm != "learned" and base_file.exists():
            base = np.load(base_file)
            d = returns - base["returns"]
            print(f"  paired gain over learned {d.mean():+.1f} +- {1.96 * d.std() / np.sqrt(n):.1f}; unfinished "
                  f"{float(base['unfinished_rate']):.3f} -> {float(out['unfinished_rate']):.3f}, deaths "
                  f"{float(base['death_rate']):.3f} -> {float(out['death_rate']):.3f}", flush=True)
print("DONE", flush=True)
