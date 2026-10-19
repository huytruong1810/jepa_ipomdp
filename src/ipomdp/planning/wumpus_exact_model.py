# ABSOLUTE PATH: src/ipomdp/planning/wumpus_exact_model.py
# ==============================================================================
# EXACT-BELIEF WUMPUS SEARCH MODEL: THE ONLINE-PLANNING REFERENCE (NO LEARNING)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Why a Planning Reference (HANDOFF section 1c, gap 3; 2026-10-19):
#    - The only Wumpus yardstick so far is a hand-written heuristic on the exact belief
#      (agents/wumpus_reference.py, 573 per game), whose risk threshold was tuned on the same
#      evaluation games. A reviewer expects a standard online POMDP planner with the TRUE model
#      (POMCP, Silver & Veness 2010; DESPOT) as the upper reference for a learned planner.
#    - This model plugs the true Wumpus rules into the SAME belief-tree search the learned
#      agent uses (planning/mcts.py: PUCT selection, exact observation branching, expectimax
#      backups). The planners then differ only in the model (exact vs learned) and the leaf
#      estimator (rollouts vs the learned value), which is the comparison the paper needs.
#
# 2. Exact Beliefs Instead of Particles:
#    - POMCP represents beliefs by particles because exact beliefs are usually intractable.
#      Here the exact posterior is the uniform prior over the 16,380 layouts restricted to
#      those consistent with the history, and the hunter's own state is shared by all of them
#      (domain/wumpus.py, section 5). Exact beliefs remove particle depletion and make every
#      R(b, a) and P(o | b, a) exact.
#    - The search state is a node id (int64, shape (N, 1)). A registry holds, per node, its
#      parent, action, observation, hunter state and steps left; masks are materialised only
#      for EXPANDED nodes, as mask(child) = mask(parent) & (obs(parent, a) == o), and stored
#      bit-packed (2 KB per node instead of 16 KB), so a 512-root, 1000-simulation search
#      fits in ~1 GB of masks. This is the registry of
#      reports/2026-10-02_wumpus_diagnostics/exact_model_play.py (HybridSearchModel), with the
#      learned latent removed.
#    - Every expansion evaluates step_rules once per action over the (g, L) layout hypotheses
#      of a chunk of g nodes: R(b, a) = mean reward over consistent layouts, P(o | b, a) =
#      share of consistent layouts emitting o. Tested against a per-node reference
#      (tests/test_wumpus.py).
#
# 3. The 50-Step Window Is Part of the State:
#    - The scored return is the discounted return inside the window, so each node carries
#      its steps left k (root: max_steps - t). A node with k = 0 pays 0 for every action and
#      its children are worth 0: the search cannot collect reward after the window, and it is
#      not misled by a stationary value bootstrapping through the truncation (the problem
#      HANDOFF section 5.6 diagnosed in the learned value).
#    - Finished games are absorbing in the rules themselves (reward 0, GAME_OVER), so a
#      game-over child is worth exactly 0. With an exact model that is correct (the
#      co-adaptation caveat of exact_model_play.py, header 3a, concerns learned values only).
#
# 4. Leaf Values (`leaf`; development smoke of 2026-10-19 in
#    reports/2026-10-19_eval_protocol/exact_planner_smoke.py, 128 development games, heuristic
#    544 on the same games):
#    - "full_info" (the reference's leaf): the posterior mean of the KNOWN-LAYOUT value with the
#      child's remaining window, V(child) = mean over its consistent layouts l of V_k(l, h), where
#      V_k is the exact k-step full-information value of hunter state h in layout l
#      (full_information_values: finite-horizon value iteration over 16,380 layouts x 257 hunter
#      states with the true rules). This is hindsight optimisation (Yoon et al. 2008) and the
#      leaf upper bound of DESPOT (Somani et al. 2013): optimistic about information beyond the
#      leaf, while every risk INSIDE the tree is evaluated exactly. It is computed exactly per
#      branch by a bincount over consistent layouts (no sampling), so it is deterministic.
#    - "zero": every leaf 0. 50 simulations: 145 per game, 0% deaths, 80% unfinished, FORWARD
#      on 92% of live steps (walking into walls: a safe bump costs only -1 and an unexpanded
#      continuation is worth 0, the same as exploring).
#    - "rollout" (POMCP's estimator), described below. 50 simulations, 8 rollouts: 11 per game,
#      91% unfinished, a near-uniform action mix. A uniform hunter dies in ~55% of games, so
#      every leaf looks equally doomed and the argmax is noise; the uninformed rollout policy
#      is the known weak point of POMCP (Silver & Veness use domain "preferred actions"). The
#      200-simulation arms were stopped: more search cannot repair the leaf estimator.
#    - Rollouts: a new non-terminal child with k > 0 steps left gets the mean discounted return of
#      `rollouts` simulated continuations: each draws a layout uniformly from the child's
#      consistent set (an exact posterior sample, POMCP's particle) and plays the default
#      policy on the true rules for the child's remaining k steps.
#    - Default policy: uniform over the actions that are not certain no-ops given the
#      hunter's own percepts: FORWARD, TURN_LEFT and TURN_RIGHT always, GRAB only on glitter
#      (gold in the hunter's cell), SHOOT only with the arrow. That is the uninformed rollout
#      policy of Silver & Veness (2010) with legal-action pruning, and it uses nothing a real
#      hunter could not perceive.
#    - All rollouts of one expand call run as ONE batch on the device, with a private
#      torch.Generator, so a search is reproducible and costs ~k batched step_rules calls per
#      expand rather than per node.
#
# 5. Efficiency Notes:
#    - Exact parts are chunked (CHUNK nodes per step_rules call: (g, L) = (128, 16,380); 32 made
#      a 512-root search launch-bound, 2026-10-19 profile), the
#      only device-to-host traffic is the Expansion itself, and the registry is preallocated
#      from (roots, simulations), so no per-node Python work happens here.
#    - The registry is rebuilt from the roots at every initial_states / update: the search
#      rebuilds its tree at every decision (planning/mcts.py), so nothing older is reachable.
# ==============================================================================

import torch
import torch.nn.functional as F
from torch import Tensor

from ..domain.wumpus import WumpusAction, WumpusBeliefTracker, WumpusState, WumpusWorld
from .search_model import Expansion

CHUNK = 128
LEAVES = ("full_info", "zero", "rollout")
_OWN_FIELDS = ("cell", "heading", "arrow", "alive", "done")
HUNTER_STATES = 257          # cell * 16 + heading * 4 + arrow * 2 + alive (256), and 256 = game over


def hunter_index(state: WumpusState) -> Tensor:
    """Index of the hunter's own state in 0..256 (HUNTER_STATES); 256 for a finished game."""
    index = state.cell * 16 + state.heading * 4 + state.arrow.long() * 2 + state.wumpus_alive.long()
    return torch.where(state.done, HUNTER_STATES - 1, index)


def full_information_values(world: WumpusWorld, horizon: int) -> Tensor:
    """
    V_k(layout, hunter) for k = 0..horizon steps left, float32 (horizon + 1, L, 257): the exact
    finite-horizon value with the layout KNOWN (section 4), V_0 = 0 and the game-over state worth
    0, by backward induction with the domain's own step_rules (as
    reports/2026-10-02_wumpus_diagnostics/full_info_vi.py, which runs it to convergence instead).
    """
    num_layouts, device = world.num_layouts, world.device
    hunters = torch.arange(HUNTER_STATES - 1, device=device)
    layouts = torch.arange(num_layouts, device=device)[:, None].expand(num_layouts, HUNTER_STATES - 1)
    h = hunters[None].expand(num_layouts, -1)
    state = WumpusState(layouts, h // 16, (h // 4) % 4, ((h // 2) % 2).bool(), (h % 2).bool(),
                        torch.zeros_like(h, dtype=torch.bool))
    successors, rewards = [], []
    for a in range(world.num_actions):
        nxt, _, reward = world.step_rules(state, torch.full_like(h, a))
        successors.append(hunter_index(nxt))
        rewards.append(reward.double())
    successors, rewards = torch.stack(successors, -1).view(num_layouts, -1), torch.stack(rewards, -1)
    values = torch.zeros(horizon + 1, num_layouts, HUNTER_STATES, dtype=torch.float32, device=device)
    value = torch.zeros(num_layouts, HUNTER_STATES, dtype=torch.float64, device=device)
    for k in range(1, horizon + 1):
        q = rewards + world.discount * value.gather(1, successors).view(num_layouts, HUNTER_STATES - 1, -1)
        value = torch.cat([q.max(-1).values, value[:, -1:]], 1)
        values[k] = value.float()
    return values


class ExactWumpusSearchModel:
    """Exact layout-mask beliefs, exact R and P, exact or rollout leaf values (module header)."""

    def __init__(self, world: WumpusWorld, batch_size: int, num_simulations: int, max_steps: int, leaf: str,
                 rollouts: int, seed: int):
        """
        Args:
            world: The Wumpus rules (its device is the model's device).
            batch_size: Number of roots B (parallel games).
            num_simulations: Simulations per search; sizes the registry (one expansion per root and simulation).
            max_steps: The window length; roots start with max_steps steps left (section 3).
            leaf: "full_info", "zero" or "rollout" (section 4).
            rollouts: Rollouts per new non-terminal leaf; > 0 exactly when leaf is "rollout".
            seed: Seed of the rollout generator.
        """
        if leaf not in LEAVES or (rollouts > 0) != (leaf == "rollout") or rollouts < 0:
            raise ValueError(f"leaf must be one of {LEAVES}, with rollouts > 0 exactly for 'rollout'; got {leaf}, {rollouts}.")
        if max_steps < 1 or num_simulations < 1:
            raise ValueError(f"need max_steps >= 1 and num_simulations >= 1; got {max_steps}, {num_simulations}.")
        self.leaf = leaf
        self.full_info = full_information_values(world, max_steps).view(-1) if leaf == "full_info" else None
        self.world, self.device = world, world.device
        self.num_actions, self.num_observations, self.discount = world.num_actions, world.num_observations, world.discount
        self.batch, self.max_steps, self.rollouts = batch_size, max_steps, rollouts
        self.tracker = WumpusBeliefTracker(world)
        self.num_layouts = world.num_layouts
        self.packed_width = (self.num_layouts + 7) // 8
        self._bits = 2 ** torch.arange(8, device=self.device, dtype=torch.uint8)
        self._shifts = torch.arange(8, device=self.device, dtype=torch.uint8)
        self.layouts = torch.arange(self.num_layouts, device=self.device)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

        bank_cap = batch_size * (num_simulations + 2)
        cap = bank_cap * self.num_actions * self.num_observations + batch_size
        self.bank = torch.zeros(bank_cap, self.packed_width, dtype=torch.uint8, device=self.device)
        self.bank_row = torch.full((cap,), -1, dtype=torch.int64, device=self.device)
        self.parent = torch.full((cap,), -1, dtype=torch.int64, device=self.device)
        self.action = torch.zeros(cap, dtype=torch.int8, device=self.device)
        self.obs = torch.zeros(cap, dtype=torch.int8, device=self.device)
        self.steps = torch.zeros(cap, dtype=torch.int16, device=self.device)
        self.own = {k: torch.zeros(cap, dtype=torch.int8, device=self.device) for k in _OWN_FIELDS}
        self.root_consistent: Tensor | None = None
        self.root_own: WumpusState | None = None
        self.root_steps = 0
        self.next_id = self.next_row = 0

    # --- bit-packed masks (section 2) ---
    def _pack(self, masks: Tensor) -> Tensor:
        padded = F.pad(masks, (0, 8 * self.packed_width - self.num_layouts)).view(-1, self.packed_width, 8)
        return (padded.to(torch.uint8) * self._bits).sum(-1, dtype=torch.uint8)

    def _unpack(self, packed: Tensor) -> Tensor:
        bits = (packed.unsqueeze(-1) >> self._shifts) & 1
        return bits.view(packed.shape[0], -1)[:, :self.num_layouts].bool()

    # --- the registry ---
    def _reset_registry(self) -> Tensor:
        b = self.batch
        self.bank[:b] = self._pack(self.root_consistent)
        self.bank_row[:max(self.next_id, b)] = -1
        self.bank_row[:b] = torch.arange(b, device=self.device)
        self.steps[:b] = self.root_steps
        own = self.root_own
        for k, v in zip(_OWN_FIELDS, (own.cell, own.heading, own.arrow, own.wumpus_alive, own.done)):
            self.own[k][:b] = v.to(torch.int8)
        self.next_id = self.next_row = b
        return torch.arange(b, device=self.device).unsqueeze(1)

    def _own_state(self, ids: Tensor, layouts: Tensor) -> WumpusState:
        """Hunter state of nodes `ids` broadcast against `layouts` (same leading shape as ids, or (g, L))."""
        shape = layouts.shape
        field = lambda k: self.own[k][ids].long().view(-1, *([1] * (len(shape) - 1))).expand(shape)  # noqa: E731
        return WumpusState(layouts, field("cell"), field("heading"), field("arrow").bool(), field("alive").bool(),
                           field("done").bool())

    def _materialise(self, ids: Tensor) -> Tensor:
        """Unpacked masks (g, L) of `ids`, creating the masks of lazy children from their parents (section 2)."""
        lazy = ids[self.bank_row[ids] < 0]
        if lazy.numel():
            parents = self.parent[lazy]
            if bool((self.bank_row[parents] < 0).any()):
                raise RuntimeError("a lazy node's parent has no materialised mask.")
            _, obs, _ = self.world.step_rules(self._own_state(parents, self.layouts.expand(parents.shape[0], -1)),
                                              self.action[lazy].long().unsqueeze(1))
            masks = self._unpack(self.bank[self.bank_row[parents]]) & (obs == self.obs[lazy].long().unsqueeze(1))
            if not bool(masks.any(1).all()):
                raise RuntimeError("an expanded node has no consistent layout (an exact-zero branch was expanded).")
            rows = torch.arange(self.next_row, self.next_row + lazy.numel(), device=self.device)
            if self.next_row + lazy.numel() > self.bank.shape[0]:
                raise RuntimeError("mask bank full: num_simulations does not match the search.")
            self.bank[rows] = self._pack(masks)
            self.bank_row[lazy] = rows
            self.next_row += lazy.numel()
        return self._unpack(self.bank[self.bank_row[ids]])

    # --- the SearchModel protocol ---
    def initial_states(self, batch_size: int, initial_observations: Tensor | None) -> Tensor:
        if batch_size != self.batch or initial_observations is None:
            raise ValueError(f"needs the o_0 of exactly {self.batch} games.")
        self.root_consistent, self.root_own = self.tracker.start(initial_observations)
        self.root_steps = self.max_steps
        return self._reset_registry()

    def update(self, states: Tensor, actions: Tensor, observations: Tensor) -> Tensor:
        self.root_consistent, self.root_own = self.tracker.update(self.root_consistent, self.root_own, actions,
                                                                  observations)
        self.root_steps -= 1
        if self.root_steps < 0:
            raise RuntimeError("update() past the end of the window.")
        return self._reset_registry()

    @torch.no_grad()
    def expand(self, states: Tensor) -> Expansion:
        num, num_a, num_o = states.shape[0], self.num_actions, self.num_observations
        ids = states[:, 0].long()
        child_ids = torch.arange(self.next_id, self.next_id + num * num_a * num_o, device=self.device)
        child_ids = child_ids.view(num, num_a, num_o)
        self.next_id += num * num_a * num_o
        if self.next_id > self.parent.shape[0]:
            raise RuntimeError("node registry full: num_simulations does not match the search.")
        rewards, probs, live, samples, leaves = [], [], [], [], []
        for i in range(0, num, CHUNK):
            r, p, alive, sample, leaf = self._exact(ids[i:i + CHUNK], child_ids[i:i + CHUNK])
            rewards.append(r), probs.append(p), live.append(alive), samples.append(sample), leaves.append(leaf)
        rewards, probs, live, values = torch.cat(rewards), torch.cat(probs), torch.cat(live), torch.cat(leaves)
        # Chunks are contiguous node ranges and each chunk's samples are in (g, A, O) row-major order,
        # so their concatenation is in the order of values[live].
        samples = [s for s in samples if s is not None]
        if samples:
            values[live] = self._rollout(child_ids[live], torch.cat(samples))
        return Expansion(rewards, probs, child_ids.unsqueeze(-1), values)

    def _exact(self, ids: Tensor, child_ids: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor]:
        """
        Exact R(b, a) (g, A), P(o | b, a) (g, A, O), the live-leaf flags (g, A, O) (reachable, not
        game over, steps left), for every live child in (g, A, O) order `rollouts` layouts drawn
        from its consistent set (n_live, rollouts) or None, and the exact leaf values (g, A, O)
        (full_info; zeros otherwise, rollouts fill them in expand); registers the children.
        """
        g, num_a, num_o = ids.shape[0], self.num_actions, self.num_observations
        masks = self._materialise(ids)                                                 # (g, L)
        hypotheses = self._own_state(ids, self.layouts.expand(g, -1))
        count = masks.sum(1).double()
        steps = self.steps[ids]
        in_window = steps > 0
        child_steps = (steps - 1).clamp(min=0)
        observations = torch.arange(num_o, device=self.device)
        rows = torch.arange(g, device=self.device).unsqueeze(1).expand(g, self.num_layouts)
        layout_ids = self.layouts.expand(g, -1)[masks]                               # consistent layouts, flat
        rewards = torch.zeros(g, num_a, dtype=torch.float64, device=self.device)
        probs = torch.zeros(g, num_a, num_o, dtype=torch.float64, device=self.device)
        live = torch.zeros(g, num_a, num_o, dtype=torch.bool, device=self.device)
        leaf_values = torch.zeros(g, num_a, num_o, dtype=torch.float64, device=self.device)
        samples = []
        for a in range(num_a):
            nxt, obs, reward = self.world.step_rules(hypotheses, torch.full((g, 1), a, dtype=torch.int64,
                                                                            device=self.device))
            rewards[:, a] = torch.where(in_window, (reward.double() * masks).sum(1) / count, 0.0)
            # Branch key (node, o) of every consistent layout; O(g L), never a dense (g, O, L) tensor.
            key = (rows * num_o + obs)[masks]
            branch = torch.bincount(key, minlength=g * num_o)                          # (g O,)
            probs[:, a] = branch.view(g, num_o).double() / count[:, None]
            first = torch.full((g * num_o,), self.num_layouts - 1, dtype=torch.int64, device=self.device)
            first.scatter_reduce_(0, key, layout_ids, reduce="amin")                   # a consistent layout per o
            first = first.view(g, num_o)
            pick = lambda t: t.gather(1, first)  # noqa: E731
            cid = child_ids[:, a]
            self.parent[cid] = ids.unsqueeze(1).expand(g, num_o)
            self.action[cid] = a
            self.obs[cid] = observations.to(torch.int8).expand(g, num_o)
            self.steps[cid] = child_steps.unsqueeze(1).expand(g, num_o)
            for k, field in zip(_OWN_FIELDS, (nxt.cell, nxt.heading, nxt.arrow, nxt.wumpus_alive, nxt.done)):
                self.own[k][cid] = pick(field).to(torch.int8)
            live[:, a] = (branch.view(g, num_o) > 0) & ~pick(nxt.done) & (child_steps > 0).unsqueeze(1)
            if self.leaf == "full_info":
                # Posterior mean of V_k(l, h) per branch: one bincount weighted by the table entries.
                entry = ((child_steps.long().unsqueeze(1) * self.num_layouts + self.layouts) * HUNTER_STATES
                         + hunter_index(nxt))
                sums = torch.bincount(key, weights=self.full_info[entry[masks]].double(), minlength=g * num_o)
                mean = sums.view(g, num_o) / branch.view(g, num_o).clamp(min=1)
                leaf_values[:, a] = torch.where(live[:, a], mean, 0.0)
            if self.rollouts > 0 and bool(live[:, a].any()):
                # Uniform posterior samples per live branch: sort the consistent layouts by branch key,
                # then draw offsets inside each branch's contiguous run.
                sorted_layouts = layout_ids[torch.argsort(key, stable=True)]
                starts = torch.cumsum(branch, 0) - branch
                branches = torch.nonzero(live[:, a].flatten()).squeeze(1)
                offsets = (torch.rand(branches.shape[0], self.rollouts, generator=self.generator, device=self.device)
                           * branch[branches, None]).long()
                samples.append((a, sorted_layouts[starts[branches, None] + offsets]))
        if not samples:
            return rewards, probs, live, None, leaf_values
        # Reorder the per-action samples into (g, A, O) row-major order, the order of values[live].
        flat = torch.cat([s for _, s in samples])
        action_of = torch.cat([torch.full((s.shape[0],), a, device=self.device) for a, s in samples])
        node, obs_index = torch.cat([torch.nonzero(live[:, a]) for a, _ in samples]).unbind(1)
        order = torch.argsort(node * num_a * num_o + action_of * num_o + obs_index)
        return rewards, probs, live, flat[order], leaf_values

    def _rollout(self, leaves: Tensor, layouts: Tensor) -> Tensor:
        """Mean discounted return of the default policy from each leaf (section 4); leaves (n,), layouts (n, K)."""
        n, k = layouts.shape
        ids = leaves.unsqueeze(1).expand(n, k).reshape(-1)
        state = self._own_state(ids, layouts.reshape(-1))
        horizon = self.steps[ids].long()
        total = torch.zeros(n * k, dtype=torch.float64, device=self.device)
        weight = 1.0
        allowed = torch.ones(n * k, self.num_actions, dtype=torch.bool, device=self.device)
        for t in range(int(horizon.max())):
            allowed[:, WumpusAction.GRAB] = self.world.gold[state.layout] == state.cell
            allowed[:, WumpusAction.SHOOT] = state.arrow
            action = torch.multinomial(allowed.float(), 1, generator=self.generator).squeeze(1)
            state, _, reward = self.world.step_rules(state, action)
            total += weight * torch.where(t < horizon, reward.double(), 0.0)
            weight *= self.discount
            if t % 10 == 9 and not bool((~state.done & (t + 1 < horizon)).any()):
                break
        return total.view(n, k).mean(1)
