# ABSOLUTE PATH: src/ipomdp/agents/wumpus_reference.py
# ==============================================================================
# EXACT-BELIEF REFERENCE AGENT FOR WUMPUS WORLD (A LOWER BOUND ON V*)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Why a Reference Agent (user decision, 2026-10-02):
#    - Wumpus World has no exact solver (domain/wumpus.py, section 4), so a learned agent's
#      return has no optimum to be compared with. This agent supplies the missing yardstick
#      from below: any policy's return is a lower bound on V*(b0). The full-information bound
#      (the mean over layouts of each known layout's MDP value, about 936) bounds it from above.
#    - It is a hand-written policy on the EXACT belief (WumpusBeliefTracker), in the spirit of
#      R&N's hybrid Wumpus agent. It is not optimal; it is a reference, never a training signal.
#
# 2. The Policy, Evaluated in Priority Order at Every Step:
#    a. GLITTER here (P(gold at own cell) = 1): GRAB, which wins.
#    b. Explore safely: walk to the nearest provably safe unvisited cell. A cell is provably
#       safe when P(pit) = 0 and, while the wumpus lives, P(wumpus) = 0. The posterior is the
#       uniform prior restricted to consistent layouts, so "probability 0" is an exact zero.
#       A cell with P(gold) = 1 is always provably safe (pits, wumpus and gold occupy distinct
#       cells) and is preferred over every other target.
#    c. Shoot to clear: with the arrow, a living wumpus, and no safe target left, walk to the
#       nearest (safe cell, heading) whose cell straight ahead is pit-free, unvisited and may
#       hold the wumpus, and SHOOT. Either outcome makes that cell provably safe: SCREAM means
#       the wumpus is dead, silence means it is on no cell of the arrow's line.
#    d. Gamble: otherwise step into the frontier cell (unvisited, next to a reachable safe
#       cell) with the lowest death probability P(pit) + P(wumpus alive there), if that
#       probability is at most `risk_threshold`.
#    e. Idle: otherwise TURN_LEFT. There is no Climb (user decision), so waiting out the window
#       at -1 per step is the remaining choice.
#    - risk_threshold = 0 never gambles, so the agent can never die (tested); larger values
#      trade deaths for wins. The analysis reports a small grid of thresholds.
#
# 3. Movement Is Exact Shortest-Path Search:
#    - Breadth-first search over (cell, heading) with unit-cost FORWARD / TURN_LEFT /
#      TURN_RIGHT, entering provably safe cells only. Every action costs -1 (shooting -10), so
#      the fewest actions are the cheapest.
#    - Search runs per game in Python on the CPU (64 nodes); the exact filter runs batched on
#      the simulator's device. Marginals are copied to the CPU synchronously (CLAUDE.md:
#      non_blocking copies corrupted memory before).
#
# 4. Deterministic:
#    - The policy has no randomness (ties go to the first action in FORWARD, TURN_LEFT,
#      TURN_RIGHT order and the lowest cell index), so it needs no seed and no checkpoint.
# ==============================================================================

from collections import deque

import torch
from torch import Tensor

from ..domain.wumpus import GRID, NUM_CELLS, WumpusAction, WumpusBeliefTracker, WumpusState, WumpusWorld

_DELTA = ((1, 0), (0, 1), (-1, 0), (0, -1))          # E, N, W, S (domain/wumpus.py, Heading)
_ZERO = 1e-12                                          # exact-zero test for enumerated probabilities


def _ahead(cell: int, heading: int) -> int | None:
    """The cell in front of the hunter, or None at a wall."""
    x, y = cell % GRID + _DELTA[heading][0], cell // GRID + _DELTA[heading][1]
    return x + GRID * y if 0 <= x < GRID and 0 <= y < GRID else None


def _first_actions(cell: int, heading: int, passable: list[bool]) -> dict[tuple[int, int], WumpusAction]:
    """BFS over (cell, heading) through passable cells: the first action of a shortest path to each node."""
    start = (cell, heading)
    first: dict[tuple[int, int], WumpusAction] = {}
    queue = deque([start])
    seen = {start}
    while queue:
        node = queue.popleft()
        c, h = node
        ahead = _ahead(c, h)
        moves = ((WumpusAction.FORWARD, (ahead, h) if ahead is not None and passable[ahead] else None),
                 (WumpusAction.TURN_LEFT, (c, (h + 1) % 4)),
                 (WumpusAction.TURN_RIGHT, (c, (h + 3) % 4)))
        for action, nxt in moves:
            if nxt is None or nxt in seen:
                continue
            seen.add(nxt)
            first[nxt] = first.get(node, action)
            queue.append(nxt)
    return first


class WumpusReferenceAgent:
    """Section 2's policy on the exact enumeration belief; implements the rollout Agent protocol."""

    def __init__(self, world: WumpusWorld, batch_size: int, risk_threshold: float):
        if not 0.0 <= risk_threshold <= 1.0:
            raise ValueError(f"risk_threshold must lie in [0, 1], got {risk_threshold}.")
        self.world = world
        self.batch_size = batch_size
        self.risk_threshold = risk_threshold
        self._tracker = WumpusBeliefTracker(world)
        self._consistent: Tensor | None = None
        self._own: WumpusState | None = None
        self._visited: list[list[bool]] = []

    def reset(self, initial_observation: Tensor | None) -> None:
        if initial_observation is None:
            raise ValueError("Wumpus World emits the start-cell percept o_0.")
        self._consistent, self._own = self._tracker.start(initial_observation)
        self._visited = [[c == 0 for c in range(NUM_CELLS)] for _ in range(self.batch_size)]

    def update(self, action: Tensor, observation: Tensor) -> None:
        self._consistent, self._own = self._tracker.update(self._consistent, self._own, action, observation)
        for row, cell in enumerate(self._own.cell.tolist()):
            self._visited[row][cell] = True

    def act(self) -> Tensor:
        m = self._tracker.marginals(self._consistent, self._own)
        pit, wumpus, gold = m.pit.cpu().tolist(), m.wumpus.cpu().tolist(), m.gold.cpu().tolist()
        cells, headings = m.cell.tolist(), m.heading.tolist()
        arrows, alive, done = m.arrow.tolist(), m.wumpus_alive.tolist(), m.done.tolist()
        actions = []
        for b in range(self.batch_size):
            if done[b]:
                actions.append(WumpusAction.TURN_LEFT)          # absorbing: every action pays 0
                continue
            danger = [pit[b][c] + (wumpus[b][c] if alive[b] else 0.0) for c in range(NUM_CELLS)]
            actions.append(self._decide(cells[b], headings[b], arrows[b], alive[b], danger, wumpus[b], gold[b],
                                        self._visited[b]))
        return torch.tensor([int(a) for a in actions], dtype=torch.int64, device=self.world.device)

    def _decide(self, cell: int, heading: int, arrow: bool, alive: bool, danger: list[float],
                wumpus: list[float], gold: list[float], visited: list[bool]) -> WumpusAction:
        """Section 2, rules a-e, for one game."""
        if gold[cell] > 1.0 - _ZERO:
            return WumpusAction.GRAB                                                                # (a)
        safe = [danger[c] < _ZERO for c in range(NUM_CELLS)]
        first = _first_actions(cell, heading, safe)
        reachable = {c for c, _ in first} | {cell}

        known_gold = [c for c in range(NUM_CELLS) if gold[c] > 1.0 - _ZERO and c in reachable]
        targets = known_gold or [c for c in range(NUM_CELLS) if safe[c] and not visited[c] and c in reachable]
        if targets:                                                                                 # (b)
            return self._towards(first, lambda c, h: c in targets)

        if arrow and alive:                                                                         # (c)
            def shooting_spot(c: int, h: int) -> bool:
                target = _ahead(c, h)
                return target is not None and not visited[target] and wumpus[target] > _ZERO \
                    and danger[target] - wumpus[target] < _ZERO
            if shooting_spot(cell, heading):
                return WumpusAction.SHOOT
            if any(shooting_spot(c, h) for c, h in first):
                return self._towards(first, shooting_spot)

        # (d) Lowest risk first, then the nearest stepping-off node (BFS order; the current node is nearest).
        nodes = [(cell, heading), *first]
        frontier = [(danger[t], distance, c, h) for distance, (c, h) in enumerate(nodes)
                    if (t := _ahead(c, h)) is not None and not visited[t] and not safe[t]]
        if frontier:
            risk, _, c, h = min(frontier)
            if risk <= self.risk_threshold:
                if (c, h) == (cell, heading):
                    return WumpusAction.FORWARD
                return self._towards(first, lambda cc, hh: (cc, hh) == (c, h))
        return WumpusAction.TURN_LEFT                                                              # (e)

    @staticmethod
    def _towards(first: dict[tuple[int, int], WumpusAction], goal) -> WumpusAction:
        """First action of a shortest path to the nearest node satisfying goal(cell, heading); BFS order is distance order."""
        for node, action in first.items():
            if goal(*node):
                return action
        raise RuntimeError("_towards called without a reachable goal node.")
