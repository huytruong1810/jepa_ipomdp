# ABSOLUTE PATH: src/ipomdp/domain/wumpus.py
# ==============================================================================
# WUMPUS WORLD (4x4, SINGLE HUNTER): RULES, BATCHED SIMULATOR, EXACT ENUMERATION FILTER
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. The World (Russell & Norvig's Wumpus World, the variant chosen by the user, 2026-10-01):
#    - A 4x4 grid. Cell index c = x + 4 y with x, y in 0..3; R&N's square [1,1] is c = 0,
#      [2,1] is c = 1, [1,2] is c = 4. The hunter starts at c = 0 facing East.
#    - Exactly 2 pits, 1 wumpus and 1 gold, on pairwise distinct cells, none at the start.
#      Every such layout is equally likely: 105 pit pairs x 13 wumpus cells x 12 gold cells =
#      16,380 layouts. Unwinnable layouts (gold walled off by pits) are kept, as in R&N.
#    - Actions (no Climb; grabbing the gold wins): FORWARD, TURN_LEFT, TURN_RIGHT, GRAB, SHOOT.
#    - Rewards (R&N): -1 per action, -10 for shooting the arrow (instead of -1), +1000 for
#      grabbing the gold, -1000 for entering a pit or the live wumpus's cell. The step that ends
#      the game also pays its action cost: winning pays 999, dying -1001.
#    - The arrow flies straight ahead from the hunter's cell to the wall; it kills the wumpus
#      if the wumpus is anywhere on that line. A shot without the arrow costs -1 and does
#      nothing. The dead wumpus's cell is safe to enter.
#    - Discount gamma = 0.99 (user decision). R&N's performance measure is undiscounted; the
#      discount keeps values finite and its effective horizon (100) matches the 100-step
#      window (conf/env/wumpus.yaml).
#
# 2. Percepts (the 5 R&N bits, user decision; position and heading are NOT observed):
#      STENCH   the wumpus (alive or dead) is in the hunter's cell or a 4-neighbour
#      BREEZE   a pit is in a 4-neighbour
#      GLITTER  the gold is in the hunter's cell
#      BUMP     this action walked into a wall (the hunter did not move)
#      SCREAM   this action's arrow killed the wumpus
#    - Observation index = STENCH + 2 BREEZE + 4 GLITTER + 8 BUMP + 16 SCREAM (0..31), plus
#      GAME_OVER = 32. The hunter must dead-reckon its position from its own actions and the
#      bumps, as in R&N; position and heading are a deterministic function of the history.
#    - Initial observation (user decision): the hunter perceives its start cell before acting,
#      as in R&N. reset() therefore returns o_0 (the percepts at c = 0; never BUMP or SCREAM),
#      unlike canonical Tiger, whose reset() emits nothing (env.py, section 2).
#
# 3. End of the Game Is an Absorbing State (user decision):
#    - After a win or a death the row stays in an absorbing state: observation GAME_OVER and
#      reward 0 on every remaining step of the fixed 100-step window. The window holds exactly
#      one game, the training objective is the standard episodic discounted return, and the
#      replay buffer still stores fixed-length episodes (no masks). The learned model has to
#      learn that GAME_OVER is absorbing with value 0; its Bellman targets reach that fixed
#      point by themselves.
#    - The window may end before the game does (truncation, not termination; env.py
#      section 3): the learned value then bootstraps through it.
#
# 4. One Rules Function, Two Users (single source of truth, as FinitePOMDP is for Tiger):
#    - step_rules() advances a WumpusState of ANY batch shape. The simulator applies it to
#      (B,) states with their sampled layouts; the exact filter applies it to (B, L) states,
#      one hypothetical copy per layout, and keeps the layouts whose predicted observation
#      equals the real one. The filter therefore cannot disagree with the simulator.
#    - The tensors T[a, s, s'] of a FinitePOMDP would need ~6M states; they are not built.
#      There is no exact solver for this domain (user decision): learned agents are evaluated
#      by return, win and death rates, and their latents against the exact filter below.
#
# 5. Exact Belief by Enumeration (ground truth for probes and latent figures):
#    - Every percept is a deterministic function of (layout, hunter state), and the hunter's
#      own state (cell, heading, arrow) depends only on its actions and the observed bumps.
#      The exact posterior over layouts is therefore the uniform prior restricted to the
#      layouts consistent with every observation so far; it is tracked as a (B, L) mask.
#    - Wumpus life is shared by all consistent layouts (a kill is heard as SCREAM), so the
#      tracked hunter state is exact. Marginals P(pit at c), P(wumpus at c), P(gold at c) are
#      the probe targets (interpretability/wumpus_latents.py).
#    - Cost: (B, L) = (512, 16380) booleans per step on the GPU; a few ms per step.
# ==============================================================================

from dataclasses import dataclass
from enum import IntEnum
from itertools import combinations

import torch
from torch import Tensor

from .env import StepOutput

GRID = 4
NUM_CELLS = GRID * GRID
START_CELL = 0
NUM_PITS = 2
GOLD_REWARD = 1000.0
DEATH_REWARD = -1000.0
STEP_COST = -1.0
SHOOT_COST = -10.0
DISCOUNT = 0.99


class WumpusAction(IntEnum):
    FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    GRAB = 3
    SHOOT = 4


class Heading(IntEnum):
    EAST = 0
    NORTH = 1
    WEST = 2
    SOUTH = 3


class Percept(IntEnum):
    """Bit values of the observation index (section 2)."""

    STENCH = 1
    BREEZE = 2
    GLITTER = 4
    BUMP = 8
    SCREAM = 16


GAME_OVER = 32
NUM_OBSERVATIONS = GAME_OVER + 1


def _observation_name(index: int) -> str:
    if index == GAME_OVER:
        return "GAME_OVER"
    bits = [p.name for p in Percept if index & p.value]
    return "+".join(bits) if bits else "NONE"


@dataclass(frozen=True)
class WumpusState:
    """
    Hunter and world state; every field has the same batch shape.

    Attributes:
        layout: Index into WumpusWorld's layout table, int64.
        cell: Hunter cell x + 4 y, int64.
        heading: Heading (0 E, 1 N, 2 W, 3 S), int64.
        arrow: The hunter still has its arrow, bool.
        wumpus_alive: The wumpus is alive, bool.
        done: The game is over (absorbing, section 3), bool.
    """

    layout: Tensor
    cell: Tensor
    heading: Tensor
    arrow: Tensor
    wumpus_alive: Tensor
    done: Tensor

    def as_tensor(self) -> Tensor:
        """Fields stacked into int64 of shape (..., 6), for diagnostics and checkpoints."""
        return torch.stack([self.layout, self.cell, self.heading, self.arrow.long(), self.wumpus_alive.long(),
                            self.done.long()], dim=-1)

    @staticmethod
    def from_tensor(packed: Tensor) -> "WumpusState":
        """Inverse of as_tensor()."""
        return WumpusState(packed[..., 0], packed[..., 1], packed[..., 2], packed[..., 3].bool(),
                           packed[..., 4].bool(), packed[..., 5].bool())


def _enumerate_layouts() -> tuple[Tensor, Tensor, Tensor]:
    """Every (pit pair, wumpus, gold) layout on distinct non-start cells (section 1)."""
    free = range(1, NUM_CELLS)
    pits, wumpus, gold = [], [], []
    for pit_pair in combinations(free, NUM_PITS):
        for w in free:
            if w in pit_pair:
                continue
            for g in free:
                if g in pit_pair or g == w:
                    continue
                pits.append(pit_pair)
                wumpus.append(w)
                gold.append(g)
    pit_mask = torch.zeros(len(pits), NUM_CELLS, dtype=torch.bool)
    pit_mask[torch.arange(len(pits)).unsqueeze(1), torch.tensor(pits)] = True
    return pit_mask, torch.tensor(wumpus), torch.tensor(gold)


def _adjacency() -> Tensor:
    """(16, 16) bool: 4-neighbourhood of the grid."""
    adjacent = torch.zeros(NUM_CELLS, NUM_CELLS, dtype=torch.bool)
    for c in range(NUM_CELLS):
        x, y = c % GRID, c // GRID
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < GRID and 0 <= ny < GRID:
                adjacent[c, nx + GRID * ny] = True
    return adjacent


class WumpusWorld:
    """The domain specification: names, discount, value bound and the layout tables (section 1)."""

    name = "wumpus"
    action_names = tuple(a.name for a in WumpusAction)
    observation_names = tuple(_observation_name(o) for o in range(NUM_OBSERVATIONS))
    discount = DISCOUNT
    has_initial_observation = True

    def __init__(self, device: torch.device):
        """
        Args:
            device: Device of the layout tables (the simulator's and the filter's device).
        """
        self.device = device
        pit, wumpus, gold = _enumerate_layouts()
        adjacent = _adjacency()
        self.pit = pit.to(device)                                                      # (L, 16)
        self.wumpus = wumpus.to(device)                                                # (L,)
        self.gold = gold.to(device)                                                    # (L,)
        self.breeze = (pit.float() @ adjacent.float() > 0).to(device)                  # (L, 16)
        self.stench = (adjacent | torch.eye(NUM_CELLS, dtype=torch.bool))[wumpus].to(device)   # (L, 16)
        self.num_layouts = len(wumpus)
        delta = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]])                        # E, N, W, S
        self._delta = delta.to(device)

    @property
    def num_actions(self) -> int:
        return len(self.action_names)

    @property
    def num_observations(self) -> int:
        return len(self.observation_names)

    @property
    def value_bound(self) -> float:
        """max|r| / (1 - gamma), with max|r| = 1001 (dying pays the step cost too)."""
        return (abs(DEATH_REWARD) + abs(STEP_COST)) / (1.0 - self.discount)

    def initial_state(self, layout: Tensor) -> WumpusState:
        """The hunter at the start cell facing East with its arrow, the wumpus alive."""
        zeros = torch.zeros_like(layout)
        return WumpusState(layout, zeros + START_CELL, zeros + Heading.EAST, torch.ones_like(layout, dtype=torch.bool),
                           torch.ones_like(layout, dtype=torch.bool), torch.zeros_like(layout, dtype=torch.bool))

    def percepts(self, state: WumpusState, bump: Tensor, scream: Tensor) -> Tensor:
        """Observation index of a state (section 2); GAME_OVER once the game is over."""
        index = (self.stench[state.layout, state.cell].long() * Percept.STENCH
                 + self.breeze[state.layout, state.cell].long() * Percept.BREEZE
                 + (self.gold[state.layout] == state.cell).long() * Percept.GLITTER
                 + bump.long() * Percept.BUMP + scream.long() * Percept.SCREAM)
        return torch.where(state.done, GAME_OVER, index)

    def initial_observation(self, state: WumpusState) -> Tensor:
        """o_0: the percepts of the start cell (no bump, no scream)."""
        no = torch.zeros_like(state.done)
        return self.percepts(state, no, no)

    def step_rules(self, state: WumpusState, action: Tensor) -> tuple[WumpusState, Tensor, Tensor]:
        """
        The rules of section 1 for states of any batch shape (section 4).

        Args:
            state: Current states.
            action: Actions broadcastable to the state's shape, int64.

        Returns:
            (next state, observation o_{t+1}, reward r_t as float32).
        """
        active = ~state.done
        x, y = state.cell % GRID, state.cell // GRID
        dx, dy = self._delta[state.heading, 0], self._delta[state.heading, 1]

        forward = active & (action == WumpusAction.FORWARD)
        nx, ny = x + dx, y + dy
        inside = (nx >= 0) & (nx < GRID) & (ny >= 0) & (ny < GRID)
        bump = forward & ~inside
        cell = torch.where(forward & inside, nx + GRID * ny, state.cell)

        heading = torch.where(active & (action == WumpusAction.TURN_LEFT), (state.heading + 1) % 4, state.heading)
        heading = torch.where(active & (action == WumpusAction.TURN_RIGHT), (state.heading + 3) % 4, heading)

        shoot = active & (action == WumpusAction.SHOOT) & state.arrow
        wumpus = self.wumpus[state.layout]
        wx, wy = wumpus % GRID, wumpus // GRID
        # The wumpus is strictly ahead on the hunter's row (E/W) or column (N/S).
        ahead = ((dx != 0) & (wy == y) & ((wx - x) * dx > 0)) | ((dy != 0) & (wx == x) & ((wy - y) * dy > 0))
        scream = shoot & state.wumpus_alive & ahead
        wumpus_alive = state.wumpus_alive & ~scream

        death = forward & (self.pit[state.layout, cell] | (wumpus_alive & (wumpus == cell)))
        win = active & (action == WumpusAction.GRAB) & (self.gold[state.layout] == state.cell)

        cost = torch.where(shoot, SHOOT_COST, STEP_COST)
        reward = torch.where(active, cost + GOLD_REWARD * win + DEATH_REWARD * death, 0.0).float()
        next_state = WumpusState(state.layout, cell, heading, state.arrow & ~shoot, wumpus_alive,
                                 state.done | death | win)
        return next_state, self.percepts(next_state, bump, scream), reward


class WumpusSimulator:
    """B independent games in lock-step; same interface as BatchedPOMDPEnv (training/rollout.py)."""

    def __init__(self, world: WumpusWorld, batch_size: int, max_steps: int, seed: int, device: torch.device):
        """
        Args:
            world: The domain (its tables must live on `device`).
            batch_size: Number of parallel games B.
            max_steps: Steps per window (truncation, section 3).
            seed: Seed of the private layout generator.
            device: Device of every simulator tensor.
        """
        if batch_size < 1 or max_steps < 1:
            raise ValueError(f"batch_size and max_steps must be >= 1, got {batch_size}, {max_steps}.")
        if world.device != device:
            raise ValueError(f"world tables live on {world.device}, simulator on {device}.")
        self.world = world
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.device = device
        self._generator = torch.Generator(device=device)
        self._generator.manual_seed(seed)
        self._state = world.initial_state(torch.zeros(batch_size, dtype=torch.int64, device=device))
        self._elapsed = torch.zeros(batch_size, dtype=torch.int64, device=device)

    @property
    def state(self) -> Tensor:
        """Packed hidden state (WumpusState.as_tensor), int64 shape (B, 6). Diagnostics only."""
        return self._state.as_tensor()

    def state_dict(self) -> dict:
        return {"generator": self._generator.get_state(), "state": self.state, "elapsed": self._elapsed.clone()}

    def load_state_dict(self, state: dict) -> None:
        self._generator.set_state(state["generator"])
        self._state = WumpusState.from_tensor(state["state"].to(self.device))
        self._elapsed = state["elapsed"].to(self.device)

    def reset(self) -> Tensor:
        """New uniformly random layouts in every row; returns o_0, int64 shape (B,) (section 2)."""
        layout = torch.randint(0, self.world.num_layouts, (self.batch_size,), generator=self._generator,
                               device=self.device)
        self._state = self.world.initial_state(layout)
        self._elapsed.zero_()
        return self.world.initial_observation(self._state)

    def step(self, action: Tensor) -> StepOutput:
        """Advances every row by one step; finished games stay absorbed (section 3)."""
        if action.shape != (self.batch_size,) or action.dtype != torch.int64:
            raise ValueError(f"action must be int64 of shape ({self.batch_size},), got {action.dtype} {tuple(action.shape)}.")
        if (action < 0).any() or (action >= self.world.num_actions).any():
            raise ValueError(f"action values must lie in [0, {self.world.num_actions}).")
        if (self._elapsed >= self.max_steps).any():
            raise RuntimeError("step() called after max_steps; call reset() first.")
        self._state, observation, reward = self.world.step_rules(self._state, action)
        self._elapsed += 1
        return StepOutput(observation=observation, reward=reward, truncated=self._elapsed >= self.max_steps)


def game_outcomes(rewards: Tensor) -> dict[str, float]:
    """
    Win, death and unfinished rates of played windows, read from their rewards (section 1).

    Args:
        rewards: r_0..r_{T-1} of shape (B, T); a win pays +999, a death -1001, nothing else exceeds |10|.

    Returns:
        win_rate, death_rate, unfinished_rate (game still running when the window ended) and
        mean_win_step (steps until the winning grab, over won games; NaN if none was won).
    """
    win = rewards > 0.0
    won, died = win.any(dim=1), (rewards < DEATH_REWARD / 2).any(dim=1)
    win_step = win.float().argmax(dim=1) + 1
    return {"win_rate": float(won.double().mean()), "death_rate": float(died.double().mean()),
            "unfinished_rate": float((~won & ~died).double().mean()),
            "mean_win_step": float(win_step[won].double().mean()) if bool(won.any()) else float("nan")}


@dataclass(frozen=True)
class WumpusMarginals:
    """Exact posterior marginals, each (B, 16) float64, and the hunter's exact own state (B,)."""

    pit: Tensor
    wumpus: Tensor
    gold: Tensor
    cell: Tensor
    heading: Tensor
    arrow: Tensor
    wumpus_alive: Tensor
    done: Tensor
    consistent_layouts: Tensor


class WumpusBeliefTracker:
    """Exact Bayes filter by layout enumeration (section 5)."""

    def __init__(self, world: WumpusWorld):
        self.world = world
        self._layouts = torch.arange(world.num_layouts, device=world.device)
        self._wumpus_onehot = torch.nn.functional.one_hot(world.wumpus, NUM_CELLS).double()
        self._gold_onehot = torch.nn.functional.one_hot(world.gold, NUM_CELLS).double()
        self._pit = world.pit.double()

    def start(self, initial_observation: Tensor) -> tuple[Tensor, WumpusState]:
        """
        Conditions the uniform prior on o_0.

        Returns:
            (consistent-layout mask of shape (B, L), the hunter state as a (B,) WumpusState whose
             layout field is meaningless (-1)).
        """
        batch = initial_observation.shape[0]
        hypotheses = self.world.initial_state(self._layouts.expand(batch, -1))
        consistent = self.world.initial_observation(hypotheses) == initial_observation.unsqueeze(1)
        return consistent, self._own_state(hypotheses)

    @staticmethod
    def _own_state(hypotheses: WumpusState) -> WumpusState:
        """The (B,) hunter state; identical across consistent layouts (section 5), read at column 0."""
        return WumpusState(torch.full_like(hypotheses.cell[:, 0], -1), hypotheses.cell[:, 0], hypotheses.heading[:, 0],
                           hypotheses.arrow[:, 0], hypotheses.wumpus_alive[:, 0], hypotheses.done[:, 0])

    def update(self, consistent: Tensor, own: WumpusState, action: Tensor,
               observation: Tensor) -> tuple[Tensor, WumpusState]:
        """One exact filter step with (a_t, o_{t+1}), int64 tensors of shape (B,)."""
        batch, num_layouts = consistent.shape
        expand = lambda t: t.unsqueeze(1).expand(batch, num_layouts)  # noqa: E731
        hypotheses = WumpusState(self._layouts.expand(batch, -1), expand(own.cell), expand(own.heading),
                                 expand(own.arrow), expand(own.wumpus_alive), expand(own.done))
        successors, predicted, _ = self.world.step_rules(hypotheses, action.unsqueeze(1))
        consistent = consistent & (predicted == observation.unsqueeze(1))
        if not bool(consistent.any(dim=1).all()):
            raise RuntimeError("An observation is inconsistent with every layout: simulator and filter disagree.")
        # Every consistent layout predicts the same hunter state; read it from one of them.
        first = consistent.float().argmax(dim=1, keepdim=True)
        pick = lambda t: t.gather(1, first).squeeze(1)  # noqa: E731
        own = WumpusState(own.layout, pick(successors.cell), pick(successors.heading), pick(successors.arrow),
                          pick(successors.wumpus_alive), pick(successors.done))
        return consistent, own

    def marginals(self, consistent: Tensor, own: WumpusState) -> WumpusMarginals:
        """Posterior marginals of the hazards and the gold, plus the exact hunter state."""
        weights = consistent.double()
        weights = weights / weights.sum(dim=1, keepdim=True)
        return WumpusMarginals(weights @ self._pit, weights @ self._wumpus_onehot, weights @ self._gold_onehot,
                               own.cell, own.heading, own.arrow, own.wumpus_alive, own.done, consistent.sum(dim=1))

    def filter_episodes(self, initial_observations: Tensor, actions: Tensor, observations: Tensor) -> list[WumpusMarginals]:
        """Exact marginals after o_0, o_1, ..., o_T for whole episodes ((B,), (B, T), (B, T)); length T + 1."""
        consistent, own = self.start(initial_observations)
        out = [self.marginals(consistent, own)]
        for t in range(actions.shape[1]):
            consistent, own = self.update(consistent, own, actions[:, t], observations[:, t])
            out.append(self.marginals(consistent, own))
        return out


__all__ = ["GAME_OVER", "NUM_CELLS", "Heading", "Percept", "WumpusAction", "WumpusBeliefTracker", "WumpusMarginals",
           "WumpusSimulator", "WumpusState", "WumpusWorld", "game_outcomes"]
