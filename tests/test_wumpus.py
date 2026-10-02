# ABSOLUTE PATH: tests/test_wumpus.py
"""
Small-scale checks for the Wumpus World domain (src/ipomdp/domain/wumpus.py).

What is verified, and why it matters downstream:
    1. The layout table is the uniform distribution over the user's placement rule.
    2. Hand-built scenarios reproduce every R&N rule: percepts, bumps, death, the arrow, the
       scream, winning and the absorbing end of the game (rewards to the unit).
    3. The simulator is seeded, checkpointable, and returns the start-cell percept from reset().
    4. The exact enumeration filter never loses the true layout, tracks the hunter's own state
       exactly, and its marginals are calibrated (5-standard-error test on random play).
The filter is the ground truth the learned latent is probed against, so an error here would
silently invalidate every latent figure.
"""

import pytest
import torch

from ipomdp.domain.wumpus import (GAME_OVER, Heading, Percept, WumpusAction, WumpusBeliefTracker, WumpusSimulator,
                                  WumpusState, WumpusWorld)

CPU = torch.device("cpu")
A = WumpusAction


@pytest.fixture(scope="module")
def world() -> WumpusWorld:
    return WumpusWorld(CPU)


def _layout(world: WumpusWorld, pits: tuple[int, int], wumpus: int, gold: int) -> int:
    match = (world.pit[:, pits[0]] & world.pit[:, pits[1]] & (world.wumpus == wumpus) & (world.gold == gold))
    (index,) = match.nonzero().flatten().tolist()
    return index


def _play(world: WumpusWorld, layout: int, actions: list[WumpusAction]) -> list[tuple[int, float, WumpusState]]:
    state = world.initial_state(torch.tensor([layout]))
    out = []
    for action in actions:
        state, observation, reward = world.step_rules(state, torch.tensor([int(action)]))
        out.append((int(observation), float(reward), state))
    return out


class TestLayouts:

    def test_uniform_placement_rule(self, world):
        assert world.num_layouts == 105 * 13 * 12
        assert bool((world.pit.sum(1) == 2).all())
        assert not bool(world.pit[:, 0].any()) and bool((world.wumpus != 0).all()) and bool((world.gold != 0).all())
        assert not bool(world.pit.gather(1, world.wumpus.unsqueeze(1)).any())
        assert not bool(world.pit.gather(1, world.gold.unsqueeze(1)).any())
        assert bool((world.wumpus != world.gold).all())
        # Uniform prior marginals: 2/15 pit, 1/15 wumpus and gold on every non-start cell.
        assert torch.allclose(world.pit[:, 1:].double().mean(0), torch.full((15,), 2 / 15, dtype=torch.float64))
        assert torch.allclose(torch.bincount(world.wumpus, minlength=16)[1:].double() / world.num_layouts,
                              torch.full((15,), 1 / 15, dtype=torch.float64))

    def test_names_and_bound(self, world):
        assert world.action_names == ("FORWARD", "TURN_LEFT", "TURN_RIGHT", "GRAB", "SHOOT")
        assert world.num_observations == 33 and world.observation_names[GAME_OVER] == "GAME_OVER"
        assert world.observation_names[Percept.STENCH | Percept.BREEZE] == "STENCH+BREEZE"
        assert world.value_bound == pytest.approx(1001 / 0.01)


class TestRules:
    # Cells: c = x + 4 y. Start c = 0 facing East; c = 1 is East of it, c = 4 North of it.

    def test_initial_percepts(self, world):
        breezy = _layout(world, (1, 10), wumpus=15, gold=14)      # pit East of the start
        smelly = _layout(world, (10, 11), wumpus=4, gold=14)      # wumpus North of the start
        state = world.initial_state(torch.tensor([breezy, smelly]))
        assert world.initial_observation(state).tolist() == [Percept.BREEZE, Percept.STENCH]

    def test_bump_turns_and_death_in_pit(self, world):
        layout = _layout(world, (5, 10), wumpus=15, gold=14)      # pit at (1,1), diagonal to the start
        steps = _play(world, layout, [A.TURN_RIGHT, A.FORWARD, A.TURN_LEFT, A.FORWARD, A.TURN_LEFT, A.FORWARD])
        assert steps[0][2].heading.item() == Heading.SOUTH and steps[0][1] == -1.0
        assert steps[1][0] == Percept.BUMP and steps[1][2].cell.item() == 0            # wall to the South
        assert steps[3][2].cell.item() == 1 and steps[3][0] == Percept.BREEZE           # (1,0) is next to the pit
        observation, reward, state = steps[5]                                            # North into (1,1): the pit
        assert observation == GAME_OVER and reward == -1001.0 and bool(state.done)

    def test_game_over_is_absorbing(self, world):
        layout = _layout(world, (1, 10), wumpus=15, gold=14)
        steps = _play(world, layout, [A.FORWARD, A.FORWARD, A.SHOOT, A.TURN_LEFT])
        assert steps[0][:2] == (GAME_OVER, -1001.0)
        assert [s[:2] for s in steps[1:]] == [(GAME_OVER, 0.0)] * 3
        assert bool(steps[-1][2].arrow)                                                  # nothing changes once over

    def test_arrow_scream_and_safe_dead_wumpus(self, world):
        layout = _layout(world, (10, 11), wumpus=2, gold=3)       # wumpus two cells East, gold behind it
        steps = _play(world, layout, [A.SHOOT, A.SHOOT, A.FORWARD, A.FORWARD, A.FORWARD, A.GRAB])
        assert steps[0][:2] == (Percept.SCREAM, -10.0)            # killed from two cells away
        assert not bool(steps[0][2].arrow) and not bool(steps[0][2].wumpus_alive)
        assert steps[1][:2] == (0, -1.0)                           # no arrow left: a plain -1 action
        assert steps[2][0] == Percept.STENCH                       # (1,0): next to the dead wumpus
        assert steps[3][0] == Percept.STENCH and not bool(steps[3][2].done)   # its own cell still smells, safe
        assert steps[4][0] == Percept.STENCH | Percept.GLITTER     # (3,0): the gold, next to the dead wumpus
        assert steps[5][:2] == (GAME_OVER, 999.0)

    def test_missed_shot_and_live_wumpus_kills(self, world):
        layout = _layout(world, (10, 11), wumpus=4, gold=3)       # wumpus North of the start
        steps = _play(world, layout, [A.SHOOT, A.TURN_LEFT, A.FORWARD])
        assert steps[0][:2] == (Percept.STENCH, -10.0) and bool(steps[0][2].wumpus_alive)   # shot East: missed
        assert steps[2][:2] == (GAME_OVER, -1001.0)

    def test_grab_off_the_gold_does_nothing(self, world):
        layout = _layout(world, (10, 11), wumpus=15, gold=3)
        steps = _play(world, layout, [A.GRAB])
        assert steps[0][:2] == (0, -1.0) and not bool(steps[0][2].done)


class TestSimulator:

    def test_reset_returns_start_percepts_and_is_seeded(self, world):
        def run(seed):
            sim = WumpusSimulator(world, 64, 20, seed, CPU)
            first = sim.reset()
            start = WumpusState.from_tensor(sim.state)
            generator = torch.Generator().manual_seed(0)
            outs = [sim.step(torch.randint(0, 5, (64,), generator=generator)) for _ in range(20)]
            return first, start, torch.stack([o.observation for o in outs]), outs[-1].truncated
        first, start, observations, truncated = run(3)
        assert torch.equal(first, world.initial_observation(start)) and bool(truncated.all())
        assert start.cell.eq(0).all() and start.heading.eq(Heading.EAST).all()
        again = run(3)
        assert torch.equal(first, again[0]) and torch.equal(observations, again[2])
        assert not torch.equal(start.layout, run(4)[1].layout)

    def test_state_dict_roundtrip(self, world):
        sim = WumpusSimulator(world, 8, 10, 0, CPU)
        sim.reset()
        sim.step(torch.zeros(8, dtype=torch.int64))
        saved = sim.state_dict()
        expected = [sim.step(torch.full((8,), a, dtype=torch.int64)).observation for a in (1, 0, 4)]
        restored = WumpusSimulator(world, 8, 10, 99, CPU)
        restored.load_state_dict(saved)
        assert all(torch.equal(restored.step(torch.full((8,), a, dtype=torch.int64)).observation, e)
                   for a, e in zip((1, 0, 4), expected))

    def test_step_after_window_raises(self, world):
        sim = WumpusSimulator(world, 2, 1, 0, CPU)
        sim.reset()
        sim.step(torch.zeros(2, dtype=torch.int64))
        with pytest.raises(RuntimeError):
            sim.step(torch.zeros(2, dtype=torch.int64))


def _random_play(world: WumpusWorld, batch: int, steps: int, seed: int):
    sim = WumpusSimulator(world, batch, steps, seed, CPU)
    generator = torch.Generator().manual_seed(seed)
    first = sim.reset()
    actions, observations, states = [], [], [WumpusState.from_tensor(sim.state)]
    for _ in range(steps):
        action = torch.randint(0, 5, (batch,), generator=generator)
        observations.append(sim.step(action).observation)
        actions.append(action)
        states.append(WumpusState.from_tensor(sim.state))
    return first, torch.stack(actions, 1), torch.stack(observations, 1), states


class TestExactFilter:

    def test_true_layout_stays_consistent_and_own_state_is_exact(self, world):
        first, actions, observations, states = _random_play(world, 32, 40, seed=1)
        tracker = WumpusBeliefTracker(world)
        consistent, own = tracker.start(first)
        rows = torch.arange(32)
        for t in range(actions.shape[1] + 1):
            if t > 0:
                consistent, own = tracker.update(consistent, own, actions[:, t - 1], observations[:, t - 1])
            truth = states[t]
            assert bool(consistent[rows, truth.layout].all())
            for field in ("cell", "heading", "arrow", "wumpus_alive", "done"):
                assert torch.equal(getattr(own, field), getattr(truth, field)), (t, field)
            marginals = tracker.marginals(consistent, own)
            assert torch.allclose(marginals.pit.sum(1), torch.full((32,), 2.0, dtype=torch.float64))
            assert torch.allclose(marginals.gold.sum(1), torch.ones(32, dtype=torch.float64))

    def test_marginals_are_calibrated(self, world):
        # E[1(hazard at c) - P(hazard at c | h_t)] = 0 for every cell and time (Bayes calibration).
        first, actions, observations, states = _random_play(world, 400, 12, seed=2)
        marginals = WumpusBeliefTracker(world).filter_episodes(first, actions, observations)[-1]
        layout = states[-1].layout
        for truth, belief in ((world.pit[layout].double(), marginals.pit),
                              (torch.nn.functional.one_hot(world.wumpus[layout], 16).double(), marginals.wumpus),
                              (torch.nn.functional.one_hot(world.gold[layout], 16).double(), marginals.gold)):
            error = truth - belief
            stderr = error.std(0) / error.shape[0] ** 0.5
            assert bool((error.mean(0).abs() <= 5 * stderr + 1e-12).all())
        assert bool((marginals.consistent_layouts < world.num_layouts).any())    # the percepts were informative
