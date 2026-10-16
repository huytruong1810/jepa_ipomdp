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
    5. The exact-belief reference agent never dies when it does not gamble, walks to visible
       gold and shoots a possible wumpus to clear a cell (it is the lower bound on V*).
The filter is the ground truth the learned latent is probed against, so an error here would
silently invalidate every latent figure.
"""

import pytest
import torch

from ipomdp.domain.wumpus import (GAME_OVER, Heading, Percept, WumpusAction, WumpusBeliefTracker, WumpusSimulator,
                                  DISCOUNT, WumpusState, WumpusWorld, game_outcomes)

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
    # Cells: c = x + 4 y. Start c = 0 facing North; c = 1 is East of it, c = 4 North of it.

    def test_initial_percepts(self, world):
        breezy = _layout(world, (1, 10), wumpus=15, gold=14)      # pit East of the start
        smelly = _layout(world, (10, 11), wumpus=4, gold=14)      # wumpus North of the start
        state = world.initial_state(torch.tensor([breezy, smelly]))
        assert world.initial_observation(state).tolist() == [Percept.BREEZE, Percept.STENCH]

    def test_bump_turns_and_death_in_pit(self, world):
        layout = _layout(world, (5, 10), wumpus=15, gold=14)      # pit at (1,1), diagonal to the start
        steps = _play(world, layout, [A.TURN_LEFT, A.FORWARD, A.TURN_RIGHT, A.TURN_RIGHT, A.FORWARD, A.TURN_LEFT,
                                      A.FORWARD])
        assert steps[0][2].heading.item() == Heading.WEST and steps[0][1] == -1.0
        assert steps[1][0] == Percept.BUMP and steps[1][2].cell.item() == 0            # wall to the West
        assert steps[3][2].heading.item() == Heading.EAST
        assert steps[4][2].cell.item() == 1 and steps[4][0] == Percept.BREEZE           # (1,0) is next to the pit
        observation, reward, state = steps[6]                                            # North into (1,1): the pit
        assert observation == GAME_OVER and reward == -1001.0 and bool(state.done)

    def test_game_over_is_absorbing(self, world):
        layout = _layout(world, (4, 10), wumpus=15, gold=14)      # pit North of the start
        steps = _play(world, layout, [A.FORWARD, A.FORWARD, A.SHOOT, A.TURN_LEFT])
        assert steps[0][:2] == (GAME_OVER, -1001.0)
        assert [s[:2] for s in steps[1:]] == [(GAME_OVER, 0.0)] * 3
        assert bool(steps[-1][2].arrow)                                                  # nothing changes once over

    def test_arrow_scream_and_safe_dead_wumpus(self, world):
        layout = _layout(world, (10, 11), wumpus=8, gold=12)      # wumpus two cells North, gold behind it
        steps = _play(world, layout, [A.SHOOT, A.SHOOT, A.FORWARD, A.FORWARD, A.FORWARD, A.GRAB])
        assert steps[0][:2] == (Percept.SCREAM, -10.0)            # killed from two cells away
        assert not bool(steps[0][2].arrow) and not bool(steps[0][2].wumpus_alive)
        assert steps[1][:2] == (0, -1.0)                           # no arrow left: a plain -1 action
        assert steps[2][0] == Percept.STENCH                       # (0,1): next to the dead wumpus
        assert steps[3][0] == Percept.STENCH and not bool(steps[3][2].done)   # its own cell still smells, safe
        assert steps[4][0] == Percept.STENCH | Percept.GLITTER     # (0,3): the gold, next to the dead wumpus
        assert steps[5][:2] == (GAME_OVER, 999.0)

    def test_missed_shot_and_live_wumpus_kills(self, world):
        layout = _layout(world, (10, 11), wumpus=1, gold=3)       # wumpus East of the start
        steps = _play(world, layout, [A.SHOOT, A.TURN_RIGHT, A.FORWARD])
        assert steps[0][:2] == (Percept.STENCH, -10.0) and bool(steps[0][2].wumpus_alive)   # shot North: missed
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
        assert start.cell.eq(0).all() and start.heading.eq(Heading.NORTH).all()
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


class TestLearningStack:
    """The shared training stack on Wumpus World: o_0, pruned search, exact resume, probes."""

    @staticmethod
    def _config(**overrides):
        from ipomdp.models import ValueEstimator
        from ipomdp.training import Representation, RunConfig, TrainerConfig
        values = dict(
            seed=3, episode_length=12, env_batch_size=4, warmup_episodes=4, buffer_capacity=32, batch_size=4,
            updates_per_collection=2, eval_episodes=4, latent_dim=16, hidden_dim=32, num_blocks=1,
            representation=Representation.JEPA, observation_horizon=0, ema_momentum=0.99, acting_ema_momentum=0.9, num_bins=255,
            trainer=TrainerConfig(learning_rate=3e-4, weight_decay=1e-4, grad_clip_norm=1.0, value_target_momentum=0.99),
            num_simulations=4, observation_prune=1e-3, c_puct=1.25, dirichlet_alpha=0.3, dirichlet_epsilon=0.25,
            temperature=1.0, temperature_min=0.1, temperature_decay=0.5, epsilon=0.0, epsilon_min=0.0, store_eval_episodes=False,
            time_aware_value=False, value_nstep=0, value_nstep_mix=0.5,
            detached_planning_head=False, privileged_belief_loss=False, value_estimator=ValueEstimator.MAX,
            retrain_every=0, retrain_updates=0)
        values.update(overrides)
        return RunConfig(**values)

    def test_training_stores_o0_and_resumes_bit_for_bit(self, tmp_path):
        from ipomdp.experiments import wumpus_setup
        from ipomdp.training import TrainingRun, load_checkpoint, save_checkpoint
        setup = wumpus_setup(CPU, DISCOUNT)
        reference = TrainingRun(setup, self._config(), CPU)
        expected = [reference.collect_and_train() for _ in range(3)] + [reference.evaluate()]
        assert reference.buffer.sample(2).initial_observations.shape == (2,)
        assert {"eval_win_rate", "eval_death_rate", "collect_win_rate"} <= (expected[-1].keys() | expected[0].keys())
        first = TrainingRun(setup, self._config(), CPU)
        first.collect_and_train()
        save_checkpoint(tmp_path / "latest.pt", first.state_dict())
        resumed = TrainingRun(setup, self._config(seed=99), CPU)
        resumed.load_state_dict(load_checkpoint(tmp_path / "latest.pt"))
        continued = [resumed.collect_and_train() for _ in range(2)] + [resumed.evaluate()]
        assert repr(continued) == repr(expected[1:])      # repr: NaN (no game won yet) equals NaN

    def test_time_aware_value_trains_and_resumes(self, tmp_path):
        # training/run.py, section 9: the agent's state carries the steps left; resume stays bit-exact.
        from ipomdp.experiments import wumpus_setup
        from ipomdp.models import HorizonValueHead
        from ipomdp.training import TrainingRun, load_checkpoint, save_checkpoint
        setup = wumpus_setup(CPU, DISCOUNT)
        reference = TrainingRun(setup, self._config(time_aware_value=True), CPU)
        assert isinstance(reference.value_head, HorizonValueHead)
        expected = [reference.collect_and_train() for _ in range(3)] + [reference.evaluate()]
        assert reference.eval_agent.state.shape == (4, 16 + 1)
        assert bool((reference.eval_agent.state[:, -1] == 0).all())     # the window was played to its end
        first = TrainingRun(setup, self._config(time_aware_value=True), CPU)
        first.collect_and_train()
        save_checkpoint(tmp_path / "latest.pt", first.state_dict())
        resumed = TrainingRun(setup, self._config(seed=99, time_aware_value=True), CPU)
        resumed.load_state_dict(load_checkpoint(tmp_path / "latest.pt"))
        continued = [resumed.collect_and_train() for _ in range(2)] + [resumed.evaluate()]
        assert repr(continued) == repr(expected[1:])

    def test_nstep_greedy_value_trains_and_resumes(self, tmp_path):
        # training/run.py, section 10: greedy games feed the n-step target; resume stays bit-exact.
        from ipomdp.experiments import wumpus_setup
        from ipomdp.training import TrainingRun, load_checkpoint, save_checkpoint
        setup = wumpus_setup(CPU, DISCOUNT)
        config = lambda **kw: self._config(value_nstep=3, store_eval_episodes=True, **kw)  # noqa: E731
        with pytest.raises(ValueError):
            TrainingRun(setup, self._config(value_nstep=3), CPU)
        reference = TrainingRun(setup, config(), CPU)
        expected = [reference.collect_and_train(), reference.evaluate(), reference.collect_and_train(),
                    reference.collect_and_train(), reference.evaluate()]
        assert {"value_nstep_rows", "value_nstep_minus_bellman"} <= expected[2].keys() & expected[3].keys()
        assert bool(reference.buffer._greedy.any()) and not bool(reference.buffer._greedy.all())
        first = TrainingRun(setup, config(), CPU)
        first.collect_and_train()
        first.evaluate()
        save_checkpoint(tmp_path / "latest.pt", first.state_dict())
        resumed = TrainingRun(setup, config(seed=99), CPU)
        resumed.load_state_dict(load_checkpoint(tmp_path / "latest.pt"))
        continued = [resumed.collect_and_train(), resumed.collect_and_train(), resumed.evaluate()]
        assert repr(continued) == repr(expected[2:])

    def test_supervision_targets_match_the_tracker(self):
        # domain/wumpus.py, section 6: start at cell 0 facing North; targets equal the marginals; live until the end.
        from ipomdp.agents import UniformRandomAgent
        from ipomdp.domain.wumpus import START_HEADING
        from ipomdp.training import play_episodes
        world = WumpusWorld(CPU, DISCOUNT)
        episodes, _ = play_episodes(WumpusSimulator(world, 64, 20, 5, CPU), UniformRandomAgent(world.num_actions, 64, 6, CPU))
        tracker = WumpusBeliefTracker(world)
        cell, heading, soft, live = tracker.supervision_targets(episodes.initial_observations, episodes.actions,
                                                                episodes.observations)
        marginals = tracker.filter_episodes(episodes.initial_observations, episodes.actions, episodes.observations)
        assert cell.shape == heading.shape == live.shape == (64, 21) and soft.shape == (64, 21, 49)
        assert bool((cell[:, 0] == 0).all()) and bool((heading[:, 0] == START_HEADING).all()) and bool(live[:, 0].all())
        for t, m in enumerate(marginals):
            assert torch.equal(cell[:, t], m.cell) and torch.equal(live[:, t], ~m.done)
            assert torch.allclose(soft[:, t, :48], torch.cat([m.pit, m.wumpus, m.gold], -1).float())
        assert torch.allclose(soft[:, 0, 48], soft[:, 0, 4] + soft[:, 0, 16 + 4])   # cell 4 is ahead; the wumpus lives
        assert bool((~live).any()) and bool(((soft >= 0) & (soft <= 1)).all())

    def test_privileged_term_trains_and_resumes(self, tmp_path):
        # training/run.py, section 12; trainer.py, section 2e.
        from ipomdp.experiments import wumpus_setup
        from ipomdp.training import TrainingRun, load_checkpoint, save_checkpoint
        setup = wumpus_setup(CPU, DISCOUNT)
        config = lambda **kw: self._config(privileged_belief_loss=True, **kw)  # noqa: E731
        reference = TrainingRun(setup, config(), CPU)
        assert "privileged_readout" in reference.networks
        assert {id(p) for p in reference.privileged.parameters()} <= {id(p) for p in reference.trainer.parameters}
        expected = [reference.collect_and_train() for _ in range(3)] + [reference.evaluate()]
        assert expected[0]["loss_privileged_belief"] > 0
        first = TrainingRun(setup, config(), CPU)
        first.collect_and_train()
        save_checkpoint(tmp_path / "latest.pt", first.state_dict())
        resumed = TrainingRun(setup, config(seed=99), CPU)
        resumed.load_state_dict(load_checkpoint(tmp_path / "latest.pt"))
        continued = [resumed.collect_and_train() for _ in range(2)] + [resumed.evaluate()]
        assert repr(continued) == repr(expected[1:])

    def test_privileged_cache_matches_fresh_targets_after_overwrites(self):
        # privileged.py, section 2c: 10 collections of 4 episodes wrap the 32-slot buffer, so slots cached
        # early are rewritten; every cached target must still equal the exact filter on the stored episode.
        from ipomdp.experiments import wumpus_setup
        from ipomdp.training import TrainingRun
        from ipomdp.training.episode_buffer import EpisodeBatch
        setup = wumpus_setup(CPU, DISCOUNT)
        run = TrainingRun(setup, self._config(privileged_belief_loss=True, updates_per_collection=8), CPU)
        for _ in range(10):
            run.collect_and_train()
        assert int(run.privileged._cached.sum()) > 0
        b = run.buffer
        stored = EpisodeBatch(b._actions, b._observations, b._rewards, b._initial_observations, b._greedy,
                              torch.arange(b.capacity))
        cached, fresh = run.privileged.targets(stored), setup.privileged.targets(stored)
        for got, want in zip(cached.classes, fresh.classes):
            assert torch.equal(got, want)
        assert torch.equal(cached.soft, fresh.soft) and torch.equal(cached.live, fresh.live)
        with pytest.raises(ValueError):
            run.privileged.targets(EpisodeBatch(b._actions, b._observations, b._rewards, b._initial_observations))

    def test_multistep_observation_term_trains_and_resumes(self, tmp_path):
        from ipomdp.experiments import wumpus_setup
        from ipomdp.training import TrainingRun, load_checkpoint, save_checkpoint
        setup = wumpus_setup(CPU, DISCOUNT)
        reference = TrainingRun(setup, self._config(observation_horizon=3), CPU)
        expected = [reference.collect_and_train() for _ in range(2)]
        assert "loss_multistep_observation" in expected[0] and "multistep_observation" in reference.networks
        first = TrainingRun(setup, self._config(observation_horizon=3), CPU)
        first.collect_and_train()
        save_checkpoint(tmp_path / "latest.pt", first.state_dict())
        resumed = TrainingRun(setup, self._config(observation_horizon=3, seed=99), CPU)
        resumed.load_state_dict(load_checkpoint(tmp_path / "latest.pt"))
        assert repr([resumed.collect_and_train()]) == repr(expected[1:])

    def test_multistep_predictor_aligns_targets_and_reaches_the_filter(self):
        from ipomdp.models import BeliefFilter, MultiStepObservationPredictor
        torch.manual_seed(0)
        belief_filter, predictor = BeliefFilter(5, 33, 8, 16, 1), MultiStepObservationPredictor(8, 5, 33, 16, 1, 4)
        actions = torch.nn.functional.one_hot(torch.randint(0, 5, (2, 6)), 5).float()
        observations = torch.nn.functional.one_hot(torch.randint(0, 33, (2, 6)), 33).float()
        latents = belief_filter.unroll(actions, observations, observations[:, 0])
        horizons = predictor(latents[:, :-1], actions)
        assert [h.shape for h in horizons] == [torch.Size([2, 6 - j, 33]) for j in range(4)]
        sum(h.sum() for h in horizons).backward()
        # The GRU cell and the learned z_0 receive gradient (inner residual weights get none at
        # initialisation: the block's output projection is zero-initialised, models/layers.py).
        reached = [p for n, p in belief_filter.named_parameters() if n.startswith("cell.") or n == "initial_latent"]
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in reached)

    def test_agent_starts_from_the_o0_conditioned_latent(self):
        from ipomdp.experiments import wumpus_setup
        from ipomdp.training import TrainingRun
        run = TrainingRun(wumpus_setup(CPU, DISCOUNT), self._config(), CPU)
        model = run.eval_agent.model
        breezy, quiet = torch.tensor([Percept.BREEZE]), torch.tensor([0])
        assert not torch.equal(model.initial_states(1, breezy), model.initial_states(1, quiet))
        expected = run.acting_filter.step(run.acting_filter.initial(1), torch.zeros(1, 5),
                                          torch.nn.functional.one_hot(breezy, 33).float())
        assert torch.allclose(model.initial_states(1, breezy), expected)

    def test_pruned_search_keeps_likely_branches_renormalised(self):
        from ipomdp.experiments import wumpus_setup
        from ipomdp.planning import BeliefTreeSearch
        from ipomdp.training import TrainingRun
        run = TrainingRun(wumpus_setup(CPU, DISCOUNT), self._config(), CPU)
        model = run.eval_agent.model
        search = BeliefTreeSearch(model, 8, 1.25, 0.3, 0.0, seed=0, observation_prune=0.02)
        search.search(model.initial_states(2, torch.tensor([0, 3])), temperature=0.0)
        for root in search.roots:
            for edge in root.edges:
                assert len(edge.children) == len(edge.observations) < 33
                assert edge.observation_probs.sum() == pytest.approx(1.0)
        assert 0.0 < search.statistics.max_pruned_mass <= 33 * 0.02
        with pytest.raises(ValueError):
            BeliefTreeSearch(model, 8, 1.25, 0.3, 0.0, seed=0, observation_prune=1 / 33)

    def test_probe_dataset_and_figures(self):
        import matplotlib.pyplot as plt
        from ipomdp.agents import UniformRandomAgent
        from ipomdp.interpretability import (build_wumpus_dataset, evaluate_wumpus_probe, fit_linear_wumpus_probe,
                                             plot_belief_maps, plot_latent_pca)
        from ipomdp.models import BeliefFilter
        from ipomdp.training import play_episodes
        world = WumpusWorld(CPU)
        episodes, states = play_episodes(WumpusSimulator(world, 16, 10, 0, CPU), UniformRandomAgent(5, 16, 0, CPU))
        data = build_wumpus_dataset(world, BeliefFilter(5, 33, 8, 16, 1), episodes, chunk=5)
        running = ~WumpusState.from_tensor(states).done
        assert data.latents.shape == (int(running.sum()), 8) and data.hazards.shape[1] == 48
        assert torch.allclose(data.hazards[:, :16].sum(1), torch.full((len(data.cell),), 2.0, dtype=torch.float64))
        report = evaluate_wumpus_probe(fit_linear_wumpus_probe(data), data, data)
        assert set(report.hazard_error) == {"pit", "wumpus", "gold"} and 0.0 <= report.cell_accuracy <= 1.0
        probe = fit_linear_wumpus_probe(data)
        plt.close(plot_belief_maps(data, probe, episode=int(data.episode[0]), num_steps=3))
        plt.close(plot_latent_pca(data))


class TestReferenceAgent:
    """The exact-belief reference agent (agents/wumpus_reference.py): safe by construction, and it uses the arrow."""

    @staticmethod
    def _play_layout(world, layout: int, agent, steps: int) -> list[tuple[int, int, float]]:
        state = world.initial_state(torch.tensor([layout]))
        agent.reset(world.initial_observation(state))
        out = []
        for _ in range(steps):
            action = agent.act()
            state, observation, reward = world.step_rules(state, action)
            agent.update(action, observation)
            out.append((int(action), int(observation), float(reward)))
        return out

    def test_never_dies_without_gambling(self, world):
        from ipomdp.agents import WumpusReferenceAgent
        from ipomdp.training import play_episodes
        episodes, _ = play_episodes(WumpusSimulator(world, 128, 30, 5, CPU), WumpusReferenceAgent(world, 128, 0.0))
        outcomes = game_outcomes(episodes.rewards)
        assert outcomes["death_rate"] == 0.0 and outcomes["win_rate"] > 0.3

    def test_walks_to_visible_gold(self, world):
        from ipomdp.agents import WumpusReferenceAgent
        layout = _layout(world, (10, 11), wumpus=15, gold=4)      # quiet start, gold straight ahead (North)
        steps = self._play_layout(world, layout, WumpusReferenceAgent(world, 1, 0.0), 2)
        assert [s[0] for s in steps] == [A.FORWARD, A.GRAB] and steps[1][2] == 999.0

    def test_shoots_a_possible_wumpus_ahead(self, world):
        from ipomdp.agents import WumpusReferenceAgent
        layout = _layout(world, (10, 11), wumpus=4, gold=14)      # stench, no breeze: the wumpus is at c = 1 or 4
        steps = self._play_layout(world, layout, WumpusReferenceAgent(world, 1, 0.0), 2)
        assert steps[0] == (A.SHOOT, Percept.STENCH | Percept.SCREAM, -10.0)
        assert steps[1][0] == A.FORWARD and steps[1][2] == -1.0     # (0,1) is safe now: the wumpus is dead


class TestConfigurableDiscount:
    """env.discount (domain/wumpus.py, section 1; experiments/runs.py, section 2)."""

    @staticmethod
    def _cfg(discount):
        from omegaconf import OmegaConf
        cfg = OmegaConf.load("conf/config.yaml")
        cfg.merge_with(OmegaConf.load("conf/env/wumpus.yaml"))
        cfg.model.num_blocks, cfg.model.latent_dim, cfg.model.hidden_dim = 1, 8, 16
        if discount is None:
            del cfg.env["discount"]
        else:
            cfg.env.discount = discount
        return cfg

    def test_default_is_the_domain_value(self):
        assert self._cfg(DISCOUNT).env.discount == DISCOUNT == 0.99
        assert WumpusWorld(CPU).discount == DISCOUNT

    def test_discount_reaches_domain_codec_and_trainer(self):
        from ipomdp.experiments import build_training_run
        setup, run = build_training_run(self._cfg(0.95), CPU)
        assert setup.domain.discount == run.trainer.discount == 0.95
        assert setup.domain.value_bound == pytest.approx(1001.0 / 0.05)
        assert run.codec.max_abs_value == pytest.approx(1001.0 / 0.05)

    def test_missing_or_invalid_discount_fails_loudly(self):
        from omegaconf.errors import ConfigAttributeError
        from ipomdp.experiments import build_training_run
        with pytest.raises(ConfigAttributeError):
            build_training_run(self._cfg(None), CPU)
        with pytest.raises(ValueError):
            WumpusWorld(CPU, 1.0)
