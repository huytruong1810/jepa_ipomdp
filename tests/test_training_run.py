# ABSOLUTE PATH: tests/test_training_run.py
"""Phase-5 checks for the training run: warm-up/planning switch, evaluation, checkpoints, exact resume."""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from ipomdp.experiments import tiger_setup
from ipomdp.models import ValueEstimator, ValueHead, ValuePair, ema_update
from ipomdp.training import (Representation, RunConfig, RunStream, TrainerConfig, TrainingRun, load_checkpoint, save_checkpoint,
                             stream_seed)

CPU = torch.device("cpu")


def _config(**overrides) -> RunConfig:
    values = dict(
        seed=7, episode_length=8, env_batch_size=4, warmup_episodes=4, buffer_capacity=64, batch_size=4,
        updates_per_collection=2, eval_episodes=4, latent_dim=8, hidden_dim=16, num_blocks=1,
        representation=Representation.JEPA, observation_horizon=0, ema_momentum=0.99,
        acting_ema_momentum=0.9,
        num_bins=255, trainer=TrainerConfig(learning_rate=3e-4, weight_decay=1e-4, grad_clip_norm=1.0,
                                            value_target_momentum=0.99),
        num_simulations=4, observation_prune=0.0, c_puct=1.25, dirichlet_alpha=0.3, dirichlet_epsilon=0.25, temperature=1.0,
        temperature_min=0.1, temperature_decay=0.5, epsilon=0.0, epsilon_min=0.0, store_eval_episodes=False,
        time_aware_value=False, value_nstep=0, value_nstep_mix=0.5,
        detached_planning_head=False, privileged_belief_loss=False, value_estimator=ValueEstimator.MAX)
    values.update(overrides)
    return RunConfig(**values)


def _parameters(run: TrainingRun) -> list[torch.Tensor]:
    return [p.detach().clone() for net in run.networks.values() for p in net.state_dict().values()]


class TestTrainingRun:

    def test_warmup_then_planning_and_temperature_annealing(self):
        run = TrainingRun(tiger_setup(CPU), _config(), CPU)
        first = run.collect_and_train()
        assert first["warmup"] == 1.0 and run.train_agent.temperature == 1.0
        second = run.collect_and_train()
        assert second["warmup"] == 0.0 and run.train_agent.temperature == 0.5
        assert run.collection == 2 and run.buffer.size == 8

    def test_greedy_evaluation_games_are_stored_only_when_enabled(self):
        for store in (False, True):
            run = TrainingRun(tiger_setup(CPU), _config(store_eval_episodes=store), CPU)
            run.collect_and_train()
            before = run.buffer.size
            run.evaluate()
            assert run.buffer.size == before + (run.cfg.eval_episodes if store else 0)

    def test_epsilon_decays_with_the_temperature_to_its_floor(self):
        run = TrainingRun(tiger_setup(CPU), _config(epsilon=0.4, epsilon_min=0.15), CPU)
        assert run.collect_and_train()["epsilon"] == 0.4          # warm-up: no decay
        assert run.collect_and_train()["epsilon"] == 0.2
        assert run.collect_and_train()["epsilon"] == 0.15
        assert run.eval_agent.epsilon == 0.0

    def test_evaluation_reports_discounted_return(self):
        run = TrainingRun(tiger_setup(CPU), _config(), CPU)
        run.collect_and_train()
        evaluation = run.evaluate()
        assert {"eval_discounted_return", "eval_discounted_return_stderr"} <= evaluation.keys()
        assert run.best_eval_return == evaluation["eval_discounted_return"]
        fractions = [v for k, v in evaluation.items() if k.startswith("eval_action_fraction/")]
        assert sum(fractions) == pytest.approx(1.0)

    @pytest.mark.parametrize("representation", list(Representation))
    def test_resume_continues_bit_for_bit(self, tmp_path: Path, representation: Representation):
        setup = tiger_setup(CPU)
        uninterrupted = TrainingRun(setup, _config(representation=representation), CPU)
        reference = [uninterrupted.collect_and_train() for _ in range(4)]
        reference_eval = uninterrupted.evaluate()

        first_half = TrainingRun(setup, _config(representation=representation), CPU)
        for _ in range(2):
            first_half.collect_and_train()
        save_checkpoint(tmp_path / "latest.pt", first_half.state_dict())

        resumed = TrainingRun(setup, _config(seed=123, representation=representation), CPU)  # different init, fully overwritten by the checkpoint
        resumed.load_state_dict(load_checkpoint(tmp_path / "latest.pt"))
        continued = [resumed.collect_and_train() for _ in range(2)]
        assert continued == reference[2:]
        assert resumed.evaluate() == reference_eval
        assert all(torch.equal(a, b) for a, b in zip(_parameters(resumed), _parameters(uninterrupted)))


    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_resume_on_cuda(self, tmp_path: Path):
        # Generator states are CPU tensors; a CUDA run must still restore them (regression).
        cuda = torch.device("cuda")
        run = TrainingRun(tiger_setup(cuda), _config(), cuda)
        run.collect_and_train()
        save_checkpoint(tmp_path / "latest.pt", run.state_dict())
        resumed = TrainingRun(tiger_setup(cuda), _config(), cuda)
        resumed.load_state_dict(load_checkpoint(tmp_path / "latest.pt"))
        assert resumed.collection == 1
        resumed.collect_and_train()


class TestRepresentation:
    """training/run.py, section 5: the decoder baseline builds no JEPA parts; all else is shared."""

    def test_decoder_run_has_no_jepa_parts(self):
        run = TrainingRun(tiger_setup(CPU), _config(representation=Representation.DECODER), CPU)
        assert run.jepa is None and "world_model" not in run.networks and "belief_filter" in run.networks
        assert run.trainer.belief_filter is run.belief_filter
        assert "loss_prediction" not in run.collect_and_train()

    def test_same_seed_gives_the_same_initial_filter(self):
        jepa = TrainingRun(tiger_setup(CPU), _config(), CPU)
        decoder = TrainingRun(tiger_setup(CPU), _config(representation=Representation.DECODER), CPU)
        assert jepa.jepa.belief_filter is jepa.belief_filter
        assert all(torch.equal(p, q) for p, q in zip(jepa.belief_filter.parameters(), decoder.belief_filter.parameters()))


class TestDetachedPlanningHead:
    """training/run.py, section 11; trainer.py, section 3d."""

    @staticmethod
    def _run(seed: int = 7, **overrides) -> TrainingRun:
        return TrainingRun(tiger_setup(CPU), _config(seed=seed, representation=Representation.DECODER, **overrides), CPU)

    def test_planners_and_value_targets_read_the_planning_head(self):
        run = self._run(detached_planning_head=True)
        assert run.eval_agent.model.observation_head is run.acting_planning_observation_head
        assert run.train_agent.model.observation_head is run.acting_planning_observation_head
        assert run.trainer.target_model.observation_head is run.planning_observation_head
        assert {"planning_observation", "acting_planning_observation"} <= run.networks.keys()
        assert "loss_planning_observation" in run.collect_and_train()

    def test_off_by_default_and_jepa_raises(self):
        run = self._run()
        assert run.planning_observation_head is None and run.eval_agent.model.observation_head is run.acting_observation_head
        with pytest.raises(ValueError):
            TrainingRun(tiger_setup(CPU), _config(detached_planning_head=True), CPU)

    def test_planning_loss_does_not_reach_the_filter(self):
        # Only the planning term trains here (the other heads' losses are scaled to zero), so any
        # filter gradient would have to come through it.
        run = self._run(detached_planning_head=True)
        run.collect_and_train()
        trainer, batch = run.trainer, run.buffer.sample(4)
        before = [p.detach().clone() for p in run.planning_observation_head.parameters()]
        actions = F.one_hot(batch.actions, trainer.num_actions).float()
        latents = trainer.belief_filter.unroll(actions, F.one_hot(batch.observations, trainer.num_observations).float(), None)
        current = latents[:, :-1].reshape(-1, latents.shape[-1])
        loss = F.cross_entropy(trainer.planning_observation_head(current.detach(), actions.reshape(-1, trainer.num_actions)),
                               batch.observations.reshape(-1))
        trainer.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert all(p.grad is None for p in trainer.belief_filter.parameters())
        assert all(p.grad is not None for p in run.planning_observation_head.parameters())
        assert all(torch.equal(p, q) for p, q in zip(before, run.planning_observation_head.parameters()))

    def test_resume_continues_bit_for_bit(self, tmp_path: Path):
        reference = self._run(detached_planning_head=True)
        expected = [reference.collect_and_train() for _ in range(3)] + [reference.evaluate()]
        first = self._run(detached_planning_head=True)
        first.collect_and_train()
        save_checkpoint(tmp_path / "latest.pt", first.state_dict())
        resumed = self._run(seed=123, detached_planning_head=True)
        resumed.load_state_dict(load_checkpoint(tmp_path / "latest.pt"))
        assert [resumed.collect_and_train() for _ in range(2)] + [resumed.evaluate()] == expected[1:]
        assert all(torch.equal(a, b) for a, b in zip(_parameters(resumed), _parameters(reference)))


class TestValuePair:
    """training/run.py, section 13; trainer.py, section 3e; models/heads.py, section 2c."""

    @staticmethod
    def _run(estimator: ValueEstimator, seed: int = 7) -> TrainingRun:
        return TrainingRun(tiger_setup(CPU), _config(seed=seed, value_estimator=estimator), CPU)

    @pytest.mark.parametrize("estimator", [ValueEstimator.DOUBLE, ValueEstimator.CLIPPED_DOUBLE])
    def test_planners_targets_and_acting_copies_read_the_pair(self, estimator: ValueEstimator):
        run = self._run(estimator)
        for pair in (run.value_head, run.trainer.target_value_head, run.acting_value_head):
            assert isinstance(pair, ValuePair) and pair.estimator is estimator
        assert run.eval_agent.model.value_head is run.acting_value_head is run.train_agent.model.value_head
        for _ in range(3):
            run.collect_and_train()
        assert not all(torch.equal(p, q) for p, q in zip(run.value_head.a.parameters(), run.value_head.b.parameters()))
        z = torch.randn(5, 8)
        va, vb = run.acting_value_head.head_values(z, run.codec)
        expected = 0.5 * (va + vb) if estimator is ValueEstimator.DOUBLE else torch.minimum(va, vb)
        assert torch.equal(run.eval_agent.model.values(z), expected)

    @pytest.mark.parametrize("estimator", [ValueEstimator.DOUBLE, ValueEstimator.CLIPPED_DOUBLE])
    def test_targets_follow_their_definitions(self, estimator: ValueEstimator):
        run = self._run(estimator)
        for _ in range(3):
            run.collect_and_train()
        trainer, z = run.trainer, torch.randn(6, 8)
        target_a, target_b = trainer.pair_value_targets(z)
        e = trainer.target_model.expand(z)
        heads = trainer.target_value_head
        children = e.next_states.flatten(0, 2)
        va, vb = (run.codec.mean(h(children)).view_as(e.next_values) for h in (heads.a, heads.b))
        qa = e.rewards + trainer.discount * (e.observation_probs * va).sum(-1)
        qb = e.rewards + trainer.discount * (e.observation_probs * vb).sum(-1)
        if estimator is ValueEstimator.DOUBLE:
            rows = torch.arange(6)
            assert torch.allclose(target_a, qb[rows, qa.argmax(-1)]) and torch.allclose(target_b, qa[rows, qb.argmax(-1)])
        else:
            clipped = e.rewards + trainer.discount * (e.observation_probs * torch.minimum(va, vb)).sum(-1)
            assert torch.allclose(target_a, clipped.max(-1).values) and torch.equal(target_a, target_b)

    def test_each_episode_trains_one_head(self):
        # A one-episode batch is episode 0, so only head A may receive a value gradient.
        run = self._run(ValueEstimator.DOUBLE)
        run.collect_and_train()
        run.trainer.train_step(run.buffer.sample(1))
        assert all(p.grad is not None and not p.grad.any() for p in run.value_head.b.parameters())
        assert any(p.grad.any() for p in run.value_head.a.parameters())

    def test_raises_with_time_aware_or_nstep_and_off_builds_nothing(self):
        for options in (dict(time_aware_value=True), dict(value_nstep=4, store_eval_episodes=True)):
            with pytest.raises(ValueError):
                TrainingRun(tiger_setup(CPU), _config(value_estimator=ValueEstimator.DOUBLE, **options), CPU)
        assert isinstance(TrainingRun(tiger_setup(CPU), _config(), CPU).value_head, ValueHead)

    def test_resume_continues_bit_for_bit(self, tmp_path: Path):
        reference = self._run(ValueEstimator.CLIPPED_DOUBLE)
        expected = [reference.collect_and_train() for _ in range(3)] + [reference.evaluate()]
        first = self._run(ValueEstimator.CLIPPED_DOUBLE)
        first.collect_and_train()
        save_checkpoint(tmp_path / "latest.pt", first.state_dict())
        resumed = self._run(ValueEstimator.CLIPPED_DOUBLE, seed=123)
        resumed.load_state_dict(load_checkpoint(tmp_path / "latest.pt"))
        assert [resumed.collect_and_train() for _ in range(2)] + [resumed.evaluate()] == expected[1:]
        assert all(torch.equal(a, b) for a, b in zip(_parameters(resumed), _parameters(reference)))


class TestPrivilegedBeliefTerm:
    """training/run.py, section 12: Tiger has no exact-belief targets, so the option raises; off builds nothing."""

    def test_tiger_raises_and_off_builds_nothing(self):
        with pytest.raises(ValueError):
            TrainingRun(tiger_setup(CPU), _config(privileged_belief_loss=True), CPU)
        run = TrainingRun(tiger_setup(CPU), _config(), CPU)
        assert run.privileged is None and "privileged_readout" not in run.networks


class TestCheckpointFiles:

    def test_roundtrip_and_atomic_write(self, tmp_path: Path):
        path = tmp_path / "nested" / "state.pt"
        save_checkpoint(path, {"x": torch.arange(3)})
        assert torch.equal(load_checkpoint(path)["x"], torch.arange(3))
        assert not path.with_suffix(".pt.tmp").exists()

    def test_missing_checkpoint_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "absent.pt")


class TestSeedStreams:
    """Phase-8 finding: base + offset seeds made neighbouring runs of a sweep share streams."""

    def test_stream_seeds_are_distinct_across_bases_and_streams(self):
        seeds = [stream_seed(base, stream) for base in range(200) for stream in RunStream]
        assert len(set(seeds)) == len(seeds)
        assert all(0 <= s < 2 ** 63 for s in seeds)

    def test_neighbouring_runs_do_not_share_simulator_streams(self):
        # Under base + offset seeding, run 1's collection env replayed run 0's evaluation env.
        run_0 = TrainingRun(tiger_setup(CPU), _config(seed=0), CPU)
        run_1 = TrainingRun(tiger_setup(CPU), _config(seed=1), CPU)
        generators = [run_0.env.state_dict()["generator"], run_0.eval_env.state_dict()["generator"],
                      run_1.env.state_dict()["generator"], run_1.eval_env.state_dict()["generator"]]
        for i, first in enumerate(generators):
            for second in generators[i + 1:]:
                assert not torch.equal(first, second)


class TestActingWeights:
    """Phase-8 decision: the planner acts with Polyak-averaged copies of the filter and heads."""

    def test_ema_update_is_exact(self):
        target, online = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
        expected = [0.9 * t.detach() + 0.1 * o.detach() for t, o in zip(target.parameters(), online.parameters())]
        ema_update(target, online, 0.9)
        assert all(torch.allclose(t, e) for t, e in zip(target.parameters(), expected))

    def test_agents_plan_with_the_acting_copies(self):
        run = TrainingRun(tiger_setup(CPU), _config(), CPU)
        for agent in (run.train_agent, run.eval_agent):
            model = agent.model
            assert model.belief_filter is run.acting_filter and model.value_head is run.acting_value_head
            assert model.reward_head is run.acting_reward_head
            assert model.observation_head is run.acting_observation_head
        initial = [p.clone() for p in run.acting_value_head.parameters()]
        run.collect_and_train()
        acting = list(run.acting_value_head.parameters())
        online = list(run.value_head.parameters())
        assert not all(torch.equal(a, i) for a, i in zip(acting, initial))      # it tracks training ...
        assert not all(torch.equal(a, o) for a, o in zip(acting, online))       # ... but lags the online weights
        assert all(not p.requires_grad for p in acting)
