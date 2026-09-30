# ABSOLUTE PATH: tests/test_training_run.py
"""Phase-5 checks for the training run: warm-up/planning switch, evaluation, checkpoints, exact resume."""

from pathlib import Path

import pytest
import torch

from ipomdp.domain import build_tiger_pomdp
from ipomdp.models import ema_update
from ipomdp.training import (RunConfig, RunStream, TrainerConfig, TrainingRun, load_checkpoint, save_checkpoint,
                             stream_seed)

CPU = torch.device("cpu")


def _config(**overrides) -> RunConfig:
    values = dict(
        seed=7, episode_length=8, env_batch_size=4, warmup_episodes=4, buffer_capacity=64, batch_size=4,
        updates_per_collection=2, eval_episodes=4, latent_dim=8, hidden_dim=16, num_blocks=1, ema_momentum=0.99,
        acting_ema_momentum=0.9,
        num_bins=255, trainer=TrainerConfig(learning_rate=3e-4, weight_decay=1e-4, grad_clip_norm=1.0,
                                            value_target_momentum=0.99),
        num_simulations=4, c_puct=1.25, dirichlet_alpha=0.3, dirichlet_epsilon=0.25, temperature=1.0,
        temperature_min=0.1, temperature_decay=0.5)
    values.update(overrides)
    return RunConfig(**values)


def _parameters(run: TrainingRun) -> list[torch.Tensor]:
    return [p.detach().clone() for net in run.networks.values() for p in net.state_dict().values()]


class TestTrainingRun:

    def test_warmup_then_planning_and_temperature_annealing(self):
        run = TrainingRun(build_tiger_pomdp(), _config(), CPU)
        first = run.collect_and_train()
        assert first["warmup"] == 1.0 and run.train_agent.temperature == 1.0
        second = run.collect_and_train()
        assert second["warmup"] == 0.0 and run.train_agent.temperature == 0.5
        assert run.collection == 2 and run.buffer.size == 8

    def test_evaluation_reports_discounted_return(self):
        run = TrainingRun(build_tiger_pomdp(), _config(), CPU)
        run.collect_and_train()
        evaluation = run.evaluate()
        assert {"eval_discounted_return", "eval_discounted_return_stderr"} <= evaluation.keys()
        assert run.best_eval_return == evaluation["eval_discounted_return"]
        fractions = [v for k, v in evaluation.items() if k.startswith("eval_action_fraction/")]
        assert sum(fractions) == pytest.approx(1.0)

    def test_resume_continues_bit_for_bit(self, tmp_path: Path):
        pomdp = build_tiger_pomdp()
        uninterrupted = TrainingRun(pomdp, _config(), CPU)
        reference = [uninterrupted.collect_and_train() for _ in range(4)]
        reference_eval = uninterrupted.evaluate()

        first_half = TrainingRun(pomdp, _config(), CPU)
        for _ in range(2):
            first_half.collect_and_train()
        save_checkpoint(tmp_path / "latest.pt", first_half.state_dict())

        resumed = TrainingRun(pomdp, _config(seed=123), CPU)  # different init, fully overwritten by the checkpoint
        resumed.load_state_dict(load_checkpoint(tmp_path / "latest.pt"))
        continued = [resumed.collect_and_train() for _ in range(2)]
        assert continued == reference[2:]
        assert resumed.evaluate() == reference_eval
        assert all(torch.equal(a, b) for a, b in zip(_parameters(resumed), _parameters(uninterrupted)))


    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_resume_on_cuda(self, tmp_path: Path):
        # Generator states are CPU tensors; a CUDA run must still restore them (regression).
        cuda = torch.device("cuda")
        run = TrainingRun(build_tiger_pomdp(), _config(), cuda)
        run.collect_and_train()
        save_checkpoint(tmp_path / "latest.pt", run.state_dict())
        resumed = TrainingRun(build_tiger_pomdp(), _config(), cuda)
        resumed.load_state_dict(load_checkpoint(tmp_path / "latest.pt"))
        assert resumed.collection == 1
        resumed.collect_and_train()


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
        run_0 = TrainingRun(build_tiger_pomdp(), _config(seed=0), CPU)
        run_1 = TrainingRun(build_tiger_pomdp(), _config(seed=1), CPU)
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
        run = TrainingRun(build_tiger_pomdp(), _config(), CPU)
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
