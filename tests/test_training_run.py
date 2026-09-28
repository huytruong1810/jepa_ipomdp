# ABSOLUTE PATH: tests/test_training_run.py
"""Phase-5 checks for the training run: warm-up/planning switch, evaluation, checkpoints, exact resume."""

from pathlib import Path

import pytest
import torch

from ipomdp.domain import build_tiger_pomdp
from ipomdp.training import RunConfig, TrainerConfig, TrainingRun, load_checkpoint, save_checkpoint

CPU = torch.device("cpu")


def _config(**overrides) -> RunConfig:
    values = dict(
        seed=7, episode_length=8, env_batch_size=4, warmup_episodes=4, buffer_capacity=64, batch_size=4,
        updates_per_collection=2, eval_episodes=4, latent_dim=8, hidden_dim=16, num_blocks=1, ema_momentum=0.99,
        num_bins=255, trainer=TrainerConfig(learning_rate=3e-4, weight_decay=1e-4, grad_clip_norm=1.0,
                                            lambda_return=0.95),
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
