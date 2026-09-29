# ABSOLUTE PATH: tests/test_experiments.py
"""Phase-7 checks for the experiments layer: config -> run, run-directory reload, seed aggregation."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import torch

from ipomdp.experiments import (PAIRED_GAPS, REPORT_METRICS, aggregate_reports, build_run_config, build_training_run,
                                load_trained_run, seed_statistic)
from ipomdp.training import save_checkpoint

CPU = torch.device("cpu")
CONF_DIR = str(Path(__file__).resolve().parents[1] / "conf")
TINY = ["training.env_batch_size=4", "training.warmup_episodes=4", "training.batch_size=4",
        "training.updates_per_collection=2", "training.eval_episodes=4", "env.max_steps=8", "model.latent_dim=8",
        "model.hidden_dim=16", "mcts.num_simulations=4"]


def _compose(overrides: list[str]):
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        return compose(config_name="config", overrides=overrides)


class TestRunConstruction:

    def test_default_config_maps_onto_run_config(self):
        cfg = _compose([])
        run_config = build_run_config(cfg)
        assert run_config.seed == cfg.seed and run_config.episode_length == cfg.env.max_steps
        assert run_config.updates_per_collection == cfg.training.updates_per_collection
        assert run_config.num_simulations == cfg.mcts.num_simulations
        assert run_config.trainer.value_target_momentum == cfg.trainer.value_target_momentum

    def test_run_directory_reloads_best_networks(self, tmp_path: Path):
        cfg = _compose(TINY)
        _, run = build_training_run(cfg, CPU)
        run.collect_and_train()
        (tmp_path / ".hydra").mkdir()
        OmegaConf.save(cfg, tmp_path / ".hydra" / "config.yaml")
        save_checkpoint(tmp_path / "checkpoints" / "best.pt", {
            "collection": 1, "eval_return": 3.5,
            "networks": {name: net.state_dict() for name, net in run.networks.items()}})

        trained = load_trained_run(tmp_path, CPU)
        assert trained.checkpoint_collection == 1 and trained.checkpoint_eval_return == 3.5
        assert not trained.run.world_model.training
        for name, net in run.networks.items():
            for key, value in net.state_dict().items():
                assert torch.equal(trained.run.networks[name].state_dict()[key], value), f"{name}.{key}"


def _report(optimal: float, planner: float, decoded: float) -> dict:
    report: dict = {}
    for path in REPORT_METRICS.values():
        node = report
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = 0.0
    report["returns"]["optimal"]["mean"] = optimal
    report["returns"]["learned_planner"]["mean"] = planner
    report["returns"]["decoded_mlp"]["mean"] = decoded
    return report


class TestSeedAggregation:

    def test_student_t_interval(self):
        stat = seed_statistic([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stat.n == 5 and stat.mean == 3.0 and stat.std == pytest.approx(2.5 ** 0.5)
        assert stat.ci95 == pytest.approx(2.7764451 * 2.5 ** 0.5 / 5 ** 0.5, rel=1e-6)   # t_{0.975,4}

    def test_single_seed_has_no_interval(self):
        with pytest.raises(ValueError):
            seed_statistic([1.0])

    def test_gaps_are_paired_within_seeds(self):
        # The returns vary a lot between seeds but the planner is always exactly 1 below the optimum.
        statistics = aggregate_reports([_report(10.0, 9.0, 10.0), _report(20.0, 19.0, 20.0), _report(30.0, 29.0, 30.0)])
        assert set(PAIRED_GAPS) <= statistics.keys()
        gap = statistics["gap/learned_planner - optimal"]
        assert gap.mean == pytest.approx(-1.0) and gap.std == pytest.approx(0.0)
        assert statistics["return/learned_planner"].std == pytest.approx(10.0)
