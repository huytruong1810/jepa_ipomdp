# ABSOLUTE PATH: tests/test_experiments.py
"""Phase-7 checks for the experiments layer: config -> run, run-directory reload, seed aggregation."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import torch

import numpy as np

from ipomdp.experiments import (TIGER_GAPS, TIGER_METRICS, WUMPUS_GAME_METRICS, WUMPUS_GAPS, WUMPUS_METRICS, aggregate_games,
                                aggregate_reports, build_run_config, build_training_run, hierarchical_bootstrap,
                                interquartile_mean, load_trained_run, seed_statistic)
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
        assert not trained.run.belief_filter.training
        for name, net in run.networks.items():
            for key, value in net.state_dict().items():
                assert torch.equal(trained.run.networks[name].state_dict()[key], value), f"{name}.{key}"


def _report(optimal: float, planner: float, decoded: float) -> dict:
    report: dict = {}
    for path in TIGER_METRICS.values():
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
        statistics = aggregate_reports([_report(10.0, 9.0, 10.0), _report(20.0, 19.0, 20.0), _report(30.0, 29.0, 30.0)],
                                       "tiger")
        assert set(TIGER_GAPS) <= statistics.keys()
        gap = statistics["gap/learned_planner - optimal"]
        assert gap.mean == pytest.approx(-1.0) and gap.std == pytest.approx(0.0)
        assert statistics["return/learned_planner"].std == pytest.approx(10.0)

    def test_wumpus_gaps_are_paired_against_the_reference(self):
        def report(planner: float) -> dict:
            out: dict = {}
            for path in WUMPUS_METRICS.values():
                node = out
                for key in path[:-1]:
                    node = node.setdefault(key, {})
                node[path[-1]] = 0.0
            out["returns"]["learned_planner"]["discounted_return"] = planner
            out["returns"]["reference_risk0.4"]["discounted_return"] = 573.0
            return out
        statistics = aggregate_reports([report(100.0), report(200.0), report(300.0)], "wumpus")
        assert set(WUMPUS_GAPS) <= statistics.keys()
        assert statistics["gap/learned_planner - reference_risk0.4"].mean == pytest.approx(-373.0)


class TestRobustStatistics:
    """aggregate.py, section 6: IQM over seeds and the hierarchical bootstrap over seeds and games."""

    def test_interquartile_mean_cuts_a_quarter_from_each_end(self):
        assert float(interquartile_mean(np.array([-1000.0, 1.0, 2.0, 3.0, 1000.0]))) == pytest.approx(2.0)
        assert float(interquartile_mean(np.arange(8.0))) == pytest.approx(3.5)        # mean of 2..5

    def test_iqm_interval_contains_the_iqm_and_ignores_an_outlier_seed(self):
        stat = seed_statistic([200.0, 190.0, 210.0, 205.0, -800.0])
        assert stat.iqm == pytest.approx(198.333333, rel=1e-6)
        assert stat.iqm_ci95[0] <= stat.iqm <= stat.iqm_ci95[1]
        # The outlier drags the mean to 1 but not the IQM. At n = 5 a resample can hold it twice, so
        # the IQM interval still reaches down; it must stay narrower than the Student-t interval.
        assert stat.mean == pytest.approx(1.0)
        assert stat.iqm_ci95[1] - stat.iqm_ci95[0] < 2.0 * stat.ci95

    def test_hierarchical_interval_covers_the_true_mean(self):
        # 5 seeds with a seed effect (sd 30) and game noise (sd 400, 512 games): the 95% interval
        # must cover the true mean in about 95% of 40 independent replications.
        rng = np.random.default_rng(0)
        covered = 0
        for _ in range(40):
            seeds = 100.0 + 30.0 * rng.standard_normal((5, 1))
            stat = hierarchical_bootstrap(seeds + 400.0 * rng.standard_normal((5, 512)), chunk=1000)
            covered += stat.ci95[0] <= 100.0 <= stat.ci95[1]
        # Binomial(40, 0.95) has sd 1.4; the percentile bootstrap at 5 seeds under-covers a little.
        assert covered >= 40 * 0.95 - 5 * 1.4 * 1.5

    def test_identical_agents_have_a_zero_paired_gap(self):
        rng = np.random.default_rng(1)
        games = []
        for _ in range(3):
            returns = rng.choice([999.0, -40.0, -1001.0], size=64)
            won, died = returns > 0, returns < -500
            row = {}
            for agent in ("learned_planner", "random", "reference_risk0.4"):
                row.update({f"{agent}.return": returns, f"{agent}.won": won, f"{agent}.died": died})
            games.append(row)
        statistics = aggregate_games(games)
        assert set(WUMPUS_GAME_METRICS) == statistics.keys()
        gap = statistics["gap/learned_planner - reference_risk0.4"]
        assert gap.mean == 0.0 and gap.ci95 == (0.0, 0.0)
        rates = [statistics[f"{k}_rate/learned_planner"].mean for k in ("win", "death", "unfinished")]
        assert sum(rates) == pytest.approx(1.0)

    def test_unequal_game_counts_raise(self):
        row = lambda n: {"learned_planner.return": np.zeros(n)}  # noqa: E731
        with pytest.raises(ValueError):
            aggregate_games([row(4), row(5)])

    def test_exact_planner_rows_are_paired_and_checked(self):
        rng = np.random.default_rng(2)
        heuristic = rng.choice([999.0, -40.0, -1001.0], size=32)
        games = []
        for _ in range(3):
            learned = rng.choice([999.0, -40.0, -1001.0], size=32)
            row = {"reference_risk0.4.return": heuristic, "reference_risk0.4.won": heuristic > 0,
                   "reference_risk0.4.died": heuristic < -500}
            for agent in ("learned_planner", "random"):
                row.update({f"{agent}.return": learned, f"{agent}.won": learned > 0, f"{agent}.died": learned < -500})
            games.append(row)
        exact = {"exact_planner.return": heuristic + 1.0, "exact_planner.won": heuristic > 0,
                 "exact_planner.died": heuristic < -500, "reference_risk0.4.return": heuristic.copy(),
                 "analysis_seed": np.int64(0)}
        statistics = aggregate_games(games, exact)
        gap = statistics["gap/exact_planner - reference_risk0.4"]
        assert gap.mean == pytest.approx(1.0) and gap.ci95 == pytest.approx((1.0, 1.0))
        assert statistics["gap/learned_planner - exact_planner"].mean == pytest.approx(
            np.mean([g["learned_planner.return"] for g in games]) - heuristic.mean() - 1.0)
        exact["reference_risk0.4.return"] = heuristic[::-1].copy()
        with pytest.raises(ValueError):
            aggregate_games(games, exact)
        baseline = [{"recurrent_dqn.return": g["learned_planner.return"] - 2.0, "recurrent_dqn.won": g["learned_planner.won"],
                     "recurrent_dqn.died": g["learned_planner.died"], "reference_risk0.4.return": heuristic}
                    for g in games]
        gap = aggregate_games(games, baseline=baseline)["gap/learned_planner - recurrent_dqn"]
        assert gap.mean == pytest.approx(2.0) and gap.ci95 == pytest.approx((2.0, 2.0))
        with pytest.raises(ValueError):
            aggregate_games(games, baseline=baseline[:2])


class TestSweepCondition:
    """sweep.py, sections 3-4: a sweep directory is one condition; continuing must not raise."""

    @staticmethod
    def _sweep():
        import importlib.util
        spec = importlib.util.spec_from_file_location("sweep", Path(__file__).resolve().parents[1] / "sweep.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_wumpus_condition_survives_the_json_round_trip(self, tmp_path):
        from ipomdp.experiments import WumpusAnalysisSettings
        sweep = self._sweep()
        sweep.record_condition(tmp_path / "s", ["env=wumpus"], WumpusAnalysisSettings())
        sweep.record_condition(tmp_path / "s", ["env=wumpus"], WumpusAnalysisSettings())     # continuing: no error
        with pytest.raises(ValueError):
            sweep.record_condition(tmp_path / "s", ["env=wumpus", "seed_x=1"], WumpusAnalysisSettings())
        assert sweep.domain_of([]) == "tiger" and sweep.domain_of(["env=wumpus"]) == "wumpus"
