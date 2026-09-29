# ABSOLUTE PATH: analyze.py
# ==============================================================================
# BELIEF ANALYSIS OF A TRAINED RUN
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Input Is a Run Directory:
#    - `uv run analyze.py runs/tiger/<run>` rebuilds the run from its resolved Hydra config
#      (<run>/.hydra/config.yaml), loads the networks of <run>/checkpoints/best.pt (the best
#      greedy evaluation), and runs ipomdp.interpretability.analyze_beliefs.
#
# 2. Exact References:
#    - V* and the per-action Q* vector sets are computed by the certified solver
#      (solve_infinite_horizon, then one per-action backup); the certified error is written into
#      the report and enters the error bounds as documented in interpretability/error_bounds.py.
#
# 3. Outputs (all under <run>/analysis/):
#      report.json     every probe report, bound, return estimate and the geometry summary
#      geometry.png    latents of held-out random-policy histories coloured by the exact posterior
# ==============================================================================

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import torch

from ipomdp.agents import PlanningAgent, UniformRandomAgent
from ipomdp.domain import BatchedPOMDPEnv, action_value_functions, solve_infinite_horizon
from ipomdp.interpretability import analyze_beliefs, build_probe_dataset
from ipomdp.planning import BeliefTreeSearch
from ipomdp.telemetry import BeliefGeometryVisualizer
from ipomdp.training import TrainingRun, load_checkpoint, play_episodes

from main import DOMAIN_BUILDERS, build_run_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Belief analysis of a trained run (see module header).")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--episodes", type=int, default=512, help="Episodes per data set and return estimate.")
    parser.add_argument("--solver-tolerance", type=float, default=0.01, help="Certified ||V* - V_n|| bound.")
    parser.add_argument("--mlp-probe-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.run_dir / ".hydra" / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pomdp = DOMAIN_BUILDERS[cfg.env.name]()
    run = TrainingRun(pomdp, build_run_config(cfg), device)
    best = load_checkpoint(args.run_dir / "checkpoints" / "best.pt")
    for name, net in run.networks.items():
        net.load_state_dict(best["networks"][name])
    run.world_model.eval()

    solution = solve_infinite_horizon(pomdp, tolerance=args.solver_tolerance, prune_epsilon=1e-6)
    action_values = action_value_functions(pomdp, solution.value_function, 1e-6)
    model = run.eval_agent.model
    planner = PlanningAgent(
        model, BeliefTreeSearch(model, cfg.mcts.num_simulations, cfg.mcts.c_puct, cfg.mcts.dirichlet_alpha,
                                dirichlet_epsilon=0.0, seed=args.seed),
        args.episodes, temperature=0.0, seed=args.seed, device=device)
    analysis = analyze_beliefs(pomdp, run.world_model.belief_filter, planner, solution.value_function, action_values,
                               solution.error_bound, cfg.env.max_steps, args.episodes, args.mlp_probe_steps,
                               args.seed, device)

    out = args.run_dir / "analysis"
    out.mkdir(exist_ok=True)
    report = asdict(analysis) | {"checkpoint_collection": best["collection"], "checkpoint_eval_return": best["eval_return"]}
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    episodes, _ = play_episodes(BatchedPOMDPEnv(pomdp, args.episodes, cfg.env.max_steps, args.seed + 99, device),
                                UniformRandomAgent(pomdp.num_actions, args.episodes, args.seed + 99, device))
    dataset = build_probe_dataset(pomdp, run.world_model.belief_filter, episodes)
    plt.close(BeliefGeometryVisualizer(str(out)).plot(dataset.latents, dataset.posteriors[:, 0], pomdp.state_names[0],
                                                      filename="geometry"))

    returns = analysis.returns
    print(f"V*(b0) = {analysis.optimal_value_b0:.3f} (certified error {analysis.solver_error_bound:.3g})")
    for name, estimate in returns.items():
        print(f"  {name:<16} discounted return {estimate.mean:8.3f} +- {estimate.stderr:.3f}")
    for name, probe in analysis.probes.items():
        print(f"  probe {name:<14} KL mean {probe.mean_kl:.5f}  L1 mean {probe.mean_l1:.4f}  L1 max {probe.max_l1:.4f}")
    for name, bound in analysis.bounds.items():
        print(f"  bounds {name:<12} L_V {bound.lipschitz_value:.1f} L_Q {bound.lipschitz_q:.1f} | "
              f"worst case: value error {bound.value_error_max:.3f} <= {bound.value_error_bound_max:.3f}, "
              f"regret {bound.regret_max:.3f} <= {bound.regret_bound_max:.3f} | "
              f"expected: value error {bound.value_error_mean:.3f} <= {bound.value_error_bound_mean:.3f}, "
              f"regret {bound.regret_mean:.4f} <= {bound.regret_bound_mean:.4f} | "
              f"suboptimal decisions {bound.suboptimal_decision_rate:.4f}")
    geometry = analysis.geometry
    print(f"  geometry: explained variance {[round(r, 3) for r in geometry.explained_variance_ratio]}; "
          f"minimality ratio {geometry.minimality_ratio:.3f}")
    if geometry.log_odds_spearman is not None:
        print(f"    |Spearman(PC1..3, log-odds)| {[round(r, 3) for r in geometry.log_odds_spearman]}, "
              f"|Spearman(PC1..3, |log-odds|)| {[round(r, 3) for r in geometry.confidence_spearman]}")
    print(f"report: {out / 'report.json'}")


if __name__ == "__main__":
    main()
