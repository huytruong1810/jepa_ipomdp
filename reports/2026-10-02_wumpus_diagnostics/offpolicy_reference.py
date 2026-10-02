# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/offpolicy_reference.py
# ==============================================================================
# OFF-POLICY DIAGNOSTIC: TRAIN ON REFERENCE-AGENT + RANDOM GAMES, EVALUATE THE GREEDY PLANNER
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (user decision, 2026-10-02):
#    - The baseline planner (runs/wumpus/north50_seed0) has a good latent but wanders instead
#      of exploring new safe cells (HANDOFF.md, section 4c). Is the bottleneck the DATA its own
#      exploration produces, or the LEARNING (value through the learned model)?
#    - Train the unchanged stack on data that already contains directed exploration: if the
#      greedy planner then approaches the reference agent, the data/exploration is the
#      bottleneck; if it stays near the baseline, value learning is.
#    - Diagnostic only: the reference agent uses the privileged exact belief, so this is never
#      a training method for the agent.
#
# 2. Identical to the Baseline Except the Behaviour Policy:
#    - Same config (the baseline's resolved .hydra/config.yaml), seed, TrainingRun, replay
#      buffer, 100 collections of env_batch_size games, updates_per_collection updates each,
#      Polyak acting copies, greedy evaluation every eval_every collections and best.pt
#      selection by that evaluation.
#    - Each collection: env_batch_size / 2 games of the reference agent (risk 0.4, the best of
#      the analysed grid) and env_batch_size / 2 uniformly random games, on the run's own
#      collection simulator (fresh layouts each collection). The planner never collects.
#
# 3. Output:
#    - A run directory (default runs/wumpus/offpolicy_reference_seed0) with .hydra/ copied
#      from the baseline, checkpoints/best.pt and main.log, so `uv run analyze.py <dir>`
#      analyses it exactly like the baseline.
# ==============================================================================

import argparse
import logging
import shutil
from pathlib import Path

import torch

from ipomdp.agents import UniformRandomAgent, WumpusReferenceAgent
from ipomdp.experiments.runs import build_training_run
from ipomdp.training import EpisodeBatch, play_episodes, save_checkpoint, stream_seed
from omegaconf import OmegaConf


def _concat(a: EpisodeBatch, b: EpisodeBatch) -> EpisodeBatch:
    return EpisodeBatch(torch.cat([a.actions, b.actions]), torch.cat([a.observations, b.observations]),
                        torch.cat([a.rewards, b.rewards]), torch.cat([a.initial_observations, b.initial_observations]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("runs/wumpus/north50_seed0"))
    parser.add_argument("--out", type=Path, default=Path("runs/wumpus/offpolicy_reference_seed0"))
    parser.add_argument("--risk", type=float, default=0.4)
    parser.add_argument("--overrides", nargs="*", default=[], help="Config overrides (smoke tests only), e.g. training.total_episodes=128")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.baseline / ".hydra", args.out / ".hydra")
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        handlers=[logging.FileHandler(args.out / "main.log"), logging.StreamHandler()])
    log = logging.getLogger("offpolicy")

    device = torch.device("cuda")
    cfg = OmegaConf.merge(OmegaConf.load(args.out / ".hydra" / "config.yaml"), OmegaConf.from_dotlist(args.overrides))
    setup, run = build_training_run(cfg, device)
    t = cfg.training
    half = t.env_batch_size // 2
    simulator = setup.make_simulator(half, cfg.env.max_steps, stream_seed(cfg.seed, 1001), device)
    reference = WumpusReferenceAgent(setup.domain, half, args.risk)
    random_agent = UniformRandomAgent(setup.domain.num_actions, half, stream_seed(cfg.seed, 1002), device)
    total_collections = t.total_episodes // t.env_batch_size
    log.info(f"off-policy diagnostic: {total_collections} collections of {half} reference (risk {args.risk}) "
             f"+ {half} random games; baseline config {args.baseline}")

    for collection in range(1, total_collections + 1):
        episodes = _concat(play_episodes(simulator, reference)[0], play_episodes(simulator, random_agent)[0])
        run.buffer.add(episodes)
        for _ in range(t.updates_per_collection):
            run.trainer.train_step(run.buffer.sample(t.batch_size))
            run._update_acting()
        if collection % t.eval_every == 0:
            evaluation = run.evaluate()
            outcomes = ", ".join(f"{k.removeprefix('eval_')} {v:.3f}" for k, v in evaluation.items()
                                 if k.removeprefix("eval_") in run.domain_metric_names)
            log.info(f"collection {collection}: greedy eval discounted return "
                     f"{evaluation['eval_discounted_return']:.2f} +- {evaluation['eval_discounted_return_stderr']:.2f} "
                     f"(best {run.best_eval_return:.2f}), {outcomes}")
            if evaluation["eval_discounted_return"] >= run.best_eval_return:
                save_checkpoint(args.out / "checkpoints" / "best.pt", {
                    "collection": collection, "eval_return": evaluation["eval_discounted_return"],
                    "networks": {name: net.state_dict() for name, net in run.networks.items()}})
    save_checkpoint(args.out / "checkpoints" / "final.pt", {
        "collection": total_collections, "eval_return": float("nan"),
        "networks": {name: net.state_dict() for name, net in run.networks.items()}})


if __name__ == "__main__":
    main()
