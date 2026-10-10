# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/objective_ab.py
# ==============================================================================
# OFFLINE OBJECTIVE A/B: WHICH REAL (NON-PRIVILEGED) OBJECTIVE MAKES THE LATENT A BELIEF?
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (user decision, 2026-10-03):
#    - The same GRU filter learns cell, heading and the danger ahead almost perfectly from
#      privileged supervision (supervised_filter.py), but not from the agent's objectives. Which
#      objective closes the gap? Arms: representation (jepa | decoder) x observation_horizon
#      (0 | k), the multi-step observation term of training/trainer.py, section 2c.
#
# 2. Protocol (fast, no planning, identical data for every arm):
#    - The REAL TrainingRun and WorldModelTrainer (value targets, Polyak acting copies and all)
#      built from the baseline's config with the arm's overrides and --seed.
#    - Training data: the cached pool of supervised_filter.py (reference-agent risk 0.4 and
#      random games, 50 steps), uniformly sampled minibatches of the config's batch_size; the
#      update count defaults to the full agent run's (100 collections x 64 updates).
#    - Scoring on the held-out pool, from the ACTING filter (what the planner uses):
#      a. probes on frozen latents (MLP, fitted on 1024 training games): hazard marginals,
#         cell, heading and the danger ahead (exactly the supervised_filter.py targets);
#      b. the planning model's death prediction: P_hat(GAME_OVER | z, FORWARD) from the acting
#         observation head against the exact danger ahead, overall and in the safe and
#         certain-death bins. This is what the planner's step decisions depend on.
#    - Output: "RESULT name value" lines after training (parsed by ab_summary.py), plus
#      "RESULT@<updates> name value" lines at the --score-at update counts (learning curves).
#
# 2b. Overfitting A/B (user decision, 2026-10-06; head_shift.py found the jointly trained filter and
#    observation head calibrated in-sample but overconfident on fresh games):
#    - Arms: --weight-decay (AdamW, every trained network) and --train-games (the first n games of
#      each half of the training pool, so 2n games), plus the --score-at learning curve as the
#      early-stopping lever.
#    - Extra scores: the observation head's next-observation NLL on 1024 training games vs the
#      held-out pool (alive steps, every action; the generalisation gap), and head_safe_miss: the
#      mean exact danger at held-out states where the head predicts P(death | FORWARD) < 0.02 (the
#      confident "safe" errors the greedy planner exploits; 0 for a perfect head).
#
# 2c. Horizon A/B and saved networks (user decision, 2026-10-09: latent work after every value-side
#    remedy for the unfinished games failed):
#    - Arms: --horizon 5 (the current agent's recipe) against 10 and 20, decoder, 3 seeds each, with
#      the current agent's config (--config runs/wumpus/u64_store_eval_eps_decoder_k5_seed0/...).
#      Over longer action sequences the future BUMPs and percepts depend on position and heading, so
#      the multi-step term is the natural non-privileged pressure towards them.
#    - --save writes every network's state (run.networks, online and acting copies) after training,
#      so later diagnostics need no retraining (HANDOFF section 7).
#
# 3. Usage:
#    uv run python reports/2026-10-02_wumpus_diagnostics/objective_ab.py --representation jepa \
#        --horizon 5 --seed 1 --pool-cache runs/wumpus/supervised_filter/pools.pt
# ==============================================================================

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from ipomdp.domain.wumpus import GAME_OVER, NUM_CELLS
from ipomdp.experiments.runs import build_training_run
from ipomdp.training import EpisodeBatch

sys.path.insert(0, str(Path(__file__).parent))
from supervised_filter import pool  # noqa: E402  (same diagnostics directory)


def latents_of(belief_filter, data, rows, num_a, num_o):
    with torch.no_grad():
        z = belief_filter.unroll(F.one_hot(data["actions"][rows], num_a).float(),
                                 F.one_hot(data["observations"][rows], num_o).float(),
                                 F.one_hot(data["initial"][rows], num_o).float())
    alive = data["alive"][rows]
    return z[alive].float(), {k: data[k][rows][alive] for k in ("hazards", "cell", "heading", "danger")}


def observation_nll(run, data, rows, num_a: int, num_o: int) -> torch.Tensor:
    """Acting head's next-observation NLL at alive steps of data[rows] (every action taken)."""
    with torch.no_grad():
        actions = F.one_hot(data["actions"][rows], num_a).float()
        z = run.acting_filter.unroll(actions, F.one_hot(data["observations"][rows], num_o).float(),
                                     F.one_hot(data["initial"][rows], num_o).float())[:, :-1]
        logits = run.acting_observation_head(z, actions).float()
        nll = F.cross_entropy(logits.flatten(0, 1), data["observations"][rows].flatten(), reduction="none")
    return nll[data["alive"][rows][:, :-1].flatten()].mean()


def score(run, train, held, num_a: int, num_o: int, device: torch.device) -> dict[str, float]:
    """Section 2's scores of the run's acting filter and observation head on the held-out pool."""
    # (a) Probes on frozen acting latents.
    acting = run.acting_filter
    acting.eval()
    fit_z, fit_t = latents_of(acting, train, slice(0, 1024), num_a, num_o)
    fit_z2, fit_t2 = latents_of(acting, train, slice(4096, 5120), num_a, num_o)          # the random half
    fit_z = torch.cat([fit_z, fit_z2])
    fit_t = {k: torch.cat([fit_t[k], fit_t2[k]]) for k in fit_t}
    held_z, held_t = latents_of(acting, held, slice(None), num_a, num_o)
    mean, std = fit_z.mean(0), fit_z.std(0) + 1e-6
    torch.manual_seed(0)
    probe = nn.Sequential(nn.Linear(fit_z.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                          nn.Linear(256, 3 * NUM_CELLS + NUM_CELLS + 4 + 1)).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), 1e-3)
    probe_generator = torch.Generator(device=device).manual_seed(0)
    for _ in range(3000):
        rows = torch.randint(0, len(fit_z), (512,), device=device, generator=probe_generator)
        out = probe((fit_z[rows] - mean) / std)
        loss = (F.binary_cross_entropy_with_logits(out[:, :48], fit_t["hazards"][rows])
                + F.cross_entropy(out[:, 48:64], fit_t["cell"][rows]) + F.cross_entropy(out[:, 64:68], fit_t["heading"][rows])
                + F.binary_cross_entropy_with_logits(out[:, 68], fit_t["danger"][rows]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        out = probe((held_z - mean) / std)
        hazards, danger = torch.sigmoid(out[:, :48]), torch.sigmoid(out[:, 68])
        forward = F.one_hot(torch.zeros(len(held_z), dtype=torch.int64, device=device), num_a).float()
        head = F.softmax(run.acting_observation_head(held_z, forward).float(), -1)[:, GAME_OVER]
    exact = held_t["danger"]
    safe, certain = exact < 1e-9, exact > 0.6
    results = {
        "probe_pit_error": (hazards[:, :16] - held_t["hazards"][:, :16]).abs().mean(),
        "probe_wumpus_error": (hazards[:, 16:32] - held_t["hazards"][:, 16:32]).abs().mean(),
        "probe_gold_error": (hazards[:, 32:48] - held_t["hazards"][:, 32:48]).abs().mean(),
        "probe_cell_acc": (out[:, 48:64].argmax(1) == held_t["cell"]).float().mean(),
        "probe_heading_acc": (out[:, 64:68].argmax(1) == held_t["heading"]).float().mean(),
        "probe_danger_error": (danger - exact).abs().mean(),
        "probe_certain_death": danger[certain].mean(),
        "head_danger_error": (head - exact).abs().mean(),
        "head_safe": head[safe].mean(),
        "head_certain_death": head[certain].mean(),
        "head_safe_miss": exact[head < 0.02].mean(),
        "head_nll_train": observation_nll(run, train, slice(0, 1024), num_a, num_o),
        "head_nll_held": observation_nll(run, held, slice(None), num_a, num_o),
    }
    acting.train()
    return {name: float(value) for name, value in results.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="runs/wumpus/north50_seed0/.hydra/config.yaml")
    parser.add_argument("--representation", choices=("jepa", "decoder"), required=True)
    parser.add_argument("--horizon", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=6400)
    parser.add_argument("--pool-cache", type=Path, required=True)
    parser.add_argument("--weight-decay", type=float, default=None, help="AdamW weight decay (default: the config's).")
    parser.add_argument("--train-games", type=int, default=None,
                        help="Train on the first n games of each half of the pool (2n games; default: all).")
    parser.add_argument("--save", type=Path, default=None, help="Write run.networks' states here after training.")
    parser.add_argument("--score-at", type=int, nargs="*", default=[],
                        help="Also score at these update counts (lines 'RESULT@<updates> name value').")
    args = parser.parse_args()

    device = torch.device("cuda")
    cfg = OmegaConf.load(args.config)
    cfg.model.representation, cfg.model.observation_horizon, cfg.seed = args.representation, args.horizon, args.seed
    if args.weight_decay is not None:
        cfg.trainer.weight_decay = args.weight_decay
    setup, run = build_training_run(cfg, device)
    world = setup.domain
    num_a, num_o = world.num_actions, world.num_observations
    length = cfg.env.max_steps
    if args.pool_cache.exists():
        cached = torch.load(args.pool_cache, map_location=device)
        if cached["length"] != length or "rewards" not in cached["train"]:
            raise ValueError(f"{args.pool_cache} does not match ({cached['length']} steps, keys {list(cached['train'])}).")
        train, held = cached["train"], cached["held"]
    else:
        train, held = pool(world, 4096, length, 2026, device), pool(world, 512, length, 2027, device)
        torch.save({"games": 4096, "length": length, "train": train, "held": held}, args.pool_cache)
    half = train["actions"].shape[0] // 2
    fit_rows = (torch.arange(train["actions"].shape[0], device=device) if args.train_games is None else
                torch.cat([torch.arange(args.train_games, device=device), half + torch.arange(args.train_games, device=device)]))
    print(f"arm representation={args.representation} horizon={args.horizon} seed={args.seed} "
          f"weight_decay={cfg.trainer.weight_decay}; {len(fit_rows)} training games", flush=True)

    t0 = time.time()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    for update in range(1, args.updates + 1):
        rows = fit_rows[torch.randint(0, len(fit_rows), (cfg.training.batch_size,), device=device, generator=generator)]
        metrics = run.trainer.train_step(EpisodeBatch(train["actions"][rows], train["observations"][rows],
                                                      train["rewards"][rows], train["initial"][rows]))
        run._update_acting()
        if update in args.score_at:
            for name, value in score(run, train, held, num_a, num_o, device).items():
                print(f"RESULT@{update} {name} {value:.4f}", flush=True)
        if update % 1600 == 0:
            print(f"update {update}: " + " ".join(f"{k} {v:.4f}" for k, v in metrics.items() if k.startswith("loss")),
                  flush=True)
    print(f"training took {time.time() - t0:.0f}s", flush=True)
    if args.save is not None:
        torch.save({name: net.state_dict() for name, net in run.networks.items()}, args.save)

    for name, value in score(run, train, held, num_a, num_o, device).items():
        print(f"RESULT {name} {value:.4f}")

if __name__ == "__main__":
    main()
