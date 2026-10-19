# ABSOLUTE PATH: reports/2026-10-19_eval_protocol/recurrent_dqn_baseline.py
# ==============================================================================
# TRAIN AND ANALYSE ONE SEED OF THE MODEL-FREE RECURRENT BASELINE (R2D2-LITE)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Usage:
#      uv run reports/2026-10-19_eval_protocol/recurrent_dqn_baseline.py <name> <seed> \
#          [--analysis-seed 12345] [--output analysis] [--set key=value ...]
#    - Trains src/ipomdp/baselines/recurrent_dqn.py on training seed <seed> with the matched
#      budget (its header, section 2), then plays the best and the final network greedily on the
#      analysis seed's 512 games (the stream of experiments/wumpus.py, stream_seed(seed, 1)).
#    - --set overrides RecurrentDQNConfig fields (the development screen, PROTOCOL.md C-c).
#    - Writes runs/baselines/<name>/seed<seed>/: config.json (condition; re-running with a
#      different one raises), best.pt / final.pt (weights), train.log, and
#      <output>/games.npz + <output>/report.json. Re-running skips a finished training.
#
# 2. games.npz keys (paired with every other agent on the same analysis seed):
#    - "recurrent_dqn.return" / ".won" / ".died" (best network), "recurrent_dqn_final.*" (final
#      network), "reference_risk0.4.return" (the heuristic on the same rows: the pairing check
#      of experiments/aggregate.py, section 6), "analysis_seed".
#
# 3. Blindness:
#    - On a held-out analysis seed (not 12345) this script prints no return; the numbers are
#      read once, in the registered aggregate of claim C-c.
# ==============================================================================

import argparse
from dataclasses import asdict, fields, replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from ipomdp.agents import WumpusReferenceAgent
from ipomdp.baselines import RecurrentDQNAgent, RecurrentDQNConfig, RecurrentQNetwork, train_recurrent_dqn
from ipomdp.domain import WumpusSimulator, WumpusWorld, game_outcome_flags
from ipomdp.experiments.wumpus import WumpusAnalysisSettings
from ipomdp.training import discounted_returns, play_episodes, stream_seed

MAX_STEPS, RISK, DEVELOPMENT_SEED = 50, 0.4, 12345

parser = argparse.ArgumentParser()
parser.add_argument("name")
parser.add_argument("seed", type=int)
parser.add_argument("--analysis-seed", type=int, default=DEVELOPMENT_SEED)
parser.add_argument("--output", default="analysis")
parser.add_argument("--set", nargs="*", default=[], help="RecurrentDQNConfig overrides, key=value.")
args = parser.parse_args()

types = {f.name: f.type for f in fields(RecurrentDQNConfig)}
overrides = {}
for item in args.set:
    key, value = item.split("=", 1)
    if key not in types:
        raise ValueError(f"Unknown RecurrentDQNConfig field {key!r}.")
    overrides[key] = (float if types[key] in (float, "float") else int)(value)
cfg = replace(RecurrentDQNConfig(), **overrides)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
world = WumpusWorld(device)
run = Path("runs/baselines") / args.name / f"seed{args.seed}"
run.mkdir(parents=True, exist_ok=True)
condition = {"config": asdict(cfg), "seed": args.seed}
if (run / "config.json").exists():
    if json.loads((run / "config.json").read_text()) != condition:
        raise ValueError(f"{run} was trained with another condition; use a new name.")
else:
    (run / "config.json").write_text(json.dumps(condition, indent=2))

if not (run / "final.pt").exists():
    started = time.time()
    with open(run / "train.log", "w") as log_file:
        def log(line: str) -> None:
            print(line, flush=True)
            log_file.write(line + "\n")
            log_file.flush()
        result = train_recurrent_dqn(cfg, world, MAX_STEPS, args.seed, device, log)
        log(f"trained in {time.time() - started:.0f} s; best collection {result.best_collection}")
    torch.save(result.best_state, run / "best.pt")
    torch.save({"state": result.final_state, "evaluations": result.evaluations,
                "best_collection": result.best_collection}, run / "final.pt")

settings = WumpusAnalysisSettings(seed=args.analysis_seed)
simulator = lambda: WumpusSimulator(world, settings.episodes, MAX_STEPS, stream_seed(settings.seed, 1), device)  # noqa: E731
rows, report = {"analysis_seed": np.int64(args.analysis_seed)}, {"analysis_seed": args.analysis_seed}
for key, state in (("recurrent_dqn", torch.load(run / "best.pt", map_location=device)),
                   ("recurrent_dqn_final", torch.load(run / "final.pt", map_location=device)["state"])):
    network = RecurrentQNetwork(world.num_observations, world.num_actions, cfg.hidden_dim).to(device)
    network.load_state_dict(state)
    agent = RecurrentDQNAgent(network, torch.zeros(settings.episodes), stream_seed(settings.seed, 3), device)
    episodes, _ = play_episodes(simulator(), agent)
    won, died = game_outcome_flags(episodes.rewards)
    returns = discounted_returns(episodes.rewards, world.discount).cpu().numpy()
    rows.update({f"{key}.return": returns, f"{key}.won": won.cpu().numpy(), f"{key}.died": died.cpu().numpy()})
    report[key] = {"return": float(returns.mean()), "win_rate": float(won.double().mean()),
                   "death_rate": float(died.double().mean())}
reference, _ = play_episodes(simulator(), WumpusReferenceAgent(world, settings.episodes, RISK))
rows[f"reference_risk{RISK}.return"] = discounted_returns(reference.rewards, world.discount).cpu().numpy()
out = run / args.output
out.mkdir(exist_ok=True)
np.savez(out / "games.npz", **rows)
(out / "report.json").write_text(json.dumps(report, indent=2))
if args.analysis_seed == DEVELOPMENT_SEED:
    print(json.dumps(report, indent=2))
print(f"games: {out / 'games.npz'}")
