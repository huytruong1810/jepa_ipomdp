# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/retrain_play.py
# ==============================================================================
# DOES THE BETTER LATENT FROM A WHOLE-BUFFER RETRAIN PLAY BETTER WITH THE DOUBLE VALUE? (objective_ab.py 2h)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-16, after objective_ab.py 2e-2g; representation direction, HANDOFF 6.2):
#    - Trained uniformly offline on the agent's own final buffer, the agent's objective gives cell / heading
#      85 / 87% (2e buffer arm) where the agent's growing-buffer training gives 76 / 68% (2f: primacy). Resets
#      with a budget-matched catch-up do not close the gap (2g FAIL): the final network has to be trained long
#      and uniformly on the full buffer, i.e. be the buffer arm itself.
#    - The representation direction (user, 2026-10-15) says: a better own latent, then the double value. C6
#      says the double value fails on the current latent because the planner exploits the model's errors. So
#      the question that decides whether to build a "retrain on the whole buffer" step into the agent is:
#      do networks retrained from scratch on the final buffer, WITH model.value_estimator=double, play better
#      than the double run's own best.pt networks, on the same games?
#    - Cheapest test first (section 1b): no agent change. objective_ab.py --train-buffer <double run> --save
#      trains every network (filter, heads, the value pair and their acting copies) with the double run's own
#      config; this script swaps them into that run's planner and plays.
#
# 2. Method (usage: uv run python <this> <double run dir> <saved networks .pt> [<games>]):
#    - Baseline: the run's best.pt planner on the 512 analysis games of report.json (stream_seed(settings.seed,
#      1), planner streams 2 and 3, Dirichlet off, temperature 0); must reproduce report.json (raises otherwise).
#    - Arm: the same planner and games with EVERY network replaced by the saved retrained ones (run.networks
#      keys, online and acting; the planner reads the acting copies).
#    - Retraining (runs/wumpus/retrain_play_chain.sh): objective_ab.py, the double run's .hydra config,
#      --representation decoder --horizon 5 --seed <s> --train-buffer <double run> (uniform over latest.pt's
#      buffer), 6400 updates (the agent's budget), --save. Its probe lines also score the retrained latent.
#    - Measures: return, paired gain per game (95%), win / death / unfinished, live action mix.
#
# 3. Pre-registered readout (2026-10-16, before any run), u64_store_eval_eps_decoder_k5_double_seed{0,1,2}
#    (best.pt baselines 163.8 / 81.7 / 168.0):
#    - PASS: paired gain >= +50 on 2+ of 3 seeds with the death rate not up on those seeds -> a default-off
#      "retrain the world model from scratch on the whole buffer" step in the agent (Tiger fingerprint), then
#      3 seeds with model.value_estimator=double paired with the double runs.
#    - FAIL: otherwise. A latent 9-19 points better on cell/heading does not make the double-value planner
#      safe; the representation gap is not the lever at these accuracies, and the decision goes to the user
#      (HANDOFF 6.2 options 2 and 3).
#    - Caveat stated in advance: the retrained value has never been trained on the agent's own later play,
#      and best.pt is a max over evaluations (biased upward relative to a single network), which favours the
#      baseline; a PASS is therefore conservative.
#
# 4. Result: (pending)
# ==============================================================================

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ipomdp.agents import PlanningAgent
from ipomdp.domain import WumpusSimulator, game_outcomes
from ipomdp.domain.wumpus import WumpusAction
from ipomdp.experiments.runs import load_trained_run
from ipomdp.experiments.wumpus import WumpusAnalysisSettings
from ipomdp.planning import BeliefTreeSearch
from ipomdp.training import discounted_returns, play_episodes, stream_seed

run_dir, saved = Path(sys.argv[1]), Path(sys.argv[2])
dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(0.3, torch.cuda.current_device())    # three jobs share the GPU
settings = WumpusAnalysisSettings()
n = int(sys.argv[3]) if len(sys.argv) > 3 else settings.episodes
trained = load_trained_run(run_dir, dev)
world, run, cfg = trained.setup.domain, trained.run, trained.cfg
if cfg.model.value_estimator != "double":
    raise ValueError(f"{run_dir} has value_estimator={cfg.model.value_estimator}; this screen is for double runs.")
model = run.eval_agent.model
GAMMA, NAMES = world.discount, [a.name for a in WumpusAction]


class Progress:
    """Pass-through agent that prints a decision counter (a stall shows within minutes)."""

    def __init__(self, actor):
        self.actor, self.batch_size, self.decisions, self.started = actor, actor.batch_size, 0, time.time()

    def reset(self, initial_observation):
        self.actor.reset(initial_observation)

    def update(self, action, observation):
        self.actor.update(action, observation)

    def act(self):
        self.decisions += 1
        if self.decisions % 10 == 0:
            print(f"    decision {self.decisions}; {time.time() - self.started:.0f} s", flush=True)
        return self.actor.act()


def play():
    planner = PlanningAgent(
        model, BeliefTreeSearch(model, cfg.mcts.num_simulations, cfg.mcts.c_puct, cfg.mcts.dirichlet_alpha,
                                dirichlet_epsilon=0.0, seed=stream_seed(settings.seed, 2),
                                observation_prune=cfg.mcts.observation_prune),
        n, temperature=0.0, seed=stream_seed(settings.seed, 3), device=dev)
    episodes, _ = play_episodes(WumpusSimulator(world, n, cfg.env.max_steps, stream_seed(settings.seed, 1), dev),
                                Progress(planner))
    del planner                                   # BeliefTreeSearch keeps its last trees
    torch.cuda.empty_cache()
    return episodes


def report(name: str, episodes, base: torch.Tensor | None) -> torch.Tensor:
    returns = discounted_returns(episodes.rewards, GAMMA).double()
    out = {k: round(float(v), 4) for k, v in game_outcomes(episodes.rewards).items()}
    ended = (episodes.rewards.abs() >= 998).long().cumsum(1)
    live = (ended - (episodes.rewards.abs() >= 998).long()) == 0
    acts = episodes.actions[live]
    mix = ", ".join(f"{NAMES[a]} {(acts == a).float().mean():.3f}" for a in range(len(NAMES)))
    line = f"  {name:14s}: return {returns.mean():7.1f} +- {returns.std() / np.sqrt(n):5.1f}  {out}"
    if base is not None:
        d = returns - base
        line += f"  paired gain {d.mean():+7.1f} +- {1.96 * d.std() / np.sqrt(n):5.1f}"
    print(line + f"\n      live action mix {mix}", flush=True)
    return returns


print(f"== {run_dir.name} best.pt (collection {trained.checkpoint_collection}) vs {saved}, {n} games", flush=True)
episodes = play()
base = report("best.pt", episodes, None)
if n == settings.episodes:
    reported = json.loads((run_dir / "analysis" / "report.json").read_text())["returns"]["learned_planner"]
    out = game_outcomes(episodes.rewards)
    for key in ("win_rate", "death_rate", "unfinished_rate"):
        if abs(reported[key] - float(out[key])) > 1e-9:
            raise RuntimeError(f"Replay is not faithful: {key} {float(out[key])} vs report.json {reported[key]}.")
    print("  replay reproduces report.json", flush=True)

states = torch.load(saved, map_location=dev)
if set(states) != set(run.networks):
    raise ValueError(f"saved networks {sorted(states)} do not match the run's {sorted(run.networks)}.")
for name, net in run.networks.items():
    net.load_state_dict(states[name])
report("retrained", play(), base)
print("DONE", flush=True)
