# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/head_swap_screen.py
# ==============================================================================
# DO NON-JOINT OBSERVATION HEADS STOP THE DOUBLE-VALUE PLANNER'S RISKY DEATHS? (R1a, HEAD SIDE)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-15, representation plan R1, HANDOFF 6.2; after head_shift.py item 5 = branch (M)):
#    - With model.value_estimator=double the current recipe loses to the oracle recipe (138 vs 223 on seeds
#      0-2), largely by deaths. On best.pt's fresh greedy games those deaths sit in the head's MODERATE bins
#      (predicted 0.04-0.06, realised 0.13-0.35 on seeds 1-2), and at exact-danger >= 0.4 FORWARDs a fresh head
#      on the same frozen latents reads 0.35 / 0.83 where the agent's jointly trained head reads 0.07 / 0.05.
#      The danger is in the latent; the planning model's head under-reports it, and the greedy planner selects
#      exactly the under-reported steps.
#    - Does a head that is not jointly trained (fresh, on frozen acting latents) give the planner better risk
#      estimates IN PLAY? An ensemble adds the classic model-exploitation guard: disagreement shows where data
#      are thin (deep ensembles, Lakshminarayanan et al. 2017; pessimism over a model ensemble as in MOPO,
#      Yu et al. 2020), and the greedy planner is exactly an optimiser that seeks the model's errors.
#
# 2. Method (usage: uv run python <this> <run dir> [<games>]; 512 = the analysis games):
#    - Networks: best.pt (load_trained_run). Baseline: the unchanged planner on the analysis games of
#      report.json (stream_seed(settings.seed, 1), planner streams 2 and 3, Dirichlet off, temperature 0); it must
#      reproduce report.json (raises otherwise).
#    - Fresh heads: K = 5 ObservationHeads (the agent's architecture; seeds 0-4), each fitted from scratch on the
#      frozen ACTING latents of the buffer slots that existed at best.pt's collection (through its stored greedy
#      block; latest.pt's later games are excluded, unlike head_shift.py), head_shift.py's recipe: Adam 3e-4,
#      6000 steps, minibatch 4096, cross-entropy on o_{t+1} at every step (the trainer's target).
#    - Arms, each replacing the planner's observation head (model.observation_head, which expand() reads);
#      the value, reward head and filter stay best.pt's:
#        (a) fresh:       head 0 alone;
#        (b) mean:        the ensemble's mixture, P(o) = mean_k P_k(o);
#        (c) pessimistic: P(GAME_OVER) = max_k P_k(GAME_OVER), the other observations take the mixture's
#                         conditional shares of the remaining mass.
#    - Measures per arm: return, paired gain per game over the baseline, win / death / unfinished, the live
#      action mix, and the arm's own head calibration at its greedy FORWARDs (bins as head_shift.py), which
#      shows whether the swap reaches the decisions (fewer moderate-bin FORWARDs, deaths in fewer bins).
#
# 3. Pre-registered readout (2026-10-15, before any run), current-recipe double runs seeds 0-2
#    (u64_store_eval_eps_decoder_k5_double_seed{0,1,2}):
#    - PASS for an arm: paired gain >= +50 on 2+ of 3 seeds with the death rate down on those seeds.
#    - (a) passes -> train with the EXISTING model.detached_planning_head (a second head on detached latents
#      for the planner; it failed on 2026-10-10 under the max value at 128 updates per collection) plus
#      model.value_estimator=double at 64 updates per collection, 3 seeds paired with the double runs.
#    - only (b) or (c) passes -> a default-off ensemble planning head in the trainer (Tiger fingerprint), then
#      the same training comparison.
#    - none passes -> swapping the head on frozen networks is not enough (the value was trained with the joint
#      head's model, so a head swap alone may not be read as "the head is not the lever"): report per seed and
#      take the decision to the user.
#
# 4. Result 2026-10-15 (128-game smoke plays, logs runs/wumpus/<run>.head_swap_smoke.log for seeds 0 and 2;
#    seed 1 printed to the console): NONE PASSES; stopped before the 512-game plays because no arm can reach
#    the pass on 2 of 3 seeds (seeds 0 and 1 are strongly negative for every arm). Paired gain over the joint
#    head (95%), deaths joint -> arm:
#                   (a) fresh            (b) mean             (c) pessimistic      deaths
#      seed 0       -359 +- 114          -134 +- 88           -187 +- 93           1.6% -> 41 / 14 / 20%
#      seed 1       -164 +- 133          -275 +- 117          -159 +- 122          8.6% -> 42 / 47 / 37%
#      seed 2        -77 +- 85            -81 +- 100           +34 +- 77           3.9% ->  8 / 12 /  2%
#    - Mechanism, from each arm's own head at its greedy FORWARDs: the fresh heads' CONFIDENT "safe" bin
#      [0, 0.02) dies 1.3-6.2% in play (the joint head's 0.0-0.3%). The planner finds and exploits the swapped
#      heads' errors, which are worse where the planner goes than the joint head's. head_shift.py's fresh-head
#      advantage (item 5) was measured at the JOINT planner's states only: a selection effect, not a better
#      head. Ensembling (b) and per-observation pessimism (c) do not prevent it: 5 seeds of the same recipe
#      on the same latents share their errors.
#    - Caveat (stated in 3): the value was trained with the joint head's model, so a swapped head also breaks
#      the value/model consistency. The swap is not a clean test of "a better head", but every arm moved
#      the decisions (action mix and FORWARD bins change), so the intervention did reach them.
#    - Reading: the joint head is not the weak part that a non-joint head can replace; any learned head's
#      errors are selected by the greedy double-value planner. Per 3: the decision goes to the user.
# ==============================================================================

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ipomdp.agents import PlanningAgent
from ipomdp.domain import WumpusSimulator, game_outcomes
from ipomdp.domain.wumpus import GAME_OVER, WumpusAction
from ipomdp.experiments.runs import load_trained_run
from ipomdp.experiments.wumpus import WumpusAnalysisSettings
from ipomdp.models import ObservationHead
from ipomdp.planning import BeliefTreeSearch
from ipomdp.training import discounted_returns, play_episodes, stream_seed
from ipomdp.training.checkpointing import load_checkpoint

run_dir = Path(sys.argv[1])
dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(0.3, torch.cuda.current_device())    # three jobs share the GPU
settings = WumpusAnalysisSettings()
n = int(sys.argv[2]) if len(sys.argv) > 2 else settings.episodes
trained = load_trained_run(run_dir, dev)
world, run, cfg = trained.setup.domain, trained.run, trained.cfg
model = run.eval_agent.model
if model.horizon is not None:
    raise ValueError("Time-aware runs carry steps-left in the state; this screen reads plain latents only.")
GAMMA, FORWARD, NAMES = world.discount, int(WumpusAction.FORWARD), [a.name for a in WumpusAction]
num_a, num_o = world.num_actions, world.num_observations
K = 5

# --- Training data: the buffer as it was at best.pt's collection (phantom_reward_diag.py's slot logic) ---
buf = load_checkpoint(run_dir / "checkpoints" / "latest.pt")["buffer"]
size, t_cfg, c = buf["size"], cfg.training, trained.checkpoint_collection
if size != buf["actions"].shape[0] and buf["next"] != size:
    raise RuntimeError("the buffer wrapped around; insertion-order reconstruction does not apply.")
start = c * t_cfg.env_batch_size + (c // t_cfg.eval_every - 1) * t_cfg.eval_episodes
end = start + t_cfg.eval_episodes
if "greedy" in buf and (not bool(buf["greedy"][start:end].all()) or bool(buf["greedy"][start - 1])):
    raise RuntimeError(f"slot {start}..{end} is not collection {c}'s greedy block.")
actions, observations, initial = (buf[k][:end].to(dev) for k in ("actions", "observations", "initial_observations"))
del buf


def latents(actions, observations, initial, chunk: int = 512) -> torch.Tensor:
    """Acting-filter latents z_0..z_{T-1} (B, T, D), float32, no autocast."""
    out = []
    with torch.no_grad():
        for i in range(0, actions.shape[0], chunk):
            a = F.one_hot(actions[i:i + chunk], num_a).float()
            o = F.one_hot(observations[i:i + chunk], num_o).float()
            z0 = F.one_hot(initial[i:i + chunk], num_o).float()
            out.append(run.acting_filter.unroll(a, o, z0)[:, :-1].float())
    return torch.cat(out)


z_train = latents(actions, observations, initial).reshape(-1, cfg.model.latent_dim)
a_train, o_train = actions.reshape(-1), observations.reshape(-1)


def fit_head(seed: int, steps: int = 6000, batch: int = 4096) -> nn.Module:
    torch.manual_seed(seed)
    head = ObservationHead(cfg.model.latent_dim, num_a, num_o, cfg.model.hidden_dim, cfg.model.num_blocks).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=3e-4)
    g = torch.Generator(device=dev).manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(0, len(z_train), (batch,), device=dev, generator=g)
        loss = F.cross_entropy(head(z_train[i], F.one_hot(a_train[i], num_a).float()), o_train[i])
        opt.zero_grad(); loss.backward(); opt.step()
    return head.eval().requires_grad_(False)


class Ensemble(nn.Module):
    """Logits of the ensemble's mixture (b) or of its pessimistic death estimate (c)."""

    def __init__(self, heads: list[nn.Module], pessimistic: bool):
        super().__init__()
        self.heads, self.pessimistic = nn.ModuleList(heads), pessimistic

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        p = torch.stack([F.softmax(h(latent, action).float(), -1) for h in self.heads])     # (K, B, O)
        mix = p.mean(0)
        if self.pessimistic:
            death = p[..., GAME_OVER].max(0).values
            rest = mix.clone()
            rest[:, GAME_OVER] = 0.0
            mix = rest / rest.sum(-1, keepdim=True).clamp_min(1e-12) * (1.0 - death)[:, None]
            mix[:, GAME_OVER] = death
        return mix.clamp_min(1e-12).log()


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
    # The arm's own head at its greedy FORWARDs (live), predicted vs realised death per bin.
    z = latents(episodes.actions, episodes.observations, episodes.initial_observations)
    fwd = live & (episodes.actions == FORWARD)
    with torch.no_grad():
        p = F.softmax(model.observation_head(z[fwd], F.one_hot(episodes.actions[fwd], num_a).float()).float(),
                      -1)[:, GAME_OVER].double()
    died = (episodes.rewards[fwd] <= -1000).double()
    bins = []
    for lo, hi in ((0, 0.02), (0.02, 0.2), (0.2, 0.5), (0.5, 1.01)):
        m = (p >= lo) & (p < hi)
        if m.any():
            bins.append(f"[{lo},{hi}) n={int(m.sum())} pred {p[m].mean():.3f} died {died[m].mean():.3f}")
    print(line + f"\n      live action mix {mix}\n      own head at greedy FORWARDs: " + " | ".join(bins), flush=True)
    return returns


print(f"== {run_dir.name} best.pt (collection {c}), {n} games; {K} fresh heads on {end} buffer games "
      f"({len(z_train)} transitions)", flush=True)
joint = model.observation_head
episodes = play()
base = report("joint (agent)", episodes, None)
if n == settings.episodes:
    reported = json.loads((run_dir / "analysis" / "report.json").read_text())["returns"]["learned_planner"]
    out = game_outcomes(episodes.rewards)
    for key in ("win_rate", "death_rate", "unfinished_rate"):
        if abs(reported[key] - float(out[key])) > 1e-9:
            raise RuntimeError(f"Replay is not faithful: {key} {float(out[key])} vs report.json {reported[key]}.")
    print("  replay reproduces report.json", flush=True)

t0 = time.time()
heads = [fit_head(k) for k in range(K)]
print(f"  {K} heads fitted in {time.time() - t0:.0f} s", flush=True)
arms = {"(a) fresh": heads[0], "(b) mean": Ensemble(heads, False), "(c) pessimistic": Ensemble(heads, True)}
for name, head in arms.items():
    model.observation_head = head                 # expand() reads it; the value/reward heads stay best.pt's
    try:
        report(name, play(), base)
    finally:
        model.observation_head = joint
print("DONE", flush=True)
