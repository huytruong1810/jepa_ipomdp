# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/pessimism_play.py
# ==============================================================================
# DOES A PLANNING-TIME RISK MARGIN TURN THE COVERAGE NETWORKS' IDLING CURE INTO RETURN? (HANDOFF 6.2b, E0')
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-21, HANDOFF 6.2b, user-approved): the coverage screens left two separable data deficits and a
#    planner that exploits whatever model error remains (C10). The privileged cover networks cure most idling
#    (unfinished -31 points) but die in 27-43% of games, and risk_localise.py found the deaths are under-rated risks
#    the argmax picks out (83-100% of fatal FORWARDs rated below half the exact danger): the optimizer's curse
#    (Smith & Winkler 2006). Pessimism in acting (a lower confidence bound on the model's predictions; MOPO, Yu et
#    al. 2020; MOReL, Kidambi et al. 2020; SPIBB, Laroche et al. 2019) is the standard counter. Before building an
#    uncertainty-SCALED margin (E2), this screen asks whether the crudest margin, a FIXED inflation of the predicted
#    death risk, already turns the cover networks' idling cure into return, and what it does to the control.
#
# 2. Method (usage: uv run python <this> <run dir> <k> <saved .pt> [<saved .pt> ...] [<games>, smoke only]):
#    - Each saved network set (runs/wumpus/coverage/{control,cover}_seed<s>.pt) plays the run's 512 analysis games
#      exactly as retrain_play.py does (same streams, greedy, Dirichlet off), with the search model wrapped in
#      PessimisticModel(k): at every expansion, for FORWARD only (the one action that can kill),
#        P(GAME_OVER | z, FORWARD) -> min(k p_O, 1), the other outcomes rescaled by (1 - min(k p_O, 1)) / (1 - p_O);
#        R(z, FORWARD) -> R - 1000 (min(k p_R, 1) - p_R), with p_R = clip(-(R + 1) / 1000, 0, 1) the reward
#        head's implied death risk (FORWARD pays -1, a death -1001; as in risk_localise.py).
#      Both heads carry the death risk in the backup (the probability weights the GAME_OVER child, the reward pays
#      the -1000), so both are inflated. GRAB's GAME_OVER is the win and is not touched.
#    - k = 1 is the identity (asserted bit-for-bit on the expansion in the smoke; then the k = 1 returns are the
#      saved <set>.returns.npy of retrain_play.py, not replayed).
#    - Paired per game against the same set at k = 1 and against control at k = 1 (control_seed<s>.returns.npy).
#    - Outputs: <set>.pess<k>.returns.npy, the log's outcome lines and live action mix.
#    - Check that the margin reaches decisions (feedback-efficiency rule): the smoke (32 games, k = 4) must lower the
#      share of live FORWARDs, else the wrapper is not on the search's path.
#
# 3. Pre-registered readout (2026-10-21, before any full run; seeds 0-4 of runs/sweeps/wumpus_retrain; k in {2, 4};
#    the control and cover retrains of coverage_games.py):
#    - PASS-k (the planning-time half of E2 works): cover at k minus control at k = 1, mean over seeds >= +50 with
#      the t95 interval above 0, AND deaths at most control's + 2 points on >= 4 of 5 seeds, AND unfinished below
#      control's on >= 4 of 5 seeds. Two k are read, so a pass at one k only is reported as such (no correction;
#      the method would set its margin by uncertainty, not by this k).
#    - Branches: PASS at some k -> pessimism in acting is a necessary half; E2 = directed coverage + pessimism.
#      FAIL with deaths still above control + 2 -> a uniform margin cannot cover the under-rating (it is too
#      state-dependent: 0.5-17% of exact at the chosen risky steps): E2 needs the uncertainty-scaled margin, and E1
#      decides whether a signal exists. FAIL with deaths at control but unfinished back at control's -> the uniform
#      margin re-creates the idling (it also penalises the safe new cells, the phantom risk of risk_localise.py):
#      again an uncertainty-scaled margin.
#    - Also read (secondary, no branch): control at k minus control at k = 1 (pessimism alone, no coverage).
#
# 4. Result 2026-10-21 (runs/wumpus/coverage/pess_k{2,4}_seed*.log; <run>/analysis/pessimism_k*.json). **Readout: FAIL
#    at both k (deaths still far above control).**
#    | Seed | cover k1 / k2 / k4 - ctl k1  | cover deaths k1 -> k2 / k4 | ctl k2 - ctl k1 | ctl k4 - ctl k1 | ctl deaths k1 -> k2 / k4 |
#    | 0    | -190 / -166 / -161           | 28 -> 22 / 20%             | +17 +- 23       | +11 +- 25       | 4.5 -> 1.0 / 0.2%        |
#    | 1    | -3 / +68 / +47               | 33 -> 20 / 20%             | +19 +- 23       | +39 +- 27       | 3.1 -> 2.0 / 0.6%        |
#    | 2    | -339 / -337 / -324           | 42 -> 42 / 40%             | +18 +- 22       | +42 +- 28       | 5.5 -> 4.1 / 2.3%        |
#    | 3    | -225 / -191 / -80            | 43 -> 32 / 22%             | +80 +- 35       | +100 +- 38      | 19.0 -> 3.9 / 2.3%       |
#    | 4    | -212 / -234 / -246           | 27 -> 27 / 27%             | +29 +- 21       | +11 +- 23       | 2.7 -> 0.6 / 0.8%        |
#    - Cover + pessimism: -172 [-357, +13] (k = 2) and -153 [-332, +26] (k = 4) vs control; the cover networks still die
#      in 20-42% of games. A uniform inflation of the predicted risk barely touches deaths whose risk was rated at
#      2-5% of exact (risk_localise.py): x4 of 0.03 x danger is still far below the danger. Registered branch: the
#      margin must be state-dependent (uncertainty-scaled), and E1 found no cheap signal for it.
#    - Secondary (no branch; exploratory): pessimism alone on the CONTROL networks gains on 5 of 5 seeds at both k,
#      +33 [-1, +66] (k = 2) and +41 [-5, +86] (k = 4) over seeds, deaths -> 0.2-4.1%, unfinished about unchanged.
#      On networks without coverage data the under-rating is milder and a fixed margin removes most deaths.
#
# 5. Follow-up (2026-10-21 evening, my call under research autonomy; exploratory, pre-registered here before launch):
#    own-arm networks (coverage_games.py section 5: non-privileged, unfiltered trials; +66 [+16, +116] over best.pt,
#    from safety) with the same margin, k in {2, 4}. Question: do the two non-privileged pieces that each helped
#    (own trials in collection, a fixed risk margin in acting) add up? Read against control at k = 1
#    (control_seed<s>.returns.npy, paired per game) and against control at the same k.
#    - PROMISING: own at some k - control at k = 1 >= +50 mean with the t95 interval over seeds above 0, and deaths
#      at most control's + 2 points on >= 4 of 5 -> propose to the user an in-agent recipe (default-off options:
#      own-trial collection rule, fixed risk margin at planning), 3 then 5 development seeds.
#    - Otherwise: the fixed margin is a small, separate gain (section 4 secondary) and the method question returns
#      to the user with the representation branch and a whole-model uncertainty signal as the options.
#    - Result 2026-10-21 (pess_own_k*_seed*.log, own_pess_chain.log; ~50 min). **Readout: not PROMISING by the rule
#      (interval includes 0); deaths criterion met on 5 of 5.**
#      | Seed | own k2 - ctl k1 | own k4 - ctl k1 | own k2 return | deaths k2 / k4 | unfinished k2 |
#      | 0    | +34 +- 32       | +37 +- 30       | 290.3         | 2.3 / 0.4%     | 60.0%         |
#      | 1    | +156 +- 40      | +137 +- 39      | 220.8         | 0.0 / 0.2%     | 71.5%         |
#      | 2    | +59 +- 32       | +57 +- 32       | 278.2         | 0.0 / 0.0%     | 65.0%         |
#      | 3    | +144 +- 50      | +122 +- 51      | 303.7         | 0.4 / 0.4%     | 62.1%         |
#      | 4    | -22 +- 34       | -37 +- 32       | 284.3         | 1.9 / 0.6%     | 60.4%         |
#      - k = 2: +74 [-19, +168]; k = 4: +63 [-24, +150] over seeds (4 of 5 positive). Mean return 275 (k = 2), the best
#        non-privileged Wumpus result so far; exploratory against best.pt: +75 [+33, +118], positive on 5 of 5.
#      - The gain is safety (deaths 0-2.3%), not idling: unfinished stays 60-72%, as in every arm without targeted
#        value-of-information data. Per the rule the method question returns to the user; this pair (own trials +
#        a fixed margin) is the cheapest candidate to train in-agent if they want a safety-only improvement.
# ==============================================================================

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ipomdp.agents import PlanningAgent
from ipomdp.domain import WumpusSimulator, game_outcomes
from ipomdp.domain.wumpus import GAME_OVER, WumpusAction
from ipomdp.experiments.runs import load_trained_run
from ipomdp.experiments.wumpus import WumpusAnalysisSettings
from ipomdp.planning import BeliefTreeSearch
from ipomdp.planning.search_model import Expansion
from ipomdp.training import discounted_returns, play_episodes, stream_seed

run_dir, k = Path(sys.argv[1]), float(sys.argv[2])
saved_sets = [Path(a) for a in sys.argv[3:] if a.endswith(".pt")]
rest = [a for a in sys.argv[3:] if not a.endswith(".pt")]
if not saved_sets or len(rest) > 1 or k < 1.0:
    raise ValueError("usage: <run dir> <k >= 1> <saved .pt> [<saved .pt> ...] [<games>]")
dev = torch.device("cuda")
settings = WumpusAnalysisSettings()
n = int(rest[0]) if rest else settings.episodes
smoke = n != settings.episodes
trained = load_trained_run(run_dir, dev)
world, run, cfg = trained.setup.domain, trained.run, trained.cfg
model = run.eval_agent.model
GAMMA, NAMES, FWD = world.discount, [a.name for a in WumpusAction], int(WumpusAction.FORWARD)
DEATH = 1000.0


class PessimisticModel:
    """The learned search model with FORWARD's predicted death risk inflated by k (section 2)."""

    def __init__(self, inner, k: float):
        self.inner, self.k = inner, k
        self.num_actions, self.num_observations, self.discount = inner.num_actions, inner.num_observations, inner.discount

    def initial_states(self, batch_size, initial_observations):
        return self.inner.initial_states(batch_size, initial_observations)

    def update(self, states, actions, observations):
        return self.inner.update(states, actions, observations)

    @torch.no_grad()
    def expand(self, states) -> Expansion:
        e = self.inner.expand(states)
        if self.k == 1.0:
            return e
        probs, rewards = e.observation_probs.clone(), e.rewards.clone()
        p_o = probs[:, FWD, GAME_OVER]
        inflated = (self.k * p_o).clamp(max=1.0)
        rest_scale = torch.where(p_o < 1.0, (1.0 - inflated) / (1.0 - p_o).clamp(min=1e-12), torch.zeros_like(p_o))
        probs[:, FWD] = probs[:, FWD] * rest_scale.unsqueeze(-1)
        probs[:, FWD, GAME_OVER] = inflated
        p_r = (-(rewards[:, FWD] + 1.0) / DEATH).clamp(0.0, 1.0)
        rewards[:, FWD] = rewards[:, FWD] - DEATH * ((self.k * p_r).clamp(max=1.0) - p_r)
        return Expansion(rewards, probs, e.next_states, e.next_values)


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


def play(search_model):
    planner = PlanningAgent(
        search_model, BeliefTreeSearch(search_model, cfg.mcts.num_simulations, cfg.mcts.c_puct,
                                       cfg.mcts.dirichlet_alpha, dirichlet_epsilon=0.0,
                                       seed=stream_seed(settings.seed, 2), observation_prune=cfg.mcts.observation_prune),
        n, temperature=0.0, seed=stream_seed(settings.seed, 3), device=dev)
    episodes, _ = play_episodes(WumpusSimulator(world, n, cfg.env.max_steps, stream_seed(settings.seed, 1), dev),
                                Progress(planner))
    del planner                                   # BeliefTreeSearch keeps its last trees
    torch.cuda.empty_cache()
    return episodes


def outcome_line(name: str, episodes) -> tuple[torch.Tensor, dict]:
    returns = discounted_returns(episodes.rewards, GAMMA).double()
    out = {key: round(float(v), 4) for key, v in game_outcomes(episodes.rewards).items()}
    ended = (episodes.rewards.abs() >= 998).long().cumsum(1)
    live = (ended - (episodes.rewards.abs() >= 998).long()) == 0
    acts = episodes.actions[live]
    mix = {NAMES[a]: round(float((acts == a).float().mean()), 4) for a in range(len(NAMES))}
    print(f"  {name:22s}: return {returns.mean():7.1f} +- {returns.std() / np.sqrt(n):5.1f}  {out}\n"
          f"      live action mix {mix}", flush=True)
    return returns, out | {"mix": mix}


def paired(label: str, a: torch.Tensor, b: torch.Tensor):
    d = a - b
    print(f"      {label}: {d.mean():+7.1f} +- {1.96 * d.std() / np.sqrt(len(d)):5.1f}", flush=True)


def load_networks(saved: Path):
    states = torch.load(saved, map_location=dev)
    if set(states) != set(run.networks):
        raise ValueError(f"saved networks {sorted(states)} do not match the run's {sorted(run.networks)}.")
    for name, net in run.networks.items():
        net.load_state_dict(states[name])


print(f"== {run_dir.name}: k = {k:g}, sets {[p.stem for p in saved_sets]}, {n} games", flush=True)
if smoke:
    # Identity at k = 1 on real expansions, then the margin must reach decisions (fewer live FORWARDs at k).
    load_networks(saved_sets[0])
    probe_states = model.initial_states(8, torch.zeros(8, dtype=torch.int64, device=dev))
    a, b = model.expand(probe_states), PessimisticModel(model, 1.0).expand(probe_states)
    if not (torch.equal(a.rewards, b.rewards) and torch.equal(a.observation_probs, b.observation_probs)):
        raise RuntimeError("PessimisticModel(k = 1) is not the identity.")
    c = PessimisticModel(model, k).expand(probe_states)
    if not torch.allclose(c.observation_probs.sum(-1), torch.ones_like(c.observation_probs.sum(-1)), atol=1e-5):
        raise RuntimeError("Inflated observation probabilities do not sum to 1.")
    print(f"  identity at k = 1 holds; start-state P(GAME_OVER | FORWARD) {a.observation_probs[0, FWD, GAME_OVER]:.4f} "
          f"-> {c.observation_probs[0, FWD, GAME_OVER]:.4f}, R(FORWARD) {a.rewards[0, FWD]:.2f} -> {c.rewards[0, FWD]:.2f}",
          flush=True)
    base, _ = outcome_line(f"{saved_sets[0].stem} k=1", play(model))
    returns, out = outcome_line(f"{saved_sets[0].stem} k={k:g}", play(PessimisticModel(model, k)))
    paired("vs k = 1", returns, base)
    print("SMOKE DONE", flush=True)
    sys.exit(0)

summary = {}
control_base = None
for saved in saved_sets:
    load_networks(saved)
    base = torch.from_numpy(np.load(saved.with_suffix(".returns.npy"))).to(dev).double()
    control_base = base if control_base is None else control_base
    returns, out = outcome_line(f"{saved.stem} k={k:g}", play(PessimisticModel(model, k)))
    paired(f"vs {saved.stem} k = 1", returns, base)
    paired(f"vs {saved_sets[0].stem} k = 1", returns, control_base)
    np.save(saved.with_suffix(f".pess{k:g}.returns.npy"), returns.cpu().numpy())
    summary[saved.stem] = out | {"mean": float(returns.mean()), "vs_self_k1": float((returns - base).mean()),
                                 "vs_first_k1": float((returns - control_base).mean())}
(run_dir / "analysis" / f"pessimism_k{k:g}.json").write_text(json.dumps(summary, indent=2))
print("DONE", flush=True)
