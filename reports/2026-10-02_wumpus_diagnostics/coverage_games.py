# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/coverage_games.py
# ==============================================================================
# COVERAGE SCREEN: DOES DATA AT THE DECLINED "TURN, THEN STEP INTO THE NEW CELL" STATES REPAIR THE MODEL?
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-19, user's choice after feature_localise.py, HANDOFF 6.2 Step D read MIXED / mostly SHIFT):
#    - On 5 of 5 seeds P(gold ahead) and "cell ahead unvisited" decode on ordinary buffer states but not at the
#      turned latents of the declined states (gold under-rated ~2x), and the composition is not the failure.
#      Conjecture (HANDOFF 1b, the feedback loop): the agent declines "turn towards the unvisited safe cell, then
#      step in", so its data rarely holds that transition from those states, the filter/model never learn what
#      it reveals, and the backup misses the value of information. This is the objective-mismatch / compounding-
#      error view of model learning (Lambert et al. 2020; Ross & Bagnell 2012 for model-based DAgger: a model is
#      only accurate on the state-action distribution it is trained on, so the data must cover the planner's
#      counterfactual choices).
#    - The screen is an UPPER BOUND on coverage: the data come from a privileged override (the exact-belief
#      heuristic picks the turn), never an agent. If the maximal dose does not repair the turned latents and
#      the play, coverage is refuted as the lever; if it does, a non-privileged coverage mechanism is next.
#
# 2. Method:
#    - This script (usage: uv run python <this> <run dir> <cover|control|own> <out .pt> [<batches>=4] [<games>=512]): 4 x 512
#      games with the run's best.pt greedy planner (Dirichlet off, temperature 0) and a shadow exact-belief
#      heuristic (WumpusReferenceAgent, risk 0.4). Simulator and planner streams stream_seed(COVERAGE_BASE, k),
#      disjoint from the analysis games (settings.seed) and identical for both arms (paired data).
#        cover:   at every declined distance-2 near-turn state (q_localise_diag.py's selection: the heuristic turns
#                 towards a provably safe unvisited cell or known gold two actions away, the planner does something
#                 else) the heuristic's turn is played, and FORWARD is forced on the next step. Elsewhere the
#                 planner acts. Dose: every declined state (the maximal dose; a dose-response comes later).
#        control: the same games with the planner only (the same amount of extra data from the agent's policy).
#      Output: an EpisodeBatch-shaped dict (actions, observations, rewards, initial_observations; (G, T)) plus the
#      override count and outcomes.
#    - Retrain (objective_ab.py section 2j, --extra-games): every network from scratch, the run's own config,
#      6400 updates uniform over latest.pt's buffer plus the arm's extra games, --save.
#    - Readout (i), the mechanism, at the declined states of Step D: feature_localise.py --networks <saved .pt>
#      (the same states from the same best.pt replay; latents recomputed from the histories with the retrained
#      filter; probes fitted on the same base-buffer slots for every arm, so the probe data do not differ).
#      Measures at (c): joint-head glitter ratio, joint-head TV median, q_localise's decision share M (share of
#      states with Q_mac > Q2(chosen)), probe gold ratio and unvisited accuracy.
#    - Readout (ii), play: retrain_play.py <run> <control .pt> <cover .pt> on the 512 analysis games, paired per
#      game, cover vs control.
#
# 3. Pre-registered readout (written 2026-10-19 before any run; runs/sweeps/wumpus_retrain seeds 0-4):
#    - MECHANISM PASS: cover - control at (c): joint-head glitter ratio >= +0.15 AND decision share M >= +10
#      points, on >= 4 of 5 seeds.
#    - PLAY PASS: cover - control paired gain, mean over the 5 seeds >= +50 with the Student-t 95% interval over
#      seeds above 0; unfinished rate down on >= 4 of 5 seeds; mean death rate up by at most 2 points.
#    - Both pass -> coverage is the lever: design a non-privileged coverage mechanism (planning-time optimism or
#      novelty for unvisited cells, or exploration over turn-then-step pairs) and screen it the same way.
#      MECHANISM pass, PLAY fail -> the repaired model does not reach the decisions; report and localise again
#      (q_localise on the cover networks). MECHANISM fail -> coverage at the maximal dose does not repair the
#      shift; the representation branch (D3, HANDOFF 1b candidate 1) next.
#    - Also reported: control vs best.pt (the retrain noise floor, -39 to +58 in objective_ab.py 2i) and the
#      cover arm's probe accuracies on the base buffer (does coverage hurt the latent elsewhere?).
#
# 4. Result 2026-10-20 (runs/wumpus/coverage: compare.log, play_seed*.log, fl_*_seed*.log; every best.pt replay
#    = report.json). **Readout: MECHANISM PASS (4 of 5), PLAY FAIL.**
#    | Seed | glitter ratio ctl -> cov | share M ctl -> cov | return ctl / cov | cov - ctl (95%) | deaths ctl -> cov | unfinished ctl -> cov |
#    | 0    | 0.45 -> 0.77             | 26 -> 58%          | 256 / 66         | -190 +- 59      | 4.5 -> 27.9%      | 59.4 -> 34.2%         |
#    | 1    | 0.39 -> 0.61             | 42 -> 66%          | 65 / 62          | -3 +- 62        | 3.1 -> 33.0%      | 81.3 -> 26.4%         |
#    | 2    | 0.33 -> 0.59             | 42 -> 55%          | 219 / -121       | -339 +- 59      | 5.5 -> 41.6%      | 61.5 -> 29.1%         |
#    | 3    | 0.44 -> 0.58             | 46 -> 47% (fail)   | 160 / -65        | -225 +- 63      | 19.0 -> 43.0%     | 42.2 -> 21.3%         |
#    | 4    | 0.60 -> 0.84             | 50 -> 71%          | 307 / 95         | -212 +- 58      | 2.7 -> 27.3%      | 57.0 -> 33.4%         |
#    - Play: mean cover - control -194, t95 [-345, -43] over seeds (significantly WORSE); unfinished down on 5 of 5
#      (-31 points on average) but deaths up +28 points (5 of 5) and wins only +4: the steps freed from idling end
#      in deaths, not in gold. Control vs best.pt (the retrain noise floor): +19 / -92 / -14 / -15 / +107.
#    - Mechanism: the coverage data repair the turned-latent model (glitter ratio +0.14 to +0.32, TV down), so the
#      Step D shift is a data-coverage effect, as the feedback-loop conjecture says. The probe on base-buffer
#      states mostly does not move (unvisited at (a) drops 0.955 -> 0.851 on seed 2).
#    - Per section 3 the branch is "localise again". The failure is deaths, not idling, so the localisation is
#      risk_localise.py (selection bias vs knowingly taken risk) rather than q_localise; the forced steps
#      themselves were provably safe (exact danger 0; every death in both arms is a FORWARD with exact danger > 0).
#      Coverage that comes from a privileged, safety-filtered teacher is not a usable lever as it stands; section 5
#      tests coverage from the agent's own, unfiltered choices.
#
# 5. Arm "own" (2026-10-19 night; my call under research autonomy, user away): coverage from the agent's OWN
#    choices, no privilege.
#    - Why: on seeds 0-1 the cover arm cures most idling (unfinished 59 -> 34% and 81 -> 26%; wins up 2 and 25
#      points) but deaths explode (4.5 -> 28%, 3.1 -> 33%). Every forced step was provably safe, so the cover data
#      are a safety-SELECTED sample of "turn, step into a new cell": ~2700 survivals per 2048 games and no deaths.
#      A latent that separates safe from risky new cells only partly then learns "new cells are safe" (the risk_
#      localise.py smoke: phantom risk at provably safe new cells 0.043 -> 0.005; the risk of chosen risky FORWARDs
#      rated at ~3% of exact). A non-privileged mechanism has no safety filter: its trials include the risky ones
#      the agent misjudges, and their deaths are in its data (model-based DAgger, Ross & Bagnell 2012; temporally
#      extended exploration, Dabney et al. 2021 "Temporally-Extended epsilon-Greedy Exploration").
#    - Rule: at every decision where the greedy planner's action is not FORWARD, with probability OWN_EPS = 0.2
#      (own generator, stream_seed(COVERAGE_BASE, 300 + k)) play the turn whose root Q is higher (the agent's own
#      estimate), then FORWARD on the next step. Nothing privileged chooses the action; the shadow heuristic's exact
#      belief only COUNTS the dose (overrides whose step enters an unvisited cell, enters a cell with exact danger
#      > 0, or starts at a declined distance-2 state).
#    - Pre-registered readout (written before the run; own vs the SAME control games and control retrains, seeds
#      0-4, retrain/feature_localise/retrain_play exactly as for cover):
#        PLAY PASS and MECHANISM PASS: the criteria of section 3, with own in place of cover.
#        RISK: risk_localise.py on own: the calibration ratio on risky unvisited-ahead states >= 0.85 x control's on
#              >= 4 of 5 seeds (no selection bias), and death rate up by at most 2 points (as in PLAY PASS).
#      PLAY PASS -> build the rule into collection as a default-off agent option (Tiger fingerprint) and train 5
#      seeds with it: the honest test, with the data inside the loop. MECHANISM fail with a small new-cell dose ->
#      untargeted trials do not reach the declined states; targeting needs a novelty signal the latent does not
#      carry (Step D's visited deficit): the representation branch. MECHANISM pass, PLAY fail with RISK pass ->
#      localise again with q_localise on the own networks.
#    - Result 2026-10-21 (runs/wumpus/coverage: compare_own.log, play_own_seed*.log, risk_own_seed*.log; own_chain.log
#      had a pause for the held-out sweep's VRAM, with no effect on results). **Readout: PLAY FAIL, MECHANISM FAIL (0 of
#      5), RISK PASS (5 of 5).**
#      | Seed | dose/game: unvisited, risky, declined | glitter ctl -> own | share M ctl -> own | own - ctl (95%) | own - best.pt | deaths ctl -> own | unfinished ctl -> own | risk ratio own/ctl |
#      | 0    | 1.3, 0.55, 0.34                       | 0.45 -> 0.53       | 26 -> 23%          | +30 +- 35       | +49           | 4.5 -> 3.5%       | 59.4 -> 58.2%         | 0.98               |
#      | 1    | 1.4, 0.49, 0.41                       | 0.39 -> 0.54       | 42 -> 48%          | +174 +- 40      | +82           | 3.1 -> 0.4%       | 81.3 -> 69.0%         | 1.10               |
#      | 2    | 1.4, 0.61, 0.39                       | 0.33 -> 0.37       | 42 -> 27%          | +68 +- 31       | +53           | 5.5 -> 0.4%       | 61.5 -> 63.3%         | 1.24               |
#      | 3    | 1.3, 0.63, 0.39                       | 0.44 -> 0.68       | 46 -> 51%          | +141 +- 48      | +126          | 19.0 -> 0.4%      | 42.2 -> 62.5%         | 1.24               |
#      | 4    | 1.7, 0.71, 0.50                       | 0.60 -> 0.60       | 50 -> 52%          | -87 +- 41       | +20           | 2.7 -> 16.0%      | 57.0 -> 42.2%         | 0.85               |
#      (Dose per game from batch 3 of each seed's games log; cover had ~1.3 declined-state entries per game.)
#      - Play: own - control mean +65, t95 [-62, +192] over seeds; unfinished down on 3 of 5 only; deaths -2.8 points on
#        average. FAIL by the rule. Exploratory, not registered: own - best.pt +66, t95 [+16, +116], positive on 5 of 5;
#        control - best.pt +1 [-88, +90]. The gain comes from safety (deaths 0.4% on seeds 1-3), not from idling.
#      - Mechanism: unfiltered trials reach the declined states ~3x less often than the privileged arm and do not
#        repair the turned-latent prediction (glitter +0.01 to +0.24; share M -15 to +7 points).
#      - Risk: no selection bias (calibration 0.85-1.24x control's). Seed 4 is the exception in play: its deaths
#        rise, and its fatal steps are mostly not under-rated (37%); those risks were taken knowingly.
#      - Registered branch: MECHANISM fail with a small declined-state dose -> "targeting needs a novelty signal the
#        latent does not carry: the representation branch". Its premise is contradicted: "cell ahead unvisited" decodes
#        at 0.97-0.98 on buffer states under the own networks. The direction is re-argued in HANDOFF section 6.2 (2026-10-21).
# 6. Arm "union" (2026-10-21, HANDOFF 6.2b E0, user-approved): the DATA UPPER BOUND of both coverage deficits.
#    - Why: cover repairs the value-of-information deficit (V) but its safety-filtered data deepen the risk deficit
#      (R); own keeps the risk calibrated but is too diluted to repair (V) (C10). Is the union of both data enough
#      to get both at once? If not even the union helps, no collection rule that only reshapes these data will.
#    - Method: no new games. objective_ab.py 2j with BOTH extra files (games_cover_seed<s>.pt games_own_seed<s>.pt,
#      4096 extra games; twice either arm's extra data, which favours the union and is acceptable for an upper
#      bound), the same retrain settings, then feature_localise.py --networks, retrain_play.py control vs union,
#      risk_localise.py, exactly as for the other arms (runs/wumpus/e0_chain.sh).
#    - Pre-registered readout (written before any union run; seeds 0-4; against the same control):
#        PASS: union - control mean >= +50 with the t95 interval over seeds above 0, unfinished below control's on
#              >= 4 of 5 seeds, deaths at most control's + 2 points on >= 4 of 5 seeds.
#      PASS -> both deficits together are sufficient at this latent: build E2 (uncertainty-directed collection
#      plus pessimistic planning, HANDOFF 6.2b). FAIL with deaths up -> even mixed data leave an exploitable risk
#      error: pessimism (E0', pessimism_play.py) is required on top. FAIL with the idling kept -> data alone cannot
#      fix it at this latent: the representation branch (HANDOFF 1b, candidate 1).
#      Also reported: glitter ratio and decision share M (MECHANISM as in section 3) and the risk calibration ratio
#      (RISK as in section 5), so the two deficits are read separately.
#    - Result 2026-10-21 (runs/wumpus/coverage: play_union_seed*.log, compare_union.log, risk_union_seed*.log; e0_chain.log;
#      every best.pt replay = report.json). **Readout: FAIL (deaths up).**
#      | Seed | union - ctl (95%) | deaths ctl -> union | unfinished ctl -> union | glitter ctl -> union | share M ctl -> union | risk ratio ctl / union | at chosen risky FWD ctl / union |
#      | 0    | -90 +- 56         | 4.5 -> 16.4%        | 59.4 -> 46.3%           | 0.45 -> 0.69         | 26 -> 44%            | 0.995 / 0.875          | 0.155 / 0.051                   |
#      | 1    | +135 +- 47        | 3.1 -> 5.3%         | 81.3 -> 65.0%           | 0.39 -> 0.55         | 42 -> 60%            | 0.820 / 0.834          | 0.034 / 0.044                   |
#      | 2    | -21 +- 46         | 5.5 -> 15.6%        | 61.5 -> 45.1%           | 0.33 -> 0.59         | 42 -> 52%            | 0.828 / 0.847          | 0.027 / 0.028                   |
#      | 3    | +77 +- 55         | 19.0 -> 10.9%       | 42.2 -> 49.8%           | 0.44 -> 0.66         | 46 -> 48%            | 0.704 / 0.942          | 0.464 / 0.043                   |
#      | 4    | -161 +- 53        | 2.7 -> 20.1%        | 57.0 -> 42.0%           | 0.60 -> 0.71         | 50 -> 74%            | 1.007 / 0.907          | 0.046 / 0.024                   |
#      - Play: union - control -12, t95 [-162, +138] over seeds; unfinished down on 4 of 5 (-10 points on average);
#        deaths up by > 2 points on 4 of 5 (5.3% -> 13.7% on average).
#      - Mechanism: the turned-latent prediction is repaired on 5 of 5 by the glitter ratio (+0.11 to +0.26), the
#        decision share bar is met on 2 of 5 only. Risk: the AVERAGE calibration on risky new cells stays near
#        control's (0.83-0.94; cover alone 0.53-0.73), yet at the risky FORWARDs the planner chooses, the predicted
#        risk is still 2-5% of exact. The deaths are the optimizer's curse on a calibrated-on-average model.
#      - Registered branch "FAIL with deaths up": even mixed data leave an exploitable risk error, so pessimism is
#        required on top; but E0' (pessimism_play.py section 4) shows a uniform margin does not remove it either.
# ==============================================================================

import sys
import time
from collections import deque
from pathlib import Path

import torch

from ipomdp.agents import PlanningAgent, WumpusReferenceAgent
from ipomdp.agents.wumpus_reference import _ZERO, _ahead, _first_actions
from ipomdp.domain import WumpusSimulator, game_outcomes
from ipomdp.domain.wumpus import NUM_CELLS, WumpusAction
from ipomdp.experiments.runs import load_trained_run
from ipomdp.planning import BeliefTreeSearch
from ipomdp.training import play_episodes, stream_seed

COVERAGE_BASE = 20261020
OWN_EPS = 0.2
run_dir, arm, out_path = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
batches = int(sys.argv[4]) if len(sys.argv) > 4 else 4
GAMES = int(sys.argv[5]) if len(sys.argv) > 5 else 512
if arm not in ("cover", "control", "own"):
    raise ValueError(f"arm must be cover, control or own, not {arm}.")
dev = torch.device("cuda")
trained = load_trained_run(run_dir, dev)
world, run, cfg = trained.setup.domain, trained.run, trained.cfg
model = run.eval_agent.model
FWD, TURNS = int(WumpusAction.FORWARD), (int(WumpusAction.TURN_LEFT), int(WumpusAction.TURN_RIGHT))


def target_distance(cell: int, heading: int, safe: list[bool], is_target) -> int:
    """Fewest actions (FORWARD through safe cells, turns) until the hunter stands on a target cell (override_diag.py)."""
    if is_target(cell):
        return 0
    queue, seen = deque([(cell, heading, 0)]), {(cell, heading)}
    while queue:
        c, h, d = queue.popleft()
        ahead = _ahead(c, h)
        for nxt in ((ahead, h) if ahead is not None and safe[ahead] else None, (c, (h + 1) % 4), (c, (h + 3) % 4)):
            if nxt is None or nxt in seen:
                continue
            if nxt[0] != c and is_target(nxt[0]):
                return d + 1
            seen.add(nxt)
            queue.append((*nxt, d + 1))
    return 10 ** 6


class CoverageAgent:
    """The greedy planner; in the cover arm the declined distance-2 near-turn states play the turn, then FORWARD."""

    def __init__(self, planner: PlanningAgent, k: int):
        self.planner, self.batch_size = planner, GAMES
        self.shadow = WumpusReferenceAgent(world, GAMES, 0.4)
        self.force = [False] * GAMES
        self.overrides, self.t, self.start = 0, 0, time.time()
        self.generator = torch.Generator().manual_seed(stream_seed(COVERAGE_BASE, 300 + k))
        self.dose = {"unvisited": 0, "risky": 0, "declined": 0}         # own arm: privileged counts, never choices

    def reset(self, initial_observation):
        self.planner.reset(initial_observation)
        self.shadow.reset(initial_observation)
        self.force = [False] * GAMES

    def update(self, action, observation):
        self.planner.update(action, observation)
        self.shadow.update(action, observation)

    def act(self):
        s = self.shadow
        m = s._tracker.marginals(s._consistent, s._own)
        heuristic = s.act().tolist()
        chosen = self.planner.act().tolist()
        pit, wumpus, gold = m.pit.cpu().tolist(), m.wumpus.cpu().tolist(), m.gold.cpu().tolist()
        alive, done = m.wumpus_alive.tolist(), m.done.tolist()
        q = self.planner.planner.statistics.q_values
        draws = torch.rand(GAMES, generator=self.generator).tolist() if arm == "own" else None
        for b in range(GAMES):
            if self.force[b]:
                chosen[b], self.force[b] = FWD, False
                continue
            if arm == "own":
                if chosen[b] == FWD or draws[b] >= OWN_EPS:
                    continue
                turn = TURNS[0] if q[b, TURNS[0]] >= q[b, TURNS[1]] else TURNS[1]
                if not done[b]:
                    self.count_dose(b, turn, heuristic[b], pit[b], wumpus[b], gold[b], alive[b], int(m.cell[b]),
                                    int(m.heading[b]))
                chosen[b], self.force[b] = turn, True
                continue
            if arm == "control" or done[b] or heuristic[b] not in TURNS or chosen[b] == heuristic[b]:
                continue
            cell, heading = int(m.cell[b]), int(m.heading[b])
            if gold[b][cell] > 1.0 - _ZERO:
                continue
            danger = [pit[b][c] + (wumpus[b][c] if alive[b] else 0.0) for c in range(NUM_CELLS)]
            safe = [d < _ZERO for d in danger]
            reachable = {c for c, _ in _first_actions(cell, heading, safe)} | {cell}
            known = [c for c in reachable if gold[b][c] > 1.0 - _ZERO]
            goal = (lambda c: c in known) if known else (lambda c: safe[c] and not s._visited[b][c] and c in reachable)
            if not (known or any(goal(c) for c in range(NUM_CELLS))):
                continue
            if target_distance(cell, heading, safe, goal) != 2:
                continue
            chosen[b], self.force[b] = heuristic[b], True
            self.overrides += 1
        self.t += 1
        if self.t % 10 == 0:
            print(f"   [{time.strftime('%H:%M:%S')}] decision {self.t}/{cfg.env.max_steps}, {self.overrides} overrides"
                  + (f" (dose {self.dose})" if arm == "own" else "") + ", "
                  f"{time.time() - self.start:.0f}s", flush=True)
        return torch.tensor(chosen, dtype=torch.int64, device=dev)

    def count_dose(self, b, turn, heuristic, pit, wumpus, gold, alive, cell, heading):
        """Own arm: one live override, counted by what the exact belief says about it (reporting only)."""
        self.overrides += 1
        target = _ahead(cell, (heading + 1) % 4 if turn == int(WumpusAction.TURN_LEFT) else (heading + 3) % 4)
        if target is None:
            return
        self.dose["unvisited"] += int(not self.shadow._visited[b][target])
        self.dose["risky"] += int(pit[target] + (wumpus[target] if alive else 0.0) >= _ZERO)
        if heuristic == turn and gold[cell] <= 1.0 - _ZERO:
            danger = [pit[c] + (wumpus[c] if alive else 0.0) for c in range(NUM_CELLS)]
            safe = [d < _ZERO for d in danger]
            reachable = {c for c, _ in _first_actions(cell, heading, safe)} | {cell}
            known = [c for c in reachable if gold[c] > 1.0 - _ZERO]
            goal = (lambda c: c in known) if known else (lambda c: safe[c] and not self.shadow._visited[b][c]
                                                          and c in reachable)
            self.dose["declined"] += int(target_distance(cell, heading, safe, goal) == 2)


print(f"== {run_dir.name} best.pt (collection {trained.checkpoint_collection}), arm {arm}, {batches} x {GAMES} games",
      flush=True)
parts, overrides, dose = [], 0, {}
for k in range(batches):
    planner = PlanningAgent(
        model, BeliefTreeSearch(model, cfg.mcts.num_simulations, cfg.mcts.c_puct, cfg.mcts.dirichlet_alpha,
                                dirichlet_epsilon=0.0, seed=stream_seed(COVERAGE_BASE, 100 + k),
                                observation_prune=cfg.mcts.observation_prune),
        GAMES, temperature=0.0, seed=stream_seed(COVERAGE_BASE, 200 + k), device=dev)
    agent = CoverageAgent(planner, k)
    episodes, _ = play_episodes(WumpusSimulator(world, GAMES, cfg.env.max_steps, stream_seed(COVERAGE_BASE, k), dev),
                                agent)
    overrides += agent.overrides
    dose = {key: dose.get(key, 0) + v for key, v in agent.dose.items()}
    print(f"  batch {k}: {agent.overrides} overrides" + (f" (dose {agent.dose})" if arm == "own" else "") + "; outcomes "
          + str({key: round(float(v), 4) for key, v in game_outcomes(episodes.rewards).items()}), flush=True)
    parts.append(episodes)
    del planner, agent
    torch.cuda.empty_cache()
data = {key: torch.cat([getattr(p, key) for p in parts]).cpu()
        for key in ("actions", "observations", "rewards", "initial_observations")}
outcomes = {key: float(v) for key, v in game_outcomes(data["rewards"]).items()}
out_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(data | {"arm": arm, "overrides": overrides, "dose": dose, "outcomes": outcomes, "run": str(run_dir)},
           out_path)
print(f"  saved {out_path}: {data['actions'].shape[0]} games, {overrides} overrides, outcomes "
      + str({k: round(v, 4) for k, v in outcomes.items()}), flush=True)
print("DONE", flush=True)
