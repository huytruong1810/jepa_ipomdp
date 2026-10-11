# HANDOFF: JEPA I-POMDP (as of 2026-10-09 night, after the offline screens and the γ = 0.95 runs)

Read `CLAUDE.md` first: it covers the architecture, commands and conventions. This file covers goals, the user's rules, the current state, every result that matters, the next steps, the dead ends and the files touched. Each module starts with a `DESIGN DECISIONS & THEORETICAL FOUNDATIONS` header with the evidence behind its design; read it before changing that module. Every diagnostic script in `reports/` explains its question, usage and result in its header. The user's auto-memory (`~/.claude/projects/-home-andyj1810-projects-jepa-ipomdp/memory/`) repeats the standing rules of section 2.

## 0. Where we are

- **Canonical Tiger: done.** The learned planner matches the exact optimum over 5 seeds (gap −0.03 ± 0.06). A decoder baseline is indistinguishable from JEPA there.
- **Single-hunter 4x4 Wumpus World: the current agent works but is far from the reference.**
  - The agent: decoder representation, 64 updates per collection, ε-uniform exploration, greedy evaluation games stored in the buffer.
  - It earns **189 ± 62** per game over 3 seeds (185, 166 and 215) with 0.2–3.9% deaths.
  - A hand-written agent on the exact belief earns 573. About 70% of the learned agent's games end unfinished after 50 steps, against 29% for the heuristic.
- **What the latent encodes.** Its UMAP is organised first by the percept just received, and only weakly by position and heading. Probes recover cell and heading 55–60% of the time on random histories, against 99.9–100% for the same network trained on exact targets.
- **The main diagnosis of the last week.** The jointly trained filter and observation head **overfit** when they train too many times per game. They were calibrated in-sample and overconfident on fresh games, and the greedy planner exploited exactly those confident "safe" errors. Going back to 64 updates per collection fixed most deaths.
- **The reviewer meeting (2026-10-07, 3pm) is over.** The page is https://claude.ai/artifact/1m1jkSud5xbFCidGMkc6VN (section 3).
- **JEPA with the same recipe (2 seeds, finished 2026-10-07): no detectable difference from the decoder in return** (paired: −16 and +26). Its probes are slightly worse on random histories (section 5.3).
- **Why games end unfinished, diagnosed 2026-10-07/08** (section 5.6):
  - the greedy planner idles in front of provably safe unvisited cells;
  - the latent DOES encode progress (probes 93–97%);
  - the bootstrapped **value** is self-consistent but ungrounded on idle chains (V̂ ≈ 650 against a realised −22);
  - a value fitted to real returns of an *exploring* policy (the heuristic's games) on the same frozen latents ranks "step forward" first.
- **Two value remedies were tried and both failed (3 seeds each, section 5.6):**
  - the time-aware (finite-horizon) value: 200 / 108 / 143 against 186 / 166 / 215;
  - n-step returns of the agent's own greedy games: 88 / 93 / 126.
  Both options are in the code, default off, and unused.
- **Offline screens of value targets (2026-10-09, section 5.6), on the current agent's frozen latents:**
  - the optimistic target max(V̂, G) fails on all 3 seeds;
  - the agent's own Monte Carlo returns are inconsistent;
  - **the heuristic-return result (M4) is not robust**: the FORWARD preference depends on the fit protocol and initialisation, and early-stopped fits fail seed 2 even with 2,048 games. That is extrapolation to idle states the heuristic never visits.
  - No value-target remedy currently has offline support.
- **Lower γ (0.95, `1c05c16`; 3 seeds, finished 2026-10-09 20:44) did not fix the idling** (section 5.6, remedy 3).
  - Unfinished share: 66.8 / 62.5 / 67.0% against 66.6 / 70.3 / 70.7%. The paired mean change is −3.8 points, 95% interval ±9.9.
  - Win rate: 30.9 / 34.0 / 26.0% against 29.5 / 26.4 / 29.1%.
  - Idle V̂ halved (650 → 200–330), and FORWARD at safe cells in unfinished games rose on 2 of 3 seeds.
- **The user accepted the idling plateau (2026-10-09 night)** and chose latent work plus the recipe ablations. Both are running:
  - **Horizon A/B, finished: longer horizons do not help, and k = 20 hurts** (`runs/wumpus/horizon_ab/summary.txt`, `objective_ab.py` section 2c).
    - k20 − k5: cell −3.8 ± 5.2 points (lower on 3 of 3 seeds), heading −13 ± 24 (lower on 3 of 3). k10 lies in between.
    - Heading accuracy is still rising at 6,400 updates for k = 5 (one seed reaches 84%), but more updates overfit the observation head (section 5.5).
    - Networks are saved as `obj_*.pt`.
  - **Decoupled planning head A/B, finished** (`runs/wumpus/planning_head_ab/`, `objective_ab.py` section 2d; 3 seeds, 25,600 updates).
    - **The latent improves with longer training:** cell +3 to +5 points and heading +5 to +8 by 12,800 updates; one seed reaches 86% / 91%.
    - **The joint head overfits** (held-out NLL 0.23 → 0.32–0.39).
    - **A fresh head early-stopped on NLL does not overfit, but is worse on confident "safe" misses** (3 of 3 seeds; paired +0.025 ± 0.062). It is no better on danger error. Decoupling alone is not a clear win.
  - **Detached planning head at 128 updates per collection (`8b3d621`): failed.**
    - Seeds 0 and 1 returned 67 and −45, against 186 and 166 for the current agent. Deaths were 8.4% and 28.7%.
    - In-training evaluations show about 30% deaths on all 3 seeds, so seed 2 was stopped by hand at collection 55.
    - The online latent did not improve: random-history probes for cell are 52/53% and heading 50/49%, against 56/59%. The offline latent gain did not transfer.
    - By the decision rule, the 128-update joint-head control was not run.
  - **Ablations, finished** (section 5.3): storing the greedy games matters (−110 ± 97 without it, deaths 6–18%); ε does not (+8.8 ± 110).

## 1. Goals

Three research contributions:
1. **Belief filter.** A recurrent JEPA belief filter (no observation decoder) learned while the agent learns the dynamics.
2. **Planning.** Belief-tree search over that filter, for POMDPs now and I-POMDPs later.
3. **Interpretability.** A map from the latent to the exact (later interactive) belief, with quantified error bounds.

Stages:
1. Canonical Tiger matches the exact solver: **done** (section 4).
2. Baseline comparison: **done** (decoder world model, section 4b).
3. **Single-hunter Wumpus World, now.** The agent should learn to play well, and the latent should be shown against the exact belief. Multi-agent behaviour is explicitly **not** wanted yet.
4. Later: multi-agent domains and I-POMDP levels.

**Honest status of "optimality"** (the user asked): the evidence is empirical and Tiger-only. There is no proof or convergence guarantee. Wumpus has no optimum to compare against.

**Tension to keep in mind:** contribution (1) is the JEPA filter, but the working Wumpus agent uses the decoder representation, because JEPA learned the danger ahead worse in the offline A/B (section 5.5). With the fixed recipe JEPA plays as well as the decoder (2 seeds, section 5.3), so contribution (1) is viable on Wumpus; its latent is a little less position- and heading-aware.

## 2. Rules the user set (follow them)

**Scope**
- Never add non-canonical elements to canonical Tiger (`tiger.95.POMDP`, γ = 0.95). Tiger must stay bit-reproducible.
- Every change needs a small-scale check before a larger run.
- Wumpus has no exact solver (user decision). Do not test multi-agent behaviour yet.
- **The known-layout value (936) is not a meaningful comparison** (user, 2026-10-07: "we do not do MDP here"). Compare against random play and the exact-belief heuristic only.

**Code**
- No fallback paths, no deprecated code, no backward compatibility, no schema versions. Fail loudly.
- Code should read as if written from scratch, with verbose design-decision comments; DDD/DRY/KISS/SOLID.
- Every source file starts with `# ABSOLUTE PATH:` and a design-decisions header. Exception: Hydra files must have `# @package _global_` on the very first line, above the header.
- When a config key is added, add it with its old value to the saved `.hydra/config.yaml` of every existing run so they still reload. Done for `env.discount` (0.99), `representation`, `observation_prune`, `observation_horizon`, `agent.epsilon`/`epsilon_min`, `training.store_eval_episodes`, `model.time_aware_value` (false) and `model.value_nstep`/`value_nstep_mix` (0 / 0.5).
- A new option must default to off and leave Tiger bit-identical. Prove it with a fingerprint: a small CPU `TrainingRun` (the test `_config`), 4 collections with evaluations, then compare metrics and every parameter sum against the previous commit in a `git worktree`. This was done for ε, `store_eval_episodes`, `time_aware_value` and `value_nstep`.
  - Fingerprint recipe: a script that imports `_config` and `tiger_setup` from `tests/test_training_run.py`, runs both representations for 4 collections plus evaluations, and prints `repr(metrics)` and every parameter sum. Run it in the worktree with `PYTHONPATH=<worktree>/src` and a symlinked `.venv`, then `cmp` the two outputs. Check that the worktree run imported the worktree's `src`.

**Architecture and efficiency** (user guidance of 2026-10-03 and 2026-10-04, also in memory)
- **The network architecture must never be the bottleneck.** When results look wrong, A/B-test state-of-the-art upgrades: a Transformer over history tokens with a KV-cache, residuals, (flash) attention, MoE, GQA, MLA. Use good statistics (several seeds, paired comparisons, 95% intervals).
- **Always use the most memory- and runtime-efficient practice that leaves results unchanged.** A bit-identical speed-up can go in once a fingerprint confirms it. A summation-order change needs the user's approval. Never change code under runs in flight; record which commit a run used.

**Workflow**
- Commit on `main` (remote `github.com/huytruong1810/jepa_ipomdp`). **Ask before every push.**
- **Since 2026-10-10 the user pre-authorises running the recommended experiments without asking** ("have a good taste for research; don't waste compute"). Still ask before pushes and before any change to canonical Tiger.
- **Ask the user about every research or design decision that is theirs**, with a recommended option first. They usually pick it, but not always. On 2026-10-06/07 they pre-authorised the recommended options while away for the reviewer deadline. **That authorisation has expired: ask again.**
- State results plainly, including unfavourable ones and your own mistakes.
- **Reviewer pages:** minimal, figure-first, honest.
  - The user cut the 2026-10-07 page to: domain, returns, curves, latent UMAP, a synced game player and probes.
  - No previous agents, no diagnosis or status sections, no known-layout bound and no subtext under the title.
  - The owner may edit pages live: read them with the Artifact tool before republishing.

**Machine** (WSL2 Ubuntu; RTX 5080 Laptop with 16 GB; 24 GB RAM; `uv`, Python 3.12, cu128)
- `~/projects/ipomcp` often runs heavy CPU jobs here (load 10–24 on 24 cores, ≥ 5 workers at about 1.45–1.75 GB RAM each). Check `uptime`, `free -m` and `ps -eo pid,rss,args --sort=-rss | head` before launching anything.
- **The tree search is CPU-bound.** Under ipomcp load a Wumpus run slowed from about 1.5–2 min to about 9 min per collection, which killed the JEPA runs on 2026-10-07. A full 100-collection Wumpus run takes about 2.5–3.2 h alone, and about 4 h with two in parallel.
- **RAM is the binding limit for parallel runs.** A Wumpus `main.py` holds about 2.6 GB, a Tiger acceptance run about 1.8 GB, and an `objective_ab.py` run about 2.1 GB plus the pool on the GPU. Six A/B runs plus ipomcp exhausted RAM on 2026-10-06, and Claude Code reaped a background watcher. Run at most 2–3 heavy jobs while ipomcp is busy.
- **Six parallel `objective_ab.py` runs saturate the GPU** (0.8 s per update each against 0.1 s alone), so running them in parallel gains nothing. Run 2–3 at a time.
- Detach long runs with `setsid nohup ... > log 2>&1 < /dev/null & disown`, then confirm the python process is alive. For multi-step overnight work, put the whole pipeline in one detached shell script (as `runs/wumpus/reviewer_chain.sh`); it survives reaped watchers.
- **Waiting on runs:**
  - Never use `pgrep -f`/`pkill -f "<pattern>"` with a pattern that appears in your own command line: it matches the shell itself. On 2026-10-06 `pkill -f run.sh` killed the calling shell.
  - Wait on PIDs (`while kill -0 <pid>; do sleep 60; done`) or match the python process exactly (`ps -eo pid,args | grep "^ *[0-9]* [^ ]*python[^ ]* main.py"`).
  - Background watchers stop after at most 2 h, and the memory reaper may kill them earlier: re-arm them.
- Cap CPU threads in parallel runs (`OMP_NUM_THREADS=2–4`). Tests pin torch to one thread (`tests/conftest.py`).
- **Tools outside the project** (deliberately not dependencies; they live in the session scratchpad and disappear with it):
  - UMAP: `uv venv <env>; VIRTUAL_ENV=<env> uv pip install umap-learn`.
  - Page screenshots: Playwright Chromium in a scratch venv. The browser binary is cached in `~/.cache/ms-playwright`. It needs `libnspr4`, `libnss3` and `libasound2`, which are not installed: `apt-get download` them (no sudo), unpack with `dpkg-deb -x` and set `LD_LIBRARY_PATH`.

## 3. Current state

**Git**
- **`origin/main` is at `9a9b726`.** Everything after it (`61991c7` … HEAD) is local and unpushed: **ask before pushing.** The previous HANDOFF wrongly said `61991c7` was pushed.
- The working tree is clean after the HANDOFF commit. Nothing is running.

**Tests**
- The fast suite passes: 159 tests, about 2 min.
- **The slow suite passed 7/7 on 2026-10-05**, at `cd9d116` plus the stale-test fix `d4911a8`. Later source changes (ε, `store_eval_episodes`, `time_aware_value`, `value_nstep`) are default-off, fingerprinted and covered by fast tests. Rerunning the slow suite once at HEAD is still advisable (about 50 min).
- **Checkpoints from before `52ac3fc` cannot be resumed:** their buffer state has no `greedy` flags (no back-compat, by rule). They still load for analysis (`load_trained_run` reads networks only).
- **The Tiger acceptance test now trains 6000 updates.** At 3000 the value gate (mean |V − V*| < 8) failed at 3–5 of 9 seeds. The two worst seeds drop to 5.3 and 1.1 at 6000. The seed table is in the test header; the logs are in `runs/tiger_acceptance_seeds/`.
- The input-network hoist (`6af8f8f`) stays: its paired effect on the value error is +4.2 ± 6.2, not significant.

**Runs** (`runs/` is untracked)

| Path | What |
|---|---|
| `runs/sweeps/polyak` | Tiger JEPA, 5 seeds: **the Tiger numbers to cite** |
| `runs/sweeps/decoder` | Tiger decoder baseline, 5 seeds |
| `runs/wumpus/u64_store_eval_eps_decoder_k5_seed{0,1,2}` | **The current Wumpus agent**, analysed (`analysis/report.json`); seed 0 also has `analysis/latents_umap_input.npz` |
| `runs/wumpus/u64_store_eval_eps_jepa_k5_seed{0,1}` | **JEPA with the same recipe, finished and analysed** (`analysis/report.json`). Collections 1–40 were trained at `f5740e4`; collections 41–100 were resumed bit-exactly on 2026-10-07 at `2a6e326` (no source change) by `runs/wumpus/jepa_resume_chain.sh` |
| `runs/wumpus/u64_store_eval_eps_decoder_k5_seed*.{unfinished_diag,value_chain_diag}.log` | Unfinished-games and value-chain diagnoses of the current agent (section 5.6) |
| `runs/wumpus/u64_store_eval_eps_decoder_k5_time_seed{0,1,2}` | Current recipe + `model.time_aware_value=true` (`8ca58d0`), analysed, plus both diagnostics (time seed 0 only for value-chain): **failed** (section 5.6) |
| `runs/wumpus/u64_store_eval_eps_decoder_k5_nstep10_seed{0,1,2}` | Current recipe + `model.value_nstep=10` (`52ac3fc`), analysed, plus both diagnostics: **failed** (section 5.6) |
| `runs/wumpus/u64_store_eval_eps_decoder_k5_seed*.{grounding_screen,teacher_robustness}.log`, `{screen,teacher}_chain.sh` | The offline value-target screens of 2026-10-09 (section 5.6) |
| `runs/wumpus/u64_store_eval_eps_decoder_k5_gamma95_seed{0,1,2}`, `gamma95_chain.sh` (+ `.log`) | Current recipe + `env.discount=0.95` (`1c05c16`), analysed, plus both diagnostics: **no significant gain** (section 5.6) |
| `runs/wumpus/u128_store_eval_eps_decoder_k5_planhead_seed{0,1,2}`, `planning_head_agent_chain.sh` | Detached planning head, 128 updates: **failed** (seed 2 stopped at collection 55) |
| `runs/wumpus/smoke_gamma95` | 6-collection small check at γ = 0.95 (clean) |
| `runs/wumpus/smoke_time_aware`, `smoke_nstep` | 6-collection small checks of the two options |
| `runs/wumpus/{jepa_resume,time_aware,value_chain,nstep}_chain.sh` (+ `.log`) | The detached pipelines of 2026-10-07/08 |
| `runs/wumpus/store_eval_eps_decoder_k5_u256_seed0`, `eps_decoder_k5_u256_seed0` | The two failed 256-update runs (−84 and −283) |
| `runs/wumpus/north50_seed0`, `north50_decoder_seed0` | The first JEPA and decoder agents (110 and 85) |
| `runs/wumpus/overfit_ab/` | Offline overfitting A/B logs (stopped at about 12,800 updates) |
| `runs/wumpus/supervised_filter/pools.pt` | Cached game pool: 4096 reference + 4096 random training games and 1024 held-out games, with exact targets |
| `runs/wumpus/objective_ab*/`, `offpolicy_reference_seed0`, `head_coverage.log`, `full_seed0*`, `smoke_seed0` | Older diagnostics (section 5.5) |
| `runs/wumpus/reviewer_chain.sh`, `jepa_chain.sh` | The overnight pipeline and the JEPA launch-plus-analysis script |
| `runs/tiger_acceptance_seeds/` | Value-gate seed study logs |

**Reviewer pages**

| Page | Content | Sources |
|---|---|---|
| Tiger (2026-09-30): https://claude.ai/artifact/Rboz6xV7xA7Ejrjomy4As5 | Tiger only; the method diagrams | `reports/2026-09-30_reviewer_page/` |
| Wumpus (2026-10-07, version 6): https://claude.ai/artifact/1m1jkSud5xbFCidGMkc6VN | Domain and rules (the heuristic explained); returns (random, current agent over 3 seeds, heuristic); training curves; latent UMAP with selectable colourings; a synced game player (true map with trail, UMAP path and exact vs decoded belief, for a win, an unfinished game and a death); probes vs the supervised bound | `reports/2026-10-07_wumpus_reviewer_page/` |

- Rebuilding the Wumpus page data:
  - `scripts/export.py` writes `data.json`. The full export takes about 15–25 min because of the calibration step. `export.py games …` refreshes the example games only; `export.py add <data.json> <runs…>` merges new runs' curves and reports.
  - `scripts/dump_latents.py` then `scripts/umap_latents.py` (in the UMAP environment) write `umap.json`. The example games' latents are placed with the fitted map's `transform()`.
  - `scripts/page_meta.json` holds labels, the main-run prefix and `jepa_prefix`.
- The page already shows a "JEPA, same recipe" row and curves once the JEPA runs exist in `data.json` (merge with `export.py add`). The data still contains a now-unused `calibration` key and old runs' reports and curves, which the page ignores.
- Publish with the Artifact tool, `files` = `data.json` and `umap.json`, to the same URL.

## 4. Results: canonical Tiger (`runs/sweeps/polyak`, 5 seeds)

All agents play the same 512 held-out episodes. Gaps are paired with the optimal agent (Student-t 95% intervals over seeds). The optimal agent earns 18.48 on these episodes; V*(b₀) = 19.37.

| Agent | Gap to optimal |
|---|---|
| **Learned planner** | **−0.03 ± 0.06** (per seed −0.11, −0.02, −0.02, 0.00, +0.00) |
| Decoded belief, MLP probe → exact Q* | −0.23 ± 0.53 |
| Decoded belief, linear probe → exact Q* | −5.38 ± 3.68 (1.3% suboptimal decisions) |

- **Policy:** 99.94% of 128,000 greedy decisions are optimal; the only deviation is one extra listen at the tightest margin (0.70).
- **Latent:** the MLP probe's KL is below 1e-4 nats; minimality ratio 0.27 ± 0.03. Sufficient, but neither minimal nor linearly sufficient.
- **Observation prediction:** KL 1–6e-4 nats; accuracy 58.3% against a Bayes ceiling of 57.8%.
- **Runtime:** exact solving is far cheaper on Tiger (a 72 s solve and 0.096 ms per decision, against 52–59 min of training and 4.6 ms per decision).
- **Exact references:** V_h(b₀) = −1, −1.95, 2.3098, 1.7955, 2.7631 for h = 1..5; V*(b₀) = 19.3713.

### 4b. Decoder baseline on Tiger (indistinguishable from JEPA)

| | JEPA | Decoder |
|---|---|---|
| Learned planner − optimal | −0.03 ± 0.06 | +0.06 ± 0.12 |
| MLP-decoded agent − optimal | −0.23 ± 0.53 | −0.10 ± 0.76 |
| Linear probe KL (random / agent histories) | 0.0011 / 0.0005 | 0.0007 / 0.0005 |
| Minimality ratio | 0.27 ± 0.03 | 0.31 ± 0.04 |

Likely a ceiling effect of Tiger's one-bit observation. On Wumpus the decoder learns the better latent (section 5.5).

## 5. Wumpus World

### 5.1 Domain (`src/ipomdp/domain/wumpus.py`, `conf/env/wumpus.yaml`; user decisions of 2026-10-01/02)

- **Grid:** 4x4; start at cell 0 (bottom left) **facing North**; cell index c = x + 4y; headings 0 E, 1 N, 2 W, 3 S.
- **Placement:** 2 pits, 1 wumpus and 1 gold on pairwise distinct non-start cells: 16,380 equally likely layouts. Unwinnable layouts are kept.
- **Actions:** FORWARD, TURN_LEFT, TURN_RIGHT, GRAB, SHOOT. **No Climb**; grabbing the gold wins.
- **Rewards:** −1 per action (−10 for a shot that uses the arrow), +1000 for the gold, −1000 for death; γ = 0.99.
- **Arrow:** flies to the wall. A dead wumpus still smells, and its cell is safe.
- **Observations:** 5 percept bits (STENCH 1, BREEZE 2, GLITTER 4, BUMP 8, SCREAM 16), plus GAME_OVER = 32. Position and heading are not observed. `reset()` emits o₀, the start-cell percept.
- **End of game:** an absorbing GAME_OVER state with reward 0 fills the rest of the 50-step window.
- **Ground truth:** `WumpusBeliefTracker`, an exact enumeration filter: marginals plus the hunter's own state.
- **Model and search:** latent 128, hidden 256, observation branches pruned below 1e-3, 50 simulations. A full run is `training.total_episodes=6400` (100 collections; the default 2560 gives only 40).

### 5.2 The exact-belief heuristic (`src/ipomdp/agents/wumpus_reference.py`)

A hand-written, deterministic policy on the exact belief. It is **not optimal** and must not be called "optimal" or "V*". Each step it applies the first rule that fits:
1. P(gold here) = 1 → GRAB.
2. Walk the shortest path (BFS over (cell, heading) through provably safe cells) to the nearest provably safe unvisited cell, i.e. P(pit) = 0 and P(live wumpus) = 0. A cell with known gold comes first.
3. With the arrow and a live wumpus: from a safe spot, shoot at an unvisited, pit-free cell that may hold the wumpus. Either outcome makes that cell safe.
4. Step into the lowest-risk frontier cell if its risk (P(pit) + P(live wumpus)) is at most the threshold.
5. Otherwise TURN_LEFT (wait).

| Risk threshold (512 games) | Return | Win | Death | Unfinished |
|---|---|---|---|---|
| 0 (never gambles) | 526 ± 20 | 62.3% | 0% | 37.7% |
| **0.4 (used as "the reference")** | **573 ± 20** | 69.7% | 1.8% | 28.5% |
| 1 | 517 ± 32 | 81.2% | 18.8% | 0% |

0.4 was the best of three thresholds tried on the same 512 analysis games, so 573 is slightly optimistic. The random hunter earns −388 ± 26.

### 5.3 The current agent and its results

**Recipe.** `env=wumpus model.representation=decoder model.observation_horizon=5 training.updates_per_collection=64 training.total_episodes=6400 training.store_eval_episodes=true agent.epsilon=0.25 agent.epsilon_min=0.1`, at commit `f5740e4`. ε decays with `temperature_decay` (0.97 per collection) to 0.1.

| Seed (512 fresh games, best.pt) | Return | Win | Death | Unfinished |
|---|---|---|---|---|
| 0 | 185.5 ± 20.0 | 29.5% | 3.9% | 66.6% |
| 1 | 166.0 ± 19.6 | 26.4% | 3.3% | 70.3% |
| 2 | 215.2 ± 18.6 | 29.1% | 0.2% | 70.7% |
| **Mean, 3 seeds** | **189 ± 62** (t-interval) | | | |

- **The death model generalises** (seed 0, `latest.pt`, 256 fresh games per policy). Steps predicted below 0.02 killed 0.2% of the time under the exploring policy and 0.1% under the greedy planner. Before the fix this was 4.1% and 9.6%.
- **The remaining gap is unfinished games, not deaths.** Diagnosed on 2026-10-07 (`unfinished_diag.py`, 3 seeds × 512 games; continued in section 5.6):
  - In unfinished games the hunter stops finding new cells around step 11–14 and then idles for about 35 steps, mostly with no-ops: SHOOT without the arrow, GRAB on an empty cell, turns.
  - In 47–50% of them a provably safe, unvisited, reachable cell still existed at the end.
  - Facing such a cell, the planner steps in only 29–42% of the time. The death head is not the reason (median P(death) about 0.001). The value head rates the latent after the step about 15 below the latent after a no-op (median γ·E[V̂ next] gap −15 to −17).
  - Under the bootstrapped (truncated-window) objective a wasted step costs only about 1 + (1−γ)·V̂ ≈ 7.5, less than those V̂ errors. Exploration progress is not valued.
- **Value overestimation, measured on the current agent:** V̂ ≈ 620–660 on idle states of unfinished games against a realised −22, and 680–750 on progress steps (`value_chain_diag.py`). The full mechanism is in section 5.6.
- **JEPA, same recipe** (2 seeds, `best.pt`, the same 512 analysis games, paired with decoder seeds 0–1):

| Seed | Decoder return | JEPA return | JEPA win / death / unfinished | MLP probe cell / heading, random histories (decoder → JEPA) | Same, agent histories |
|---|---|---|---|---|---|
| 0 | 185.5 ± 20.0 | 169.6 ± 18.9 | 24.2% / 1.8% / 74.0% | 56.4 / 59.0% → 53.6 / 50.8% | 85.5 / 86.8% → 79.9 / 76.5% |
| 1 | 166.0 ± 19.6 | 191.5 ± 18.7 | 26.4% / 0.8% / 72.9% | 58.0 / 59.5% → 53.5 / 49.7% | 83.0 / 87.9% → 87.0 / 91.0% |

  - The paired return differences are −16 and +26: **no detectable difference** with 2 seeds.
  - On random histories JEPA's latent decodes heading about 9 points worse, in line with the offline A/B (section 5.5).
- **Recipe ablations (2026-10-10, 3 seeds each, paired with seeds 0–2 of the current agent; `runs/wumpus/ablation_chain.sh`):**

| Arm | Return per seed | Death | Unfinished | Paired Δ return |
|---|---|---|---|---|
| Current agent | 185.5 / 166.0 / 215.2 | 3.9 / 3.3 / 0.2% | 66.6 / 70.3 / 70.7% | |
| ε off (`u64_store_eval_noeps_…`) | 228.9 / 190.0 / 174.2 | 0.4 / 0.8 / 3.1% | 69.7 / 74.2 / 70.3% | +8.8 ± 109.8 (no detectable effect) |
| store_eval off (`u64_eps_nostore_…`) | 110.6 / 13.7 / 111.5 | 12.5 / 17.8 / 6.1% | 60.5 / 60.9 / 71.1% | **−110 ± 97** (significant) |

  - **Storing the greedy evaluation games is what matters.** Without them, deaths return: the head is not trained on the greedy planner's own risky FORWARDs (section 5.5).
  - **ε is not needed.** Neither arm changes the idling.

### 5.4 Interpretability findings (current agent, seed 0)

- **Probes (MLP; held-out games):** cell 55–58% and heading 55–60% on random-action histories; 83–88% and 87–90% on the agent's own games. The same GRU trained on exact targets (`supervised_filter.py`) reaches 99.9% and 100%. Hazard marginals beat the constant baseline only modestly (pit error 0.070 vs 0.085).
- **UMAP** (`reports/2026-10-07_wumpus_reviewer_page/scripts/umap_latents.py`; trustworthiness 0.983):
  - The map is organised first by the percept just received.
  - A point's 10 nearest neighbours share its percept 93% of the time on random histories (36% by chance), its heading 44% (25%) and its cell 43% (24%). On the agent's own games: 96%, 74% and 71%.
  - The danger ahead varies only inside the stench/breeze clusters.
- **Example death game in the player:** on its final step the exact danger is 1.00 and the model says 0.00, and the decoded pit map puts the pit in the wrong cell. Rare deaths still come from latent errors.

### 5.5 How we got here (condensed diagnosis chain; details in the script headers under `reports/2026-10-02_wumpus_diagnostics/`)

1. **The first agents (JEPA 110 ± 20, decoder 85 ± 21) wandered and died.** Their death prediction was badly miscalibrated (`death_calibration.py`).
2. **Ruled out:**
   - search depth: 200 simulations gave no gain (`sims_diag.py`);
   - the reward head, which is accurate (`reward_diag.py`);
   - exploration data for the representation: training on reference data made play worse (`offpolicy_reference.py`);
   - the architecture: a supervised GRU reaches about 100%, and a causal Transformer was no better (`supervised_filter.py`, `ab_summary.py`).
3. **The objective is the limit for the latent.** In the offline objective A/B (`objective_ab.py`) the decoder beat JEPA significantly: heading +20 points, cell +7, danger error −0.016. The multi-step term (k = 5) did not help JEPA and was seed-dependent for the decoder. Longer offline training improved the decoder latent but made the head's death error worse. That was the first sign of overfitting.
4. **Coverage** (`head_coverage.py`): forcing dangerous FORWARD outcomes into the data halved a fresh head's danger error.
5. **The combined-fix run** (decoder, k = 5, ε, **256** updates per collection) failed at **−283 ± 17**. `greedy_deaths.py`: the planner stepped forward where its head confidently said "safe".
6. **Storing the greedy games** (still at 256 updates) gave **−84 ± 27**.
7. **`head_shift.py` (with a correction):** the head is calibrated **in-sample** but overconfident on **fresh games of the exploring policy too**, so the problem is overfitting, not only a policy shift. A fresh head on frozen latents generalised better than the jointly trained one.
8. **The offline overfitting A/B** (`objective_ab.py` arms plus `overfit_summary.py`, 2 seeds per arm):
   - with 2,048 games, held-out NLL rose from 0.30 to 0.61 while training NLL fell;
   - even with 8,192 games it overfits past about 6,400 updates;
   - weight decay up to 0.1 had no effect.
9. **Back to 64 updates per collection, keeping the other changes:** 189 ± 62 (section 5.3).

### 5.6 The unfinished games: diagnosis and two failed value remedies (2026-10-07 to 10-09)

**Diagnosis** (`unfinished_diag.py`, `value_chain_diag.py`; current agent, 3 seeds × 512 games; both replay the analysis games exactly, with outcomes equal to `report.json`):
- **Behaviour:** in unfinished games the hunter stops finding new cells around step 11–14, then spends about 35 steps on no-ops (SHOOT without the arrow, GRAB on an empty cell, turns).
- **It isn't just stuck behind risk:** in 47–50% of those games a provably safe, unvisited, reachable cell still existed at the end. Facing such a cell, the planner steps in only 29–42% of the time.
- **Not the death head:** median P(death | FORWARD) there is about 0.001. The whole one-step deficit is γ·E[V̂(next)]: the value rates the latent after the step about 15 below the latent after a no-op.
- **H2 rejected, the latent knows about progress.** Held-out MLP probes on frozen latents:
  - "safe unvisited cell directly ahead": 93–97% correct, against 79–84% for always guessing the majority answer;
  - cells visited: off by 0.15–0.37, against 1.6–2.3 for always guessing the average;
  - steps since the last new cell: off by 1.9, against 12.
- **H1, how idling sustains itself:** on idle steps V̂ ≈ 620–660 against a realised −22, and the Bellman residual is small. A no-op backup lowers V̂ by only about 1 + (1−γ)·V̂ ≈ 7 per iteration. On seeds 0 and 1 the latent barely moves under idle actions (0.17 against 4–6 on progress steps); seed 2 drifts and idles anyway.
- **H3 confirmed, the value learning is the failure.** A fresh value fitted on the same frozen latents to the **exact-belief heuristic's** Monte Carlo returns (held-out error 59–65, against 450 for a constant) prefers FORWARD at the agent's declined safe-cell steps: median +76 to +225, FORWARD first 66–73% of the time. The learned value prefers the no-op there (median −9 to −27).

**Remedy 1, time-aware value** (`model.time_aware_value`, `8ca58d0`; user choice: value head only). V(z, k), where k is the steps left in the window; the search counts k down to a terminal 0.

| Seed | Current | Time-aware | Unfinished |
|---|---|---|---|
| 0 | 185.5 ± 20.0 | 200.2 ± 19.0 | 66.6% → 70.7% |
| 1 | 166.0 ± 19.6 | 107.9 ± 19.7 | 70.3% → 69.3% |
| 2 | 215.2 ± 18.6 | 143.1 ± 20.4 | 70.7% → 67.6% |

The agent idles all the same, with GRAB instead of SHOOT. The window being treated as a cut-off is not the cause.

**Remedy 2, n-step greedy-return grounding** (`model.value_nstep=10`, mix 0.5, `52ac3fc`). On stored greedy evaluation games the value target is 0.5·Bellman + 0.5·Gₜ⁽¹⁰⁾, where G is the 10-step return of real rewards bootstrapped with the target value head.

| Seed | Current | n-step | Unfinished |
|---|---|---|---|
| 0 | 185.5 ± 20.0 | 87.9 ± 20.5 | 66.6% → 68.2% |
| 1 | 166.0 ± 19.6 | 93.4 ± 16.0 | 70.3% → 79.1% |
| 2 | 215.2 ± 18.6 | 126.1 ± 17.0 | 70.7% → 77.5% |

- **Idle V̂ fell** (650 → 290–380), as intended.
- **But the agent steps into safe cells *less*:** 14–27% of the time. Its value's no-op preference widened to −30 to −37.
- **Reading:** the greedy policy's own games end unfinished whether a state is "about to explore" or "idling", so their returns pull every value down without separating the two.

**Remedy 3, lower γ** (`env.discount=0.95`, `1c05c16`; user choice). The current recipe otherwise unchanged. Returns are at γ = 0.95, so compare with the heuristic re-scored at 0.95: 362.4 (random −227.6).

| Seed | Return (γ 0.95) | Win | Death | Unfinished (current agent) | FORWARD at safe cells, unfinished games (current) |
|---|---|---|---|---|---|
| 0 | 177.8 ± 15.7 | 30.9% | 2.3% | 66.8% (66.6%) | 55% (29%) |
| 1 | 196.3 ± 16.2 | 34.0% | 3.5% | 62.5% (70.3%) | 52% (31%) |
| 2 | 130.8 ± 16.1 | 26.0% | 7.0% | 67.0% (70.7%) | 32% (42%) |

- **Paired with the current agent:** unfinished −3.8 points (95% interval ±9.9) and wins +2.0 (±13.3). Neither is significant.
- **Normalised score** (agent − random) / (heuristic − random): 0.67 against 0.60 before.
- **Idle V̂ fell** to 200–330 (from about 650), still +210 to +330 above the realised return. The learned one-step preference at the declined steps is still the no-op (median −17 / −61 / −35).
- **Confound:** `value_bound` drops to 20,020, so the two-hot support is 5× finer. A finer support is therefore not the missing piece either.

**Offline screens of value targets (2026-10-09; user choice "screen offline, then train").** On frozen latents, a fresh MLP value is scored with the M4 test: Q(FORWARD) − Q(chosen) at the agent's declined safe-cell steps.

`grounding_screen.py` (the agent's own replay buffer from `latest.pt`, 8,960 games; early-stopped fits). FORWARD preferred, seeds 0 / 1 / 2:

| Target | FORWARD preferred | Note |
|---|---|---|
| learned V̂ | 23 / 13 / 34% | |
| **A** max(V̂, G), all buffer games | 23 / 16 / 35% | **fails**: G > V̂ on only 5% of steps, since V̂ already overestimates by 480–690 |
| G, exploring collection games | 40 / 27 / 65% | returns explain 9–13% of variance (layout luck) |
| G, all buffer games | 53 / 25 / 54% | |
| G, heuristic games (M4's target) | 69 / 62 / 39% | weaker than M4 |

A correction to the earlier write-up: a single realised return is **not** ≤ V*(b). Only its expectation over layouts is, so max(V̂, G) is also biased upward by luck.

`teacher_robustness.py` (heuristic returns; 4 protocols × 3 initialisations; 2,048 fresh heuristic games):
- Long (3000-step) fits pass on every seed, but their median gaps vary 3–6× across initialisations.
- Early-stopped fits (150–550 steps) **fail seed 2** with 512 and with 2,048 games (39–47%).
- Held-out fit quality barely differs between protocols (MSE explained 0.79–0.86).
- **Reading:** the test evaluates V on imagined children of idle-region states, which the heuristic's games barely cover. The ranking is extrapolation, not a property of the teacher's returns. M4's 66–73% is one draw from that spread, so H3 in `value_chain_diag.py` is weaker than stated: the information is in the latent, but no screened return target robustly ranks FORWARD first.

## 6. Next steps (in order; ask the user at each research decision, recommendation first)

1. **The idling problem: the user's decision.** Every value-side remedy has failed:
   - in training: time-aware, n-step, γ = 0.95 (which also gave a 5× finer support);
   - offline: optimistic, own Monte Carlo, teacher returns.

   Candidates left:
   - **Planner-side idle check:** treat a child latent ≈ its parent (a no-op) as no-progress at search time. Engineering rather than learning; weigh it against the research goals.
   - **Accept the plateau** and move to steps 2 and 4. The latent (step 4) may be what makes idle states look valuable: they are percept-organised and lie close to progress states.
   - Diagnostics to rerun on any new agent: `unfinished_diag.py <run> 512` and `value_chain_diag.py <run> 512`.
2. **Done (2026-10-10): recipe ablations** (section 5.3). Store the greedy games; ε is optional.
4. **Push the latent towards position and heading.** It is organised by percept; probes are 55–60% against 100% supervised. Candidates: longer horizons for the multi-step term at the fixed update rate; an auxiliary objective that does not use privileged state. Use offline A/Bs on `pools.pt` first, and save the networks this time.
5. **Seeds and tooling.**
   - Done: `sweep.py` and `experiments/aggregate.py` handle Wumpus (2026-10-10). Existing runs join a sweep by symlink plus a `TRAINED` marker. `runs/sweeps/wumpus_current` links the current agent's seeds.
   - Seeds 3 and 4 are training (`runs/wumpus/current_seeds34_chain.sh`). Once they are analysed, link them and rerun the same `sweep.py wumpus_current --seeds 0,1,2,3,4 <the recipe's overrides>`.
6. **Rerun the slow suite at HEAD once** (section 3, Tests).
7. **Ask before pushing** `61991c7` onward.
8. **Deferred:** the hybrid JEPA + observation condition; a model-free recurrent Q-learner baseline; multi-agent Tiger (verify its tables against Gmytrasiewicz & Doshi first); I-POMDP levels via the S × M_j reduction. Cheap Tiger questions: an observation-head detach ablation in JEPA mode; Polyak momentum 0.98.

## 7. Failed paths and pitfalls: do not repeat

**Representation and model**
- **VICReg** (any placement) made the latent less belief-like. **Pure JEPA without reward grounding** learns no belief.
- **A stochastic (DreamerV3 discrete) latent transition for imagination** imagined biased beliefs, and predicting online targets collapsed the representation.
- **Free bits** on the balanced KL: the prior never trained.
- **Symlog-space two-hot decoding** is Jensen-biased; **unbounded bins** wreck the real-space means.
- **A decoder of the *current* o_t:** the GRU can copy its last input. Use next-observation prediction.
- **Wumpus, JEPA as is (64 updates, no ε):** the latent does not encode the danger ahead.
- **Wumpus, multi-step k = 5 on JEPA:** no effect.
- **Wumpus, a causal Transformer filter (untuned):** no better than the GRU, and worse on cell.

**Training and data**
- **256 updates per collection on Wumpus** (replay ratio 64): the filter and head overfit; −283 ± 17, and −84 ± 27 even with greedy games stored. Keep 64 per collection, or justify more updates with held-out NLL.
- **Weight decay** (1e-2, 1e-1) against that overfitting: no effect.
- **Training the Wumpus agent on reference-agent data** (off-policy): made play worse.
- **TD(λ) value targets** learn the exploring policy's value, so the agent listens forever (Tiger).
- **Bellman value targets with gradients into the filter** collapse the latent.
- **20-step episodes:** the bootstrap is unanchored. **Chunked prioritised replay** from a zero belief: replaced by whole-episode uniform replay.
- **Acting with online (non-averaged) weights:** head jitter flips close decisions. Keep the Polyak copies.

- **Wumpus, time-aware (finite-horizon) value** (`model.time_aware_value`): no gain; still ~69% unfinished (section 5.6).
- **Wumpus, offline value-target screens** (`grounding_screen.py`, `teacher_robustness.py`): optimistic max(V̂, G) cannot lower an overestimated V̂ (fails 3/3); the agent's own MC returns are luck-dominated; heuristic returns rank FORWARD first only under some fit protocols (extrapolation). Do not build any of them as a training option without new evidence.
- **Trusting one frozen-latent fit's ranking at imagined children** (M4): check fit protocols and initialisations first.
- **Wumpus, longer multi-step horizons** (k = 10, 20, offline on `pools.pt`): no better than k = 5; k = 20 lowers cell and heading on 3 of 3 seeds.
- **Wumpus, detached planning head + 128 updates per collection** (`model.detached_planning_head`): deaths return (8–29%, returns 67 / −45). The online latent is no better, even though the offline A/B improved it with more updates. Offline latent gains on `pools.pt` do not predict the agent.
- **Wumpus, γ = 0.95** (`env.discount`): halves idle V̂ but leaves about 65% of games unfinished (paired −3.8 ± 9.9 points).
- **Wumpus, n-step returns of the agent's own greedy games mixed into the value target** (`model.value_nstep=10`, mix 0.5): worse (88–126 against 166–215). The greedy policy's returns are uniformly poor and push all values down without separating exploring from idling.

**Planning**
- **Mean (MuZero-style) backups with a uniform prior** degrade as search grows. Use expectimax.
- **More simulations on Wumpus** (200 vs 50): no gain.

**Analysis and methodology**
- **Judging calibration on the replay buffer:** it is in-sample. Always check fresh games (`head_shift.py` does both). This mistake was made once (the first `head_shift.py` reading).
- **`cfg.seed + offset` seeds** couple runs. Use `stream_seed`.
- **R² as a probe metric:** an untrained GRU already reaches 0.93. Use KL/L1 on Tiger, and the error against the constant (and majority) baseline on Wumpus.
- **Reading in-training evaluation dips, or citing the `best.pt` score:** noise and a winner's curse. Re-estimate on fresh seeds with `analyze.py`.
- **Judging a gate or an A/B from one seed:** the Tiger value gate ranged from 0.7 to 24 across seeds.
- **Choosing the reference threshold and reporting it on the same games** inflates it a little. Say so when citing 573.
- **Forgetting `training.total_episodes=6400`** on a Wumpus run gives only 40 collections.
- **Changing initialisation order in `TrainingRun`** breaks bit-exact reproduction. Build new modules after the existing ones and only when enabled.
- **Hydra `@package` below a header comment** is silently ignored. It must be line 1.
- **Comparing metric dicts with `==`:** `mean_win_step` is NaN when no game is won. Compare `repr()`.
- **Harness runs that do not save networks** (`objective_ab.py`): later diagnostics need retraining. Save the final networks next time.
- **Expanding a whole data set at once through `LearnedSearchModel.expand`** (N × A × O children; 20,000 states → 6.4 M rows) runs out of GPU memory. Chunk it (`value_chain_diag.py`, `one_step`).
- **`echo "... $(date ...) exit $?"`** logs the exit status of `date`, not of the job, so crashed runs looked successful once. Capture `status=$?` first.

**Machine and tooling**
- **Running heavy jobs under ipomcp load:** CPU-bound search slows about 4×, and parallel runs can exhaust RAM (section 2, Machine).
- **Six parallel GPU jobs** (pool builds or A/Bs) gain nothing. **`pkill -f`/`pgrep -f` self-matches** killed a shell once.
- **The page-data export plays 2 × 512 greedy games per run** for calibration; it took over 50 min under load. It now uses 256.
- **Scratchpad files vanish with the session.** Keep anything reproducible in `reports/` or `runs/`.

## 8. History (commit, then the gist)

**Pushed (through `9a9b726`)**
- `ba1d3fc` Phase 1: canonical Tiger, exact Bayes filter, certified solver, seeded env. `c267482`: non-blocking copy corruption fixed.
- `ccc985b` Phase 2 (`BeliefFilter`, reward grounding plus JEPA); `94c4b1e`, `2191999` Phase 3 (unbiased two-hot, observation-branching imagination); `d110ad9` Phase 4 (search model, expectimax tree search); `682060a`, `db625d9`, `80ee623` Phase 5 (`TrainingRun`, Bellman targets); `7a16fea` Phase 6 (probes, error bounds); `01bbe62` Phase 7 (experiments package); `fd4d628`, `e2620c4` Phase 8.
- `5c79deb` … `065b1c1`: Polyak acting weights; `polyak` sweep (Tiger gate met). `18f7454`: Tiger reviewer-page sources. `d8871f1`: decoder baseline switch. `9a9b726`: decoder sweep launched.

**Local, unpushed (`61991c7` onward)**
- `61991c7`: Tiger decoder result.
- `4d1ace5`, `90b489d`, `ae2d108`, `123fa81`: Wumpus domain, the domain-generic stack (Tiger bit-identical), the North start and 50-step window, the reference heuristic.
- `feb65bd` … `674a865`: Wumpus diagnostics, supervised bound, Transformer A/B, multi-step term (`2bbebac`), objective A/Bs.
- `62deb68`, `6af8f8f`: runtime profile; the hoisted `input_net` (user-approved).
- `a2aa606` … `07d3e08`: coverage diagnostic, acceptance seed study, previous HANDOFF rewrite.
- `5dbeca4`: **ε-uniform exploration** (`agent.epsilon`, default off; fingerprinted).
- `cab3894`, `cd9d116`, `d4911a8`: value-gate study; acceptance at 6000 updates; stale slow-test fix.
- `48a442d`, `07fdcba`, `cd076b3`, `27f28e9`: failed combined-fix run and its diagnostics (`absorbing_value.py`, `greedy_deaths.py`, `start_danger.py`, `head_shift.py`).
- `1d7801b`: **`training.store_eval_episodes`** (default off; fingerprinted).
- `87cfa37`, `aa46d46`, `68d66b4`, `f5740e4`, `d4b476e`: store-eval run, overfitting finding, overfitting A/B, 64-update runs launched.
- `03caa22` … `d8325b7`: Wumpus reviewer page (versions 1–6), UMAP, synced player, export tooling, results in HANDOFF.
- `2a6e326`: previous HANDOFF rewrite. `784b317`: `unfinished_diag.py`, JEPA arm resumed. `80a3628`: JEPA same-recipe result.
- `8ca58d0`, `483f1c3`: **`model.time_aware_value`** (default off; fingerprinted) and its launch.
- `59bd83c`: `value_chain_diag.py` (latent has progress info; value ungrounded).
- `52ac3fc`: **`model.value_nstep`** and the buffer's `greedy` flags (default off; fingerprinted).
- `2eec2bb`: HANDOFF rewrite. `fb57b11`, `45bb861`: offline value-target screens (no remedy has support).
- `8b3d621`: **`model.detached_planning_head`** (default off; fingerprinted); failed on Wumpus.
- `1c05c16`: **`env.discount`**, a configurable Wumpus γ (default 0.99; fingerprinted).

## 9. Files touched since the last push (`9a9b726`)

**New source**
- `src/ipomdp/domain/wumpus.py`: rules, `WumpusSimulator`, `WumpusBeliefTracker`, `game_outcomes`, `START_HEADING`.
- `src/ipomdp/agents/wumpus_reference.py`: `WumpusReferenceAgent` (section 5.2).
- `src/ipomdp/interpretability/wumpus_latents.py`: probe dataset, probes, `plot_belief_maps`, `plot_latent_pca`.
- `src/ipomdp/experiments/wumpus.py`: `analyze_wumpus_run`, which also reports the reference agents.
- `conf/env/wumpus.yaml`; `tests/test_wumpus.py`; `tests/conftest.py`.

**Changed source**
- `agents/planning_agent.py`: **ε-uniform exploration** (section 3 of its header).
- `training/run.py`:
  - section 7: ε, decayed with the temperature;
  - section 8: `store_eval_episodes`;
  - also `Domain`, `DomainSetup`, `observation_horizon`, `multistep_predictor`.
- `training/trainer.py`: section 2c, the multi-step term; section 3b, finite-horizon targets; section 3c, n-step greedy-return mixing (`nstep_returns`, `_mix_nstep`).
- `training/run.py`, sections 9 (`time_aware_value`) and 10 (`value_nstep`), each with its negative result; `training/episode_buffer.py`, section 4 (the `greedy` flag; `add(episodes, greedy)`).
- `planning/search_model.py`, section 5 (finite-horizon state [z, k], `LearnedSearchModel.values`); `models/heads.py`, section 2b (`HorizonValueHead`).
- `models/world_model.py` (`start`, `unroll(..., initial_observation)`, section 7, the hoisted `input_net`); `models/heads.py` (`MultiStepObservationPredictor`).
- `training/episode_buffer.py`, `training/rollout.py`, `planning/search_model.py`, `planning/mcts.py` (section 6, pruning), `domain/pomdp.py`, `domain/__init__.py`, `experiments/runs.py`, `interpretability/*`, `telemetry/visualizers.py`, `main.py`, `analyze.py`.
- `conf/config.yaml`: `model.observation_horizon`, `agent.epsilon`, `agent.epsilon_min`, `training.store_eval_episodes`, `model.time_aware_value`, `model.value_nstep`, `model.value_nstep_mix`.
- Tests:
  - `test_world_model_acceptance.py`: 6000 updates and the seed table;
  - `test_planner_and_agent.py`: ε test and the stale-signature fix;
  - `test_training_run.py`: ε decay and stored-evaluation tests;
  - also `test_wumpus.py`, `test_training.py`, `test_models.py`, `test_integration_e2e.py`.

**Reports**
- `reports/2026-10-02_wumpus_diagnostics/` (every header has usage and result):

| Script | What it does |
|---|---|
| `value_diag.py`, `reward_diag.py`, `sims_diag.py`, `split_diag.py` | Value and reward heads, simulation count, start-percept split |
| `full_info_vi.py` | Known-layout value (no longer cited) |
| `offpolicy_reference.py` | Off-policy training diagnostic |
| `death_calibration.py`, `danger_probe.py`, `reference_probe.py` | Death calibration and latent probes |
| `supervised_filter.py`, `ab_summary.py` | Supervised bound; GRU/Transformer arms; pool cache; A/B summaries |
| `objective_ab.py`, `overfit_summary.py` | Offline objective and overfitting A/Bs (weight-decay and data-size arms, held-out NLL, `head_safe_miss`); learning-curve summary |
| `head_coverage.py` | Coverage diagnostic |
| `absorbing_value.py`, `greedy_deaths.py`, `start_danger.py`, `head_shift.py` | V̂ by game status; danger at the greedy planner's FORWARDs; first-step danger; **calibration in-sample vs fresh exploring vs fresh greedy** |
| `unfinished_diag.py` | **Why games end unfinished:** action mix, idling, safe cells left, root Q gaps, the one-step Q(FORWARD) − Q(chosen) split into reward and γ·E[V̂ next] (section 5.6) |
| `grounding_screen.py`, `teacher_robustness.py` | **Offline screens of value targets** on frozen latents (section 5.6): optimistic, own MC, heuristic returns; robustness of the heuristic result to protocol, data and initialisation |
| `value_chain_diag.py` | **Why the value prefers idling:** Bellman residual and drift on idle chains, progress probes, a value refit to the heuristic's returns (section 5.6) |
| `tiger_acceptance_seed.py` | Tiger acceptance at another seed and update count |
| `hoist_forward_check.py`, `hoist_training_check.py`, `profile_train_step.py` | Hoist equivalence and runtime profile |

- `reports/2026-10-07_wumpus_reviewer_page/`: `index.html`, `data.json`, `umap.json`, `games_latents.npz`; `scripts/export.py`, `dump_latents.py`, `umap_latents.py`, `page_meta.json`.
