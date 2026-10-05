# HANDOFF: JEPA I-POMDP (as of 2026-10-02)

Read `CLAUDE.md` first. It describes the architecture, commands and conventions. This file covers the project's goals, current state, results, history, dead ends and next steps. The module headers (`DESIGN DECISIONS & THEORETICAL FOUNDATIONS` blocks) record the evidence behind each design choice; read the relevant header before changing a module.

**Where we are in one paragraph.** The learned agent matches the exact optimum on canonical Tiger across 5 seeds. A decoder-based baseline matches it too, so Tiger cannot tell the two representations apart. The project has moved on to the single-hunter 4x4 **Wumpus World**, which has no exact solver. On 2026-10-02 the user changed the setup to a **North start and a 50-step window**. The first full run (old setup) was analysed: **the learned planner wins no more often than a random hunter (18% vs 17%)**; it learned not to die by idling at the start. An **exact-belief reference agent** now bounds V*(b₀) from below (≈ 573; the full-information upper bound is 936). A baseline run on the new setup (`runs/wumpus/north50_seed0`) was launched 2026-10-02 15:35 (section 6, step 1).

## 1. Goals

Three research contributions:

1. **Belief filter.** A recurrent JEPA belief filter (no observation decoder) learned while the agent learns the dynamics.
2. **Planning.** Belief-tree search over that filter, for POMDPs now and I-POMDPs later.
3. **Interpretability.** A map from the latent to the exact (later interactive) belief, with quantified error bounds.

Stages:
1. Canonical Tiger matches the exact solver (**done**, section 4).
2. Baseline comparison (**done**: decoder world model, section 4b).
3. **Single-hunter Wumpus World, now:** the agent should learn to play well, and the learned latent space should be shown against the exact belief. Multi-agent behaviour is explicitly **not** wanted yet.
4. Later, multi-agent domains and I-POMDP levels.

**Honest status of "optimality"** (the user asked): the evidence is empirical and Tiger-only. There is no proof and no convergence guarantee, and on Wumpus there is no optimum to compare against.

## 2. Rules the user set (follow them)

- **Scope:**
  - Never add non-canonical elements to canonical Tiger (`tiger.95.POMDP`, γ = 0.95).
  - Every change needs a small-scale check before a larger run.
  - Wumpus: no exact solver (user decision). Do not test multi-agent behaviour yet.
- **Code:**
  - No fallback paths, no deprecated code, no backward compatibility, no schema versions. Fail loudly.
  - Code reads as if written from scratch, with verbose design-decision comments. DDD/DRY/KISS/SOLID.
  - Every source file starts with `# ABSOLUTE PATH:` and a design-decisions header. The exception is Hydra files, which must put `# @package _global_` on the very first line, above the header.
- **Workflow:**
  - Commit on `main` (remote `github.com/huytruong1810/jepa_ipomdp`). **Ask before every push.**
  - **Ask the user about every research or design decision that is theirs**, offering a recommended option. They usually pick it, but not always: on Wumpus they chose "no Climb" over the recommended option.
- **Communication:**
  - State results plainly, including unfavourable ones.
  - Reviewer pages: minimal, figures over prose. The 2026-09-30 page was loved.
- **Machine:** WSL2 Ubuntu, RTX 5080, `uv`, Python 3.12, cu128.
  - `~/projects/ipomcp` often runs CPU-heavy jobs here (load average 10–24 on 24 cores). Check `uptime` and `pgrep -af "python.*(main|sweep|analyze).py"` before long runs.
  - The search is CPU/Python-bound, so load stretches wall time but not results, because everything is seeded.
  - **Detach long runs** with `setsid nohup uv run main.py ... > log 2>&1 < /dev/null &`, and then confirm the process is alive. The "Done" job line printed by `setsid` refers to the launcher, not the run.
  - **Never wait on a run with `pgrep -f "<pattern>"` inside a shell loop.** The loop's own `bash -c` command line contains the pattern, so `pgrep` always matches itself and the loop never ends. That is why no watcher ever reported the Wumpus run finishing at 05:39. Match the python process instead, e.g. `ps -eo args | grep "^[^ ]*python[^ ]* main.py env=wumpus"`, or wait on the PID with `while kill -0 <pid>; do sleep 60; done`.

## 3. Current state

- **Git:** `main` is pushed through `61991c7`. Everything after it is local and unpushed (ask before pushing), from `4d1ace5` to the latest HANDOFF commit.
- **Tests:**
  - The fast suite passes (149 tests, about 45 s unloaded, 80 s under load). `tests/conftest.py` pins torch to one CPU thread; with 24 threads, a loaded machine made it 204 s.
  - The slow suite passed 7/7 on 2026-09-30 (post-Phase 8, Tiger). It has not been rerun since the Wumpus refactor. Tiger is proven unchanged by a GPU fingerprint instead (section 5).
- **Runs** (`runs/` is untracked):
  - `runs/sweeps/polyak`: Tiger JEPA, 5 seeds, **the numbers to cite**.
  - `runs/sweeps/default`: Tiger, before Polyak averaging.
  - `runs/sweeps/decoder`: Tiger decoder baseline, 5 seeds.
  - `runs/wumpus/smoke_seed0`: 10 collections, analysed with 128 games.
  - **`runs/wumpus/full_seed0`**: 100 collections (6,400 games), seed 0, default Wumpus config. It finished 2026-10-02 05:39 after 4 h 58 min. `best.pt` is from collection 20 and `latest.pt` from collection 100. **Not analysed.**
  - The saved `.hydra/config.yaml` files of all Tiger runs gained `model.representation: jepa` and `mcts.observation_prune: 0.0` (what they ran with), so they reload after those keys were added.
- **Reviewer page:** https://claude.ai/artifact/Rboz6xV7xA7Ejrjomy4As5 (Tiger only; sources in `reports/2026-09-30_reviewer_page/`). The owner edits it live, so always read it with the Artifact tool before republishing and merge onto the live version. It does not yet include the decoder baseline or Wumpus.

## 4. Results: canonical Tiger (`runs/sweeps/polyak`, 5 seeds)

All agents play the same 512 held-out episodes. Gaps are paired with the optimal agent, with Student-t 95% intervals over seeds. The optimal agent earns 18.48 on these episodes; V*(b₀) = 19.36.

| Agent | Gap to optimal |
|---|---|
| **Learned planner** | **−0.03 ± 0.06** (per seed −0.11, −0.02, −0.02, 0.00, +0.00) |
| Decoded belief, MLP probe → exact Q* | −0.23 ± 0.53 |
| Decoded belief, linear probe → exact Q* | −5.38 ± 3.68 (1.3% suboptimal decisions) |

- **Policy:** 99.94% of 128,000 greedy decisions match the optimal policy. The only deviations are one extra listen at the tightest decision margin (0.70).
- **Representation:** MLP probe KL below 1e-4 nats; minimality ratio 0.27 ± 0.03. The latent is sufficient but neither minimal nor linearly sufficient.
- **Observation prediction:** KL 1–6e-4 nats; accuracy 58.3% against the Bayes ceiling of 57.8%.
- **Runtime:** exact is far cheaper on Tiger: a 72 s solve and 0.096 ms per decision, against 52–59 min of training and 4.6 ms per decision. The case for the learned agent is model-freeness and scaling.
- **Search horizon:** about 3.6 actions ahead on average (at most 6) with 50 simulations.
- **Exact references:** V_h(b₀) = −1, −1.95, 2.3098, 1.7955, 2.7631 for h = 1..5; V*(b₀) = 19.3713.

## 4b. Baseline: decoder world model (Tiger; indistinguishable from JEPA)

`model.representation=decoder` (`training/trainer.py`, section 2b). There is no JEPA predictor or EMA target filter. The planner's observation head P(o′|z,a) is the decoder, and its cross-entropy flows into the filter; everything else is identical. The 5-seed sweep is compared with `polyak`:

| | JEPA | Decoder |
|---|---|---|
| Learned planner − optimal | −0.03 ± 0.06 | +0.06 ± 0.12 |
| MLP-decoded agent − optimal | −0.23 ± 0.53 | −0.10 ± 0.76 |
| Linear probe KL (random / agent histories) | 0.0011 / 0.0005 | 0.0007 / 0.0005 |
| Minimality ratio | 0.27 ± 0.03 | 0.31 ± 0.04 |

- Every interval overlaps, so **JEPA buys nothing measurable over a decoder on Tiger.**
- The decoder escapes always-listen slightly earlier: at collection 10, 4 of 5 decoder seeds had escaped against 1 of 5 JEPA seeds.
- Hypothesis (untested): Tiger's one-bit observation makes reconstruction free, which is a ceiling effect. The user deferred the hybrid condition (JEPA plus observation gradients) because it cannot be informative on Tiger.

## 4c. Wumpus World (current work)

**Domain:** the user's decisions of 2026-10-01, implemented in `src/ipomdp/domain/wumpus.py`.
- **Layout:** a 4x4 grid with the start at cell 0 = R&N's [1,1], **facing North** (user decision 2026-10-02; R&N and every run before that date face East). Cell index c = x + 4y.
- **Placement:** 2 pits, 1 wumpus and 1 gold on pairwise distinct non-start cells, giving 16,380 equally likely layouts. Unwinnable layouts are allowed.
- **Actions:** FORWARD, TURN_LEFT, TURN_RIGHT, GRAB, SHOOT. **No Climb** (the user's choice, against the recommendation); grabbing the gold wins.
- **Rewards:**
  - −1 per action, or −10 for a shot that uses the arrow.
  - +1000 for the gold and −1000 for death; the ending step also pays its cost, so a win pays 999 and a death −1001.
  - γ = 0.99.
- **Arrow:** it flies to the wall. A dead wumpus still smells, and its cell is safe.
- **Observations:**
  - The 5 percept bits (STENCH, BREEZE, GLITTER, BUMP, SCREAM), giving observations 0..31, plus GAME_OVER = 32.
  - Position and heading are **not** observed; the hunter must dead-reckon.
  - **An initial observation o₀** (the start-cell percept) is emitted by `reset()`. Tiger emits none.
- **End of game:** win or death leads to an **absorbing GAME_OVER state** with reward 0 for the rest of the fixed **50-step window** (user decision 2026-10-02; was 100), which holds one game. With the layout known, every winnable layout (98.8%) is won within 50 steps, and the full-information bound is V*(b₀) ≤ 936.
- **Ground truth:**
  - An exact enumeration filter, `WumpusBeliefTracker`, provides marginals P(pit / wumpus / gold at c) and the hunter's exact own state.
  - One `step_rules()` serves both the simulator and the filter.
  - There is no solver.
- **Model and search** (`conf/env/wumpus.yaml`): latent 128 and hidden 256 (Tiger uses 32 / 64), and observation branches with P̂ < 1e-3 pruned and renormalised (`mcts.observation_prune`). Everything else uses Tiger's defaults: 64 games per collection, 64 updates per collection, 50 simulations, 256 warm-up games and JEPA representation.

**Full run `runs/wumpus/full_seed0`** (old setup: East start, 100-step window; not comparable with runs after 2026-10-02): greedy in-training evaluation, 128 games each, standard error about 30:

| Collection | Return | Win | Death | Unfinished |
|---|---|---|---|---|
| 5 | −4.2 | 7% | 0% | 93% |
| 10 | 51.7 | 12% | 0% | 88% |
| 20 | **107.3** (best.pt) | 20% | 2% | 77% |
| 25–95 | −55 … +70 | 4–23% | 0–16% | 69–93% |
| 100 | **−268.7** | 0% | 29% | 71% |

- **Plateau after about collection 20.** The hunter is cautious and rarely finishes a game; returns wander within noise, and the last evaluation collapsed (outside noise).
- The `best.pt` score of 107 is a winner's-curse maximum of 20 noisy evaluations; re-estimate it on fresh seeds.
- Planned collections took about 3 min each under load.

**Baseline on the new setup, `runs/wumpus/north50_seed0`** (100 collections, 1 h 18 min after the resume; `best.pt` from collection 85; analysis with 512 fresh games):
- In-training evaluations wander between −83 and +134 with no trend after collection 55.

| | Return | Win | Death | Unfinished |
|---|---|---|---|---|
| Learned planner | 110 ± 20 | 19.3% | 3.7% | 77.0% |
| Reference, risk 0 / 0.4 / 1 | 526 / 573 / 517 | 62 / 70 / 81% | 0 / 1.8 / 19% | |
| Random hunter | −388 ± 26 | 12.9% | 55.5% | 31.6% |

- **The representation is good on the agent's own histories** (MLP probe): heading 85% (chance 25%), cell 95%; pit error 0.017 vs constant 0.100, wumpus 0.008 vs 0.063, gold 0.007 vs 0.016. The belief map matches the exact one closely.
- **Split by start percept** (256 games, scratch diagnostic):

| | Quiet start (66%) | Percept at start (34%) |
|---|---|---|
| Planner | 204, win 26%, unfinished 74% | −72, win 3%, unfinished 91% |
| Reference (risk 0.4) | 826, win 98% | 124, win 19% |

- With a quiet start the planner **does move** (FORWARD is 49% of all its actions; every quiet-start game has a FORWARD) but rarely finds the gold. It wanders instead of exploring new safe cells. GRAB is 22% of actions. **The gap is directed exploration toward unvisited cells, not representation and not search depth.**
- The value support is loose for Wumpus: `value_bound` = 1001/(1 − γ) = 100,100, so only 153 of 255 bins lie in the reachable return range ±1001 and the spacing near 500 is 49.

**Supervised upper bound** (`reports/2026-10-02_wumpus_diagnostics/supervised_filter.py`; the agent's `BeliefFilter` 128/256/1 trained directly on exact targets from 4096 reference + 4096 random games; 1024 held-out games):

| Held-out | Agent latent (JEPA, probe) | Supervised GRU, 3000 steps | 12000 steps |
|---|---|---|---|
| Cell / heading | 87% / 66% (reference games) | 99.9% / 100% | 99.9% / 99.9% |
| Danger ahead, mean error | 0.112 | 0.032 | 0.016 |
| Exact 0.99 decoded as | 0.47 | 0.69 | 0.90 |
| Pit / wumpus error (constant 0.103 / 0.061) | | 0.061 / 0.046 | 0.032 / 0.021 |

- **The GRU architecture and capacity suffice; the training objectives are the bottleneck.** JEPA plus reward, or the decoder's observation loss, never force the latent to track position, heading and the danger ahead precisely.
- **Offline objective A/B** (user's choice, 2026-10-03; `objective_ab.py`; real trainer offline on the cached pool, 6400 updates; 3 seeds per arm; about 70 min per run under load; summary: `ab_summary.py runs/wumpus/objective_ab/obj_*.log`). k is `model.observation_horizon` (multi-step observation term, commit `2bbebac`).

| Held-out | a JEPA | b decoder | c JEPA k=5 | d decoder k=5 | supervised |
|---|---|---|---|---|---|
| Heading | 54% ± 5 | 74% ± 6 | 56% ± 3 | 75% ± 22 | 100% |
| Cell | 75% ± 1 | 81% ± 4 | 75% ± 1 | 80% ± 4 | 99.9% |
| Danger probe error | 0.110 ± 0.004 | 0.094 ± 0.008 | 0.109 ± 0.003 | 0.076 ± 0.081 | 0.033 |
| Certain death decoded | 0.53 | 0.62 | 0.54 | 0.62 | 0.69 |
| Head danger error | 0.124 ± 0.006 | 0.108 ± 0.001 | 0.119 ± 0.004 | 0.102 ± 0.031 | |

  - Decoder − JEPA (paired, significant): heading +20 ± 7 points, cell +6.6 ± 5.0, danger probe −0.016 ± 0.012, head danger error −0.016 ± 0.005, certain death +0.086 ± 0.054.
  - Multi-step on JEPA: no effect (all paired differences near 0).
  - Decoder + k=5 varies by seed: seeds 1 and 2 are like the decoder; **seed 3 reaches heading 84% and danger error 0.039, close to the supervised 0.033.** A near-supervised latent is reachable from a non-privileged objective, but not reliably in 6400 updates.
  - Caveat: the full decoder agent run (on its own data) did not play better than JEPA (85 vs 110).
- **Longer offline A/B** (user's choice, 2026-10-04; decoder vs decoder + k=5, 25,600 updates, 3 seeds, scored at 6400 / 12800 / 25600; old unroll code; about 6 h for six parallel runs):
  - **The latent keeps improving with training.** Danger-probe error: decoder 0.094 → 0.064 ± 0.062, decoder + k=5 0.076 → 0.051 ± 0.075. Heading 83% / 87%. Three of six runs end near the supervised bound (0.030–0.038).
  - k=5 vs k=0: no significant difference at 25,600 (all paired intervals include 0).
  - **The observation head's death prediction does not improve; it gets slightly worse:** head danger error 0.108 → 0.136 ± 0.045 (decoder) and 0.102 → 0.120 ± 0.106 (k=5). It stays bad even where the latent holds the information (k=5 seed 1: probe 0.037, head 0.158). P(GAME_OVER) on provably safe steps is still ≈ 0.05, which costs ≈ 50 per safe step.
  - **Hypothesis (untested): data coverage for the head.** P(death | FORWARD) is learned from sampled outcomes, and dangerous FORWARD steps are rare in the data (the reference agent avoids them; random hunters die early).
- **Runtime profile (2026-10-04, while six runs shared the GPU):** about 950 ms per update; value targets are only 12%. The step-by-step filter unroll is kernel-launch-bound. Hoisting `BeliefFilter.input_net` out of the loop (it depends only on (a, o)) makes the unroll 4.5–7.6× faster. Its forward pass is bit-identical under bf16 autocast, but **training is not bit-identical** (gradient summation order): parameter sums differ by 0.004 (Tiger) and 0.025 (Wumpus) after 16 updates. **Adopted (user decision, 2026-10-04; commit after `62deb68`)**: `BeliefFilter.unroll` now hoists `input_net` (`world_model.py`, section 7). Cited Tiger runs reproduce bit-for-bit only at their own commits. **To do once the GPU is free:** check statistical equivalence on Tiger with `uv run pytest -m slow tests/test_world_model_acceptance.py` (gate thresholds). The longer objective A/B (`objective_ab_long`) was launched before the change and runs the old unroll. Scratch checks: `hoist_check.py`, `h.py` in the session scratchpad (re-create from this description if needed).
- **Intel iGPU (user asked, 2026-10-04):** Windows shows an integrated "Intel(R) Graphics" chip (shared memory). Using it needs a separate XPU PyTorch environment plus Intel's WSL compute runtime (no `/dev/dri` here; the venv is cu128). Not set up: low value next to the RTX 5080. Revisit only if GPU throughput becomes the limit.
- **User guidance (2026-10-03, in memory):** the architecture must never be the bottleneck; A/B-test state-of-the-art upgrades (Transformer over history tokens with a KV-cache, residuals, MoE, GQA, MLA) with good statistics. **GRU vs Transformer A/B** (supervised, 3 seeds each, 3000 steps, identical cached pools; Student-t 95% intervals; `ab_summary.py`): GRU (618k parameters) vs causal Transformer (1.06M; 4 pre-norm blocks, RMSNorm, SwiGLU, flash SDPA; untuned, same learning rate, no warm-up). Danger error 0.033 ± 0.003 vs 0.044 ± 0.020; certain death decoded 0.69 ± 0.07 vs 0.71 ± 0.04; **cell 99.9% vs 92.0% (paired −7.8 ± 3.3 points)**; heading 100% vs 96.5% ± 6.4; **gold error 0.0225 vs 0.0253 (paired +0.0028 ± 0.0007)**. The GRU is equal or significantly better, so the architecture is not the bottleneck on this task. Pitfall: six parallel pool builds filled the 16 GB GPU and stalled for 2 h; build the pools once with `--pool-cache`.

**Decoder run, `runs/wumpus/north50_decoder_seed0`** (`model.representation=decoder`, otherwise identical to `north50_seed0`; 2 h 35 min under load; `best.pt` from collection 25; logs of the checks in the run directory):

| | JEPA (`north50_seed0`) | Decoder |
|---|---|---|
| Planner, 512 games | 110 ± 20, win 19.3%, death 3.7% | 85 ± 21, win 21.7%, death 9.0% |
| Quiet start: win / death | 26% / 1% | 35% / 11% |
| Predicted P(GAME_OVER) at exact danger 0 / 0.99 | 0.19 / 0.39 | 0.15 / 0.46 |
| Q(FORWARD) − Q(TURN_LEFT) at provably safe states | −199 | −132 |
| Danger probe decodes exact 0.99 as | 0.47 | 0.54 |
| Agent-history probe: cell / heading | 95% / 85% | 89% / 75% |

- **The decoder does not fix it; the hypothesis is refuted.** Observation gradients into the filter do not make the danger ahead decodable. The limitation is shared by both representations.
- Decoder FORWARD fraction is only 9.5% (JEPA 49%).
- A small error in P(death) is amplified by the ±1000 rewards: predicting 0.15 instead of 0 on a safe step costs about 150, which outweighs exploring.

**Off-policy diagnostic, `runs/wumpus/offpolicy_reference_seed0`** (`reports/2026-10-02_wumpus_diagnostics/offpolicy_reference.py`; 32 reference + 32 random games per collection, the planner never collects; otherwise the baseline's config):
- **The planner got worse, not better.** In-training evaluations: best −20 (collection 45); 0% wins from collection 55 on; deaths 26–81%. Analysis of `best.pt` (512 games): −75 ± 32, win 29%, death 35%, unfinished 36%. Good data does not fix it, so **the bottleneck is what the model learns, not the exploration that generates the data.**
- **Death prediction is badly calibrated** (`death_calibration.py`, baseline `best.pt`, states of reference games): exact P(death | FORWARD) 0 / 0.5 / 0.99 → predicted P(GAME_OVER) 0.19 / 0.37 / 0.39, and Q(FORWARD) − Q(TURN_LEFT) = −199 / −394 / −598. The model barely separates a provably safe step from certain death and penalises every step, which explains the wandering.
- **The latent is the bottleneck, not the head** (`danger_probe.py`, reference + random states, held-out): a fresh MLP probe trained directly on the exact danger ahead decodes 0.06 / 0.34 / 0.47 for exact 0 / 0.5 / 0.99 (mean error 0.112 vs head 0.141 vs constant 0.212). Same for the off-policy run.
- **Localisation is partial on travelling histories** (`reference_probe.py`, reference games): cell 87% (majority 57%), heading 62–66% (majority 31%). The danger ahead needs both, plus the hazard map.
- **Hypothesis:** in JEPA mode only the JEPA loss and the reward shape the latent. Observation cross-entropy (BUMP, BREEZE, STENCH, GAME_OVER) does not reach the filter, and those percepts are what pin down cell, heading and danger. The decoder baseline (`model.representation=decoder`) routes exactly that gradient into the filter.

**Analysis of `full_seed0`** (2026-10-02; 512 fresh games, old setup; `runs/wumpus/full_seed0_c100` is a copy holding collection 100's networks as `best.pt`):

| | Return | Win | Death | Unfinished |
|---|---|---|---|---|
| Planner, collection 20 (`best.pt`) | 83 ± 17 | 18.4% | 2.1% | 79.5% |
| Planner, collection 100 | −241 ± 14 | 0.4% | 25.6% | 74.0% |
| Random hunter | −441 ± 27 | 17.4% | 69.7% | 12.9% |

- **Probes on agent histories (MLP, collection 20 / 100):** pit error 0.046 / 0.029 vs constant 0.093 / 0.097; wumpus 0.030 / 0.015 vs 0.061 / 0.077; gold no better than constant at collection 20. Heading 32% / 65% (chance 25%). Cell 72% / 86%, probably inflated by the hunter rarely leaving cell 0 (majority baseline not computed). On random histories the probes barely beat the constant.
- **Belief map** of the longest held-out game: breeze and stench at the start; the hunter turns in place for 100 steps and never shoots. The decoded hazard maps follow the exact ones but are overconfident.
- **Value head** (scratch diagnostic, random histories): V̂(z₀) = 251 (collection 20) and 529 (collection 100). V̂ correlates 0.32 with realised return-to-go and scores idling states at ≈ 250, where idling is worth −63 over 100 steps. V̂ equals its own one-step Bellman backup through the model, so the model is self-consistent, but V̂ is nearly flat across states. Mean Q per action (collection 20): FORWARD 148, turns, GRAB and SHOOT ≈ 251–253, so any move with death risk loses to waiting.
- **Reward head is accurate:** GRAB predicts 25.4 on average against the exact 25.4; −1.00 where P(gold here) = 0; +998 where P(gold here) > 0.5.
- **Telemetry:** the target value mean rose from 124 (collection 13) to 332 (collection 100) while collection returns stayed negative; collection action fractions stayed near uniform even at temperature 0.1; greedy evaluation spent 30–58% of actions on GRAB.
- **Search depth is not the bottleneck:** `best.pt` with 200 simulations (instead of 50; 256 games, same analysis seed) earned 77 ± 24, with 17.2% wins, 1.6% deaths and 81% unfinished, the same as with 50. 73% of its actions were TURN_LEFT.
- **Working diagnosis (untested):** the flat value makes progress look nearly free to postpone, so the planner prefers zero-risk idling to any move with death risk. The planner learned "don't die", not "find the gold".

**Reference bounds (new setup, 512 games, `agents/wumpus_reference.py`):**
- **Upper:** with the layout known, mean V(b₀) = 936.3, and 98.83% of layouts are winnable, all within 50 steps (scratch value iteration over all 16,380 layouts with `step_rules`).
- **Lower:** the exact-belief reference agent (safe exploration by shortest path, shoot to clear, gamble below a risk threshold):

| Risk threshold | Return | Win | Death |
|---|---|---|---|
| 0 (never gambles) | 526 ± 20 | 62.3% | 0% |
| 0.4 (best of the grid) | 573 ± 20 | 69.7% | 1.8% |
| 1 (always gambles) | 517 ± 32 | 81.2% | 18.8% |

**Smoke run** (`runs/wumpus/smoke_seed0`, 10 collections; `analyze.py` with 128 games):
- **Returns:** the planner earned 36.5 ± 27.1 (win 10%, death 0%). A random hunter earned −462 (win 16%, death 73%).
- **Probe error against a constant guess, MLP probe on agent histories:** pit 0.040 vs 0.094, wumpus 0.027 vs 0.065, gold 0.011 vs 0.013. Cell decoded 76% correctly. **Heading at chance, 31%.**
- **On random-policy histories** the probes barely beat the constant guess. Random games are short, because the random hunter dies quickly.

**Hypotheses for the plateau** (before the analysis above; 1 is weakened by the heading result at collection 100, 2 is refuted by the 200-simulation check above; 3 and 4 still stand):
1. **The latent may not track heading or position well enough.** The smoke heading accuracy was 31%. Without heading the hunter cannot plan routes, and caution is the safe fixed point.
2. **The search is too shallow** for routes of 6–15 actions, which include turns; leaves rely on a value head trained by fitted VI at reward scale ±1000.
3. **Exploration:** after warm-up, Dirichlet noise and the visit-count temperature over 5 actions may rarely produce long successful excursions. Unwinnable layouts also reward caution.
4. **Final collapse:** possibly value or online-head instability at this reward scale, comparable to Tiger's head jitter that Polyak averaging fixed.

## 5. History (commit, then the gist)

- `ba1d3fc` **Phase 1**: canonical Tiger as the single source of truth; exact Bayes filter, certified alpha-vector solver, seeded batched env.
- `c267482`: non-blocking device-to-host copies had corrupted every earlier replay buffer.
- `ccc985b` **Phase 2**: `BeliefFilter` (learned z₀, one latent, whole episodes); reward grounding plus JEPA.
- `94c4b1e`, `2191999` **Phase 3**: unbiased real-space two-hot codec; observation-branching imagination; 100-step episodes.
- `d110ad9` **Phase 4**: `SearchModel` protocol and `BeliefTreeSearch` (exact observation branching, expectimax, argmax Q).
- `682060a`, `db625d9`, `80ee623` **Phase 5**: `TrainingRun` with bit-exact resume; Bellman-optimality value targets through the learned model, read from detached latents.
- `7a16fea` **Phase 6**: probes, error bounds, decoded-belief agent, geometry, `analyze.py`.
- `01bbe62` **Phase 7**: `ipomdp.experiments`; thin scripts.
- `fd4d628` **Phase 8**: hashed seed streams. `e2620c4`: consistency pass.
- `5c79deb` … `065b1c1`: Polyak-averaged acting weights; the `polyak` sweep result (Tiger gate met).
- `18f7454`: reviewer-page sources; docs.
- `d8871f1`: decoder baseline switch, `model.representation`.
- `9a9b726`, `61991c7`: decoder sweep launched; result.
- `4d1ace5`: Wumpus domain, simulator and exact enumeration filter, with 13 tests.
- `90b489d`: **domain-generic stack**:
  - Initial observation o₀ through the filter (`BeliefFilter.start`), buffer, rollout, agents, trainer and search models.
  - `DomainSetup` in place of `FinitePOMDP` in `TrainingRun`.
  - Observation pruning in the search.
  - `conf/env/*.yaml` as `_global_` overrides.
  - Wumpus analysis without a solver.
  - Tests pinned to one thread.
  - **Tiger verified bit-identical:** a 3-collection GPU fingerprint of metrics and parameter sums matches the pre-refactor commit exactly.
- `a1506f0`: fixed colour range for the latent-PCA cell panel; HANDOFF.
- `ae2d108`: North start and 50-step window (user decision).
- `HEAD`: exact-belief reference agent (`agents/wumpus_reference.py`, 3 tests) in the Wumpus analysis; analysis of `full_seed0`.

## 5b. Files touched in the Wumpus work (`4d1ace5`, `90b489d`, `a1506f0`)

- **New:**
  - `src/ipomdp/domain/wumpus.py`: rules, `WumpusSimulator`, `WumpusBeliefTracker`, `game_outcomes`.
  - `src/ipomdp/interpretability/wumpus_latents.py`: dataset, probes, `plot_belief_maps`, `plot_latent_pca`.
  - `src/ipomdp/experiments/wumpus.py`: `analyze_wumpus_run`.
  - `conf/env/wumpus.yaml`.
  - `tests/test_wumpus.py`: 17 tests.
  - `tests/conftest.py`.
- **Changed for o₀ and the generic domain:**
  - `models/world_model.py`: `start`, and `unroll(..., initial_observation)`.
  - `training/episode_buffer.py`: `initial_observations`.
  - `training/rollout.py`: `Simulator` protocol, and `agent.reset(o0)`.
  - `training/run.py`: `Domain`, `DomainSetup`, `domain_metric_names`.
  - `training/trainer.py`.
  - `planning/search_model.py`: `initial_states(B, o0)`.
  - `planning/mcts.py`: section 6 pruning, `Edge.observations`, `max_pruned_mass`.
  - `agents/*`.
  - `domain/pomdp.py`: `has_initial_observation = False`.
  - `experiments/runs.py`: `tiger_setup`, `wumpus_setup`, `DOMAIN_BUILDERS`, `latent_figure`.
  - `main.py`, `analyze.py` (dispatch by `env.name`), `conf/config.yaml` and `conf/env/tiger.yaml`.
  - `telemetry/visualizers.py`.
  - The reviewer scripts in `reports/2026-09-30_reviewer_page/scripts/`, ported to the new API.
- **Diagnostics of 2026-10-02** (all reproducible): `reports/2026-10-02_wumpus_diagnostics/` (value and reward heads, full-information bound, simulation count, start-percept split, off-policy diagnostic).
- **Not yet generic:** `sweep.py` and `experiments/aggregate.py` are still Tiger-only (they call `analyze_run`, which needs the solver). A Wumpus sweep needs a Wumpus aggregation first.

## 6. Next actions (in order)

1. **Off-policy diagnostic: done** (section 4c). The latent does not encode the danger ahead. Options offered next (2026-10-02): the decoder representation on Wumpus (recommended; existing switch), a hybrid JEPA plus observation gradients (new code), or longer training.
2. **Decoder run: done; it does not fix the latent** (section 4c). Next step options offered 2026-10-02 (ask the user): a supervised upper bound for the filter architecture (train the same GRU filter directly on the exact belief and danger; privileged diagnostic: is it the objective or the architecture/capacity?), longer training, or pausing Wumpus.
3. **Supervised upper bound (user's choice, 2026-10-03, over longer training, a larger filter and pausing Wumpus):** `reports/2026-10-02_wumpus_diagnostics/supervised_filter.py` trains the same `BeliefFilter` (128/256/1) directly on exact targets; it is running at 3000 and 12000 steps, with logs in `runs/wumpus/supervised_filter/`. Smoke test (256 games, 500 steps): heading 47% (majority 29%), certain death decoded as 0.39, so it is not trivially easy.
4. Once Wumpus plays well: build a Wumpus seed sweep and aggregation, then a reviewer-page section with the latent figures.
4. **Ask before pushing** `4d1ace5` onward.
5. **Deferred:**
   - The hybrid JEPA+decoder condition (only on a domain where it can separate the two).
   - A model-free recurrent Q-learner baseline.
   - Multi-agent Tiger (verify its tables against Gmytrasiewicz & Doshi first).
   - I-POMDP levels via the S × M_j reduction.
6. **Open Tiger questions, cheap:**
   - Observation-head detach ablation in JEPA mode.
   - Polyak momentum 0.98.
   - Common-random-number training curves.

## 7. Failed paths and pitfalls: do not repeat

Representation and model:
- **VICReg** (any placement) made the latent less belief-like. **Pure JEPA without reward grounding** learns no belief.
- **Stochastic (DreamerV3 discrete) latent transition for imagination:** it imagined biased beliefs, and predicting online targets collapsed the representation.
- **Free bits** on the balanced KL: the prior never trained.
- **Symlog-space two-hot decoding** is Jensen-biased. **Unbounded bins** wreck the real-space means.
- **A decoder of the *current* o_t** (DreamerV3 style) was not built: the GRU can copy its last input. Use next-observation prediction instead.

Planning:
- **Mean (MuZero-style) backups with a uniform prior** degrade as search grows. Use expectimax.

Training:
- **TD(λ) value targets** learn the exploring policy's value, so the agent listens forever.
- **Bellman value targets with gradients into the filter** collapse the latent.
- **20-step episodes:** the bootstrap is unanchored.
- **Chunked prioritised replay** from a zero belief: replaced by whole-episode uniform replay.
- **Acting with online (non-averaged) weights:** head jitter flips close decisions. Keep the Polyak copies.

Methodology and tooling:
- **`cfg.seed + offset` seeds** couple runs. Use `stream_seed`.
- **R² as the probe metric:** an untrained GRU reaches 0.93. On Tiger use KL/L1; on Wumpus use the error against the constant baseline.
- **Reading in-training evaluation dips, or citing the `best.pt` score:** both are noise or a winner's curse. Re-estimate on fresh seeds.
- **Running pytest or sweeps under `ipomcp` load:** slow. Torch CPU multithreading on tiny ops is pathological there; tests are already pinned to one thread.
- **Changing initialisation order in `TrainingRun`** would break bit-exact reproduction of the cited Tiger runs. New modules must be built after the existing ones.
- **Hydra `@package` placed below a header comment** is silently ignored. It must be line 1.
- **Comparing metric dicts with `==`:** `mean_win_step` is NaN when no game is won, and NaN ≠ NaN. Compare `repr()`.
- **`pgrep -f` in a watcher loop matches the loop itself** and never exits (section 2, Machine). Match the python process or wait on its PID.
