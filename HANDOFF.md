# HANDOFF: JEPA I-POMDP (as of 2026-10-05)

Read `CLAUDE.md` first: it covers the architecture, commands and conventions. This file covers goals, rules, current state, every result, the dead ends and the next steps. Each module starts with a `DESIGN DECISIONS & THEORETICAL FOUNDATIONS` header that holds the evidence behind its design; read it before changing that module. The user's auto-memory (`~/.claude/projects/-home-andyj1810-projects-jepa-ipomdp/memory/`) repeats the standing rules in section 2.

**Where we are in one paragraph.**
- **Tiger:** the learned agent matches the exact optimum across 5 seeds. A decoder baseline matches it too.
- **Wumpus World (current work):** the single-hunter 4x4 game, which has no exact solver. The learned planner earns 110 ± 20 per game, far below an exact-belief reference agent's 573 ± 20; the full-information upper bound is 936.
- **Diagnosis:** a long chain of diagnostics (section 5) narrows the cause.
  1. Search depth, rewards, data coverage for the representation, and network architecture are **not** the bottleneck.
  2. The learned latent never encodes the danger of the step ahead precisely enough, because JEPA/reward and one-step observation prediction don't force it to. With 4× longer training the decoder latent nearly reaches a privileged supervised bound, but the planning model's death prediction stays poorly calibrated.
  3. That death prediction is partly limited by data coverage: dangerous forward steps are rare in the data.
- **Open item:** a runtime speed-up (hoisting the filter's input network) was adopted. Tiger's slow value-head check fails at 3/9 seeds without it and 5/9 with it; the paired difference is not significant, so it stays. The gate is being re-specified (section 3).
- **The combined-fix Wumpus run failed (2026-10-05):** −283 ± 17 per game. The greedy planner steps where its death prediction is confidently wrong, and value overestimation grows with updates (section 6, step 2).

## 1. Goals

Three research contributions:
1. **Belief filter.** A recurrent JEPA belief filter (no observation decoder) learned while the agent learns the dynamics.
2. **Planning.** Belief-tree search over that filter, for POMDPs now and I-POMDPs later.
3. **Interpretability.** A map from the latent to the exact (later interactive) belief, with quantified error bounds.

Stages:
1. Canonical Tiger matches the exact solver: **done** (section 4).
2. Baseline comparison: **done** (decoder world model, section 4b).
3. **Single-hunter Wumpus World, now.** The agent should learn to play well, and the learned latent space should be shown against the exact belief. Multi-agent behaviour is explicitly **not** wanted yet.
4. Later: multi-agent domains and I-POMDP levels.

**Honest status of "optimality"** (the user asked): the evidence is empirical and Tiger-only. There is no proof or convergence guarantee, and Wumpus has no optimum to compare against, only bounds (section 5.2).

## 2. Rules the user set (follow them)

**Scope**
- Never add non-canonical elements to canonical Tiger (`tiger.95.POMDP`, γ = 0.95).
- Every change needs a small-scale check before a larger run.
- Wumpus has no exact solver (user decision). Do not test multi-agent behaviour yet.

**Code**
- No fallback paths, no deprecated code, no backward compatibility, no schema versions. Fail loudly.
- Code should read as if written from scratch, with verbose design-decision comments; DDD/DRY/KISS/SOLID.
- Every source file starts with `# ABSOLUTE PATH:` and a design-decisions header. Exception: Hydra files must have `# @package _global_` on the very first line, above the header.
- When a config key is added, add it with its old value to the saved `.hydra/config.yaml` of every existing run so they still reload. This was done for `representation`, `observation_prune` and `observation_horizon`.

**Architecture and efficiency** (user guidance of 2026-10-03 and 2026-10-04, also in memory)
- **The network architecture must never be the bottleneck.** When results look wrong, A/B-test state-of-the-art upgrades: a Transformer over history tokens with a KV-cache, residuals, (flash) attention, MoE, GQA, MLA. Use good statistics (several seeds, paired comparisons, 95% intervals); intuition only chooses what to test first.
- **Always use the most memory- and runtime-efficient practice that leaves results unchanged.**
  - A bit-identical speed-up can go in once a fingerprint confirms it.
  - A speed-up that only changes floating-point summation order needs the user's approval, because Tiger must stay bit-reproducible.
  - Never change code under runs in flight; record which commit a run used.

**Workflow**
- Commit on `main` (remote `github.com/huytruong1810/jepa_ipomdp`). **Ask before every push.**
- **Ask the user about every research or design decision that is theirs**, with a recommended option. They usually pick the recommendation, but not always: they chose "no Climb" and the North start themselves.
- State results plainly, including unfavourable ones and your own mistakes.
- Reviewer pages: minimal, figures over prose. The 2026-09-30 page was loved.

**Machine** (WSL2 Ubuntu; RTX 5080 Laptop with 16 GB; 24 GB RAM; `uv`, Python 3.12, cu128)
- `~/projects/ipomcp` often runs heavy jobs here: load average 10–24 on 24 cores, and ≥ 5 workers at about 1.45 GB RAM each. Check `uptime`, `free -m` and `ps -eo pid,rss,args --sort=-rss | head` before launching anything.
- **RAM is the binding limit for parallel runs.** One Tiger acceptance run takes about 1.8 GB RAM; one objective_ab run holds the pool on the GPU (six in parallel used 12.8 GB of VRAM).
  - On 2026-10-05, six acceptance runs plus ipomcp pushed the system to critically low memory. Claude Code then reaped an idle background watcher.
  - Run at most 2–3 heavy jobs at a time while ipomcp is busy.
- Detach long runs with `setsid nohup ... > log 2>&1 < /dev/null & disown`, then confirm the python process is alive. The "Done" line printed by `setsid` refers to the launcher, not the run.
- **Waiting on runs:**
  - Never use `pgrep -f "<pattern>"` inside a shell loop: the loop's own command line matches the pattern, so it never exits. The same self-match once returned the wrong PID.
  - Wait on the PID instead (`while kill -0 <pid>; do sleep 60; done`) or match the python process exactly, e.g. `ps -eo args | grep "^[^ ]*python[^ ]* main.py"`.
  - Background watchers in Claude Code stop after at most 2 h, so re-arm them for longer runs.
- Cap CPU threads in parallel runs (`OMP_NUM_THREADS=2`). Tests pin torch to one thread (`tests/conftest.py`).
- **Intel iGPU:** Windows also shows an integrated "Intel(R) Graphics" chip with shared memory. Using it would need a separate XPU PyTorch environment and Intel's WSL compute runtime (there is no `/dev/dri`). It was not set up: it adds little next to the 5080 and the bottleneck has been CPU and RAM. Revisit only if GPU throughput becomes the limit.

## 3. Current state

**Git**
- `main` is pushed through `61991c7`. Everything after it (`4d1ace5` … latest) is local and unpushed: **ask before pushing.**
- The working tree is clean after the HANDOFF commit.

**Tests**
- The fast suite passes: 151 tests, about 60–80 s under load.
- **Slow suite at HEAD (2026-10-05): 7/7 pass** (acceptance at 6000 updates). History:
  - Before the hoist (`62deb68`) and before Wumpus (`61991c7`): 5/5 at seed 0.
  - At HEAD (with the hoist): 4/5. `test_value_head_approaches_optimal_value` fails with mean |V − V*| = 22.3 (gate 8).
  - **The value gate is seed-fragile.** Value error by network-init seed (`reports/2026-10-02_wumpus_diagnostics/tiger_acceptance_seed.py`; logs in `runs/tiger_acceptance_seeds/`):

| Seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | Mean | Median | Fails |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Before hoist (`62deb68`) | 4.9 | 1.87 | **13.02** | 2.19 | 1.45 | 1.69 | 5.13 | **12.35** | **15.68** | 6.48 | 4.90 | 3/9 |
| With hoist (HEAD) | **22.34** | **15.09** | 6.61 | 1.54 | 0.68 | 3.83 | **13.22** | **8.94** | **24.12** | 10.71 | 8.94 | 5/9 |

  - Paired difference (with − before), seeds 0–8: **+4.2 ± 6.2** (95% t-interval, −1.9 to +10.4), Wilcoxon p = 0.25. The interval includes 0, so **the hoist stays** (user rule of 2026-10-05). The point estimate leans worse; 9 seeds cannot exclude moderate harm.
  - Planning returns in all 18 runs: 20.9–22.1 (target 19.28). Linear-probe KL ≤ 0.0011. The value error does not reach the policy.
  - **Gate re-specification (user decision 2026-10-05: check more updates first).** At 6000 updates HEAD seeds 0 and 8 (the worst) give 5.3 and 1.1 (`accept_post_seed{0,8}_u6000.log`), so the test now runs 6000 updates with the bound kept at 8 (`cd9d116`; about 40 min).
  - How to run the pre-hoist arm: `git worktree add --detach <dir> 62deb68`, then run inside it with `PYTHONPATH=<dir>/src OMP_NUM_THREADS=2 <repo>/.venv/bin/python reports/2026-10-02_wumpus_diagnostics/tiger_acceptance_seed.py <seed>`. Remove the worktree afterwards.

**Runs** (`runs/` is untracked)

| Path | What |
|---|---|
| `runs/sweeps/polyak` | Tiger JEPA, 5 seeds: **the numbers to cite** |
| `runs/sweeps/default`, `runs/sweeps/decoder` | Tiger before Polyak averaging; Tiger decoder baseline |
| `runs/wumpus/smoke_seed0`, `full_seed0`, `full_seed0_c100` | Wumpus, **old setup** (East start, 100 steps). `_c100` holds collection 100's networks as `best.pt` |
| `runs/wumpus/north50_seed0` | **Wumpus baseline, current setup** (JEPA, 100 collections; `main_part1.log` and `.hydra_part1/` hold the first 40-collection launch) |
| `runs/wumpus/north50_decoder_seed0` | Decoder agent, current setup (plus logs of the diagnostics) |
| `runs/wumpus/offpolicy_reference_seed0` | Off-policy diagnostic (reference + random data) |
| `runs/wumpus/supervised_filter/` | Supervised upper bound and GRU/Transformer A/B logs; **`pools.pt`** = the cached game pool (4096 reference + 4096 random training games, 1024 held-out, 50 steps, with rewards and exact targets); `pools_no_rewards.pt` is the older pool |
| `runs/wumpus/objective_ab/`, `objective_ab_long/` | Offline objective A/Bs (logs only; **networks were not saved**) |
| `runs/wumpus/head_coverage.log` | Coverage diagnostic |
| `runs/tiger_acceptance_seeds/` | Value-gate seed study logs |

**Reviewer page:** https://claude.ai/artifact/Rboz6xV7xA7Ejrjomy4As5 (Tiger only; sources in `reports/2026-09-30_reviewer_page/`). The owner edits it live, so read it with the Artifact tool before republishing and merge onto the live version. It does not yet include the decoder baseline or Wumpus.

## 4. Results: canonical Tiger (`runs/sweeps/polyak`, 5 seeds)

All agents play the same 512 held-out episodes. Gaps are paired with the optimal agent (Student-t 95% intervals over seeds). The optimal agent earns 18.48 on these episodes; V*(b₀) = 19.37.

| Agent | Gap to optimal |
|---|---|
| **Learned planner** | **−0.03 ± 0.06** (per seed −0.11, −0.02, −0.02, 0.00, +0.00) |
| Decoded belief, MLP probe → exact Q* | −0.23 ± 0.53 |
| Decoded belief, linear probe → exact Q* | −5.38 ± 3.68 (1.3% suboptimal decisions) |

- **Policy:** 99.94% of 128,000 greedy decisions are optimal. The only deviation is one extra listen at the tightest margin (0.70).
- **Latent:** the MLP probe's KL is below 1e-4 nats; minimality ratio 0.27 ± 0.03. The latent is sufficient but neither minimal nor linearly sufficient.
- **Observation prediction:** KL of 1–6e-4 nats; accuracy 58.3%, against a Bayes ceiling of 57.8%.
- **Runtime:** solving exactly is far cheaper on Tiger: a 72 s solve and 0.096 ms per decision, against 52–59 min of training and 4.6 ms per decision. The case for the learned agent is that it needs no model and should scale.
- **Search** looks about 3.6 actions ahead on average (at most 6) with 50 simulations.
- **Exact references:** V_h(b₀) = −1, −1.95, 2.3098, 1.7955, 2.7631 for h = 1..5, and V*(b₀) = 19.3713.

### 4b. Decoder baseline on Tiger (indistinguishable from JEPA)

`model.representation=decoder` (`training/trainer.py`, section 2b): no JEPA predictor or EMA target, and the observation head's cross-entropy flows into the filter.

| | JEPA | Decoder |
|---|---|---|
| Learned planner − optimal | −0.03 ± 0.06 | +0.06 ± 0.12 |
| MLP-decoded agent − optimal | −0.23 ± 0.53 | −0.10 ± 0.76 |
| Linear probe KL (random / agent histories) | 0.0011 / 0.0005 | 0.0007 / 0.0005 |
| Minimality ratio | 0.27 ± 0.03 | 0.31 ± 0.04 |

- All intervals overlap. The decoder escapes always-listen slightly earlier.
- Likely a ceiling effect of Tiger's one-bit observation. On Wumpus the decoder *is* better on the latent (section 5.4).

## 5. Wumpus World (current work)

### 5.1 Domain (`src/ipomdp/domain/wumpus.py`, `conf/env/wumpus.yaml`)

User decisions of 2026-10-01 and 2026-10-02:
- **Grid:** 4x4, start at cell 0 = R&N's [1,1], **facing North**. R&N faces East, as did every run before 2026-10-02; the user calls North "more canonical". Cell index c = x + 4y.
- **Placement:** 2 pits, 1 wumpus and 1 gold on pairwise distinct non-start cells, giving 16,380 equally likely layouts. Unwinnable layouts are kept.
- **Actions:** FORWARD, TURN_LEFT, TURN_RIGHT, GRAB, SHOOT. **No Climb**; grabbing the gold wins.
- **Rewards:** −1 per action (−10 for a shot that uses the arrow), +1000 for the gold, −1000 for death. The ending step also pays its cost: a win pays 999 and a death −1001. γ = 0.99.
- **Arrow:** flies to the wall. A dead wumpus still smells, and its cell is safe.
- **Observations:** 5 percept bits (STENCH, BREEZE, GLITTER, BUMP, SCREAM), giving indices 0..31, plus GAME_OVER = 32. Position and heading are **not** observed. `reset()` emits **o₀**, the start-cell percept.
- **End of game:** an absorbing GAME_OVER state with reward 0 fills the rest of the **50-step window** (was 100), which holds one game.
- **Ground truth:** `WumpusBeliefTracker` is an exact enumeration filter giving marginals and the hunter's own state. One `step_rules()` serves both simulator and filter.
- **Model and search:** latent 128, hidden 256, observation branches pruned below 1e-3. Everything else follows the Tiger defaults (64 games and 64 updates per collection, 50 simulations, 256 warm-up games, JEPA). A full run uses `training.total_episodes=6400` (100 collections). **The default of 2560 gives only 40 collections, and that mistake was made once.**

### 5.2 Bounds and reference agent

- **Upper bound:** with the layout known, the mean of each layout's optimal value is V(b₀) = 936.3. 98.83% of layouts are winnable, all within 50 steps (`reports/.../full_info_vi.py`).
- **Lower bound:** `src/ipomdp/agents/wumpus_reference.py` acts on the exact belief.
  1. Grab on glitter.
  2. Otherwise walk the shortest path to the nearest provably safe unvisited cell.
  3. Otherwise shoot at a possible wumpus to clear a cell.
  4. Otherwise gamble below a risk threshold, or idle.

  `analyze_wumpus_run` reports it on the same layouts as the planner. 512 games:

| Risk threshold | Return | Win | Death |
|---|---|---|---|
| 0 (never gambles) | 526 ± 20 | 62.3% | 0% |
| **0.4 (best of the grid)** | **573 ± 20** | 69.7% | 1.8% |
| 1 (always gambles) | 517 ± 32 | 81.2% | 18.8% |

### 5.3 Agent results (512 fresh games each, via `analyze.py`)

| Run | Return | Win | Death | Unfinished |
|---|---|---|---|---|
| `north50_seed0` (JEPA, `best.pt` from collection 85) | **110 ± 20** | 19.3% | 3.7% | 77.0% |
| `north50_decoder_seed0` (decoder, `best.pt` from collection 25) | 85 ± 21 | 21.7% | 9.0% | 69% |
| `offpolicy_reference_seed0` (trained on reference + random data) | −75 ± 32 | 29.1% | 35.4% | 35.5% |
| Random hunter | −388 ± 26 | 12.9% | 55.5% | 31.6% |
| Old setup: `full_seed0` collection 20 / collection 100 | 83 / −241 | 18.4 / 0.4% | 2.1 / 25.6% | |

- In-training evaluations wander (−83 to +134) with no trend after about 50 collections. `best.pt` scores are a winner's curse.
- **Split by start percept** (`split_diag.py`, 256 games). With a quiet start the first step is provably safe; that is 66% of games.

| | Quiet start: win | Percept at start: win |
|---|---|---|
| JEPA planner | 26% (death 1%) | 3% |
| Decoder planner | 35% (death 11%) | 0% |
| Reference (risk 0.4) | **98%** | 19% |

### 5.4 The diagnosis chain (what was tested, in order, and what it showed)

1. **Value head** (`value_diag.py`, old setup): V̂ is high and flat (V̂(z₀) = 251 at collection 20 and 529 at collection 100), correlates only 0.32 with realised return, and scores idling at about 250 where idling is worth −63. It is self-consistent with its own one-step backup.
2. **Reward head is accurate** (`reward_diag.py`): GRAB predicts 25.4 against an exact 25.4, −1.00 where the gold is impossible, and 998 where it is likely. Rewards are not the problem.
3. **Search depth is not the bottleneck** (`sims_diag.py`): 200 simulations gave 77 ± 24, the same as 50.
4. **The JEPA latent is good on the agent's own short histories** (heading 85%, cell 95%), but the planner wanders: FORWARD is 49% of its actions, yet it rarely reaches new cells.
5. **Off-policy diagnostic** (`offpolicy_reference.py`): training on reference-agent data made play *worse* (−75). Exploration data is not the bottleneck.
6. **Death prediction is badly calibrated** (`death_calibration.py`, JEPA): exact P(death | FORWARD) of 0 / 0.5 / 0.99 is predicted as P(GAME_OVER) 0.19 / 0.37 / 0.39. Q(FORWARD) − Q(TURN_LEFT) is −199 even on provably safe steps; with ±1000 rewards a small error costs a lot. **This explains the wandering.**
7. **The latent is the bottleneck** (`danger_probe.py`): even a fresh probe trained on the exact danger reads certain death as only 0.47. Localisation on travelling (reference) histories is partial: cell 87% (majority 57%), heading 62–66% (majority 31%) (`reference_probe.py`).
8. **The decoder agent does not fix it** (full run): its death prediction reads 0.15 / 0.46 at exact 0 / 0.99, and its danger probe reads 0.54.
9. **Supervised upper bound** (`supervised_filter.py`): the same GRU filter (128/256/1) trained on privileged exact targets reaches cell 99.9%, heading 100%, and a danger error of 0.032 at 3,000 steps and **0.016 at 12,000 steps** (certain death read 0.90). **The architecture and capacity suffice; the training objective is the bottleneck.**
10. **GRU vs causal Transformer** (supervised; 3 seeds each; identical pools; `ab_summary.py`). The Transformer had 4 pre-norm blocks, RMSNorm, SwiGLU and flash SDPA, 1.06M parameters, untuned; the GRU had 618k.
    - **GRU equal or better:** cell 99.9% vs 92.0% (paired −7.8 ± 3.3 points); gold error significantly better; danger error 0.033 vs 0.044 (not significant).
    - The architecture is not the bottleneck at this scale.
11. **New option: multi-step observation term** (`model.observation_horizon = k`, commit `2bbebac`). From z_t, an action-conditioned GRU rolls open-loop through the actual next k actions and predicts o_{t+1..t+k}; its cross-entropy goes into the filter. The default k = 0 builds nothing, so Tiger is unchanged.
12. **Offline objective A/B** (`objective_ab.py`): the real `TrainingRun`/trainer trained offline on `pools.pt` for 6,400 updates, 3 seeds per arm, scoring the acting latent by probes and the acting observation head by death calibration.

| Held-out | JEPA | Decoder | JEPA, k = 5 | Decoder, k = 5 | Supervised |
|---|---|---|---|---|---|
| Heading | 54% ± 5 | 74% ± 6 | 56% ± 3 | 75% ± 22 | 100% |
| Cell | 75% ± 1 | 81% ± 4 | 75% ± 1 | 80% ± 4 | 99.9% |
| Danger-probe error | 0.110 | 0.094 | 0.109 | 0.076 ± 0.081 | 0.033 |
| Head danger error | 0.124 | 0.108 | 0.119 | 0.102 | |

    - Decoder − JEPA is significant: heading +20 ± 7 points, cell +6.6 ± 5.0, danger-probe −0.016 ± 0.012, head −0.016 ± 0.005.
    - k = 5 does nothing for JEPA. For the decoder it is seed-dependent: seed 3 reached a danger-probe error of 0.039.
13. **Longer offline A/B** (decoder vs decoder + k = 5, 25,600 updates, 3 seeds, scored at 6,400 / 12,800 / 25,600; old unroll code). The 6,400-update scores reproduced item 12 exactly, so the harness is deterministic.
    - **The latent keeps improving:** danger-probe error falls from 0.094 to 0.064 (decoder) and from 0.076 to 0.051 (k = 5). Three of six runs end at 0.030–0.038, near the supervised bound. Heading reaches 83% and 87%. k = 5 vs k = 0 is not significant.
    - **The observation head's death prediction does not improve:** head danger error 0.108 → 0.136 (decoder) and 0.102 → 0.120 (k = 5). It stays bad even where the probe reads the danger well (k = 5 seed 1: probe 0.037, head 0.158). P(death) on provably safe steps stays about 0.05, which costs about 50 per step.
14. **Coverage diagnostic** (`head_coverage.py`; frozen supervised latent; fresh heads of the agent's architecture; 1 seed):
    - Normal data: head danger error 0.063, safe 0.007, certain death 0.65.
    - **Adding one forced FORWARD step from every live reference state** (outcome from the true rules, no privileged label): **0.031**, safe 0.007, certain death **0.83**.
    - Coverage of dangerous FORWARD outcomes matters: they make up 4.7% of FORWARD transitions in the pool, 14.9% after augmentation. Even without it, a head on a near-exact latent beats the agents' heads.

**Current understanding:** the agent needs (a) a latent that tracks position, heading and the danger ahead, which longer training of the decoder (± k = 5) nearly provides, and (b) a death prediction that is calibrated in exactly those states, which needs more dangerous FORWARD experience in the data, and possibly more head training or capacity. Neither has been tested in a full agent run yet.

### 5.5 Superseded early hypotheses

These predate the diagnosis chain:
1. The latent doesn't track heading: confirmed as part of item 7.
2. The search is too shallow: refuted (item 3).
3. Exploration fails to reach the gold: refuted for the representation (item 5); partly true for the head's death prediction (item 14).
4. The final collapse at collection 100 comes from value instability: untested. The longer agent runs show no such collapse.

## 6. Next steps (in order; ask the user at each research decision)

1. **Tiger value-gate study: done (2026-10-05).** The hoist stays; the acceptance test runs 6000 updates (HEAD seeds 0 and 8: 22.3/24.1 at 3000 -> 5.3/1.1 at 6000; seed table in the test header). Full slow suite at `cd9d116`: 6/7 passed; the failure was a stale `OptimalAgent.reset()` signature in `test_exact_model_planner_matches_optimal_policy_return` (broken since `90b489d`), fixed and passing. The 6000-update value gate has been checked at seeds 0 and 8 only.
2. **Wumpus: the combined-fix run failed (2026-10-05).** User decisions: option a with ε-uniform exploration.
   - `runs/wumpus/eps_decoder_k5_u256_seed0`, commit `5dbeca4`, 3 h 18 min: `env=wumpus model.representation=decoder model.observation_horizon=5 training.updates_per_collection=256 training.total_episodes=6400 agent.epsilon=0.25 agent.epsilon_min=0.1` (ε decays with `temperature_decay` to 0.1).
   - In-training greedy evaluations: −199 to −531 at every evaluation, with no trend; 59% deaths at collection 100.
   - `analyze.py` (best.pt = collection 25, 512 fresh games): **−283 ± 17**, win 0.4%, death 29%, against the random hunter at −388 and `north50_seed0` at 110. Linear probes are no better than the old runs (cell 48%, heading 41%).
   - **Diagnosis** (`absorbing_value.py`, `greedy_deaths.py`, `start_danger.py`; final checkpoint):
     - **The head is confidently wrong on the greedy planner's own histories.** FORWARDs taken at exact danger 0.4–0.6 (n = 75) and ≥ 0.6 (n = 36) were predicted at P(death) 0.004 and 0.017; the agent died on 57% and 97% of them (the step alignment was checked). The same head reads 0.5–0.9 at such dangers on random and reference histories and 0.71–0.73 at z₀ with a percept. The planner steps where the head is wrong (optimizer's curse). Older runs show the same at a smaller scale (34 and 12 such steps).
     - **Value overestimation grows with updates:** V̂ at live states is 1262 at the end (1565 at about collection 40), against 868 (`north50_decoder_seed0`) and 559 (`north50_seed0`); the maximum achievable return is 999. The absorbing state is fine (V̂ ≈ 103).
   - Options not chosen earlier: (b) an offline A/B of head-side fixes on frozen latents; (c) tightening the Wumpus value support (`value_bound` = 100,100, so only 153 of 255 bins are reachable and the spacing near 500 is 49; a domain decision for the user). Exploration options not chosen: more root noise early; reweighting GAME_OVER in the loss.
   - **Head-shift diagnostic** (`head_shift.py`, user decision 2026-10-05; the full result is in its header):
     - The head is **calibrated on its training data** (buffer FORWARDs: predicted 0.071, died 0.071, every bin) but not on the greedy planner's histories (predicted 0.002, died 0.097).
     - The acting copies are not the cause: the online pair is just as wrong.
     - The latent carries most of the danger: on held-out greedy dangerous FORWARDs (died 67%), a fresh head trained on buffer + greedy games reads 0.53, one trained on the buffer alone 0.17, and the agent's head 0.007.
     - **Cause: the exploring policy does not visit the states the greedy planner reaches**, and the greedy planner exploits the head's errors there.
   - **Next: ask the user** how to get the greedy policy's own outcomes into the data (or make planning robust to the errors).
3. **Only once Wumpus plays near the reference:** build a Wumpus seed sweep and aggregation (`sweep.py` and `experiments/aggregate.py` are still Tiger-only), then a reviewer-page section with the latent figures.
4. **Ask before pushing** `4d1ace5` onward.
5. **Deferred:** the hybrid JEPA + observation condition; a model-free recurrent Q-learner baseline; multi-agent Tiger (verify its tables against Gmytrasiewicz & Doshi first); I-POMDP levels via the S × M_j reduction.
6. **Cheap open Tiger questions:** an observation-head detach ablation in JEPA mode; Polyak momentum 0.98; common-random-number training curves.

## 7. Failed paths and pitfalls: do not repeat

**Representation and model**
- **VICReg** (any placement) made the latent less belief-like. **Pure JEPA without reward grounding** learns no belief.
- **A stochastic (DreamerV3 discrete) latent transition for imagination** imagined biased beliefs, and predicting online targets collapsed the representation.
- **Free bits** on the balanced KL: the prior never trained.
- **Symlog-space two-hot decoding** is Jensen-biased; **unbounded bins** wreck the real-space means.
- **A decoder of the *current* o_t:** the GRU can copy its last input. Use next-observation prediction.
- **Wumpus, JEPA as is:** the latent does not encode the danger ahead (section 5.4, item 7).
- **Wumpus, decoder at 6,400 updates:** better than JEPA but not enough to play well.
- **Wumpus, multi-step k = 5 on JEPA:** no effect.
- **Wumpus, decoder + k = 5 + 4× updates + ε-uniform exploration (full run, 2026-10-05):** −283 ± 17, worse than every earlier run (section 6, step 2).
- **Wumpus, a causal Transformer filter (untuned):** no better than the GRU, and worse on cell.

**Planning**
- **Mean (MuZero-style) backups with a uniform prior** degrade as search grows. Use expectimax.
- **More simulations on Wumpus** (200 vs 50): no gain.

**Training and data**
- **TD(λ) value targets** learn the exploring policy's value, so the agent listens forever.
- **Bellman value targets with gradients into the filter** collapse the latent.
- **20-step episodes:** the bootstrap is unanchored.
- **Chunked prioritised replay** from a zero belief: replaced by whole-episode uniform replay.
- **Acting with online (non-averaged) weights:** head jitter flips close decisions. Keep the Polyak copies.
- **Training the Wumpus agent on reference-agent data** (off-policy): made play worse.

**Methodology and tooling**
- **`cfg.seed + offset` seeds** couple runs. Use `stream_seed`.
- **R² as a probe metric:** an untrained GRU already reaches 0.93. Use KL/L1 on Tiger, and the error against the constant (and majority) baseline on Wumpus.
- **Reading in-training evaluation dips, or citing the `best.pt` score:** noise and a winner's curse. Re-estimate on fresh seeds.
- **Judging a gate or an A/B from one seed:** the Tiger value gate ranges from 0.7 to 22 across seeds. Use several seeds and paired intervals.
- **Running heavy jobs under `ipomcp` load:** slow, and parallel runs can exhaust RAM (section 2, Machine).
- **Six parallel builds of the supervised-filter pools** filled the 16 GB GPU and stalled for 2 h. Build once with `--pool-cache` (`runs/wumpus/supervised_filter/pools.pt`).
- **Forgetting `training.total_episodes=6400`** on a Wumpus run gives only 40 collections. Resume with the larger total; this is bit-exact, because temperature decays per collection.
- **Changing initialisation order in `TrainingRun`** breaks bit-exact reproduction. Build new modules after the existing ones and only when enabled (as for `multistep_predictor`).
- **Hydra `@package` below a header comment** is silently ignored. It must be line 1.
- **Comparing metric dicts with `==`:** `mean_win_step` is NaN when no game is won. Compare `repr()`.
- **`pgrep -f` in a watcher loop** matches itself.
- **Harness runs that do not save networks** (`objective_ab.py`): later diagnostics then need retraining. Save the final networks next time.
- **Scratchpad files vanish with the session.** Keep anything reproducible in `reports/` or `runs/`.

## 8. History (commit, then the gist)

**Tiger phases (pushed)**
- `ba1d3fc` **Phase 1**: canonical Tiger as the single source of truth; exact Bayes filter, certified alpha-vector solver, seeded batched env.
- `c267482`: non-blocking device-to-host copies had corrupted every earlier replay buffer.
- `ccc985b` **Phase 2**: `BeliefFilter`; reward grounding plus JEPA.
- `94c4b1e`, `2191999` **Phase 3**: unbiased real-space two-hot codec; observation-branching imagination; 100-step episodes.
- `d110ad9` **Phase 4**: `SearchModel` protocol and `BeliefTreeSearch` (exact observation branching, expectimax, argmax Q).
- `682060a`, `db625d9`, `80ee623` **Phase 5**: `TrainingRun` with bit-exact resume; Bellman-optimality value targets through the learned model, read from detached latents.
- `7a16fea` **Phase 6**: probes, error bounds, decoded-belief agent, geometry, `analyze.py`.
- `01bbe62` **Phase 7**: `ipomdp.experiments`; thin scripts.
- `fd4d628` **Phase 8**: hashed seed streams. `e2620c4`: consistency pass.
- `5c79deb` … `065b1c1`: Polyak-averaged acting weights; the `polyak` sweep (Tiger gate met).
- `18f7454`: reviewer-page sources. `d8871f1`: decoder baseline switch. `9a9b726`, `61991c7`: decoder sweep and its result.

**Wumpus (local, unpushed)**
- `4d1ace5`: Wumpus domain, simulator and exact enumeration filter.
- `90b489d`: domain-generic stack (o₀ through filter, buffer, rollout, agents, trainer and search models; `DomainSetup`; observation pruning; `_global_` env configs; Wumpus analysis; tests pinned to one thread). Tiger was verified bit-identical by a GPU fingerprint.
- `a1506f0`, `17f3b9b`, `43e748e`: figure fix; HANDOFF; the `pgrep` watcher pitfall.
- `ae2d108`: North start and 50-step window (user decision).
- `123fa81`: exact-belief reference agent (+3 tests) in the Wumpus analysis.
- `feb65bd` … `e67e2e1`: 200-simulation check; `north50_seed0` extended to 100 collections and analysed.
- `ce1c18e`, `b6fcb07`, `027850b`, `ac536ed`: diagnostics scripts; off-policy diagnostic; decoder run and its analysis.
- `e2d72ea`, `31aebd6`, `aa976b0`, `8ef8d90`, `6373c9a`, `cc0e937`: supervised upper bound; Transformer arm; A/B summary; pool caching; GRU-vs-Transformer result.
- `2bbebac`: **`model.observation_horizon`** (multi-step observation term) + 2 tests.
- `740b24c`, `5e8e761`, `c610603`, `b64c6e1`, `674a865`: offline objective A/B harness, results, `--score-at`, longer A/B.
- `62deb68`, `6af8f8f`: runtime profile; **hoisted `input_net` in `BeliefFilter.unroll`** (user-approved; not bit-identical in training).
- `a2aa606` … `ea66ecb`: coverage diagnostic; acceptance bisection and seed study; memory-pressure note.
- The HANDOFF rewrite commit (this file).

## 9. Files touched since `61991c7` (the last push)

**New source**
- `src/ipomdp/domain/wumpus.py`: rules, `WumpusSimulator`, `WumpusBeliefTracker`, `game_outcomes`, `START_HEADING`.
- `src/ipomdp/agents/wumpus_reference.py`: `WumpusReferenceAgent`.
- `src/ipomdp/interpretability/wumpus_latents.py`: probe dataset, probes, `plot_belief_maps`, `plot_latent_pca`.
- `src/ipomdp/experiments/wumpus.py`: `analyze_wumpus_run`, which also reports the reference agents.
- `conf/env/wumpus.yaml`; `tests/test_wumpus.py` (22 tests); `tests/conftest.py`.

**Changed source**
- `models/world_model.py`: `start`; `unroll(..., initial_observation)`; **section 7, the hoisted `input_net`**.
- `models/heads.py`: **`MultiStepObservationPredictor`** (section 4).
- `models/__init__.py`.
- `training/trainer.py`: section 2c, the multi-step term, and the `multistep_predictor` argument.
- `training/run.py`: `Domain`, `DomainSetup`, `domain_metric_names`, `observation_horizon`, `multistep_predictor`.
- `training/episode_buffer.py`: `initial_observations`.
- `training/rollout.py`: `Simulator` protocol, `agent.reset(o0)`.
- `planning/search_model.py`: `initial_states(B, o0)`.
- `planning/mcts.py`: section 6 pruning.
- `agents/*`.
- `domain/pomdp.py`, `domain/__init__.py`.
- `experiments/runs.py`: domain builders, `observation_horizon`.
- `interpretability/*`; `telemetry/visualizers.py`.
- `main.py`, `analyze.py`.
- `conf/config.yaml`: `model.observation_horizon: 0`. `conf/env/tiger.yaml`.
- Tests: `test_training_run.py`, `test_training.py`, `test_models.py`, `test_planner_and_agent.py`, `test_integration_e2e.py`.

**Reports**
- `reports/2026-09-30_reviewer_page/scripts/`: ported to the new API.
- **`reports/2026-10-02_wumpus_diagnostics/`** (each script's header explains its use and result):

| Script | What it does |
|---|---|
| `value_diag.py`, `reward_diag.py`, `sims_diag.py`, `split_diag.py` | Value and reward heads, simulation count, start-percept split |
| `full_info_vi.py` | Full-information upper bound |
| `offpolicy_reference.py` | Off-policy training diagnostic |
| `death_calibration.py`, `danger_probe.py`, `reference_probe.py` | Death calibration and latent probes |
| `supervised_filter.py` | Supervised bound; GRU/Transformer arms; pool cache |
| `objective_ab.py`, `ab_summary.py` | Objective A/B harness; Student-t and paired summaries |
| `head_coverage.py` | Coverage diagnostic |
| `tiger_acceptance_seed.py` | Tiger acceptance at other seeds |
| `hoist_forward_check.py`, `hoist_training_check.py`, `profile_train_step.py` | Hoist equivalence and runtime profile |

**Not yet generic:** `sweep.py` and `experiments/aggregate.py` (Tiger only).
