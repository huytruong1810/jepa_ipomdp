# HANDOFF — JEPA I-POMDP phased review (as of 2026-09-28)

Read `CLAUDE.md` first: it describes the current architecture and conventions. This file covers the review's state, history and next steps. The module headers (`DESIGN DECISIONS & THEORETICAL FOUNDATIONS` blocks) record the evidence behind every design choice. Read the relevant header before changing a module.

## 1. Goal and ground rules

The research code has three contributions:
1. A JEPA-style recurrent **belief filter** learned while the agent learns the dynamics.
2. **MCTS planning** over that filter, for POMDPs and later I-POMDPs.
3. **Interpretability** that maps the latent to the (interactive) belief with quantified error bounds.

The user asked for a rigorous, phased, bottom-up review:

| Phase | Scope | Status |
|---|---|---|
| 1 | domain | done |
| 2 | JEPA filter | done |
| 3 | DreamerV3 parts | done |
| 4 | MCTS/agent | done |
| 5 | training loop | done |
| 6 | interpretability / error bounds | done |
| 7 | scripts, configs, results layout | done |
| 8 | holistic review | in progress |

Rules the user set:
- **Scope:**
  - Canonical single-agent Tiger only (`tiger.95.POMDP`, γ = 0.95). Never add non-canonical elements to Tiger.
  - The learned agent must match the exact solver at small scale before anything larger is run.
- **Code quality:**
  - No fallback paths, no deprecated code, no backward compatibility, no schema versions. Fail loudly.
  - Code should read as if written from scratch. Keep comments and docstrings verbose about design decisions.
  - Follow DDD/DRY/KISS/SOLID. Restructure, rename or delete freely when it improves the design.
- **Workflow:**
  - Commit on `main` (remote `github.com/huytruong1810/jepa_ipomdp`). Ask before pushing.
  - Ask the user whenever a research or design decision is genuinely theirs. So far they have picked the recommended option each time, but they still want to be asked.
  - Every phase needs a small-scale check against exact references before any larger run.
- **Environment:** WSL2 Ubuntu, RTX 5080, `uv`, Python 3.12, cu128. Another project (`~/projects/ipomcp`) sometimes runs CPU-heavy experiments on the same machine, which inflates timings. Check `uptime`, and check `pgrep -fl main.py` before launching training.

## 2. Current git state

- Pushed through `01bbe62` (Phase 7). Phase 8 commits `fd4d628` (seeding fix) and `e2620c4` (consistency pass) are local; ask before pushing.
- Fast suite: 121 passed at `e2620c4`. The slow acceptance suite last passed at Phase 6; Phase 8 changed only seeding and comments in the code it exercises, so its numbers will differ slightly from section 4 but it has not been rerun.
- **Running:** `uv run sweep.py default --seeds 0,1,2,3,4` (default config, new seed streams), started 2026-09-28 22:09, log `runs/sweep_default.log`, results `runs/sweeps/default/aggregate.json`. About 45-60 min per seed. Re-running the same command continues it if it is interrupted. Avoid CPU-heavy work (e.g. pytest) while it runs: search is CPU-bound and slows down.

## 3. What was done (commit, then the gist)

- `db269d3`: baseline snapshot. Old run artifacts deleted (user's decision).
- `ba1d3fc`, **Phase 1 (domain).**
  - The old Tiger was not canonical: it emitted a free observation at t = 0, the growl stayed informative after a door opened, and γ was 0.99.
  - Built `src/ipomdp/domain/`:
    - `FinitePOMDP`, the single source of truth;
    - an exact batched Bayes filter;
    - an exact alpha-vector value iteration (incremental pruning plus a certified infinite-horizon bound; V*(b0) = 19.3713);
    - a seeded batched env.
  - Wumpus, UAV and the old scripts deleted.
- `c267482`, **critical bug.** Non-blocking device-to-host copies stored stale memory in the replay buffer (477 of 500 reads stale). That silently corrupted every earlier GPU run.
- `ccc985b`, **Phase 2 (JEPA filter).**
  - The old world model's latent was no more belief-like than a random network's.
  - Now: `BeliefFilter` with a learned z0, a single latent vector (no slots), whole-episode training, and VICReg removed.
  - Decision: **filter objective = reward grounding + JEPA self-prediction.**
- `94c4b1e` and `2191999`, **Phase 3 (DreamerV3 parts).**
  - Two-hot decoding was Jensen-biased: the −100/+10 door gamble (mean −45) decoded to −2.9. Replaced with an unbiased real-space codec bounded by `FinitePOMDP.value_bound`.
  - The prior never trained because free bits were swallowing the KL.
  - Stochastic latent imagination was biased.
  - Decision: **observation-branching imagination.** A learned `ObservationHead` predicts P(o′|z,a) and the real `BeliefFilter.step` produces the child latents. The JEPA predictor became deterministic and is used for representation learning only.
  - Tiger episodes are 100 steps. A 20-step truncation left the bootstrap latent unanchored and biased values by 8%.
- `d110ad9`, **Phase 4 (planner/agent).**
  - Decision: **no opponent inputs anywhere.** The opponent is folded into the env.
  - `SearchModel` protocol with `ExactSearchModel` and `LearnedSearchModel`.
  - `BeliefTreeSearch`: exact branching over observations, **expectimax backups** (mean backups got worse with more search), argmax-Q greedy actions, seeded RNG.
  - The exact-model agent matches the optimal return (slow test).
- `682060a`, `db625d9`, `80ee623`, **Phase 5 (training loop).**
  - `TrainingRun` gives bit-exact resume, including buffer and RNG states.
  - Greedy evaluation of the discounted return; `best.pt` chosen by evaluation.
  - Decision: **value target = Bellman optimality backup through the learned model** (EMA target value head, momentum 0.9), with the value head on **detached** latents.
  - The full loop reaches roughly V*: greedy returns about 12.6–22.4 once learning starts. Default config: 64 episodes × 40 collections with 64 updates each, about 40 min.
  - Learning is gated by roughly 1000 gradient updates, not by collected data.
- `7a16fea`, **Phase 6 (interpretability).**
  - Probes: `build_probe_dataset` from any recorded episodes, fitted `BeliefProbe`s that decode latents, evaluation on held-out episodes.
  - `error_bounds.py`: span-Hölder bounds (value, one-step regret, discounted loss) in worst-case and expected form.
  - `analysis.py`: decoded-belief agent versus optimal versus the learned planner on common seeds, plus a geometry report.
  - Solver: `action_value_functions` (per-action Q* sets).
  - Agents: `UniformRandomAgent`, and an `Agent` protocol for `play_episodes`.
  - `BeliefGeometryVisualizer`.
  - `analyze.py <run_dir>` writes `analysis/report.json` and `geometry.png`.

- Phase 7 commit, **Phase 7 (scripts, configs, results layout).** User decisions: thin scripts over a library package; delete stale files; sequential multi-seed runner with aggregation.
  - `src/ipomdp/experiments/`: `runs.py` (`DOMAIN_BUILDERS`, `build_run_config`, `build_training_run`, `load_trained_run`, `analyze_run`) and `aggregate.py` (Student-t 95% intervals over seeds, paired within-seed gaps). No script imports another any more.
  - `sweep.py <name> --seeds ... [overrides]`: one `main.py` process per seed into `runs/sweeps/<name>/seed<k>/`, then `analyze_run`, then `aggregate.json`. Continues an interrupted sweep; a changed condition raises.
  - `main.py` exits non-zero on SIGINT (130) and guardrail aborts (1), so only completed runs are analysed.
  - Deleted `BACKLOG.md`, `plot_rewards.py`, `src/__init__.py`. Dropped `torchvision`, `torchaudio`, `scikit-learn`; `pytest` moved to the `dev` dependency group. `.gitignore` pruned. `recreate_venv.sh` now just `uv sync` + CUDA check. README rewritten.

- `fd4d628`, `e2620c4`, **Phase 8 (holistic review), part 1.** Every module re-read end to end against its header and the others.
  - **Seed streams were shared across runs** (user decision: stop, fix, restart the sweep). `cfg.seed + offset` made run k's collection env replay run k-1's evaluation env, and run k's buffer share run k+2's env stream, so sweep replicates were coupled. Now `training/seeding.py` hashes `(seed, stream)` with numpy `SeedSequence` for every consumer, in training and analysis. Tested (the cross-run test fails under the old scheme). Consequence: runs made before `fd4d628` (including `phase5_ratio64_seed0`) are not reproducible from their seed any more; section 4 numbers were produced with the old streams.
  - Verified with no change needed: the solver's ε-pruning accounting (2|O|ε per backup) and certified bound; expectimax/PUCT search; the Bellman value target through `LearnedSearchModel.expand`; the error-bound derivations (span-Hölder, 2 L_Q ε regret, performance-difference form of (c)); resume state coverage.
  - Stale comments fixed (see the `e2620c4` message) and dead code removed (`symlog`/`symexp`, silent `None` skipping in `MetricsLogger`, cwd-relative visualizer defaults).

## 4. Key measured results (cite from the module headers)

- **Multi-seed result (Phase 8, the numbers to cite):** `runs/sweeps/default/aggregate.json`, 5 seeds, default config, new seed streams, about 57 min per seed.
  - Paired gaps to the optimal agent on common analysis episodes (Student-t 95% over seeds):

    | Agent | Gap |
    |---|---|
    | Decoded-belief, MLP probe | −0.08 ± 0.44 |
    | Learned planner | −0.51 ± 1.02 (seed 1: −1.97; others ≥ −0.27) |
    | Decoded-belief, linear probe | −6.16 ± 7.00 (seed 3: −16.06) |

  - The optimal agent earns 18.48 on those 512 episodes (V*(b0) = 19.36).
  - Probe KL: MLP < 1e-4, linear 0.0009. Suboptimal decisions: MLP 0.17%, linear 2.1%. Minimality ratio 0.27 ± 0.03.
  - Every seed escaped always-listen by collection 10 (seed 4: 15).
  - The biased training selection score was 22.01 ± 2.26, against an unbiased 17.96 ± 1.02 for the same checkpoints.
  - Late-run evaluation dips persist: seed 1 scored 11.40 at collection 30, seed 2 scored 14.58 at collection 40. The final checkpoint is not the best one.
- **Where the planner loses return (Phase 8 diagnosis, 2026-09-29):** a scratch script replayed greedy episodes with exact beliefs tracked alongside.
  - Seed 1's entire shortfall is one decision. At posterior 0.97/0.03 (two net growls) it LISTENs instead of opening the correct door, at 18% of steps, with regret ≈ 0.7 each.
  - The exact margin Q*(door) − Q*(listen) there is only **+0.70**. The observation head is essentially exact (KL ≈ 0.002). The margin is flipped by the reward head (correct-door error −1.79) and a belief-dependent value bias (+0.5 at confident beliefs up to +2.6 at b = 0.5). The search cannot repair a root-edge reward error.
  - **The head errors are optimisation noise, not bias.** Between best.pt and latest.pt, door-reward and value errors at decision-relevant beliefs move by 1–3 with sign changes: seed 1 from −1.76 to +0.19, seed 2 from +0.48 to −2.79 (its evaluation dip to 14.58 at collection 40), seed 3 from −0.66 to +0.91.
  - The late-run evaluation dips have the same cause. The constant learning rate on high-variance bimodal door rewards (−100/+10) and bootstrapped values makes the heads jitter by more than the 0.70 decision margin.
- **Polyak-averaged acting weights (`430f20c`), interim result on seeds 1 and 2** (`runs/sweeps/polyak`):
  - Learned planner gap to optimal: seed 1 went from −1.97 to −0.015, seed 2 from −0.04 to −0.019.
  - Cost: escape from always-listen moved from collection 10 to collection 15, the averaging lag.
  - Correction to the "late dips" reading: a 128-episode in-training evaluation has a standard error of about 2.8 (per-episode return std ≈ 31), so scores of 13–24 are within noise of the optimum. Only seed 1's 11.4 at collection 30 of `default` stood out. The in-training evaluation is too noisy to diagnose dips; the analysis's common-episode gaps are the reliable measure.
  - Running now: seeds 0, 3 and 4, extending `polyak` to 5 seeds.
- The seed-0 numbers below predate the seeding fix and are kept for history.


- **Exact references:** V_h(b0) for h = 1..5 is −1, −1.95, 2.3098, 1.7955, 2.7631; V*(b0) = 19.37.
- **Off-policy acceptance** (random data, 3000 updates):

  | Check | Result |
  |---|---|
  | Belief probe KL, linear / MLP | 0.0007 / 0.00004 |
  | Door-reward error | about 2 |
  | Mean \|V − V*\| | 4.9 |
  | Learned planner return | 21.19 ± 1.69 (V* = 19.28) |

- **Phase 6 analysis** of `runs/tiger/phase5_ratio64_seed0` (`analysis/report.json` there, regenerated at Phase 7 with deterministic latents; the pre-fix Phase 6 report differed by < 1 stderr):

  | Agent | Discounted return |
  |---|---|
  | Optimal | 19.54 ± 1.35 |
  | Decoded-belief, MLP probe | **19.46 ± 1.31** (the latent is Bayes-sufficient for decisions) |
  | Decoded-belief, linear probe | 8.59 ± 0.64 (not linearly sufficient: 2.1% wrong decisions) |
  | Learned planner | 17.08 ± 1.20 |

  - Geometry: PC1 (68% of variance) is confidence (|Spearman| with |log-odds| 0.755); PC2 (19%) is side (0.816 with log-odds). Minimality ratio 0.283, so the latent is sufficient but not minimal.
  - Bounds: the expected-form value-error bound is 0.33 against a measured 0.068. Worst-case bounds are loose (for example 55 against 0.70) because of a few rare latents.

## 5. Approaches that failed — do not repeat

- **VICReg**: in any placement it made the latent less belief-like. Pure JEPA self-prediction without reward grounding learns no belief.
- **Stochastic latent transition** (DreamerV3 discrete z) for MCTS imagination:
  - The EMA-target latent space differs from the online space the heads read.
  - Predicting the online target collapses the representation.
  - The discrete z learned even the two-outcome growl poorly.
- **Free bits on the balanced KL**: the prior never trained.
- **Symlog-space two-hot decoding**: biased. **Unbounded bins (±4.85e8)**: tail mass wrecks real-space means.
- **Mean (MuZero-style) backups with a uniform prior**: estimates degrade with more search.
- **TD(λ) value targets**: they learn the exploring policy's value, and the greedy agent listens forever (return −19.88 = Σγ^t·(−1)).
- **Bellman value targets with gradients into the filter**: the representation collapses to the always-listen fixed point.
- **20-step episodes**: the value is biased because the bootstrap latent is unanchored.
- **Using R² as the probe metric**: an untrained GRU already reaches R² = 0.93. Use KL and L1.
- **Chunked PER replay with a zero-initialized belief mid-episode**: replaced by uniform whole-episode replay. PER was not reintroduced for lack of demonstrated benefit.

## 6. Next actions (in order)

1. **Done:** the sweep was read (section 4) and README updated.
2. **Ask before pushing** `fd4d628`, `e2620c4` and later Phase 8 commits.
3. **Phase 8 (holistic), remaining:** a plan for progressively larger experiments, informed by the sweep. Proposed order:
   1. multi-seed canonical Tiger;
   2. a baseline comparison;
   3. multi-agent Tiger, whose exact tables must first be verified against Gmytrasiewicz & Doshi;
   4. I-POMDP levels via the S × M_j reduction.
4. **Known open issues to raise with the user:**
   - The learned planner (17.1) trails the decoded-belief agent (19.5); the gap comes from the learned heads, not the representation.
   - Late-run evaluation dips (5.5 at one point).
   - Slow value convergence (|V − V*| was still 4.9 after 3000 updates).
   - The latent is not linearly sufficient.
   - Worst-case bounds are loose.
   - Search throughput is about 220 transitions/s at B = 256 and is Python-bound.
