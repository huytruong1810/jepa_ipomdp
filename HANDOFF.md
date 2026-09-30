# HANDOFF: JEPA I-POMDP (as of 2026-09-30)

Read `CLAUDE.md` first. It describes the architecture, commands and conventions. This file covers the project's goals, state, history, dead ends and next steps. The module headers (`DESIGN DECISIONS & THEORETICAL FOUNDATIONS` blocks) record the evidence behind each design choice; read the relevant header before changing a module.

## 1. Goal

Three research contributions:

1. **Belief filter.** A recurrent JEPA belief filter (no observation decoder) learned while the agent learns the dynamics.
2. **Planning.** Belief-tree search over that filter, for POMDPs now and I-POMDPs later.
3. **Interpretability.** A map from the latent to the exact (later interactive) belief, with quantified error bounds.

The project gate was that the learned agent must match the exact solver on canonical single-agent Tiger before anything larger runs. **That gate is now met across seeds** (section 4). The next stage is a baseline comparison, then multi-agent Tiger, then I-POMDP levels.

## 2. Rules the user set (unchanged; follow them)

- **Scope:**
  - Canonical single-agent Tiger only (`tiger.95.POMDP`, γ = 0.95) until the user moves on. Never add non-canonical elements to Tiger.
  - Every change needs a small-scale check against the exact references before a larger run.
- **Code:**
  - No fallback paths, no deprecated code, no backward compatibility, no schema versions. Fail loudly.
  - Code should read as if written from scratch, with verbose comments about design decisions. Follow DDD/DRY/KISS/SOLID.
  - Every source file starts with `# ABSOLUTE PATH:` and a design-decisions header.
- **Workflow:**
  - Commit on `main` (remote `github.com/huytruong1810/jepa_ipomdp`). **Ask before every push.**
  - **Ask the user about every research or design decision that is theirs.** Offer a recommended option. So far they have always picked the recommendation, but they want to be asked.
- **Communication:**
  - The user is preparing reviewer material. They want results stated plainly, including unfavourable ones. For example, the runtime comparison says honestly that exact is faster on Tiger.
  - Reviewer-facing pages should be minimal: main results only, figures over prose (section 7).
- **Machine:** WSL2 Ubuntu, RTX 5080, `uv`, Python 3.12, cu128.
  - Another project (`~/projects/ipomcp`) sometimes runs CPU-heavy experiments here. Check `uptime` and `pgrep -af "python.*(main|sweep).py"` before long runs.
  - The belief-tree search is CPU/Python-bound, so running pytest during a sweep slows the sweep.
  - Pattern for watching a detached sweep: `ps -eo args | grep "^[^ ]*python[^ ]* [^ ]*sweep.py <name>"`. A plain `pgrep -f` matches its own shell.

## 3. Current state

- **Git:** `main` is pushed through `065b1c1`. The HANDOFF, CLAUDE.md and `reports/` commit that follows it is local; ask before pushing.
- **Review phases:** 1 domain, 2 JEPA filter, 3 DreamerV3 parts, 4 planner/agent, 5 training loop, 6 interpretability, 7 scripts/layout: all done. 8 holistic: consistency review done, multi-seed validation done, remaining items in section 6.
- **Tests:** the fast suite passes (123 at `430f20c`).
  - The slow suite (`uv run pytest -m slow`, about 25 min on GPU) last passed at Phase 6.
  - Phase 8 changed seeding and added Polyak acting weights inside `TrainingRun`. The acceptance test builds its components directly, so it is probably unaffected, but **it has not been rerun since Phase 8**. Rerun it once.
- **Runs:**
  - `runs/sweeps/default` and `runs/sweeps/polyak`: 5 seeds each, default config.
  - `polyak` is the current method and the numbers to cite.
  - Older `runs/tiger/phase5_*` runs predate the seeding fix and are not reproducible from their seed.
- **Reviewer page:** https://claude.ai/artifact/Rboz6xV7xA7Ejrjomy4As5 (private, owner-shared). Reviewers loved it. Its source, data and generating scripts are in `reports/2026-09-30_reviewer_page/` (see its README).
  - The owner edits the live page themselves; for example, they renamed the title to "JEPA-IPOMDP".
  - Always read the live version with the Artifact tool before republishing, and merge onto it.

## 4. Results to cite (canonical Tiger, `runs/sweeps/polyak`, 5 seeds)

All agents play the same 512 held-out episodes. The paired gap to the optimal agent is reported with a Student-t 95% interval over seeds. The optimal agent earns 18.48 on these episodes, and V*(b₀) = 19.36.

| Agent | Gap to optimal |
|---|---|
| **Learned planner** (the trained agent) | **−0.03 ± 0.06** (per seed −0.11, −0.02, −0.02, 0.00, +0.00) |
| Decoded belief, MLP probe → exact Q* | −0.23 ± 0.53 (contains 0) |
| Decoded belief, linear probe → exact Q* | −5.38 ± 3.68 (1.3% suboptimal decisions) |

- **Policy:**
  - 99.94% of 128,000 greedy decisions match the exact optimal policy: listen at net growls 0 and ±1, open the opposite door at ±2.
  - It is a *net* count, so a left growl followed by a right growl cancels.
  - At ±2, Q*(open) − Q*(listen) = 0.70, the tightest margin. Almost every deviation is one extra listen there.
  - Source: `reports/.../scripts/policy_check_output.txt`.
- **Representation:**
  - MLP probe KL to the exact posterior is below 1e-4 nats. Minimality ratio 0.27 ± 0.03.
  - PC1 separates post-door-opening states from balanced-listening states even though both have belief 0.5, so the latent is sufficient but not minimal. PC2 carries the side.
- **Observation prediction:** KL(exact ‖ learned) is 1–6e-4 nats. Log-loss on the realised growls is 0.6484 (learned) against 0.6487 (exact). Accuracy is 58.3% against 57.8%, the Bayes ceiling.
- **Runtime (honest):** on Tiger the exact solution is far cheaper.

  | | Exact | Learned |
  |---|---|---|
  | One-off cost | 72 s certified solve | 52–59 min training per seed |
  | Acting | 0.096 ms per decision (alpha vectors) | 4.6 ms per decision (learned-model search) |

  Exact-model search costs 4.0 ms per decision, so the search is Python-bound. The learned agent's case is model-freeness and scaling (|A|·|Γ|^|O| vector growth, nested I-POMDPs), not speed on Tiger.
- **Planning horizon:**
  - All methods optimise the discounted infinite-horizon return; the effective horizon is 1/(1−γ) = 20.
  - The exact solver ran 150 exact backups, certified within 0.0096 of V*. The greedy-policy guarantee from that bound is only 2γδ/(1−γ) ≈ 0.37, but the actual decisions are unaffected because the tightest margin is 0.70.
  - The search reaches about 3.6 actions ahead on average (6 at most) with 50 simulations. V̂ or V* leaves carry the tail.
- **Before Polyak averaging** (`runs/sweeps/default`): planner gap −0.51 ± 1.02, and seed 1 lost 1.97. The training-time `best.pt` score overstates performance (22.0 against an unbiased 18.0 for the same checkpoints).
- **Exact references:** V_h(b₀) = −1, −1.95, 2.3098, 1.7955, 2.7631 for h = 1..5; V*(b₀) = 19.3713.

## 5. History (commit, then the gist)

- `ba1d3fc`, **Phase 1:** canonical Tiger rebuilt as the single source of truth (the old one had a free t = 0 observation, informative post-door growls and γ = 0.99). Added the exact Bayes filter, the certified alpha-vector solver and a seeded batched env.
- `c267482`: non-blocking device-to-host copies stored stale memory in the replay buffer (477 of 500 reads stale), silently corrupting every earlier GPU run.
- `ccc985b`, **Phase 2:** `BeliefFilter` with a learned z₀, a single latent (no slots), whole-episode training. Filter objective = reward grounding + JEPA self-prediction.
- `94c4b1e`, `2191999`, **Phase 3:**
  - Unbiased real-space two-hot codec bounded by `value_bound`.
  - Observation-branching imagination: a learned P̂(o′|z,a), with children produced by the real filter.
  - 100-step episodes.
- `d110ad9`, **Phase 4:** `SearchModel` protocol (exact and learned models) and `BeliefTreeSearch` with exact observation branching, expectimax backups and argmax-Q greedy actions. No opponent inputs anywhere.
- `682060a`, `db625d9`, `80ee623`, **Phase 5:**
  - `TrainingRun` with bit-exact resume.
  - Bellman-optimality value targets through the learned model, using an EMA target value head, with the value head on **detached** latents.
- `7a16fea`, **Phase 6:** probes, span-Hölder error bounds, the decoded-belief agent, geometry, `analyze.py`.
- `01bbe62`, **Phase 7:**
  - `ipomdp.experiments` (`runs.py`, `aggregate.py`). Thin `main.py`, `analyze.py` and `sweep.py`.
  - `main.py` exits non-zero unless every collection completed.
  - Stale files and dependencies removed, README rewritten.
- `fd4d628`, **Phase 8:** hashed seed streams (`training/seeding.py`) replace `cfg.seed + offset`, which coupled neighbouring sweep seeds.
- `e2620c4`: consistency pass over module headers, plus dead-code removal.
- `d4e37b1`, `6597c57`: the `default` sweep result.
- `5c79deb`, `430f20c`, `c77ebd7`, `065b1c1`:
  - Diagnosis: head optimisation noise flipped the 0.70-margin decision.
  - Fix: **Polyak-averaged acting weights** (`model.acting_ema_momentum = 0.99`), shared EMA helpers in `models/ema.py`.
  - The `polyak` sweep result.
- Final commit of this session: this HANDOFF, CLAUDE.md, and `reports/2026-09-30_reviewer_page/` (page source, data, scripts).

## 6. Next actions (in order)

1. **Ask before pushing** the final commit of this session.
2. **Rerun the slow suite once** (`uv run pytest -m slow`, about 25 min, idle GPU) to confirm Phase 8 did not disturb the acceptance gate.
3. **Baseline comparison. The user must choose the baseline first; ask them.** Both options reuse `sweep.py` and cost about 5 GPU-hours per 5-seed condition.
   - *Recommended:* a **decoder-based world model**: the same filter, heads and search, with the representation trained by next-observation reconstruction instead of JEPA. This isolates exactly what dropping the decoder buys.
   - *Alternative:* a model-free recurrent Q-learner (answers whether planning helps at all).
4. **Small open questions worth one run each:**
   - **Observation-head detach ablation.** The observation head reads detached latents by design, never by measurement. One acceptance-protocol run with gradients allowed would settle it (about 25 min).
   - **Polyak momentum.** 0.99 delays the escape from always-listen by about 5 collections. Try 0.98 on seeds 1–2 as a cheap check. Ask before changing the default.
   - **Interpretable training curves.** The in-training evaluation (128 episodes, standard error about 2.8) cannot resolve dips. Scoring it against the exact optimal agent on the same episodes (common random numbers) would make the curves readable. Tiger-only helper: V*/Q* take about 72 s to compute once.
5. **Multi-agent Tiger:** first verify its exact joint tables against Gmytrasiewicz & Doshi (2005), including the creak observations and their 90% accuracy. Only then add it as a separate `FinitePOMDP` builder. Never modify the canonical Tiger.
6. **I-POMDP levels** via the S × M_j reduction (`domain/pomdp.py`, section 4). The same env, filter and certified solver apply.
7. **Known limitations to keep visible:**
   - The latent is not linearly sufficient.
   - Worst-case error bounds are loose; the expected-form bounds are within a small factor of the measurements.
   - Search throughput is Python-bound: about 4.6 ms per decision at B = 256.
   - Value convergence is slow off-policy: mean |V − V*| was 4.9 after 3000 updates in the acceptance protocol.

## 7. Failed paths: do not try again

Representation and model:
- **VICReg**, in any placement: the latent became less belief-like. **Pure JEPA without reward grounding** learns no belief (probe KL 0.018, the same as an untrained network).
- **Stochastic latent transition (DreamerV3 discrete z) for imagination:**
  - It imagined biased beliefs, because the EMA-target latent space differs from the online space the heads read.
  - Predicting online targets instead collapsed the representation.
  - The discrete z learned even the two-outcome growl poorly.
- **Free bits on the balanced KL:** the prior never trained.
- **Symlog-space two-hot decoding:** Jensen-biased (the −45 door gamble decoded to −2.9). **Unbounded bins (±4.85e8)**: tail mass wrecks the real-space means.

Planning:
- **Mean (MuZero-style) backups with a uniform prior:** estimates degrade as search grows (Q(open) went from 12.80 to −2.98 at 1000 simulations).

Training:
- **TD(λ) value targets:** they learn the noisy exploring policy's value, so the greedy agent listens forever (−19.88).
- **Bellman value targets with gradients into the filter:** the self-referential targets collapse the latent to the always-listen fixed point (probe KL 0.002 → 0.045).
- **20-step episodes:** the bootstrap latent is unanchored and the value is biased.
- **Chunked prioritised replay with a zero-initialised mid-episode belief:** replaced by uniform whole-episode replay. PER was not reintroduced for lack of benefit.
- **Acting with the online (non-averaged) weights:** head jitter of 1–3 exceeds the 0.70 decision margin. Keep the Polyak acting copies.

Methodology:
- **`cfg.seed + offset` seed streams:** they couple neighbouring seeds. Always use `stream_seed(base, stream)`.
- **R² as the probe metric:** an untrained GRU already reaches 0.93. Use KL and L1.
- **Reading in-training evaluation dips as real regressions:** the standard error is about 2.8. Use the common-episode gaps from `analyze_run`.
- **Citing the training-time `best.pt` score:** it is a max over noisy evaluations (winner's curse). Cite `analyze_run`'s fresh-seed returns instead.
- **Launching a sweep while pytest or `ipomcp` load runs:** it slows the CPU-bound search a lot.
