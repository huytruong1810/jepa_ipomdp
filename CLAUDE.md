# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research code for a model-based RL agent for partially observable (and eventually interactive, I-POMDP) domains. A recurrent JEPA belief filter (no observation decoder) keeps a latent belief `z_t ∈ R^D`. A belief-tree MCTS plans over those latents using learned reward, observation and value heads. README.md gives the current overview; the module headers hold the detailed design evidence. HANDOFF.md records the review's state, history, failed approaches and next steps.

The codebase is being reviewed bottom-up in phases (domain → JEPA filter → DreamerV3 parts → MCTS → training loop → interpretability → scripts/layout → holistic). Phases 1–7 (domain, JEPA belief filter, DreamerV3 components, planner and agent, training loop, interpretability, scripts/layout) are done; Phase 8 (holistic review, starting with multi-seed results) is next. The full learning loop reaches near-optimal play on canonical Tiger with the default config (about 40 minutes); see `conf/config.yaml` section 1b. Current scope is the **single-agent canonical Tiger only**; the learned agent must match the exact solver before anything larger is run.

## Commands

The project uses `uv` with Python 3.12. Torch comes from the CUDA 12.8 index (`pytorch-cu128` in `pyproject.toml`). `recreate_venv.sh` rebuilds `.venv`.

```bash
uv run main.py                                   # train; artifacts in runs/<env>/<timestamp>_seed<seed>/
uv run main.py training.total_episodes=512 mcts.num_simulations=20 seed=1   # any config key can be overridden
uv run main.py resume=runs/tiger/<run>/checkpoints/latest.pt                # continue a run bit-for-bit

uv run pytest                                    # fast suite (slow tests excluded via addopts)
uv run pytest -m slow                            # certified V* solve, exact-model planner vs optimal, world-model acceptance (GPU)
uv run pytest tests/test_domain.py::TestExactSolver::test_optimal_actions   # single test
uv run analyze.py runs/tiger/<run>               # belief analysis of best.pt -> <run>/analysis/
uv run sweep.py <name> --seeds 0,1,2,3,4 [hydra overrides]   # train + analyse each seed, Student-t 95% CIs -> runs/sweeps/<name>/
tensorboard --logdir runs
```

The three scripts are thin shells over `ipomdp.experiments`; no script imports another. `main.py` exits 0 only when every collection completed (130 on SIGINT, 1 on a guardrail abort), which `sweep.py` relies on. `sweep.py` runs seeds sequentially, one `main.py` process each, records its condition in `runs/sweeps/<name>/sweep.json`, and re-running the same command continues an interrupted sweep (different overrides raise).

## Architecture

**Domain layer (`src/ipomdp/domain/`) is the single source of truth.** `FinitePOMDP` holds validated float64 tensors `T[a,s,s']`, `O[a,s',o]`, `R[a,s]`, `b0`, `γ`. Everything that needs the true model reads these tensors:
- `tiger.py`: `build_tiger_pomdp()` reproduces Cassandra's `tiger.95.POMDP` exactly (γ = 0.95, no observation at t = 0, uniform observation after opening a door).
- `env.py`: `BatchedPOMDPEnv` steps B episodes with one tensor op per quantity. It uses a private seeded `torch.Generator` and does not auto-reset. `reset()` emits no observation; `step(a)` returns `(o_{t+1}, r_t, truncated)`; the caller calls `reset_rows(mask)` after reading the final observation. `env.state` is privileged diagnostics only.
- `belief.py`: exact batched Bayes filter `belief_update`, `observation_distribution`, `predict_state`. This is the ground truth for probing the learned latent.
- `solver.py`: exact alpha-vector value iteration by incremental pruning with ε-pruning (Lark's filter).
  - `solve_finite_horizon` returns Γ_1..Γ_h.
  - `solve_infinite_horizon` stops at a proven bound `(γ·‖V_n − V_{n−1}‖ + 2|O|ε)/(1−γ)`; the sup-norm is computed exactly by LPs.
  - Reference values: V*(b0) = 19.3713; V_h(b0) is −1, −1.95, 2.3098, 1.7955, 2.7631 for h = 1..5 and 6.6934 for h = 10.
- Extending to I-POMDPs: fold a finite opponent-model set into T/O to get a `FinitePOMDP` over S × M_j; the same env, filter and solver then apply. Multi-agent Tiger tables are deliberately absent until they have been verified against Gmytrasiewicz & Doshi.

**Config.** Hydra: `conf/config.yaml` (`seed`, `training`, `agent`, `model`, `mcts`) plus `conf/env/tiger.yaml`, which holds only `name` and `max_steps`. `|A|`, `|O|`, action names and γ are derived from the `FinitePOMDP` built by `ipomdp.experiments` (`DOMAIN_BUILDERS`, `build_run_config`, `build_training_run`) and passed explicitly to the planner and trainer. Hydra writes each run to `runs/<env>/<timestamp>_seed<seed>/`.

**Training run (`training/run.py`, `main.py`).**
- `TrainingRun` owns every piece of state: networks, optimizer, `EpisodeBuffer`, the collection and evaluation simulators, the training and evaluation agents and planners, and all random-generator states.
- `state_dict()`/`load_state_dict()` make resume bit-exact (tested). `main.py` is a thin Hydra shell around it that handles telemetry, evaluation, checkpoints, visualization and SIGINT.
- Collection: `env_batch_size` whole episodes of `env.max_steps` (100 for Tiger). Actions are uniform during `warmup_episodes`, then come from the exploring planner (root Dirichlet noise, annealed temperature). `updates_per_collection` trainer steps follow.
- Evaluation, every `eval_every` collections: a separate greedy planner (argmax Q) on a separate simulator. It reports the mean **discounted** return, comparable to V*(b₀) = 19.37. `best.pt` tracks the best evaluation and `latest.pt` holds the full state.
- There is no opponent input anywhere. The agent models the POMDP it faces, with the opponent folded into the environment (Phase 4 decision).

**World model (`models/world_model.py`).** One latent vector; there are no object slots.
- `BeliefFilter`: learned `z_0`, `z_{t+1} = GRU([a_t, o_{t+1}], z_t)`, the counterpart of the exact `tau(b, a, o')`.
- `RecurrentJEPA` holds the online filter, an EMA `target_filter` and a deterministic `LatentPredictor`, which is JEPA self-prediction of the target latent and is used for representation learning only.
- `WorldModelTrainer.train_step` (`training/trainer.py`) unrolls whole episodes and combines:
  - JEPA mean-squared error and two-hot reward, which are the only terms that shape the representation;
  - two-hot value regressing the **Bellman optimality backup through the learned model**, `max_a [R + γ Σ_o P(o) V̄(τ)]`, computed with the planner's own `LearnedSearchModel.expand` and an EMA target value head;
  - observation cross-entropy.

  Value and observation heads read **detached** latents. Self-referential Bellman targets otherwise collapse the representation. TD(λ) targets were removed because they estimate the noisy exploring policy's value, which made the greedy agent listen forever.
- The design follows from studies recorded in module headers. Pure JEPA gives no belief without reward grounding, and VICReg hurts (`world_model.py`, sections 3 and 5). The stochastic latent transition imagined biased beliefs and was removed.
- **Planning imagines by observation branching:** sample `o' ~ ObservationHead(z, a)`, then `z' = BeliefFilter.step(z, a, o')`. Imagined latents stay on the filter's manifold.
- Heads (`models/heads.py`): value, reward and observation. `TwoHotSymlog` (`models/distributions.py`) is one shared instance injected into trainer and planner. Its bins are symlog-spaced real values bounded by `FinitePOMDP.value_bound` = max|R|/(1−γ). Encoding and decoding are linear in real space, so means are unbiased. The old symlog-space decoding turned Tiger's −100/+10 door gamble (mean −45) into −2.9.

**Interpretability (`src/ipomdp/interpretability/`, `analyze.py`).** This is contribution (3): the learned latent maps to the exact belief with quantified guarantees.
- `belief_probe.py`: `build_probe_dataset(model, filter, episodes)` replays recorded episodes (random or the agent's own) through the exact Bayes filter and the learned filter. `fit_linear_probe`/`fit_mlp_probe` return a `BeliefProbe` that decodes latents into beliefs. `evaluate_probe` reports KL and L1 on held-out episodes. R² is not used: an untrained GRU already reaches R² = 0.93.
- `error_bounds.py`: the span-Hölder lemma |α·(b−b′)| ≤ span(α)/2·‖b−b′‖₁ turns the decoding error ε into bounds, each reported next to its measured value:
  - value error ≤ L_V·ε;
  - one-step regret ≤ 2·L_Q·ε;
  - discounted policy loss ≤ 2·L_Q·ε/(1−γ).

  Both worst-case and expected (E[ε]) forms are reported. The per-action Q* sets come from `domain.solver.action_value_functions`.
- `analysis.py` (`analyze_beliefs`) compares returns on common simulator seeds:
  - the optimal agent (exact beliefs, argmax Q*);
  - the **decoded-belief agent** (learned filter plus probe, then argmax Q*), which isolates the representation;
  - the trained learned-model planner.

  It also reports geometry: the PCA spectrum, per-component rank correlation with log-odds (side) and |log-odds| (confidence), and a minimality ratio.
- `experiments/runs.py` (`analyze_run`) reloads a run directory (`.hydra/config.yaml` + `checkpoints/best.pt`), re-estimates every return on fresh simulator seeds (the training `best.pt` score is a biased max) and writes `<run>/analysis/report.json` and `geometry.png`; `uv run analyze.py runs/tiger/<run>` calls it.
- `experiments/aggregate.py` aggregates one report per seed: Student-t 95% intervals, with paired within-seed gaps (planner − optimal, etc.) as the quantities to cite, since all seeds share one analysis seed.
- `tests/test_world_model_acceptance.py` (slow, GPU) is the acceptance gate. It trains off-policy on random-policy data (3000 updates, about 24 min) and checks:
  - the belief probes;
  - the reward head against b·R[a];
  - the observation head against the exact P(o′|b,a);
  - the value head against V*(b) (mean error < 8);
  - the greedy learned-model planner against V*(b₀) = 19.28.

**Planner and agent (`planning/`, `agents/`).**
- `SearchModel` protocol (`search_model.py`) with two implementations:
  - `ExactSearchModel`: exact beliefs over the `FinitePOMDP`, with optional alpha-vector leaf values;
  - `LearnedSearchModel`: filter latents plus the reward, observation and value heads.
- `BeliefTreeSearch` (`mcts.py`):
  - PUCT with a uniform prior and optional root Dirichlet noise;
  - **exact branching over observations** and **expectimax backups**: max at decision nodes, exact expectation at chance nodes. A mean backup (MuZero style) made estimates worse with more search.
- Greedy action choice (`temperature = 0`) takes argmax Q; exploration samples visit counts shaped by the temperature.
- `PlanningAgent` tracks states with the model's own filter and plans over the same model. With the exact model and V* leaves it matches the optimal policy's return of 19.37 (slow test).

**Telemetry (`ipomdp/telemetry/`).** Covers the TensorBoard writer, the belief-geometry, search-tree and reward visualizers, the system monitor, the profiler and the guardrail (thermal cooldown plus VRAM/RSS limits, which write `emergency.pt`). Logging goes through Hydra (console and `<run dir>/main.log`). Checkpoint files are written by `training/checkpointing.py`: atomic writes, always loaded on the CPU.

## Conventions and gotchas

- Rules the user set for the review:
  - The Tiger domain must stay canonical.
  - No fallback paths, deprecated code or backward-compatibility shims: fail loudly instead.
  - Keep comments and docstrings verbose about design decisions.
  - Work on `main`.
- Source files start with a `# ABSOLUTE PATH: <repo-relative path>` line and a `DESIGN DECISIONS & THEORETICAL FOUNDATIONS` comment block.
- Anything used as ground truth (domain tensors, filter, solver) is float64. Statistical tests use fixed seeds and a 5-standard-error tolerance.
- Training forward passes run under `bfloat16` autocast on CUDA; planning, value targets and analysis run in float32 without autocast. Non-finite losses or two-hot targets raise `FloatingPointError`; they are never skipped or sanitized.
- Never copy CUDA tensors to the CPU with `non_blocking=True` and then read them without synchronizing. The result is stale memory, which silently corrupted every replay buffer before commit `c267482`.
- Value/reward projection layers are zero-initialized on purpose.
- Before launching training, check whether a run is already going (`pgrep -fl main.py`). Another project on this machine (`~/projects/ipomcp`) sometimes runs CPU-heavy experiments, which inflates timings.
