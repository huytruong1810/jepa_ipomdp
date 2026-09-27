# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research code for a model-based RL agent for partially observable (and eventually interactive, I-POMDP) domains. A recurrent JEPA world model (no observation decoder) keeps a multi-object latent belief `b_t ∈ R^(B × N_obj × D_latent)`. An open-loop MCTS plans entirely in that latent space, using learned value, reward and opponent-policy heads. README.md holds the theory (VICReg, DreamerV3 two-hot symlog, KL balancing). **README's "Repository Structure" section and HANDOFF.md are stale** (they describe the pre-review codebase and a deleted training run).

The codebase is being reviewed bottom-up in phases (domain → JEPA filter → DreamerV3 parts → MCTS → training loop → interpretability → scripts/layout → holistic). Phases 1 (domain), 2 (JEPA belief filter) and 3 (DreamerV3 components) are done. Current scope is the **single-agent canonical Tiger only**; the learned agent must match the exact solver before anything larger is run.

## Commands

The project uses `uv` with Python 3.12. Torch comes from the CUDA 12.8 index (`pytorch-cu128` in `pyproject.toml`). `recreate_venv.sh` rebuilds `.venv`.

```bash
uv run main.py                                   # train (resumes from tiger_checkpoints/latest_checkpoint.pt if present)
uv run main.py training.total_steps=400 mcts.num_simulations=10 seed=1   # any config key can be overridden

uv run pytest                                    # fast suite (slow tests excluded via addopts)
uv run pytest -m slow                            # certified infinite-horizon solve (~4 min) + world-model acceptance (~3 min, GPU)
uv run pytest tests/test_domain.py::TestExactSolver::test_optimal_actions   # single test
tensorboard --logdir tiger_tensorboard
```

The evaluation, probing and interactive scripts (`eval_*`, `probe_*`, `enjoy_*`) were removed in Phase 1 because they depended on the old environment API; they are rebuilt in Phases 6–7. Their previous logic is in commit `db269d3`.

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

**Config.** Hydra: `conf/config.yaml` (`seed`, `training`, `agent`, `model`, `mcts`) plus `conf/env/tiger.yaml`, which holds only `name` and `max_steps`. `|A|`, `|O|`, action names and γ are derived from the `FinitePOMDP` in `main.py` (`DOMAIN_BUILDERS`) and passed explicitly to the planner and trainer. Hydra writes run directories to `outputs/`.

**Pipeline wiring (`main.py`).** Episode-major collection:
- Each collection starts `env_batch_size` episodes together, all at the learned z_0 via `agent.reset()`. Every step: `a = agent.act()` (uniform during `warmup_episodes`) → `env.step(a)` → `agent.update(a, o')`. After `env.max_steps` steps the whole episodes go into `EpisodeBuffer`, which is on-device and sampled uniformly. `updates_per_collection` calls to `WorldModelTrainer.train_step` follow.
- The single-agent POMDP has no opponent. World model, heads and planner still carry opponent inputs, so they receive a singleton opponent action space (`OPPONENT_ACTION_DIM = 1`) until Phases 3–4.

**World model (`models/world_model.py`).** One latent vector; there are no object slots.
- `BeliefFilter`: learned `z_0`, `z_{t+1} = GRU([a_t, o_{t+1}], z_t)`, the counterpart of the exact `tau(b, a, o')`.
- `RecurrentJEPA` holds the online filter, an EMA `target_filter` and a deterministic `LatentPredictor`, which is JEPA self-prediction of the target latent and is used for representation learning only.
- `WorldModelTrainer.train_step` (`training/trainer.py`) unrolls whole episodes and combines:
  - JEPA mean-squared error;
  - two-hot reward and TD(λ) value, bootstrapped at truncation;
  - opponent cross-entropy;
  - observation cross-entropy on **detached** latents.
- The design follows from studies recorded in module headers. Pure JEPA gives no belief without reward grounding, and VICReg hurts (`world_model.py`, sections 3 and 5). The stochastic latent transition imagined biased beliefs and was removed.
- **Planning imagines by observation branching:** sample `o' ~ ObservationHead(z, a)`, then `z' = BeliefFilter.step(z, a, o')`. Imagined latents stay on the filter's manifold.
- Heads (`models/heads.py`): value, reward, opponent policy and observation. `TwoHotSymlog` (`models/distributions.py`) is one shared instance injected into trainer and planner. Its bins are symlog-spaced real values bounded by `FinitePOMDP.value_bound` = max|R|/(1−γ). Encoding and decoding are linear in real space, so means are unbiased. The old symlog-space decoding turned Tiger's −100/+10 door gamble (mean −45) into −2.9.

**Interpretability (`src/ipomdp/interpretability/`).** `collect_probe_dataset` produces pairs of (filter latent, exact posterior). `linear_probe` and `mlp_probe` report KL(b* ‖ probe) overall, worst-case and per posterior value. R² is not used: an untrained GRU already reaches R² = 0.93. `tests/test_world_model_acceptance.py` (slow, GPU) is the acceptance gate. It checks the belief probes, the reward head against b·R[a], the observation head against the exact P(o′|b,a), and the value head against the exact random-policy value of −606.67.

**Planner.** `LatentBeliefTreeSearch` (`planning/mcts.py`) runs PUCT over latents `(B, D)`. It creates children by observation branching (`num_observation_samples` per action) and uses MinMax Q-normalization and Dirichlet root noise. It is reviewed in Phase 4.

**Telemetry (`ipomdp/telemetry/`).** Checkpointer (`tiger_checkpoints/`), TensorBoard logger, latent/MCTS/reward visualizers (`tiger_plots/`), system monitor, profiler and execution guardrail. SIGINT saves an interrupt checkpoint.

## Conventions and gotchas

- Rules the user set for the review:
  - The Tiger domain must stay canonical.
  - No fallback paths, deprecated code or backward-compatibility shims: fail loudly instead.
  - Keep comments and docstrings verbose about design decisions.
  - Work on `main`.
- Source files start with a `# ABSOLUTE PATH: <repo-relative path>` line and a `DESIGN DECISIONS & THEORETICAL FOUNDATIONS` comment block.
- Anything used as ground truth (domain tensors, filter, solver) is float64. Statistical tests use fixed seeds and a 5-standard-error tolerance.
- With `training.compile=true` (CUDA only), the four networks are wrapped in `torch.compile` and the MCTS control flow stays eager. Outputs of compiled modules must be `.clone()`d before reuse, because CUDA-graph static buffers get overwritten.
- Inference runs under `bfloat16` autocast. Non-finite losses or two-hot targets raise `FloatingPointError`; they are never skipped or sanitized.
- Never copy CUDA tensors to the CPU with `non_blocking=True` and then read them without synchronizing. The result is stale memory, which silently corrupted every replay buffer before commit `c267482`.
- Value/reward projection layers are zero-initialized on purpose.
- Before launching training, check whether a run is already going (`pgrep -fl main.py`).
