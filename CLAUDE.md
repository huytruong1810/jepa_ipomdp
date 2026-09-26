# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research code for a model-based RL agent for partially observable (and eventually interactive, I-POMDP) domains. A recurrent JEPA world model (no observation decoder) keeps a multi-object latent belief `b_t ∈ R^(B × N_obj × D_latent)`. An open-loop MCTS plans entirely in that latent space, using learned value, reward and opponent-policy heads. README.md holds the theory (VICReg, DreamerV3 two-hot symlog, KL balancing). **README's "Repository Structure" section and HANDOFF.md are stale** (they describe the pre-review codebase and a deleted training run).

The codebase is being reviewed bottom-up in phases (domain → JEPA filter → DreamerV3 parts → MCTS → training loop → interpretability → scripts/layout → holistic). Phase 1 (domain) is done. Current scope is the **single-agent canonical Tiger only**; the learned agent must match the exact solver before anything larger is run.

## Commands

The project uses `uv` with Python 3.12. Torch comes from the CUDA 12.8 index (`pytorch-cu128` in `pyproject.toml`). `recreate_venv.sh` rebuilds `.venv`.

```bash
uv run main.py                                   # train (resumes from tiger_checkpoints/latest_checkpoint.pt if present)
uv run main.py training.total_steps=400 mcts.num_simulations=10 seed=1   # any config key can be overridden

uv run pytest                                    # fast suite (slow tests excluded via addopts)
uv run pytest -m slow                            # certified infinite-horizon Tiger solve (~4 min)
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

**Pipeline wiring (`main.py`).** `main.py` builds every component by hand. The loop is: `agent.observe(o_t)` → `a_t = agent.act()` (or `act_uniformly()` during warm-up) → `env.step(a_t)` → `buffer.push(o_t, a_t, r_t)`. Truncated rows call `buffer.end_episode(final_obs=o_T, terminated=False)` and are then reset in the env, the agent and the pending observation.
- Observations reach the networks one-hot (`|O|` wide). Every episode starts with the all-zero "empty history" observation, an agent-side encoding that carries no state information.
- The single-agent POMDP has no opponent. The world model, planner and trainer still carry opponent inputs, so they receive a singleton opponent action space (`OPPONENT_ACTION_DIM = 1`) until Phases 3–4 review them.
- World model (`models/world_model.py`): `RecurrentJEPABase` wraps an extractor, the `RecurrentContextEncoder` GRU filter `b_t = f(b_{t-1}, a_{t-1}, o_t)`, and the `CausalRelationalPredictor` (stochastic latent dynamics with categorical `z_t` and KL balancing). An EMA target encoder provides the JEPA targets.
- Heads (`models/heads.py`): value and reward are 255-bin two-hot symlog (`models/distributions.py`). `DiscretePolicyHead` is the opponent model.
- `DiscreteLatentOpenLoopSearch` (`planning/mcts.py`): PUCT over latent beliefs with MinMax Q-normalization and Dirichlet root noise.
- Training: `PrioritizedSequenceBuffer` (sequence PER over a SumTree, chunks of `burn_in + train_seq_len`). `DiscreteRecurrentIPOMDPTrainer.train_sequence` combines JEPA + KL, VICReg, two-hot value/reward on TD(λ) returns, opponent cross-entropy and dream consistency.

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
- Inference runs under `bfloat16` autocast. The trainer skips the update when `combined_loss` is non-finite.
- Value/reward projection layers are zero-initialized on purpose.
- Before launching training, check whether a run is already going (`pgrep -fl main.py`).
