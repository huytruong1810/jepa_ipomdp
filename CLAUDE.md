# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research code for a model-based RL agent for multi-agent partially observable domains: a neural surrogate of a Bayesian-Adaptive Interactive POMDP (BA-I-POMDP). A recurrent JEPA world model (no observation decoder) keeps a multi-object latent belief `b_t ∈ R^(B × N_obj × D_latent)`. An open-loop MCTS plans entirely in that latent space, using learned value, reward and opponent-policy heads. README.md holds the theory (VICReg, DreamerV3 two-hot symlog, KL balancing, Tiger spec). **README's "Repository Structure" section is stale**: it names `src/ipomdp/core/`, `networks.py`, `memory.py` and similar files that no longer exist. Use the layout below instead.

`BACKLOG.md` tracks known theoretical and execution defects by ID (T1–T6, E1–E2, P0–P4). `HANDOFF.md` records the state of the long-running training run and the procedure for resuming after a crash.

## Commands

The project uses `uv` with Python 3.12. Torch comes from the CUDA 12.8 index (`pytorch-cu128` in `pyproject.toml`). `recreate_venv.sh` rebuilds `.venv`.

```bash
uv run main.py                                   # train (resumes from <env>_checkpoints/latest_checkpoint.pt if present)
uv run main.py env=wumpus                        # Hydra override: switch domain
uv run main.py training.compile=true mcts.num_simulations=100   # any config key can be overridden

uv run pytest tests/ -v                          # full suite (~60 tests, ~1 min)
uv run pytest tests/test_core_utils.py -v        # fast sanity subset
uv run pytest tests/test_envs.py::TestName::test_name -v        # single test

uv run eval_kl_tiger.py      # probe vs. exact Bayesian oracle (TigerBayesianOracle): KL, NLL, Brier, accuracy
uv run probe_tiger.py        # amnesia-gate / semantic linear probe on the belief
uv run eval_tiger.py         # policy evaluation
uv run enjoy_tiger.py        # interactive terminal walkthrough
tensorboard --logdir tiger_tensorboard
```

The eval, probe and enjoy scripts are Hydra apps. They load `${env.name}_checkpoints/latest_checkpoint.pt` by a path hard-coded in each script, not from a `--checkpoint` flag. Most of them fall back to randomly initialized weights with only a warning if loading fails. `*_wumpus.py` scripts are the Wumpus equivalents.

## Architecture

**Config.** Hydra: `conf/config.yaml` (sections `training`, `model`, `mcts`) plus `conf/env/{tiger,wumpus}.yaml`. An env config supplies `obs_dim`, `action_dim_i` / `action_dim_j`, `extractor_name`, `env_kwargs` and `action_map`. Hydra writes run directories to `outputs/`.

**Registry.** `ipomdp/telemetry/registry.py` maps config strings to classes. Envs register with `@register_env("tiger")` and extractors with `@register_extractor("mlp")`. Registration runs as an import side effect: `main.py` does `import ipomdp.envs` before `registry.make_env(...)`. A new env or extractor needs the decorator, and its module must be imported from the package `__init__`.

**Pipeline wiring (`main.py`).** `main.py` builds every component by hand; there is no builder:

- `SyncVectorEnv` (`envs/vector.py`) runs `env_batch_size` parallel channels. Envs implement `IPOMDPEnv` (`interfaces.py`). Data is passed as the frozen dataclasses in `types.py` (`State`, `Observation`, `Action`, `StepResult`), keyed by agent id (`"agent_0"`, `"agent_1"`).
- World model (`models/world_model.py`):
  - `RecurrentJEPABase` wraps the extractor (`models/extractors.py`), `RecurrentContextEncoder` and `CausalRelationalPredictor`.
  - `RecurrentContextEncoder` is the GRU belief filter, `b_t = f(b_{t-1}, a_{t-1}, o_t)`.
  - `CausalRelationalPredictor` is the stochastic latent dynamics, with prior/posterior over categorical `z_t` and KL balancing.
  - An EMA target encoder provides the JEPA targets.
- Heads (`models/heads.py`): `ValueHead` and `RewardHead` are 255-bin two-hot symlog (`models/distributions.py`, `TwoHotSymlog`). `DiscretePolicyHead` is the opponent model. `ObservationProbeHead` is used only by the eval and probe scripts. Shared blocks (RMSNorm, SwiGLU residual stack, AttentionPooler) are in `models/layers.py`.
- `DiscreteLatentOpenLoopSearch` (`planning/mcts.py`) does PUCT over latent beliefs. It uses MinMax Q-normalization and Dirichlet root noise, and samples opponent actions from the opponent head.
- Agents (`agents/jepa_agent.py`):
  - `DiscreteJEPAAgent` holds the recurrent `belief` and `prev_action`, and calls the planner.
  - `StatelessAgent` is the fixed opponent. In `main.py`, `agent_1` always plays action 0 (Listen).
  - Agents must `reset_index(i)` when channel `i` terminates.
- Training:
  - `PrioritizedSequenceBuffer` (`training/replay_buffer.py`) is sequence-level PER over a SumTree. It stores chunks of length `burn_in + train_seq_len`.
  - `DiscreteRecurrentIPOMDPTrainer.train_sequence` (`training/trainer.py`) re-runs the filter over the burn-in steps with loss masked out. It then accumulates these losses per step: JEPA (SmoothL1 to the EMA target, plus KL), VICReg, two-hot value/reward regression on TD(λ) returns, opponent cross-entropy, and dream-rollout consistency. The sum is weighted by the PER importance weights and the mask. Discounting always uses `γ·(1−done)`.
- Loop: random actions for `warmup_episodes`, then MCTS with annealed temperature. A gradient step runs every `update_freq` env steps once the buffer holds `batch_size` sequences.

**Telemetry (`ipomdp/telemetry/`).**
- `ModelCheckpointer` writes `latest_checkpoint.pt`, `best_model.pt` and `interrupt_checkpoint.pt` into `<env>_checkpoints/`. On init it restores `best_loss` from the existing `best_model.pt`, so a resumed run cannot overwrite the best model.
- `MetricsLogger` writes TensorBoard logs.
- Visualizers write latent PCA and MCTS trees to `<env>_plots/`.
- `SystemTelemetryMonitor` (thermal and VRAM), `PipelineProfiler` and `ExecutionGuardrail` (writes emergency checkpoints) run inside the training loop.
- SIGINT saves an interrupt checkpoint.

## Gotchas

- With `training.compile=true` (CUDA only), the four networks are wrapped in `torch.compile` and the MCTS control flow stays eager. Outputs of compiled modules must be `.clone()`d before reuse, because CUDA-graph static buffers get overwritten. `torch.compiler.cudagraph_mark_step_begin()` should only be called when compile is enabled on CUDA.
- Inference runs under `bfloat16` autocast. Losses such as VICReg promote to float32 explicitly. The trainer skips the update when `combined_loss` is non-finite; keep that guard.
- Value/reward projection layers are zero-initialized on purpose, so the initial expected return is exactly 0. Do not "fix" this.
- Tiger observations are `[growl, creak]`, with creak `-1.0` meaning silence. There are no pseudo-observations, to stay faithful to the canonical Kaelbling / Doshi & Gmytrasiewicz benchmark.
- Source files start with a `# ABSOLUTE PATH: <repo-relative path>` line and a `DESIGN DECISIONS & THEORETICAL FOUNDATIONS` comment block. New files should follow the same style.
- `tiger_checkpoints/`, `tiger_tensorboard/` and `tiger_plots/` contain results from a multi-day training run. Do not delete or overwrite them without asking. Before resuming or relaunching training, check whether a run is already going (`pgrep -fl main.py`).
