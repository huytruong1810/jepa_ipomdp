<!-- ABSOLUTE PATH: README.md -->

# JEPA-IPOMDP

A model-based RL agent for partially observable domains, and eventually for interactive ones (I-POMDPs). It learns a **recurrent JEPA belief filter** without an observation decoder and plans with a **belief-tree search** over the filter's latents. It also shows, with quantified error bounds, that the learned latent **is** the Bayesian belief.

The three contributions:

1. **Belief filter.** A latent belief `z_t ∈ R^D` is learned while the agent learns the dynamics.
2. **Planning.** An expectimax belief-tree search runs over that filter, branching exactly over observations.
3. **Interpretability.** Probes map the latent to the exact posterior. Span-Hölder bounds turn the decoding error into guarantees on values and decisions.

The current scope is deliberately small: the **canonical single-agent Tiger** problem (Cassandra's `tiger.95.POMDP`, γ = 0.95). The learned agent must match the exact solver there before anything larger is attempted.

## Method

### Exact reference layer (`src/ipomdp/domain/`)

A `FinitePOMDP` holds float64 tensors `T[a,s,s']`, `O[a,s',o]`, `R[a,s]`, `b0` and `γ`. It is the single source of truth for:
- the batched simulator (`env.py`);
- the exact Bayes filter τ(b, a, o′) (`belief.py`);
- an exact alpha-vector value iteration (`solver.py`). It uses incremental pruning, and a certified stopping bound computed by LPs gives V*(b₀) = **19.3713** for Tiger.

Every learned quantity is checked against these references.

### Belief filter and world model (`src/ipomdp/models/`, `src/ipomdp/training/trainer.py`)

```
z_0 = learned,      z_{t+1} = GRU([a_t, o_{t+1}], z_t)          (counterpart of τ(b, a, o'))
```

Whole episodes are unrolled and trained with the following terms:

| Term | Role |
|---|---|
| JEPA self-prediction of an EMA target filter's latent | representation |
| two-hot reward regression | representation (grounds the latent in the task) |
| observation cross-entropy, on detached latents | P(o′ \| z, a) for planning |
| two-hot value regressing `max_a [R + γ Σ_o P(o) V̄(τ(z,a,o))]` through the learned model, EMA target head, detached latents | V for search leaves |

The two-hot codec decodes linearly in real space over symlog-spaced bins bounded by max|R|/(1−γ), so its means are unbiased.

Findings that shaped this design are recorded in the module headers. For example, pure JEPA learns no belief without reward grounding. VICReg hurts. TD(λ) value targets make the greedy agent listen forever. Bellman targets with gradients into the filter collapse the representation.

### Planning (`src/ipomdp/planning/`, `src/ipomdp/agents/`)

`BeliefTreeSearch` is PUCT at decision nodes, exact branching over observations at chance nodes, and expectimax backups. It runs over a `SearchModel`:
- `ExactSearchModel` uses exact beliefs. With V* leaves it matches the optimal return.
- `LearnedSearchModel` uses filter latents plus the reward, observation and value heads. It imagines a step as `o′ ~ P̂(o′|z,a)` followed by `z′ = filter.step(z, a, o′)`, so imagined latents stay on the filter's manifold.

### Interpretability (`src/ipomdp/interpretability/`)

- Probes ψ: z ↦ b̂ (linear and MLP) are fitted on one set of episodes and scored by KL and L1 against the exact posterior on another. R² is not used, because an untrained GRU already reaches 0.93.
- The span-Hölder lemma |α·(b−b′)| ≤ span(α)/2 · ‖b−b′‖₁ gives:
  - value error ≤ L_V ε;
  - one-step regret ≤ 2 L_Q ε;
  - discounted policy loss ≤ 2 L_Q ε/(1−γ).

  Each is reported in worst-case and expected form, next to its measured value.
- A **decoded-belief agent** (learned filter, then probe, then argmax Q*) isolates the representation. Its gap to the optimal agent can only come from the latent.

## Results (canonical Tiger, 5 training seeds)

These come from `uv run sweep.py polyak --seeds 0,1,2,3,4` with the default config (about 57 min per seed). The agent plans with Polyak-averaged weights (`training/run.py`, section 4).

Every agent is evaluated on the same 512 held-out episodes. The table reports the paired gap to the optimal agent on those episodes, with Student-t 95% intervals over seeds. For reference, the optimal agent earns 18.48 on these episodes (V*(b₀) = 19.36).

| Agent | Gap to optimal (discounted return) | Per seed |
|---|---|---|
| Learned-model planner (the trained agent) | **−0.03 ± 0.06** | −0.11, −0.02, −0.02, 0.00, +0.00 |
| Decoded-belief, MLP probe | **−0.23 ± 0.53** | +0.18, −0.47, −0.22, +0.16, −0.81 |
| Decoded-belief, linear probe | −5.38 ± 3.68 | −3.09, −6.63, −2.52, −9.83, −4.83 |

- **The trained agent matches the Bayes-optimal agent** on every seed, within 0.11.
- **The latent is Bayes-sufficient for decisions.**
  - The MLP-decoded belief acting on the exact Q* matches the optimum: the interval contains 0.
  - The probe's KL to the exact posterior is below 10⁻⁴ nats.
  - 0.15% of its decisions are suboptimal.
- **The latent is not linearly sufficient.** 1.3% of linear-probe decisions are suboptimal.
- **Polyak averaging was necessary.** Without it (sweep `default`), the planner's gap was −0.51 ± 1.02, and one seed lost 1.97. The cause was online reward and value heads jittering by 1–3 across updates, while the open-vs-listen margin at posterior 0.97 is 0.70.
  - The cost is that learning escapes always-listen about 5 collections later, at collection 15 instead of 10.
- **The training-time best evaluation overstates performance** (winner's curse): 22.70 ± 1.63, against an unbiased 18.45 for the same checkpoints.
  - In-training evaluations (128 episodes, standard error about 2.8) are too noisy to compare checkpoints.

## Usage

The project uses [uv](https://docs.astral.sh/uv/) with Python 3.12 and CUDA 12.8 torch. `./recreate_venv.sh` rebuilds `.venv` from `uv.lock`.

```bash
uv run main.py                                   # train (Hydra); writes runs/<env>/<timestamp>_seed<seed>/
uv run main.py seed=1 training.total_episodes=1280 mcts.num_simulations=20   # any config key can be overridden
uv run main.py resume=runs/tiger/<run>/checkpoints/latest.pt                 # continue a run bit-for-bit

uv run analyze.py runs/tiger/<run>               # probes, bounds, returns, geometry -> <run>/analysis/
uv run sweep.py <name> --seeds 0,1,2,3,4 [overrides ...]   # train + analyse each seed, 95% CIs -> runs/sweeps/<name>/

uv run pytest                                    # fast suite
uv run pytest -m slow                            # certified solve, exact planner vs optimal, world-model acceptance (GPU)
tensorboard --logdir runs
```

With the default config one training run takes about 40 minutes on an RTX 5080.

## Layout

```
main.py  analyze.py  sweep.py       thin entry points (train / analyse a run / multi-seed experiment)
conf/                               Hydra config (config.yaml, env/tiger.yaml)
src/ipomdp/
  domain/            FinitePOMDP, canonical Tiger, simulator, exact Bayes filter, exact solver
  models/            belief filter, JEPA predictor, value/reward/observation heads, two-hot codec
  planning/          belief-tree search, exact and learned search models
  agents/            planning agent, uniform random agent
  training/          episode buffer, world-model trainer, rollouts, checkpointable TrainingRun
  interpretability/  belief probes, error bounds, belief analysis
  experiments/       config -> run, run-directory reload and analysis, seed aggregation
  telemetry/         TensorBoard, visualizers, system monitor, profiler, guardrails
tests/
```

Every source file opens with a `DESIGN DECISIONS & THEORETICAL FOUNDATIONS` block. It records why the module is built the way it is, including the approaches that were tried and failed.

## Roadmap

1. Multi-seed canonical Tiger, with confidence intervals on all of the results above.
2. A baseline comparison.
3. Multi-agent Tiger, once its exact tables have been verified against Gmytrasiewicz & Doshi.
4. I-POMDP levels, by folding a finite set of opponent models into the state (S × M_j). The same env, filter and solver then apply.
