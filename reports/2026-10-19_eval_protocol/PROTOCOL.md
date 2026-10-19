# ABSOLUTE PATH: reports/2026-10-19_eval_protocol/PROTOCOL.md

# Frozen evaluation protocol (registered 2026-10-19, before any held-out number was read)

## Why
Every Wumpus screen, threshold and design choice up to 2026-10-18 was read on training seeds 0–4 and on the same 512 analysis games (analysis seed 12345). That reuses the test set: pre-registration protects each readout, not the selection made across about 30 readouts (HANDOFF §1c, gap 2). The heuristic's risk threshold of 0.4 was also chosen on those games. From now on, headline numbers come only from the held-out set below.

## Sets
| Set | Training seeds | Analysis seed | Output directory | Status |
|---|---|---|---|---|
| Development | 0–4 | 12345 | `analysis/` | Everything read so far. **Exploratory only.** |
| Held-out | 5–9 | 20261019 | `analysis_heldout/` | Confirmatory. Each claim is read **once**. |

- If a confirmatory interval is too wide to decide a claim, extend the held-out training seeds to 10–14. Never fall back to 0–4.
- Re-analysing development runs (seeds 0–4) on analysis seed 20261019 is allowed and labelled exploratory. It measures how much the 12345 games flattered the development numbers. It does not count as confirmatory, because those training seeds took part in selecting the recipe.
- Before 2026-10-19, no Wumpus run used a training seed ≥ 5 and no script used seed 20261019 (checked by grep and directory listing).

## Frozen choices
- **Agent recipe:** whole-buffer retrain + double value, `runs/sweeps/wumpus_retrain/sweep.json`:
  - `env=wumpus model.representation=decoder model.observation_horizon=5 training.updates_per_collection=64 training.total_episodes=6400 training.store_eval_episodes=true agent.epsilon=0.25 agent.epsilon_min=0.1 model.value_estimator=double training.retrain_every=50 training.retrain_updates=6400`
  - Checkpoint: `best.pt`, as every development number used. `latest.pt` is reported beside it.
- **References:**
  - Uniform random play.
  - The exact-belief heuristic at risk 0.4, the threshold frozen from the development set.
  - The exact-belief online planner (`ExactWumpusSearchModel`, HANDOFF §6). Its simulation count and rollout count are fixed on **development** games, in its module header, before any held-out play. Fixed 2026-10-19 04:20, before any held-out play, by the smoke's pre-written choice rule: full_info leaves, 1000 simulations, no rollouts (476 ± 41 on 128 development games, against 424 ± 41 at 200 and 544 for the heuristic on the same games). It plays once per analysis seed (`exact_planner_play.py`) and enters the aggregate through `sweep.py --exact-planner-games`.
- **Games:** 512 per agent, one simulator stream per analysis seed, so every agent plays the same layouts (paired).
- **Statistics** (`src/ipomdp/experiments/aggregate.py`, sections 2, 3 and 6):
  1. The mean over seeds with a Student-t 95% interval.
  2. The interquartile mean over seeds with a percentile-bootstrap 95% interval (Agarwal et al. 2021).
  3. The hierarchical bootstrap (seeds, then games) 95% interval. For gaps it is computed on per-game paired differences. **This is the interval cited for confirmatory claims**, because it alone contains both training and evaluation noise.

## Confirmatory claims (read once each on the held-out set)
- **C-a (performance).** The recipe's mean discounted return per game, with its paired gaps to random, the heuristic (risk 0.4) and the exact-belief planner. Win, death and unfinished rates are reported beside it. There is no pass threshold: these are the paper's headline numbers.
- **C-b (representation).** MLP-probe cell and heading accuracy and pit hazard error, on random and on agent histories, from the same held-out analysis.
- **C-c (model-free baseline; registered 2026-10-19 ~05:00, before any baseline run on either set).** The recipe against R2D2-lite (`src/ipomdp/baselines/recurrent_dqn.py`: recurrent double DQN, n-step, value rescaling, Ape-X epsilon ladder) at the matched budget (448,000 environment steps, 19,200 updates, the same data schedule including stored greedy evaluations, best checkpoint by the same evaluation). Quantity: the paired gap recipe minus baseline per (training seed, game), hierarchical-bootstrap 95% interval; the baseline's return, win, death and unfinished rates beside it. No pass threshold: it is reported either way.
  - **Development screen** (training seeds 0-2, analysis seed 12345, `--output analysis`; `reports/2026-10-19_eval_protocol/recurrent_dqn_baseline.py`): a 2 x 2 grid of learning rate {1e-4 (R2D2), 3e-4} x n-step {5 (R2D2), 10}, every other field at its default. Choice rule: the highest mean best-network return over the 3 seeds; if the best arm is within one standard error (over seeds) of the R2D2 defaults (1e-4, 5), keep the defaults. The choice is recorded in the module header 4 before the held-out runs.
  - **Held-out run:** the chosen setting on training seeds 5-9, analysis seed 20261019, `--output analysis_heldout`, then paired with the recipe's held-out games.
- Later method claims (for example, a new objective against this recipe) must be added to this file **before** their held-out runs start. Each states its comparison, its interval and its decision rule.

## Change log
Any change after the first held-out read is logged here with its date and reason. A change makes the claims it affects exploratory.

- (none)
