# HANDOFF: JEPA I-POMDP (as of 2026-10-16 evening: a whole-buffer retrain is the first representation fix that plays better. Networks retrained from scratch on each double run's final buffer (2h, `retrain_play.py`) play +66.5 / -0.6 / +88.0 against best.pt (230 / 81 / 256; deaths down on all three); seed 2's 256 beats the privileged oracle + double recipe (223). Implemented as `training.retrain_every` / `retrain_updates` (default off, Tiger fingerprint identical, `d0364c6`); 3 seeds training (`runs/wumpus/retrain_chain.sh`). Earlier today: reset + catch-up (2g) FAILED; the agent's latent is limited by its growing-buffer schedule (2f))

**The next session's job:**
1. **Read the in-agent retrain runs** (`runs/wumpus/retrain_chain.sh`: current recipe + `model.value_estimator=double` + `training.retrain_every=50 training.retrain_updates=6400`, seeds 0-2, `runs/wumpus/u64_store_eval_eps_decoder_k5_double_retrain50_seed*`; launched 2026-10-16 17:26, ~6 h incl. analyze) exactly as pre-registered in `training/run.py` section 14 (PASS: >= +50 on 2 of 3 against 163.8 / 81.7 / 168.0 with deaths not up).
2. On PASS: 5 seeds with `sweep.py`, the new Wumpus recipe; then the unfinished games (6.3, macro actions). On FAIL: compare with the offline retrain (230.3 / 81.1 / 256.0) before more runs.

Section 1b holds the working conjectures and the research principles ("taste in research"), including the compute-efficiency rules (updated 2026-10-15). Section 7 lists what failed: do not repeat it.

Read `CLAUDE.md` first: it covers the architecture, commands and conventions. This file covers goals, the user's rules, the current state, every result that matters, the next steps, the dead ends and the files touched. Each module starts with a `DESIGN DECISIONS & THEORETICAL FOUNDATIONS` header with the evidence behind its design; read it before changing that module. Every diagnostic script in `reports/` explains its question, usage, pre-registered readout and result in its header. The user's auto-memory (`~/.claude/projects/-home-andyj1810-projects-jepa-ipomdp/memory/`) repeats the standing rules of section 2.

## 0. Where we are

- **Canonical Tiger: done.** The learned planner matches the exact optimum over 5 seeds (gap −0.03 ± 0.06). A decoder baseline is indistinguishable from JEPA there.
- **Single-hunter 4x4 Wumpus World: the current agent works but is far from the reference.**
  - The agent: decoder representation, 64 updates per collection, ε-uniform exploration, greedy evaluation games stored in the buffer (section 5.3 has the recipe).
  - **182 ± 62 per game over 5 seeds** (185, 166, 215, 107, 236; `runs/sweeps/wumpus_current/aggregate.json`). Wins 29.6 ± 2.7%, deaths 4.9 ± 7.8%, unfinished 65.5 ± 8.9%. Paired gap to the exact-belief heuristic (573): −391 ± 62.
  - Storing the greedy games is what keeps deaths down (−110 ± 97 without it); ε does not matter.
- **The open problem is idling.** About 65% of games end unfinished. The planner stops entering provably safe unvisited cells around step 11–14 and spends the rest of the 50-step window on no-ops (GRAB without gold, SHOOT without the arrow; section 5.9).
- **The prize (section 5.8):** letting the heuristic's navigation rules override the planner gives 482 ± 39 (+300 ± 42 paired), three quarters of it from turning towards an adjacent safe cell and stepping in.
- **Before 2026-10-14** (sections 5.6–5.9): every value-side remedy failed (time-aware value, n-step greedy returns, γ = 0.95, detached planning head, offline value targets, teacher values in play). An oracle-latent arm (`model.privileged_belief_loss`; cell 96–98%, heading 100% decodable) idled as much or more, so the value/planner was judged the bottleneck.
- **2026-10-14:** no-op self-loops are a symptom (C1 closed); the heads are exact on no-ops (no phantom reward); the oracle networks' value exceeds 999, which no game can pay, on 36–58% of live steps, and two oracle seeds refuse the gold for it. That pointed to C2, maximisation bias in the backup.

### 2026-10-15: C2 tested and confirmed on the oracle networks; the fix does not transfer to the agent's own latent

1. **FVI screen** (`fvi_screen.py`, results in its header 4; fitted value iteration on frozen buffer latents through each network's own model, then played in the planner):
   - (i) the plain max target **diverges**: mean target ~9,000–50,000 on the oracle networks, ~130,000–145,000 on the current agent; V > 999 everywhere. The backup itself creates the optimism (gate passed on all 5 networks).
   - **Oracle networks: pass.** At 50 simulations, (ii) double gains +191.1 ± 38.3 / +54.4 ± 33.7 / +212.4 ± 37.2 and (iii) clipped +177.8 ± 39.6 / +40.2 ± 34.0 / +135.7 ± 47.0 (seeds 0 / 1 / 2). Unfinished falls 98–99% → 63–77% on seeds 0 and 2. V > 999 falls to 0.4–18%.
   - **Current agent: fail.** Double −370.0 ± 44.9 / −160.3 ± 41.0, clipped −296.3 ± 44.6 / −64.6 ± 51.8 (seeds 0 / 1; seed 2 stopped because it could not change the verdict). Deaths rise to 5–16%. Its learned value is rarely impossible (V > 999 on < 1% of live steps): the converged values exploit its model errors instead.
2. **`model.value_estimator: max | double | clipped_double`** (`5dc41ce`; default `max`, Tiger bit-identical; section 9).
3. **Training with `value_estimator=double`** (`runs/wumpus/double_chain.sh`, 3 seeds per recipe, analysis on the same 512 games against each baseline seed; pass ≥ +100 on 2 of 3 with unfinished down, pre-registered):

| Recipe, seeds 0 / 1 / 2 | Baseline | With double value | Difference | Unfinished | Deaths |
|---|---|---|---|---|---|
| Oracle latent (`privileged_belief_loss`) | 55.6 / 168.0 / −5.7 | **215.7 / 231.4 / 223.1** | **+160 / +63 / +229: PASS** | 76–96% → 68–70% | 1–2% |
| Current agent | 185.5 / 166.0 / 215.2 | 163.8 / 81.7 / 168.0 | −22 / −84 / −47: FAIL | 65–70% (unchanged) | 0–4% → 4–9% |

   - Oracle + double: V̂ > 999 on ~1% of live steps (baseline 36%), and every game that perceived the gold took it (baseline: refused in 13 of 13). `best.pt` at collections 85 / 70 / 65, so not an early-network artefact. **Mean 223, the best Wumpus result so far**, but this is the privileged diagnostic arm, not an agent.
   - Current + double: idle V̂ falls (no-op GRAB 431–635 vs 626–660) but stays flat; deaths rise.
4. **Reading (C6, section 1b): the value bias and the model's accuracy are coupled.** Removing the maximisation bias helps only when the latent and heads are accurate; otherwise the corrected value exploits the model's errors (confident "safe" mistakes of the observation head, section 5.5). The 2026-10-12 conclusion ("the value/planner, not the representation, is the bottleneck") held for the max backup only. With an unbiased value, the representation is again what separates the agent (cell/heading decodable 56–59% on random histories) from the oracle arm (96–100%).
5. **User decision (2026-10-15): the representation direction.** Bring the agent's own latent closer to the oracle latent **without privileged targets**, then train with `value_estimator=double`. Plan in section 6.2.
6. Even oracle + double leaves ~68% of games unfinished (the heuristic makes 573), so the value fix is necessary but not sufficient.

- **Nothing is running.** Everything is committed and pushed (through the commit that wrote this file). Fast suite 179 passed (2026-10-15). The slow suite last passed 7/7 at `f1e5bbf`; not run since `privileged_belief_loss` and `value_estimator` (both default off, fingerprinted). Run it when the machine is free (~45 min).

### 2026-10-15 night: R0, R0', C4 and the R1a head-swap screen (all `src/` untouched)

1. **R0** (`head_shift.py` item 4, `latest.pt` of the six double runs): every head is calibrated in its confident-"safe" bin on fresh greedy games (miss rate <= 0.002); no pre-registered branch matched. **My miss:** R0 read `latest.pt` (collection 100), not the `best.pt` networks whose returns were compared.
2. **R0'** (`head_shift.py` item 5, `best.pt` networks on `latest.pt`'s buffer; pre-registered): **branch (M), moderate-risk underestimation.** On current seeds 1 and 2 most greedy deaths sit in the head's [0.02, 0.5) bins with realised / predicted 2.2 and 8.0; on no oracle run. The greedy planner selects the head's underestimates (on fresh exploring games both recipes underrate that bin ~2x). At exact danger >= 0.4 a fresh head on the same frozen latents reads 0.35 / 0.83 where the joint head reads 0.07 / 0.05, but see 4.
3. **C4** (`noop_drift_diag.py`, `noop_drift_cost.py`): the current latent moves 4-7x more on exact no-ops than the oracle latent (mean |dz|/|z| 0.10-0.22 vs 0.02-0.03), but at matched steps the drift costs no cell/heading information, and skipping no-op updates gains only +1 to +4 points. The cell/heading gap (59-66% vs 95-100% on the held-out random pool) is in tracking moves and turns. C4 is not an R1 arm.
4. **R1a** (`head_swap_screen.py`, pre-registered; 128-game smoke on seeds 0-2, stopped there): fresh heads on the frozen acting latents (single, 5-ensemble mean, 5-ensemble pessimistic) swapped into the `best.pt` current + double planner. **No arm passes:** paired -359 / -164 / -77 (fresh), -134 / -275 / -81 (mean), -187 / -159 / +34 (pessimistic); deaths 2-9% -> up to 47%. The planner exploits the swapped heads' confident "safe" errors (their [0, 0.02) bin dies 1.3-6.2% in play). The fresh-head advantage in item 2 was a selection effect at the joint planner's states.
5. **Reading (C6 sharpened):** with the double value, deaths come from the greedy planner selecting a learned head's errors; no head trained on the current latent (joint, fresh or ensembled) resists that, while the oracle latent's joint head does (deaths 1-2%). The lever stays the representation (dead reckoning of moves and turns), or robustness of the planner to model error. Also: oracle + double still leaves 68-70% of games unfinished, the same as the current recipe, so a representation fix bounds at ~223, not 573.

### 2026-10-16 early: data vs objective (user choice), then the schedule (`objective_ab.py` 2e-2g)

1. **Agent references** (scored like every `objective_ab.py` arm, on the full held-out pool; `runs/wumpus/agent_reference_probes.txt`): current recipe cell / heading 74-76% / 62-65% at `best.pt` (76-77% / 65-75% at `latest.pt`); oracle + double 97-100%. The 59-61% I quoted earlier came from the harder random-only half.
2. **2e, data vs objective** (3 seeds, 6400 updates, same code): training offline on the agent's OWN buffer (uniform over the final buffer) gives **cell 85 / 84 / 87%, heading 92 / 78 / 92%**, probe danger error 0.03-0.09, better than the pool arm (80 / 77 / 81%, 72 / 63 / 69%). Neither data nor objective limits the agent's latent: **its training does.**
3. **2f, growing-buffer schedule** (the agent's exact data schedule, offline): cell 76.6%, heading 71.9% (means), within 5 points of the agent, and held-out NLL 0.34-0.35 vs 0.26-0.29: **SCHEDULE (primacy bias, Nikishin et al. 2022).** The first few dozen games are replayed hundreds of times and the latent never recovers.
4. **2g, reset + catch-up** (running): (i) one reset at update 3200 + 3200 uniform catch-up updates; (ii) resets at 1600/3200/4800 + 1600 each. Pre-registered PASS: within 5 points of the buffer arm on cell and heading.
5. Implication if it passes: the representation direction does not need a new objective; the agent needs to retrain its world model on its whole buffer (resets with catch-up), then the double value.
6. **2g result (2026-10-16): FAIL.** The first launch (00:17) died with the machine before any reset fired; cause unknown, not memory (a rebuilt `TrainingRun` costs ~0.4 GB RAM); relaunched unchanged at 13:10 with a per-minute system log (`runs/wumpus/reset_chain_sysmon.log`). (i) one reset: cell / heading 80.2 / 76.3% (+3.6 / +4.4 over the growing arm); (ii) three resets: 76.8 / 68.1%. Both cut the held-out head NLL (0.30-0.31 vs 0.35). Matching the total budget left the final network with less whole-buffer training than the buffer arm (3200 + 3200 and 1600 + 1600 updates), which is why more resets did worse. No reset schedule goes into the agent.
7. **2h result: PASS** (`retrain_play.py` section 4). Retrained latent cell 85 / 84 / 83%, heading 91 / 90 / 79%. Paired against best.pt on the 512 analysis games: +66.5 ± 41.9 / -0.6 ± 44.3 / +88.0 ± 43.2 (230.3 / 81.1 / 256.0 vs 163.8 / 81.7 / 168.0); deaths 4.1 / 9.0 / 5.9% -> 2.9 / 4.3 / 0.6%; unfinished still 64-76%. The baseline's SHOOT idling (57% of live actions on seeds 1-2) mostly disappears. C6 confirmed from the other side: with a better own latent the double value works.
8. **Implemented** `training.retrain_every` / `training.retrain_updates` (`d0364c6`; run.py section 14): re-initialise every network and the optimizer after every N-th collection, then U uniform updates on the whole buffer. Default off; Tiger fingerprint identical; tests for raising, reset and bit-exact resume across a retrain (fast suite 182 passed). 3 seeds training (next session's job 1).

## 1. Goals

Three research contributions:
1. **Belief filter.** A recurrent JEPA belief filter (no observation decoder) learned while the agent learns the dynamics.
2. **Planning.** Belief-tree search over that filter, for POMDPs now and I-POMDPs later.
3. **Interpretability.** A map from the latent to the exact (later interactive) belief, with quantified error bounds.

Stages:
1. Canonical Tiger matches the exact solver: **done** (section 4).
2. Baseline comparison: **done** (decoder world model, section 4b).
3. **Single-hunter Wumpus World, now.** The agent should learn to play well, and the latent should be shown against the exact belief. Multi-agent behaviour is explicitly **not** wanted yet.
4. Later: multi-agent domains and I-POMDP levels.

**Honest status of "optimality"** (the user asked): the evidence is empirical and Tiger-only. There is no proof or convergence guarantee. Wumpus has no optimum to compare against.

**Tension to keep in mind:** contribution (1) is the JEPA filter, but the working Wumpus agent uses the decoder representation, because JEPA learned the danger ahead worse in the offline A/B (section 5.5). With the fixed recipe JEPA plays as well as the decoder (2 seeds, section 5.3), so contribution (1) is viable on Wumpus; its latent is a little less position- and heading-aware.

## 1b. Working conjectures and research principles ("taste in research")

The user trusts the agent's research judgement (2026-10-10: "run the experiments you think will not waste meaningless compute and post-analysis"; 2026-10-12: "decide on the best direction forward without asking me to pick an option"). These are the beliefs that currently steer the work, strongest evidence first. Each is a conjecture with a stated test, not a result.

**C6 (new, 2026-10-15; the lead). The value bias and the model's accuracy are coupled.** The double value turns the oracle networks into the best agent so far (223) and makes the current agent worse (138 vs 189 on the same seeds; deaths up). On frozen latents the converged pair values exploit the current model's errors. Implication: fix the representation and heads first, then use the double value. **Sharpened 2026-10-15 night (R0', R1a):** what the planner exploits are the head's under-reported MODERATE risks, not confident "safe" errors, and swapping in fresh or ensembled heads on the same latent makes it worse (the planner finds their errors). Only the oracle latent's head resists. Next: section 6.2 options.

**C2. Confirmed 2026-10-15 for the oracle networks: the value is flat and impossibly optimistic, sustained by the max over noisy children.** The plain max diverges on frozen latents. Double/clipped targets remove the impossible values in the screen and in training (V > 999: 36% → 1%; gold taken). For the current agent the bias is not the main problem (C6).
- **The mechanism:** V̂ has no gradient towards progress. A flat optimistic V is self-consistent only if each step's backup supplies about (1−γ)V + 1 ≈ 8. Taking the max over 5 actions × observation branches of noisy child values supplies that (maximisation bias, van Hasselt 2010).
- **Evidence:**
  - the oracle networks hold V̂ > 999, which no policy can reach, and refuse the gold for it;
  - the idle bias is ≈ 950 and the Bellman residual ≈ 5 (self-consistent, not under-trained);
  - with loops removed, the idling moves to turn cycles (C1 result);
  - on the current agent the filter drifts on no-ops (0.46·|z|), so every no-op child is a fresh noise draw for the max.
- **Test (done):** `fvi_screen.py` (header 4), then training with `model.value_estimator=double` (section 0).
- **Caveat to keep in mind:** the planner's own tree search also maxes over noisy leaves. A value trained with a double target can still be lifted by a 50-simulation search, which is why the screen plays at 1 and at 50 simulations.

**C1. Closed 2026-10-14: no-op self-loops are a symptom.** A loop child is scored r + γV̂(z), and the exact identity V(z) = max(M, r/(1−γ)) removes it without biasing decisions. Applied correctly (v2) it removes the loops; the idling moves to 2-step turn cycles and the gains stay below +40. The identity is not worth building into the method on its own. Revisit only if C2's fix leaves loops behind.

**C5. Refuted 2026-10-14: a phantom no-op reward.** The heads are exact on no-ops (`phantom_reward_diag.py`).

**C3. Values learned from another policy's states fail by compounding error.** Shown twice now (teacher plug-in, Q/A heads): offline ranking on the teacher's states says nothing about play. Any value fix must be judged on the agent's own state distribution, in play.

**C4. The current agent's filter is not belief-consistent on uninformative actions.** (Measured 2026-10-15 night: real, 4-7x the oracle latent's drift, but it costs no cell/heading information; not an R1 arm. `noop_drift_cost.py`.) The exact belief is unchanged by a failed GRAB, but the latent moves by half its norm. It is untested as a cause. With the double value in place it is one candidate difference between the agent's latent and the oracle's (which is nearly idempotent on no-ops); a cheap CPU check on stored games can measure it on the double runs.

**Principles that picked the experiments (keep using them):**
- Cheapest informative test first: replay the trained networks with one piece swapped before any training run; train only when a frozen-network test passes.
- Pre-register the readout in the script header before launching (thresholds, seeds, which networks), and read it out as written. If something learned before launch forces a change (e.g. a tolerance), write the change and its reason into the header before the runs.
- Every replay must reproduce `analysis/report.json` exactly; scripts raise otherwise.
- Separate "ranks well offline" from "plays well". Only play counts.
- Check which checkpoint a readout uses (`best.pt` can be a collection-10 network).
- State negative results and your own corrections plainly (e.g. the "action-gap" reading of 2026-10-12, corrected on 2026-10-13).
- Privileged arms (exact belief, teacher, true layout) are diagnostics that bound what is possible, never agents.
- Stop arms that can no longer change a decision; record each stop.
- **Check that an intervention actually reaches the decision.**
  - Before reading a planning-time test, confirm the modified quantity survives the search. The self-loop v1 changed leaf values that `mcts.py` overwrites on expansion; 9 GPU-hours were spent on a void test.
  - Read the planner's code path, and in the smoke check confirm the intended effect on the action mix (e.g. the flagged action's share falls).
- **When a result contradicts the intended effect** (GRAB rose when loops were penalised), stop and find out why before reading the readout.
- **Prefer "impossible-value" checks.** Values outside the attainable range (> 999 here) show a backup artefact directly, without any comparison policy. Look for such bounds first: they are cheap, and no teacher or privileged information is needed.

**Compute-efficiency rules (learned 2026-10-14; the user asked for efficient experimentation):**
- **Use CPU diagnostics on stored data before GPU plays.** The buffer holds the checkpoint's own greedy games at its collection (`store_eval_episodes`; slot = c·64 + (c/5 − 1)·128). `phantom_reward_diag.py` answers head questions in 45 s on the CPU without any play.
- **Chunk every `expand`** to ≤ a few hundred latents on the CPU and ≤ 2048 on the GPU. One unchunked 16k-latent CPU expand (~10+ GB) set off a memory reap.
- **No per-node or per-edge device syncs in search subclasses.** Batch the tensor work per `_expand` call and sync once. A per-edge `.tolist()` made the plays ~10× slower.
- **Budget the GPU before launching.**
  - A 512-game, 50-simulation play takes ~20 min when the machine is free and ~1–2 h under ipomcp's 16-worker load (load ~25). Plan chains with that factor, or ask the user to pause ipomcp.
  - 1-simulation plays take ~3 min.
  - At most 3 planner jobs on the GPU at once; the caching allocator then shows ~15.9 of 16.3 GB.
- **Give every long play a progress line** (decision counter, timestamps), so a stall shows within minutes rather than hours.
- **Profile a smoke run before a long launch** (2026-10-15): the FVI fit spent 85% of each refresh re-expanding frozen latents; expanding the children once made it ~4× cheaper with identical output. `OMP_NUM_THREADS` made no difference to the GPU-bound jobs, so use 1–2 threads beside ipomcp.
- **Cap GPU memory per job when several share the GPU** (`torch.cuda.set_per_process_memory_fraction(f, torch.cuda.current_device())`; it needs an indexed device). Without a cap, three planner jobs held ~11.6 GB of allocator cache each, and WSL spilled into system memory (a refresh took 78 s instead of 4.7 s). `BeliefTreeSearch` keeps its last trees (`self.roots`): drop the planner after a big play. A 512-game, 50-simulation play peaks at ~5.4 GB; a Wumpus `main.py` at ~3.4 GB.
- **Smoke-check every edit, even a one-liner,** before relaunching (a cap call without a device index failed 5 queued jobs at start-up, 2026-10-15).
- **Order chains by information per GPU-hour.** Put the arms that decide the branch first, and confirmations and low-prior arms last, where they can be dropped.


## 2. Rules the user set (follow them)

**Scope**
- Never add non-canonical elements to canonical Tiger (`tiger.95.POMDP`, γ = 0.95). Tiger must stay bit-reproducible.
- Every change needs a small-scale check before a larger run.
- Wumpus has no exact solver (user decision). Do not test multi-agent behaviour yet.
- **The known-layout value (936) is not a meaningful comparison** (user, 2026-10-07: "we do not do MDP here"). Compare against random play and the exact-belief heuristic only.

**Code**
- No fallback paths, no deprecated code, no backward compatibility, no schema versions. Fail loudly.
- Code should read as if written from scratch, with verbose design-decision comments; DDD/DRY/KISS/SOLID.
- Every source file starts with `# ABSOLUTE PATH:` and a design-decisions header. Exception: Hydra files must have `# @package _global_` on the very first line, above the header.
- When a config key is added, add it with its old value to the saved `.hydra/config.yaml` of every existing run so they still reload. Done for `env.discount` (0.99), `representation`, `observation_prune`, `observation_horizon`, `agent.epsilon`/`epsilon_min`, `training.store_eval_episodes`, `model.time_aware_value` (false) and `model.value_nstep`/`value_nstep_mix` (0 / 0.5).
- A new option must default to off and leave Tiger bit-identical. Prove it with a fingerprint: a small CPU `TrainingRun` (the test `_config`), 4 collections with evaluations, then compare metrics and every parameter sum against the previous commit in a `git worktree`. This was done for ε, `store_eval_episodes`, `time_aware_value` and `value_nstep`.
  - Fingerprint recipe: a script that imports `_config` and `tiger_setup` from `tests/test_training_run.py`, runs both representations for 4 collections plus evaluations, and prints `repr(metrics)` and every parameter sum. Run it in the worktree with `PYTHONPATH=<worktree>/src` and a symlinked `.venv`, then `cmp` the two outputs. Check that the worktree run imported the worktree's `src`.

**Architecture and efficiency** (user guidance of 2026-10-03 and 2026-10-04, also in memory)
- **The network architecture must never be the bottleneck.** When results look wrong, A/B-test state-of-the-art upgrades: a Transformer over history tokens with a KV-cache, residuals, (flash) attention, MoE, GQA, MLA. Use good statistics (several seeds, paired comparisons, 95% intervals).
- **Always use the most memory- and runtime-efficient practice that leaves results unchanged.** A bit-identical speed-up can go in once a fingerprint confirms it. A summation-order change needs the user's approval. Never change code under runs in flight; record which commit a run used.

**Workflow**
- Commit on `main` (remote `github.com/huytruong1810/jepa_ipomdp`). **Since 2026-10-11: after every implementation, commit and push right away, without asking** (user: "After every implementation, please go ahead and commit and push to git. No need to ask me for permissions on those."). Never force-push or rewrite history.
- **Since 2026-10-10 the user pre-authorises running the recommended experiments without asking** ("have a good taste for research; don't waste compute"). Still ask before any change to canonical Tiger.
- **2026-10-12 (user away for 24 h): "decide on the best direction forward without asking me to pick an option. I trust your 'taste in research'."** Within the Wumpus value/planner work, pick the next step yourself and record the reasoning here (section 1b). Domain changes, stage changes and canonical Tiger still need the user.
- **Still ask the user** about changes of research direction or project stage (e.g. moving to multi-agent or I-POMDP), domain changes, and anything touching canonical Tiger. Offer a recommended option first. Within the agreed direction, pick and run the experiment with the best expected information per GPU-hour. Use cheap screens first, pre-register a pass criterion, stop arms that can no longer change a decision, and record each stop.
- State results plainly, including unfavourable ones and your own mistakes.
- **Reviewer pages:** minimal, figure-first, honest.
  - The user cut the 2026-10-07 page to: domain, returns, curves, latent UMAP, a synced game player and probes.
  - No previous agents, no diagnosis or status sections, no known-layout bound and no subtext under the title.
  - The owner may edit pages live: read them with the Artifact tool before republishing.

**Machine** (WSL2 Ubuntu; RTX 5080 Laptop with 16 GB; 24 GB RAM; `uv`, Python 3.12, cu128)
- `~/projects/ipomcp` often runs heavy CPU jobs here (load 10–24 on 24 cores, ≥ 5 workers at about 1.45–1.75 GB RAM each). Check `uptime`, `free -m` and `ps -eo pid,rss,args --sort=-rss | head` before launching anything.
- **The tree search is CPU-bound.** Under ipomcp load a Wumpus run slowed from about 1.5–2 min to about 9 min per collection, which killed the JEPA runs on 2026-10-07. A full 100-collection Wumpus run takes about 2.5–3.2 h alone, and about 4 h with two in parallel.
- **RAM is the binding limit for parallel runs.** A Wumpus `main.py` holds about 2.6 GB, a Tiger acceptance run about 1.8 GB, and an `objective_ab.py` run about 2.1 GB plus the pool on the GPU. Six A/B runs plus ipomcp exhausted RAM on 2026-10-06, and Claude Code reaped a background watcher. Run at most 2–3 heavy jobs while ipomcp is busy.
- **Six parallel `objective_ab.py` runs saturate the GPU** (0.8 s per update each against 0.1 s alone), so running them in parallel gains nothing. Run 2–3 at a time.
- Detach long runs with `setsid nohup ... > log 2>&1 < /dev/null & disown`, then confirm the python process is alive. For multi-step overnight work, put the whole pipeline in one detached shell script (as `runs/wumpus/reviewer_chain.sh`); it survives reaped watchers.
- **Waiting on runs:**
  - Never use `pgrep -f`/`pkill -f "<pattern>"` with a pattern that appears in your own command line: it matches the shell itself. On 2026-10-06 `pkill -f run.sh` killed the calling shell.
  - Wait on PIDs (`while kill -0 <pid>; do sleep 60; done`) or match the python process exactly (`ps -eo pid,args | grep "^ *[0-9]* [^ ]*python[^ ]* main.py"`).
  - Background watchers stop after at most 2 h, and the memory reaper may kill them earlier: re-arm them.
- Cap CPU threads in parallel runs (`OMP_NUM_THREADS=2–4`). Tests pin torch to one thread (`tests/conftest.py`).
- **Tools outside the project** (deliberately not dependencies; they live in the session scratchpad and disappear with it):
  - UMAP: `uv venv <env>; VIRTUAL_ENV=<env> uv pip install umap-learn`.
  - Page screenshots: Playwright Chromium in a scratch venv. The browser binary is cached in `~/.cache/ms-playwright`. It needs `libnspr4`, `libnss3` and `libasound2`, which are not installed: `apt-get download` them (no sudo), unpack with `dpkg-deb -x` and set `LD_LIBRARY_PATH`.

## 3. Current state

**Git**
- **Everything is pushed** (2026-10-15). The working tree is clean. `runs/` is untracked: keep it.
- **Nothing is running** (2026-10-15 19:06). The R0 chain, `runs/wumpus/head_shift_double_chain.sh`, was stopped on the user's request 3 minutes after launch; its two partial logs were deleted. Relaunch it as in the next-session job above.
- Chains of 2026-10-14/15, all finished or stopped on purpose, with logs beside them: `c2_chain.sh` (self-loop v2), `fvi_chain.sh` then `fvi_queue.sh` (FVI screen; `fvi_queue.log` records the relaunches and the stop of current seed 2), `double_chain.sh` (training with `value_estimator=double`, both recipes, then `analyze.py`).
- **Health checks for any chain:** `cat <chain>.log`; `tail -3 <run>.<diag>.log` (progress lines should advance); `ps -eo pid,etime,args | grep "^ *[0-9]* *[0-9:-]* [^ ]*python[^ ]* \(main.py\|reports/\)"`; `nvidia-smi`; `uptime` (ipomcp load). **Never `pkill -f`** a pattern in your own command line; kill by PID (chain shell, `xargs`, then the python jobs).

**Tests**
- The fast suite passes: 179 tests (2026-10-15; 7 new in `TestValuePair`), about 1.5 min on an idle machine (6 min under load).
- **The slow suite passed 7/7 at `f1e5bbf` on 2026-10-11** (42.5 min; log `runs/slow_suite_f1e5bbf.log`). It covers every option added since 2026-10-05.
- **Checkpoints from before `52ac3fc` cannot be resumed:** their buffer state has no `greedy` flags (no back-compat, by rule). They still load for analysis (`load_trained_run` reads networks only).
- **The Tiger acceptance test now trains 6000 updates.** At 3000 the value gate (mean |V − V*| < 8) failed at 3–5 of 9 seeds. The two worst seeds drop to 5.3 and 1.1 at 6000. The seed table is in the test header; the logs are in `runs/tiger_acceptance_seeds/`.
- The input-network hoist (`6af8f8f`) stays: its paired effect on the value error is +4.2 ± 6.2, not significant.

**Runs** (`runs/` is untracked)

| Path | What |
|---|---|
| `runs/sweeps/polyak` | Tiger JEPA, 5 seeds: **the Tiger numbers to cite** |
| `runs/sweeps/decoder` | Tiger decoder baseline, 5 seeds |
| `runs/sweeps/wumpus_current/` | **The Wumpus numbers to cite**: symlinks to the 5 current-agent seeds plus `aggregate.json` (182 ± 62). Re-aggregate with `uv run sweep.py wumpus_current --seeds 0,1,2,3,4 env=wumpus model.representation=decoder model.observation_horizon=5 training.updates_per_collection=64 training.total_episodes=6400 training.store_eval_episodes=true agent.epsilon=0.25 agent.epsilon_min=0.1` |
| `runs/wumpus/u64_store_eval_eps_decoder_k5_seed{0,1,2,3,4}` | **The current Wumpus agent**, analysed (`analysis/report.json`) with `unfinished_diag` logs. Seeds 0–2 were trained at `f5740e4`, and seeds 3–4 at `f1e5bbf` (default-off options only in between). Seed 0 also has `analysis/latents_umap_input.npz` |
| `runs/wumpus/u64_store_eval_noeps_decoder_k5_seed{0,1,2}`, `u64_eps_nostore_decoder_k5_seed{0,1,2}`, `ablation_chain.sh` | Recipe ablations (ε off; store_eval off), analysed (section 5.3) |
| `runs/wumpus/horizon_ab/`, `planning_head_ab/` (+ `*_chain.sh`) | Offline A/Bs on `pools.pt` with saved networks (`obj_*.pt`): horizon k = 5/10/20, and the decoupled planning head over 25,600 updates (section 5.7) |
| `runs/slow_suite_f1e5bbf.log` | Slow suite 7/7 |
| `runs/wumpus/privileged_chain.sh`, `smoke_privileged`, `aborted_privileged_2026-10-11/` | The oracle-latent launch script (ready; section 6.1), its 6-collection small check (clean), and the cancelled first launch |
| `runs/wumpus/u64_store_eval_eps_decoder_k5_seed*.{counterfactual_diag,override_diag,override_near_diag}.log`, `analysis/{counterfactual,override}.npz`, `runs/wumpus/{counterfactual,override,override_near}_chain.sh` | The counterfactual and override diagnostics of 2026-10-11 (section 5.8) |
| `runs/wumpus/u64_store_eval_eps_jepa_k5_seed{0,1}` | **JEPA with the same recipe, finished and analysed** (`analysis/report.json`). Collections 1–40 were trained at `f5740e4`; collections 41–100 were resumed bit-exactly on 2026-10-07 at `2a6e326` (no source change) by `runs/wumpus/jepa_resume_chain.sh` |
| `runs/wumpus/u64_store_eval_eps_decoder_k5_seed*.{unfinished_diag,value_chain_diag}.log` | Unfinished-games and value-chain diagnoses of the current agent (section 5.6) |
| `runs/wumpus/u64_store_eval_eps_decoder_k5_time_seed{0,1,2}` | Current recipe + `model.time_aware_value=true` (`8ca58d0`), analysed, plus both diagnostics (time seed 0 only for value-chain): **failed** (section 5.6) |
| `runs/wumpus/u64_store_eval_eps_decoder_k5_nstep10_seed{0,1,2}` | Current recipe + `model.value_nstep=10` (`52ac3fc`), analysed, plus both diagnostics: **failed** (section 5.6) |
| `runs/wumpus/u64_store_eval_eps_decoder_k5_seed*.{grounding_screen,teacher_robustness}.log`, `{screen,teacher}_chain.sh` | The offline value-target screens of 2026-10-09 (section 5.6) |
| `runs/wumpus/u64_store_eval_eps_decoder_k5_gamma95_seed{0,1,2}`, `gamma95_chain.sh` (+ `.log`) | Current recipe + `env.discount=0.95` (`1c05c16`), analysed, plus both diagnostics: **no significant gain** (section 5.6) |
| `runs/wumpus/u128_store_eval_eps_decoder_k5_planhead_seed{0,1,2}`, `planning_head_agent_chain.sh` | Detached planning head, 128 updates: **failed** (seed 2 stopped at collection 55) |
| `runs/wumpus/smoke_gamma95` | 6-collection small check at γ = 0.95 (clean) |
| `runs/wumpus/smoke_time_aware`, `smoke_nstep` | 6-collection small checks of the two options |
| `runs/wumpus/{jepa_resume,time_aware,value_chain,nstep}_chain.sh` (+ `.log`) | The detached pipelines of 2026-10-07/08 |
| `runs/wumpus/store_eval_eps_decoder_k5_u256_seed0`, `eps_decoder_k5_u256_seed0` | The two failed 256-update runs (−84 and −283) |
| `runs/wumpus/north50_seed0`, `north50_decoder_seed0` | The first JEPA and decoder agents (110 and 85) |
| `runs/wumpus/overfit_ab/` | Offline overfitting A/B logs (stopped at about 12,800 updates) |
| `runs/wumpus/supervised_filter/pools.pt` | Cached game pool: 4096 reference + 4096 random training games and 1024 held-out games, with exact targets |
| `runs/wumpus/objective_ab*/`, `offpolicy_reference_seed0`, `head_coverage.log`, `full_seed0*`, `smoke_seed0` | Older diagnostics (section 5.5) |
| `runs/wumpus/reviewer_chain.sh`, `jepa_chain.sh` | The overnight pipeline and the JEPA launch-plus-analysis script |
| `runs/tiger_acceptance_seeds/` | Value-gate seed study logs |
| `runs/wumpus/u64_store_eval_eps_decoder_k5_privileged_seed{0,1,2}` | **The oracle-latent arm** (`model.privileged_belief_loss=true`, trained at `5416237`), analysed on `best.pt` (section 5.9) |
| `runs/wumpus/u64_store_eval_eps_decoder_k5_privileged_seed{0,1,2}_final/` | The same runs' **final networks** (`.hydra` and `checkpoints/latest.pt` symlinked; `checkpoints/best.pt` is a best.pt-format copy of latest.pt's networks). All 2026-10-12/13 diagnostics use these. Logs beside them: `<run>_final.{analyze,unfinished_diag,value_chain_diag,grounding_screen,teacher_robustness,teacher_plugin_diag,teacher_plugin_sims,q_head_screen,self_loop_diag}.log` |
| `runs/wumpus/{privileged,privileged_final,privileged_screen,teacher_plugin,teacher_plugin_sims,q_head,self_loop}_chain.sh` (+ `.log`) | The detached pipelines of 2026-10-11 to 10-14 |
| `runs/sweeps/wumpus_current/seed{0,1,2}.self_loop_diag.log` | Self-loop v1 on the current agent: baselines only (stopped; v1 void) |
| `<run>.self_loop_frozen.log` (oracle `_final` runs; later the current seeds) | Self-loop v2 (frozen loops): the C1 result |
| `<run>.phantom_reward_diag.log` (oracle `_final` runs and `runs/sweeps/wumpus_current/seed{0,1,2}`) | No phantom reward; V̂ > 999 and gold refusal |
| `<run>.fvi_screen.log`, `<run>.fvi_nets.pt` (oracle `_final` runs, `runs/sweeps/wumpus_current/seed{0,1}`); `runs/wumpus/fvi_chain.sh`, `fvi_queue.sh` (+ `.log`) | The C2 screen and its fitted nets (`fvi_screen.py` header 4). `seed2.fvi_screen.log` of the current agent holds a stale start-up failure: that seed was stopped on purpose |
| `runs/wumpus/u64_store_eval_eps_decoder_k5_privileged_double_seed{0,1,2}` | **Oracle recipe + `value_estimator=double`** (trained at `da06d34`), analysed; `.phantom_reward_diag.log` beside: **the best Wumpus result (223)** |
| `runs/wumpus/u64_store_eval_eps_decoder_k5_double_seed{0,1,2}` | Current recipe + `value_estimator=double`, analysed, `.phantom_reward_diag.log`: failed (138) |
| `runs/wumpus/smoke_double` | 6-collection small check of the double value (oracle recipe; clean) |
| `runs/wumpus/double_chain.sh` (+ `.log`), `head_shift_double_chain.sh` (+ `.log`) | The double-value training chain (done); R0 (ready to relaunch) |

**Reviewer pages**

| Page | Content | Sources |
|---|---|---|
| Tiger (2026-09-30): https://claude.ai/artifact/Rboz6xV7xA7Ejrjomy4As5 | Tiger only; the method diagrams | `reports/2026-09-30_reviewer_page/` |
| Wumpus (2026-10-07, version 6): https://claude.ai/artifact/1m1jkSud5xbFCidGMkc6VN | Domain and rules (the heuristic explained); returns (random, current agent over 3 seeds, heuristic); training curves; latent UMAP with selectable colourings; a synced game player (true map with trail, UMAP path and exact vs decoded belief, for a win, an unfinished game and a death); probes vs the supervised bound | `reports/2026-10-07_wumpus_reviewer_page/` |

- Rebuilding the Wumpus page data:
  - `scripts/export.py` writes `data.json`. The full export takes about 15–25 min because of the calibration step. `export.py games …` refreshes the example games only; `export.py add <data.json> <runs…>` merges new runs' curves and reports.
  - `scripts/dump_latents.py` then `scripts/umap_latents.py` (in the UMAP environment) write `umap.json`. The example games' latents are placed with the fitted map's `transform()`.
  - `scripts/page_meta.json` holds labels, the main-run prefix and `jepa_prefix`.
- The page already shows a "JEPA, same recipe" row and curves once the JEPA runs exist in `data.json` (merge with `export.py add`). The data still contains a now-unused `calibration` key and old runs' reports and curves, which the page ignores.
- Publish with the Artifact tool, `files` = `data.json` and `umap.json`, to the same URL.

## 4. Results: canonical Tiger (`runs/sweeps/polyak`, 5 seeds)

All agents play the same 512 held-out episodes. Gaps are paired with the optimal agent (Student-t 95% intervals over seeds). The optimal agent earns 18.48 on these episodes; V*(b₀) = 19.37.

| Agent | Gap to optimal |
|---|---|
| **Learned planner** | **−0.03 ± 0.06** (per seed −0.11, −0.02, −0.02, 0.00, +0.00) |
| Decoded belief, MLP probe → exact Q* | −0.23 ± 0.53 |
| Decoded belief, linear probe → exact Q* | −5.38 ± 3.68 (1.3% suboptimal decisions) |

- **Policy:** 99.94% of 128,000 greedy decisions are optimal; the only deviation is one extra listen at the tightest margin (0.70).
- **Latent:** the MLP probe's KL is below 1e-4 nats; minimality ratio 0.27 ± 0.03. Sufficient, but neither minimal nor linearly sufficient.
- **Observation prediction:** KL 1–6e-4 nats; accuracy 58.3% against a Bayes ceiling of 57.8%.
- **Runtime:** exact solving is far cheaper on Tiger (a 72 s solve and 0.096 ms per decision, against 52–59 min of training and 4.6 ms per decision).
- **Exact references:** V_h(b₀) = −1, −1.95, 2.3098, 1.7955, 2.7631 for h = 1..5; V*(b₀) = 19.3713.

### 4b. Decoder baseline on Tiger (indistinguishable from JEPA)

| | JEPA | Decoder |
|---|---|---|
| Learned planner − optimal | −0.03 ± 0.06 | +0.06 ± 0.12 |
| MLP-decoded agent − optimal | −0.23 ± 0.53 | −0.10 ± 0.76 |
| Linear probe KL (random / agent histories) | 0.0011 / 0.0005 | 0.0007 / 0.0005 |
| Minimality ratio | 0.27 ± 0.03 | 0.31 ± 0.04 |

Likely a ceiling effect of Tiger's one-bit observation. On Wumpus the decoder learns the better latent (section 5.5).

## 5. Wumpus World

### 5.1 Domain (`src/ipomdp/domain/wumpus.py`, `conf/env/wumpus.yaml`; user decisions of 2026-10-01/02)

- **Grid:** 4x4; start at cell 0 (bottom left) **facing North**; cell index c = x + 4y; headings 0 E, 1 N, 2 W, 3 S.
- **Placement:** 2 pits, 1 wumpus and 1 gold on pairwise distinct non-start cells: 16,380 equally likely layouts. Unwinnable layouts are kept.
- **Actions:** FORWARD, TURN_LEFT, TURN_RIGHT, GRAB, SHOOT. **No Climb**; grabbing the gold wins.
- **Rewards:** −1 per action (−10 for a shot that uses the arrow), +1000 for the gold, −1000 for death; γ = 0.99.
- **Arrow:** flies to the wall. A dead wumpus still smells, and its cell is safe.
- **Observations:** 5 percept bits (STENCH 1, BREEZE 2, GLITTER 4, BUMP 8, SCREAM 16), plus GAME_OVER = 32. Position and heading are not observed. `reset()` emits o₀, the start-cell percept.
- **End of game:** an absorbing GAME_OVER state with reward 0 fills the rest of the 50-step window.
- **Ground truth:** `WumpusBeliefTracker`, an exact enumeration filter: marginals plus the hunter's own state.
- **Model and search:** latent 128, hidden 256, observation branches pruned below 1e-3, 50 simulations. A full run is `training.total_episodes=6400` (100 collections; the default 2560 gives only 40).

### 5.2 The exact-belief heuristic (`src/ipomdp/agents/wumpus_reference.py`)

A hand-written, deterministic policy on the exact belief. It is **not optimal** and must not be called "optimal" or "V*". Each step it applies the first rule that fits:
1. P(gold here) = 1 → GRAB.
2. Walk the shortest path (BFS over (cell, heading) through provably safe cells) to the nearest provably safe unvisited cell, i.e. P(pit) = 0 and P(live wumpus) = 0. A cell with known gold comes first.
3. With the arrow and a live wumpus: from a safe spot, shoot at an unvisited, pit-free cell that may hold the wumpus. Either outcome makes that cell safe.
4. Step into the lowest-risk frontier cell if its risk (P(pit) + P(live wumpus)) is at most the threshold.
5. Otherwise TURN_LEFT (wait).

| Risk threshold (512 games) | Return | Win | Death | Unfinished |
|---|---|---|---|---|
| 0 (never gambles) | 526 ± 20 | 62.3% | 0% | 37.7% |
| **0.4 (used as "the reference")** | **573 ± 20** | 69.7% | 1.8% | 28.5% |
| 1 | 517 ± 32 | 81.2% | 18.8% | 0% |

0.4 was the best of three thresholds tried on the same 512 analysis games, so 573 is slightly optimistic. The random hunter earns −388 ± 26.

### 5.3 The current agent and its results

**Recipe.** `env=wumpus model.representation=decoder model.observation_horizon=5 training.updates_per_collection=64 training.total_episodes=6400 training.store_eval_episodes=true agent.epsilon=0.25 agent.epsilon_min=0.1`, at commit `f5740e4`. ε decays with `temperature_decay` (0.97 per collection) to 0.1.

| Seed (512 fresh games, best.pt) | Return | Win | Death | Unfinished |
|---|---|---|---|---|
| 0 | 185.5 ± 20.0 | 29.5% | 3.9% | 66.6% |
| 1 | 166.0 ± 19.6 | 26.4% | 3.3% | 70.3% |
| 2 | 215.2 ± 18.6 | 29.1% | 0.2% | 70.7% |
| 3 | 107.0 | 31.1% | 15.8% | 53.1% |
| 4 | 236.2 | 32.0% | 1.2% | 66.8% |
| **Mean, 5 seeds** (`runs/sweeps/wumpus_current/aggregate.json`) | **182 ± 62** (t-interval; 3 seeds gave 189 ± 62) | 29.6 ± 2.7% | 4.9 ± 7.8% | 65.5 ± 8.9% |

- **The death model generalises** (seed 0, `latest.pt`, 256 fresh games per policy). Steps predicted below 0.02 killed 0.2% of the time under the exploring policy and 0.1% under the greedy planner. Before the fix this was 4.1% and 9.6%.
- **The remaining gap is unfinished games, not deaths.** Diagnosed on 2026-10-07 (`unfinished_diag.py`, 3 seeds × 512 games; continued in section 5.6):
  - In unfinished games the hunter stops finding new cells around step 11–14 and then idles for about 35 steps, mostly with no-ops: SHOOT without the arrow, GRAB on an empty cell, turns.
  - In 47–50% of them a provably safe, unvisited, reachable cell still existed at the end.
  - Facing such a cell, the planner steps in only 29–42% of the time. The death head is not the reason (median P(death) about 0.001). The value head rates the latent after the step about 15 below the latent after a no-op (median γ·E[V̂ next] gap −15 to −17).
  - Under the bootstrapped (truncated-window) objective a wasted step costs only about 1 + (1−γ)·V̂ ≈ 7.5, less than those V̂ errors. Exploration progress is not valued.
- **Value overestimation, measured on the current agent:** V̂ ≈ 620–660 on idle states of unfinished games against a realised −22, and 680–750 on progress steps (`value_chain_diag.py`). The full mechanism is in section 5.6.
- **JEPA, same recipe** (2 seeds, `best.pt`, the same 512 analysis games, paired with decoder seeds 0–1):

| Seed | Decoder return | JEPA return | JEPA win / death / unfinished | MLP probe cell / heading, random histories (decoder → JEPA) | Same, agent histories |
|---|---|---|---|---|---|
| 0 | 185.5 ± 20.0 | 169.6 ± 18.9 | 24.2% / 1.8% / 74.0% | 56.4 / 59.0% → 53.6 / 50.8% | 85.5 / 86.8% → 79.9 / 76.5% |
| 1 | 166.0 ± 19.6 | 191.5 ± 18.7 | 26.4% / 0.8% / 72.9% | 58.0 / 59.5% → 53.5 / 49.7% | 83.0 / 87.9% → 87.0 / 91.0% |

  - The paired return differences are −16 and +26: **no detectable difference** with 2 seeds.
  - On random histories JEPA's latent decodes heading about 9 points worse, in line with the offline A/B (section 5.5).
- **Recipe ablations (2026-10-10, 3 seeds each, paired with seeds 0–2 of the current agent; `runs/wumpus/ablation_chain.sh`):**

| Arm | Return per seed | Death | Unfinished | Paired Δ return |
|---|---|---|---|---|
| Current agent | 185.5 / 166.0 / 215.2 | 3.9 / 3.3 / 0.2% | 66.6 / 70.3 / 70.7% | |
| ε off (`u64_store_eval_noeps_…`) | 228.9 / 190.0 / 174.2 | 0.4 / 0.8 / 3.1% | 69.7 / 74.2 / 70.3% | +8.8 ± 109.8 (no detectable effect) |
| store_eval off (`u64_eps_nostore_…`) | 110.6 / 13.7 / 111.5 | 12.5 / 17.8 / 6.1% | 60.5 / 60.9 / 71.1% | **−110 ± 97** (significant) |

  - **Storing the greedy evaluation games is what matters.** Without them, deaths return: the head is not trained on the greedy planner's own risky FORWARDs (section 5.5).
  - **ε is not needed.** Neither arm changes the idling.

### 5.4 Interpretability findings (current agent, seed 0)

- **Probes (MLP; held-out games):** cell 55–58% and heading 55–60% on random-action histories; 83–88% and 87–90% on the agent's own games. The same GRU trained on exact targets (`supervised_filter.py`) reaches 99.9% and 100%. Hazard marginals beat the constant baseline only modestly (pit error 0.070 vs 0.085).
- **UMAP** (`reports/2026-10-07_wumpus_reviewer_page/scripts/umap_latents.py`; trustworthiness 0.983):
  - The map is organised first by the percept just received.
  - A point's 10 nearest neighbours share its percept 93% of the time on random histories (36% by chance), its heading 44% (25%) and its cell 43% (24%). On the agent's own games: 96%, 74% and 71%.
  - The danger ahead varies only inside the stench/breeze clusters.
- **Example death game in the player:** on its final step the exact danger is 1.00 and the model says 0.00, and the decoded pit map puts the pit in the wrong cell. Rare deaths still come from latent errors.

### 5.5 How we got here (condensed diagnosis chain; details in the script headers under `reports/2026-10-02_wumpus_diagnostics/`)

1. **The first agents (JEPA 110 ± 20, decoder 85 ± 21) wandered and died.** Their death prediction was badly miscalibrated (`death_calibration.py`).
2. **Ruled out:**
   - search depth: 200 simulations gave no gain (`sims_diag.py`);
   - the reward head, which is accurate (`reward_diag.py`);
   - exploration data for the representation: training on reference data made play worse (`offpolicy_reference.py`);
   - the architecture: a supervised GRU reaches about 100%, and a causal Transformer was no better (`supervised_filter.py`, `ab_summary.py`).
3. **The objective is the limit for the latent.** In the offline objective A/B (`objective_ab.py`) the decoder beat JEPA significantly: heading +20 points, cell +7, danger error −0.016. The multi-step term (k = 5) did not help JEPA and was seed-dependent for the decoder. Longer offline training improved the decoder latent but made the head's death error worse. That was the first sign of overfitting.
4. **Coverage** (`head_coverage.py`): forcing dangerous FORWARD outcomes into the data halved a fresh head's danger error.
5. **The combined-fix run** (decoder, k = 5, ε, **256** updates per collection) failed at **−283 ± 17**. `greedy_deaths.py`: the planner stepped forward where its head confidently said "safe".
6. **Storing the greedy games** (still at 256 updates) gave **−84 ± 27**.
7. **`head_shift.py` (with a correction):** the head is calibrated **in-sample** but overconfident on **fresh games of the exploring policy too**, so the problem is overfitting, not only a policy shift. A fresh head on frozen latents generalised better than the jointly trained one.
8. **The offline overfitting A/B** (`objective_ab.py` arms plus `overfit_summary.py`, 2 seeds per arm):
   - with 2,048 games, held-out NLL rose from 0.30 to 0.61 while training NLL fell;
   - even with 8,192 games it overfits past about 6,400 updates;
   - weight decay up to 0.1 had no effect.
9. **Back to 64 updates per collection, keeping the other changes:** 189 ± 62 (section 5.3).

### 5.6 The unfinished games: diagnosis and the failed value remedies (2026-10-07 to 10-09)

**Diagnosis** (`unfinished_diag.py`, `value_chain_diag.py`; current agent, 3 seeds × 512 games; both replay the analysis games exactly, with outcomes equal to `report.json`):
- **Behaviour:** in unfinished games the hunter stops finding new cells around step 11–14, then spends about 35 steps on no-ops (SHOOT without the arrow, GRAB on an empty cell, turns).
- **It isn't just stuck behind risk:** in 47–50% of those games a provably safe, unvisited, reachable cell still existed at the end. Facing such a cell, the planner steps in only 29–42% of the time.
- **Not the death head:** median P(death | FORWARD) there is about 0.001. The whole one-step deficit is γ·E[V̂(next)]: the value rates the latent after the step about 15 below the latent after a no-op.
- **H2 rejected, the latent knows about progress.** Held-out MLP probes on frozen latents:
  - "safe unvisited cell directly ahead": 93–97% correct, against 79–84% for always guessing the majority answer;
  - cells visited: off by 0.15–0.37, against 1.6–2.3 for always guessing the average;
  - steps since the last new cell: off by 1.9, against 12.
- **H1, how idling sustains itself:** on idle steps V̂ ≈ 620–660 against a realised −22, and the Bellman residual is small. A no-op backup lowers V̂ by only about 1 + (1−γ)·V̂ ≈ 7 per iteration. On seeds 0 and 1 the latent barely moves under idle actions (0.17 against 4–6 on progress steps); seed 2 drifts and idles anyway.
- **H3 confirmed, the value learning is the failure.** A fresh value fitted on the same frozen latents to the **exact-belief heuristic's** Monte Carlo returns (held-out error 59–65, against 450 for a constant) prefers FORWARD at the agent's declined safe-cell steps: median +76 to +225, FORWARD first 66–73% of the time. The learned value prefers the no-op there (median −9 to −27).

**Remedy 1, time-aware value** (`model.time_aware_value`, `8ca58d0`; user choice: value head only). V(z, k), where k is the steps left in the window; the search counts k down to a terminal 0.

| Seed | Current | Time-aware | Unfinished |
|---|---|---|---|
| 0 | 185.5 ± 20.0 | 200.2 ± 19.0 | 66.6% → 70.7% |
| 1 | 166.0 ± 19.6 | 107.9 ± 19.7 | 70.3% → 69.3% |
| 2 | 215.2 ± 18.6 | 143.1 ± 20.4 | 70.7% → 67.6% |

The agent idles all the same, with GRAB instead of SHOOT. The window being treated as a cut-off is not the cause.

**Remedy 2, n-step greedy-return grounding** (`model.value_nstep=10`, mix 0.5, `52ac3fc`). On stored greedy evaluation games the value target is 0.5·Bellman + 0.5·Gₜ⁽¹⁰⁾, where G is the 10-step return of real rewards bootstrapped with the target value head.

| Seed | Current | n-step | Unfinished |
|---|---|---|---|
| 0 | 185.5 ± 20.0 | 87.9 ± 20.5 | 66.6% → 68.2% |
| 1 | 166.0 ± 19.6 | 93.4 ± 16.0 | 70.3% → 79.1% |
| 2 | 215.2 ± 18.6 | 126.1 ± 17.0 | 70.7% → 77.5% |

- **Idle V̂ fell** (650 → 290–380), as intended.
- **But the agent steps into safe cells *less*:** 14–27% of the time. Its value's no-op preference widened to −30 to −37.
- **Reading:** the greedy policy's own games end unfinished whether a state is "about to explore" or "idling", so their returns pull every value down without separating the two.

**Remedy 3, lower γ** (`env.discount=0.95`, `1c05c16`; user choice). The current recipe otherwise unchanged. Returns are at γ = 0.95, so compare with the heuristic re-scored at 0.95: 362.4 (random −227.6).

| Seed | Return (γ 0.95) | Win | Death | Unfinished (current agent) | FORWARD at safe cells, unfinished games (current) |
|---|---|---|---|---|---|
| 0 | 177.8 ± 15.7 | 30.9% | 2.3% | 66.8% (66.6%) | 55% (29%) |
| 1 | 196.3 ± 16.2 | 34.0% | 3.5% | 62.5% (70.3%) | 52% (31%) |
| 2 | 130.8 ± 16.1 | 26.0% | 7.0% | 67.0% (70.7%) | 32% (42%) |

- **Paired with the current agent:** unfinished −3.8 points (95% interval ±9.9) and wins +2.0 (±13.3). Neither is significant.
- **Normalised score** (agent − random) / (heuristic − random): 0.67 against 0.60 before.
- **Idle V̂ fell** to 200–330 (from about 650), still +210 to +330 above the realised return. The learned one-step preference at the declined steps is still the no-op (median −17 / −61 / −35).
- **Confound:** `value_bound` drops to 20,020, so the two-hot support is 5× finer. A finer support is therefore not the missing piece either.

**Offline screens of value targets (2026-10-09; user choice "screen offline, then train").** On frozen latents, a fresh MLP value is scored with the M4 test: Q(FORWARD) − Q(chosen) at the agent's declined safe-cell steps.

`grounding_screen.py` (the agent's own replay buffer from `latest.pt`, 8,960 games; early-stopped fits). FORWARD preferred, seeds 0 / 1 / 2:

| Target | FORWARD preferred | Note |
|---|---|---|
| learned V̂ | 23 / 13 / 34% | |
| **A** max(V̂, G), all buffer games | 23 / 16 / 35% | **fails**: G > V̂ on only 5% of steps, since V̂ already overestimates by 480–690 |
| G, exploring collection games | 40 / 27 / 65% | returns explain 9–13% of variance (layout luck) |
| G, all buffer games | 53 / 25 / 54% | |
| G, heuristic games (M4's target) | 69 / 62 / 39% | weaker than M4 |

A correction to the earlier write-up: a single realised return is **not** ≤ V*(b). Only its expectation over layouts is, so max(V̂, G) is also biased upward by luck.

`teacher_robustness.py` (heuristic returns; 4 protocols × 3 initialisations; 2,048 fresh heuristic games):
- Long (3000-step) fits pass on every seed, but their median gaps vary 3–6× across initialisations.
- Early-stopped fits (150–550 steps) **fail seed 2** with 512 and with 2,048 games (39–47%).
- Held-out fit quality barely differs between protocols (MSE explained 0.79–0.86).
- **Reading:** the test evaluates V on imagined children of idle-region states, which the heuristic's games barely cover. The ranking is extrapolation, not a property of the teacher's returns. M4's 66–73% is one draw from that spread, so H3 in `value_chain_diag.py` is weaker than stated: the information is in the latent, but no screened return target robustly ranks FORWARD first.

### 5.7 Latent and planning-head experiments (2026-10-09/10; after the user accepted the idling plateau)

- **Horizon A/B, finished: longer horizons do not help, and k = 20 hurts** (`runs/wumpus/horizon_ab/summary.txt`, `objective_ab.py` section 2c).
  - k20 − k5: cell −3.8 ± 5.2 points (lower on 3 of 3 seeds), heading −13 ± 24 (lower on 3 of 3). k10 lies in between.
  - Heading accuracy is still rising at 6,400 updates for k = 5 (one seed reaches 84%), but more updates overfit the observation head (section 5.5).
  - Networks are saved as `obj_*.pt`.
- **Decoupled planning head A/B, finished** (`runs/wumpus/planning_head_ab/`, `objective_ab.py` section 2d; 3 seeds, 25,600 updates).
  - **The latent improves with longer training:** cell +3 to +5 points and heading +5 to +8 by 12,800 updates; one seed reaches 86% / 91%.
  - **The joint head overfits** (held-out NLL 0.23 → 0.32–0.39).
  - **A fresh head early-stopped on NLL does not overfit, but is worse on confident "safe" misses** (3 of 3 seeds; paired +0.025 ± 0.062). It is no better on danger error. Decoupling alone is not a clear win.
- **Detached planning head at 128 updates per collection (`8b3d621`): failed.**
  - Seeds 0 and 1 returned 67 and −45, against 186 and 166 for the current agent. Deaths were 8.4% and 28.7%.
  - In-training evaluations show about 30% deaths on all 3 seeds, so seed 2 was stopped by hand at collection 55.
  - The online latent did not improve: random-history probes for cell are 52/53% and heading 50/49%, against 56/59%. The offline latent gain did not transfer.
  - By the decision rule, the 128-update joint-head control was not run.

### 5.8 Counterfactual and override diagnostics (2026-10-11)

**Counterfactual rollouts** (`counterfactual_diag.py`; current agent, 5 seeds, 300 sampled declined safe steps per seed). Each step is branched three ways from its exact history: the agent's action (A), a replicate (A2) and a forced FORWARD (F), each continued by the greedy planner. Means over seeds with t-intervals:

| Declined steps in… | Δ_true = G_F − G_A, mean | Median | Δ win | Model's one-step estimate (median) | Spearman(model, Δ) |
|---|---|---|---|---|---|
| unfinished games | **+90 ± 38** (5/5 seeds > 0) | 0 | +10.9 ± 4.9 pts | −18 ± 10 | −0.07 ± 0.20 |
| won games | −178 ± 122 | −14 | −17 pts | −9 | +0.15 |

- **Against the pre-registered readout:** (i) fails, since the median is 0, not > +15. It is mostly (ii): the planner declines a later safe step within 5 steps 49 ± 9% of the time and keeps declining 59–72% of later safe steps. Not (iii): forced steps rarely kill.
- The model's sign is wrong in unfinished games and it carries no information about the true advantage.
- In won games declining was usually right: the planner was heading for the gold. "Safe unvisited cell ahead" is not always the best target.
- Noise floor: A2 − A has median 0. Branch A repeats the original game exactly at 69–89% of steps.

**Persistent exact-belief overrides** (`override_diag.py`; the same 512 analysis games; privileged diagnostics, not agents):

| Arm | Return (5 seeds) | Paired with the planner |
|---|---|---|
| planner | 182 ± 62 | |
| forward (FORWARD whenever a provably safe unvisited cell is ahead) | 190 ± 74 | +8 ± 24 (n.s.) |
| near (the heuristic's rule a/b action when its target is ≤ 3 actions away) | 406 ± 89 | **+224 ± 75** |
| navigate (the heuristic's rule a/b action whenever it applies) | **482 ± 39** | **+300 ± 42** |

- **Idling and navigation explain about 77% of the 391-point gap to the heuristic.** The rest is risk decisions: seed 3 keeps 13.5% deaths under `navigate`.
- **Three quarters of the prize is short-range:** turn towards an adjacent safe cell, then step in (navigate − near = +76 ± 55). FORWARD alone gains nothing, so the failure is the setting-up turn, a 2–3-step credit problem. The latent decodes heading only 58% of the time on random histories (section 5.4), which may be why turns look like no-ops to the value; this is untested.

### 5.9 The oracle-latent arm (2026-10-12): a near-exact latent does not stop the idling

**Updated 2026-10-15:** this holds for the max backup only. With `value_estimator=double` the oracle recipe reaches 223 (section 0); its idling was the maximisation bias (C2). The representation matters again (C6).

Current recipe + `model.privileged_belief_loss=true`, 3 seeds, trained at `5416237` (`runs/wumpus/u64_store_eval_eps_decoder_k5_privileged_seed{0,1,2}`; `privileged_chain.sh`). Paired with `wumpus_current` seeds 0–2, same 512 analysis games, Student-t 95% intervals over 3 seeds:

| | Final networks (collection 100) | Current agent | Paired |
|---|---|---|---|
| Return | −47 / 140 / −54 | 186 / 166 / 215 | −176 ± 326 |
| Win | 0.4 / 24.2 / 0.4% | 29.5 / 26.4 / 29.1% | −20 ± 38 pts |
| Death | 0.8 / 5.3 / 1.2% | 3.9 / 3.3 / 0.2% | −0.1 ± 6.7 pts |
| Unfinished | 98.8 / 70.5 / 98.4% | 66.6 / 70.3 / 70.7% | +20 ± 43 pts |
| MLP probe, cell / heading (random histories) | 98/100, 96/100, 97/100% | 56/59, 58/60, 55/55% | +40 ± 5 / +42 ± 6 pts |
| MLP probe, pit / wumpus error (constant 0.085 / 0.058) | 0.047 / 0.031 (all seeds within 0.003) | 0.070 / 0.046 | −0.024 ± 0.003 / −0.014 ± 0.004 |

- **Premise met** on the final networks. `loss_privileged_belief` fell from about 3.3 to 0.25 on every seed.
- **Readout: value/planner (pre-registered criterion 3, more than met).** Seed 1 is within ±10 points of the current agent's unfinished rate. Seeds 0 and 2 idle almost every game (step of the last new cell 5.0 and 8.2, against 11–14). FORWARD at a provably safe unvisited cell ahead: 21 / 25 / 21% (current 29–42%). The learned Q prefers the no-op there by a median of 48 / 19 / 49 (current 9–27).
- **The navigation override** (`override_diag.py`, on `best.pt`) is worth +399 / +326 / +461 per game, more than on the current agent (+300).
- **Protocol deviation, stated:** the pre-registered readout used `best.pt`. For seeds 0 and 2, `best.pt` came from collections 10 and 20, the winner's curse on noisy in-training evaluations: a barely trained latent (cell 64 / 70%), so the premise could not be judged there. The final networks were therefore analysed as well, through `<run>_final/` (`.hydra` and `latest.pt` linked; `best.pt` is a best.pt-format copy of `latest.pt`'s networks; `privileged_final_chain.sh`). On `best.pt`: returns 56 / 168 / −6, unfinished 79 / 76 / 96%, probes 64/54, 96/100, 70/62%. The conclusion does not depend on the choice: seed 1, the one seed whose `best.pt` is late (collection 90), already met the premise there and idled as much as the current agent.
- **In-training evaluations:** seeds 0 and 2 sat at about −40 (almost all idling) from collection 25 to 100. Seed 1 recovered only at collection 90.
- **Reading:** representation errors don't explain the idling. With the hunter's cell, heading and hazards known almost exactly, the jointly trained value still ranks no-ops above the turn-then-step moves, and more confidently than with the poorer latent. The sharper latent may let the self-consistent Bellman fixed point (section 5.6, H1) separate idle states more cleanly, making the idling worse rather than better. That is a hypothesis, not tested.

**The value-target screens re-run on the oracle latents** (`privileged_screen_chain.sh`; `grounding_screen.py` and `teacher_robustness.py` on `<run>_final`, same protocol as section 5.6; logs `<run>_final.{grounding_screen,teacher_robustness}.log`):
- **The agent's own value is worse on the better latent.** V̂ − realised return on idle steps is 982 / 891 / 983 (current agent 637 / 675 / 683). FORWARD is preferred at declined safe steps 8 / 22 / 5% of the time (current 23 / 13 / 34%).
- **The agent's own returns still don't help.** max(V̂, G) gives 12 / 29 / 6%; its Monte Carlo returns give 41 / 35 / 17% (exploring games) and 33 / 43 / 13% (all buffer games). These returns explain 6–8% of the variance.
- **Teacher (heuristic) returns now rank FORWARD first under every early-stopped fit:** ES and big-ES, 18 of 18 fits over seeds × initialisations, FORWARD preferred 51–87%, medians +2 to +95. On the current agent's latents, seed 2 failed all 6 of its early-stopped fits.
  - Long (3000-step) fits got slightly less robust: 13 of 18 pass, against 18 of 18 before. Seed 2 big gives 41 / 32%, and seed 0 M4 gives 49 / 41%.
  - Held-out fit quality is about the same (MSE explained 0.83–0.86).
- **Reading:** with a near-exact latent, a value that steps into safe cells is learnable from the right returns, at least under early stopping. The agent's own Bellman fixed point moves the other way: more optimistic on idle states, and more sure of the no-op. The failure is in the value target that the agent's own data and backup produce, not in what the latent can represent.

**The teacher-value plug-in test (2026-10-12; user choice; `teacher_plugin_diag.py`, `teacher_plugin_chain.sh`).** A value fitted to 2,048 fresh heuristic games on the oracle latents replaces the learned value inside the unchanged 50-simulation search, on the same 512 analysis games. The learned-value replays reproduce `report.json` exactly.

| Seed | big-ES gain per init (median) | big (long fit) gain | Unfinished, learned → big-ES |
|---|---|---|---|
| 0 | +13 / +124 / +40 (**+40**) | −33 / +2 / −38 | 98.8% → 88–99% |
| 1 | −86 / −56 / −69 (**−69**) | −186 / −186 / −188 | 70.5% → 87–88% |
| 2 | +15 / +15 / +15 (**+15**) | +10 / +18 / +11 | 98.4% → 100% (all games identical) |

- **Pre-registered readout: the model/search also blocks progress** (gain < +50 on all 3 seeds). A value that ranks FORWARD first offline does not make the search play better.
- **`value_chain_diag.py` on the same networks:**
  - On idle steps V̂ is about 962 and nearly flat. The latent barely moves (step size 0.03–0.08, against 5.2–5.8 on progress steps), and the Bellman residual is about 5.
  - **The search picks the one-step argmax on idle steps only 27 / 35 / 27% of the time** (51–62% on progress steps).
  - With the grounded V_g the one-step ranking prefers FORWARD only weakly: 53 / 53 / 55% (medians +6, +6, +41).
- **Simulation sweep (script section 2c, `teacher_plugin_sims_chain.sh`; logs `<run>_final.teacher_plugin_sims.log`): the count does not matter.** big-ES, median over 3 initialisations, return (paired gain over the learned value at the same count):

  | Seed | 1 simulation | 10 | 50 | Learned value 1 / 10 / 50 |
  |---|---|---|---|---|
  | 0 | +37 (+79) | +4 (+54) | −7 (+40) | −42 / −50 / −47 |
  | 1 | +28 (−108) | +30 (−146) | +71 (−69) | 136 / 176 / 140 |
  | 2 | −36 (+4) | −40 (+12) | −40 (+15) | −40 / −52 / −54 |

  No seed reaches the pre-registered +100 at 1 or 10 simulations, and unfinished stays at 89–100% at 1 simulation. **Readout: the deep max backups are not the blocker; the teacher value's one-step preference is too weak to drive play.**
- **Reading at the time (the action-gap problem), corrected below:** an idle step costs about 1 + (1−γ)·V ≈ 7, and I read the fit's residual (per-game sd ≈ 460, 11% unexplained) as value errors of order 100 that swamp the gap. The action-value screen showed that residual is mostly the layout luck of single-game returns.

**The action-value head screen (2026-10-13; `q_head_screen.py`, `q_head_chain.sh`; header sections 2b and 3).** From 15,070 heuristic states, every action was branched and continued by the heuristic in the same layout (every teacher branch reproduced its game). On the oracle latents a state value V (acting by a one-step lookahead), a Q head and a common-layout advantage head A were fitted and played greedily (no search) on the 512 analysis games:

| Seed | A head gain over V, per init (median) | Q head (median) | Deaths / unfinished: V, A, Q |
|---|---|---|---|
| 0 | −34 / +61 / +38 (**+38**) | −86 | 0 / 93–96%; 2–16 / 68–84%; 14–21 / 65–74% |
| 1 | −22 / −176 / −80 (**−80**) | −114 | 0 / 93%; 6–29 / 54–83%; 11–19 / 70–79% |
| 2 | +54 / +58 / −21 (**+54**) | −129 | 0 / 99–100%; 3–18 / 71–88%; 20–30 / 59–73% |

- **Pre-registered readout: not resolvable from this latent and data** (A gains < +50 on seeds 0 and 1).
- **Offline, on held-out teacher states, every arm picks almost as well as the teacher:** G(chosen) − G(teacher) is −5 to −11 (V through the learned children), −6 to −9 (Q) and −5 to −9 (A). The "V exact children" arm (+36 to +38) is not fair: the true next percept leaks the layout.
- **Reading: compounding error off the teacher's state distribution,** not the size of the on-distribution error. V idles without dying; the action heads leave the idle states and die 2–30% of the time. **Teacher-derived values are a dead end without on-policy states**, so the teacher line (plug-in, Q/A heads) is closed.

**What the idling is made of (2026-10-13; `unfinished_diag.py` logs, all 8 runs):** after the last new cell, 50–90% of actions are GRAB without gold or SHOOT, mostly without the arrow. These are exact no-ops: the world and the exact belief stay put. On the oracle latents the filter is nearly idempotent there (GRAB's child moves 0.026·|z| at the median, against 0.6–0.9 for turns and FORWARD). On the current agent it is not (0.46·|z|): the filter drifts on no-ops. At the root, all 5 actions' Q sit within 5–20 of each other and of V̂ ≈ 744.
- **Hypothesis (self-loop):** a no-op child is its own parent, so the backup scores it r + γ·V̂(z), the parent's own optimistic value. In the value target that branch feeds V̂(z) back to itself, so an overestimate shrinks by 1% per backup. Exact identity for a deterministic self-loop: V(z) = max(M, r/(1−γ)), with M the best non-loop action. Valuing a loop child at r/(1−γ) therefore changes no fixed point.
- **Running:** `self_loop_diag.py`, see section 6.1.

## 6. Next steps

**Direction (user decision 2026-10-15): representation.** Make the agent's own latent approach the oracle latent without privileged targets, then train with `model.value_estimator=double`. Experiments within this direction need no permission (section 2). Domain changes, project-stage changes and anything touching canonical Tiger do.

### 6.1 Done 2026-10-15
- **FVI screen** (`fvi_screen.py`, header 4): oracle networks pass, current agent fails (section 0).
- **`model.value_estimator`** (`5dc41ce`): `ValuePair` (models/heads.py 2c), `WorldModelTrainer.pair_value_targets` (trainer.py 3e), run.py section 13 with both training results.
- **Double-value training, both recipes** (section 0 table): oracle + double passes (223), current + double fails (138).

### 6.2 The representation plan

**What is already known** (do not redo; sections 5.5, 5.7, 7):
- The architecture is not the limit: the same GRU reaches ~100% cell/heading from exact targets, and a causal Transformer was no better (`supervised_filter.py`). The objective is the limit.
- The decoder objective beats JEPA on the latent (heading +20 points, cell +7); longer multi-step horizons (k = 10, 20) do not help; more offline updates improve the latent but overfit the jointly trained observation head (held-out NLL up, confident "safe" misses up). Weight decay does not help; a detached, NLL-early-stopped head is not more conservative where it matters.
- What the double-value planner exploits on the current agent are deaths: the observation head's confident "safe" errors (`head_shift.py`: calibrated in-sample, overconfident on fresh games).

**Done 2026-10-15 night** (section 0, night): R0, R0' (branch M), C4 (not an arm), R1a head swaps (fail).

**2026-10-16: the user chose option 1 (data vs objective); result: training-limited, the growing-buffer schedule (section 0, 2026-10-16). Reset + catch-up (2g) FAILED; whole-buffer retrain-and-play (2h) PASSED; in-agent retrain training (run.py 14).**

**R1 options for the user (2026-10-15 night; my recommendation first):**
1. **(Recommended) Data vs objective for dead reckoning, offline.** `objective_ab.py` on the cached pool reaches cell 77-81% / heading 63-84% at the agent's 6400 updates; the agent's `best.pt` filters reach only 59-61% / 58-66% on the same held-out pool. Train the same objective offline on the agent's own buffer (a `--train-buffer <run>` option) vs the pool, 3 seeds, same updates, scored on the held-out pool. If the buffer arm stays near 60%, the agent's own (idle-heavy) data limits dead reckoning: exploration/data arms (e.g. a share of random-action games in the buffer) come next. If it reaches ~80%, the in-agent training (value loss, co-adaptation, Polyak) limits it. ~1 h, no `src/` change.
2. **Planner robustness to model error, in training:** a default-off jointly trained head ensemble whose disagreement penalises the value targets (MOPO-style), with the double value. Larger change (trainer, fingerprint, 3 x 3 h runs); R1a suggests same-recipe ensembles share their errors, so the prior is low.
3. **Accept the ~223 bound and turn to the unfinished games** on the oracle + double networks (turn-then-forward macro actions, section 6.3), i.e. the remaining 68% that no representation fix has touched.

### 6.3 Other candidates (weaker or more expensive; reasons stated)
- **A learned policy prior (AlphaZero-style).** A prior distilled from the search's own visits inherits the idling, and argmax-Q acting ignores the prior. Only worth it if acting follows visits or the prior comes from outside the search.
- **Advantage learning (action-gap increasing operator, Bellemare et al. 2016).** The counterfactual test found the one-step estimates uncorrelated with the true advantage (section 5.8), so amplifying the gap amplifies noise.
- **Turn-then-forward macro actions.** Worth revisiting on the oracle + double networks, whose value now has a gradient: about 68% of their games still end unfinished.
- **A privileged DAgger diagnostic** (teacher labels on the agent's own states) would bound what a value learned on the right states can do (C3). Diagnostic only; lower priority than (B).
- **The remaining ~23% of the gap is risk decisions** (seed 3 keeps 13.5% deaths even with the navigation override). Not diagnosed.
- If an arm is significant at 3 seeds, extend it to 5 with `sweep.py`.
- Run the slow suite once the code settles (`uv run pytest -m slow`, ~45 min).

### 6.4 Ask the user before starting
- **Reviewer page refresh**, only if the user wants a meeting. The 2026-10-07 page shows 3 seeds (189); the 5-seed numbers are in `runs/sweeps/wumpus_current`. Merge with `scripts/export.py add`, read the live page with the Artifact tool first, then republish to the same URL.
- **Direction calls:** the hybrid JEPA + observation condition; a model-free recurrent Q-learner baseline; multi-agent Tiger (verify its tables against Gmytrasiewicz & Doshi first); I-POMDP levels via the S × M_j reduction. Cheap Tiger questions: an observation-head detach ablation in JEPA mode; Polyak momentum 0.98.

## 7. Failed paths and pitfalls: do not repeat

**Representation and model**
- **VICReg** (any placement) made the latent less belief-like. **Pure JEPA without reward grounding** learns no belief.
- **A stochastic (DreamerV3 discrete) latent transition for imagination** imagined biased beliefs, and predicting online targets collapsed the representation.
- **Free bits** on the balanced KL: the prior never trained.
- **Symlog-space two-hot decoding** is Jensen-biased; **unbounded bins** wreck the real-space means.
- **A decoder of the *current* o_t:** the GRU can copy its last input. Use next-observation prediction.
- **Wumpus, JEPA as is (64 updates, no ε):** the latent does not encode the danger ahead.
- **Wumpus, multi-step k = 5 on JEPA:** no effect.
- **Wumpus, a causal Transformer filter (untuned):** no better than the GRU, and worse on cell.

**Training and data**
- **256 updates per collection on Wumpus** (replay ratio 64): the filter and head overfit; −283 ± 17, and −84 ± 27 even with greedy games stored. Keep 64 per collection, or justify more updates with held-out NLL.
- **Weight decay** (1e-2, 1e-1) against that overfitting: no effect.
- **Training the Wumpus agent on reference-agent data** (off-policy): made play worse.
- **TD(λ) value targets** learn the exploring policy's value, so the agent listens forever (Tiger).
- **Bellman value targets with gradients into the filter** collapse the latent.
- **20-step episodes:** the bootstrap is unanchored. **Chunked prioritised replay** from a zero belief: replaced by whole-episode uniform replay.
- **Acting with online (non-averaged) weights:** head jitter flips close decisions. Keep the Polyak copies.

- **Wumpus, time-aware (finite-horizon) value** (`model.time_aware_value`): no gain; still ~69% unfinished (section 5.6).
- **Wumpus, offline value-target screens** (`grounding_screen.py`, `teacher_robustness.py`): optimistic max(V̂, G) cannot lower an overestimated V̂ (fails 3/3); the agent's own MC returns are luck-dominated; heuristic returns rank FORWARD first only under some fit protocols (extrapolation). Do not build any of them as a training option without new evidence.
- **Trusting one frozen-latent fit's ranking at imagined children** (M4): check fit protocols and initialisations first.
- **Wumpus, longer multi-step horizons** (k = 10, 20, offline on `pools.pt`): no better than k = 5; k = 20 lowers cell and heading on 3 of 3 seeds.
- **Wumpus, detached planning head + 128 updates per collection** (`model.detached_planning_head`): deaths return (8–29%, returns 67 / −45). The online latent is no better, even though the offline A/B improved it with more updates. Offline latent gains on `pools.pt` do not predict the agent.
- **Wumpus, γ = 0.95** (`env.discount`): halves idle V̂ but leaves about 65% of games unfinished (paired −3.8 ± 9.9 points).
- **Wumpus, "always step into a provably safe cell ahead"** (exact-belief override, `override_diag.py`): +8 ± 24, nothing. Do not build a one-step FORWARD remedy; the missing skill is turning towards the safe cell (section 5.8).
- **Wumpus, n-step returns of the agent's own greedy games mixed into the value target** (`model.value_nstep=10`, mix 0.5): worse (88–126 against 166–215). The greedy policy's returns are uniformly poor and push all values down without separating exploring from idling.

- **Wumpus, an oracle latent (`model.privileged_belief_loss`) as a fix for the idling:** cell/heading decode 96–100%, the agent idles as much or more. The representation is not the bottleneck **under the max backup** (section 5.9; with the double value it is, C6).
- **Wumpus, teacher (heuristic) values on the oracle latent, in play** (section 5.9): a fitted state value plugged into the search at 1, 10 or 50 simulations (gains +4 to +79 on two seeds, −69 to −146 on seed 1); action-value and common-layout advantage heads acting greedily (A head medians +38 / −80 / +54; Q heads −86 to −129, deaths 11–30%). They rank as well as the teacher on the teacher's states and fail on the agent's own. Do not pursue teacher-derived values without on-policy states.
- **Reading a fit's residual on single-game returns as its value error:** the residual is mostly layout luck (the "action-gap problem" reading of 2026-10-12, corrected on 2026-10-13). Measure ranking quality directly (G(chosen) − G(teacher) on branched states).
- **Using the true next percept to score a one-step lookahead** ("V exact children"): it leaks the layout (it shows whether FORWARD kills) and looks +37 better than the teacher. Not a fair arm.

**Planning**
- **Mean (MuZero-style) backups with a uniform prior** degrade as search grows. Use expectimax.
- **Wumpus, the no-op self-loop identity at planning time** (`self_loop_diag.py` v2, frozen loops): +22 / +14 / +38. The idling moves to turn cycles. Do not build it into the method alone (C1 closed).
- **Changing a child's value by wrapping `LearnedSearchModel.expand`** only changes *unexpanded leaves*. `BeliefTreeSearch` expands every child it selects, and an expanded node's value is max_a Q over its own children (`mcts.py`, `DecisionNode.value`). Self-loop v1 was void for this reason (9 GPU-hours). To pin a node's value, stop it from being expanded (`FrozenLoopSearch` in `self_loop_diag.py`).
- **Per-edge Python/GPU work inside a search subclass** (one `.tolist()` per edge): ~10× slower plays. Batch per `_expand` call.
- **More simulations on Wumpus** (200 vs 50): no gain.

**Analysis and methodology**
- **Inferring a head error from an action-mix shift without checking the head:** the "phantom GRAB reward" guess of 2026-10-14 was wrong (the heads are exact). Measure the head directly on stored states first (`phantom_reward_diag.py`, 45 s on the CPU).
- **Judging calibration on the replay buffer:** it is in-sample. Always check fresh games (`head_shift.py` does both). This mistake was made once (the first `head_shift.py` reading).
- **`cfg.seed + offset` seeds** couple runs. Use `stream_seed`.
- **R² as a probe metric:** an untrained GRU already reaches 0.93. Use KL/L1 on Tiger, and the error against the constant (and majority) baseline on Wumpus.
- **Reading in-training evaluation dips, or citing the `best.pt` score:** noise and a winner's curse. Re-estimate on fresh seeds with `analyze.py`.
- **Reading a pre-registered readout off `best.pt` alone:** the winner's curse can pick a collection-10 network (oracle-latent seeds 0 and 2), which says nothing about the trained agent. Check `checkpoint_collection`, and analyse `latest.pt` as well when it is early (section 5.9).
- **Judging a gate or an A/B from one seed:** the Tiger value gate ranged from 0.7 to 24 across seeds.
- **Choosing the reference threshold and reporting it on the same games** inflates it a little. Say so when citing 573.
- **Forgetting `training.total_episodes=6400`** on a Wumpus run gives only 40 collections.
- **Changing initialisation order in `TrainingRun`** breaks bit-exact reproduction. Build new modules after the existing ones and only when enabled.
- **Hydra `@package` below a header comment** is silently ignored. It must be line 1.
- **Comparing metric dicts with `==`:** `mean_win_step` is NaN when no game is won. Compare `repr()`.
- **Harness runs that do not save networks** (`objective_ab.py`): later diagnostics need retraining. Save the final networks next time.
- **Expanding a whole data set at once through `LearnedSearchModel.expand`** (N × A × O children; 20,000 states → 6.4 M rows) runs out of GPU memory. Chunk it (`value_chain_diag.py`, `one_step`).
- **`echo "... $(date ...) exit $?"`** logs the exit status of `date`, not of the job, so crashed runs looked successful once. Capture `status=$?` first.

**Value targets (2026-10-15)**
- **The double value on the agent's own latent** (`value_estimator=double`, current recipe): −22 / −84 / −47, deaths up. The frozen-latent screen predicted it. Only combine it with a better representation (C6).
- **Fitted value iteration with the plain max through the learned model** diverges (to 10^4–10^5) on frozen latents. Never use it as a value target.

**Machine and tooling**
- **Several uncapped GPU planner jobs on WSL** (2026-10-15): three FVI jobs held ~11.6 GB of cache each and spilled into system memory (fit 78 s per refresh instead of 4.7 s; ~2.5 h lost with the two relaunches below). Cap per process (section 1b).
- **`torch.cuda.set_per_process_memory_fraction(f, torch.device("cuda"))`** raises: it needs an index (`torch.cuda.current_device()`).
- **A recorder holding the planner after a play** keeps the search trees (`BeliefTreeSearch.roots`, ~2+ GB at 512 games); under a cap the next step ran out of memory.
- **`kill -0 0` always succeeds** (it signals the process group): a "wait for PID" loop given 0 waits forever.
- **Running heavy jobs under ipomcp load:** CPU-bound search slows about 4×, and parallel runs can exhaust RAM (section 2, Machine).
- **Six parallel GPU jobs** (pool builds or A/Bs) gain nothing. **Six parallel planner diagnostics** (`self_loop_diag.py`, 2.6 GB each) filled the 16 GB GPU; WSL spilled to shared memory and the plays slowed > 10× (13 h lost, 2026-10-14). Run at most 3, and check `nvidia-smi` (it can take 1–2 min to answer under load) shortly after launch.
- **`pkill -f`/`pgrep -f` self-matches** killed the calling shell twice (2026-10-06, 2026-10-14). Kill by PID.
- **The page-data export plays 2 × 512 greedy games per run** for calibration; it took over 50 min under load. It now uses 256.
- **Scratchpad files vanish with the session.** Keep anything reproducible in `reports/` or `runs/`.
- **An unchunked CPU `expand` of 16,000 latents** (≈ 2.6M children, ~10+ GB) set off Claude Code's memory reaper, alongside ipomcp's 16 workers (2026-10-14). Chunk.
- **Long chains without progress lines:** a 10× slowdown went unnoticed for hours. Print a decision or refresh counter with timestamps.

## 8. History (commit, then the gist)

**Pushed (through `9a9b726`)**
- `ba1d3fc` Phase 1: canonical Tiger, exact Bayes filter, certified solver, seeded env. `c267482`: non-blocking copy corruption fixed.
- `ccc985b` Phase 2 (`BeliefFilter`, reward grounding plus JEPA); `94c4b1e`, `2191999` Phase 3 (unbiased two-hot, observation-branching imagination); `d110ad9` Phase 4 (search model, expectimax tree search); `682060a`, `db625d9`, `80ee623` Phase 5 (`TrainingRun`, Bellman targets); `7a16fea` Phase 6 (probes, error bounds); `01bbe62` Phase 7 (experiments package); `fd4d628`, `e2620c4` Phase 8.
- `5c79deb` … `065b1c1`: Polyak acting weights; `polyak` sweep (Tiger gate met). `18f7454`: Tiger reviewer-page sources. `d8871f1`: decoder baseline switch. `9a9b726`: decoder sweep launched.

**Pushed 2026-10-11 (`61991c7` … `43a4dee`)**
- `61991c7`: Tiger decoder result.
- `4d1ace5`, `90b489d`, `ae2d108`, `123fa81`: Wumpus domain, the domain-generic stack (Tiger bit-identical), the North start and 50-step window, the reference heuristic.
- `feb65bd` … `674a865`: Wumpus diagnostics, supervised bound, Transformer A/B, multi-step term (`2bbebac`), objective A/Bs.
- `62deb68`, `6af8f8f`: runtime profile; the hoisted `input_net` (user-approved).
- `a2aa606` … `07d3e08`: coverage diagnostic, acceptance seed study, previous HANDOFF rewrite.
- `5dbeca4`: **ε-uniform exploration** (`agent.epsilon`, default off; fingerprinted).
- `cab3894`, `cd9d116`, `d4911a8`: value-gate study; acceptance at 6000 updates; stale slow-test fix.
- `48a442d`, `07fdcba`, `cd076b3`, `27f28e9`: failed combined-fix run and its diagnostics (`absorbing_value.py`, `greedy_deaths.py`, `start_danger.py`, `head_shift.py`).
- `1d7801b`: **`training.store_eval_episodes`** (default off; fingerprinted).
- `87cfa37`, `aa46d46`, `68d66b4`, `f5740e4`, `d4b476e`: store-eval run, overfitting finding, overfitting A/B, 64-update runs launched.
- `03caa22` … `d8325b7`: Wumpus reviewer page (versions 1–6), UMAP, synced player, export tooling, results in HANDOFF.
- `2a6e326`: previous HANDOFF rewrite. `784b317`: `unfinished_diag.py`, JEPA arm resumed. `80a3628`: JEPA same-recipe result.
- `8ca58d0`, `483f1c3`: **`model.time_aware_value`** (default off; fingerprinted) and its launch.
- `59bd83c`: `value_chain_diag.py` (latent has progress info; value ungrounded).
- `52ac3fc`: **`model.value_nstep`** and the buffer's `greedy` flags (default off; fingerprinted).
- `2eec2bb`: HANDOFF rewrite. `fb57b11`, `45bb861`: offline value-target screens (no remedy has support).
- `1c05c16`: **`env.discount`**, a configurable Wumpus γ (default 0.99; fingerprinted); γ = 0.95 result `eb3c757`.
- `5de6900`, `7f65d90`, `dede693`: horizon A/B and recipe ablations. `3a4da1b`, `f57cbbb`: decoupled planning-head A/B.
- `8b3d621`, `f1e5bbf`: **`model.detached_planning_head`** (default off; fingerprinted) and its failure on Wumpus.
- `ae75473`, `43a4dee`: Wumpus support in `sweep.py`/aggregation; 5-seed headline; slow suite 7/7.

**Pushed 2026-10-11 night (after `43a4dee`)**
- HANDOFF rewrites (`2d8df2e` and earlier).
- `c95e68b`: `counterfactual_diag.py`, `override_diag.py` and the section 5.8 results.
- `25123d0`: **`model.privileged_belief_loss`** (default off; fingerprinted); `ed7b564`: its pre-registered readout; `cad6836` and later: HANDOFF (the first launch was cancelled by the user so the next session can plan it).

**Pushed 2026-10-12 to 10-14**
- `5416237`: privileged targets cached per buffer slot (46 s → 14 s per collection; Tiger bit-identical against `77f7679`). `1f72a1e`, `2b996a6`, `2afd1db`: oracle-latent launch, result (value/planner), screens.
- `b0d3848`, `968d008`, `0e27e80`: `teacher_plugin_diag.py` and its result; `ef315c6`, `bb3f8f0`: simulation sweep readout and `q_head_screen.py`.
- `5bf5099`, `00e3704`, `3924a9d`: `self_loop_diag.py`, the Q-head result and the correction of the action-gap reading, the 3-at-a-time relaunch.
- HANDOFF rewrites.

**Pushed 2026-10-14**
- `5475e6d`: self-loop chain relaunched after a WSL shutdown.
- `b2c3a91`: `phantom_reward_diag.py` (no phantom reward; V̂ > 999; gold refusal) and `fvi_screen.py` (C2, pre-registered).
- `690a737`: `self_loop_diag.py` v2 (`FrozenLoopSearch`; v1 void); `fvi_screen.py` smoke-checked; `c2_chain.sh` launched.
- `e91dc00`, `2061a9b`: the self-loop v2 result (C1 a symptom); `FrozenLoopSearch` vectorised.
- `46f1750`, `a8b4f40`: HANDOFF rewrite; chain stopped on request.

**Pushed 2026-10-15**
- `3790114`: `fvi_screen.py` compute changes (children expanded once, information-first order, saved nets, progress lines).
- `61ba76c`, `5246ce9`, `d223e6d`: GPU memory cap per job, its indexed-device fix, the planner released after each play.
- `b8cc795`, `96bbdaf`, `ac63ebd`, `d40e511`, `da06d34`: FVI results (oracle pass, current fail).
- `5dc41ce`: `model.value_estimator` (Tiger fingerprint identical; 179 tests).
- `f9beec1`, `bd374b7`: the two training results (oracle + double passes, current + double fails).
- `6278dd3`, `314e992`: the user's representation decision; the R0 plan.
- `3624c1c`: HANDOFF rewrite.
- `cbac03f`, `2a741ac`, `95901d1`: R0 pre-registration, result (no branch) and R0' pre-registration, R0' result (branch M).
- `1f7f216`: `noop_drift_diag.py`, `noop_drift_cost.py` (C4).
- `ddedd5d`: R1a `head_swap_screen.py` and the HANDOFF update.
- `af4fbee`, `684ffa6`, `13060d6`: `objective_ab.py` 2e (`--train-buffer`, `--agent-reference`, `--agent-checkpoint`), 2f (`--buffer-schedule growing`), 2g (`--reset-at`, `--catch-up`), their pre-registrations and results; this HANDOFF update.

## 9. Files touched since `9a9b726` (all pushed)

**Added or changed 2026-10-15 night** (diagnostics only)
- `reports/2026-10-02_wumpus_diagnostics/head_shift.py` (items 4, 5; optional 4th argument: another checkpoint's networks), `noop_drift_diag.py`, `noop_drift_cost.py`, `head_swap_screen.py` (new).
- `runs/wumpus/head_shift_best_chain.sh` (untracked).

**Added or changed 2026-10-15**
- `src/ipomdp/models/heads.py` (section 2c: `ValueHead.value`, `ValueEstimator`, `ValuePair`), `models/__init__.py`; `planning/search_model.py` (`values` reads `value_head.value`); `training/trainer.py` (section 3e, `pair_value_targets`, per-episode head rows); `training/run.py` (section 13 with both results; `RunConfig.value_estimator`; head B built last); `experiments/runs.py`; `conf/config.yaml` (`model.value_estimator: max`, back-filled into all 63 saved run configs).
- Tests: `TestValuePair` in `tests/test_training_run.py` (wiring, target definitions, one head per episode, raises, resume); `value_estimator` in the test configs of `test_training_run.py` and `test_wumpus.py`.
- `reports/2026-10-02_wumpus_diagnostics/fvi_screen.py` (compute changes 2b a–e, result header 4).
- `runs/wumpus/fvi_queue.sh`, `double_chain.sh`, `head_shift_double_chain.sh` (untracked).
- `HANDOFF.md`, `CLAUDE.md` (project paragraph), memory files.

**Added or changed 2026-10-14** (diagnostics only; `src/` untouched, so Tiger is unaffected)
- `reports/2026-10-02_wumpus_diagnostics/phantom_reward_diag.py` (new): the heads on no-op GRAB and SHOOT at the checkpoint's own stored greedy games, on the CPU; V̂ > 999; gold refusal.
- `reports/2026-10-02_wumpus_diagnostics/fvi_screen.py` (new): fitted value iteration with max, double and clipped-double targets, played in the planner (C2).
- `reports/2026-10-02_wumpus_diagnostics/self_loop_diag.py`: v2 `FrozenLoopSearch` (header 2c), results in header 3.
- `runs/wumpus/c2_chain.sh` (stopped) and `runs/wumpus/fvi_chain.sh` (ready to launch); untracked, like all of `runs/`.
- `HANDOFF.md`, `CLAUDE.md` (project paragraph), memory files.

**Added or changed 2026-10-12 to 10-14**
- `src/ipomdp/training/privileged.py` (section 2c: the per-slot target cache, `forget()`, `targets()`), `training/episode_buffer.py` (`EpisodeBatch.index`; `add` returns the slots it wrote), `training/run.py` (`_store()`, `forget()` on load); test `test_privileged_cache_matches_fresh_targets_after_overwrites` in `tests/test_wumpus.py`.
- `reports/2026-10-02_wumpus_diagnostics/teacher_plugin_diag.py` (teacher value plugged into the search; section 2c simulation sweep), `q_head_screen.py` (branched heuristic returns; V, Q and A heads in play), `self_loop_diag.py` (the self-loop identity at planning time; running).
- `HANDOFF.md`, `CLAUDE.md` (project paragraph), memory files.

**Added 2026-10-11 night**
- `src/ipomdp/training/privileged.py` (new): `PrivilegedTargets`, `PrivilegedSpec`, `PrivilegedBeliefLoss`.
- `domain/wumpus.py`, section 6: `WumpusBeliefTracker.supervision_targets`.
- `training/trainer.py`, section 2e; `training/run.py`, section 12 (`DomainSetup.privileged`, `RunConfig.privileged_belief_loss`); `experiments/runs.py` (`wumpus_setup` supplies the spec); `conf/config.yaml` (`model.privileged_belief_loss: false`, also back-filled into all 56 saved run configs).
- Tests: `test_wumpus.py` (targets vs tracker; train and resume), `test_training_run.py` (Tiger raises; off builds nothing).
- `reports/2026-10-02_wumpus_diagnostics/counterfactual_diag.py`, `override_diag.py`.

**New source**
- `src/ipomdp/domain/wumpus.py`: rules, `WumpusSimulator`, `WumpusBeliefTracker`, `game_outcomes`, `START_HEADING`.
- `src/ipomdp/agents/wumpus_reference.py`: `WumpusReferenceAgent` (section 5.2).
- `src/ipomdp/interpretability/wumpus_latents.py`: probe dataset, probes, `plot_belief_maps`, `plot_latent_pca`.
- `src/ipomdp/experiments/wumpus.py`: `analyze_wumpus_run`, which also reports the reference agents.
- `conf/env/wumpus.yaml`; `tests/test_wumpus.py`; `tests/conftest.py`.

**Changed source**
- `agents/planning_agent.py`: **ε-uniform exploration** (section 3 of its header).
- `training/run.py`:
  - section 7: ε, decayed with the temperature;
  - section 8: `store_eval_episodes`;
  - also `Domain`, `DomainSetup`, `observation_horizon`, `multistep_predictor`.
- `training/trainer.py`: section 2c, the multi-step term; section 3b, finite-horizon targets; section 3c, n-step greedy-return mixing (`nstep_returns`, `_mix_nstep`).
- `training/run.py`, sections 9 (`time_aware_value`) and 10 (`value_nstep`), each with its negative result; `training/episode_buffer.py`, section 4 (the `greedy` flag; `add(episodes, greedy)`).
- `planning/search_model.py`, section 5 (finite-horizon state [z, k], `LearnedSearchModel.values`); `models/heads.py`, section 2b (`HorizonValueHead`).
- `models/world_model.py` (`start`, `unroll(..., initial_observation)`, section 7, the hoisted `input_net`); `models/heads.py` (`MultiStepObservationPredictor`).
- `training/episode_buffer.py`, `training/rollout.py`, `planning/search_model.py`, `planning/mcts.py` (section 6, pruning), `domain/pomdp.py`, `domain/__init__.py`, `experiments/runs.py`, `interpretability/*`, `telemetry/visualizers.py`, `main.py`, `analyze.py`.
- `conf/config.yaml`: `model.observation_horizon`, `agent.epsilon`, `agent.epsilon_min`, `training.store_eval_episodes`, `model.time_aware_value`, `model.value_nstep`, `model.value_nstep_mix`, `model.detached_planning_head`. `conf/env/wumpus.yaml`: `env.discount`.
- 2026-10-09/10 additions:
  - `domain/wumpus.py`, section 1: a configurable γ (`WumpusWorld(device, discount)`);
  - `experiments/runs.py`: `DOMAIN_BUILDERS` take `(env_cfg, device)`;
  - `training/trainer.py`, section 3d, and `training/run.py`, section 11: the detached planning head (failed; kept off);
  - `experiments/aggregate.py`, section 5: per-domain metric tables (`DOMAIN_METRICS`, `aggregate_reports(reports, domain)`);
  - `sweep.py`, section 4: Wumpus sweeps, and runs joining a sweep by symlink.
- Tests:
  - `test_world_model_acceptance.py`: 6000 updates and the seed table;
  - `test_planner_and_agent.py`: ε test and the stale-signature fix;
  - `test_training_run.py`: ε decay and stored-evaluation tests;
  - also `test_wumpus.py`, `test_training.py`, `test_models.py`, `test_integration_e2e.py`.

**Reports**
- `reports/2026-10-02_wumpus_diagnostics/` (every header has usage and result):

| Script | What it does |
|---|---|
| `value_diag.py`, `reward_diag.py`, `sims_diag.py`, `split_diag.py` | Value and reward heads, simulation count, start-percept split |
| `full_info_vi.py` | Known-layout value (no longer cited) |
| `offpolicy_reference.py` | Off-policy training diagnostic |
| `death_calibration.py`, `danger_probe.py`, `reference_probe.py` | Death calibration and latent probes |
| `supervised_filter.py`, `ab_summary.py` | Supervised bound; GRU/Transformer arms; pool cache; A/B summaries |
| `objective_ab.py`, `overfit_summary.py` | **2e-2g: the agent's own buffer offline (training-limited), its growing schedule (primacy), resets + catch-up.** Offline objective and overfitting A/Bs (weight-decay and data-size arms, held-out NLL, `head_safe_miss`); section 2c horizon A/B, section 2d decoupled planning head (`--planning-head`), `--save`; learning-curve summary |
| `head_coverage.py` | Coverage diagnostic |
| `absorbing_value.py`, `greedy_deaths.py`, `start_danger.py`, `head_shift.py` | V̂ by game status; danger at the greedy planner's FORWARDs; first-step danger; **calibration in-sample vs fresh exploring vs fresh greedy** |
| `unfinished_diag.py` | **Why games end unfinished:** action mix, idling, safe cells left, root Q gaps, the one-step Q(FORWARD) − Q(chosen) split into reward and γ·E[V̂ next] (section 5.6) |
| `grounding_screen.py`, `teacher_robustness.py` | **Offline screens of value targets** on frozen latents (section 5.6): optimistic, own MC, heuristic returns; robustness of the heuristic result to protocol, data and initialisation |
| `value_chain_diag.py` | **Why the value prefers idling:** Bellman residual and drift on idle chains, progress probes, a value refit to the heuristic's returns (section 5.6) |
| `counterfactual_diag.py`, `override_diag.py` | **Exact counterfactual branches at declined safe steps; persistent exact-belief overrides** (section 5.8) |
| `teacher_plugin_diag.py` | **A heuristic-return value plugged into the unchanged search**, at any simulation count (section 5.9) |
| `q_head_screen.py` | **Every action branched from heuristic states in the same layout; V, Q and advantage heads fitted and played** (section 5.9) |
| `self_loop_diag.py` | **The exact self-loop identity at planning time**; v2 freezes loop children (C1: a symptom) |
| `phantom_reward_diag.py` | **The reward/observation heads on no-ops (exact); V̂ > 999 and gold refusal** (CPU, stored greedy games) |
| `fvi_screen.py` | **Fitted value iteration: max vs double vs clipped double, played** (C2) |
| `noop_drift_diag.py`, `noop_drift_cost.py` | **C4: latent drift on exact no-ops; it costs no cell/heading information** |
| `head_swap_screen.py` | **R1a: fresh / ensemble heads swapped into the double-value planner (fail: the planner exploits them)** |
| `tiger_acceptance_seed.py` | Tiger acceptance at another seed and update count |
| `hoist_forward_check.py`, `hoist_training_check.py`, `profile_train_step.py` | Hoist equivalence and runtime profile |

- `reports/2026-10-07_wumpus_reviewer_page/`: `index.html`, `data.json`, `umap.json`, `games_latents.npz`; `scripts/export.py`, `dump_latents.py`, `umap_latents.py`, `page_meta.json`.
