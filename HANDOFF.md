# HANDOFF: JEPA I-POMDP (as of 2026-10-19. **Phase: algorithm design and improvement (user decision 2026-10-19).** Wumpus recipe (whole-buffer retrain + double value): 200 ± 44 per game on development seeds (`runs/sweeps/wumpus_retrain`). The performance ladder on development games: random -420, R2D2-lite (model-free, matched budget) ~90, **ours ~200**, ours with the EXACT one-step model 408, exact-belief planner (1000 sims) ~476, heuristic 573. The open problem is the learned model: it costs ~+200 per game and the missing information is in the latent (Steps A/B, 2026-10-18). The value-equivalent observation term failed (Step C). Publication groundwork done 2026-10-19: frozen protocol, exact-planner reference, R2D2-lite baseline; the confirmatory held-out runs are still going (section 0, 2026-10-19). Second domain (RockSample) SHELVED until the method works)

**The user's goal for the paper (2026-10-17):** "combine state-of-the-art theories into the algorithm to have the strongest publication in deep learning and decision theory and representation learning". Every next step should be tied to a theory and tested by a pre-registered screen; score-chasing by trial and error is out (two weeks of it did not close the gap).

**The next session's job (algorithm design; everything is pushed; held-out jobs may still be running):**
1. **Read section 1b first, especially "Algorithm-design phase (2026-10-19)"**: it is the taste that should pick the experiments. Then section 6.2 (the plan) and section 7 (failed paths).
2. **Start with Step D of section 6.2** (feature-level localisation of the latent gap at the declined states; CPU, pre-authorised). Its readout picks the method branch (D1 shift / D2 objective / D3 representation). Pre-register the readout in the script header before running.
3. **Background duty: the confirmatory readout.** When `runs/sweeps/wumpus_retrain_heldout` has all 5 seeds (`runs/wumpus/heldout_chain.log`) and `runs/references/wumpus_exact_planner/seed20261019/games.npz` exists (`runs/references/exact_play_20261019.log`; resumable per chunk if interrupted: rerun `uv run reports/2026-10-19_eval_protocol/exact_planner_play.py 20261019`), re-run the held-out sweep command (in `runs/wumpus/heldout_chain.sh`) with `--exact-planner-games runs/references/wumpus_exact_planner/seed20261019/games.npz --baseline-runs runs/baselines/r2d2_heldout`. Read C-a/C-b/C-c ONCE, record them in PROTOCOL.md and section 0. It is the honest anchor for the current method; it does not change the design direction.
4. **Algorithm work is judged on the DEVELOPMENT set only** (training seeds 0-4, analysis seed 12345; paired, hierarchical bootstrap from `games.npz`). A method claim is registered in PROTOCOL.md before it ever touches the held-out set (seeds 5-9 / analysis seed 20261019, or fresh seeds 10-14 for a second claim).
5. Check first: `ps -eo pid,args | grep "[p]ython"`, `uptime` (ipomcp load), `nvidia-smi`. At most 3 GPU planner jobs.

Section 1c is the publication-gap list (self-critique, 2026-10-18): read it before planning experiments. Section 1b holds the working conjectures and the research principles ("taste in research"), including the compute-efficiency rules (updated 2026-10-17). Section 7 lists what failed: do not repeat it.

Read `CLAUDE.md` first: it covers the architecture, commands and conventions. This file covers goals, the user's rules, the current state, every result that matters, the next steps, the dead ends and the files touched. Each module starts with a `DESIGN DECISIONS & THEORETICAL FOUNDATIONS` header with the evidence behind its design; read it before changing that module. Every diagnostic script in `reports/` explains its question, usage, pre-registered readout and result in its header. The user's auto-memory (`~/.claude/projects/-home-andyj1810-projects-jepa-ipomdp/memory/`) repeats the standing rules of section 2.

## 0. Where we are

- **Canonical Tiger: done.** The learned planner matches the exact optimum over 5 seeds (gap −0.03 ± 0.06). A decoder baseline is indistinguishable from JEPA there.
- **Wumpus, newest recipe (2026-10-17): the whole-buffer retrain + double value**, 200 ± 44 over 5 seeds (`runs/sweeps/wumpus_retrain`, section 5.3b). Same return as the 182 recipe within noise (paired +18 ± 53), but fewer deaths (3.1 vs 4.9%), a smaller seed spread (worst seed 157 vs 107) and a much better latent. It is the base for further work; both are cited until a significant gain.
- **Single-hunter 4x4 Wumpus World: the current agent works but is far from the reference.**
  - The agent: decoder representation, 64 updates per collection, ε-uniform exploration, greedy evaluation games stored in the buffer (section 5.3 has the recipe).
  - **182 ± 62 per game over 5 seeds** (185, 166, 215, 107, 236; `runs/sweeps/wumpus_current/aggregate.json`). Wins 29.6 ± 2.7%, deaths 4.9 ± 7.8%, unfinished 65.5 ± 8.9%. Paired gap to the exact-belief heuristic (573): −391 ± 62.
  - Storing the greedy games is what keeps deaths down (−110 ± 97 without it); ε does not matter.
- **The open problem is idling.** About 65% of games end unfinished. The planner stops entering provably safe unvisited cells around step 11–14 and spends the rest of the 50-step window on no-ops (GRAB without gold, SHOOT without the arrow; section 5.9).
- **The prize (section 5.8):** letting the heuristic's navigation rules override the planner gives 482 ± 39 (+300 ± 42 paired), three quarters of it from turning towards an adjacent safe cell and stepping in.
- **Before 2026-10-14** (sections 5.6–5.9): every value-side remedy failed (time-aware value, n-step greedy returns, γ = 0.95, detached planning head, offline value targets, teacher values in play). An oracle-latent arm (`model.privileged_belief_loss`; cell 96–98%, heading 100% decodable) idled as much or more, so the value/planner was judged the bottleneck.
- **2026-10-14:** no-op self-loops are a symptom (C1 closed); the heads are exact on no-ops (no phantom reward); the oracle networks' value exceeds 999, which no game can pay, on 36–58% of live steps, and two oracle seeds refuse the gold for it. That pointed to C2, maximisation bias in the backup.

### 2026-10-20 (early): coverage screen READOUT: MECHANISM PASS (4 of 5), PLAY FAIL (`coverage_games.py` header 4 has the table)

- **Mechanism:** the privileged coverage data repair the turned-latent model on 4 of 5 seeds (joint-head glitter ratio +0.14 to +0.32; decision share M +14 to +32 points, seed 3 +1). So Step D's shift is a data-coverage effect, as the feedback-loop conjecture says.
- **Play:** cover - control **-194 per game, t95 [-345, -43]** (significantly worse; -190 / -3 / -339 / -225 / -212). Unfinished games fall on 5 of 5 seeds (-31 points on average), deaths rise on 5 of 5 (+28 points: 4.5 -> 28%, 3.1 -> 33%, 5.5 -> 42%, 19 -> 43%, 2.7 -> 27%) and wins barely move (+4). The steps freed from idling end in deaths, not gold. Control vs best.pt (the retrain noise floor): +19 / -92 / -14 / -15 / +107.
- Per the registered branches this is "localise again". The failure is deaths, so the localisation is `risk_localise.py` (below) instead of q_localise. Coverage from a privileged, safety-filtered teacher is a failed path as it stands (section 7).

### 2026-10-20 (afternoon): own coverage arm, interim (seed 0 only; NOT the readout)

- Own games done for 5 of 5 seeds (per 2048 games: 5600-7800 trials, ~2800 into unvisited cells, ~1200 into risky cells, ~800 at declined states (cover: 2500-3150); collection deaths 34-46%). All 5 retrains done.
- **Seed 0:** own 286 vs control 256, **+30 +- 35** (n.s.); deaths 4.5 -> 3.5%; risk calibration on risky new cells 0.972 (control 0.995, cover 0.712). Unfiltered coverage removes the selection bias. Mechanism weak: glitter ratio 0.45 -> 0.53, decision share M 26 -> 23%. Seed 1 mechanism: glitter 0.39 -> 0.54, M 42 -> 48%.
- **VRAM incident 2:** the held-out sweep's parent keeps ~5 GB after each analysis, and the device reached 14.6 GB during seed 1's own play (34 guardrail alerts, no abort). I stopped the own chain at 13:18. `runs/wumpus/own_resume_chain.sh` runs the rest (seed 1 play/risk, seeds 2-4) once the held-out sweep has finished, in two streams.

### 2026-10-20 (morning): risk localisation of the coverage deaths: MIXED, acted on as SELECTION (`risk_localise.py` header 4)

- **VALUE refuted on 5 of 5:** the deaths are not knowingly taken risks. 83-100% of the cover arm's fatal FORWARDs had P(death) rated below half the exact danger.
- **Selection is real but moderate:** the cover data lower the calibration on risky new cells by 12-36% (cover/control ratio 0.64-0.88; the 0.67 bar is met on 2 seeds) and the phantom risk at safe new cells by 30-70%, on 5 of 5 seeds.
- **The planner amplifies it** (optimizer's curse): 2-15x more risky FORWARDs are taken, and at those the risk estimate is 0.5-17% of exact.
- Next as queued: the non-privileged own arm (`runs/wumpus/own_coverage_chain.sh`). Noted for after it: whatever the coverage, the planner needs robustness to the risk head's error where it chooses (pessimism on P(GAME_OVER)); this is C6 again.

### 2026-10-19 (night): coverage interim and the queue while the user is away (8 h, "proceed with your taste in research")

- Seeds 0-1 showed the pattern above first (cover vs control -190 +- 59 and -3 +- 62).
- The forced steps are safe (exact danger of every FORWARD in `games_cover_seed1.pt`: +1791 after-turn FORWARDs, +15 risky; every death in both arms is a FORWARD with exact danger > 0). **Hypothesis: sample-selection bias.** The privileged data contain ~2700 survived "turn, step into a new cell" transitions per 2048 games and no risky ones, so a latent that separates safe from risky new cells only partly learns "new cells are safe". The 32-game smoke of the new `risk_localise.py` agrees: phantom risk at provably safe new cells falls from 0.043 to 0.005, and the chosen risky FORWARDs are rated at ~3% of their exact danger.
- **Queued (my calls, pre-registered in the script headers):**
  1. `risk_localise.py` (SELECTION vs VALUE; `runs/wumpus/risk_localise_chain.sh`, logs `runs/wumpus/coverage/risk_seed*.log`).
  2. `coverage_games.py` arm **own** (section 5): non-privileged coverage. Where the planner does not step, with probability 0.2 it plays its own better turn, then FORWARD, with no safety filter, so its misjudged risky trials and their deaths enter the data. Retrain, feature_localise, play and risk are all read against the same control (`runs/wumpus/own_coverage_chain.sh`, log `runs/wumpus/coverage/own_chain.log`, readout `compare_own.log`, `play_own_seed*.log`). Smoke (32 games): 2.4 trials per game, 1.0 into unvisited cells (cover 1.3), 0.47 into risky cells, 0.16 at declined states; collection deaths 37%.
- **Held-out sweep incident:** seed 7 was aborted at 14:27 by the device-wide VRAM guardrail, because I started the coverage chain beside it. Seeds 5 and 6 are intact. `runs/wumpus/heldout_resume_chain.sh` moves the aborted directory to `runs/wumpus/heldout_seed7_aborted_20261019` and reruns the same sweep command once the coverage chain ends. The confirmatory readout is still unread. Later chains gate every planner job on device VRAM < 9 GB.

### 2026-10-19 (evening): coverage screen running (user's choice after Step D)

- **User chose the coverage branch** (D1) over the self-predictive latent (D3) and a combined screen.
- `coverage_games.py` (header: question, method, **pre-registered readout**): a privileged UPPER BOUND on coverage. In the cover arm, 2048 extra games per seed have the heuristic's turn played at every declined distance-2 state, then FORWARD forced; the control arm plays the same seeds with the planner only. Both arms retrain from scratch on the buffer plus their extra games (`objective_ab.py` 2j `--extra-games`), then (i) `feature_localise.py --networks` (decision share M, joint-head glitter ratio, probes at the same declined states; `compare` mode) and (ii) `retrain_play.py` paired cover vs control.
- Readout: MECHANISM = glitter ratio >= +0.15 and decision share >= +10 points on >= 4 of 5 seeds; PLAY = mean paired >= +50 with the t interval over seeds above 0, unfinished down on >= 4 of 5, deaths up at most 2 points.
- Chain `runs/wumpus/coverage_chain.sh` (launched 14:22 at `ae39722`; log `runs/wumpus/coverage/chain.log`, sysmon beside it, readout `runs/wumpus/coverage/compare.log` and `play_seed*.log`). Smokes passed: 32-game cover arm (50 overrides), 20-update retrain with extra games, and `--networks` with best.pt's own networks reproducing the base run exactly.

### 2026-10-19 (afternoon): Step D, feature-level localisation: MIXED, mostly a turn-specific SHIFT (`feature_localise.py`)

- Probes for the factors of the exact P(o' | b, FORWARD) (wall / unvisited ahead, P(gold, pit, live wumpus) ahead, glitter, breeze, stench) trained on each run's buffer, scored on (a) held-out buffer states, (b) the declined roots, (c) the turned latents. 5 seeds of `wumpus_retrain` best.pt; every replay = report.json; header section 4 has the table.
- **Registered readout MIXED** (D3 3 of 5, D1 2 of 5, D2 0 of 5; random-filter control selective on 5 of 5). D3 fires only through "unvisited ahead" at (a), 0.89-0.90 against the 0.90 bar, on the three seeds with the earliest best.pt.
- **On 5 of 5 seeds:** P(gold ahead) is calibrated on buffer states (ratio 0.94-1.09) but under-rated ~2x at the turned latents (0.51-0.59), and the turned latent reads the always-unvisited cell ahead as visited in 19-35% of states. The composed glitter (1 - wall^) gold^ equals the direct probe (~0.5-0.6 of exact), so composition is not the failure (D2 refuted). Step A's "not decodable" is this shift.
- Reading: a support shift specific to "turn towards the unvisited safe cell at a state where the agent declines" (rare in its own data), plus a marginal "visited"-tracking deficit on weaker seeds. Supports the feedback-loop conjecture (section 1b). Per section 3 of the script, the branch is the user's call.
- `declined_states.pt` per seed (`<run>/analysis/`) holds the declined root and turned latents with exact targets for the next screens.

### 2026-10-19 (later): back to algorithm design (user decision)

- I asked whether to plan RockSample (groundwork item 4). The user asked why we were adding a domain and whether the current design and results were good. My answer: no. Tiger is solved but a toy. On Wumpus the agent makes ~35% of the heuristic and the learned model costs ~+200 per game (Step B), and a second domain would only replicate that weakness. **User: "let's go back to algorithm design and improvement."** RockSample is shelved until the method works; the groundwork (protocol, exact planner, R2D2-lite) becomes the test bench.
- Lesson (now in section 1b): do not carry a plan item forward without re-checking that the core result justifies it.

### 2026-10-19: publication groundwork (user's choice; HANDOFF 1c items 1-2)

- **Frozen evaluation protocol** (`3f41017`; `reports/2026-10-19_eval_protocol/PROTOCOL.md`). Held-out set = training seeds 5-9 + analysis seed 20261019 into `analysis_heldout/`. `games.npz` (per-game returns and outcome flags per agent) next to every new report. Aggregates add the IQM over seeds with a bootstrap CI and a hierarchical bootstrap (seeds, then games; paired per game for gaps), `aggregate.py` section 6. The seed-0 development report was reproduced exactly.
- **Running** (`runs/wumpus/heldout_chain.sh`, launched 23:20 2026-10-18, sysmon `heldout_sysmon.log`):
  - (i) the confirmatory sweep `runs/sweeps/wumpus_retrain_heldout` (retrain + double recipe, seeds 5-9; ~7 min per collection under ipomcp load ~25-35, so ~12 h per seed);
  - (ii) the exploratory re-analysis of development seeds 0-4 on the held-out games, `runs/sweeps/wumpus_retrain_devseeds_heldoutgames` (symlinks to the same runs; aggregate goes to `aggregate_analysis_heldout.json`).
  - Do not read the held-out sweep's numbers before C-a/C-b are read as registered.
- **Exact-belief online planner** (`fdd8866`, `33f6f6e`; `planning/wumpus_exact_model.py`): the agent's own BeliefTreeSearch with the true rules over exact layout-mask beliefs (bit-packed registry, steps-left window, exact R/P by bincount).
  - Leaf screen on 128 development games (heuristic 544 on them; `exact_planner_smoke.py`):
    - zero leaves: 145 (FORWARD into walls);
    - uninformed POMCP rollouts: 11 (all leaves look doomed; 200-sim arms stopped);
    - **full-information (hindsight-optimisation) leaf: 335 at 50 sims, 424 ± 41 at 200 sims** (win 56%, deaths 2.3%, unfinished 41%; paired -120 ± 28 vs the heuristic).
    - **1000 sims: 476 ± 41** (win 61%, deaths 2.3%, unfinished 37%; paired -68 ± 25 vs the heuristic; 1000 minus 200 paired +51 ± 14; 193 s per decision of 128 roots).
  - **Frozen (`d08492d`, 04:20, before any held-out play) by the pre-written choice rule: full_info, 1000 sims, no rollouts** (200 sims lies outside one SE of the best). Recorded in the module header 4 and PROTOCOL.md.
  - Reading: a standard exact-model planner beats the learned agent by ~+250 but stays below the hand-written heuristic. The QMDP-style leaf undervalues information-gathering risks. Cite both references.
- **Exact planner on the held-out games** (running since 04:22 2026-10-19): `uv run reports/2026-10-19_eval_protocol/exact_planner_play.py 20261019`, 4 chunks of 128 roots (~2.7 h each, resumable per chunk), log `runs/references/exact_play_20261019.log`, output `runs/references/wumpus_exact_planner/seed20261019/games.npz`. Blind: prints no return. It enters C-a through `sweep.py ... --exact-planner-games runs/references/wumpus_exact_planner/seed20261019/games.npz` (re-run the held-out sweep command with it once all seeds are done; aggregate.py section 6 checks the game pairing against the heuristic's per-game returns).
- **Model-free baseline R2D2-lite** (`36d4b79`, `335a622`; `src/ipomdp/baselines/recurrent_dqn.py`, `reports/2026-10-19_eval_protocol/recurrent_dqn_baseline.py`, claim C-c in PROTOCOL.md): recurrent double DQN, n-step, value rescaling, Ape-X epsilon ladder, matched budget (448k steps, 19.2k updates, same data schedule and layout stream per seed). Development screen (seeds 0-2): R2D2 defaults lr 1e-4, n 5 at 87 ± 20 kept by the registered rule (best arm n 10: 91 ± 9; lr 3e-4 unstable); table in the module header 4 (`6a2f305`). **Held-out seeds 5-9 are DONE and unread** (`runs/baselines/r2d2_heldout/seed{5..9}/analysis_heldout/games.npz`); they enter C-c through `sweep.py ... --baseline-runs runs/baselines/r2d2_heldout`.
- **Exploratory: development seeds 0-4 on the held-out games** (finished ~04:00): learned planner 164 ± 49 (t; hierarchical [129, 197]), heuristic 580, random -420; probes cell/heading 72/77% random, 88/91% agent histories. On the development games the same seeds gave 200 ± 44 against a heuristic of 573, so the development games flattered the learned agent by ~35 (within noise).

### 2026-10-18 evening: Step C, value-equivalent observation term, FAILS its retrain screen

- **Built** (`51abf72`): `model.value_equivalent_weight` w (trainer.py 2f, run.py 15): w * mean(((sum_o P(o | z_t, a_t) sg V-bar(tau(z_t, a_t, o)) - sg V-bar(z_{t+1})) / max|R|)^2), gradients through the observation head into the filter, decoder only; branch values reused from the value-target expansion. Default off; Tiger fingerprint identical (both representations, max and double); fast suite 185 passed; every run's `.hydra/config.yaml` back-filled with 0.0.
- **Screen** (`objective_ab.py` 2i, pre-registered; `--set` overrides; `retrain_play.py` section 5 pairs several saved sets with the first): whole-buffer retrains on `wumpus_retrain` seeds 0-2, control (w = 0) vs w = 25 vs w = 100 (doses from a smoke: at w = 1 the term's filter gradient is 1-2% of the rest), 512 paired games.
- **Result: FAIL.** vs control: w = 25 -57 / +2 / +27, w = 100 -64 / +31 / +12; unfinished unchanged (60-70%). w = 25 leaves the latent unchanged; w = 100 damages it (heading -23 / -17 points on seeds 0-1, head danger error doubled) and gold information stays as it was. Control retrains alone vary -39 to +58 against best.pt, so n = 3 screens resolve only effects above ~+50.
- **Mechanism hypothesis (untested):** the term matches only the next value. Death and gold pay through the reward r_t, and V-bar at game-over latents is co-adapted (exact_model_play.py 3a), so the term is blind to the outcomes that matter and trades danger calibration for value matching. A Q-equivalent variant needs observation-conditioned rewards.

### 2026-10-18: Step A LATENT, Step B PASS (5 of 5): the learned one-step model costs ~+200 per game

**Step A, fit vs latent** (`obs_fit_vs_latent.py`, pre-registered; ~12 min per seed). At the `q_localise_diag.py` states, fresh observation heads on the frozen acting latents of `best.pt`, trained on the run's own buffer and early-stopped: (ii) the agent's hard CE, (iii) exact soft targets at the buffer's actions, (iv) exact soft targets for all 5 actions (decodability bound). Readout **LATENT**: (iv) is far from the exact P(o' | turned latent, FORWARD) on 4 of 5 seeds (TV 0.27-0.40, glitter ratio 0.49-0.62; seed 0 between; margins thin on seeds 1 and 4). Each step away from the joint head helps (glitter ratio 0.22-0.44 -> ~0.5, decision share 8-14% -> 19-34%), but (iv) recovers well under half of the EP - M gap. About half of the glitter under-prediction is the fit/schedule, the rest is not decodable from the latent by this probe.

**Step B, privileged play bound** (`exact_model_play.py`, pre-registered; a hybrid search model with exact layout-enumeration beliefs per tree node, the learned value and filter, the unchanged 50-simulation search; the learned arm reproduces report.json on every seed):

| Seed | learned | EPR (exact P, R) | paired | EP (exact P) | paired | unfinished learned -> EPR |
|---|---|---|---|---|---|---|
| 0-4 | 236.7 / 156.8 / 233.1 / 174.7 / 199.7 | 460.4 / 321.7 / 400.3 / 433.7 / 423.8 | +224 / +165 / +167 / +259 / +224 | 408 / 253 / 355 / 389 / 382 | +172 / +96 / +122 / +214 / +182 | 67 / 68 / 62 / 72 / 67 -> 45 / 57 / 52 / 48 / 49% |

- **PASS on 5 of 5** (>= +100 on 3 of 5 needed): mean 408 vs 200 (paired +208 ± 48 over seeds), deaths 0 on every seed (learned 0.8-4.7%), wins 26-34% -> 43-55%. The observation head carries ~75% of it (EP +157), the reward head the rest.
- **Protocol change after the 32-game smoke (before any full run; header 3a):** with V = 0 at exact game-over children (q_localise's convention) the planner never won: at the gold it bumped into walls, because V^ at glitter latents exceeds ~1,010 > the 999 a GRAB pays, and the learned planner only grabs because its own post-GRAB latent value is not 0. The value is co-adapted with its own terminal latents; the full runs keep the learned value at every child.
- **Reading (C9 confirmed in play):** the agent's own value is good enough to plan with once the one-step model is right; the learned model (mostly the observation head) costs ~+200 per game and all the deaths. Step A says the missing information is largely not in the latent, so the method must shape the representation, not just the head. Half of the games still end unfinished with the exact model: the remaining ~165 to the heuristic is value/search.

### 2026-10-17 evening (2): model vs value localisation, MODEL on 5 of 5 seeds

The user asked for a direction with the strongest publication (combining SOTA theory in deep learning, decision theory and representation learning) and approved this screen. `q_localise_diag.py` (pre-registered; same states as the macro check): depth-2 expectimax at each declined state, swapping exact parts of the one-step model (layout-enumeration filter) for the learned ones, the learned value kept at the filter's latents. Share of states where TURN -> FORWARD beats the chosen action:

| Seed | learned (M) | exact P (EP) | exact P and R (EPR) |
|---|---|---|---|
| 0-4 | 10 / 14 / 11 / 11 / 8% | 43 / 42 / 46 / 62 / 58% | 69 / 48 / 64 / 81 / 68% |

- EPR - M = +34 to +70 points on every seed (MODEL needs >= +30 on 3 of 5). The observation head carries 56-85% of it. At the turned latent its P(o' | FORWARD) is far from exact (TV 0.40-0.51), and it predicts glitter 2.3-4.5x too rarely (0.03-0.06 vs 0.12): **the value of information of entering a new cell is missing from the backup.** The rest is a phantom death risk in the reward head after an imagined turn (mean R(FORWARD) error -39 to -107, median 0). Root rewards are exact, and the learned value at the stepped-into latents is already higher than at the chosen children (+13 to +21 on 4 seeds).
- **Reading (C9):** the idling is a decision-relevant model error, not value bias or search. That calls for decision-aware / value-equivalent model learning (Grimm et al. 2020/2021; MuZero's principle): the observation head must be accurate where its errors flip decisions, not just on average under the agent's own (idling) data. Caveat: offline depth-2 ranking, not play (C3).
- **Next (user's call):** (a) a privileged play bound: plan with the exact P and R and the learned value and filter, 512 paired games per seed (it would show whether fixing the model is worth the method); (b) a CPU split of the head error into "fit" vs "latent" (a fresh head on frozen latents at these states). Then the method.

### 2026-10-17 evening: the macro-action reach check (user's choice of direction), VOID

`reports/2026-10-02_wumpus_diagnostics/macro_reach_diag.py` (pre-registered in its header; 5 retrain seeds, best.pt, the 512 analysis games, every replay = report.json; ~9 min per seed). At the planner's declined "near-turn" states (the heuristic turns towards a provably safe/gold target 2 actions away, the planner does something else; 1,400-2,500 states per seed), it reads the root tree:
- The turn child is expanded at 99.7-100% of those states, so the tree already contains TURN -> FORWARD (expectimax: Q(TURN) >= Q_mac, asserted).
- Q_mac > Q(chosen) at **0.000 on 5 of 5 seeds** (median gap −29 to −45); distance-3 states the same. Leaf-only one-step values prefer the macro only 9-15% of the time.
- **Readout VOID** (threshold < 10% on >= 4 of 5): the macro was not built and not played. The search finds the safe step and the model's value ranks it below no-ops (SHOOT/GRAB 26-58% of the chosen actions) or other moves. C8 sharpened: with a good latent and the double value, the idling is the value's ranking of the stepped latent, not search coverage. Cost: ~45 min of plays, no `src/` change.

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

- **Nothing is running** (2026-10-17). Everything is committed and pushed (through the commit that wrote this file). Fast suite 182 passed (2026-10-16). The slow suite last passed 7/7 at `f1e5bbf`; not run since `privileged_belief_loss` and `value_estimator` (both default off, fingerprinted). Run it when the machine is free (~45 min).

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
9. **In-agent result: PASS on all three** (run.py section 14). 236.7 / 156.8 / 233.1 vs 163.8 / 81.7 / 168.0 (+72.9 / +75.1 / +65.1), deaths 4.1 / 9.0 / 5.9% -> 0.8 / 4.7 / 4.1%, unfinished 62-68%. Against the current recipe (max value) on the same seeds: +51 / -9 / +18. Agent-history probes cell 86-92%, heading 91-95%. Caveat: between the retrains (collections 55-95) evaluations were no better than the double runs' and deaths higher; the final retrain repairs it. best.pt at collections 100 / 55 / 80.
10. **5-seed sweep (2026-10-17, `runs/sweeps/wumpus_retrain`, seeds 3-4 added by `runs/wumpus/retrain_seeds34_chain.sh`):** 200.2 ± 43.7 (std 35) vs `wumpus_current` 182.0 ± 61.9. Paired per seed: +51.2 / -9.2 / +17.9 / +67.7 / -36.5 = **+18.2 ± 53.0, not significant**. Deaths 3.1 ± 1.9% vs 4.9 ± 7.8%; unfinished 67.2% vs 65.5%; wins 29.7%. Probes: cell / heading on random histories 68.6 / 75.3% vs 57.0 / 58.3%, on agent histories 87.4 / 91.5% vs 82.3 / 85.8%. Reading: the retrain repairs the double value (C6) and gives a much better, more consistent latent, but in return it only matches the max-value recipe on average, because ~67% of games still end unfinished. The representation direction has done what it can; the idling is a planner/value problem on top of a good-enough latent.

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

**Where Wumpus stands against the stage-3 goal (2026-10-18):** the latent can now be shown against the exact belief convincingly (agent-history probes cell 87%, heading 92%; random histories 69 / 75%). Playing well is not reached: 200 per game against the heuristic's 573, and ~67% of games end unfinished. The open problem is localised (C9): the learned observation (and, less, reward) head is wrong exactly where it decides whether to enter a new cell; the value and the search are not the blocker.

**Publication framing (2026-10-17, my proposal to the user; superseded in detail by section 1c's self-critique and preferred framing, 2026-10-18).** Three theory-backed findings plus a method:
- Representation learning: a JEPA belief filter that matches the exact optimum on Tiger, with span-Hölder error bounds against the exact belief (contribution 3).
- Two biases that compound in learned belief-space planning: maximisation bias (van Hasselt 2010; double estimation only helps once the model is accurate, C2/C6) and primacy bias (Nikishin et al. 2022; the growing-buffer schedule caps the latent, whole-buffer retraining lifts it, C7).
- Decision theory: exact-component localisation (exact P, R swapped in at the planner's own states) shows the residual failure is a lost value of information in the model, not value bias or search (C9).
- The method to build (Step C): decision-aware / value-equivalent model learning for belief-space planning.

## 1b. Working conjectures and research principles ("taste in research")

The user trusts the agent's research judgement (2026-10-10: "run the experiments you think will not waste meaningless compute and post-analysis"; 2026-10-12: "decide on the best direction forward without asking me to pick an option"). These are the beliefs that currently steer the work, strongest evidence first. Each is a conjecture with a stated test, not a result.

### Algorithm-design phase (2026-10-19): what I believe and how I would pick experiments

**Where the problem is (my reading, strongest evidence first).**
- **The bottleneck is the learned one-step model, and inside it the latent.** The exact one-step model with the agent's OWN value and filter plays 408 vs 200 (+208 ± 48, 5 of 5 seeds, deaths 0); exact P alone gives +157 (Step B). No head on the frozen latent recovers the exact P(o' | FORWARD) at the declined states (Step A, LATENT). That +208 is the **known headroom**, the budget any model-side method competes for.
- **The value and search are second-order for now.** With the exact model, half the games are still unfinished and ~165 remain to the heuristic. The exact planner itself (true rules and belief, 1000 sims) only reaches ~476, so the last stretch is hard for any search. Do not work on value or search until the model gap is mostly closed.
- **The specific failure is a rare, high-value outcome: glitter in a newly entered cell is predicted 2.3-4.5x too rarely.** Likelihood (CE) training underweights rare outcomes, and the agent's own data rarely shows "turn, then step into a new cell" from the states where it declines. That is a feedback loop: idle -> no such data -> the model says new cells are dull -> no value of information -> idle. It is the model-learning version of objective mismatch (Lambert et al. 2020) and of C3's compounding error. **Untested conjecture: this loop, not capacity, is why the latent lacks the information.**
- **Partly known:** "safe unvisited cell directly ahead" and "cells visited" decode well from the latent on agent states (section 5.6: 93-97%, off by 0.15-0.37), yet the exact P(glitter) at the TURNED (imagined) latents does not. So either the turned latents are off the filter's training support, or the composition into P(o') is what fails. Step D (section 6.2) separates these. I would not build a method before it reads out.

**How I would choose experiments in this phase.**
- **Localise, then remedy. Every candidate method states which part of the +208 it targets and is screened first where the error lives.** Cheapest is the declined-state metrics of `q_localise_diag.py` (TV to the exact P(o' | FORWARD), the glitter ratio, and the share of states where TURN -> FORWARD wins): a method that does not move these at the declined states will not move play. Next is an offline whole-buffer retrain (`objective_ab.py` pattern, saved networks) with 512 paired games (`retrain_play.py`). Only then train in-agent on 3 and 5 seeds.
- **Anchor every result on the ladder**: random -420, R2D2-lite ~90, ours ~200, exact one-step model + own value 408, exact planner ~476, heuristic 573. Report where a method lands and what fraction of the +208 it recovers.
- **Theory first, one mechanism per arm.** Each arm names the theory it tests (self-predictive sufficiency, value equivalence, decision-aware weighting, coverage) and the readout branch that would refute it. No stacking of untested pieces.
- **Effects below ~+50 need 5 seeds or a paired control per seed** (control retrains alone vary -39 to +58). Use the hierarchical bootstrap on `games.npz` for every development comparison; it is already wired into `sweep.py`.
- **Keep the development / held-out wall.** All design reads use the development set. A method claim goes into PROTOCOL.md before its held-out run, and each held-out readout is read once.
- **Do not widen scope before the core result works** (2026-10-19). I carried the "second domain" item forward from an approved plan without asking whether the main result justified it; the user caught it. A plan item is re-justified against the current results before it starts.
- **The privileged and exact pieces stay diagnostics.** The exact planner and the exact one-step model bound what is possible; the method must learn everything from the agent's own percepts and rewards.
- **Architecture is a legitimate lever but has to earn it** (user standing rule: architecture must never be the bottleneck). A supervised GRU reaches ~100% on cell and heading, and an untuned causal Transformer was no better (section 7). Run an architecture A/B only if Step D says the information is missing from the latent everywhere, not just at turned latents.

**Method candidates, in the order I would try them (after Step D picks the branch).**
1. **Self-predictive latent on the retrain recipe (Ni et al. 2024; the ZP condition plus reward/value prediction makes the latent sufficient for Q\*).** This is JEPA's latent self-prediction, so a positive result also rescues contribution 1. JEPA on the retrain recipe has never been run, and the old JEPA loss on Wumpus came before the schedule fix (C7). Cheap: one offline retrain arm per seed.
2. **Decision-aware weighting of the observation likelihood (VaGraM, Voelcker et al. 2022):** weight each transition's CE by the value spread across its observation branches, so the rare glitter branch with a +999 continuation costs what it is worth. It fits the rare-outcome mechanism directly and, unlike Step C, uses the value at the CHILD latents where gold value lives.
3. **Q-equivalent observation term:** Step C's untested successor. It matches r + γV over branches instead of V alone, so outcomes paid through the reward (death, gold) are seen.
4. **Coverage of decision-relevant counterfactuals:** collect data where the planner declines (e.g. optimism or information-gain bonuses at planning time, or ε on the declined action pair), so the model sees what entering new cells reveals. This targets the feedback loop rather than the objective.

**C7 (new 2026-10-16; confirmed). The agent's latent was limited by its training SCHEDULE, not its objective, data or architecture.** Offline with the same code: uniform training on the agent's own final buffer gives cell / heading 85 / 87%; the agent's growing-buffer schedule replayed offline gives 77 / 72% (the agent: 76 / 68%). Primacy bias (Nikishin et al. 2022): the first few dozen games are replayed hundreds of times and later data does not repair the latent. Budget-matched resets do not fix it (2g); retraining from scratch, long and uniformly on the whole buffer, does (2h and in-agent; run.py section 14). **Open sub-question:** the growing schedule after a mid-run retrain re-degrades the networks (collections 55-95); a schedule that stays near-uniform is untested.

**C9 (new 2026-10-17 evening; offline, 5 of 5 seeds). The idling is a decision-relevant MODEL error: the observation head under-predicts what a new cell can reveal (glitter 2.3-4.5x too rare after turn + step), so the backup misses the value of information.** Exact P and R with the learned value flip the declined turn-then-step decision at 48-81% of states (learned: 8-14%); `q_localise_diag.py`. **Confirmed in play 2026-10-18** (`exact_model_play.py`: +208 per game, deaths 0, 5 of 5 seeds) and **localised to the latent** (`obs_fit_vs_latent.py`: no head on the frozen latent recovers it). A next-value-equivalent term failed (6.2 Step C, 2026-10-18 evening); a reward-aware (Q-equivalent) variant is untested.

**C8 (new 2026-10-17; the lead for the next direction). With a good-enough latent and an unbiased value, the idling is a planner/value problem.** The retrain recipe has an 87-92% latent, a double value without impossible values and few deaths, yet ~67% of games end unfinished, the same as every recipe before it. The oracle + double arm (latent 97-100%) also stays at 68-70% unfinished. Neither more latent accuracy nor removing the maximisation bias touches the count of unfinished games; the navigation override (+300, section 5.8) shows the missing behaviour is "turn towards an adjacent safe cell and step in". Test next: planner-side changes that make that two-step behaviour cheap to find (macro actions), judged in play on the retrain recipe. **Tested 2026-10-17 (`macro_reach_diag.py`): VOID.** The search already finds the turn-then-step path and its value is 29-45 below the chosen action, so the idling is in the value's ranking of the stepped latent, not search coverage. Any remedy must change that ranking on the agent's own states (C3), and the value-target remedies of section 7 are what failed before.

**C6 (2026-10-15; resolved 2026-10-16). The value bias and the model's accuracy are coupled.** Confirmed from the other side: with the retrained latent the double value goes from -22 / -84 / -47 (vs max) to +73 / +75 / +65 (vs double alone), and deaths fall. The double value turns the oracle networks into the best agent so far (223) and makes the current agent worse (138 vs 189 on the same seeds; deaths up). On frozen latents the converged pair values exploit the current model's errors. Implication: fix the representation and heads first, then use the double value. **Sharpened 2026-10-15 night (R0', R1a):** what the planner exploits are the head's under-reported MODERATE risks, not confident "safe" errors, and swapping in fresh or ensembled heads on the same latent makes it worse (the planner finds their errors). Only the oracle latent's head resists. Next: section 6.2 options.

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
- **When swapping exact parts into a learned planner, keep the learned value everywhere, terminal latents included** (2026-10-18). The learned value is co-adapted with its own model; zeroing it at exact game-over children made the planner bump into walls at the gold (`exact_model_play.py` 3a). A 32-game smoke with per-game trajectories found it in minutes.
- **Size a new loss term by its gradient share before screening** (2026-10-18): at w = 1 the value-equivalent term was 1-2% of the filter gradient. Measure on trained networks (the zero-initialised value makes such terms vanish at init).
- **Know the noise floor of the screen** (2026-10-18): two control retrains of the same run differ from best.pt by -39 to +58 per seed, so a 3-seed "+50 on 2 of 3" bar only resolves large effects. Use 5+ seeds, or a paired control per seed, for effects below ~+50.
- **Tie every step to a theory and to the paper** (user, 2026-10-17). Before proposing an experiment, say which theory it tests and which branch of the readout leads to which method.
- **Localise with exact components before building a remedy** (2026-10-17). Where an exact model exists, swap exact parts (P, R, terminal flags) into the learned Q at the planner's own declined states, keeping the rest learned. One CPU-scale screen (`q_localise_diag.py`, ~50 min for 5 seeds) located what two weeks of value remedies had not.
- **Before adding a search edge or option, check whether the tree already dominates it** (2026-10-17). In expectimax, an expanded child's value is max over its actions, so a macro scored by the same model is never above its first action (`macro_reach_diag.py`: void, ~45 min instead of a 5-seed play screen plus training).
- **Distrust a recommendation you would not bet on** (2026-10-17): I proposed re-screening the old value remedies, then said I did not believe in it when asked. State confidence honestly up front.
- **Ask "data, objective, or training?" before changing an objective** (2026-10-16). Train the same objective offline on (a) a reference pool, (b) the agent's own buffer uniformly, (c) the agent's exact data schedule, with the same code and update count. It separated the three causes in ~3 GPU-hours, where months of objective changes had not; the answer (schedule) needed no new loss.
- **When a fix is budget-matched, check what the FINAL network was trained on** (2g lesson). Matching total updates left the post-reset network with half the whole-buffer training of the reference arm, and more resets made it worse. Compare the final network's training, not the run's total.
- **Screen the whole fix on frozen data before building it into the agent** (2h): train the would-be agent's networks offline on a real run's buffer, swap them into that run's planner, and play paired games. 2 GPU-hours told us the in-agent version would pass (it did: +73 / +75 / +65 vs the screen's +67 / -1 / +88).
- **Watch the in-training evaluations between interventions,** not only the final readout: the retrain runs degraded between retrains (deaths up to 25%) although the final readout passed. A pass at the end can hide a fragile middle.
- **Report the honest comparison as well as the pre-registered one:** the retrain recipe passes its pre-registered test (vs double alone) but is not significantly better than the best previous recipe (vs max value). Say both.
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
- **A 512-game, 50-simulation play needs > 4.8 GB of GPU memory** (2026-10-16): the 0.3 cap that suits 128-game screens OOMs. Run 512-game plays one at a time, uncapped (`retrain_play.py`).
- **Save the networks of every offline training** (`objective_ab.py --save`): the 2h plays reused the saved retrained networks after the plays OOMed, with no retraining.
- **Run a per-minute system log beside long jobs** (`nvidia-smi` temperature/memory/power, RAM, swap, load). The 2026-10-16 machine crash left no evidence because there was none.


## 1c. Publication-gap list (self-critique, 2026-10-18; the user asked for it to steer towards a publishable paper)

Verdict: not yet a top-venue paper. The method results are weak and from one real domain. The strongest asset is the **diagnostic methodology**: exact-component localisation inside a learned belief-space planner, pre-registered and paired, at the planner's own decision points. The chain of findings it produced (maximisation bias -> primacy bias -> missing value of information in the model) is the novel content. Gaps, worst first:

1. **The headline results do not support the stated contributions.**
   - Contribution (1), the JEPA filter: it ties the decoder on Tiger and loses on Wumpus; every Wumpus result uses the decoder. Either make JEPA work on the real domain or drop it from the title.
   - Tiger (two states) is a sanity check, not evidence.
   - On Wumpus the agent makes 35% of a NON-optimal heuristic (200 vs 573), and the best recipe is not significantly better than the previous one (+18 ± 53). The big numbers (408, 482, 223) are privileged arms, not agents.
   - With the exact one-step model, half the games are still unfinished, and the learned value is co-adapted with the model: it needs its own non-zero values at game-over latents to grab the gold (`exact_model_play.py` 3a). This weakens "the value is fine, only the model is wrong".
2. **Evaluation hygiene: the most serious methodological gap.**
   - Development and reporting share seeds and games: every screen, threshold and design choice was read on the same training seeds (0-4) and the same 512 analysis games. That is test-set reuse; pre-registration protects each readout, not the selection across ~30 readouts.
     - Fix: freeze the protocol, then report headline numbers on untouched training seeds (e.g. 5-9) and fresh game seeds.
   - Multiple comparisons: many screens, several arms each, "2 of 3" / "3 of 5" pass rules at n = 3.
     - Fix: 10 seeds for headline claims; IQM with stratified-bootstrap intervals (Agarwal et al. 2021, rliable); confirmatory vs exploratory results separated.
   - Protocol changes after smoke runs (Step B's terminal values): documented, but keep them rare and principled.
3. **No baselines.** Expected by reviewers:
   - model-free recurrent RL (R2D2/DRQN or PPO-LSTM) at matched environment steps;
   - DreamerV3;
   - MuZero / Stochastic MuZero-style value-equivalent latent models (Step C is their principle applied to observation branching);
   - self-predictive representations (Ni et al. 2024);
   - an exact-simulator online planner (POMCP/DESPOT with the true Wumpus model) as a real upper reference instead of the hand-written heuristic.
4. **Scope and generality.**
   - Two tiny hand-built domains. Need 3+ POMDPs with exact filters and graded difficulty: RockSample(7,8)/(11,11) (exact belief, SARSOP references; also makes the alpha-vector bounds meaningful beyond Tiger), larger Wumpus boards or battleship/minesweeper-style domains, and a POPGym or memory task for contact with deep RL.
   - Planning and the value targets enumerate all |O| = 49 observations; large or continuous observation spaces need a sampled-branching (progressive widening) variant shown to match.
   - Cost: 6.5 h per seed on a 4x4 grid; report wall-clock and scaling.
5. **Interpretability (contribution 3) is thin.**
   - The span-Hölder bound is a one-line Lipschitz lemma, instantiated only on Tiger; its worst-case form is likely vacuous on harder domains.
   - Probes need control tasks / selectivity (Hewitt & Liang 2019) and capacity vs accuracy, beyond the untrained-GRU baseline.
   - The stronger contribution: tie probe error to decision loss on the planner's own states. Steps A and B already gesture at it ("not decodable, and worth +208 in play"); formalise which belief features are decision-relevant and whether the latent has them.
6. **The theory is borrowed; sharpen it.**
   - Maximisation bias (van Hasselt), primacy bias (Nikishin), value equivalence (Grimm) and VaGraM are known; "observed in belief-space planning" is a finding, not a theory contribution.
   - Candidate result: in a belief-tree backup, observation-model error enters weighted by the spread of child values (errors in value of information are amplified where branches differ), and the value-equivalent objective bounds that term. That ties Step C to a proposition.
   - C6 (bias correction helps only with an accurate model) rests on 3 seeds and one domain. Make it a controlled experiment: interpolate exact and learned P with weight alpha, and plot the double-vs-max gain against alpha.
7. **Missing ablations.**
   - The recipe stacks decoder, k = 5, epsilon, stored greedy games, Polyak acting copies, double value, retrain and VE. A paper needs one leave-one-out table on held-out seeds.
   - Search design choices (expectimax vs mean backup, observation pruning, simulation count) need the same.

**Framings.**
- **Preferred:** a diagnostic paper with the method it motivates. Working title: "Where do learned belief-space planners fail? Exact-component localisation in POMDPs with tractable filters."
  - The protocol, applied to two agent families (ours plus a MuZero- or Dreamer-style one) on 3-4 exact-filter domains.
  - The value-equivalent observation objective as the fix it points to, validated on held-out seeds against baselines.
  - It plays to the real strength, and the negative results become content.
- **Alternative:** a method paper on decision-aware latent models for belief-tree search. Needs a large significant VE gain on several domains, beating DreamerV3, MuZero and model-free baselines; a higher bar with no evidence for it yet.

**Frozen evaluation protocol (registered 2026-10-19, before any held-out read): `reports/2026-10-19_eval_protocol/PROTOCOL.md`.** Development = training seeds 0-4 and analysis seed 12345 (everything so far; exploratory only). Held-out = training seeds 5-9 (then 10-14 if needed) and analysis seed 20261019, written to `analysis_heldout/`; confirmatory claims C-a (performance against random, heuristic and the exact-belief planner) and C-b (probes), each read once. The cited interval is the hierarchical bootstrap over seeds and games (`aggregate.py` section 6); the mean ± t and IQM with a bootstrap CI are reported beside it. New method claims must be registered there before their held-out runs.

**Status 2026-10-19:** gaps 2 (protocol) and part of 3 (exact-planner reference, R2D2-lite) are done; the held-out readout is pending. Gap 1 (weak method results) is now the priority (algorithm phase, sections 1b and 6.2). Gap 4 (a second domain) waits for a working method.

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
- **Everything is pushed** (2026-10-18 evening, through the commit that wrote this file). The working tree is clean. `runs/` is untracked: keep it.
- **C-a/C-b/C-c readout, once the held-out sweep has all 5 seeds and the exact play is done:** re-run the held-out sweep command (`runs/wumpus/heldout_chain.sh` has it) with `--exact-planner-games runs/references/wumpus_exact_planner/seed20261019/games.npz --baseline-runs runs/baselines/r2d2_heldout`; it skips the trained seeds and writes `aggregate_analysis_heldout.json`. Read it once and record it in PROTOCOL.md and here.
- **Running (2026-10-19):** the held-out sweep `runs/wumpus/heldout_chain.sh` (seed 5 at collection ~50 of 100 at 04:20, ~8 h per seed at load ~15-20) and the exact-planner held-out play (section 0, 2026-10-19). Earlier: **nothing was running** at 2026-10-18 22:30. The day's chains all finished: `runs/wumpus/obs_fit_chain.sh` (Step A; logs `obs_fit_seed{0..4}.log`), `exact_play_chain.sh` (Step B; logs `exact_play_{learned,EPR,EP}_seed{0..4}.log`, `exact_play_sysmon.log`; smoke logs `exact_play_smoke_seed0.log` = EPRt, `exact_play_smoke_EPRv_seed0.log` = EPR), `ve_screen_chain.sh` (Step C screen; `runs/wumpus/ve_screen/`). Earlier: `macro_reach_chain.sh`, `q_localise_chain.sh` (2026-10-17), `retrain_seeds34_chain.sh`. The machine crashed once on 2026-10-16 ~00:30 (cause unknown, not this project's memory).
- Chains of 2026-10-14/15, all finished or stopped on purpose, with logs beside them: `c2_chain.sh` (self-loop v2), `fvi_chain.sh` then `fvi_queue.sh` (FVI screen; `fvi_queue.log` records the relaunches and the stop of current seed 2), `double_chain.sh` (training with `value_estimator=double`, both recipes, then `analyze.py`).
- **Health checks for any chain:** `cat <chain>.log`; `tail -3 <run>.<diag>.log` (progress lines should advance); `ps -eo pid,etime,args | grep "^ *[0-9]* *[0-9:-]* [^ ]*python[^ ]* \(main.py\|reports/\)"`; `nvidia-smi`; `uptime` (ipomcp load). **Never `pkill -f`** a pattern in your own command line; kill by PID (chain shell, `xargs`, then the python jobs).

**Tests**
- The fast suite passes: 197 tests (2026-10-19), about 1.5 min on an idle machine (6 min under load).
- **The slow suite passed 7/7 at `6a2f305` on 2026-10-19** (50.5 min; log `runs/slow_suite_6a2f305.log`). Previous pass: `f1e5bbf`, 2026-10-11.
- **Checkpoints from before `52ac3fc` cannot be resumed:** their buffer state has no `greedy` flags (no back-compat, by rule). They still load for analysis (`load_trained_run` reads networks only).
- **The Tiger acceptance test now trains 6000 updates.** At 3000 the value gate (mean |V − V*| < 8) failed at 3–5 of 9 seeds. The two worst seeds drop to 5.3 and 1.1 at 6000. The seed table is in the test header; the logs are in `runs/tiger_acceptance_seeds/`.
- The input-network hoist (`6af8f8f`) stays: its paired effect on the value error is +4.2 ± 6.2, not significant.

**Runs** (`runs/` is untracked)

| Path | What |
|---|---|
| `runs/sweeps/polyak` | Tiger JEPA, 5 seeds: **the Tiger numbers to cite** |
| `runs/sweeps/wumpus_retrain/seed{0..4}` | **The newest Wumpus recipe** (retrain + double, 200 ± 44). `analysis/` holds `report.json`, `macro_reach.npz`, `q_localise.npz`, `obs_fit_vs_latent.json` + `obs_fit_*_pfwd.npy` (Step A) and `exact_play_{learned,EPR,EP}.npz` (Step B: per-game returns, rewards, actions) |
| `runs/wumpus/ve_screen/` | Step C retrain screen: `{control,VE25,VE100}_seed{0,1,2}.pt` (saved networks) + `.returns.npy` (per-game returns), `train_*.log` (probe RESULT lines), `play_seed*.log`, `chain.log`, `sysmon.log` |
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
| `runs/sweeps/wumpus_retrain/` | **The newest Wumpus recipe, 5 seeds** (symlinks to `runs/wumpus/u64_store_eval_eps_decoder_k5_double_retrain50_seed{0..4}` + `aggregate.json`, 200 ± 44); log `runs/sweeps/wumpus_retrain.sweep.log`. Re-aggregate with `uv run sweep.py wumpus_retrain --seeds 0,1,2,3,4 <the overrides in its sweep.json>` |
| `runs/wumpus/u64_store_eval_eps_decoder_k5_double_retrain50_seed{0..4}` | Retrain recipe runs (seeds 0-2 at `d0364c6` via `retrain_chain.sh`, seeds 3-4 via `retrain_seeds34_chain.sh`), analysed, `TRAINED` markers |
| `runs/wumpus/data_vs_objective/` | `objective_ab.py` 2e-2g logs: `pool_`, `buffer_`, `growing_`, `reset1_`, `reset3_seed*.log`; chains `data_vs_objective_chain.sh`, `schedule_chain.sh`, `reset_chain.sh` (+ `.log`), `reset_chain_sysmon.log` (system log of the relaunch) |
| `runs/wumpus/agent_reference_probes.txt` | Agent references scored like the `objective_ab.py` arms |
| `runs/wumpus/retrain_play/` | 2h: `retrained_seed*.pt` (networks retrained on each double run's whole buffer), `train_seed*.log`, `play_seed*.log`; chains `retrain_play_chain.sh` (training; its parallel plays OOMed) and `retrain_play_plays.sh` (plays one at a time) |

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

### 5.3b The whole-buffer retrain recipe (2026-10-16/17; `training/run.py` section 14)

**Recipe.** The 5.3 recipe plus `model.value_estimator=double training.retrain_every=50 training.retrain_updates=6400` (retrains after collections 50 and 100; ~6.5 h per seed, 3 in parallel). Trained at `d0364c6`.

| Seed (512 games, best.pt) | 5.3 recipe | Double alone | **Retrain + double** | best.pt from | Deaths | Unfinished |
|---|---|---|---|---|---|---|
| 0 | 185.5 | 163.8 | **236.7** | collection 100 | 0.8% | 67.2% |
| 1 | 166.0 | 81.7 | **156.8** | 55 | 4.7% | 67.8% |
| 2 | 215.2 | 168.0 | **233.1** | 80 | 4.1% | 62.3% |
| 3 | 107.0 | - | **174.7** | | 2.9% | 71.5% |
| 4 | 236.2 | - | **199.7** | | 3.1% | 67.2% |
| **Mean** | 182 ± 62 | 138 (3 seeds) | **200 ± 44** (std 35) | | 3.1 ± 1.9% | 67.2 ± 4.1% |

- Paired against the 5.3 recipe: +51 / -9 / +18 / +68 / -37 = **+18 ± 53, not significant**. Against double alone (pre-registered): +73 / +75 / +65.
- Latent (MLP probes): random histories cell 68.6%, heading 75.3% (5.3 recipe: 57.0 / 58.3%); agent histories 87.4 / 91.5% (82.3 / 85.8%).
- Between the retrains (collections 55-95) the 128-game evaluations were no better than the double runs' and deaths reached 25% (seed 0); the growing schedule after a retrain brings the planner's model exploitation back. The final retrain repairs it.

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

**Direction (2026-10-19, user: "let's go back to algorithm design and improvement"): close the learned-model gap (the +208 of Step B) with a theory-tied method, judged on the development set and confirmed on held-out seeds. Second domain shelved.** Earlier direction (2026-10-17): the unfinished games through decision-aware model learning. Experiments inside it need no permission (pre-register, cheapest first); the method itself (Step C) and any change to canonical Tiger need the user.

### 6.1 Done 2026-10-15 to 10-17
- **FVI screen** (`fvi_screen.py`, header 4): oracle networks pass, current agent fails.
- **`model.value_estimator`** (`5dc41ce`; run.py section 13): oracle + double passes (223), current + double fails (138).
- **R0, R0', C4, R1a** (2026-10-15 night): the head under-reports moderate risks; no-op drift costs no information; fresh/ensemble head swaps fail.
- **Data vs objective vs schedule** (`objective_ab.py` 2e-2f): training-limited, the growing schedule (C7).
- **Reset + catch-up** (2g): FAIL. **Retrain-and-play screen** (`retrain_play.py`, 2h): PASS.
- **`training.retrain_every` / `retrain_updates`** (`d0364c6`; run.py section 14): in-agent PASS on 3 seeds, 5-seed sweep 200 ± 44 (section 5.3b).
- **Macro reach check** (`macro_reach_diag.py`): VOID on 5 of 5. **Model-vs-value localisation** (`q_localise_diag.py`): MODEL on 5 of 5 (C9).
- **Step A** (`obs_fit_vs_latent.py`): LATENT. **Step B** (`exact_model_play.py`): PASS on 5 of 5 (408 vs 200; EP 357) (2026-10-18).
- **Step C, value-equivalent observation term** (`model.value_equivalent_weight`; `objective_ab.py` 2i retrain screen): FAIL (2026-10-18 evening).

### 6.2 The plan (2026-10-19, algorithm phase)

**Step D: feature-level localisation of the latent gap. DONE 2026-10-19: MIXED, mostly a turn-specific SHIFT (section 0; `feature_localise.py`).** Question: at the declined states, which factor of the exact P(o' | b, FORWARD) is missing from the turned latent, and is it missing only there?
- States and latents: exactly those of `q_localise_diag.py` (re-select as there; the count must equal `analysis/q_localise.npz`), on `runs/sweeps/wumpus_retrain/seed0-4` `best.pt`. Use the turned latent (after TURN with the branch observation), as in Step A.
- Targets from the exact filter (`WumpusBeliefTracker`): the factors that make up P(o' | b, FORWARD): cell ahead is a wall; cell ahead is new; P(gold in the cell ahead); P(pit), P(live wumpus) in the cell ahead; and P(glitter) itself.
- Fit MLP probes on ordinary buffer states (the run's replay buffer; `obs_fit_vs_latent.py` has the data plumbing), then evaluate them (a) on held-out buffer states, (b) on the declined turned latents.
- Pre-register (example thresholds; fix them in the header before running): a factor decodable on buffer states (accuracy or calibration close to the exact) but far at the turned latents -> **D1 SHIFT** (imagined latents off the filter's support; the remedy is coverage, candidate 4, or training the model on its own branches). Decodable at both, while P(glitter) is wrong -> **D2 OBJECTIVE/COMPOSITION** (candidates 2-3). Not decodable on buffer states either -> **D3 REPRESENTATION** (candidate 1, then an architecture A/B).

**Coverage screen (user's D1 choice). DONE 2026-10-20: MECHANISM PASS 4 of 5, PLAY FAIL (-194, deaths +28 points; section 0).** Privileged, safety-filtered coverage repairs the turned-latent model but teaches "new cells are safe". Running: `risk_localise.py` (SELECTION vs VALUE reading of the deaths), then `coverage_games.py` arm **own** (section 5 of its header: the agent's own unfiltered turn-then-step trials, epsilon 0.2, same control). If own passes PLAY: a default-off collection option and 5 in-agent seeds. If its MECHANISM fails for lack of dose at the declined states: the representation branch (candidate 1; Step D's visited deficit is what a targeted, non-privileged mechanism would need).

**Then:** the method candidate the branch points to (section 1b list), screened at the declined states first, then by an offline retrain with paired play, then in-agent on 3 and 5 development seeds. Register the claim in PROTOCOL.md and confirm on held-out seeds 10-14 (5-9 are spent on the current recipe's claims).

### 6.2a The 2026-10-18 plan (Steps A-C; done)

**Step A: fit vs latent. DONE 2026-10-18: LATENT (section 0).** Question: is the observation head's error at the turned latent a fitting/objective problem or missing information in the latent?
- Data: the `q_localise_diag.py` states (re-select them exactly as there; the count must equal `analysis/q_localise.npz`), plus the run's replay buffer (`latest.pt`; slot layout in section 1b) for training.
- Arms, on the frozen acting latents of `best.pt`: (i) the joint head (as in q_localise); (ii) a fresh observation head trained on the buffer, early-stopped on held-out NLL (`head_shift.py` / `head_swap_screen.py` have the fitting code); (iii) an MLP probe from the turned latent to the EXACT P(o' | FORWARD) (soft targets from the exact filter on buffer states; `belief_probe.py` patterns).
- Readout (pre-register in the header): TV to exact and P(glitter) ratio at the declined states. (iii) close to exact while (i)/(ii) are not -> FIT/OBJECTIVE (the CE objective under the agent's own data does not put accuracy where decisions need it: supports Step C directly). (iii) also far -> LATENT (the turned latent lacks "is the cell ahead new and could it hold the gold"; then the representation objective must change, e.g. a self-predictive/value-equivalent latent term).
- Reminder (R1a, section 7): a fresh head that looks better offline can play worse; Step A only localises, it does not pick a head.

**Step B: privileged play bound. DONE 2026-10-18: PASS on 5 of 5 (section 0); EPR ~22-38 min per 512-game seed.** Does the exact one-step model with the learned value and filter actually stop the idling in play?
- New diagnostic search model in `reports/` (no `src/` change): the state is [z, node id]; the model keeps, per node, the exact belief (consistent-layout mask (L,) = 16,380 bools, and the hunter's own state), stored lazily as (parent id, a, o) and materialised on expansion (`mask_child = mask_parent & (obs == o)`). `expand` batches `WumpusWorld.step_rules` over (N, L) hypotheses per action (chunk N to ~64 roots: 64 x 5 x 16,380 entries), returns exact R(b, a) = mean reward, exact P(o | b, a) by scatter-add, children latents from the learned filter with the true branch observation, learned V at the children, and V = 0 at exact terminal children. `q_localise_diag.py`'s `exact_outcomes` is the reference implementation to test against (assert equality on its states).
- Arms: learned planner (must reproduce report.json), EPR (exact P and R), EP (exact P, learned R). Same seeds, 512 paired games, `best.pt` of `runs/sweeps/wumpus_retrain/seed0-4`, one play at a time, uncapped (> 4.8 GB), progress lines, per-minute system log.
- Pre-registered pass: EPR >= +100 paired on 3 of 5 seeds with unfinished down and deaths up by at most 2 points. EP vs EPR splits observation head vs reward head. FAIL -> the offline ranking does not survive play (C3 again); stop and report before any method.
- Smoke first: 32 games, check that the learned arm reproduces the planner's actions bit for bit when the exact swaps are disabled, and that EPR changes the action mix (FORWARD into new cells up).

**Step C: the method.** The user chose the recommendation (2026-10-18): a latent-side value-equivalent observation term. **Tried and FAILED its retrain screen (section 0, 2026-10-18 evening).** Remaining candidates, strongest theory first:
1. **Value-equivalent observation loss** (Grimm et al. 2020, 2021 "proper value equivalence"; MuZero's principle): train P_hat(o' | z, a) so that sum_o P_hat(o) V_bar(tau(z, a, o)) matches V_bar(z_{t+1}) of the real next latent (target value head, gradient into the observation head only, beside the CE term). Default off (`model.value_equivalent_observation`), Tiger fingerprint-identical.
2. **Decision-aware weighting of the observation CE** (VaGraM, Voelcker et al. 2022): weight each transition's CE by the value spread across its observation branches, so errors that change decisions cost more.
3. **Step A said LATENT, so this is now the lead (recommended), possibly with 1 or 2 on top:** a latent-side term (self-predictive representations, Ni et al. 2024: the ZP condition plus reward/value prediction is what makes the latent sufficient for Q*), on the JEPA arm as well (contribution 1).
- Judge in training on the retrain + double recipe, 3 seeds first, paired with `runs/sweeps/wumpus_retrain` on the same 512 analysis games; pre-register >= +50 on 2 of 3 with unfinished down; then 5 seeds with `sweep.py`.

**Also open (lower priority):** a retrain schedule that does not re-degrade between retrains (C7 sub-question; collections 55-95 had deaths up to 25%); JEPA on the retrain recipe (contribution 1, one seed pair); the remaining ~23% of the gap is risk decisions (not diagnosed).

### 6.3 Other candidates (weaker or more expensive; reasons stated)
- **A learned policy prior (AlphaZero-style).** A prior distilled from the search's own visits inherits the idling, and argmax-Q acting ignores the prior. Only worth it if acting follows visits or the prior comes from outside the search.
- **Advantage learning (action-gap increasing operator, Bellemare et al. 2016).** The counterfactual test found the one-step estimates uncorrelated with the true advantage (section 5.8), so amplifying the gap amplifies noise.
- **Re-screening the old value remedies (time-aware, n-step) on the retrain recipe:** low prior; C9 puts the error in the model, not the value. I proposed it on 2026-10-17 and withdrew it.
- **A privileged DAgger diagnostic** (teacher labels on the agent's own states) would bound what a value learned on the right states can do (C3). Diagnostic only; lower priority than (B).
- **The remaining ~23% of the gap is risk decisions** (seed 3 keeps 13.5% deaths even with the navigation override). Not diagnosed.
- If an arm is significant at 3 seeds, extend it to 5 with `sweep.py`.
- Run the slow suite once the code settles (`uv run pytest -m slow`, ~45 min).

### 6.4 Ask the user before starting
- **Reviewer page refresh**, only if the user wants a meeting. The 2026-10-07 page shows 3 seeds (189); the 5-seed numbers are in `runs/sweeps/wumpus_current` and the retrain recipe's in `runs/sweeps/wumpus_retrain` (worth a row and the latent-probe comparison). Merge with `scripts/export.py add`, read the live page with the Artifact tool first, then republish to the same URL.
- **Direction calls:** the hybrid JEPA + observation condition; a second domain (RockSample(7,8); shelved 2026-10-19 until the method works); multi-agent Tiger (verify its tables against Gmytrasiewicz & Doshi first); I-POMDP levels via the S × M_j reduction. Cheap Tiger questions: an observation-head detach ablation in JEPA mode; Polyak momentum 0.98.

## 7. Failed paths and pitfalls: do not repeat

**Representation and model**
- **Coverage from a privileged, safety-filtered teacher** (`coverage_games.py` cover arm, 2026-10-20): forcing the exact-belief heuristic's provably safe turn-then-step at every declined state, then retraining on the extra games, repairs the turned-latent glitter prediction but costs -194 per game (deaths +28 points on 5 of 5 seeds). The data hold only safe entries into new cells, so the death predictions at new cells drop. Any coverage data must include the risky outcomes (sample-selection bias).
- **VICReg** (any placement) made the latent less belief-like. **Pure JEPA without reward grounding** learns no belief.
- **A stochastic (DreamerV3 discrete) latent transition for imagination** imagined biased beliefs, and predicting online targets collapsed the representation.
- **Free bits** on the balanced KL: the prior never trained.
- **Symlog-space two-hot decoding** is Jensen-biased; **unbounded bins** wreck the real-space means.
- **A decoder of the *current* o_t:** the GRU can copy its last input. Use next-observation prediction.
- **Wumpus, JEPA as is (64 updates, no ε):** the latent does not encode the danger ahead.
- **Wumpus, multi-step k = 5 on JEPA:** no effect.
- **Wumpus, a causal Transformer filter (untuned):** no better than the GRU, and worse on cell.

- **A next-value-equivalent observation term into the filter** (`model.value_equivalent_weight` = 25 / 100; 2026-10-18): no gain over a control retrain on 3 seeds, w = 100 damages heading and danger. It is blind to reward-carried outcomes (death, gold). Do not retry without observation-conditioned rewards.

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
- **Wumpus, turn-then-forward macro actions in the expectimax search** (`macro_reach_diag.py`, 2026-10-17): void. The tree already expands the turn child and values the path below the chosen action at every declined state (0% on 5 seeds). A macro scored by the same model cannot change the argmax; check the tree before adding edges.
- **Mean (MuZero-style) backups with a uniform prior** degrade as search grows. Use expectimax.
- **Wumpus, the no-op self-loop identity at planning time** (`self_loop_diag.py` v2, frozen loops): +22 / +14 / +38. The idling moves to turn cycles. Do not build it into the method alone (C1 closed).
- **Changing a child's value by wrapping `LearnedSearchModel.expand`** only changes *unexpanded leaves*. `BeliefTreeSearch` expands every child it selects, and an expanded node's value is max_a Q over its own children (`mcts.py`, `DecisionNode.value`). Self-loop v1 was void for this reason (9 GPU-hours). To pin a node's value, stop it from being expanded (`FrozenLoopSearch` in `self_loop_diag.py`).
- **Per-edge Python/GPU work inside a search subclass** (one `.tolist()` per edge): ~10× slower plays. Batch per `_expand` call.
- **More simulations on Wumpus** (200 vs 50): no gain.

**Analysis and methodology**
- **Proposing planner-side or value-side remedies before localising the error:** two weeks of value remedies (section 5.6-5.9) and the macro (VOID) missed that the observation head was wrong where it mattered. Localise first with exact-component swaps where an exact model exists (`q_localise_diag.py`).
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

**Training schedule and representation (2026-10-16)**
- **Budget-matched resets with a catch-up** (`objective_ab.py` 2g; one reset at 3200 + 3200, or three at 1600 / 3200 / 4800 + 1600): cell / heading 80 / 76% and 77 / 68% against 85 / 87% for uniform whole-buffer training. The final network is under-trained on the full buffer; more resets are worse. Retrain from scratch, long and uniformly, instead (run.py section 14).
- **New objectives to fix the latent** before checking the schedule: the objective was never the limit (C7). Check data vs objective vs schedule first.
- **A retrain seed derived from `cfg.seed`** broke checkpoint-driven resume (the resume test caught it); every random stream must come from the checkpoint (`retrain_seed` is saved).

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
- **A 0.3 GPU memory cap on 512-game plays** OOMs (2026-10-16, 2h first plays); the cap suits 128-game screens only.
- **A `tail -f` monitor over a glob** does not see log files created later (the 3-reset arms); poll with `grep` over the glob instead.
- **Three chained arms without a system log:** the 2026-10-16 machine crash at ~00:30 left nothing to diagnose. Run a per-minute system log (section 1b).

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

**Pushed 2026-10-16/17**
- `fd11f1a`: HANDOFF, data vs objective -> training-limited; reset screen launched (it died with the machine; relaunched unchanged at 13:10).
- `7511997`, `aa5ee65`: 2g reset + catch-up FAIL; `retrain_play.py` (2h) pre-registered; crash note.
- `d0364c6`: `training.retrain_every` / `retrain_updates` (run.py section 14; Tiger fingerprint identical, 182 tests); 2h PASS; in-agent test pre-registered.
- `81dce02`, `ac180a2`: in-agent retrain + double PASS on 3 seeds; seeds 3-4 launched.
- `57fb36f`: 5-seed sweep 200 ± 44 (paired +18 ± 53 vs 182, not significant); CLAUDE.md project paragraph.
- `5c54df2`: HANDOFF rewrite for the next session.
- `5506e2d`, `3f6cb73`: `macro_reach_diag.py` pre-registered, VOID (5 of 5 seeds).
- `569c1d7`, `db73bc7`: `q_localise_diag.py` pre-registered, MODEL (5 of 5 seeds), C9.
- 2026-10-18: HANDOFF rewrite for the next session (plan Steps A-C, publication framing, new principles).

## 9. Files touched since `9a9b726` (all pushed)

**Added or changed 2026-10-19**
- `src/ipomdp/domain/wumpus.py` (`game_outcome_flags`), `experiments/wumpus.py` (header 3; `WumpusAnalysisSettings.output`; `games.npz`), `experiments/aggregate.py` (section 6: IQM, `hierarchical_bootstrap`, `aggregate_games(games, exact_planner=None)`, `WUMPUS_GAME_METRICS`, `EXACT_PLANNER_GAME_METRICS`), `analyze.py` / `sweep.py` (`--analysis-output`; sweep `--exact-planner-games`, header 5).
- `src/ipomdp/planning/wumpus_exact_model.py` (new; `ExactWumpusSearchModel`, `full_information_values`; frozen configuration in header 4).
- Tests: `TestRobustStatistics` (6) in `tests/test_experiments.py`, `TestExactSearchModel` (6) in `tests/test_wumpus.py`. Fast suite 197 passed.
- `reports/2026-10-19_eval_protocol/` (new): `PROTOCOL.md`, `exact_planner_smoke.py`, `exact_planner_play.py`, `recurrent_dqn_baseline.py`.
- `src/ipomdp/baselines/` (new; R2D2-lite), `tests/test_baselines.py` (5); `aggregate_games(..., baseline=)`, `BASELINE_GAME_METRICS`, sweep `--baseline-runs`.
- `runs/sweeps/wumpus_{current,retrain}/sweep.json` back-filled with `"output": "analysis"`; `runs/wumpus/heldout_chain.sh`, `exact_smoke_*` chains (untracked).

**Added or changed 2026-10-18**
- `src/ipomdp/training/trainer.py` (section 2f; `value_equivalent_weight` / `value_equivalent_scale`; `_value_targets_and_children`, `_pair_value_targets_and_children` return the expansion's child values; the public `value_targets` / `pair_value_targets` are unchanged), `training/run.py` (section 15; `RunConfig.value_equivalent_weight`; passes max|R| = value_bound * (1 - gamma)), `experiments/runs.py`, `conf/config.yaml` (`model.value_equivalent_weight: 0.0`). Tiger fingerprint identical at `51abf72`.
- Tests: `TestValueEquivalentTerm` in `tests/test_training.py`; the field in the `_config`s of `test_training_run.py` and `test_wumpus.py`.
- Every `runs/**/.hydra/config.yaml` (75) back-filled with `value_equivalent_weight: 0.0` (untracked).
- `reports/2026-10-02_wumpus_diagnostics/obs_fit_vs_latent.py` (new, Step A), `exact_model_play.py` (new, Step B; `HybridSearchModel`, an exact layout-enumeration one-step model per tree node), `objective_ab.py` (2i and `--set` dotlist overrides), `retrain_play.py` (section 5: several saved sets, paired with the first; per-game returns saved).
- `runs/wumpus/obs_fit_chain.sh`, `exact_play_chain.sh`, `ve_screen_chain.sh` (untracked).
- `HANDOFF.md` (sections 0, 1c new, 1b C9, 3, 6, 7, 9), `CLAUDE.md` (project paragraph), memory files.

**Added or changed 2026-10-17 evening**
- `reports/2026-10-02_wumpus_diagnostics/q_localise_diag.py` (new; result in header 4); `runs/wumpus/q_localise_chain.sh` (untracked); per-seed `analysis/q_localise.npz`.
- `reports/2026-10-02_wumpus_diagnostics/macro_reach_diag.py` (new; result in header 4); `runs/wumpus/macro_reach_chain.sh` (untracked); per-seed `analysis/macro_reach.npz`.

**Added or changed 2026-10-16/17**
- `src/ipomdp/training/run.py`: section 14 (whole-buffer retrain: `RunConfig.retrain_every` / `retrain_updates`, `TrainingRun._retrain()`, `retrain_seed` checkpointed only when enabled, the `retrained` metric only when enabled; pre-registration, 3-seed and 5-seed results); `training/seeding.py` (`RunStream.RETRAIN = 9`); `experiments/runs.py` (`build_run_config`); `conf/config.yaml` (`training.retrain_every: 0`, `training.retrain_updates: 0`, back-filled into all 70 saved run configs).
- Tests: `TestWholeBufferRetrain` in `tests/test_training_run.py` (partial settings raise, off logs nothing; networks and Adam reset then catch-up; bit-exact resume across a retrain); `retrain_every` / `retrain_updates` in the test configs of `test_training_run.py` and `test_wumpus.py`.
- `reports/2026-10-02_wumpus_diagnostics/objective_ab.py` (2e `--train-buffer`, `--agent-reference`, `--agent-checkpoint`; 2f `--buffer-schedule growing`; 2g `--reset-at`, `--catch-up`; results), `retrain_play.py` (new; 2h, pre-registration and result).
- `runs/wumpus/reset_chain.sh`, `retrain_play_chain.sh`, `retrain_play_plays.sh`, `retrain_chain.sh`, `retrain_seeds34_chain.sh` (untracked).
- `HANDOFF.md`, `CLAUDE.md` (project paragraph), memory files.

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
| `obs_fit_vs_latent.py` | **Step A: fresh observation heads on frozen latents vs the exact P(o' \| FORWARD) at the declined states (LATENT)** |
| `exact_model_play.py` | **Step B: hybrid search model, exact one-step P/R per tree node + learned value/filter, in play (PASS 5/5, +208)** |
| `tiger_acceptance_seed.py` | Tiger acceptance at another seed and update count |
| `hoist_forward_check.py`, `hoist_training_check.py`, `profile_train_step.py` | Hoist equivalence and runtime profile |

- `reports/2026-10-07_wumpus_reviewer_page/`: `index.html`, `data.json`, `umap.json`, `games_latents.npz`; `scripts/export.py`, `dump_latents.py`, `umap_latents.py`, `page_meta.json`.
