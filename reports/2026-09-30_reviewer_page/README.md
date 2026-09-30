# Reviewer page, 2026-09-30

The source of the reviewer page published at https://claude.ai/artifact/Rboz6xV7xA7Ejrjomy4As5 (private; the owner controls sharing). Reviewers received it well.

- `index.html` is the page. It loads `data.json` from the same folder. To update the published page from a new session, publish `index.html` with `data.json` as a supporting file to that URL, after reading the live version with the Artifact tool, because the owner edits it in place (for example, they renamed the title to "JEPA-IPOMDP").
- `data.json` holds every number the page shows. It comes from `runs/sweeps/polyak` (5 seeds, Polyak-averaged acting weights), `runs/sweeps/default` (the same without averaging), the exact solver and the trained networks.

## Page contents (after the owner's cuts)

Method (4 SVG diagrams: execution loop, belief-tree planning, training cycle, one gradient step with its stop-gradients), Result 1 (policy by net growl count against the exact policy), Result 2 (latent geometry and probe KL), Result 2b (interactive latent traversal of 8 greedy episodes), Result 3 (learned V̂, R̂ and observation head against exact V*, b·R and P(o′|b,a)), and Runtime.

The owner asked for minimal prose: no intro paragraphs under section titles, and no headline tiles, per-seed gap chart, diagnosis, bounds or review-log sections. Only the short figure captions remain.

## Regenerating data.json

Run from the repository root, with `S = reports/2026-09-30_reviewer_page/scripts`. Both extract scripts read and write `$S/page/data.json`, so create `$S/page/` first and copy the result back next to `index.html`.

1. `qstar.pt` holds the certified V* (tolerance 0.01) and the per-action Q* sets: `{"v": AlphaVectorSet, "q": list[AlphaVectorSet]}`. Rebuild it with `solve_infinite_horizon(build_tiger_pomdp(), tolerance=0.01, prune_epsilon=1e-6)` and `action_value_functions(pomdp, v, 1e-6)` if needed (about 72 s).
2. `uv run python $S/extract.py $S` covers the sweeps, curves, exact V*/Q*, learned heads per posterior, geometry, probes and bounds (about 2 min, GPU).
3. `uv run python $S/policy_check.py $S > $S/policy_check_output.txt` gives the greedy actions per net growl count for the 5 polyak seeds (about 15 min).
4. `uv run python $S/extract2.py $S` covers runtime, trajectories, observation prediction and the policy table. It reuses `runtime_cache.json` when present. Delete that file to re-time the solver and the planners (about 5 extra min, and it needs an idle machine).

Diagnosis scripts that are not used by the page: `diagnose_heads.py` (where the planner loses return, per seed; hard-coded to `runs/sweeps/default`), `head_noise.py` (head errors at best.pt vs latest.pt) and `depth.py` (search depth). These are scratch analysis tools, not part of the package.
