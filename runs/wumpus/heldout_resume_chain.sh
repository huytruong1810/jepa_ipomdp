#!/bin/bash
# Held-out protocol, continued 2026-10-19: seed 7 was aborted at 14:27 by the device-wide VRAM guardrail when the
# coverage chain started beside it (my scheduling error). It is moved aside and retrained from scratch (training is
# deterministic per seed), after the coverage chain finishes, with the same command as heldout_chain.sh.
cd /home/andyj1810/projects/jepa_ipomdp
export OMP_NUM_THREADS=2
until grep -qE "ALL DONE|exit [1-9]" runs/wumpus/coverage/chain.log; do sleep 120; done
mv runs/sweeps/wumpus_retrain_heldout/seed7 runs/wumpus/heldout_seed7_aborted_20261019
echo "$(date) restart held-out training sweep (confirmatory), seeds 5-9"
uv run sweep.py wumpus_retrain_heldout --seeds 5,6,7,8,9 --analysis-seed 20261019 --analysis-output analysis_heldout env=wumpus model.representation=decoder model.observation_horizon=5 training.updates_per_collection=64 training.total_episodes=6400 training.store_eval_episodes=true agent.epsilon=0.25 agent.epsilon_min=0.1 model.value_estimator=double training.retrain_every=50 training.retrain_updates=6400 > runs/wumpus/heldout_sweep_resume.log 2>&1
echo "$(date) held-out sweep exit $?"
