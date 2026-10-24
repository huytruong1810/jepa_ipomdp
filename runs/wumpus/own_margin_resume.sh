#!/bin/bash
# 2026-10-24: the machine rebooted on 2026-10-22 ~20:30 during wumpus_own_margin seed 1 (collection 84). Resume it
# bit-exactly from latest.pt (collection 80; resume is tested bit-exact), mark it trained, then continue the sweep
# with the identical command (analyses seed 1, trains and analyses seed 2, aggregates).
cd /home/andyj1810/projects/jepa_ipomdp
R="env=wumpus model.representation=decoder model.observation_horizon=5 training.updates_per_collection=64 training.total_episodes=6400 training.store_eval_episodes=true agent.epsilon=0.25 agent.epsilon_min=0.1 model.value_estimator=double training.retrain_every=50 training.retrain_updates=6400 agent.trial_prob=0.2 mcts.risk_margin=2"
S=runs/sweeps/wumpus_own_margin/seed1
start=$(date +%s)
.venv/bin/python main.py seed=1 hydra.run.dir=$S $R resume=$S/checkpoints/latest.pt >> runs/wumpus/sweep_own_margin.log 2>&1
status=$?
echo "[resume] seed 1 main.py exit $status" >> runs/wumpus/sweep_own_margin.log
[ $status -eq 0 ] || exit $status
echo "$(( $(date +%s) - start )) s (resumed from latest.pt at collection 80 after the 2026-10-22 reboot)" > $S/TRAINED
.venv/bin/python sweep.py wumpus_own_margin --seeds 0,1,2 $R >> runs/wumpus/sweep_own_margin.log 2>&1
echo "[resume] sweep exit $?" >> runs/wumpus/sweep_own_margin.log
