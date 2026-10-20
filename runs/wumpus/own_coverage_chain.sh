#!/bin/bash
# coverage_games.py arm "own" (section 5), after risk_localise_chain.sh. Beside the held-out sweep, so one planner
# job at a time and a VRAM gate (the held-out main.py's guardrail aborts above 13.5 GB device-wide).
cd /home/andyj1810/projects/jepa_ipomdp
O=runs/wumpus/coverage
D=reports/2026-10-02_wumpus_diagnostics
export OMP_NUM_THREADS=2
log() { echo "[$(date '+%F %T')] $*" >> $O/own_chain.log; }
gate() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 9000 ]; do sleep 60; done; }
until grep -q "RISK DONE" $O/risk_chain.log 2>/dev/null; do sleep 120; done
log "commit $(git log -1 --format=%h)"
TRAIN=
for s in 0 1 2 3 4; do
  R=runs/sweeps/wumpus_retrain/seed$s
  gate
  .venv/bin/python $D/coverage_games.py $R own $O/games_own_seed$s.pt > $O/games_own_seed$s.log 2>&1
  log "seed $s own games exit $?"
  [ -n "$TRAIN" ] && wait $TRAIN
  .venv/bin/python $D/objective_ab.py --config $R/.hydra/config.yaml --representation decoder --horizon 5 \
    --pool-cache runs/wumpus/supervised_filter/pools.pt --seed $s --train-buffer $R \
    --extra-games $O/games_own_seed$s.pt --save $O/own_seed$s.pt > $O/train_own_seed$s.log 2>&1 &
  TRAIN=$!
done
wait $TRAIN
log "trained: $(grep -c 'training took' $O/train_own_seed*.log | tr '\n' ' ')"
for s in 0 1 2 3 4; do
  R=runs/sweeps/wumpus_retrain/seed$s
  gate
  .venv/bin/python $D/feature_localise.py $R cuda --networks $O/own_seed$s.pt > $O/fl_own_seed$s.log 2>&1
  log "seed $s feature_localise own exit $?"
  gate
  .venv/bin/python $D/retrain_play.py $R $O/control_seed$s.pt $O/own_seed$s.pt > $O/play_own_seed$s.log 2>&1
  log "seed $s play exit $?"
  gate
  .venv/bin/python $D/risk_localise.py $R $O/own_seed$s.pt > $O/risk_own_seed$s.log 2>&1
  log "seed $s risk exit $?"
done
.venv/bin/python $D/feature_localise.py compare control own runs/sweeps/wumpus_retrain/seed{0,1,2,3,4} > $O/compare_own.log 2>&1
log "compare exit $?"
log "OWN DONE"
