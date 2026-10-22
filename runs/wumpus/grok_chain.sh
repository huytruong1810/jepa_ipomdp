#!/bin/bash
# objective_ab.py 2m: grokking screen on seed 0 (weight decay 1, 51200 updates, checkpoints), then feature_localise
# at each checkpoint (window: device memory < 7.5 GB, no main.py older than 3 h).
cd /home/andyj1810/projects/jepa_ipomdp
O=runs/wumpus/grok_screen
D=reports/2026-10-02_wumpus_diagnostics
R=runs/sweeps/wumpus_retrain/seed0
log() { echo "[$(date '+%F %T')] $*" >> $O/chain.log; }
mem() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }
window() {
  while true; do
    m=$(mem); old=$(ps -eo etimes,args | awk '/[m]ain.py seed=/ && $1 > 10800' | wc -l)
    [ "$old" -eq 0 ] && [ "$m" -lt 7500 ] && return; sleep 60
  done
}
log "commit $(git log -1 --format=%h)"
OMP_NUM_THREADS=2 .venv/bin/python $D/objective_ab.py --config $R/.hydra/config.yaml --representation decoder --horizon 5 \
  --pool-cache runs/wumpus/supervised_filter/pools.pt --seed 0 --train-buffer $R --weight-decay 1.0 --updates 51200 \
  --score-at 6400 12800 25600 51200 --save $O/wd1_seed0.pt > $O/train_wd1_seed0.log 2>&1
log "train exit $?"
for k in 6400 12800 25600 51200; do
  window; log "feature_localise u$k"
  OMP_NUM_THREADS=2 .venv/bin/python $D/feature_localise.py $R cuda --networks $O/wd1_seed0_u$k.pt > $O/fl_u$k.log 2>&1
  log "feature_localise u$k exit $?"
done
log "ALL DONE"
