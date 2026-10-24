#!/bin/bash
# objective_ab.py 2m extension: seeds 1-2 at weight decay 1.0, 25600 updates (checkpoint 12800), then
# feature_localise at both checkpoints and a play of both (paired offline with the 2i ctl's stored returns).
cd /home/andyj1810/projects/jepa_ipomdp
source runs/wumpus/gpu_window.sh
O=runs/wumpus/grok_screen
D=reports/2026-10-02_wumpus_diagnostics
log() { echo "[$(date '+%F %T')] $*" >> $O/chain.log; }
log "seeds 1-2 extension, commit $(git log -1 --format=%h)"
for s in 1 2; do
  R=runs/sweeps/wumpus_retrain/seed$s
  if [ ! -f $O/wd1_seed${s}_u25600.pt ]; then
    while [ "$(mem)" -ge 9500 ]; do sleep 60; done
    log "seed $s train"
    OMP_NUM_THREADS=2 .venv/bin/python $D/objective_ab.py --config $R/.hydra/config.yaml --representation decoder --horizon 5 \
      --pool-cache runs/wumpus/supervised_filter/pools.pt --seed $s --train-buffer $R --weight-decay 1.0 --updates 25600 \
      --score-at 12800 25600 --save $O/wd1_seed$s.pt > $O/train_wd1_seed$s.log 2>&1
    log "seed $s train exit $?"
  fi
  for k in 25600 12800; do
    grep -q "^DONE" $O/fl_seed${s}_u$k.log 2>/dev/null && continue
    log "seed $s feature_localise u$k"
    OMP_NUM_THREADS=2 big $O/fl_seed${s}_u$k.log .venv/bin/python $D/feature_localise.py $R cuda --networks $O/wd1_seed${s}_u$k.pt
    log "seed $s feature_localise u$k exit $?"
  done
  if ! grep -q "^DONE" $O/play_seed$s.log 2>/dev/null; then
    log "seed $s play"
    OMP_NUM_THREADS=2 big $O/play_seed$s.log .venv/bin/python $D/retrain_play.py $R $O/wd1_seed${s}_u25600.pt $O/wd1_seed${s}_u12800.pt
    log "seed $s play exit $?"
  fi
done
log "SEEDS 1-2 DONE"
