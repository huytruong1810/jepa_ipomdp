#!/bin/bash
# objective_ab.py 2m readout after the 2026-10-22 reboot (the run died past 40000 updates; checkpoints 6400 / 12800 /
# 25600 survive, 51200 was never written). feature_localise at each checkpoint, then play ctl, u12800, u25600 on the
# 512 analysis games (paired with ctl, the first set).
cd /home/andyj1810/projects/jepa_ipomdp
source runs/wumpus/gpu_window.sh
O=runs/wumpus/grok_screen
D=reports/2026-10-02_wumpus_diagnostics
R=runs/sweeps/wumpus_retrain/seed0
log() { echo "[$(date '+%F %T')] $*" >> $O/chain.log; }
log "readout restart, commit $(git log -1 --format=%h)"
ln -sf ../sp_screen/ctl_seed0.pt $O/ctl_seed0.pt
for k in 25600 12800 6400; do
  grep -q "^DONE" $O/fl_u$k.log 2>/dev/null && continue
  log "feature_localise u$k"
  OMP_NUM_THREADS=2 big $O/fl_u$k.log .venv/bin/python $D/feature_localise.py $R cuda --networks $O/wd1_seed0_u$k.pt
  log "feature_localise u$k exit $?"
done
log "play"
OMP_NUM_THREADS=2 big $O/play_seed0.log .venv/bin/python $D/retrain_play.py $R $O/ctl_seed0.pt $O/wd1_seed0_u12800.pt $O/wd1_seed0_u25600.pt
log "play exit $?"
log "ALL DONE"
