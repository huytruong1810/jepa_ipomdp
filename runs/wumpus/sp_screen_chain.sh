#!/bin/bash
# objective_ab.py 2l: self-predictive latent retrain screen (zp, jepa vs the 2i control), one GPU job at a time
# beside the option (a) sweeps. Retrains wait for device memory < 9.5 GB; plays and feature_localise (~5.4 GB)
# wait for a window: every running main.py younger than 3 h (far from its 5.4 GB analysis phase) and < 7.5 GB used,
# or no main.py and < 3 GB used.
cd /home/andyj1810/projects/jepa_ipomdp
O=runs/wumpus/sp_screen
D=reports/2026-10-02_wumpus_diagnostics
log() { echo "[$(date '+%F %T')] $*" >> $O/chain.log; }
mem() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }
wait_mem() { while [ "$(mem)" -ge "$1" ]; do sleep 60; done; }
window() {
  while true; do
    m=$(mem); ages=$(ps -eo etimes,args | awk '/[m]ain.py seed=/{print $1}')
    if [ -z "$ages" ]; then [ "$m" -lt 3000 ] && return; else
      old=$(echo "$ages" | awk '$1 > 10800' | wc -l); [ "$old" -eq 0 ] && [ "$m" -lt 7500 ] && return; fi
    sleep 60
  done
}
log "commit $(git log -1 --format=%h)"
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=temperature.gpu,memory.used,power.draw --format=csv,noheader) | $(cut -d' ' -f1-3 /proc/loadavg)"; sleep 60; done ) > $O/sysmon.log 2>&1 &
MON=$!
for s in 0 1 2; do
  R=runs/sweeps/wumpus_retrain/seed$s
  ln -sf ../ve_screen/control_seed$s.pt $O/ctl_seed$s.pt
  for arm in zp jepa; do
    [ -f $O/${arm}_seed$s.pt ] && continue
    if [ $arm = zp ]; then A="--representation decoder --set model.self_predictive_weight=1.0"; else A="--representation jepa"; fi
    wait_mem 9500
    log "seed $s train $arm"
    OMP_NUM_THREADS=2 .venv/bin/python $D/objective_ab.py --config $R/.hydra/config.yaml $A --horizon 5 \
      --pool-cache runs/wumpus/supervised_filter/pools.pt --seed $s --train-buffer $R --save $O/${arm}_seed$s.pt \
      > $O/train_${arm}_seed$s.log 2>&1
    log "seed $s train $arm exit $?"
  done
  window; log "seed $s play"
  OMP_NUM_THREADS=2 .venv/bin/python $D/retrain_play.py $R $O/ctl_seed$s.pt $O/zp_seed$s.pt $O/jepa_seed$s.pt \
    > $O/play_seed$s.log 2>&1
  log "seed $s play exit $?"
  for arm in ctl zp jepa; do
    window; log "seed $s feature_localise $arm"
    OMP_NUM_THREADS=2 .venv/bin/python $D/feature_localise.py $R cuda --networks $O/${arm}_seed$s.pt > $O/fl_${arm}_seed$s.log 2>&1
    log "seed $s feature_localise $arm exit $?"
  done
done
kill $MON
log "ALL DONE"
