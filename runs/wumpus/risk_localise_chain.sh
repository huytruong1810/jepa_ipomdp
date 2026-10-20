#!/bin/bash
# risk_localise.py on the coverage arms, after the coverage chain; one 512-game play at a time beside the held-out sweep.
cd /home/andyj1810/projects/jepa_ipomdp
export OMP_NUM_THREADS=2
until grep -qE "ALL DONE|exit [1-9]" runs/wumpus/coverage/chain.log; do sleep 120; done
sleep 600   # let the held-out sweep's seed 7 allocate first
for s in 0 1 2 3 4; do
  .venv/bin/python reports/2026-10-02_wumpus_diagnostics/risk_localise.py runs/sweeps/wumpus_retrain/seed$s \
    runs/wumpus/coverage/control_seed$s.pt runs/wumpus/coverage/cover_seed$s.pt > runs/wumpus/coverage/risk_seed$s.log 2>&1
  echo "[$(date '+%F %T')] seed $s risk_localise exit $?" >> runs/wumpus/coverage/risk_chain.log
done
echo "[$(date '+%F %T')] RISK DONE" >> runs/wumpus/coverage/risk_chain.log
