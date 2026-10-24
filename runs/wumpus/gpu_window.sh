# Sourced by the chains: big GPU jobs (plays, feature_localise, ~5.4 GB) take a shared flock and run only when the
# sweep is training (its main.py alive, < 7.5 GB used) or no sweep is running at all (< 3 GB used), never during a
# sweep's analysis (which used > 8 GB on 2026-10-22 and tripped the guardrail).
mem() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }
big_window() {
  while true; do
    m=$(mem); sweeps=$(pgrep -fc "[s]weep.py wumpus_"); mains=$(pgrep -fc "[m]ain.py seed=")
    if [ "$sweeps" -eq 0 ] && [ "$mains" -eq 0 ] && [ "$m" -lt 3000 ]; then return; fi
    if [ "$mains" -ge 1 ] && [ "$m" -lt 7500 ]; then return; fi
    sleep 60
  done
}
# usage: big <log> <cmd...>   (holds the lock for the whole job)
big() { local out=$1; shift; ( flock 9; big_window; "$@" > "$out" 2>&1 ) 9> runs/wumpus/gpu_big.lock; }
