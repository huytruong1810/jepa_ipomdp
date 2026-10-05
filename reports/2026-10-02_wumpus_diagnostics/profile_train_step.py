# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/profile_train_step.py
# ==============================================================================
# WALL TIME OF ONE TRAINING UPDATE AND ITS VALUE-TARGET SHARE
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Diagnostic of 2026-10-04/05, kept so the HANDOFF numbers are reproducible:
#    - Times WorldModelTrainer.train_step (decoder, k = 5, Wumpus) and the value-target part.
#      Result (2026-10-04, GPU shared by six runs): ~950 ms per update, value targets 12%.
# ==============================================================================

import time, torch
from omegaconf import OmegaConf
from ipomdp.experiments.runs import build_training_run
from ipomdp.training import EpisodeBatch
dev = torch.device("cuda")
cfg = OmegaConf.load("runs/wumpus/north50_seed0/.hydra/config.yaml"); cfg.model.representation = "decoder"; cfg.model.observation_horizon = 5
setup, run = build_training_run(cfg, dev)
c = torch.load("runs/wumpus/supervised_filter/pools.pt", map_location=dev)["train"]
g = torch.Generator(device=dev).manual_seed(0)
tr = run.trainer
orig_vt = tr.value_targets
acc = {"value_targets": 0.0}
def timed_vt(x):
    torch.cuda.synchronize(); t = time.perf_counter(); r = orig_vt(x); torch.cuda.synchronize(); acc["value_targets"] += time.perf_counter() - t; return r
tr.value_targets = timed_vt
for i in range(30):
    rows = torch.randint(0, 8192, (16,), device=dev, generator=g)
    b = EpisodeBatch(c["actions"][rows], c["observations"][rows], c["rewards"][rows], c["initial"][rows])
    if i == 10: torch.cuda.synchronize(); t0 = time.perf_counter(); acc["value_targets"] = 0.0
    tr.train_step(b); run._update_acting()
torch.cuda.synchronize(); total = (time.perf_counter() - t0) / 20
print(f"train_step+acting {total*1000:.0f} ms/update; value_targets {acc['value_targets']/20*1000:.0f} ms ({acc['value_targets']/20/total:.0%})")
