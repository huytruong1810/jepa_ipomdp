# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/hoist_training_check.py
# ==============================================================================
# TRAINING FINGERPRINT: PER-STEP VS HOISTED UNROLL
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Diagnostic of 2026-10-04/05, kept so the HANDOFF numbers are reproducible:
#    - Two collections on Tiger and Wumpus with each unroll variant (monkeypatched); compares
#      metrics and parameter sums. Result (2026-10-04): not bit-identical (parameter sums differ by
#      0.004 on Tiger and 0.025 on Wumpus after 16 updates; gradient summation order).
# ==============================================================================

import os, torch
from omegaconf import OmegaConf
from hydra import compose, initialize
from ipomdp.experiments.runs import build_training_run
from ipomdp.models import BeliefFilter
dev = torch.device("cuda")
original = BeliefFilter.unroll
def hoisted(self, actions, observations, initial_observation):
    z = self.start(actions.shape[0], initial_observation); zs = [z]
    x = self.input_net(torch.cat([actions, observations], -1))
    for t in range(actions.shape[1]):
        z = self.cell(x[:, t], z); zs.append(z)
    return torch.stack(zs, 1)
def fingerprint(variant, env):
    BeliefFilter.unroll = variant
    with initialize(version_base=None, config_path=os.path.relpath("/home/andyj1810/projects/jepa_ipomdp/conf", os.path.dirname(__file__))):
        cfg = compose("config", overrides=[f"env={env}", "training.warmup_episodes=128", "training.updates_per_collection=8"])
    _, run = build_training_run(cfg, dev)
    m = [run.collect_and_train() for _ in range(2)]
    return repr(m), [float(p.double().sum()) for p in run.trainer.parameters]
for env in ("tiger", "wumpus"):
    a, b = fingerprint(original, env), fingerprint(hoisted, env)
    print(env, "metrics identical", a[0] == b[0], "| parameters identical", a[1] == b[1],
          "| max param-sum diff", max(abs(x - y) for x, y in zip(a[1], b[1])))
