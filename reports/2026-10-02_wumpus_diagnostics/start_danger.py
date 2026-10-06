# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/start_danger.py
# ==============================================================================
# PREDICTED VS EMPIRICAL DEATH OF FORWARD FROM THE START CELL, BY START PERCEPT
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-05): does the head use o_0 for the first step's danger?
#    - Usage: uv run python <this> <run dir>:<checkpoint> [...]. 4096 fresh layouts; FORWARD from
#      the start; empirical death rate vs P(GAME_OVER | z_0(o_0), FORWARD), and V_hat(z_0).
#    - Result 2026-10-05: yes, in every run (eps_decoder_k5_u256_seed0: breeze 0.53 -> 0.71,
#      stench 0.50 -> 0.73, both 1.00 -> 1.00, quiet 0 -> 0.001). The fatal misses are mid-game.
#      V_hat(z_0) is 1490-1670 in that run, above the largest achievable return (999).
# ==============================================================================

import sys
from pathlib import Path
import torch, torch.nn.functional as F
from ipomdp.domain import WumpusSimulator
from ipomdp.domain.wumpus import GAME_OVER
from ipomdp.experiments.runs import build_training_run
from ipomdp.training import load_checkpoint, stream_seed
from omegaconf import OmegaConf
dev = torch.device("cuda")
for spec in sys.argv[1:]:
    run_dir, name = spec.split(":"); run_dir = Path(run_dir)
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml"); setup, run = build_training_run(cfg, dev)
    for k, net in run.networks.items(): net.load_state_dict(load_checkpoint(run_dir / "checkpoints" / name)["networks"][k])
    world, model = setup.domain, run.eval_agent.model
    sim = WumpusSimulator(world, 4096, 50, stream_seed(99, 1), dev); o0 = sim.reset()
    out = sim.step(torch.zeros(4096, dtype=torch.int64, device=dev))   # FORWARD from the start
    died = (out.reward <= -1000).double()
    with torch.no_grad():
        z = model.initial_states(4096, o0)
        p = F.softmax(model.observation_head(z, F.one_hot(torch.zeros(4096, dtype=torch.int64, device=dev), 5).float()).float(), -1)[:, GAME_OVER]
        v = run.codec.mean(run.acting_value_head(z).float())
    print(f"{run_dir.name}:{name}")
    for o in torch.unique(o0).tolist():
        m = o0 == o
        print(f"   o0={o:2d} n={int(m.sum()):5d}  empirical P(death|FORWARD) {died[m].mean():.3f}  predicted {p[m].mean():.3f}  V_hat(z0) {v[m].mean():7.1f}")
