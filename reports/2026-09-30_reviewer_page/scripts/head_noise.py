"""Reward/value head error at best.pt vs latest.pt, per exact-posterior bucket (random-policy beliefs)."""
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from ipomdp.agents import UniformRandomAgent
from ipomdp.domain import BatchedPOMDPEnv
from ipomdp.experiments import load_trained_run
from ipomdp.interpretability import build_probe_dataset
from ipomdp.training import load_checkpoint, play_episodes, stream_seed
d = torch.device("cuda")
ref = torch.load(sys.argv[1], weights_only=False)
V = ref["v"]
for seed in range(5):
    run_dir = Path(f"runs/sweeps/default/seed{seed}")
    t = load_trained_run(run_dir, d)
    ep, _ = play_episodes(BatchedPOMDPEnv(t.pomdp, 512, 100, stream_seed(99, 1), d),
                          UniformRandomAgent(3, 512, stream_seed(99, 2), d))
    latest = load_checkpoint(run_dir / "checkpoints" / "latest.pt")["networks"]
    rows = []
    for name, nets in (("best", None), ("latest", latest)):
        if nets is not None:
            for n, net in t.run.networks.items():
                net.load_state_dict(nets[n])
        ds = build_probe_dataset(t.pomdp, t.run.world_model.belief_filter, ep)
        z, b = ds.latents, ds.posteriors
        p = b[:, 0]
        with torch.no_grad():
            # door that is correct when P(tiger left) = p: open right if p > 0.5
            door = torch.where(p > 0.5, 2, 1)
            r_hat = t.run.codec.mean(t.run.reward_head(z, F.one_hot(door, 3).float())).double()
            r_true = (b * t.pomdp.reward.to(d)[door]).sum(-1)
            v_err = t.run.codec.mean(t.run.value_head(z)).double() - V.value(b.cpu()).to(d)
        conf = torch.minimum(p, 1 - p).round(decimals=4)
        cells = []
        for k in (0.5, 0.15, 0.0302, 0.0055):
            m = conf == k
            cells.append(f"{k}: r {float((r_hat - r_true)[m].mean()):+.2f} v {float(v_err[m].mean()):+.2f}")
        rows.append(f"  {name:<6} " + " | ".join(cells))
    print(f"seed {seed}\n" + "\n".join(rows), flush=True)
