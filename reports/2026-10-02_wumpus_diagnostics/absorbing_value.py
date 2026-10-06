# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/absorbing_value.py
# ==============================================================================
# LEARNED VALUE OF LIVE, DEAD AND WON STATES
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-05, combined-fix run eps_decoder_k5_u256_seed0):
#    - The greedy agent died in 40-60% of evaluation games through collection 40 and the mean
#      Bellman value target rose to 800-1550, above the largest achievable return (999). If the
#      value head also rates the absorbing GAME_OVER state highly, dying is worth
#      -1001 + gamma * V_hat(dead) to the planner, and that can beat playing on.
#    - The exact value of any state after the game ended is 0 (absorbing, reward 0).
#    - Usage: uv run python <this> <run dir> <games> <checkpoint name>. Uniform-random games on a
#      fixed seed, latents from the acting filter; V_hat = codec mean of the acting value head.
#      Reports V_hat by status (alive / after death / after a win) and the one-step learned
#      Q(a) at alive states.
#    - Result 2026-10-05: the absorbing state is NOT the problem (V_hat after death 114, after a
#      win 114; exact 0). Live states are overestimated and the overestimate grows with updates:
#      mean V_hat alive 1565 (eps_decoder_k5_u256_seed0, 256 updates/collection, ~collection 40)
#      vs 868 (north50_decoder_seed0, 64/collection, collection 100) and 559 (north50_seed0
#      best.pt); the largest achievable return is 999 and the reference agent earns 573.
# ==============================================================================

import sys
from pathlib import Path

import torch

from ipomdp.agents import UniformRandomAgent
from ipomdp.domain import WumpusSimulator
from ipomdp.experiments.runs import build_training_run
from ipomdp.training import load_checkpoint, play_episodes, stream_seed
from omegaconf import OmegaConf

run_dir, n, name = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
dev = torch.device("cuda")
cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
setup, run = build_training_run(cfg, dev)
state = load_checkpoint(run_dir / "checkpoints" / name)
for key, net in run.networks.items():
    net.load_state_dict(state["networks"][key])
world, model = setup.domain, run.eval_agent.model
episodes, _ = play_episodes(WumpusSimulator(world, n, cfg.env.max_steps, stream_seed(4321, 1), dev),
                            UniformRandomAgent(world.num_actions, n, stream_seed(4321, 2), dev))
with torch.no_grad():
    z = model.initial_states(n, episodes.initial_observations)
    latents = [z]
    for t in range(episodes.actions.shape[1]):
        z = model.update(z, episodes.actions[:, t], episodes.observations[:, t])
        latents.append(z)
    latents = torch.stack(latents, 1)                                  # (B, T + 1, D): z_0 .. z_T
    values = run.codec.mean(run.acting_value_head(latents.flatten(0, 1)).float()).view(n, -1).double()
    rewards = episodes.rewards.double()
    died = torch.cumsum((rewards <= -1000).double(), 1) > 0             # after step t's reward: dead
    won = torch.cumsum((rewards >= 998).double(), 1) > 0
    status_dead = torch.cat([torch.zeros(n, 1, dtype=torch.bool, device=dev), died], 1)
    status_won = torch.cat([torch.zeros(n, 1, dtype=torch.bool, device=dev), won], 1)
    alive = ~(status_dead | status_won)
    print(f"== {run_dir.name} {name}: {n} uniform-random games")
    for label, m in (("alive", alive), ("after death", status_dead), ("after a win", status_won)):
        v = values[m]
        print(f"  V_hat {label:12s} n={int(m.sum()):6d}  mean {v.mean():8.1f}  median {v.median():8.1f}  "
              f"min {v.min():8.1f}  max {v.max():8.1f}" if len(v) else f"  {label}: none")
    live = latents[alive]
    expansion = model.expand(live[:4096])
    q = (expansion.rewards + world.discount * (expansion.observation_probs * expansion.next_values).sum(-1)).double()
    print("  one-step Q at alive states, mean per action: "
          + ", ".join(f"{a} {q[:, i].mean():.1f}" for i, a in enumerate(world.action_names)))
