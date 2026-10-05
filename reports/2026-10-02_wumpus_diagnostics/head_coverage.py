# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/head_coverage.py
# ==============================================================================
# IS THE OBSERVATION HEAD'S DEATH PREDICTION LIMITED BY DATA COVERAGE?
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (user decision, 2026-10-05):
#    - With longer training the latent nearly reaches the supervised bound, but the observation
#      head's P(GAME_OVER | z, FORWARD) stays badly calibrated (objective_ab_long). Hypothesis:
#      the head learns it only from SAMPLED outcomes, and dangerous FORWARD steps are rare in
#      the data (the reference agent avoids them, random hunters die early).
#
# 2. Protocol (the latent is held fixed, so only the head's data differs):
#    - Latent: the agent's BeliefFilter architecture trained with privileged supervision
#      (supervised_filter.py targets, --filter-steps updates), then frozen. A near-exact latent
#      removes representation quality as a confound.
#    - Head: a fresh ObservationHead of the agent's architecture (config latent / hidden /
#      blocks), fitted with cross-entropy on detached latents, the same steps and minibatch
#      size in both arms:
#        (a) NORMAL     every (z_t, a_t, o_{t+1}) transition of the training pool;
#        (b) AUGMENTED  (a) plus one forced FORWARD transition from every alive state of the
#                       pool's reference games, its outcome o' from the true rules
#                       (WumpusWorld.step_rules on the true state). No privileged LABEL is
#                       used, only extra sampled experience, which exploration could collect.
#    - Score: on held-out games, P_hat(GAME_OVER | z, FORWARD) against the exact danger ahead:
#      mean error, and the means in the safe and certain-death bins.
#    - The reference games are regenerated with their true states (same seeds and agent as
#      the cached pool, so they are identical games); --check verifies that.
#
# 3. Usage:
#    uv run python reports/2026-10-02_wumpus_diagnostics/head_coverage.py --pool-cache runs/wumpus/supervised_filter/pools.pt
# ==============================================================================

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from ipomdp.agents import WumpusReferenceAgent
from ipomdp.domain import WumpusSimulator, WumpusWorld
from ipomdp.domain.wumpus import GAME_OVER, NUM_CELLS, WumpusAction, WumpusState
from ipomdp.models import BeliefFilter, ObservationHead
from ipomdp.training import play_episodes, stream_seed

sys.path.insert(0, str(Path(__file__).parent))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="runs/wumpus/north50_seed0/.hydra/config.yaml")
    parser.add_argument("--pool-cache", type=Path, required=True)
    parser.add_argument("--filter-steps", type=int, default=6000)
    parser.add_argument("--head-steps", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda")
    cfg = OmegaConf.load(args.config)
    world = WumpusWorld(device)
    num_a, num_o, length = world.num_actions, world.num_observations, cfg.env.max_steps
    cached = torch.load(args.pool_cache, map_location=device)
    train, held = cached["train"], cached["held"]
    games = cached["games"]

    # 1. Supervised latent (supervised_filter.py's targets and readout), then frozen.
    torch.manual_seed(args.seed)
    belief_filter = BeliefFilter(num_a, num_o, cfg.model.latent_dim, cfg.model.hidden_dim, cfg.model.num_blocks).to(device)
    readout = nn.Sequential(nn.Linear(cfg.model.latent_dim, 256), nn.SiLU(),
                            nn.Linear(256, 3 * NUM_CELLS + NUM_CELLS + 4 + 1)).to(device)
    optimizer = torch.optim.Adam([*belief_filter.parameters(), *readout.parameters()], lr=3e-4)

    def unroll(data, rows):
        return belief_filter.unroll(F.one_hot(data["actions"][rows], num_a).float(),
                                    F.one_hot(data["observations"][rows], num_o).float(),
                                    F.one_hot(data["initial"][rows], num_o).float())       # (b, T+1, D)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    for _ in range(args.filter_steps):
        rows = torch.randint(0, train["actions"].shape[0], (64,), device=device, generator=generator)
        alive = train["alive"][rows]
        out = readout(unroll(train, rows)[alive])
        loss = (F.binary_cross_entropy_with_logits(out[:, :48], train["hazards"][rows][alive])
                + F.cross_entropy(out[:, 48:64], train["cell"][rows][alive])
                + F.cross_entropy(out[:, 64:68], train["heading"][rows][alive])
                + F.binary_cross_entropy_with_logits(out[:, 68], train["danger"][rows][alive]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    belief_filter.eval()

    # 2. Transitions (z_t, a_t, o_{t+1}) of alive steps, and the forced-FORWARD augmentation.
    with torch.no_grad():
        z_parts, a_parts, o_parts = [], [], []
        for start in range(0, train["actions"].shape[0], 512):
            rows = slice(start, start + 512)
            z = unroll(train, rows)[:, :-1]
            alive = train["alive"][rows][:, :-1]
            z_parts.append(z[alive]); a_parts.append(train["actions"][rows][alive]); o_parts.append(train["observations"][rows][alive])
        normal = (torch.cat(z_parts), torch.cat(a_parts), torch.cat(o_parts))

        reference, states = play_episodes(WumpusSimulator(world, games, length, stream_seed(2026, 1), device),
                                          WumpusReferenceAgent(world, games, 0.4))
        if not torch.equal(reference.actions, train["actions"][:games]):
            raise RuntimeError("Regenerated reference games differ from the cached pool.")
        states = WumpusState.from_tensor(states[:, :-1])                                         # s_0..s_{T-1}
        alive = ~states.done
        forward = torch.full_like(states.cell, int(WumpusAction.FORWARD))
        _, forced_observation, _ = world.step_rules(states, forward)
        z_ref = torch.cat([unroll(train, slice(s, s + 512))[:, :-1] for s in range(0, games, 512)])
        augmented = (torch.cat([normal[0], z_ref[alive]]), torch.cat([normal[1], forward[alive]]),
                     torch.cat([normal[2], forced_observation[alive]]))
        held_z = torch.cat([unroll(held, slice(s, s + 512)) for s in range(0, held["actions"].shape[0], 512)])
        held_z, held_danger = held_z[held["alive"]], held["danger"][held["alive"]]
    print(f"transitions: normal {len(normal[0])} (FORWARD {(normal[1] == 0).float().mean():.2f}, "
          f"GAME_OVER after FORWARD {(normal[2][normal[1] == 0] == GAME_OVER).float().mean():.3f}); "
          f"augmented {len(augmented[0])} (GAME_OVER after FORWARD "
          f"{(augmented[2][augmented[1] == 0] == GAME_OVER).float().mean():.3f})", flush=True)

    # 3. Fresh heads on frozen latents; calibration on held-out games.
    for name, (z, a, o) in (("normal", normal), ("augmented", augmented)):
        torch.manual_seed(args.seed)
        head = ObservationHead(cfg.model.latent_dim, num_a, num_o, cfg.model.hidden_dim, cfg.model.num_blocks).to(device)
        head_optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
        head_generator = torch.Generator(device=device).manual_seed(args.seed)
        for _ in range(args.head_steps):
            rows = torch.randint(0, len(z), (16 * length,), device=device, generator=head_generator)
            loss = F.cross_entropy(head(z[rows], F.one_hot(a[rows], num_a).float()), o[rows])
            head_optimizer.zero_grad()
            loss.backward()
            head_optimizer.step()
        with torch.no_grad():
            p = F.softmax(head(held_z, F.one_hot(torch.zeros(len(held_z), dtype=torch.int64, device=device), num_a).float()), -1)
            death = p[:, GAME_OVER]
        safe, certain = held_danger < 1e-9, held_danger > 0.6
        print(f"RESULT {name}_head_danger_error {(death - held_danger).abs().mean():.4f}")
        print(f"RESULT {name}_head_safe {death[safe].mean():.4f}")
        print(f"RESULT {name}_head_certain_death {death[certain].mean():.4f}")


if __name__ == "__main__":
    main()
