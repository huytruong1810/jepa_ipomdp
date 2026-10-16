# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/head_shift.py
# ==============================================================================
# WHY THE DEATH PREDICTION IS CONFIDENTLY WRONG ON THE GREEDY PLANNER'S HISTORIES
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (user decision 2026-10-05, after the combined-fix run failed):
#    - greedy_deaths.py showed that on the greedy planner's own histories the acting observation
#      head predicts P(GAME_OVER | z, FORWARD) ~ 0.01 at FORWARD steps that kill 57-97% of the
#      time, while it reads 0.5-0.9 at such dangers on random/reference histories. Which of these
#      explains it?
#      H1 Acting copies: the planner pairs the Polyak-averaged filter with the Polyak-averaged
#         head; training fits the ONLINE head on ONLINE-filter latents. Averaging two networks'
#         weights separately need not average their composition.
#      H2 Distribution shift: the head is calibrated on its training data (the replay buffer)
#         but greedy histories reach latents the buffer does not cover.
#      H3 Missing information: the latent on greedy histories does not carry the danger at all.
#    - Usage: uv run python <this> <run dir> <checkpoint, e.g. latest.pt> [greedy games].
#      latest.pt carries the replay buffer, which is the head's training data.
#
# 2. Measurements (all at alive states where the action taken was FORWARD):
#    - Calibration by predicted-probability bin vs the REALISED death on that step, on buffer
#      transitions and on greedy transitions, for the acting pair and the online pair (H1, H2).
#    - Nearest-neighbour distance in latent space from greedy dangerous FORWARD latents to the
#      buffer's FORWARD latents, against buffer-to-buffer distances of held-out buffer episodes (H2).
#    - Fresh-head fits (the agent's ObservationHead architecture, Adam 3e-4, from scratch on frozen
#      acting latents): trained on buffer only vs buffer + half the greedy episodes, scored on the
#      held-out greedy half by realised death at FORWARD steps with exact danger >= 0.4. A fresh
#      head that learns the danger from greedy data shows the information is in the latent (H3
#      rejected) and that coverage is what is missing (H2).
#
# 3. Result 2026-10-05 (eps_decoder_k5_u256_seed0, latest.pt, 512 greedy games):
#    - H1 rejected: the online pair is as wrong as the acting pair (greedy FORWARDs: predicted
#      0.003 vs realised death 0.097; acting 0.002 vs 0.097).
#    - The head is CALIBRATED on its training data: all buffer FORWARDs predicted 0.071, died
#      0.071, every bin matches; the last 10 collections alone 0.061 vs 0.069. On greedy histories
#      the bin predicted < 0.02 dies 9.2% of the time (0.7% in the recent buffer).
#    - H2, but not as "far away" latents: greedy dangerous latents are as close to buffer FORWARD
#      latents as held-out buffer latents are (median NN distance 3.89 vs 4.08). Nearby buffer
#      latents carry the wrong outcome, so the shift is in which outcomes the data hold, not in
#      latent coverage.
#    - H3 mostly rejected: on held-out greedy FORWARDs at exact danger >= 0.4 (exact 0.63, died
#      0.67) the agent's head reads 0.007, a fresh head on the buffer 0.166, and a fresh head on the
#      buffer plus the other half of the greedy games 0.529 (safe steps 0.009). The latent carries
#      most of the danger; the data lack the greedy policy's own outcomes.
#    - So: the exploring policy (temperature >= 0.1, root Dirichlet noise 0.25, epsilon >= 0.1)
#      does not visit the states the greedy planner reaches, and the greedy planner steps exactly
#      where the head is wrong there.
#    - CORRECTED the same day by the fresh-exploring-games check (added afterwards): the head is
#      also miscalibrated on FRESH games of the exploring policy itself, so the cause is
#      OVERFITTING to the buffer's games, not a policy shift. Predicted < 0.02: died 0.1% in the
#      buffer, 4.3% on fresh exploring games, 9.2% on fresh greedy games; predicted 0.5-1 (mean
#      0.89): died 96% in the buffer, 40% on fresh exploring games. store_eval_eps_decoder_k5_u256
#      _seed0 (greedy games stored): 0.2% / 2.7% / 5.5% and 89% / 43%. The greedy planner is the
#      worst case because it selects the confident "safe" errors.
#
# 4. R0 of the representation plan (HANDOFF 6.2; pre-registered 2026-10-15 BEFORE any result was read):
#    - Networks: latest.pt of the six double-value runs, u64_store_eval_eps_decoder_k5_double_seed{0,1,2}
#      (current recipe) and u64_store_eval_eps_decoder_k5_privileged_double_seed{0,1,2} (oracle recipe),
#      512 greedy games each (runs/wumpus/head_shift_double_chain.sh). No code change from item 3.
#    - Question: on the planner's own greedy histories, is the oracle-recipe head calibrated where the
#      current-recipe head is confidently "safe" and wrong (C6), and is the danger in the current latent
#      anyway?
#    - Metrics per run:
#      * confident-safe miss rate = realised death of the acting pair's greedy FORWARDs in the predicted
#        bin [0, 0.02). MISCALIBRATED if >= 0.04 (twice the bin edge) with n >= 200; CALIBRATED if
#        <= 0.02; in between is ambiguous and counts against neither.
#      * fresh-head danger = mean predicted death on held-out greedy FORWARDs at exact danger >= 0.4, for
#        the fresh head on the buffer only and on the buffer + greedy half. RECOVERS if >= 0.40, LACKS if
#        < 0.25.
#    - Decision (each recipe by majority of its 3 seeds):
#      (A) current miscalibrated, oracle calibrated, buffer-only fresh head on current latents recovers:
#          the danger is in the latent and the head overfits its games -> R1 screens head-side arms
#          (ensemble/uncertainty heads; the NLL-early-stopped detached head already failed, HANDOFF 7).
#      (B) as (A) but only buffer + greedy recovers: outcome coverage of the data is missing -> R1
#          screens head/data arms.
#      (C) neither fresh head recovers on current latents: the latent lacks the danger -> R1 screens
#          representation-objective arms.
#      (D) the oracle runs are miscalibrated too: calibration does not separate the recipes -> R1 starts
#          from the C4 no-op drift check.
#    - Result 2026-10-15 (logs <run>.head_shift.log, ~20 min per run, 2 at a time):
#      * Confident-safe miss rate, current 0.002 / 0.000 / 0.000, oracle 0.000 / 0.001 / 0.002: ALL SIX
#        CALIBRATED. No pre-registered branch matches: (A)-(C) need the current head miscalibrated, (D) the
#        oracle head. The premise of R0 (current head confidently wrong where the oracle head is right) is
#        false for these networks. Read by (D)'s logic, stated as a deviation: the observation head's
#        calibration does not separate the recipes, so R1 starts from C4 (no-op drift, which does separate
#        them: below).
#      * Fresh-head danger on current latents, buffer only 0.609 / 0.141 / 0.381, buffer + greedy 0.755 /
#        0.214 / 0.590; on oracle latents 0.000 / 0.253 / 0.285 and 0.004 / 0.171 / 0.436. But the held-out
#        exact-danger samples are n = 11 / 7 / 5 and 1 / 12 / 63: uninformative, no verdict.
#      * Not pre-registered: greedy deaths at latest.pt are ~5.7% / 1.6% / 1.7% (current) and 0% / 2.5% /
#        14% (oracle seed 2: 66 of ~72 in the predicted [0.2, 0.5) bin, realised 0.56). Deaths sit in the
#        moderate bins (current seed 0: [0.02, 0.2) pred 0.044, died 0.198), where the planner takes risks
#        rated 2-4x too low, not in confident-"safe" errors.
#      * CAVEAT, my miss at pre-registration: this reads latest.pt (collection 100, it carries the buffer),
#        whereas the compared returns (223 vs 138) and deaths (analysis) are of best.pt (collections 55-90).
#        The final networks differ in behaviour (oracle seed 2 dies 14% here).
#      * C4 check (scratch, best.pt's 128 stored greedy games, CPU 7 s per network): relative latent drift
#        |dz|/|z| on exact no-ops (GRAB without GLITTER, SHOOT without the arrow) is mean 0.217 / 0.117 /
#        0.098 (median 0.102 / 0.022 / 0.016) for the current recipe and 0.030 / 0.023 / 0.021 (median
#        0.004) for the oracle recipe, against 0.47-0.81 on other actions in both. The agent's own latent
#        moves 4-7x more on no-ops; this is a consistent difference between the two latents.
#      * C4 follow-up (scratch, best.pt, held-out random pool games, MLP probes): the drift does NOT cost
#        cell/heading information. Unmatched, accuracy falls with the number of no-ops seen (cell 0.97 ->
#        0.32), but at matched steps (15-25) it RISES with it (fewer moves to dead-reckon), and skipping the
#        filter update on exact no-ops gains only +1 to +4 points overall (current seeds; oracle unchanged at
#        95-100%). The agent's cell/heading gap is in tracking moves and turns, not in the no-ops. So a C4
#        consistency term is not a promising R1 arm.
#
# 5. R0' (2026-10-15; corrects item 4's checkpoint miss). Usage: <run> latest.pt 512 best.pt, i.e. best.pt's
#    networks (the ones whose returns were compared: 223 vs 138) on latest.pt's buffer for the fresh heads.
#    Pre-registered before any result, the six double runs:
#    - Per run: greedy FORWARD deaths in the acting head's [0, 0.02) bin (n x died) as a share of all greedy
#      FORWARD deaths, and the miss rate in [0, 0.02) and in [0.02, 0.5) (realised / predicted).
#    - (S) confident-safe errors: on 2+ of 3 current runs the [0, 0.02) bin holds >= half the deaths with miss
#      rate >= 0.04, and not on 2+ oracle runs -> R1 on the head's confident errors (item 4's (A)-(C) logic,
#      with the fresh-head numbers where n >= 30).
#    - (M) moderate-risk underestimation: on 2+ current runs most deaths are in [0.02, 0.5) with realised /
#      predicted >= 2, and not on 2+ oracle runs -> R1 on calibration in the moderate range.
#    - (N) neither separates the recipes -> the death gap is not the observation head's calibration; R1 moves
#      to what the representation lacks for dead reckoning (moves and turns), per item 4's C4 follow-up.
#    - Result 2026-10-15 (logs <run>.head_shift_best.log): (M) MODERATE-RISK UNDERESTIMATION.
#      deaths in [0, 0.02) / in [0.02, 0.5) (n x died), realised / predicted in [0.02, 0.2):
#        current seed 0   ~8 (miss 0.001) / ~10   0.044 / 0.058 = 0.76
#        current seed 1   ~5 (miss 0.002) / ~41   0.129 / 0.059 = 2.2
#        current seed 2   ~0              / ~44   0.346 / 0.043 = 8.0
#        oracle seeds     ~0 / 0 / ~6 (miss <= 0.001)  /  ~1 / 0 / ~1   (0.37 / 0 / 0.22)
#      (S) on 0 of 3 current runs; (M) on 2 of 3 current (seeds 1, 2) and 0 of 3 oracle runs. The greedy
#      current planner also steps into the moderate bin 2-4x as often (227 / 311 / 127 FORWARDs vs 75 / 47 /
#      84). On fresh EXPLORING games both recipes' heads underrate this bin ~2x, so the greedy planner selects
#      the head's underestimates (a winner's curse on the model).
#      Not pre-registered: at held-out greedy FORWARDs with exact danger >= 0.4, where n is usable (current
#      seeds 1, 2: n = 25, 21), the agent's head reads 0.066 / 0.045 and a FRESH head on the same frozen
#      latents (buffer only) 0.354 / 0.826 (died 0.64 / 1.00). The danger is in the latent; the jointly trained
#      head does not report it. Caveat: the fresh heads train on latest.pt's buffer, which includes the
#      collections after best.pt.
#      -> R1 on the head side: the planning model's death estimate in the moderate range (HANDOFF 6.2).
# ==============================================================================

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from ipomdp.domain import WumpusSimulator
from ipomdp.domain.wumpus import GAME_OVER, GRID
from ipomdp.experiments.runs import build_training_run
from ipomdp.interpretability import build_wumpus_dataset
from ipomdp.models import ObservationHead
from ipomdp.training import EpisodeBatch, load_checkpoint, play_episodes, stream_seed

run_dir, name = Path(sys.argv[1]), sys.argv[2]
games = int(sys.argv[3]) if len(sys.argv) > 3 else 512
dev = torch.device("cuda")
cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
setup, run = build_training_run(cfg, dev)
run.load_state_dict(load_checkpoint(run_dir / "checkpoints" / name))
if len(sys.argv) > 4:                     # header 5: another checkpoint's networks on <name>'s buffer
    networks = load_checkpoint(run_dir / "checkpoints" / sys.argv[4])["networks"]
    for net_name, net in run.networks.items():
        net.load_state_dict(networks[net_name])
    name = f"{sys.argv[4]} networks, {name} buffer"
world, num_a, num_o = setup.domain, setup.domain.num_actions, setup.domain.num_observations
FORWARD = 0
torch.manual_seed(0)


def unroll(belief_filter, episodes: EpisodeBatch, chunk: int = 512) -> torch.Tensor:
    """Latents z_0..z_T (B, T + 1, D) of `belief_filter` in float32, without autocast."""
    out = []
    with torch.no_grad():
        for i in range(0, episodes.actions.shape[0], chunk):
            a = F.one_hot(episodes.actions[i:i + chunk], num_a).float()
            o = F.one_hot(episodes.observations[i:i + chunk], num_o).float()
            o0 = F.one_hot(episodes.initial_observations[i:i + chunk], num_o).float()
            out.append(belief_filter.unroll(a, o, o0).float())
    return torch.cat(out)


def forward_transitions(latents: torch.Tensor, episodes: EpisodeBatch):
    """(latents, died) at alive FORWARD steps; alive = no GAME_OVER observed before step t."""
    obs_before = torch.cat([episodes.initial_observations[:, None], episodes.observations[:, :-1]], 1)
    alive = torch.cumsum((obs_before == GAME_OVER).int(), 1) == 0
    mask = alive & (episodes.actions == FORWARD)
    return latents[:, :-1][mask], (episodes.rewards[mask] <= -1000).double(), mask


def p_death(head, latents: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        a = F.one_hot(torch.full((len(latents),), FORWARD, device=dev), num_a).float()
        return F.softmax(head(latents, a).float(), -1)[:, GAME_OVER].double()


def calibration(label: str, predicted: torch.Tensor, died: torch.Tensor) -> None:
    rows = []
    for lo, hi in ((0, 0.02), (0.02, 0.2), (0.2, 0.5), (0.5, 1.01)):
        m = (predicted >= lo) & (predicted < hi)
        if m.any():
            rows.append(f"[{lo},{hi}) n={int(m.sum())} pred {predicted[m].mean():.3f} died {died[m].mean():.3f}")
    print(f"  {label:34s} overall pred {predicted.mean():.3f} died {died.mean():.3f} | " + " | ".join(rows))


# --- Data: the replay buffer (training data) and fresh greedy games --------------------------------
buf = run.buffer
n = buf.size
buffer_eps = EpisodeBatch(buf._actions[:n], buf._observations[:n], buf._rewards[:n], buf._initial_observations[:n])
eval_agent = run.eval_agent
greedy_parts = []
for k in range(games // eval_agent.batch_size):
    ep, _ = play_episodes(WumpusSimulator(world, eval_agent.batch_size, cfg.env.max_steps, stream_seed(4321 + k, 1), dev),
                          eval_agent)
    greedy_parts.append(ep)
greedy_eps = EpisodeBatch(*(torch.cat([getattr(p, f) for p in greedy_parts]) for f in
                            ("actions", "observations", "rewards", "initial_observations")))
print(f"== {run_dir.name} {name}: buffer {n} episodes, greedy {greedy_eps.actions.shape[0]} episodes")

pairs = {"acting": (run.acting_filter, run.acting_observation_head), "online": (run.belief_filter, run.observation_head)}
for pair, (filt, head) in pairs.items():
    for label, eps in (("buffer", buffer_eps), ("greedy", greedy_eps)):
        z, died, _ = forward_transitions(unroll(filt, eps), eps)
        calibration(f"{pair} pair, {label} FORWARDs", p_death(head, z), died)
recent = n - 10 * cfg.training.env_batch_size          # the buffer is chronological (never wrapped here)
recent_eps = EpisodeBatch(*(t[recent:] for t in (buffer_eps.actions, buffer_eps.observations, buffer_eps.rewards,
                                                  buffer_eps.initial_observations)))
z, died, _ = forward_transitions(unroll(run.acting_filter, recent_eps), recent_eps)
calibration("acting pair, last 10 collections", p_death(run.acting_observation_head, z), died)
# Fresh games of the EXPLORING policy (new layouts): separates overfitting to the buffer's games
# (wrong here too) from a shift between the exploring and the greedy policy (right here).
train_agent = run.train_agent
explore_parts = [play_episodes(WumpusSimulator(world, train_agent.batch_size, cfg.env.max_steps,
                                               stream_seed(8765 + k, 1), dev), train_agent)[0]
                 for k in range(games // train_agent.batch_size)]
explore_eps = EpisodeBatch(*(torch.cat([getattr(p, f) for p in explore_parts]) for f in
                             ("actions", "observations", "rewards", "initial_observations")))
z, died, _ = forward_transitions(unroll(run.acting_filter, explore_eps), explore_eps)
calibration("acting pair, fresh exploring games", p_death(run.acting_observation_head, z), died)

# --- Exact danger on greedy histories (for the held-out score) ------------------------------------
acting_filter, acting_head = pairs["acting"]
g_lat = unroll(acting_filter, greedy_eps)
z_g, died_g, mask_g = forward_transitions(g_lat, greedy_eps)
d = build_wumpus_dataset(world, acting_filter, greedy_eps)
delta = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=dev)
x, y = d.cell % GRID + delta[d.heading, 0], d.cell // GRID + delta[d.heading, 1]
inside = (x >= 0) & (x < GRID) & (y >= 0) & (y < GRID)
ahead = (x + GRID * y).clamp(0, 15)
exact_all = torch.where(inside, d.hazards[:, :16].gather(1, ahead[:, None]).squeeze(1)
                        + d.wumpus_alive.double() * d.hazards[:, 16:32].gather(1, ahead[:, None]).squeeze(1), 0.0)
exact_grid = torch.full(g_lat.shape[:2], float("nan"), dtype=torch.float64, device=dev)
exact_grid[d.episode, d.step] = exact_all
exact_g = exact_grid[:, :-1][mask_g]
episode_g = torch.arange(greedy_eps.actions.shape[0], device=dev)[:, None].expand_as(mask_g)[mask_g]

# --- H2: nearest-neighbour distance to the buffer's FORWARD latents -------------------------------
b_lat = unroll(acting_filter, buffer_eps)
z_b, died_b, mask_b = forward_transitions(b_lat, buffer_eps)
episode_b = torch.arange(n, device=dev)[:, None].expand_as(mask_b)[mask_b]
held_b = episode_b >= int(0.9 * n)
reference_b, query_b = z_b[~held_b], z_b[held_b][:4096]


def nn_distance(query: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.cdist(q, reference_b).min(1).values for q in query.split(1024)])


danger = exact_g >= 0.4
print(f"  NN distance to buffer FORWARD latents (median): held-out buffer {nn_distance(query_b).median():.3f}, "
      f"greedy safe {nn_distance(z_g[~danger][:4096]).median():.3f}, greedy danger>=0.4 {nn_distance(z_g[danger]).median():.3f} "
      f"(n={int(danger.sum())})")

# --- H2/H3: fresh heads on frozen acting latents --------------------------------------------------
all_b = b_lat[:, :-1].reshape(-1, b_lat.shape[-1])
act_b, obs_b = buffer_eps.actions.reshape(-1), buffer_eps.observations.reshape(-1)
train_g = torch.arange(greedy_eps.actions.shape[0], device=dev) < greedy_eps.actions.shape[0] // 2
all_g = g_lat[train_g][:, :-1].reshape(-1, g_lat.shape[-1])
act_g, obs_g = greedy_eps.actions[train_g].reshape(-1), greedy_eps.observations[train_g].reshape(-1)
test = ~train_g[episode_g]


def fit_head(latents, actions, observations, steps=6000, batch=4096):
    head = ObservationHead(latents.shape[-1], num_a, num_o, cfg.model.hidden_dim, cfg.model.num_blocks).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=3e-4)
    for _ in range(steps):
        i = torch.randint(0, len(latents), (batch,), device=dev)
        loss = F.cross_entropy(head(latents[i], F.one_hot(actions[i], num_a).float()), observations[i])
        opt.zero_grad(); loss.backward(); opt.step()
    return head.eval()


print("  held-out greedy FORWARDs (second half of the greedy games):")
for label, head in (("agent's acting head", acting_head),
                    ("fresh head, buffer only", fit_head(all_b, act_b, obs_b)),
                    ("fresh head, buffer + greedy half", fit_head(torch.cat([all_b, all_g]), torch.cat([act_b, act_g]),
                                                                   torch.cat([obs_b, obs_g])))):
    p = p_death(head, z_g[test])
    m = test & danger
    print(f"    {label:34s} danger>=0.4: n={int((exact_g[test] >= 0.4).sum())} pred {p[exact_g[test] >= 0.4].mean():.3f} "
          f"exact {exact_g[m].mean():.3f} died {died_g[m].mean():.3f} | safe (exact 0): pred {p[exact_g[test] == 0].mean():.3f}")
