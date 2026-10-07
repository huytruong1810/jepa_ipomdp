# ABSOLUTE PATH: reports/2026-10-07_wumpus_reviewer_page/scripts/export.py
# ==============================================================================
# DATA EXPORT FOR THE WUMPUS REVIEWER PAGE (data.json)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. One script, one JSON file, every number on the page traceable to a run directory:
#    - returns:      <run>/analysis/report.json of every listed run (512 fresh games on the shared
#                    analysis seed; random hunter and exact-belief reference agents from the same
#                    report), plus the full-information bound 936.3 (full_info_vi.py).
#    - curves:       greedy in-training evaluations parsed from <run>/main.log.
#    - games:        held-out games of the MAIN run's greedy planner, step by step: exact hazard
#                    marginals (enumeration filter) vs the MLP probe's decoding of the acting latent,
#                    hunter cell and heading, action, percept, reward, exact danger ahead and the
#                    planning model's P(death | FORWARD). The probe is fitted on the other games.
#    - calibration:  the observation head's P(death | FORWARD) against the realised death, in bins,
#                    on the run's own replay buffer (in-sample), on fresh exploring-policy games and
#                    on fresh greedy games (as head_shift.py), for the main run and the "before" run.
#    - overfit_ab:   objective_ab.py logs of runs/wumpus/overfit_ab (head NLL train/held, safe-miss).
#    - probes:       probe scores from each report.json, plus the privileged supervised bound.
#    - page_meta.json (next to this script, hand-written): run labels, the main runs' name prefix and
#      the status paragraphs; merged into data.json unchanged.
#    - games also carry the TRUE layout (pits, wumpus, gold, from the simulator's hidden state) and
#      the true arrow / wumpus-alive flags per step, so a reviewer can follow the real map; each
#      game's per-step latents go to games_latents.npz next to the JSON, for umap_latents.py to
#      project into the same UMAP map.
#    - Usage: uv run python <this> <main run dir> <before run dir> <out json> [other run dirs...]
#             uv run python <this> games <main run dir> <existing out json>   (refresh games only)
#      The main run dir supplies the games and the "after" calibration; every run dir whose name
#      starts with page_meta's main_prefix is pooled as the current agent's seeds.
# ==============================================================================

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ipomdp.agents import PlanningAgent
from ipomdp.domain import WumpusSimulator
from ipomdp.domain.wumpus import GAME_OVER, GRID
from ipomdp.experiments.runs import build_training_run
from ipomdp.interpretability import build_wumpus_dataset, fit_mlp_wumpus_probe
from ipomdp.planning import BeliefTreeSearch
from ipomdp.training import EpisodeBatch, load_checkpoint, play_episodes, stream_seed
from omegaconf import OmegaConf

DEV = torch.device("cuda")
FORWARD = 0
DELTA = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=DEV)
BINS = ((0, 0.02), (0.02, 0.2), (0.2, 0.5), (0.5, 1.01))


def load(run_dir: Path, name: str):
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    setup, run = build_training_run(cfg, DEV)
    run.load_state_dict(load_checkpoint(run_dir / "checkpoints" / name))
    return cfg, setup.domain, run


def cat(parts: list[EpisodeBatch]) -> EpisodeBatch:
    return EpisodeBatch(*(torch.cat([getattr(p, f) for p in parts]) for f in
                          ("actions", "observations", "rewards", "initial_observations")))


def play(world, agent, games: int, length: int, seed: int) -> EpisodeBatch:
    return cat([play_episodes(WumpusSimulator(world, agent.batch_size, length, stream_seed(seed + k, 1), DEV), agent)[0]
                for k in range(games // agent.batch_size)])


def unroll(belief_filter, eps: EpisodeBatch, num_a: int, num_o: int) -> torch.Tensor:
    out = []
    with torch.no_grad():
        for i in range(0, eps.actions.shape[0], 512):
            out.append(belief_filter.unroll(F.one_hot(eps.actions[i:i + 512], num_a).float(),
                                            F.one_hot(eps.observations[i:i + 512], num_o).float(),
                                            F.one_hot(eps.initial_observations[i:i + 512], num_o).float()).float())
    return torch.cat(out)


def p_death(head, latents: torch.Tensor, num_a: int) -> torch.Tensor:
    with torch.no_grad():
        a = F.one_hot(torch.full((len(latents),), FORWARD, device=DEV), num_a).float()
        return F.softmax(head(latents, a).float(), -1)[:, GAME_OVER].double()


def calibration(run, eps: EpisodeBatch, num_a: int, num_o: int) -> list[dict]:
    z = unroll(run.acting_filter, eps, num_a, num_o)
    obs_before = torch.cat([eps.initial_observations[:, None], eps.observations[:, :-1]], 1)
    alive = torch.cumsum((obs_before == GAME_OVER).int(), 1) == 0
    mask = alive & (eps.actions == FORWARD)
    predicted = p_death(run.acting_observation_head, z[:, :-1][mask], num_a)
    died = (eps.rewards[mask] <= -1000).double()
    rows = []
    for lo, hi in BINS:
        m = (predicted >= lo) & (predicted < hi)
        rows.append({"lo": lo, "hi": min(hi, 1.0), "n": int(m.sum()),
                     "predicted": float(predicted[m].mean()) if m.any() else None,
                     "died": float(died[m].mean()) if m.any() else None})
    return rows


def run_calibration(run_dir: Path, games: int) -> dict:
    print(f"calibration of {run_dir.name} ({games} fresh games per policy)", flush=True)
    cfg, world, run = load(run_dir, "latest.pt")
    num_a, num_o = world.num_actions, world.num_observations
    n = run.buffer.size
    b = run.buffer
    buffer_eps = EpisodeBatch(b._actions[:n], b._observations[:n], b._rewards[:n], b._initial_observations[:n])
    return {"run": run_dir.name,
            "buffer": calibration(run, buffer_eps, num_a, num_o),
            "exploring": calibration(run, play(world, run.train_agent, games, cfg.env.max_steps, 8765), num_a, num_o),
            "greedy": calibration(run, play(world, run.eval_agent, games, cfg.env.max_steps, 4321), num_a, num_o)}


def danger_ahead(d) -> torch.Tensor:
    x, y = d.cell % GRID + DELTA[d.heading, 0], d.cell // GRID + DELTA[d.heading, 1]
    inside = (x >= 0) & (x < GRID) & (y >= 0) & (y < GRID)
    ahead = (x + GRID * y).clamp(0, 15)
    return torch.where(inside, d.hazards[:, :16].gather(1, ahead[:, None]).squeeze(1)
                       + d.wumpus_alive.double() * d.hazards[:, 16:32].gather(1, ahead[:, None]).squeeze(1), 0.0)


def export_games(run_dir: Path, latents_out: Path, count: int = 3) -> list[dict]:
    """Held-out greedy games of best.pt (the analysed checkpoint), with exact vs decoded beliefs."""
    print(f"games of {run_dir.name}", flush=True)
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    setup, run = build_training_run(cfg, DEV)
    best = load_checkpoint(run_dir / "checkpoints" / "best.pt")
    for key, net in run.networks.items():
        net.load_state_dict(best["networks"][key])
    world = setup.domain
    num_a = world.num_actions
    model = run.eval_agent.model
    n = 256
    agent = PlanningAgent(model, BeliefTreeSearch(model, cfg.mcts.num_simulations, cfg.mcts.c_puct, cfg.mcts.dirichlet_alpha,
                                                  dirichlet_epsilon=0.0, seed=stream_seed(2027, 2),
                                                  observation_prune=cfg.mcts.observation_prune),
                          n, temperature=0.0, seed=stream_seed(2027, 3), device=DEV)
    eps, states = play_episodes(WumpusSimulator(world, n, cfg.env.max_steps, stream_seed(2027, 1), DEV), agent)
    half = n // 2
    pick = lambda rows: EpisodeBatch(eps.actions[rows], eps.observations[rows], eps.rewards[rows],  # noqa: E731
                                     eps.initial_observations[rows])
    fit = build_wumpus_dataset(world, run.acting_filter, pick(slice(0, half)))
    held_eps = pick(slice(half, None))
    held = build_wumpus_dataset(world, run.acting_filter, held_eps)
    probe = fit_mlp_wumpus_probe(fit, steps=3000)
    with torch.no_grad():
        decoded = probe.hazards(held.latents).double()
        danger = danger_ahead(held)
        predicted = p_death(run.acting_observation_head, held.latents, num_a)
    # Pick games that travel: most distinct cells visited; prefer one win, one death, one other.
    games = []
    outcome = lambda e: "win" if bool((held_eps.rewards[e] >= 998).any()) else (  # noqa: E731
        "death" if bool((held_eps.rewards[e] <= -1000).any()) else "unfinished")
    ranked = sorted(range(half), key=lambda e: -len(set(held.cell[held.episode == e].tolist())))
    chosen, seen = [], set()
    for e in ranked:
        if outcome(e) not in seen:
            chosen.append(e)
            seen.add(outcome(e))
        if len(chosen) == count:
            break
    game_latents = {}
    for gi, e in enumerate(chosen):
        rows = (held.episode == e).nonzero().flatten()
        hidden = states[half + e]                                      # (T + 1, 6): layout, cell, heading, arrow, wumpus alive, done
        layout = int(hidden[0, 0])
        truth = {"pits": world.pit[layout].nonzero().flatten().tolist(), "wumpus": int(world.wumpus[layout]),
                 "gold": int(world.gold[layout])}
        game_latents[f"game{gi}"] = held.latents[rows].cpu().numpy()
        steps = []
        for r in rows.tolist():
            t = int(held.step[r])
            steps.append({
                "t": t, "cell": int(held.cell[r]), "heading": int(held.heading[r]),
                "action": int(held_eps.actions[e, t]) if t < held_eps.actions.shape[1] else None,
                "percept": int(held_eps.initial_observations[e]) if t == 0 else int(held_eps.observations[e, t - 1]),
                "reward": float(held_eps.rewards[e, t]) if t < held_eps.rewards.shape[1] else None,
                "exact": [round(float(v), 4) for v in held.hazards[r]],
                "decoded": [round(float(v), 4) for v in decoded[r]],
                "danger": round(float(danger[r]), 4), "p_death": round(float(predicted[r]), 4),
                "arrow": int(hidden[t, 3]), "wumpus_alive": int(hidden[t, 4])})
        games.append({"outcome": outcome(e), "truth": truth, "steps": steps})
    np.savez_compressed(latents_out, **game_latents)
    return games


def curves(run_dirs: list[Path]) -> dict:
    out = {}
    for d in run_dirs:
        pts = re.findall(r"collection (\d+): greedy eval discounted return (-?[\d.]+) \+- ([\d.]+)",
                         (d / "main.log").read_text())
        out[d.name] = [[int(c), float(v), float(s)] for c, v, s in pts]
    return out


def overfit_ab(log_dir: Path) -> dict:
    out: dict = {}
    for path in sorted(log_dir.glob("obj_*_seed*.log")):
        m = re.fullmatch(r"obj_(.+)_seed(\d+)\.log", path.name)
        arm, seed = m.group(1), m.group(2)
        for updates, name, value in re.findall(r"RESULT@(\d+) (\w+) ([-\d.]+)", path.read_text()):
            out.setdefault(arm, {}).setdefault(name, {}).setdefault(updates, {})[seed] = float(value)
    return out


def main() -> None:
    if sys.argv[1] == "games":
        main_run, out_path = Path(sys.argv[2]), Path(sys.argv[3])
        data = json.loads(out_path.read_text())
        data["games"] = export_games(main_run, out_path.parent / "games_latents.npz")
        out_path.write_text(json.dumps(data, separators=(",", ":")))
        print(f"refreshed games in {out_path}")
        return
    main_run, before_run, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    return_runs = [Path(p) for p in sys.argv[4:]]
    reports = {d.name: json.loads((d / "analysis" / "report.json").read_text()) for d in return_runs + [main_run]}
    data = {
        "upper_bound": 936.3,
        "reports": {k: {"returns": v["returns"], "probes": v["probes"], "checkpoint_collection": v["checkpoint_collection"]}
                    for k, v in reports.items()},
        "curves": curves(return_runs + [main_run]),
        "supervised_bound": {"cell_acc": 0.999, "heading_acc": 1.0, "danger_error_3000": 0.032, "danger_error_12000": 0.016},
        "overfit_ab": overfit_ab(Path("runs/wumpus/overfit_ab")),
        "calibration": {"after": run_calibration(main_run, 256), "before": run_calibration(before_run, 256)},
        "games": export_games(main_run, out_path.parent / "games_latents.npz"),
        **json.loads((Path(__file__).parent / "page_meta.json").read_text()),
    }
    out_path.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
