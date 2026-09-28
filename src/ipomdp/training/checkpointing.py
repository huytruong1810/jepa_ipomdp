# ABSOLUTE PATH: src/ipomdp/training/checkpointing.py
# ==============================================================================
# ATOMIC CHECKPOINT FILES
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. What Is Saved Is Decided by the Owner of the State:
#    - TrainingRun.state_dict() defines the complete run state (networks, optimiser, replay
#      buffer, simulators, planners, agents, all random generators). This module only writes
#      and reads such dictionaries; it has no knowledge of their contents.
#
# 2. Two Files per Run Directory:
#      latest.pt  full run state, written every save_every collections and on SIGINT; the file
#                 to pass as `resume=` to continue a run bit-for-bit.
#      best.pt    networks + collection index of the best GREEDY EVALUATION return so far.
#    - An earlier version selected "best" by training loss, which says nothing about the
#      quality of the policy in an RL loop.
#
# 3. Atomic Writes, Loud Reads:
#    - Files are written to a temporary name and renamed, so a crash never leaves a truncated
#      checkpoint. Loading a missing file raises; there is no silent "start from scratch".
#
# 4. Always Load on the CPU:
#    - Random-generator states are CPU ByteTensors and must stay there (torch.Generator and
#      torch.set_rng_state reject CUDA tensors). Every owner's load_state_dict moves its own
#      tensors to its device, so checkpoints are read with map_location="cpu".
# ==============================================================================

from pathlib import Path

import torch


def save_checkpoint(path: Path, state: dict) -> None:
    """Atomically writes `state` to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path) -> dict:
    """Reads a checkpoint written by save_checkpoint onto the CPU (raises if it does not exist)."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)
