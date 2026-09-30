# ABSOLUTE PATH: src/ipomdp/models/ema.py
# ==============================================================================
# EXPONENTIAL MOVING AVERAGES OF NETWORK WEIGHTS
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. One Update Rule, Three Uses:
#      target <- m * target + (1 - m) * online        (per parameter, in place, no gradients)
#    - JEPA target filter (models/world_model.py, section 4): stop-gradient self-prediction
#      targets.
#    - Target value head (training/trainer.py, section 3): the slow V-bar of the Bellman targets.
#    - Polyak-averaged acting model (training/run.py, section 4): the weights the planner acts
#      and is evaluated with.
#    - Each use keeps its own copy and momentum; only the arithmetic is shared.
# ==============================================================================

import copy

import torch
import torch.nn as nn


def frozen_copy(module: nn.Module) -> nn.Module:
    """A deep copy of `module` that receives no gradients (the initial EMA target)."""
    return copy.deepcopy(module).requires_grad_(False)


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, momentum: float) -> None:
    """target <- momentum * target + (1 - momentum) * online, parameter by parameter."""
    for target_param, online_param in zip(target.parameters(), online.parameters(), strict=True):
        target_param.mul_(momentum).add_(online_param, alpha=1.0 - momentum)
