# ABSOLUTE PATH: src/ipomdp/domain/tiger.py
# ==============================================================================
# CANONICAL TIGER POMDP (Kaelbling, Littman & Cassandra, 1998)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Verbatim Reproduction of the Reference Specification:
#    - The model below reproduces Cassandra's reference file `tiger.95.POMDP`
#      (https://www.pomdp.org/examples/tiger.95.POMDP) entry for entry:
#         discount: 0.95
#         start:    0.5 0.5
#         T:listen      identity      O:listen      [[0.85, 0.15], [0.15, 0.85]]
#         T:open-left   uniform       O:open-left   uniform
#         T:open-right  uniform       O:open-right  uniform
#         R:listen -1 ; R:open-<door> +10 if the tiger is behind the other door, -100 otherwise.
#    - Nothing may be added to or changed in this domain: published optimal values
#      (V*(b0) = 19.37 for gamma = 0.95) are only comparable for the exact model.
#
# 2. Consequences That Earlier Iterations Got Wrong (kept here as guard rails):
#    - There is NO observation at t = 0. The agent acts first, from b0 = (0.5, 0.5).
#      Emitting a growl at reset hands the agent a free 85%-accurate observation.
#    - After OPEN_LEFT / OPEN_RIGHT the tiger is re-placed uniformly AND the observation is
#      uniform, i.e. carries no information. The belief after any door opening is exactly
#      b0 regardless of the observation received.
#    - The problem is continuing (infinite horizon, discounted). Opening a door is not a
#      terminal event; any episode boundary used for training is an artificial truncation.
#
# 3. Multi-Agent Tiger Is Deliberately Absent:
#    - The two-agent Tiger of Gmytrasiewicz & Doshi (2005) adds creak observations (90%
#      accurate) that depend on the other agent's action. Its exact joint tables will be
#      added, verified against the paper, when the interactive setting is studied. Until
#      then the single-agent canonical model is the only Tiger in the codebase.
# ==============================================================================

from enum import IntEnum

import torch

from .pomdp import FinitePOMDP

# Both literals are written out (rather than 1 - 0.85 = 0.15000000000000002) so the tensor is
# bit-identical to the reference file.
LISTEN_ACCURACY = 0.85
LISTEN_ERROR = 0.15
LISTEN_REWARD = -1.0
TREASURE_REWARD = 10.0
TIGER_REWARD = -100.0
DISCOUNT = 0.95


class TigerState(IntEnum):
    """Hidden tiger location."""

    TIGER_LEFT = 0
    TIGER_RIGHT = 1


class TigerAction(IntEnum):
    """Agent actions."""

    LISTEN = 0
    OPEN_LEFT = 1
    OPEN_RIGHT = 2


class TigerObservation(IntEnum):
    """Growl direction heard by the agent."""

    GROWL_LEFT = 0
    GROWL_RIGHT = 1


def build_tiger_pomdp() -> FinitePOMDP:
    """
    Builds the canonical Tiger POMDP exactly as specified in `tiger.95.POMDP`.

    Returns:
        FinitePOMDP with |S| = 2, |A| = 3, |O| = 2 and discount 0.95.
    """
    num_s = len(TigerState)
    identity = torch.eye(num_s, dtype=torch.float64)
    uniform_states = torch.full((num_s, num_s), 1.0 / num_s, dtype=torch.float64)

    transition = torch.stack([
        identity,        # LISTEN: the tiger stays where it is.
        uniform_states,  # OPEN_LEFT: problem resets, tiger re-placed uniformly.
        uniform_states,  # OPEN_RIGHT: problem resets, tiger re-placed uniformly.
    ])

    listen_obs = torch.tensor([
        [LISTEN_ACCURACY, LISTEN_ERROR],  # tiger left  -> growl left w.p. 0.85
        [LISTEN_ERROR, LISTEN_ACCURACY],  # tiger right -> growl right w.p. 0.85
    ], dtype=torch.float64)
    uninformative_obs = torch.full((num_s, len(TigerObservation)), 1.0 / len(TigerObservation), dtype=torch.float64)
    observation = torch.stack([listen_obs, uninformative_obs, uninformative_obs])

    reward = torch.tensor([
        [LISTEN_REWARD, LISTEN_REWARD],      # LISTEN
        [TIGER_REWARD, TREASURE_REWARD],     # OPEN_LEFT:  tiger left -> eaten; tiger right -> treasure
        [TREASURE_REWARD, TIGER_REWARD],     # OPEN_RIGHT: tiger left -> treasure; tiger right -> eaten
    ], dtype=torch.float64)

    return FinitePOMDP(
        transition=transition,
        observation=observation,
        reward=reward,
        initial_belief=torch.full((num_s,), 1.0 / num_s, dtype=torch.float64),
        discount=DISCOUNT,
        state_names=tuple(s.name for s in TigerState),
        action_names=tuple(a.name for a in TigerAction),
        observation_names=tuple(o.name for o in TigerObservation),
    )
