# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/saved_networks.py
# ==============================================================================
# SAVED NETWORK SETS IN ANOTHER RUN'S LAYOUT (objective_ab.py 2l)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Why:
#    - retrain_play.py and feature_localise.py swap saved network sets (objective_ab.py --save) into a run's
#      planner, and require the saved keys to equal run.networks. A set trained with JEPA parts (JEPA, or the
#      decoder with the self-predictive term, training/run.py section 18) holds "world_model" (the filter, the
#      predictor and the EMA target filter) where a decoder run holds "belief_filter".
#    - The planner reads only the acting copies and the heads, which have the same keys and shapes in both
#      layouts; the predictor and target filter are training-only. So such a set plays in a decoder run once
#      "world_model" is replaced by its belief_filter part. Nothing else is renamed or dropped, and any other
#      key mismatch still raises.
# ==============================================================================

PREFIX = "belief_filter."


def in_layout(states: dict, run_keys) -> dict:
    """The saved set with "world_model" mapped to "belief_filter" when the run needs that (header 1)."""
    run_keys = set(run_keys)
    if "world_model" in states and "belief_filter" in run_keys and "world_model" not in run_keys:
        states = dict(states)
        world = states.pop("world_model")
        states["belief_filter"] = {k[len(PREFIX):]: v for k, v in world.items() if k.startswith(PREFIX)}
    if set(states) != run_keys:
        raise ValueError(f"saved networks {sorted(states)} do not match the run's {sorted(run_keys)}.")
    return states
