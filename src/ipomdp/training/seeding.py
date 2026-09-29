# ABSOLUTE PATH: src/ipomdp/training/seeding.py
# ==============================================================================
# INDEPENDENT RANDOM STREAMS FROM ONE BASE SEED
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Every Random Consumer Gets Its Own Stream:
#    - A run has several independent random consumers (network initialisation, collection and
#      evaluation simulators, replay sampling, planners, action sampling); an analysis has more
#      (one simulator per data set, agents). Each is seeded with stream_seed(base, stream), a
#      distinct stream id per consumer, so no two consumers share a generator sequence.
#
# 2. Hashed, Not Offset (Phase-8 finding):
#    - Streams used to be seeded base + offset (offsets 1-7). Within one run that kept the
#      consumers apart, but across runs it did not: in a sweep over seeds 0..4, run k's
#      collection simulator (seed k) replayed the exact CUDA random sequence of run k-1's
#      evaluation simulator (seed (k-1) + 1), and run k's replay sampler (seed k + 2) shared its
#      sequence with run k+2's collection simulator. Replicates meant to be independent were
#      coupled, which the Student-t intervals over seeds (experiments/aggregate.py) assume
#      they are not.
#    - numpy's SeedSequence hashes the entropy pair (base, stream) through a mixing function
#      designed for exactly this (spawning independent streams); distinct pairs give
#      effectively unrelated seeds, whatever the numeric relation between bases or ids.
#
# 3. Range:
#    - The 64-bit hash is shifted to 63 bits, which both torch.Generator.manual_seed and
#      numpy.random.default_rng accept.
# ==============================================================================

from enum import IntEnum

import numpy as np


def stream_seed(base: int, stream: int) -> int:
    """Seed of stream `stream` derived from `base` (module header); a non-negative 63-bit int."""
    if base < 0 or stream < 0:
        raise ValueError(f"base and stream must be non-negative, got {base}, {stream}.")
    return int(np.random.SeedSequence([base, stream]).generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))


class RunStream(IntEnum):
    """Stream ids of a TrainingRun (training/run.py)."""

    NETWORK_INIT = 0
    COLLECTION_ENV = 1
    EVAL_ENV = 2
    BUFFER = 3
    TRAIN_PLANNER = 4
    EVAL_PLANNER = 5
    TRAIN_AGENT = 6
    EVAL_AGENT = 7
    WARMUP_AGENT = 8
