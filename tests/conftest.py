# ABSOLUTE PATH: tests/conftest.py
# ==============================================================================
# TEST-SESSION SETTINGS
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. One CPU Thread:
#    - The CPU tests run thousands of tiny tensor operations (a 16-wide filter, a 4-root
#      search). With torch's default of one intra-op thread per core, each of them pays a
#      thread-pool synchronisation, which on a loaded machine dominated the run time: one
#      Wumpus training collection took 16 s with 24 threads and 0.4 s with one (2026-10-01).
#      Results do not depend on the thread count beyond float summation order, which no test
#      asserts on; GPU tests are unaffected.
# ==============================================================================

import torch

torch.set_num_threads(1)
