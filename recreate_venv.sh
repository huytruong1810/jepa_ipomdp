#!/usr/bin/env bash
# ABSOLUTE PATH: recreate_venv.sh
# ==============================================================================
# REBUILD THE VIRTUAL ENVIRONMENT FROM uv.lock
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. The Lock File Is the Environment:
#    - `uv sync` installs exactly uv.lock (torch from the CUDA 12.8 index configured in
#      pyproject.toml, the project in editable mode, and the dev group with pytest). Nothing is
#      pip-installed on the side, so a rebuilt .venv is identical to the one the tests ran in.
#
# 2. Native WSL Paths Only:
#    - Under /mnt/c the Windows filesystem makes the venv and every import extremely slow, so the
#      script refuses to run there.
#
# 3. Verify CUDA Before Anything Is Trained:
#    - Training and the slow tests expect the RTX 5080 (sm_120), which needs the cu128 wheels.
#      The script prints the device and fails if CUDA is unavailable.
# ==============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ "$PROJECT_DIR" == /mnt/* ]]; then
    echo "Refusing to run under /mnt: move the project into the native WSL filesystem (e.g. ~/projects)." >&2
    exit 1
fi
if ! command -v uv &> /dev/null; then
    echo "uv is not installed: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

rm -rf .venv .pytest_cache src/*.egg-info
find . -type d -name "__pycache__" -not -path "./runs/*" -exec rm -rf {} +

uv python install 3.12
uv sync

uv run python - <<'EOF'
import sys

import torch

print(f"Python {sys.version.split()[0]}, torch {torch.__version__}, CUDA {torch.version.cuda}")
if not torch.cuda.is_available():
    sys.exit("CUDA is not available: check the NVIDIA driver and the cu128 torch wheel.")
print(f"GPU: {torch.cuda.get_device_name(0)} (compute capability {torch.cuda.get_device_capability(0)})")
EOF
