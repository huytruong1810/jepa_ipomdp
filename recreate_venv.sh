#!/usr/bin/env bash
# ABSOLUTE PATH: recreate_venv.sh
# DESIGN DECISIONS: Native WSL Virtual Environment Recreation & Clean-Up Daemon.
# 1. Path Guardrail: Blocks execution in slow /mnt/c/ directories to ensure WSL native performance.
# 2. Complete Purge: Deletes Windows-side compiled artifacts and cache folders.
# 3. High-Performance 'uv' Provisioning: Installs Python 3.12 and creates the clean Linux .venv.
# 4. CUDA 12.8 & RTX 5080 Matching: Directs uv to pull correct GPU-accelerated PyTorch builds.
# 5. CRLF Line Ending Sanitization: Converts Windows Carriage Returns (\r\n) to Linux Line Feeds (\n).
# 6. Deep Import Verification: Tests PyTorch CUDA initialization and imports all critical packages.

set -e # Terminate immediately if any command fails

# Define color outputs for high-visibility terminal logging
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

logger_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

logger_warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

logger_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 1. Path Safety Check
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
logger_info "Active project directory workspace resolved: $PROJECT_DIR"

# Prevent running in the slow/restricted Windows mount path (/mnt/c/...)
if [[ "$PROJECT_DIR" == /mnt/c/* ]]; then
    logger_error "Execution blocked! You are running this inside a Windows mount directory (/mnt/c/)."
    logger_error "WSL symlinks and environment files will break or run extremely slowly under /mnt/."
    logger_error "Please move your project folder to your native WSL home directory (e.g., ~/projects/jepa_ipomdp) and rerun."
    exit 1
fi

# 2. Reclaim Disk Space & Purge Stale Windows/Cache Files
logger_warn "Scanning for non-compatible Windows virtual environments or cached artifacts..."
for dir in ".venv" "venv" "dist" "build" "*.egg-info" ".pytest_cache" ".hydra"; do
    if [ -d "$dir" ] || ls -d $dir &>/dev/null; then
        logger_warn "Removing incompatible or stale path: '$dir'"
        rm -rf $dir
    fi
done

# Clean stale __pycache__ folders recursively
logger_info "Cleaning up old Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} +

# 3. Ensure 'uv' Is Available Natively
if ! command -v uv &> /dev/null; then
    logger_info "'uv' package manager not detected. Running automated bootstrap installation..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Load environmental path properties immediately
    source "$HOME/.local/bin/env"
else
    logger_info "Verified 'uv' package installer is present natively."
fi

# 4. Initialize Standalone Python 3.12 Engine inside WSL
logger_info "Fetching and linking clean standalone Python 3.12 production runtime..."
uv python install 3.12

# 5. Provision Native Linux Virtual Environment (.venv)
logger_info "Creating native Linux virtual environment (.venv)..."
uv venv .venv --python 3.12

# Activate environment natively in this shell step
source .venv/bin/activate
logger_info "Virtual environment successfully activated."

# 6. Install CUDA-Aware PyTorch Wheels for RTX 5080 (cu128 targets)
logger_info "Installing PyTorch compiled with CUDA 12.8 acceleration libraries..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 7. Install Project and Developer Dependencies
logger_info "Installing package and development dependencies from pyproject.toml..."
# Installing in editable mode (-e) to allow hot-reloads during active research
uv pip install -e .

# 8. Sanitize Windows CRLF Line Endings to Linux LF
logger_info "Sanitizing source file carriage returns (converting Windows CRLF to Linux LF)..."
# Locates all Python, configuration, and shell scripts and converts endings natively
find . -type f \( -name "*.py" -o -name "*.sh" -o -name "*.toml" -o -name "*.md" -o -name "*.yaml" -o -name "*.yml" \) -not -path "*/.*" -exec sed -i 's/\r$//' {} +
chmod +x recreate_venv.sh

# 9. Verify CUDA Acceleration and GPU Presence on RTX 5080
logger_info "Running validation check on PyTorch CUDA initialization and core imports..."
python3 -c "
import sys
import torch

print('='*60)
print('NATIVE LINUX COMPILE CHECK:')
print('  Python Version:          ', sys.version.split()[0])
print('  PyTorch Version:         ', torch.__version__)
print('  GPU Device Available:    ', torch.cuda.is_available())

if torch.cuda.is_available():
    print('  Active Graphics Device:  ', torch.cuda.get_device_name(0))
    print('  Device Compute Capability:', torch.cuda.get_device_capability(0))
    print('  CUDA Version Compiled:   ', torch.version.cuda)
else:
    print('  WARNING: CUDA is not active! PyTorch fell back to CPU.')

print('\nCORE IMPORT VERIFICATION:')
try:
    import hydra
    import omegaconf
    import sklearn
    import tensorboard
    import tqdm
    import networkx
    import plotly
    print('  ✅ All pyproject.toml scientific dependencies imported cleanly!')
except ImportError as e:
    print('  ❌ Dependency Import Failure: ', e)
    sys.exit(1)
print('='*60)
"

logger_info "🎉 NATIVE ENVIRONMENT RESTORATION COMPLETE!"
logger_info "To begin working in this environment, run: source .venv/bin/activate"