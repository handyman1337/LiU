#!/usr/bin/env bash
# =============================================================================
# TDDE78 Deep Reinforcement Learning — Environment Setup
# Linköping University, Spring 2026
#
# This script sets up the development environment using uv.
# No sudo required.
#
# Usage:
#   bash setup.sh
# =============================================================================

set -euo pipefail

# ---------------------
# Configuration
# ---------------------
PYTHON_VERSION="3.10"
VENV_DIR=".venv"

# ---------------------
# Colors for output
# ---------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[INFO]${NC}  $1"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ---------------------
# Header
# ---------------------
echo ""
echo "=============================================="
echo "  TDDE78 Deep Reinforcement Learning"
echo "  Environment Setup"
echo "=============================================="
echo ""

# ---------------------
# Step 1: Check / Install uv
# ---------------------
info "Checking for uv..."

if ! command -v uv &> /dev/null; then
    warn "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add uv to PATH for this session
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

    if ! command -v uv &> /dev/null; then
        error "Failed to install uv. Please install it manually:"
        echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        echo "  Then restart your terminal and run this script again."
        exit 1
    fi
fi

ok "uv found: $(uv --version)"
echo ""

# ---------------------
# Step 2: Create virtual environment
# ---------------------
info "Creating Python ${PYTHON_VERSION} virtual environment in ${VENV_DIR}/..."

if [ -d "${VENV_DIR}" ]; then
    warn "Virtual environment already exists. Removing and recreating..."
    rm -rf "${VENV_DIR}"
fi

uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}"
ok "Virtual environment created"
echo ""

# ---------------------
# Step 3: Activate virtual environment
# ---------------------
info "Activating virtual environment..."
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
ok "Virtual environment activated ($(python --version))"
echo ""

# ---------------------
# Step 4: Install dependencies
# ---------------------
info "Installing dependencies from requirements.txt..."
echo ""

uv pip install --python "${VENV_DIR}/bin/python" -r requirements.txt

echo ""
ok "All dependencies installed"
echo ""

# ---------------------
# Step 5: Register Jupyter kernel
# ---------------------
info "Registering Jupyter kernel for this environment..."
python -m ipykernel install --user --name=tdde78 --display-name="TDDE78 Deep RL (.venv)"
ok "Jupyter kernel 'TDDE78 Deep RL (.venv)' registered"
echo ""

# ---------------------
# Step 6: Create project directory structure
# ---------------------
info "Creating project directory structure..."


# Lab directories
for lab in lab1_value_based lab2_policy_gradient lab3_actor_critic lab4_model_based lab5_multi_agent; do
    mkdir -p "labs/${lab}/starter_code"
    mkdir -p "labs/${lab}/experiments"
done

ok "Directory structure created"
echo ""

# ---------------------
# Step 7: Verify installation
# ---------------------
info "Verifying installation..."
echo ""

python << 'VERIFY'
import sys

checks_passed = 0
checks_total = 0

def check(name, import_func):
    global checks_passed, checks_total
    checks_total += 1
    try:
        result = import_func()
        print(f"  [OK]    {name}: {result}")
        checks_passed += 1
    except Exception as e:
        print(f"  [FAIL]  {name}: {e}")

# Core libraries
check("Python",      lambda: sys.version.split()[0])
check("PyTorch",     lambda: __import__('torch').__version__)
check("NumPy",       lambda: __import__('numpy').__version__)
check("Gymnasium",   lambda: __import__('gymnasium').__version__)
check("PettingZoo",  lambda: __import__('pettingzoo').__version__)
check("Matplotlib",  lambda: __import__('matplotlib').__version__)
check("Pandas",      lambda: __import__('pandas').__version__)
check("TensorBoard", lambda: __import__('tensorboard').__version__)
check("tqdm",        lambda: __import__('tqdm').__version__)

# Test Gymnasium environments
print()
check("Gymnasium CartPole-v1", lambda: (
    __import__('gymnasium').make('CartPole-v1').reset() and "initialized"
) or "initialized")

check("Gymnasium LunarLander-v3", lambda: (
    __import__('gymnasium').make('LunarLander-v3').reset() and "initialized"
) or "initialized")

# Test Atari (requires explicit registration in Gymnasium >= 1.0.0)
try:
    import ale_py
    import gymnasium as gym
    gym.register_envs(ale_py)
    env = gym.make('ALE/Pong-v5')
    env.reset()
    env.close()
    checks_total += 1
    checks_passed += 1
    print(f"  [OK]    Atari (Pong-v5): initialized")
except Exception as e:
    checks_total += 1
    print(f"  [WARN]  Atari (Pong-v5): {e}")
    print(f"          Atari ROMs may need manual setup for some games.")

# Test PettingZoo MPE
try:
    from pettingzoo.mpe import simple_spread_v3
    env = simple_spread_v3.env()
    env.reset()
    env.close()
    checks_total += 1
    checks_passed += 1
    print(f"  [OK]    PettingZoo simple_spread_v3: initialized")
except Exception as e:
    checks_total += 1
    print(f"  [WARN]  PettingZoo simple_spread_v3: {e}")

# Test PyTorch CUDA
import torch
if torch.cuda.is_available():
    print(f"\n  [OK]    CUDA: available ({torch.cuda.get_device_name(0)})")
else:
    print(f"\n  [INFO]  CUDA: not available (CPU training only)")

print(f"\n  Results: {checks_passed}/{checks_total} checks passed")

if checks_passed < checks_total:
    print("  Some optional components may need additional setup.")

VERIFY

echo ""

# ---------------------
# Done
# ---------------------
echo "=============================================="
echo -e "  ${GREEN}Setup Complete!${NC}"
echo "=============================================="
echo ""
echo "  To activate the environment:"
echo "    source ${VENV_DIR}/bin/activate"
echo ""
echo "  To deactivate:"
echo "    deactivate"
echo ""
echo "  To start working on Lab 1:"
echo "    cd labs/lab1_value_based/"
echo ""
echo "  To launch Jupyter:"
echo "    jupyter lab"
echo ""
