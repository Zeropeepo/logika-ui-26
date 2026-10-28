#!/usr/bin/env bash
# ==============================================================================
# AI Center UB HPC - Isolated User-Space Environment Setup
# Target Node: NVIDIA A100-SXM4 80GB | Driver 580.126.20 | CUDA 13.0 Capable
# NO ROOT / NO SYSTEM DRIVER MODIFICATION REQUIRED
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "======================================================================"
echo "          SETTING UP ISOLATED HPC ENVIRONMENT FOR A100 80GB           "
echo "======================================================================"

# 1. Install or locate uv in user space if not present
if command -v uv &> /dev/null; then
    UV_BIN="$(command -v uv)"
elif [ -f "${HOME}/.cargo/bin/uv" ]; then
    UV_BIN="${HOME}/.cargo/bin/uv"
elif [ -f "${HOME}/.local/bin/uv" ]; then
    UV_BIN="${HOME}/.local/bin/uv"
else
    echo "[*] Installing standalone 'uv' into user space (~/.local/bin)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
    UV_BIN="${HOME}/.local/bin/uv"
fi

echo "[*] Using uv: ${UV_BIN} ($(${UV_BIN} --version))"

# 2. Create isolated virtual environment
VENV_DIR="${SCRIPT_DIR}/.venv_hpc"
if [ ! -d "${VENV_DIR}" ]; then
    echo "[*] Creating isolated virtualenv: ${VENV_DIR}..."
    ${UV_BIN} venv "${VENV_DIR}" --python 3.11 || ${UV_BIN} venv "${VENV_DIR}"
else
    echo "[*] Existing virtualenv found at ${VENV_DIR}"
fi

export VIRTUAL_ENV="${VENV_DIR}"
export PATH="${VENV_DIR}/bin:${PATH}"
PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"

echo "[*] Python executable: ${PYTHON} ($(${PYTHON} --version))"

# 3. Install PyTorch with CUDA 12.4 (Fully compatible with NVIDIA 580.x driver & A100 Ampere SM80)
echo "[*] Installing PyTorch with CUDA 12.4..."
${UV_BIN} pip install --upgrade \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch torchvision torchaudio

# 4. Install ML & Data Science Stack
echo "[*] Installing competitive ML & Tabular Foundation Stack..."
${UV_BIN} pip install --upgrade \
    "structured-data-models>=0.2.1" \
    "lightgbm>=4.3.0" \
    "xgboost>=2.0.0" \
    "catboost>=1.2.5" \
    "polars>=0.20.0" \
    "pyarrow>=14.0.0" \
    "scikit-learn>=1.4.0" \
    "scipy>=1.12.0" \
    "optuna>=3.6.0" \
    "psutil>=5.9.0" \
    "tqdm>=4.66.0"

# 5. Run Pre-flight Verification
echo "======================================================================"
echo "          RUNNING PRE-FLIGHT COMPUTE & GPU VERIFICATION               "
echo "======================================================================"
${PYTHON} check_hpc_resources.py

echo ""
echo "======================================================================"
echo " [SUCCESS] HPC Environment Ready!"
echo " Activate with: source ${VENV_DIR}/bin/activate"
echo " Run Master Pipeline with: python hpc_master_pipeline.py"
echo "======================================================================"
