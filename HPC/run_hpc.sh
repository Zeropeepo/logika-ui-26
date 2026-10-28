#!/usr/bin/env bash
# ==============================================================================
# Interactive Execution Runner for UB AI Center HPC
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "======================================================================"
echo "    UB AI CENTER HPC - INTERACTIVE PRODUCTION EXECUTION RUNNER       "
echo "======================================================================"

# 1. Environment check
if [ ! -d ".venv_hpc" ]; then
    echo "[*] Setting up isolated user environment..."
    bash setup_env.sh
fi

source .venv_hpc/bin/activate

# 2. Resource audit
python check_hpc_resources.py

# 3. Diagnostic
python hpc_adversarial_validation.py

# 4. Master Pipeline
python hpc_master_pipeline.py "$@"

echo "======================================================================"
echo "[SUCCESS] Submissions ready in: ${SCRIPT_DIR}/submissions/"
echo "======================================================================"
