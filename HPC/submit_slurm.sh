#!/bin/bash
#SBATCH --job-name=credit_risk_hpc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=hpc_run_%j.log
#SBATCH --error=hpc_run_%j.err

# ==============================================================================
# SLURM Execution Script for UB AI Center HPC (NVIDIA A100-SXM4 80GB)
# ==============================================================================

set -euo pipefail

echo "======================================================================"
echo "Job ID: ${SLURM_JOB_ID:-LOCAL}"
echo "Node  : $(hostname)"
echo "Date  : $(date)"
echo "======================================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 1. Setup / activate environment
if [ ! -d ".venv_hpc" ]; then
    echo "[*] Setting up isolated user environment..."
    bash setup_env.sh
fi

source .venv_hpc/bin/activate

# 2. Pre-flight check
python check_hpc_resources.py

# 3. Diagnostic check (adversarial validation)
python hpc_adversarial_validation.py

# 4. Run Master Pipeline
python hpc_master_pipeline.py

echo "======================================================================"
echo "Job Completed Successfully: $(date)"
echo "Submissions located at: ${SCRIPT_DIR}/submissions/"
echo "======================================================================"
