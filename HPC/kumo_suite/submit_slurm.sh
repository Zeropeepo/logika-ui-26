#!/usr/bin/env bash
#SBATCH --job-name=kumo-screen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=kumo-%j.log
set -euo pipefail
# Submit from repository root. Add your cluster's partition/account directives.
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}"
source .venv-kumo/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export POLARS_MAX_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
python -m HPC.kumo_suite run --preset screen --query-batch 512 "$@"
python -m HPC.kumo_suite summarize
