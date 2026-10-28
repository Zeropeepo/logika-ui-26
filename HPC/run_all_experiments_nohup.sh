#!/usr/bin/env bash
# ==============================================================================
# Master Unattended Experiment Suite Runner (nohup safe)
# Target: AI Center UB HPC - NVIDIA A100-SXM4 80GB (Shared Node)
# ==============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOG_DIR="${SCRIPT_DIR}/logs"
PRED_DIR="${SCRIPT_DIR}/preds"
SUB_DIR="${SCRIPT_DIR}/submissions"
mkdir -p "${LOG_DIR}" "${PRED_DIR}" "${SUB_DIR}"

MASTER_LOG="${LOG_DIR}/master_nohup_$(date +%Y%m%d_%H%M%S).log"
SUMMARY_CSV="${SCRIPT_DIR}/hpc_experiments_summary.csv"

# Redirect stdout and stderr to master log and console
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "======================================================================"
echo "      AI CENTER UB HPC - MASTER EXPERIMENT ORCHESTRATOR               "
echo "      Start Time: $(date)"
echo "      Node      : $(hostname)"
echo "      PID       : $$"
echo "      Master Log: ${MASTER_LOG}"
echo "======================================================================"

# 1. Environment Activation
if [ ! -d ".venv_hpc" ]; then
    echo "[*] Setting up isolated user virtual environment..."
    bash setup_env.sh
fi

source .venv_hpc/bin/activate
echo "[*] Using Python: $(which python)"

# 2. Pre-Flight Compute Check
echo ""
echo "======================================================================"
echo "RUNNING PRE-FLIGHT RESOURCE AUDIT..."
echo "======================================================================"
python check_hpc_resources.py

# 3. Experiment Sequence Matrix
EXPERIMENTS=(
    "experiments/exp01_kumo_foundation_suite.py:KumoTabular_Capacity_Suite"
    "experiments/exp02_smart_pseudolabel_rounds.py:Smart_Pseudolabel_Distillation"
    "experiments/exp03_population_expert_dart.py:Decoupled_Population_Experts_DART"
    "experiments/exp04_catboost_gpu_suite.py:CatBoost_GPU_A100_Suite"
    "experiments/exp05_tab_resnet_rankgauss.py:Deep_Tab_ResNet_RankGauss"
    "experiments/exp06_adversarial_reweighted.py:Adversarial_Sample_Reweighted"
    "experiments/exp07_ensemble_optimizer.py:Master_Bayesian_Ensemble_Optimizer"
)

TOTAL_EXPS=${#EXPERIMENTS[@]}
echo ""
echo "======================================================================"
echo "QUEUED EXPERIMENTS: ${TOTAL_EXPS} EXPERIMENTAL RUNS"
echo "======================================================================"
for i in "${!EXPERIMENTS[@]}"; do
    IFS=":" read -r script desc <<< "${EXPERIMENTS[$i]}"
    echo "  [$((i + 1))/${TOTAL_EXPS}] ${desc} (${script})"
done
echo "======================================================================"
echo ""

START_TIME=$(date +%s)
SUCCESSFUL=0
FAILED=0

for i in "${!EXPERIMENTS[@]}"; do
    IFS=":" read -r script desc <<< "${EXPERIMENTS[$i]}"
    EXP_NUM=$((i + 1))
    EXP_NAME="$(basename "${script}" .py)"
    EXP_LOG="${LOG_DIR}/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
    
    echo ""
    echo "======================================================================"
    echo ">>> LAUNCHING [${EXP_NUM}/${TOTAL_EXPS}]: ${desc}"
    echo "    Script : ${script}"
    echo "    Log    : ${EXP_LOG}"
    echo "    Time   : $(date)"
    echo "======================================================================"
    
    t0=$(date +%s)
    set +e
    python -u "${script}" 2>&1 | tee "${EXP_LOG}"
    EXIT_CODE=$?
    set -e
    t1=$(date +%s)
    DUR=$((t1 - t0))
    
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "[+] [${EXP_NUM}/${TOTAL_EXPS}] SUCCESS: ${desc} (Completed in ${DUR}s)"
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        echo "[!] [${EXP_NUM}/${TOTAL_EXPS}] FAILED (Exit Code: ${EXIT_CODE}): ${desc}"
        echo "[!] Check log for details: ${EXP_LOG}"
        FAILED=$((FAILED + 1))
    fi
    
    # Clean up GPU memory before next experiment
    python -c "import gc, torch; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None" || true
    sleep 3
done

TOTAL_DUR=$(( $(date +%s) - START_TIME ))

echo ""
echo "======================================================================"
echo "      ALL EXPERIMENTAL RUNS COMPLETED                                "
echo "      End Time      : $(date)"
echo "      Total Duration: ${TOTAL_DUR}s ($((TOTAL_DUR / 60)) min)"
echo "      Summary       : ${SUCCESSFUL} Successful | ${FAILED} Failed"
echo "======================================================================"

if [ -f "${SUMMARY_CSV}" ]; then
    echo ""
    echo "======================================================================"
    echo "HPC EXPERIMENTS SUMMARY TABLE (${SUMMARY_CSV}):"
    echo "======================================================================"
    column -s, -t < "${SUMMARY_CSV}" || cat "${SUMMARY_CSV}"
fi

echo ""
echo "======================================================================"
echo "FINAL GENERATED SUBMISSIONS (READY IN submissions/):"
echo "======================================================================"
ls -lh submissions/
echo "======================================================================"
