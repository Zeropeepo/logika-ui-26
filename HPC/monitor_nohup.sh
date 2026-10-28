#!/usr/bin/env bash
# ==============================================================================
# Real-Time Monitoring Script for Running HPC Experiments
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "======================================================================"
echo "AI CENTER UB HPC - EXPERIMENTS MONITOR"
echo "Date: $(date)"
echo "======================================================================"

# 1. Check Running Python Processes
echo ""
echo "--- Active Python / Runner Processes ---"
ps aux | grep -E "python.*exp|run_all_experiments" | grep -v grep || echo "No active experiment processes found."

# 2. Check GPU Status
echo ""
echo "--- GPU VRAM & Load ---"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader
else
    echo "nvidia-smi not available."
fi

# 3. Check Summary Table
echo ""
echo "--- Experiments Completed So Far (hpc_experiments_summary.csv) ---"
if [ -f "hpc_experiments_summary.csv" ]; then
    column -s, -t < hpc_experiments_summary.csv || cat hpc_experiments_summary.csv
else
    echo "No experiments completed yet."
fi

# 4. Check Latest Master Log Output
LATEST_LOG=$(ls -t logs/master_nohup_*.log 2>/dev/null | head -n 1 || true)
if [ -n "${LATEST_LOG}" ] && [ -f "${LATEST_LOG}" ]; then
    echo ""
    echo "--- Last 25 Lines of Master Log (${LATEST_LOG}) ---"
    tail -n 25 "${LATEST_LOG}"
fi
echo "======================================================================"
