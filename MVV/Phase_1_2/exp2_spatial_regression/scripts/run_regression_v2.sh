#!/bin/bash
#SBATCH --job-name=phase12_exp2_v2
#SBATCH --partition=teaching
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-phase12-exp2-v2-%j.out

export PYTHONNOUSERSITE=1

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
SCRIPT_DIR="$HOME/CoderOCR/OCR-Coder/MVV/Phase_1_2/exp2_spatial_regression/scripts"

echo "==== Phase 1.2 Exp2 — pool4x4 + pool8x8 native CV ===="
echo "Node: $(hostname)  CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo ""

"$PYTHON" "$SCRIPT_DIR/run_regression_v2.py"

echo ""
echo "==== Plotting ===="
"$PYTHON" "$SCRIPT_DIR/plot_results_v2.py"

echo ""
echo "Done: $(date)"
