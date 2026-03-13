#!/bin/bash
#SBATCH --job-name=phase15_probe
#SBATCH --partition=teaching
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-phase15-probe-%j.out

export PYTHONNOUSERSITE=1

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
SCRIPT="$HOME/CoderOCR/OCR-Coder/MVV/Phase_1_5/scripts/run_probe_1_5.py"

echo "==== Phase 1.5 — Ridge Regression Probe ===="
echo "Node: $(hostname)  CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo ""

"$PYTHON" "$SCRIPT"

echo ""
echo "Done: $(date)"
