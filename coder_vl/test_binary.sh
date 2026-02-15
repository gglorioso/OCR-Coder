#!/bin/bash
#SBATCH --job-name=test-binary
#SBATCH --partition=dgx
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=00:15:00
#SBATCH --output=test_binary.out
#SBATCH --error=test_binary.err

# Binary classification test — verify model can use visual features for simple tasks

echo "=================================================="
echo "Binary Classification Test"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=================================================="

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
cd "$HOME/CoderOCR/OCR-Coder" || exit 1

"$PYTHON" coder_vl/test_binary_classification.py

echo ""
echo "=================================================="
echo "Done"
echo "End time: $(date)"
echo "=================================================="
