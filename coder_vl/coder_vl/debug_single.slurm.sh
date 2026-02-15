#!/bin/bash
#SBATCH --job-name=debug-single
#SBATCH --partition=dgx
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=00:15:00
#SBATCH --output=debug_single.out
#SBATCH --error=debug_single.err

# Debug a single example to understand model behavior

echo "=================================================="
echo "Debugging single example"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=================================================="

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
cd "$HOME/CoderOCR/OCR-Coder" || exit 1

"$PYTHON" coder_vl/debug_single_example.py

echo ""
echo "=================================================="
echo "Done"
echo "End time: $(date)"
echo "=================================================="
