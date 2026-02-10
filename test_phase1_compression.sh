#!/bin/bash
#SBATCH --job-name=phase1-compression
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo "================================================="
echo "Phase 1: Vision Token Compression Scaling Test"
echo "================================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time:      $(date)"
echo "================================================="
echo ""

cd "$HOME/DS OCR"

# Use the conda env Python (quotes needed — path has spaces)
PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"

echo "Python: $("$PYTHON" --version)"
echo ""

# Run the Phase 1 compression test
"$PYTHON" test_phase1_compression.py

echo ""
echo "Job finished at: $(date)"

