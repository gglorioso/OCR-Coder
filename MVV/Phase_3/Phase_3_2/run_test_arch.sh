#!/bin/bash
#SBATCH --job-name=test_arch_3_2
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-test-arch-%j.out

export PYTHONNOUSERSITE=1
export HF_HOME=$HOME/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
REPO=/home/ad.msoe.edu/gloriosog/CoderOCR/OCR-Coder

cd "$REPO"

echo "=== test_arch_3_2 started at $(date) ==="
"$PYTHON" MVV/Phase_3_2/scripts/test_arch.py
echo "=== test_arch_3_2 finished at $(date) ==="
