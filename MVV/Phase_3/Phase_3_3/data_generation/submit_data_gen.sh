#!/bin/bash
#SBATCH --job-name=datagen_phase3
#SBATCH --partition=dgx
#SBATCH --array=1-4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-datagen-%A_%a.out

export PYTHONNOUSERSITE=1
export HF_HOME=$HOME/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
REPO=/home/ad.msoe.edu/gloriosog/CoderOCR/OCR-Coder

cd "$REPO"

echo "=== Data generation array task $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT started at $(date) ==="
"$PYTHON" MVV/Phase_3/Phase_3_3/data_generation/generate_dataset.py
echo "=== Data generation array task $SLURM_ARRAY_TASK_ID finished at $(date) ==="
