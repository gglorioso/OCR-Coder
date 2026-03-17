#!/bin/bash
#SBATCH --job-name=infer_1_9c
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-infer19c-%j.out

export PYTHONNOUSERSITE=1
export HF_HOME=$HOME/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
REPO=/home/ad.msoe.edu/gloriosog/CoderOCR/OCR-Coder

cd "$REPO"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Phase 1.9c inference evaluation"
"$PYTHON" MVV/Phase_1_9/c/infer_1_9c.py
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done."
