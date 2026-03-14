#!/bin/bash
#SBATCH --job-name=phase2_align
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-phase2-%j.out

export PYTHONNOUSERSITE=1
export HF_HOME=$HOME/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
REPO=/home/ad.msoe.edu/gloriosog/CoderOCR/OCR-Coder

cd "$REPO"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Phase 2 alignment training"
"$PYTHON" MVV/Phase_2/train_alignment.py
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done."
