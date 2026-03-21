#!/bin/bash
#SBATCH --job-name=phase3_mini
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-phase3mini-%j.out

export PYTHONNOUSERSITE=1
export HF_HOME=$HOME/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
REPO=/home/ad.msoe.edu/gloriosog/CoderOCR/OCR-Coder

cd "$REPO"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 1: Preparing 100-sample mini dataset"
"$PYTHON" MVV/Phase_3/train_joint.py --prepare-data
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Data preparation complete."

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 2: Plumbing test (overfit 1 batch)"
"$PYTHON" MVV/Phase_3/train_joint.py --overfit --epochs 20
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Plumbing test complete."

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 3: Loss curve test (100 samples, 5 epochs)"
"$PYTHON" MVV/Phase_3/train_joint.py --epochs 5
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done."
