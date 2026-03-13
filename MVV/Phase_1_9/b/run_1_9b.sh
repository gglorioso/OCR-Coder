#!/bin/bash
#SBATCH --job-name=phase19b
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-phase19b-%j.out

export PYTHONNOUSERSITE=1
export HF_HOME=$HOME/.cache/huggingface

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
REPO=/home/ad.msoe.edu/gloriosog/CoderOCR/OCR-Coder

cd "$REPO"
"$PYTHON" MVV/Phase_1_9/b/infer_1_9b.py
