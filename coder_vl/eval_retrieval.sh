#!/bin/bash
#SBATCH --job-name=eval_retrieval
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-eval-retrieval-%j.out
#SBATCH --error=slurm-eval-retrieval-%j.err

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
REPO="$HOME/CoderOCR/OCR-Coder"

cd "$REPO"

$PYTHON coder_vl/eval_retrieval.py \
    --checkpoint  ./checkpoints/phase2b_v2/best.pt \
    --val_manifest data_v2b/manifests/val.jsonl \
    --features_dir ./precomputed_features_tiled \
    --output       retrieval_results_phase2b_v2.json
