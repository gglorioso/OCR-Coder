#!/usr/bin/env bash
#SBATCH --job-name=eval_retrieval
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-eval-retrieval-%j.out

# ---------------------------------------------------------------------------
# Phase 1.10 — ColBERT-style retrieval evaluation
# ---------------------------------------------------------------------------

set -euo pipefail

# Isolate from any user-installed packages (Qwen / transformers-5.x etc.)
export PYTHONNOUSERSITE=1

# HuggingFace cache — keep models off the home quota
export HF_HOME=$HOME/.cache/huggingface

# Avoid CUDA OOM from fragmented allocations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------------------------
# Activate the project Python environment
# ---------------------------------------------------------------------------
PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
REPO=/home/ad.msoe.edu/gloriosog/CoderOCR/OCR-Coder

cd "$REPO"

echo "=== Phase 1.10 Retrieval Eval ==="
echo "  Date     : $(date)"
echo "  Host     : $(hostname)"
echo "  GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "  Python   : $PYTHON"
echo ""

"$PYTHON" MVV/Phase_1_10/eval_retrieval.py

echo ""
echo "=== Done: $(date) ==="
