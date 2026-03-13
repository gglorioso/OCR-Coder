#!/bin/bash
#SBATCH --job-name=phase15_extract
#SBATCH --partition=teaching
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=03:00:00
#SBATCH --output=slurm-phase15-extract-%j.out

export PYTHONNOUSERSITE=1

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
SCRIPT="$HOME/CoderOCR/OCR-Coder/MVV/Phase_1_5/scripts/extract_features_1_5.py"
DATA_DIR="$HOME/CoderOCR/OCR-Coder/MVV/Phase_1_1/data_mvv"

echo "==== Phase 1.5 — Feature Extraction (Methods 2 & 3) ===="
echo "Node: $(hostname)  GPU: $SLURM_JOB_GPUS"
echo "Start: $(date)"
echo ""

"$PYTHON" "$SCRIPT" \
    --data-dir "$DATA_DIR" \
    --device cuda \
    --batch-size 8

echo ""
echo "Done: $(date)"
