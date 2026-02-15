#!/bin/bash
#SBATCH --job-name=eval_perfect
#SBATCH --partition=dgx
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=00:30:00
#SBATCH --output=eval_perfect.out
#SBATCH --error=eval_perfect.err

# Evaluate trained perfect features model

echo "=================================================="
echo "EVALUATE PERFECT FEATURES MODEL"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=================================================="

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
cd "$HOME/CoderOCR/OCR-Coder" || exit 1

"$PYTHON" coder_vl/eval_perfect_features.py \
    --checkpoint ./checkpoints/perfect_features/best.pt \
    --test_manifest "Data Crawling/output/manifests/test.jsonl" \
    --num_examples 10

echo ""
echo "=================================================="
echo "Done"
echo "End time: $(date)"
echo "=================================================="
