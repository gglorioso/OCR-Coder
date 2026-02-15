#!/bin/bash
#SBATCH --job-name=perfect_quick
#SBATCH --partition=dgx
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=00:30:00
#SBATCH --output=perfect_quick.out
#SBATCH --error=perfect_quick.err
#SBATCH --mail-user=gloriosog@msoe.edu
#SBATCH --mail-type=END,FAIL

# Test 1: Perfect Features (Quick Inference Test)
# Tests if token insertion works with perfect features (no training needed)

echo "=================================================="
echo "TEST 1: PERFECT FEATURES (QUICK TEST)"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=================================================="

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
cd "$HOME/CoderOCR/OCR-Coder" || exit 1

"$PYTHON" coder_vl/test_perfect_features_quick.py \
    --test_manifest "Data Crawling/output/manifests/test.jsonl" \
    --num_examples 5 \
    --max_new_tokens 256

echo ""
echo "=================================================="
echo "Done"
echo "End time: $(date)"
echo "=================================================="
