#!/bin/bash
#SBATCH --job-name=perfect_feat
#SBATCH --partition=dgx
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=04:00:00
#SBATCH --output=perfect_feat.out
#SBATCH --error=perfect_feat.err
#SBATCH --mail-user=gloriosog@msoe.edu
#SBATCH --mail-type=END,FAIL

# Test 1: Perfect Features (Full Training)
# Only needed if quick test passes and you want to train an adapter

echo "=================================================="
echo "TEST 1: PERFECT FEATURES (FULL TRAINING)"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=================================================="

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
cd "$HOME/CoderOCR/OCR-Coder" || exit 1

"$PYTHON" coder_vl/test_perfect_features.py \
    --train_manifest "Data Crawling/output/manifests/train.jsonl" \
    --val_manifest "Data Crawling/output/manifests/val.jsonl" \
    --batch_size 4 \
    --grad_accum 4 \
    --lr 1e-3 \
    --epochs 1 \
    --eval_steps 50 \
    --log_steps 10 \
    --num_visual_tokens 256 \
    --checkpoint_dir "./checkpoints/perfect_features"

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Done — exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=================================================="

exit $EXIT_CODE
