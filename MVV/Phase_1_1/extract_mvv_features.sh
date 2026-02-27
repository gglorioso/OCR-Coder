#!/bin/bash
#SBATCH --job-name=mvv-features
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=extract_mvv_features.out
#SBATCH --error=extract_mvv_features.err
#SBATCH --mail-user=gloriosog@msoe.edu
#SBATCH --mail-type=END,FAIL

# MVV feature extraction sweep — GPU required for SigLIP inference.
# Passes all 8,980 images through frozen SigLIP-SO400M at 4 token budgets.
#
# Outputs (one .pt file per image per budget):
#   MVV/Phase_1_1/data_mvv/features/budget_729/
#   MVV/Phase_1_1/data_mvv/features/budget_441/
#   MVV/Phase_1_1/data_mvv/features/budget_256/
#   MVV/Phase_1_1/data_mvv/features/budget_121/

echo "=================================================="
echo "MVV Feature Extraction Sweep"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "GPU:    $CUDA_VISIBLE_DEVICES"
echo "Start:  $(date)"
echo "=================================================="

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
cd "$HOME/CoderOCR/OCR-Coder" || exit 1

"$PYTHON" MVV/Phase_1_1/extract_mvv_features.py \
    --data-dir   MVV/Phase_1_1/data_mvv \
    --model      google/siglip-so400m-patch14-384 \
    --batch-size 32 \
    --device     cuda

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Done — exit code: $EXIT_CODE"
echo "End:  $(date)"
for budget in 729 441 256 121; do
    count=$(find MVV/Phase_1_1/data_mvv/features/budget_${budget}/ -name "*.pt" 2>/dev/null | wc -l)
    echo "  budget_${budget}: ${count} vectors"
done
echo "=================================================="

exit $EXIT_CODE
