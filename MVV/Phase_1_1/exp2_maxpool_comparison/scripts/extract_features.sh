#!/bin/bash
#SBATCH --job-name=mvv-maxpool
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=extract_maxpool_features.out
#SBATCH --error=extract_maxpool_features.err
#SBATCH --mail-user=gloriosog@msoe.edu
#SBATCH --mail-type=END,FAIL

# MVV adaptive max-pool feature extraction.
# Single SigLIP forward pass per budget; saves both pool4x4 and pool8x8
# features in one pass.
#
# Output dirs:
#   MVV/Phase_1_1/exp2_maxpool_comparison/data/features_maxpool/pool4x4/budget_{N}/  [18,432d]
#   MVV/Phase_1_1/exp2_maxpool_comparison/data/features_maxpool/pool8x8/budget_{N}/  [73,728d]

echo "=================================================="
echo "MVV Adaptive Max-Pool Feature Extraction"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "GPU:    $CUDA_VISIBLE_DEVICES"
echo "Start:  $(date)"
echo "=================================================="

export PYTHONNOUSERSITE=1

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
cd "$HOME/CoderOCR/OCR-Coder" || exit 1

"$PYTHON" MVV/Phase_1_1/exp2_maxpool_comparison/scripts/extract_features.py \
    --data-dir   MVV/Phase_1_1/data_mvv \
    --model      google/siglip-so400m-patch14-384 \
    --batch-size 32 \
    --device     cuda

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Done — exit code: $EXIT_CODE"
echo "End:  $(date)"
for ps in 4 8; do
    for budget in 729 441 256 121; do
        count=$(find MVV/Phase_1_1/exp2_maxpool_comparison/data/features_maxpool/pool${ps}x${ps}/budget_${budget}/ \
                     -name "*.pt" 2>/dev/null | wc -l)
        echo "  pool${ps}x${ps}/budget_${budget}: ${count} vectors"
    done
done
echo "=================================================="

exit $EXIT_CODE
