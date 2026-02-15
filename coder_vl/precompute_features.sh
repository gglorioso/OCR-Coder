#!/bin/bash
#SBATCH --job-name=precompute-features
#SBATCH --partition=dgx
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=2:00:00
#SBATCH --output=slurm-precompute-%j.out
#SBATCH --error=slurm-precompute-%j.err
#SBATCH --mail-user=gloriosog@msoe.edu
#SBATCH --mail-type=END,FAIL

# Pre-compute vision features for all training images.
# One-shot job: runs vision encoder over ~2175 images, saves [num_tokens, 1280] tensors.
# Expected runtime: ~10-30 minutes on 1x V100.

echo "=================================================="
echo "Pre-computing Vision Features"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=================================================="

nvidia-smi

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
cd "$HOME/CoderOCR/OCR-Coder" || exit 1

"$PYTHON" coder_vl/precompute_features.py \
    --output_dir ./precomputed_features \
    --image_size 768

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Done — exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=================================================="

exit $EXIT_CODE
