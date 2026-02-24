#!/bin/bash
#SBATCH --job-name=img-sensitivity
#SBATCH --partition=dgx
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=02:00:00
#SBATCH --output=sensitivity.out
#SBATCH --error=sensitivity.err
#SBATCH --mail-user=gloriosog@msoe.edu
#SBATCH --mail-type=END,FAIL

# Test 1: Image Sensitivity Check
# Runs 50 val examples twice — correct image vs swapped image.
# High output similarity = model ignoring visual tokens (language prior dominates).
# Low similarity = model IS reading the image.

echo "=================================================="
echo "Test 1: Image Sensitivity Check"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "=================================================="

nvidia-smi

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
cd "$HOME/CoderOCR/OCR-Coder" || exit 1

"$PYTHON" -c "import peft" 2>/dev/null || "$PYTHON" -m pip install peft --quiet

"$PYTHON" coder_vl/test_image_sensitivity.py \
    --checkpoint        ./checkpoints/phase2b/best.pt \
    --features_dir      ./precomputed_features_tiled \
    --val_manifest      data_v2b/manifests/val.jsonl \
    --max_samples       50 \
    --max_new_tokens    80 \
    --save_file         ./sensitivity_results.json

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Done — exit code: $EXIT_CODE"
echo "End: $(date)"
echo "=================================================="

exit $EXIT_CODE
