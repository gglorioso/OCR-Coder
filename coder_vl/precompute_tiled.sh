#!/bin/bash
#SBATCH --job-name=precompute-tiled
#SBATCH --partition=dgx
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=4:00:00
#SBATCH --output=slurm-precompute-tiled-%j.out
#SBATCH --error=slurm-precompute-tiled-%j.err
#SBATCH --mail-user=gloriosog@msoe.edu
#SBATCH --mail-type=END,FAIL

# Pre-compute tiled vision features for Phase 2a v6.
#
# Uses 2x2 tiling + full thumbnail (5 views per image, 1280 tokens vs 256).
# Saves to ./precomputed_features_tiled/ to preserve existing base features.
#
# Processes both v1 (monokai) and v2 (dracula) images in sequence.
# Expected runtime: ~2-3 hours on 1x V100.
#
# After this completes, run:
#   sbatch coder_vl/train_phase2a.sh

echo "=================================================="
echo "Pre-computing Tiled Vision Features (v6)"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Start:  $(date)"
echo "=================================================="

nvidia-smi

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
cd "$HOME/CoderOCR/OCR-Coder" || exit 1

# --- v1 images (monokai) ---
echo ""
echo "--- Processing v1 manifests (monokai) ---"
"$PYTHON" coder_vl/precompute_features.py \
    --manifest_dir "Data Crawling/output/manifests" \
    --output_dir   ./precomputed_features_tiled \
    --image_size   768 \
    --tiling

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: v1 precompute failed (exit $EXIT_CODE)"
    exit $EXIT_CODE
fi

# --- v2 images (dracula) ---
echo ""
echo "--- Processing v2 manifests (dracula) ---"
"$PYTHON" coder_vl/precompute_features.py \
    --manifest_dir ./data_v2/manifests \
    --output_dir   ./precomputed_features_tiled \
    --image_size   768 \
    --tiling

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Done — exit code: $EXIT_CODE"
echo "End: $(date)"
echo "=================================================="

if [ $EXIT_CODE -eq 0 ]; then
    COUNT=$(ls ./precomputed_features_tiled/*.pt 2>/dev/null | wc -l)
    echo "Tiled features saved: $COUNT .pt files in ./precomputed_features_tiled/"
    echo ""
    echo "Next: sbatch coder_vl/train_phase2a.sh"
fi

exit $EXIT_CODE
