#!/bin/bash
#SBATCH --job-name=precompute-2b
#SBATCH --partition=dgx
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=slurm-precompute-2b-%j.out
#SBATCH --error=slurm-precompute-2b-%j.err
#SBATCH --mail-user=gloriosog@msoe.edu
#SBATCH --mail-type=END,FAIL

# Phase 2b: Pre-compute tiled vision features for new images.
#
# Reads manifests from data_v2b/manifests/, encodes each image with the
# DeepSeek-OCR-2 vision encoder (2x2 tiling + thumbnail = 5×256 = 1280 tokens
# per image, 1280D features), and saves .pt files to precomputed_features_tiled/.
#
# Skips images whose .pt already exists (safe to rerun after partial completion).
#
# Prerequisites:
#   - data_v2b/manifests/ populated by data_gen_2b.sh
#   - models/vision_encoder.pt NOT required (script loads from HuggingFace cache)
#
# Expected runtime: ~1-3 hours per 5K new images on 1x V100.

echo "=================================================="
echo "Phase 2b: Pre-computing tiled vision features"
echo "=================================================="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Start:   $(date)"
echo "=================================================="
echo ""

nvidia-smi

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
cd "$HOME/CoderOCR/OCR-Coder" || exit 1

MANIFEST_DIR="$HOME/CoderOCR/OCR-Coder/data_v2b/manifests"
OUTPUT_DIR="$HOME/CoderOCR/OCR-Coder/precomputed_features_tiled"

echo "Manifest dir: $MANIFEST_DIR"
echo "Output dir:   $OUTPUT_DIR"
echo ""

NEW_IMAGES=$(cat "$MANIFEST_DIR"/*.jsonl 2>/dev/null | python3 -c "
import sys, json
imgs = set()
for line in sys.stdin:
    d = json.loads(line)
    if d.get('image'):
        imgs.add(d['image'])
print(len(imgs))
")
echo "Unique images in manifests: $NEW_IMAGES"
echo ""

"$PYTHON" coder_vl/precompute_features.py \
    --output_dir   "$OUTPUT_DIR" \
    --manifest_dir "$MANIFEST_DIR" \
    --image_size   768 \
    --tiling \
    --skip_existing

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Done — exit code: $EXIT_CODE"
echo "End: $(date)"
echo "=================================================="

if [ $EXIT_CODE -eq 0 ]; then
    TOTAL_PT=$(ls "$OUTPUT_DIR"/*.pt 2>/dev/null | wc -l)
    MONOKAI_PT=$(ls "$OUTPUT_DIR"/*_monokai.pt 2>/dev/null | wc -l)
    echo ""
    echo "Total .pt files in features dir: $TOTAL_PT"
    echo "Monokai .pt files: $MONOKAI_PT"
    echo ""
    echo "Next: sbatch coder_vl/train_phase2b.sh"
fi

exit $EXIT_CODE
