#!/usr/bin/env bash
#SBATCH --job-name=phase19_keyword_probe
#SBATCH --partition=teaching
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-phase19-%j.out

set -e

export PYTHONNOUSERSITE=1

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO="/home/ad.msoe.edu/gloriosog/CoderOCR/OCR-Coder"
PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
SCRIPTS="$REPO/MVV/Phase_1_9/scripts"

FEAT_DIR="$REPO/MVV/Phase_1_9/data/features"
LABELS_DIR="$REPO/MVV/Phase_1_9/data/labels"
GT_PATH="$REPO/MVV/Phase_1_9/data/ground_truth.jsonl"
CKPT_DIR="$REPO/MVV/Phase_1_9/checkpoints"

mkdir -p "$FEAT_DIR" "$LABELS_DIR" "$CKPT_DIR"

# ---------------------------------------------------------------------------
# Step 1 — Extract raw [1024, 1152] SigLIP features (idempotent)
# ---------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 1: Extract raw SigLIP features"

N_FEAT=$(find "$FEAT_DIR" -name "*.pt" 2>/dev/null | wc -l)
if [ "$N_FEAT" -gt 0 ]; then
    echo "  Found $N_FEAT .pt files in $FEAT_DIR — skipping extraction."
else
    echo "  Running extract_features_1_9.py …"
    "$PYTHON" "$SCRIPTS/extract_features_1_9.py" \
        --batch-size 8 \
        --device cuda
    echo "  Extraction done."
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 1 complete."

# ---------------------------------------------------------------------------
# Step 2 — Generate keyword labels (idempotent)
# ---------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 2: Generate keyword labels"

N_LABELS=$(find "$LABELS_DIR" -name "*.pt" 2>/dev/null | wc -l)
if [ "$N_LABELS" -gt 0 ]; then
    echo "  Found $N_LABELS label files — skipping label generation."
else
    echo "  Running label_generator_1_9.py …"
    "$PYTHON" "$SCRIPTS/label_generator_1_9.py"
    echo "  Label generation done."
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 2 complete."

# ---------------------------------------------------------------------------
# Step 3 — Train ConvRoPE keyword probe
# ---------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 3: Train ConvRoPE keyword probe"

"$PYTHON" "$SCRIPTS/train_1_9.py" \
    --feat-dir     "$FEAT_DIR"  \
    --labels-dir   "$LABELS_DIR" \
    --ground-truth "$GT_PATH"   \
    --out-dir      "$CKPT_DIR"  \
    --epochs       20           \
    --batch-size   32           \
    --lr           1e-3         \
    --device       cuda

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 3 complete."
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 1.9 pipeline finished."
