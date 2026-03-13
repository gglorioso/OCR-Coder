#!/usr/bin/env bash
#SBATCH --job-name=phase18_contrastive
#SBATCH --partition=teaching
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-phase18-%j.out

set -e

export PYTHONNOUSERSITE=1

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO="/home/ad.msoe.edu/gloriosog/CoderOCR/OCR-Coder"
PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
SCRIPTS="$REPO/MVV/Phase_1_8/scripts"

GROUND_TRUTH="$REPO/MVV/Phase_1_8/data/ground_truth/ground_truth.jsonl"
TEXT_EMB_OUT="$REPO/MVV/Phase_1_8/data/text_embeddings/text_embeddings.pt"
FEAT_DIR="$REPO/MVV/Phase_1_5/data/features/method2/pool8x8"
CKPT_DIR="$REPO/MVV/Phase_1_8/checkpoints"

mkdir -p "$(dirname "$TEXT_EMB_OUT")"
mkdir -p "$CKPT_DIR"

# ---------------------------------------------------------------------------
# Step 1 — Pre-compute text embeddings (skip if already done)
# ---------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 1: Pre-compute text embeddings"

if [ -f "$TEXT_EMB_OUT" ]; then
    echo "  Text embeddings already exist at $TEXT_EMB_OUT — skipping."
else
    echo "  Running precompute_text_embeddings.py …"
    "$PYTHON" "$SCRIPTS/precompute_text_embeddings.py" \
        --ground-truth "$GROUND_TRUTH" \
        --out-path     "$TEXT_EMB_OUT" \
        --device       cuda \
        --batch-size   256
    echo "  Done."
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 1 complete."

# ---------------------------------------------------------------------------
# Step 2 — Train contrastive adapter
# ---------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 2: Train contrastive adapter"

"$PYTHON" "$SCRIPTS/train_1_8.py" \
    --feat-dir      "$FEAT_DIR" \
    --ground-truth  "$GROUND_TRUTH" \
    --text-emb      "$TEXT_EMB_OUT" \
    --out-dir       "$CKPT_DIR" \
    --epochs        30 \
    --batch-size    64 \
    --lr            1e-4 \
    --device        cuda

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 2 complete."
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 1.8 pipeline finished."
