#!/bin/bash
#SBATCH --job-name=mvv-p12-exp2
#SBATCH --partition=teaching
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=mvv_p12_exp2.out
#SBATCH --error=mvv_p12_exp2.err
#SBATCH --mail-user=gloriosog@msoe.edu
#SBATCH --mail-type=END,FAIL

# MVV Phase 1.2 Exp2 — Spatial Regression (CPU-only, no GPU needed)
#
# High memory required: pool8x8 features are 73,728D × 8,980 files ≈ 2.7 GB/budget.
# PCA (randomized SVD) + Ridge run entirely on CPU.
#
# Step 1: gen_labels.py  — windowed AST labels (only nodes visible in 40-line window)
# Step 2: run_regression.py — PCA(1024) + Ridge per pool size (pool4x4, pool8x8)
#
# Reads features from : MVV/Phase_1_1/exp2_maxpool_comparison/data/features_maxpool/
# Writes labels to    : MVV/Phase_1_2/exp2_spatial_regression/data/labels.jsonl
# Writes results to   : MVV/Phase_1_2/exp2_spatial_regression/results/

set -euo pipefail
export PYTHONNOUSERSITE=1

echo "=================================================="
echo "MVV Phase 1.2 Exp2 — Spatial Regression Probe"
echo "=================================================="
echo "Job ID  : ${SLURM_JOB_ID:-local}"
echo "Node    : ${SLURMD_NODENAME:-$(hostname)}"
echo "CPUs    : ${SLURM_CPUS_PER_TASK:-?}"
echo "Mem     : 64G"
echo "Start   : $(date)"
echo "=================================================="

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
REPO="$HOME/CoderOCR/OCR-Coder"
SCRIPTS="$REPO/MVV/Phase_1_2/exp2_spatial_regression/scripts"

cd "$REPO" || exit 1

# ── Ensure matplotlib is available ───────────────────────────────────────────
"$PYTHON" -c "import matplotlib" 2>/dev/null || {
    echo "Installing matplotlib …"
    "$PYTHON" -m pip install --quiet matplotlib
}

# ── Step 1: Generate windowed labels ─────────────────────────────────────────
echo ""
echo "[Step 1/2] Generating windowed structural labels via AST …"
"$PYTHON" "$SCRIPTS/gen_labels.py"
LABEL_COUNT=$(wc -l < "$REPO/MVV/Phase_1_2/exp2_spatial_regression/data/labels.jsonl" 2>/dev/null || echo "?")
echo "  → ${LABEL_COUNT} rows written to labels.jsonl"

# ── Step 2: Run spatial regression probe ─────────────────────────────────────
echo ""
echo "[Step 2/2] Running PCA + Ridge regression probe …"
"$PYTHON" "$SCRIPTS/run_regression.py" \
    --n_components 1024 \
    --alpha 100.0

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Done — exit code: $EXIT_CODE"
echo "End: $(date)"
echo "=================================================="

exit $EXIT_CODE
