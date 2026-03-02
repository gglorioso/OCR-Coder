#!/bin/bash
#SBATCH --job-name=mvv-p12-regression
#SBATCH --partition=teaching
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=mvv_p12_regression.out
#SBATCH --error=mvv_p12_regression.err
#SBATCH --mail-user=gloriosog@msoe.edu
#SBATCH --mail-type=END,FAIL

# MVV Phase 1.2 — Structural Regression Probe (CPU-only, no GPU needed)
#
# Step 1: gen_labels.py  — AST-parse source files → labels.jsonl
# Step 2: run_regression.py — LinearRegression probe, R² sweep across budgets
#
# Reads features from  : MVV/Phase_1_1/data_mvv/features/budget_*/
# Reads source from    : Scraped Repos/
# Writes labels to     : MVV/Phase_1_2/exp1_structural_regression/data/labels.jsonl
# Writes results to    : MVV/Phase_1_2/exp1_structural_regression/results/

set -euo pipefail
export PYTHONNOUSERSITE=1

echo "=================================================="
echo "MVV Phase 1.2 — Structural Regression Probe"
echo "=================================================="
echo "Job ID  : ${SLURM_JOB_ID:-local}"
echo "Node    : ${SLURMD_NODENAME:-$(hostname)}"
echo "CPUs    : ${SLURM_CPUS_PER_TASK:-?}"
echo "Start   : $(date)"
echo "=================================================="

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
REPO="$HOME/CoderOCR/OCR-Coder"
SCRIPTS="$REPO/MVV/Phase_1_2/exp1_structural_regression/scripts"

cd "$REPO" || exit 1

# ── Ensure matplotlib is available (not in base deepseek-ocr env) ────────────
"$PYTHON" -c "import matplotlib" 2>/dev/null || {
    echo "Installing matplotlib …"
    "$PYTHON" -m pip install --quiet matplotlib
}

# ── Step 1: Generate labels ──────────────────────────────────────────────────
echo ""
echo "[Step 1/2] Generating structural labels via AST …"
"$PYTHON" "$SCRIPTS/gen_labels.py"
LABEL_COUNT=$(wc -l < "$REPO/MVV/Phase_1_2/exp1_structural_regression/data/labels.jsonl" 2>/dev/null || echo "?")
echo "  → ${LABEL_COUNT} rows written to labels.jsonl"

# ── Step 2: Run regression probe ─────────────────────────────────────────────
echo ""
echo "[Step 2/2] Running LinearRegression probe …"
"$PYTHON" "$SCRIPTS/run_regression.py"

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Done — exit code: $EXIT_CODE"
echo "End: $(date)"
echo "=================================================="

exit $EXIT_CODE
