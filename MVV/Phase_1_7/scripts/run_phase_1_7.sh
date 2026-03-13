#!/bin/bash
#SBATCH --job-name=phase17_visual_enhancements
#SBATCH --partition=teaching
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-phase17-%j.out

set -e

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
REPO="$HOME/CoderOCR/OCR-Coder"
SCRIPTS="$REPO/MVV/Phase_1_7/scripts"
REPOS_DIR="$REPO/Scraped Repos"
MANIFEST="$REPO/MVV/Phase_1_1/data_mvv/manifest.jsonl"
RESULTS_DIR="$REPO/MVV/Phase_1_7/results"

echo "==== MVV Phase 1.7 — Visual Enhancements Study ===="
echo "Node:  $(hostname)"
echo "Start: $(date)"
echo ""

# ===========================================================================
# Exp A: Syntax Highlighting Only
# ===========================================================================
IMG_DIR_A="$REPO/MVV/Phase_1_7/images/exp_A_syntax_only"
FEAT_DIR_A="$REPO/MVV/Phase_1_7/data/features/exp_A/pool8x8"
mkdir -p "$IMG_DIR_A" "$FEAT_DIR_A" "$RESULTS_DIR"

echo "=== Exp A: Rendering (syntax highlighting only) ==="
"$PYTHON" "$SCRIPTS/render_enhanced.py" \
    --repos-dir "$REPOS_DIR" \
    --output-dir "$IMG_DIR_A" \
    --syntax-highlighting

echo "=== Exp A: Extracting features ==="
"$PYTHON" "$SCRIPTS/extract_features_1_7.py" \
    --image-dir "$IMG_DIR_A" \
    --out-dir "$FEAT_DIR_A" \
    --device cuda \
    --batch-size 8

echo "=== Exp A: Running probe ==="
"$PYTHON" "$SCRIPTS/run_attention_probe_rope_1_7.py" \
    --feat-dir "$FEAT_DIR_A" \
    --out-path "$RESULTS_DIR/exp_A_results.json" \
    --exp-name "phase_1_7_exp_A_syntax_only"

echo ""
echo "Exp A complete: $(date)"
echo ""

# ===========================================================================
# Exp B: Syntax Highlighting + Line Numbers
# ===========================================================================
IMG_DIR_B="$REPO/MVV/Phase_1_7/images/exp_B_syntax_linenum"
FEAT_DIR_B="$REPO/MVV/Phase_1_7/data/features/exp_B/pool8x8"
mkdir -p "$IMG_DIR_B" "$FEAT_DIR_B"

echo "=== Exp B: Rendering (syntax highlighting + line numbers) ==="
"$PYTHON" "$SCRIPTS/render_enhanced.py" \
    --repos-dir "$REPOS_DIR" \
    --output-dir "$IMG_DIR_B" \
    --syntax-highlighting \
    --line-numbers

echo "=== Exp B: Extracting features ==="
"$PYTHON" "$SCRIPTS/extract_features_1_7.py" \
    --image-dir "$IMG_DIR_B" \
    --out-dir "$FEAT_DIR_B" \
    --device cuda \
    --batch-size 8

echo "=== Exp B: Running probe ==="
"$PYTHON" "$SCRIPTS/run_attention_probe_rope_1_7.py" \
    --feat-dir "$FEAT_DIR_B" \
    --out-path "$RESULTS_DIR/exp_B_results.json" \
    --exp-name "phase_1_7_exp_B_syntax_linenum"

echo ""
echo "Exp B complete: $(date)"
echo ""

# ===========================================================================
# Exp C: Syntax Highlighting + Line Numbers + Indent Guides
# ===========================================================================
IMG_DIR_C="$REPO/MVV/Phase_1_7/images/exp_C_syntax_linenum_guides"
FEAT_DIR_C="$REPO/MVV/Phase_1_7/data/features/exp_C/pool8x8"
mkdir -p "$IMG_DIR_C" "$FEAT_DIR_C"

echo "=== Exp C: Rendering (syntax highlighting + line numbers + indent guides) ==="
"$PYTHON" "$SCRIPTS/render_enhanced.py" \
    --repos-dir "$REPOS_DIR" \
    --output-dir "$IMG_DIR_C" \
    --syntax-highlighting \
    --line-numbers \
    --indent-guides

echo "=== Exp C: Extracting features ==="
"$PYTHON" "$SCRIPTS/extract_features_1_7.py" \
    --image-dir "$IMG_DIR_C" \
    --out-dir "$FEAT_DIR_C" \
    --device cuda \
    --batch-size 8

echo "=== Exp C: Running probe ==="
"$PYTHON" "$SCRIPTS/run_attention_probe_rope_1_7.py" \
    --feat-dir "$FEAT_DIR_C" \
    --out-path "$RESULTS_DIR/exp_C_results.json" \
    --exp-name "phase_1_7_exp_C_syntax_linenum_guides"

echo ""
echo "Exp C complete: $(date)"
echo ""

echo "==== All Phase 1.7 experiments complete ===="
echo "Done: $(date)"
