#!/usr/bin/env bash
#SBATCH --job-name=render_compare
#SBATCH --partition=dgx
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-render-compare-%j.out

set -euo pipefail

export PYTHONNOUSERSITE=1

REPO=/home/ad.msoe.edu/gloriosog/CoderOCR/OCR-Coder
PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"

cd "$REPO"

echo "=== Phase 1.11a: Line-count comparison render ==="
echo "Node   : $(hostname)"
echo "Date   : $(date)"
echo "Python : $PYTHON"
echo ""

"$PYTHON" MVV/Phase_1_11/a/render_comparison.py

echo ""
echo "=== Done ==="
