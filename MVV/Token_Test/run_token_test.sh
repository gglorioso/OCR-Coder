#!/bin/bash
#SBATCH --job-name=token_test
#SBATCH --partition=teaching
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-token-test-%j.out

# ---------------------------------------------------------------------------
# DeepSeek tokenizer baseline test
# Tokenizes a random sample of 500 source files from the Phase_1_1 manifest
# and saves summary statistics + ASCII histogram to
#   MVV/Token_Test/results/token_stats.json
# ---------------------------------------------------------------------------

# Prevent user-site packages (e.g. stray transformers installs) from
# shadowing the project environment.
export PYTHONNOUSERSITE=1

REPO_ROOT="/home/ad.msoe.edu/gloriosog/CoderOCR/OCR-Coder"
PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"

cd "$REPO_ROOT" || { echo "ERROR: could not cd to $REPO_ROOT"; exit 1; }

echo "=============================="
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $(hostname)"
echo "Date   : $(date)"
echo "Python : $PYTHON"
echo "=============================="

"$PYTHON" MVV/Token_Test/token_count.py

echo ""
echo "Done. Exit code: $?"
