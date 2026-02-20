#!/bin/bash
#SBATCH --job-name=data-gen-v2
#SBATCH --partition=teaching
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-data-gen-v2-%j.out
#SBATCH --error=slurm-data-gen-v2-%j.err

# Phase 2a Data Generation v2 — Dracula theme, full repo sweep
#
# Renders ALL usable Python files from Scraped Repos in the dracula theme.
# Output goes to ./data_v2/ (separate from original monokai data).
# Precomputed features from v2 save to the SAME ./precomputed_features/ dir
# because filenames include the theme suffix (_dracula.pt vs _monokai.pt).
#
# After this job completes, run:
#   sbatch coder_vl/precompute_features_v2.sh

echo "========================================"
echo "Phase 2a Data Generation v2 (dracula)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "CPUs:   $SLURM_CPUS_PER_TASK"
echo "Start:  $(date)"
echo ""

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
REPOS_DIR="$HOME/CoderOCR/OCR-Coder/Scraped Repos"
OUTPUT_DIR="$HOME/CoderOCR/OCR-Coder/data_v2"

NUM_REPOS=$(ls -d "$REPOS_DIR"/*/ 2>/dev/null | wc -l)
echo "Repos: $NUM_REPOS in $REPOS_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Large target — sample_by_size will naturally cap at however many
# unique files exist across all buckets (likely 5000-8000 after filters)
"$PYTHON" "$HOME/CoderOCR/OCR-Coder/Data Crawling/simple_data_gen.py" \
    --repos-dir  "$REPOS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --target     100000 \
    --style      dracula \
    --font-size  13 \
    --seed       123

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Done — exit code: $EXIT_CODE"
echo "End: $(date)"
echo "========================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Images:    $(ls "$OUTPUT_DIR/images" 2>/dev/null | wc -l) rendered"
    echo "Train:     $(wc -l < "$OUTPUT_DIR/manifests/train.jsonl" 2>/dev/null) examples"
    echo "Val:       $(wc -l < "$OUTPUT_DIR/manifests/val.jsonl" 2>/dev/null) examples"
    echo ""
    echo "Next: sbatch coder_vl/precompute_features_v2.sh"
fi

exit $EXIT_CODE
