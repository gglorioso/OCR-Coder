#!/bin/bash
#SBATCH --job-name=data-gen-v3
#SBATCH --partition=teaching
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-data-gen-v3-%j.out
#SBATCH --error=slurm-data-gen-v3-%j.err

# Phase v3 Multi-Style Data Generation
#
# Renders 8 colour themes for every code chunk:
#   monokai dracula one-dark github-dark nord default friendly vs
#
# ~8× more images than v2b (~77k images) using 16 parallel workers.
# Estimated wall time: ~2h (vs ~16h single-threaded).
#
# After this job: update precompute script to point at data_v3, then sbatch.

echo "========================================"
echo "Phase v3 Multi-Style Data Generation"
echo "========================================"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_NODELIST"
echo "CPUs:    $SLURM_CPUS_PER_TASK"
echo "Start:   $(date)"
echo ""

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
REPOS_DIR="$HOME/CoderOCR/OCR-Coder/Scraped Repos"
OUTPUT_DIR="$HOME/CoderOCR/OCR-Coder/data_v3"
FEATURES_DIR="$HOME/CoderOCR/OCR-Coder/precomputed_features_tiled"

echo "Repos:    $REPOS_DIR"
echo "Output:   $OUTPUT_DIR"
echo "Workers:  $SLURM_CPUS_PER_TASK"
echo ""

"$PYTHON" "$HOME/CoderOCR/OCR-Coder/Data Crawling/data_gen_2b.py" \
    --repos-dir    "$REPOS_DIR" \
    --output-dir   "$OUTPUT_DIR" \
    --features-dir "$FEATURES_DIR" \
    --chunk-size   500 \
    --seed         42 \
    --n-workers    "$SLURM_CPUS_PER_TASK" \
    --styles monokai dracula one-dark github-dark nord default friendly vs

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Done — exit code: $EXIT_CODE"
echo "End: $(date)"
echo "========================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Images rendered: $(ls "$OUTPUT_DIR/images" 2>/dev/null | wc -l)"
    echo "Train examples:  $(wc -l < "$OUTPUT_DIR/manifests/train.jsonl" 2>/dev/null)"
    echo "Val examples:    $(wc -l < "$OUTPUT_DIR/manifests/val.jsonl" 2>/dev/null)"
    echo "Test examples:   $(wc -l < "$OUTPUT_DIR/manifests/test.jsonl" 2>/dev/null)"
fi

exit $EXIT_CODE
