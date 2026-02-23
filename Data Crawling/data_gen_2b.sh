#!/bin/bash
#SBATCH --job-name=data-gen-2b
#SBATCH --partition=teaching
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-data-gen-2b-%j.out
#SBATCH --error=slurm-data-gen-2b-%j.err

# Phase 2b Data Generation — CPU-only image rendering + manifest generation
#
# Processes ALL valid Python files from Scraped Repos:
#   - Chunks files >500 lines into 500-line segments
#   - Renders new monokai PNGs to data_v2b/images/
#   - Generates 6 AST label types per chunk
#   - Repo-level 90/5/5 split → manifests in data_v2b/manifests/
#
# After this job: sbatch coder_vl/precompute_2b.sh

echo "========================================"
echo "Phase 2b Data Generation"
echo "========================================"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_NODELIST"
echo "CPUs:    $SLURM_CPUS_PER_TASK"
echo "Start:   $(date)"
echo ""

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
REPOS_DIR="$HOME/CoderOCR/OCR-Coder/Scraped Repos"
OUTPUT_DIR="$HOME/CoderOCR/OCR-Coder/data_v2b"
FEATURES_DIR="$HOME/CoderOCR/OCR-Coder/precomputed_features_tiled"

echo "Repos:    $REPOS_DIR"
echo "Output:   $OUTPUT_DIR"
echo "Features: $FEATURES_DIR"
echo ""

"$PYTHON" "$HOME/CoderOCR/OCR-Coder/Data Crawling/data_gen_2b.py" \
    --repos-dir   "$REPOS_DIR" \
    --output-dir  "$OUTPUT_DIR" \
    --features-dir "$FEATURES_DIR" \
    --chunk-size  500 \
    --seed        42

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
    echo ""
    echo "Next: sbatch coder_vl/precompute_2b.sh"
fi

exit $EXIT_CODE
