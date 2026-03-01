#!/bin/bash
#SBATCH --job-name=gen-mvv-images
#SBATCH --partition=teaching
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=gen_mvv_images.out
#SBATCH --error=gen_mvv_images.err
#SBATCH --mail-user=gloriosog@msoe.edu
#SBATCH --mail-type=END,FAIL

# MVV monochrome image generation (CPU-only, no GPU needed).
# Renders every valid Python file from Scraped Repos as one 800x800 grayscale
# PNG per file (AST-anchored, 40 lines, 1 file = 1 class = 1 image).
#
# Outputs:
#   MVV/Phase_1_1/data_mvv/images/      — PNG files (~8-9K expected)
#   MVV/Phase_1_1/data_mvv/manifest.jsonl

echo "=================================================="
echo "MVV Monochrome Image Generation"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "CPUs:   $SLURM_CPUS_PER_TASK"
echo "Start:  $(date)"
echo "=================================================="

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
cd "$HOME/CoderOCR/OCR-Coder" || exit 1

"$PYTHON" "MVV/Phase_1_1/gen_mvv_images.py" \
    --repos-dir "Scraped Repos" \
    --output-dir "MVV/Phase_1_1/data_mvv"

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Done — exit code: $EXIT_CODE"
echo "End:  $(date)"
echo "Image count: $(find MVV/Phase_1_1/data_mvv/images/ -name '*.png' 2>/dev/null | wc -l)"
echo "=================================================="

exit $EXIT_CODE
