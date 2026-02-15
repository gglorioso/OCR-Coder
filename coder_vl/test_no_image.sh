#!/bin/bash
#SBATCH --job-name=test-no-image
#SBATCH --partition=dgx
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=00:10:00
#SBATCH --output=test_no_image.out
#SBATCH --error=test_no_image.err

echo "Testing generation without image features"
echo "Start time: $(date)"

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
cd "$HOME/CoderOCR/OCR-Coder" || exit 1

"$PYTHON" coder_vl/test_no_image.py

echo "Done: $(date)"
