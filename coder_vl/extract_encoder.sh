#!/bin/bash
#SBATCH --job-name=extract-vision-encoder
#SBATCH --partition=dgx
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=01:00:00
#SBATCH --output=slurm-extract-encoder-%j.out
#SBATCH --error=slurm-extract-encoder-%j.err
#SBATCH --mail-user=gloriosog@msoe.edu
#SBATCH --mail-type=END,FAIL

# Extract Vision Encoder from DeepSeek-OCR-2
# Needs ~13-16 GB VRAM (fp16), so V100 32GB is sufficient
# Expected runtime: ~5-10 minutes

echo "=================================================="
echo "Extracting Vision Encoder from DeepSeek-OCR-2"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs: $SLURM_GPUS"
echo "Start time: $(date)"
echo "=================================================="
echo ""

# GPU info
nvidia-smi

echo ""
echo "=================================================="
echo "Starting extraction..."
echo "=================================================="
echo ""

# Use direct Python path (note: path has spaces, must be quoted)
PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"

# Change to project directory
cd "$HOME/CoderOCR/OCR-Coder" || exit 1

# Run extraction script
"$PYTHON" coder_vl/extract_encoder.py \
    --model_path "deepseek-ai/deepseek-ocr-2" \
    --output_path "./models/vision_encoder.pt" \
    --device "cuda"

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Extraction complete"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=================================================="

# Show file size if successful
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Saved vision encoder:"
    ls -lh ./models/vision_encoder.pt
fi

exit $EXIT_CODE
