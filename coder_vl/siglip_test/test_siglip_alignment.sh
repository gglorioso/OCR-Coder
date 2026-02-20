#!/bin/bash
#SBATCH --job-name=siglip-align
#SBATCH --partition=dgx
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=01:30:00
#SBATCH --output=siglip_alignment.out
#SBATCH --error=siglip_alignment.err
#SBATCH --mail-user=gloriosog@msoe.edu
#SBATCH --mail-type=END,FAIL

# SigLIP vs OCR-2 alignment comparison
# Compares perplexity with random adapters (no training)
# Expected runtime: ~30-60 min on 1x V100

echo "=================================================="
echo "SigLIP vs OCR-2 Alignment Test"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=================================================="

nvidia-smi

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
cd "$HOME/CoderOCR/OCR-Coder" || exit 1

"$PYTHON" coder_vl/siglip_test/test_siglip_alignment.py \
    --val_manifest "Data Crawling/output/manifests/val.jsonl" \
    --ocr2_features_dir ./precomputed_features \
    --num_examples 30 \
    --num_seeds 3

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Done — exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=================================================="

exit $EXIT_CODE
