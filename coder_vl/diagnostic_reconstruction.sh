#!/bin/bash
#SBATCH --job-name=diagnostic-recon
#SBATCH --partition=dgx
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=02:00:00
#SBATCH --output=diagnostic_recon.out
#SBATCH --error=diagnostic_recon.err
#SBATCH --mail-user=gloriosog@msoe.edu
#SBATCH --mail-type=END,FAIL

# Diagnostic Test: Visual Feature Reconstruction Quality
# Tests if trained adapter can reconstruct code from visual features
# Decision thresholds:
#   BLEU ≥0.3: Info preserved → use stronger adapter
#   BLEU <0.1: Info lost → fine-tune encoder

echo "========================================================"
echo "DIAGNOSTIC: CODE RECONSTRUCTION TEST"
echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "========================================================"

nvidia-smi

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
cd "$HOME/CoderOCR/OCR-Coder" || exit 1

"$PYTHON" coder_vl/diagnostic_reconstruction.py \
    --checkpoint ./checkpoints/phase2a/best.pt \
    --features_dir ./precomputed_features \
    --val_manifest "Data Crawling/output/manifests/val.jsonl" \
    --max_new_tokens 512 \
    --max_samples 30 \
    --output coder_vl/diagnostic_results.json

EXIT_CODE=$?

echo ""
echo "========================================================"
echo "DONE — exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "========================================================"

exit $EXIT_CODE
