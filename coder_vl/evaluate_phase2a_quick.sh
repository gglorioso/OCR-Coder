#!/bin/bash
#SBATCH --job-name=phase2a-eval-quick
#SBATCH --partition=dgx
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=00:30:00
#SBATCH --output=2a_eval_quick.out
#SBATCH --error=2a_eval_quick.err
#SBATCH --mail-user=gloriosog@msoe.edu
#SBATCH --mail-type=END,FAIL

# Phase 2a Evaluation: Quick test (15 examples)

echo "=================================================="
echo "Phase 2a: Quick Evaluation (15 examples)"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=================================================="

nvidia-smi

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
cd "$HOME/CoderOCR/OCR-Coder" || exit 1

"$PYTHON" coder_vl/evaluate_phase2a.py \
    --checkpoint ./checkpoints/phase2a/best.pt \
    --features_dir ./precomputed_features \
    --val_manifest "Data Crawling/output/manifests/val.jsonl" \
    --max_new_tokens 256 \
    --max_samples 15

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Done — exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=================================================="

exit $EXIT_CODE
