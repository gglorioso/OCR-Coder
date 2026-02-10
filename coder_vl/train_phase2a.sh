#!/bin/bash
#SBATCH --job-name=phase2a-adapter
#SBATCH --partition=dgxh100
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=24:00:00
#SBATCH --output=slurm-phase2a-%j.out
#SBATCH --error=slurm-phase2a-%j.err
#SBATCH --mail-user=gloriosog@msoe.edu
#SBATCH --mail-type=END,FAIL

# Phase 2a Training - Projection Adapter Alignment
# Trains adapter only (13.6M params), vision encoder + coder model frozen
# Expected runtime: 6-10 hours on 1× H100

echo "=================================================="
echo "Phase 2a: Projection Adapter Training"
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
echo "Starting training..."
echo "=================================================="
echo ""

# Use direct Python path (not conda activate - doesn't work in SLURM)
PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"

# Change to project directory
cd "$HOME/CoderOCR/OCR-Coder" || exit 1

# Run training script
"$PYTHON" coder_vl/train_projector.py \
    --batch_size 8 \
    --learning_rate 1e-3 \
    --checkpoint_dir "./checkpoints/phase2a"

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Training complete"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=================================================="

exit $EXIT_CODE
