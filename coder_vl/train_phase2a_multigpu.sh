#!/bin/bash
#SBATCH --job-name=phase2a-multigpu
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=slurm-phase2a-multigpu-%j.out
#SBATCH --error=slurm-phase2a-multigpu-%j.err
#SBATCH --mail-user=gloriosog@msoe.edu
#SBATCH --mail-type=END,FAIL

# Phase 2a Training - Projection Adapter Alignment (Multi-GPU)
# Trains adapter only (13.6M params), vision encoder + coder model frozen
# Expected runtime: ~3-5 hours on 4× V100 (vs 12-18 hours on 1× V100)

echo "=================================================="
echo "Phase 2a: Projection Adapter Training (Multi-GPU)"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs: $SLURM_GPUS"
echo "Tasks: $SLURM_NTASKS"
echo "Start time: $(date)"
echo "=================================================="
echo ""

# GPU info
nvidia-smi

echo ""
echo "=================================================="
echo "Setting up distributed training environment..."
echo "=================================================="
echo ""

# Set distributed training environment variables
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"

echo ""
echo "=================================================="
echo "Starting training..."
echo "=================================================="
echo ""

# Use direct Python path (not conda activate - doesn't work in SLURM)
PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"

# Change to project directory
cd "$HOME/CoderOCR/OCR-Coder" || exit 1

# Run training script with srun for multi-GPU
srun "$PYTHON" coder_vl/train_projector.py \
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
