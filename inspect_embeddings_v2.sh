#!/bin/bash
#SBATCH --job-name=inspect_embeddings
#SBATCH --output=slurm-inspect-embeddings-%j.out
#SBATCH --error=slurm-inspect-embeddings-%j.err
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00

echo "=========================================="
echo "Phase 2 Prep: Embedding Dimension Inspector"
echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time:      $(date)"
echo "=========================================="
echo ""

cd "$HOME/DS OCR"

# Use the conda env Python directly (same approach as Phase 1)
PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"

echo "Python: $("$PYTHON" --version)"
echo ""

# Run the inspection script
"$PYTHON" "DS Coder/inspect_embeddings_v2.py"

echo ""
echo "Job finished at: $(date)"
