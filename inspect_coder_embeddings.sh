#!/bin/bash
#SBATCH --job-name=inspect_coder
#SBATCH --output=slurm-inspect-coder-%j.out
#SBATCH --error=slurm-inspect-coder-%j.err
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00

echo "=========================================="
echo "Inspect DeepSeek-Coder-V2-Lite Embeddings"
echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Time:      $(date)"
echo "=========================================="
echo ""

cd "$HOME/DS OCR"

# Use the conda env Python directly (same as Phase 1)
PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"

echo "Python: $("$PYTHON" --version)"
echo ""

# Run the inspection script
"$PYTHON" "DS Coder/inspect_coder_embeddings.py"

echo ""
echo "Job finished at: $(date)"
