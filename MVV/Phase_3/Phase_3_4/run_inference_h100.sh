#!/bin/bash
#SBATCH --job-name=infer_h100
#SBATCH --partition=dgxh100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

export PYTHONNOUSERSITE=1

echo "========================================"
echo "Phase 3.4 -- Stage 1 Inference (H100)"
echo "========================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Date:      $(date)"
echo "========================================"

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"

"$PYTHON" MVV/Phase_3/Phase_3_4/run_inference_h100.py \
    --model-path "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct" \
    --ckpt-dir "MVV/Phase_3/checkpoints/stage1_4h100/epoch_best" \
    --data-dir "MVV/Phase_3/full_data/tensors_and_texts" \
    --num-samples 5 \
    --max-tokens 2048

echo "========================================"
echo "Inference complete: $(date)"
echo "========================================"
