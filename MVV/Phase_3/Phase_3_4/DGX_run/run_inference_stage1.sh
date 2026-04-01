#!/bin/bash
#SBATCH --job-name=s1_infer
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-s1-infer-%j.out
#SBATCH --mem=48G

export PYTHONNOUSERSITE=1

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"

echo "=========================================="
echo "Stage 1 Inference -- $(date)"
echo "=========================================="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

"$PYTHON" MVV/Phase_3/Phase_3_4/DGX_run/run_inference_stage1.py \
    --model-path "$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-Coder-V2-Lite-Instruct/snapshots/e434a23f91ba5b4923cf6c9d9a238eb4a08e3a11" \
    --ckpt-dir MVV/Phase_3/Phase_3_4/DGX_run/checkpoints/stage1_4gpu/epoch_step_4000 \
    --data-dir MVV/Phase_3/full_data/tensors_and_texts \
    --num-samples 3 \
    --max-tokens 512

echo ""
echo "Done -- $(date)"
