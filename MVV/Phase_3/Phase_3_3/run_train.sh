#!/bin/bash
#SBATCH --job-name=phase3_3_train
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-phase3_3-%j.out

export PYTHONNOUSERSITE=1
export HF_HOME=$HOME/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
REPO=/home/ad.msoe.edu/gloriosog/CoderOCR/OCR-Coder
MODEL_PATH="$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-Coder-V2-Lite-Instruct/snapshots/e434a23f91ba5b4923cf6c9d9a238eb4a08e3a11"

cd "$REPO"

echo "=== Phase 3.3 training started at $(date) ==="
"$PYTHON" MVV/Phase_3/Phase_3_3/train_joint.py \
    --model-path "$MODEL_PATH" \
    --data-dir MVV/Phase_3/mini_data \
    --projector-ckpt MVV/Phase_2/checkpoints/best_aligned.pt \
    --save-dir MVV/Phase_3/Phase_3_3/checkpoints \
    --epochs 10 \
    --batch-size 1
echo "=== Phase 3.3 training finished at $(date) ==="
