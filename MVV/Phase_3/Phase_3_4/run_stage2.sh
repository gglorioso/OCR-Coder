#!/bin/bash
#SBATCH --job-name=stage2_reason
#SBATCH --partition=dgxh100
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-stage2-%j.out

export PYTHONNOUSERSITE=1
export HF_HOME=$HOME/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
REPO=/home/ad.msoe.edu/gloriosog/CoderOCR/OCR-Coder
MODEL_PATH="$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-Coder-V2-Lite-Instruct/snapshots/e434a23f91ba5b4923cf6c9d9a238eb4a08e3a11"

cd "$REPO"

echo "=== Stage 2 reasoning fine-tune started at $(date) ==="
echo "=== GPUs: $(nvidia-smi -L | wc -l) ==="

"$PYTHON" -m torch.distributed.run \
    --nproc_per_node=8 \
    --master_port=29501 \
    MVV/Phase_3/Phase_3_4/train_stage2.py \
    --model-path "$MODEL_PATH" \
    --jsonl-path MVV/Phase_3/Phase_3_4/reasoning_dataset.jsonl \
    --repo-root "$REPO" \
    --stage1-ckpt-dir MVV/Phase_3/checkpoints/stage1_run/epoch_best \
    --save-dir MVV/Phase_3/checkpoints/stage2_run \
    --epochs 3 \
    --batch-size 4

echo "=== Stage 2 reasoning fine-tune finished at $(date) ==="
