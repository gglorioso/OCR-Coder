#!/bin/bash
#SBATCH --job-name=s1_cont
#SBATCH --partition=dgx
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-stage1-continue-%j.out
#SBATCH --dependency=afterany:239379

export PYTHONNOUSERSITE=1
export HF_HOME=$HOME/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
REPO=/home/ad.msoe.edu/gloriosog/CoderOCR/OCR-Coder
MODEL_PATH="$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-Coder-V2-Lite-Instruct/snapshots/e434a23f91ba5b4923cf6c9d9a238eb4a08e3a11"

cd "$REPO"

echo "=== Stage 1 continuation (V100 QLoRA) started at $(date) ==="
echo "=== Resuming from epoch_latest checkpoint ==="
echo "=== GPUs: $(nvidia-smi -L | wc -l) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PYTHON" -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=29502 \
    MVV/Phase_3/Phase_3_4/DGX_run/train_stage1.py \
    --model-path "$MODEL_PATH" \
    --data-dir MVV/Phase_3/full_data/tensors_and_texts \
    --save-dir MVV/Phase_3/Phase_3_4/DGX_run/checkpoints/stage1_4gpu_continued \
    --resume-from MVV/Phase_3/Phase_3_4/DGX_run/checkpoints/stage1_4gpu/epoch_latest \
    --epochs 2 \
    --batch-size 2

echo "=== Stage 1 continuation (V100 QLoRA) finished at $(date) ==="
