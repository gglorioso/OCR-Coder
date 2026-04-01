#!/bin/bash
#SBATCH --job-name=inference_33
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-inference-33-%j.out

export PYTHONNOUSERSITE=1
export HF_HOME=$HOME/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
REPO=/home/ad.msoe.edu/gloriosog/CoderOCR/OCR-Coder
MODEL_PATH="$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-Coder-V2-Lite-Instruct/snapshots/e434a23f91ba5b4923cf6c9d9a238eb4a08e3a11"

cd "$REPO"

echo "=== Phase 3.3 inference started at $(date) ==="
"$PYTHON" MVV/Phase_3/Phase_3_3/run_inference.py \
    --model-path "$MODEL_PATH" \
    --ckpt-dir MVV/Phase_3/Phase_3_3/checkpoints/epoch_9 \
    --data-dir MVV/Phase_3/full_data/tensors_and_texts \
    --sample black__action__main_py_chunk0 \
    --max-tokens 512
echo "=== Phase 3.3 inference finished at $(date) ==="
