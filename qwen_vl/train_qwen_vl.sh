#!/bin/bash
#SBATCH --job-name=qwen-vl-v1
#SBATCH --partition=dgx
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-qwen-vl-%j.out
#SBATCH --error=slurm-qwen-vl-%j.err

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
TORCHRUN="$HOME/DS OCR/envs/deepseek-ocr/bin/torchrun"
REPO="$HOME/CoderOCR/OCR-Coder"
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

cd "$REPO"

# Force NCCL to use P2P/sockets — IB (ibverbs/mlx4) not available on dgx nodes
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Reduce CUDA memory fragmentation (helps fit 7B model on 32GB V100)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ------------------------------------------------------------------
# Install required packages (transformers>=4.49 for Qwen2.5-VL,
# qwen-vl-utils for image processing helpers).
# --installs to ~/.local, takes priority over conda env.
# Only runs the first time; subsequent jobs skip download.
# ------------------------------------------------------------------
echo "Checking dependencies..."
"$PYTHON" -c "from transformers import Qwen2_5_VLForConditionalGeneration" 2>/dev/null || {
    echo "Upgrading transformers to >=4.49 for Qwen2.5-VL support..."
    "$PYTHON" -m pip install --user --quiet "transformers>=4.49.0" "qwen-vl-utils>=0.0.8"
    echo "Dependencies installed."
}

# ------------------------------------------------------------------
# Verify model is downloaded (must pre-download on login node)
# ------------------------------------------------------------------
"$PYTHON" -c "
from huggingface_hub import try_to_load_from_cache
f = try_to_load_from_cache('$MODEL', 'config.json')
if f is None:
    print('ERROR: Model not cached. Run on login node first:')
    print('  python -c \"from huggingface_hub import snapshot_download; snapshot_download(\\\"$MODEL\\\")\"')
    exit(1)
print('Model cache: OK')
"
if [ $? -ne 0 ]; then
    exit 1
fi

echo "Starting training..."
"$TORCHRUN" --nproc_per_node=2 qwen_vl/train_qwen_vl.py \
    --model_name      "$MODEL" \
    --train_manifest  data_v2b/manifests/train.jsonl \
    --val_manifest    data_v2b/manifests/val.jsonl \
    --checkpoint_dir  ./checkpoints/qwen_vl_v1 \
    --batch_size      1 \
    --grad_accum      8 \
    --epochs          3 \
    --lr              2e-4 \
    --lora_r          16 \
    --lora_alpha      32 \
    --lora_dropout    0.05 \
    --max_seq_len     4096 \
    --eval_steps      200 \
    --log_steps       10 \
    --ckpt_interval   1800
