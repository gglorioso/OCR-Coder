#!/bin/bash
#SBATCH --job-name=phase2b_v2
#SBATCH --partition=dgx
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-phase2b-v2-%j.out
#SBATCH --error=slurm-phase2b-v2-%j.err

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
TORCHRUN="$HOME/DS OCR/envs/deepseek-ocr/bin/torchrun"
REPO="$HOME/CoderOCR/OCR-Coder"

cd "$REPO"

"$TORCHRUN" --nproc_per_node=2 coder_vl/train_phase2b.py \
    --features_dir    ./precomputed_features_tiled \
    --train_manifest  data_v2b/manifests/train.jsonl \
    --val_manifest    data_v2b/manifests/val.jsonl \
    --checkpoint_dir  ./checkpoints/phase2b_v4 \
    --init_from       ./checkpoints/phase2a_v6/best.pt \
    --batch_size      8 \
    --grad_accum      4 \
    --epochs          2 \
    --lr_adapter      1e-5 \
    --lr_lora         2e-5 \
    --contrast_weight 0.3 \
    --eval_steps      200 \
    --log_steps       10 \
    --ckpt_interval   1800
