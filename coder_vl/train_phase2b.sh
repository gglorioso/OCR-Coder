#!/bin/bash
#SBATCH --job-name=phase2b-train
#SBATCH --partition=dgx
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00
#SBATCH --output=slurm-phase2b-%j.out
#SBATCH --error=slurm-phase2b-%j.err
#SBATCH --mail-user=gloriosog@msoe.edu
#SBATCH --mail-type=END,FAIL

# Phase 2b: QLoRA fine-tuning of DeepSeek-Coder-V2-Lite + projection adapter.
#
# Trains both the adapter and the LLM (via 4-bit QLoRA) on 40K image-grounded
# examples from 13 Python repos.  Auto-resumes from the latest step_*.pt
# checkpoint if the job is requeued.
#
# Prerequisites:
#   - ./checkpoints/phase2a_v6/best.pt  (adapter init)
#   - ./precomputed_features_tiled/     (tiled vision features, 9469 images)
#   - data_v2b/manifests/train.jsonl    (40,083 examples)
#   - data_v2b/manifests/val.jsonl      (2,018 examples)
#
# Expected runtime: ~6.5h on 4x V100 (2 epochs, 625 gradient steps)
# Memory per GPU:   ~11.9 GB / 32 GB

echo "=================================================="
echo "Phase 2b: QLoRA Fine-tuning"
echo "=================================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Start:     $(date)"
echo "=================================================="

nvidia-smi

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
TORCHRUN="$HOME/DS OCR/envs/deepseek-ocr/bin/torchrun"

cd "$HOME/CoderOCR/OCR-Coder" || exit 1

# Install peft if not already present
"$PYTHON" -c "import peft" 2>/dev/null || "$PYTHON" -m pip install peft --quiet

"$TORCHRUN" --nproc_per_node=4 coder_vl/train_phase2b.py \
    --features_dir   ./precomputed_features_tiled \
    --train_manifest data_v2b/manifests/train.jsonl \
    --val_manifest   data_v2b/manifests/val.jsonl \
    --batch_size     4 \
    --lr_adapter     1e-5 \
    --lr_lora        2e-5 \
    --epochs         2 \
    --grad_accum     4 \
    --max_seq_length 260 \
    --checkpoint_dir ./checkpoints/phase2b \
    --eval_steps     200 \
    --log_steps      10 \
    --ckpt_interval  1800 \
    --init_from      ./checkpoints/phase2a_v6/best.pt \
    --lora_r         16 \
    --lora_alpha     32 \
    --lora_dropout   0.05

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Done — exit code: $EXIT_CODE"
echo "End: $(date)"
echo "=================================================="

exit $EXIT_CODE

# To chain a continuation job (auto-resume picks up latest step_*.pt):
# sbatch --dependency=afterany:$SLURM_JOB_ID coder_vl/train_phase2b.sh
