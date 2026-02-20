#!/bin/bash
#SBATCH --job-name=contrastive-pretrain
#SBATCH --partition=dgx
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm-contrastive-%j.out
#SBATCH --error=slurm-contrastive-%j.err
#SBATCH --mail-user=gloriosog@msoe.edu
#SBATCH --mail-type=END,FAIL

# Stage 1: Contrastive pre-training of projection adapter (v3: SigLIP loss).
#
# Prereqs:
#   - precompute_features.sh must have run (./precomputed_features/*.pt exist)
#   - Coder model weights cached in HuggingFace cache
#
# What this does:
#   Trains the 13.6M-param adapter using SigLIP loss (sigmoid per-pair) so that
#   proj(visual_feats.mean()) ≈ coder.embed(code_text).mean()
#   Forces adapter outputs into the "code" region of coder's embedding space.
#
#   v3 changes vs v2 (InfoNCE):
#     - SigLIP loss: sigmoid per-pair instead of softmax, stable at small batch sizes
#     - Learnable temperature: starts at 1.0, adapts during training
#     - SigLIP random-init baseline ≈ 0.693 (vs InfoNCE ≈ 4.16); target < 0.3
#
# After this job completes, run Stage 2:
#   sbatch coder_vl/train_phase2a.sh
#
# Expected runtime: 2-5 hours on V100 (steps are fast — no transformer forward)
# VRAM: ~10-11 GB (fits V100 32 GB with headroom)

echo "=================================================="
echo "Stage 1: Contrastive Pre-training (v3: SigLIP)"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Start:  $(date)"
echo "=================================================="

nvidia-smi

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
cd "$HOME/CoderOCR/OCR-Coder" || exit 1

"$PYTHON" coder_vl/contrastive_pretrain.py \
    --features_dir    ./precomputed_features \
    --train_manifest  "Data Crawling/output/manifests/train.jsonl" \
                      "./data_v2/manifests/train.jsonl" \
    --val_manifest    "Data Crawling/output/manifests/val.jsonl" \
                      "./data_v2/manifests/val.jsonl" \
    --coder_model     deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
    --batch_size      64 \
    --lr              1e-4 \
    --epochs          150 \
    --temperature     1.0 \
    --max_text_tokens 256 \
    --checkpoint_dir  ./checkpoints/contrastive_v4 \
    --log_steps       10

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Done — exit code: $EXIT_CODE"
echo "End: $(date)"
echo "=================================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Contrastive pre-training (v4: SigLIP + bias) succeeded."
    echo "Best checkpoint: ./checkpoints/contrastive_v4/best.pt"
    echo ""
    echo "Next: submit Stage 2 generation training:"
    echo "  sbatch coder_vl/train_phase2a.sh"
    echo "  (train_phase2a.sh uses --init_from ./checkpoints/contrastive_v4/best.pt)"
fi

exit $EXIT_CODE
