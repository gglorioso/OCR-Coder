#!/bin/bash
#SBATCH --job-name=phase2a-train
#SBATCH --partition=dgx
#SBATCH --gpus=8
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00
#SBATCH --output=slurm-phase2a-%j.out
#SBATCH --error=slurm-phase2a-%j.err
#SBATCH --mail-user=gloriosog@msoe.edu
#SBATCH --mail-type=END,FAIL

# Phase 2a v6: Train projection adapter using tiled pre-computed vision features.
# Prereq: run precompute_tiled.sh first to generate ./precomputed_features_tiled/
# Expected: ~6-8 hours on 8x V100 (32 GB) for 2 epochs over 37K examples.
#
# Changes vs v5 run (job 223917):
#   - gpus 4 -> 8           (full DGX node; halves wall time)
#   - features_dir          ./precomputed_features -> ./precomputed_features_tiled
#                           (2x2 tiling + thumbnail: 1280 tokens vs 256;
#                            fixes root cause of G4/G5 failure — 88:1 compression
#                            lost fine-grained identifiers; now ~20:1)
#   - max_seq_length 2048 -> 260  (answers are ≤201 words; 2048 was pure padding
#                                  waste; 260 covers 100% of training examples)
#   - checkpoint_dir phase2a_v5 -> phase2a_v6

echo "=================================================="
echo "Phase 2a: Adapter Training (pre-computed features)"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=================================================="

nvidia-smi

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
TORCHRUN="$HOME/DS OCR/envs/deepseek-ocr/bin/torchrun"
cd "$HOME/CoderOCR/OCR-Coder" || exit 1

# Combine original + data_v2 manifests
COMBINED_TRAIN=$(mktemp /tmp/phase2a_train_XXXX.jsonl)
COMBINED_VAL=$(mktemp /tmp/phase2a_val_XXXX.jsonl)
cat "Data Crawling/output/manifests/train.jsonl" "data_v2/manifests/train.jsonl" > "$COMBINED_TRAIN"
cat "Data Crawling/output/manifests/val.jsonl"   "data_v2/manifests/val.jsonl"   > "$COMBINED_VAL"
echo "Combined train: $(wc -l < "$COMBINED_TRAIN") examples"
echo "Combined val:   $(wc -l < "$COMBINED_VAL") examples"

"$TORCHRUN" --nproc_per_node=8 coder_vl/train_projector.py \
    --features_dir   ./precomputed_features_tiled \
    --train_manifest "$COMBINED_TRAIN" \
    --val_manifest   "$COMBINED_VAL" \
    --batch_size 4 \
    --lr 1e-5 \
    --epochs 2 \
    --grad_accum 4 \
    --max_seq_length 260 \
    --checkpoint_dir ./checkpoints/phase2a_v6 \
    --eval_steps 200 \
    --log_steps 10 \
    --init_from ./checkpoints/contrastive_v4/best.pt

EXIT_CODE=$?
rm -f "$COMBINED_TRAIN" "$COMBINED_VAL"

echo ""
echo "=================================================="
echo "Done — exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=================================================="

exit $EXIT_CODE
