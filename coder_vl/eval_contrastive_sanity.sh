#!/bin/bash
#SBATCH --job-name=eval-contrastive-sanity
#SBATCH --partition=dgx
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=00:30:00
#SBATCH --output=eval_contrastive_sanity.out
#SBATCH --error=eval_contrastive_sanity.err
#SBATCH --mail-user=gloriosog@msoe.edu
#SBATCH --mail-type=END,FAIL

# Sanity check: run eval on the PURE contrastive_v4 checkpoint (before LM fine-tuning).
# If outputs here are also Chinese/garbage -> contrastive alignment wasn't enough.
# If outputs here are coherent -> LM fine-tuning undid the alignment (catastrophic forgetting).

echo "=================================================="
echo "Sanity Check: Eval on contrastive_v4 checkpoint (pre-LM)"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=================================================="

nvidia-smi

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
cd "$HOME/CoderOCR/OCR-Coder" || exit 1

COMBINED_VAL=$(mktemp /tmp/phase2a_eval_val_XXXX.jsonl)
cat "Data Crawling/output/manifests/val.jsonl" "data_v2/manifests/val.jsonl" > "$COMBINED_VAL"
echo "Combined val: $(wc -l < "$COMBINED_VAL") examples (using first 15)"

"$PYTHON" coder_vl/evaluate_phase2a.py \
    --checkpoint ./checkpoints/contrastive_v4/best.pt \
    --features_dir ./precomputed_features \
    --val_manifest "$COMBINED_VAL" \
    --max_new_tokens 256 \
    --max_samples 15

EXIT_CODE=$?
rm -f "$COMBINED_VAL"

echo ""
echo "=================================================="
echo "Done — exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=================================================="

exit $EXIT_CODE
