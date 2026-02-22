#!/bin/bash
#SBATCH --job-name=phase2a-eval
#SBATCH --partition=dgx
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00
#SBATCH --output=2a_eval.out
#SBATCH --error=2a_eval.err
#SBATCH --mail-user=gloriosog@msoe.edu
#SBATCH --mail-type=END,FAIL

# Phase 2a Evaluation: Gates G4 (ROUGE-L), G5 (exact-match), G6 (Distinct-1)
# Prereq: training complete, best.pt exists in ./checkpoints/phase2a_v6/
# Expected: ~14-16 hours on 1x V100 (2086 val examples, max_new_tokens=100)
# Saves partial results to ./eval_results_v6.json every 100 examples (resumable).

echo "=================================================="
echo "Phase 2a: Evaluation (G4/G5/G6)"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=================================================="

nvidia-smi

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
cd "$HOME/CoderOCR/OCR-Coder" || exit 1

# Combine val manifests (same as training)
COMBINED_VAL=$(mktemp /tmp/phase2a_eval_val_XXXX.jsonl)
cat "Data Crawling/output/manifests/val.jsonl" "data_v2/manifests/val.jsonl" > "$COMBINED_VAL"
echo "Combined val: $(wc -l < "$COMBINED_VAL") examples"

"$PYTHON" coder_vl/evaluate_phase2a.py \
    --checkpoint ./checkpoints/phase2a_v6/best.pt \
    --features_dir ./precomputed_features_tiled \
    --val_manifest "$COMBINED_VAL" \
    --max_new_tokens 100 \
    --repetition_penalty 1.3 \
    --save_file ./eval_results_v6.json \
    --save_every 100

EXIT_CODE=$?
rm -f "$COMBINED_VAL"

echo ""
echo "=================================================="
echo "Done — exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=================================================="

exit $EXIT_CODE
