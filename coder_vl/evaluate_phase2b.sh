#!/bin/bash
#SBATCH --job-name=phase2b-eval
#SBATCH --partition=dgx
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00
#SBATCH --output=2b_eval.out
#SBATCH --error=2b_eval.err
#SBATCH --mail-user=gloriosog@msoe.edu
#SBATCH --mail-type=END,FAIL

# Phase 2b Evaluation: Gates G4 (ROUGE-L), G5 (exact-match), G6 (Distinct-1)
# Prereq: Phase 2b training complete, best.pt in ./checkpoints/phase2b/
# Uses same gates as Phase 2a; compare 2b vs 2a results.
# Saves partial results to ./eval_results_2b.json every 100 examples (resumable).

echo "=================================================="
echo "Phase 2b: Evaluation (G4/G5/G6)"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=================================================="

nvidia-smi

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
cd "$HOME/CoderOCR/OCR-Coder" || exit 1

"$PYTHON" -c "import peft" 2>/dev/null || "$PYTHON" -m pip install peft --quiet

"$PYTHON" coder_vl/evaluate_phase2b.py \
    --checkpoint ./checkpoints/phase2b/best.pt \
    --features_dir ./precomputed_features_tiled \
    --val_manifest data_v2b/manifests/val.jsonl \
    --max_new_tokens 100 \
    --repetition_penalty 1.3 \
    --save_file ./eval_results_2b.json \
    --save_every 100

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Done — exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=================================================="

exit $EXIT_CODE
