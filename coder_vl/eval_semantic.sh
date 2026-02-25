#!/bin/bash
#SBATCH --job-name=eval-semantic
#SBATCH --partition=teaching
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=eval_semantic.out
#SBATCH --error=eval_semantic.err
#SBATCH --mail-user=gloriosog@msoe.edu
#SBATCH --mail-type=END,FAIL

# Semantic evaluation of Phase 2b outputs using BERT embeddings.
# CPU-only — no GPU needed.
# Runs on existing eval_results_2b.json (no new inference required).
#
# Outputs:
#   eval_semantic.out / eval_semantic.err  — logs
#   eval_semantic_results.json             — full results

echo "=================================================="
echo "Semantic Evaluation (BERTScore + Retrieval)"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Start:  $(date)"
echo "=================================================="

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
cd "$HOME/CoderOCR/OCR-Coder" || exit 1

"$PYTHON" coder_vl/eval_semantic.py \
    --results eval_results_2b.json \
    --model   distilbert-base-uncased \
    --save    eval_semantic_results.json

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Done — exit code: $EXIT_CODE"
echo "End: $(date)"
echo "=================================================="

exit $EXIT_CODE
