#!/bin/bash
#SBATCH --job-name=precompute-v2
#SBATCH --partition=dgx
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=2:00:00
#SBATCH --output=slurm-precompute-v2-%j.out
#SBATCH --error=slurm-precompute-v2-%j.err
#SBATCH --mail-user=gloriosog@msoe.edu
#SBATCH --mail-type=END,FAIL

# Pre-compute vision features for v2 (dracula) images.
#
# Prereq: data_gen_v2.sh must have completed (./data_v2/manifests/*.jsonl exist)
#
# Saves to the SAME ./precomputed_features/ directory as v1.
# No conflicts because v2 image stems end in _dracula vs v1's _monokai.
#
# After this completes, run:
#   sbatch coder_vl/contrastive_pretrain.sh
#   (now passes both v1 + v2 manifests)

echo "=================================================="
echo "Pre-computing Vision Features (v2 / dracula)"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Start:  $(date)"
echo "=================================================="

nvidia-smi

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
cd "$HOME/CoderOCR/OCR-Coder" || exit 1

"$PYTHON" coder_vl/precompute_features.py \
    --manifest_dir ./data_v2/manifests \
    --output_dir   ./precomputed_features \
    --image_size   768

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Done — exit code: $EXIT_CODE"
echo "End: $(date)"
echo "=================================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Features in ./precomputed_features/: $(ls ./precomputed_features/*.pt | wc -l) total .pt files"
    echo "  (includes both _monokai and _dracula)"
    echo ""
    echo "Next: sbatch coder_vl/contrastive_pretrain.sh"
fi

exit $EXIT_CODE
