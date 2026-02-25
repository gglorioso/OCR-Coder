#!/bin/bash
#SBATCH --job-name=linear-probe
#SBATCH --partition=teaching
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=probe.out
#SBATCH --error=probe.err
#SBATCH --mail-user=gloriosog@msoe.edu
#SBATCH --mail-type=END,FAIL

# Test 3: Linear Probe on Visual Features
# Tests whether precomputed visual features contain code-structure information
# WITHOUT using the LLM. No GPU required.
#
# Probe A: Can logistic regression predict source_file from avg-pooled features?
# Probe B: Binary — does the image contain class definitions?

echo "=================================================="
echo "Test 3: Linear Probe on Visual Features"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "=================================================="

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
cd "$HOME/CoderOCR/OCR-Coder" || exit 1

# Ensure sklearn is available
"$PYTHON" -c "import sklearn" 2>/dev/null || "$PYTHON" -m pip install scikit-learn --quiet

"$PYTHON" coder_vl/test_linear_probe.py \
    --features_dir  ./precomputed_features_tiled \
    --val_manifest  data_v2b/manifests/val.jsonl \
    --min_samples   3 \
    --mlp \
    --save_file     ./probe_results.json

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Done — exit code: $EXIT_CODE"
echo "End: $(date)"
echo "=================================================="

exit $EXIT_CODE
