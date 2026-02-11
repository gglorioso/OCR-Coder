#!/bin/bash
#SBATCH --job-name=data-gen-phase2a
#SBATCH --partition=teaching
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-data-gen-%j.out
#SBATCH --error=slurm-data-gen-%j.err

# DeepSeek-Coder-VL Phase 2a Data Generation
# Generates ~10K training examples from cloned repos
#
# Usage:
#   sbatch simple_data_gen.sh

echo "========================================"
echo "Phase 2a Data Generation"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo ""

# Python environment
PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"

# Directories (using home directory - repos in "Scraped Repos" folder)
REPOS_DIR="$HOME/CoderOCR/OCR-Coder/Scraped Repos"
OUTPUT_DIR="$HOME/CoderOCR/OCR-Coder/Data Crawling/output"

# Check if repos directory exists
if [ ! -d "$REPOS_DIR" ]; then
    echo "❌ Repos directory not found: $REPOS_DIR"
    echo ""
    echo "You need to clone repos first. Quick setup:"
    echo ""
    echo "  mkdir -p $REPOS_DIR && cd $REPOS_DIR"
    echo "  git clone --depth 1 https://github.com/django/django"
    echo "  git clone --depth 1 https://github.com/pallets/flask"
    echo "  git clone --depth 1 https://github.com/tiangolo/fastapi"
    echo "  git clone --depth 1 https://github.com/psf/requests"
    echo "  git clone --depth 1 https://github.com/encode/httpx"
    echo "  git clone --depth 1 https://github.com/pallets/click"
    echo "  git clone --depth 1 https://github.com/pydantic/pydantic"
    echo "  git clone --depth 1 https://github.com/python/cpython"
    echo "  git clone --depth 1 https://github.com/numpy/numpy"
    echo "  git clone --depth 1 https://github.com/pandas-dev/pandas"
    echo "  git clone --depth 1 https://github.com/scikit-learn/scikit-learn"
    echo "  git clone --depth 1 https://github.com/pytorch/pytorch"
    echo "  git clone --depth 1 https://github.com/huggingface/transformers"
    echo "  git clone --depth 1 https://github.com/fastapi/fastapi"
    echo "  git clone --depth 1 https://github.com/psf/black"
    echo ""
    exit 1
fi

echo "Repos directory: $REPOS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Count repos (handle spaces in path)
NUM_REPOS=$(ls -d "$REPOS_DIR"/*/ 2>/dev/null | wc -l)
echo "Found $NUM_REPOS repositories"
echo ""

# Run the data generation pipeline
"$PYTHON" "$HOME/CoderOCR/OCR-Coder/Data Crawling/simple_data_gen.py" \
    --repos-dir "$REPOS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --target 10000 \
    --style monokai \
    --font-size 13 \
    --seed 42

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Job complete!"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "========================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Data generation successful!"
    echo ""
    echo "Check outputs:"
    echo "  Images:    ls $OUTPUT_DIR/images | head"
    echo "  Manifests: ls $OUTPUT_DIR/manifests"
    echo "  Train:     wc -l $OUTPUT_DIR/manifests/train.jsonl"
    echo ""
fi

exit $EXIT_CODE

