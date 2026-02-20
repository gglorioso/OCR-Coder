#!/bin/bash
#SBATCH --job-name=linear-probe
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm-probe-%j.out
#SBATCH --error=slurm-probe-%j.err

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
cd /home/ad.msoe.edu/gloriosog/CoderOCR/OCR-Coder

echo "========================================"
echo "LINEAR PROBE TEST"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "========================================"

# Step 1: Generate labels from manifests (CPU only, fast)
echo ""
echo "=== Step 1: Generate probe labels ==="
"$PYTHON" coder_vl/linear_probe/generate_probe_labels.py
if [ $? -ne 0 ]; then
    echo "FAILED: Label generation"
    exit 1
fi

# Step 2: Extract OCR-2 features (loads precomputed .pt files, no GPU needed)
echo ""
echo "=== Step 2: Extract features (OCR-2) ==="
"$PYTHON" coder_vl/linear_probe/extract_probe_features.py --encoder ocr2
if [ $? -ne 0 ]; then
    echo "FAILED: OCR-2 feature extraction"
    exit 1
fi

# Step 3: Train probes on OCR-2 features
echo ""
echo "=== Step 3: Train linear probes (OCR-2) ==="
"$PYTHON" coder_vl/linear_probe/train_linear_probe.py --encoder ocr2

# Step 4: Extract SigLIP features (needs GPU)
if [ -f models/siglip_encoder.pt ]; then
    echo ""
    echo "=== Step 4: Extract features (SigLIP) ==="
    "$PYTHON" coder_vl/linear_probe/extract_probe_features.py --encoder siglip

    echo ""
    echo "=== Step 5: Train linear probes (SigLIP) ==="
    "$PYTHON" coder_vl/linear_probe/train_linear_probe.py --encoder siglip
else
    echo ""
    echo "Skipping SigLIP: models/siglip_encoder.pt not found"
fi

echo ""
echo "========================================"
echo "DONE: $(date)"
echo "========================================"
