#!/bin/bash
#SBATCH --job-name=phase16_attn_rope
#SBATCH --partition=teaching
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-phase16-attention-rope-%j.out

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
SCRIPT="$HOME/CoderOCR/OCR-Coder/MVV/Phase_1_6/scripts/run_attention_probe_rope.py"

echo "==== Phase 1.6 — Attention Probe (Experiment B, 2D SinCos Positional Encoding) ===="
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

"$PYTHON" "$SCRIPT"

echo ""
echo "Done: $(date)"
