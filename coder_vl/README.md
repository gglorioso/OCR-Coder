# CoderVL Implementation

Vision-enabled DeepSeek-Coder model implementation for Phase 2.

## Files

### Core Modules

- **`projector.py`** - Projection adapter (13.6M params)
  - 2-layer MLP: `1280D → 4096D → 2048D`
  - Maps SigLIP visual features to DeepSeek-Coder embedding space
  - Includes built-in test function

- **`model.py`** - Complete CoderVL model
  - Loads vision encoder + coder model + adapter
  - Implements LLaVA-style token replacement
  - Handles forward pass with image + text input
  - Supports both training and inference

- **`train_projector.py`** - Phase 2a training script
  - Trains adapter only (vision encoder + coder frozen)
  - Loads manifests from `Data Crawling/output/manifests/`
  - Implements proper loss masking (assistant tokens only)
  - Includes checkpointing, evaluation, wandb logging

### Job Scripts

- **`train_phase2a.sh`** - SLURM job script for dgxh100
  - Partition: `dgxh100` (H100 80GB)
  - Runtime: ~6-10 hours
  - Saves checkpoints to `./checkpoints/phase2a`

### Testing

- **`test_adapter.py`** - Local test script
  - Verifies adapter shape transformation
  - Runs without GPU
  - Use before submitting SLURM job

## Usage

### 1. Test Adapter Locally

```bash
python coder_vl/test_adapter.py
```

Expected output: ✅ shape transformation verified (1120×1280 → 1120×2048)

### 2. Submit Phase 2a Training Job

```bash
sbatch coder_vl/train_phase2a.sh
```

Monitor with:
```bash
tail -f slurm-phase2a-<jobid>.out
```

### 3. Check Training Progress

Checkpoints saved to: `./checkpoints/phase2a/`

Wandb dashboard: https://wandb.ai/[your-username]/deepseek-coder-vl

## Phase 2a Success Gates

From [PHASE2_PLAN.md](../Context /PHASE2_PLAN.md) Section 8:

- ✅ Training loss < 3.0
- ✅ Validation loss < 3.5
- ✅ ROUGE-L > 0.25 on code descriptions
- ✅ Function listing accuracy > 30%
- ✅ No overfitting (train-val gap < 0.5)
- ✅ Output diversity (Distinct-1 > 0.3)

All gates must pass before proceeding to Phase 2b.

## Architecture

```
Code Image (PNG)
  ↓
SigLIP Vision Encoder (frozen, 1280D) - from DeepSeek-OCR-2
  ↓
Projection Adapter (trainable, 13.6M params)
  ↓ 1280D → 4096D → 2048D
DeepSeek-Coder-V2-Lite (frozen, 2048D)
  ↓
Code Understanding + Generation
```

## Next Steps After Phase 2a

1. Evaluate against gates (Section 8 in PHASE2_PLAN.md)
2. If gates pass → proceed to Phase 2b (adapter + LoRA)
3. If gates fail → debug, try architecture ablations (E04-E06)

## Prerequisites - Vision Encoder Extraction

⚠️ **Before running Phase 2a training, you must extract the vision encoder:**

### Step 1: Extract Vision Encoder

The full DeepSeek-OCR-2 model is ~26 GB in memory. We extract only the vision components (~1.5-2 GB) to save memory during training.

**Run extraction on GPU node:**

```bash
sbatch coder_vl/extract_encoder.sh
```

This will:
1. Load DeepSeek-OCR-2 (uses ~13-16 GB VRAM in fp16)
2. Extract: SAM + Qwen2Decoder2Encoder + MlpProjector
3. Discard: Language decoder (Qwen2)
4. Save to: `./models/vision_encoder.pt`

**Runtime:** ~5-10 minutes on V100

### Step 2: Verify Extraction

Check the output file:

```bash
ls -lh ./models/vision_encoder.pt
# Expected: ~1.5-2 GB
```

### What Gets Extracted

From DeepSeek-OCR-2's vision pipeline:

```
Image [3, H, W]
  ↓
SAM (ImageEncoderViT)
  ↓ [896, 16, 16]
Qwen2Decoder2Encoder
  ↓ [256, 896] (base view) or [1120, 896] (with patches)
MlpProjector
  ↓ [256, 1280] or [1120, 1280]
```

These three components become the `vision_encoder.pt` file.

See [PHASE2_PLAN.md](../Context /PHASE2_PLAN.md) Section 6 for technical details.
