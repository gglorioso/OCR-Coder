# Phase 3: Joint Training — Mini Plumbing Test

## Overview

Phase 3 is the joint training phase for DeepSeek-Coder-VL, where the SigLIP vision encoder,
the ConvRoPE MLP projector, and the LoRA-adapted DeepSeek-Coder-V2-Lite-Instruct are trained
end-to-end on a code generation objective.

Before committing to a full-scale cluster run, two validation tests were performed on a single
V100 (Rosie DGX partition) to verify the training pipeline.

**Questions we are trying to answer here**
The Shape Question: Can we successfully slice open the LLM's native 1D text embedding sequence and seamlessly stitch our custom 2D visual tensors inside without PyTorch throwing a dimensionality mismatch error?

The Masking Question: Is the -100 label masking working perfectly? (i.e., Is the model strictly calculating loss on the ground-truth Python code, while safely ignoring the <|pad|> placeholders and structural \n tokens?)

---

## Architecture

```
Code image → SigLIP (frozen, 1152D) → ConvRoPE MLP Projector (→ 2048D) → DeepSeek-Coder-V2-Lite (LoRA)
Bug report (text) ──────────────────────────────────────────────────────────────────────────────────────→
```

**Model parameters:**

| Component       | Parameters       | Status               |
|-----------------|-----------------|----------------------|
| SigLIP encoder  | (not counted)    | Frozen               |
| MLP Projector   | 11,867,264       | Trainable            |
| LoRA adapters   | 289,837,056      | Trainable            |
| LLM backbone    | 15,996,321,280   | Frozen (except LoRA) |
| **Trainable total** | **301,704,320** |                   |

Projector initialized from: `MVV/Phase_2/checkpoints/best_aligned.pt`

---

## Test A — Plumbing Test (Overfit 1 Batch)

**Purpose:** Verify that gradients flow correctly through the full pipeline — from the visual
embedding splice point through the projector and into the LoRA layers. A single sample is
trained for 20 epochs. Monotonically decreasing loss confirms the plumbing is intact.

**Setup:**
- 1 sample: pre-extracted SigLIP tensor [1024, 1152] + 40-line Python source .txt
- 20 epochs, no validation split
- Runtime: ~10 minutes on a single V100

**Results (SLURM job 237185):**

| Epoch | Loss   |
|-------|--------|
| 1     | 2.0337 |
| 2     | 2.0337 |
| 3     | 2.0087 |
| 4     | 1.9729 |
| 5     | 1.9396 |
| 6     | 1.8658 |
| 7     | 1.7955 |
| 8     | 1.7012 |
| 9     | 1.6195 |
| 10    | 1.5214 |
| 11    | 1.4229 |
| 12    | 1.3278 |
| 13    | 1.2436 |
| 14    | 1.1612 |
| 15    | 1.0895 |
| 16    | 1.0103 |
| 17    | 0.9223 |
| 18    | 0.8247 |
| 19    | 0.7308 |
| 20    | 0.6438 |

**Verdict: PASS.** Loss decreased steadily from 2.03 to 0.64. All gradients are flowing
correctly through the visual splice, projector, and LoRA layers.

---

## Test B — Loss Curve Test (100 Samples, 5 Epochs)

**Purpose:** Observe whether train and val loss move in the expected direction at small scale.
This is a sanity check, not a generalization test — 90 training samples is insufficient for
generalization and overfitting is expected.

**Setup:**
- 100 samples total, 90 train / 10 val split
- 5 epochs
- Runtime: ~2 hours on a single V100

**Results:**

| Epoch | Train Loss | Val Loss |
|-------|------------|----------|
| 1     | 1.6925     | 1.3504   |
| 2     | 1.1161     | 1.0607   |
| 3     | 0.8376     | 1.0004   |
| 4     | 0.4834     | 1.1208   |
| 5     | 0.2551     | 1.2034   |

Best val loss: **1.0004** at epoch 3.

**Verdict: Expected behavior.** Train loss collapses (1.69 → 0.25) while val loss diverges
after epoch 3. This is a dataset size artifact, not a model defect. With 90 samples,
memorization is inevitable.

---

## Known Issues

**LoRA learning rate too high.** The LoRA LR ramped to approximately 2e-4 by the end of
training — 20x above the safe ceiling established in Phase 1.9c (`lr_lora <= 1e-5`). This
likely accelerated val divergence. The warmup schedule in `train_joint.py` must be capped
before the full-scale run.

---

## Next Steps

1. **Fix lr schedule** — cap `lr_lora <= 1e-5` in `train_joint.py` to prevent catastrophic
   forgetting of contrastive alignment.
2. **Render remaining dataset** — approximately 7,900 additional files need to be rendered to
   build the full training corpus.
3. **Submit DGX allocation request** — the Rosie DGX supercomputer application should be
   backed by the mini-run metrics documented here.
4. **Execute full-scale Phase 3 run** — once allocation is granted, submit the full training
   job.

---

## Files

| File                  | Description                                      |
|-----------------------|--------------------------------------------------|
| `train_joint.py`      | Joint training script (projector + LoRA)         |
| `run_phase3_mini.sh`  | SLURM batch script for the mini plumbing test    |
| `mini_data/`          | 100-sample dataset used for Tests A and B        |
| `checkpoints/`        | Saved model checkpoints from mini runs           |
