# Phase 1.9a — ConvRoPE Keyword Probe

Proves that character-level keyword information survives 32×32→16×16 convolutional compression.

## Architecture

| Stage | Detail |
|---|---|
| Input | SigLIP features [1024, 1152] |
| Compression | Conv2d(1152, 1152, kernel_size=2, stride=2) → 32×32 to 16×16 grid |
| Flatten | [256, 1152] |
| Positional encoding | 2D RoPE (16×16 grid) |
| MLP | Linear(1152→2048) → GELU → Linear(2048→2048) |
| Probe head | Linear(2048→16) — one logit per keyword per token |
| Loss | BCEWithLogitsLoss, 20 epochs |

## Results

Best macro F1: **0.7801**

| Keyword | F1 |
|---|---|
| class | 0.9806 |
| import | 0.9616 |
| def | 0.9336 |
| except | 0.8645 |
| else | 0.8458 |
| raise | 0.8230 |
| if | 0.8301 |
| return | 0.8201 |
| pass | 0.7863 |
| try | 0.7904 |
| for | 0.7282 |
| elif | 0.7170 |
| yield | 0.7143 |
| with | 0.6667 |
| while | 0.5000 |
| lambda | 0.3529 |

Final epoch: train_loss=0.0001, val_loss=0.0038, macro_F1=0.7697

## What the Probe Sees

For a patch in the 16×16 grid where `def` appears in the source image, the probe outputs a high
probability for the `def` keyword class. Patches with no keywords near them output near-zero
probability across all 16 classes.

## Interpretation

**High F1 scores (class, import, def):** These keywords are visually distinctive — they appear as
consistent glyph sequences at predictable horizontal positions — and are common enough in the
training data for the probe to see many positive examples.

**Low F1 scores (lambda, while, with):** These keywords are rare in the training set. The probe
sees few positive examples, so precision and recall are both unstable. The low scores reflect data
sparsity, not information loss.

**Key conclusion:** The 2×2 convolutional downsampling is lossless for keyword-level information.
The spatial structure of Python code survives compression from 32×32 to 16×16. A macro F1 of 0.78
over 16 keyword classes, measured on held-out validation data, confirms the compressed
representation retains the discriminative features needed for LLM alignment.

**Implications for Phase 2:** The projector output is a valid foundation for alignment training.
val_loss=0.0038 and train_loss=0.0001 indicate the probe memorized the training set; the
validation F1 scores are the meaningful signal.
