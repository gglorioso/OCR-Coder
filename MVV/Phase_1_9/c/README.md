# Phase 1.9c — Large-Scale Alignment Training (8,000-Sample Scale-Up)

## Overview

Phase 1.9c is the large-scale alignment test: scaling projector-only training from 500 samples
(Phase 1.9b / Phase 2 diagnostic) to the full ~8,980-sample manifest for 1 epoch, then running
inference to determine whether 16x more data is sufficient to override the LLM's RLHF priors.

The projector initializes from Phase 2's `best_aligned.pt` (val_loss=1.3918), which itself was
trained for 2 epochs on 500 samples starting from a keyword-classification baseline.

---

## Training Sequence

Two runs led to this evaluation:

| Run | Dataset | Epochs | Init | Final val_loss |
|---|---|---|---|---|
| Phase 2 diagnostic | ~500 train samples | 2 | Phase 1.9a BCE checkpoint | **1.3918** |
| Phase 1.9c scale-up | ~8,082 train samples (90/10 split of 8,980) | 1 | `Phase_2/checkpoints/best_aligned.pt` | (see training log) |

The Phase 2 diagnostic established the projector checkpoint used as Phase 1.9c's starting point.
Phase 1.9c trains only the projector; the LLM backbone remains strictly frozen throughout.

---

## Architecture

```
Code image → SigLIP features [1024, 1152]
           → ConvRoPEProjector [256, 2048]
               strided Conv2d (32×32 → 16×16)
               2D RoPE (row/col positional encoding)
               MLP (1152 → 2048 → 2048, GELU)
           → concat with tokenized source [T, 2048]
           → DeepSeek-Coder-V2-Lite-Instruct (frozen, 8-bit)
           → cross-entropy loss on text tokens only (first 256 labels = -100)
```

Only the **ConvRoPEProjector** is trained. The LLM is strictly frozen.

---

## Training Configuration

| Parameter | Value |
|---|---|
| Manifest | `MVV/Phase_1_1/data_mvv/manifest.jsonl` (~8,980 entries) |
| Train/Val split | 90/10, fixed seed=42 |
| Train samples | ~8,082 |
| Val samples | ~898 |
| Epochs | 1 |
| Batch size | 1 |
| Gradient accumulation | 4 steps (effective batch = 4) |
| Learning rate | 1e-5 |
| Warmup steps | 100 |
| Max text tokens | 512 |
| Vision tokens masked | First 256 labels set to -100 |
| Position IDs | Explicit, dynamic padding |
| LLM quantization | 8-bit (BitsAndBytes) |
| GPU | 1× V100 (dgx partition) |
| Projector init | `MVV/Phase_2/checkpoints/best_aligned.pt` |

---

## Inference Results (20-Sample Test Set)

Evaluated with `infer_1_9c.py` on 20 held-out samples, seed=42.

| Metric | Value |
|---|---|
| Samples evaluated | 20 |
| Mean Edit Distance | **0.980** |
| Word Salad | 0 |
| Hallucination | 0 |
| Ghosting | 0 |
| Other | **20** |

All 20 samples classified as **OTHER** — coherent, instruction-following responses that do not
reconstruct the source code.

### Comparison vs. Phase 1.9b

| | Phase 1.9b Run 2 (500 samples) | Phase 1.9c (8,082 samples) |
|---|---|---|
| Mean Edit Distance | 0.981 | 0.980 |
| Ghosting count | 0 / 20 | 0 / 20 |
| Failure mode | Clarifying questions / generic code | Clarifying questions / generic code |

Edit distance improved by 0.001 despite a 16x increase in training data — effectively no change.

---

## Observed Failure Patterns

The LLM consistently ignores the visual prefix and defaults to conversational behavior. Patterns
observed across all 20 samples:

| Pattern | Example |
|---|---|
| Prompt regurgitation | Output begins with the system prompt verbatim |
| Embedding explanation | "The following 256 embeddings represent a high-resolution image..." |
| Clarification loop | "I'm unable to reconstruct... I can provide a summary..." |
| Token repetition | `code\ncode\ncode\n...` (40+ repetitions) |
| Generic typing imports | `from typing import List, Optional, Any, Union, Callable, cast` |
| Tokenization artifact | `mathboldmathboldmathbold...` (repeated 100+ times) |
| Placeholder loop | `def main():\n-    def main():\n...` |

---

## Diagnosis — The RLHF Override Problem

**Root cause:** DeepSeek-Coder-V2-Lite-**Instruct** has been fine-tuned with RLHF to behave as
a conversational assistant. When 256 "alien" visual tokens are prepended, the frozen LLM's RLHF
prior dominates: it interprets the opaque tokens as an ambiguous instruction and responds with
safe conversational behavior (explaining, asking for clarification, looping on generic code).

**Why scaling failed:** The projector is the only trainable component. It must learn to produce
256 token embeddings that, when read by a frozen RLHF-tuned LLM, reliably override years of
instruction-following conditioning. This is not achievable with a projector-only approach
regardless of dataset scale — the bottleneck is the frozen backbone, not the adapter.

**Evidence:**
- Phase 1.9b (500 samples): mean edit distance 0.981, 0/20 ghosting
- Phase 1.9c (8,082 samples, 16x scale): mean edit distance 0.980, 0/20 ghosting
- The failure mode is identical across both runs and both projector checkpoints

---

## Conclusion and Path to Phase 3

Phase 1.9c is a clean negative result. It confirms that:

1. **Projector-only training cannot override a frozen RLHF backbone**, regardless of data scale.
2. **Generative alignment requires LoRA unfreezing** of the LLM — the standard LLaVA approach
   (joint projector + backbone fine-tuning).
3. **The bottleneck is architectural, not data-scale.** 16x more data produced ~0% improvement.

However, the contrastive evaluation (Phase 1.10) showed **Recall@5 = 100% zero-shot** on the
retrieval task, demonstrating that the projector's embedding space has learned meaningful
structural representations as a byproduct of generation training — even without generative
decoding succeeding.

Phase 1.9c's failure directly motivates Phase 3: joint LoRA + projector training where the LLM
backbone is partially unfrozen and can be adapted to interpret visual token prefixes.

---

## Files

| File | Description |
|---|---|
| `train_1_9c.py` | Training script (full dataset, 1 epoch) |
| `infer_1_9c.py` | Inference evaluation (20 samples, failure classification) |
| `run_1_9c.sh` | SLURM job: trains then infers sequentially |
| `checkpoints/best.pt` | Best projector checkpoint (lowest val_loss) |
| `checkpoints/epoch_N.pt` | Per-epoch projector checkpoints |
| `results/training_log.jsonl` | Per-epoch metrics (appended during training) |
| `results/reconstruction_report.md` | Full 20-sample inference report (2026-03-17) |
