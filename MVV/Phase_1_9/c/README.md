# Phase 1.9c — Large-Scale Alignment Training

## Objective

Scale the ConvRoPEProjector alignment training from the ~500-sample subsample used in
Phase 2 to the full ~8,980-sample manifest (`MVV/Phase_1_1/data_mvv/manifest.jsonl`),
training for 5 epochs.

The primary target is **GHOSTING emergence**: outputs where Python structural keywords
(`def`, `class`, `import`) are correctly placed and indentation roughly matches the
reference, even if token-level accuracy remains low.

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

## Initialization

- Projector weights loaded from `MVV/Phase_2/checkpoints/best_aligned.pt`
- Phase 2 trained on ~500 samples for 2 epochs, achieved **val_loss = 1.392**
- No LoRA; projector-only training continues from Phase 2's best checkpoint

---

## Training Configuration

| Parameter | Value |
|---|---|
| Manifest | `MVV/Phase_1_1/data_mvv/manifest.jsonl` (~8,980 entries) |
| Train/Val split | 90/10, fixed seed=42 |
| Epochs | 5 |
| Batch size | 1 |
| Gradient accumulation | 4 steps (effective batch = 4) |
| Learning rate | 1e-5 |
| Warmup steps | 100 |
| Max text tokens | 512 |
| LLM quantization | 8-bit (BitsAndBytes) |
| GPU | 1× V100 (dgx partition) |
| PYTORCH_CUDA_ALLOC_CONF | expandable_segments:True |

---

## Results

**Status: Training in progress (2026-03-16)**

Job submitted via `sbatch MVV/Phase_1_9/c/run_1_9c.sh`. Results will be filled once training completes.

### Training Log

| Epoch | Train Loss | Val Loss | Best? |
|---|---|---|---|
| 1 | — | — | — |
| 2 | — | — | — |
| 3 | — | — | — |
| 4 | — | — | — |
| 5 | — | — | — |

### Inference Evaluation (20 samples, seed=42)

| Metric | Value |
|---|---|
| Mean Edit Distance | — |
| Word Salad | — |
| Hallucination | — |
| Ghosting | — |
| Other | — |

Full inference report: `results/reconstruction_report.md`

---

## Success Criterion

**GHOSTING** is the target outcome: outputs where structural keywords (`def`, `class`,
`import`) appear in plausible positions and indentation roughly matches the reference
file, even if exact token reconstruction remains poor (edit_distance 0.3–0.8).

A result is considered successful if at least **3 out of 20** sampled outputs are
classified as GHOSTING (vs. 0 in Phase 1.9b baseline).

---

## Files

| File | Description |
|---|---|
| `train_1_9c.py` | Training script (full dataset, 5 epochs) |
| `infer_1_9c.py` | Inference evaluation (20 samples, failure classification) |
| `run_1_9c.sh` | SLURM job: trains then infers sequentially |
| `checkpoints/best.pt` | Best projector checkpoint (lowest val_loss) |
| `checkpoints/epoch_N.pt` | Per-epoch projector checkpoints |
| `results/training_log.jsonl` | Per-epoch metrics (appended during training) |
| `results/reconstruction_report.md` | Inference evaluation report |
