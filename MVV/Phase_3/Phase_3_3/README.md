# Phase 3.3 -- Fixed LR Training

## What This Phase Is

Phase 3.3 is a surgical patch of the Phase 3 training script, fixing the lr_lora blowout that caused validation divergence. It is NOT a rewrite -- the architecture (ConvRoPEProjector, CoderVLModel, JointDataset, collate_fn) is identical to Phase 3.

## What Changed from Phase 3

| Setting | Phase 3 | Phase 3.3 |
|---------|---------|-----------|
| lr_lora | 2e-4 | 5e-6 |
| lr_projector | 1e-5 | 1e-5 (unchanged) |
| LR schedule | Cosine warmup (100 steps) | Flat (no scheduler) |
| Epochs | 5 | 10 |
| Progress | Print statements | tqdm |
| model_path | Hardcoded | CLI arg |
| Checkpointing | Per-epoch | Per-epoch (unchanged) |

## The Question This Test Answers

Does capping lr_lora at 5e-6 (vs 2e-4) fix the validation divergence observed in Phase 3 starting at epoch 3?

Phase 3 results showed train_loss collapsing (1.69 to 0.25) while val_loss diverged (1.35 to 1.20) -- classic LR overshoot. If val_loss stays flat or continues decreasing past epoch 5, the LR was confirmed as the root cause.

## Data

100-file balanced haystack (25 each from black, flask, django, numpy), 90/10 train/val split.

## How to Run

```bash
sbatch MVV/Phase_3/Phase_3_3/run_train.sh
```

## Files

- `train_joint.py` -- self-contained training script
- `run_train.sh` -- SLURM batch script (dgx, 1 GPU, 4h)
- `checkpoints/` -- per-epoch outputs (projector.pth + lora_adapter/)

## Results

Pending (not yet submitted).
