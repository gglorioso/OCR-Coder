# MVV Phase 1.5 — 256-Token Compression Method Comparison

**Core question:** Which of three token compression strategies best preserves code structure
information when reducing SigLIP features to a 256-token budget?

Phase 1.4 established baseline Ridge probes on single-scale tokens. Phase 1.5 compares three
compression methods across two pooling resolutions (4×4 and 8×8), probing for `line_count`,
`n_defs`, and `n_classes`.

---

## Methods

| Method | Description | Output shape |
|---|---|---|
| **Method 1** | Naive downsampling (resize to budget_256) | pool4x4: 16×16×1152 / pool8x8: 64×1152 |
| **Method 2** | Native 448px render + average pool | pool4x4: 16×16×1152 / pool8x8: 64×1152 |
| **Method 3** | Token pruning + average pool | pool4x4: 16×16×1152 / pool8x8: 64×1152 |

All conditions produce 18,432D (pool4x4) or 73,728D (pool8x8) flat feature vectors.

---

## Results (PCA + Ridge, 5-fold CV)

### line_count R²

| Method | pool4x4 | pool8x8 |
|---|:---:|:---:|
| Method 1 (naive downsample) | 0.957 | 0.967 |
| **Method 2 (native+avgpool)** | 0.977 | **0.982** |
| Method 3 (pruning+avgpool) | 0.941 | 0.954 |

### n_defs R²

| Method | pool4x4 | pool8x8 |
|---|:---:|:---:|
| Method 1 (naive downsample) | 0.672 | 0.692 |
| **Method 2 (native+avgpool)** | 0.757 | **0.802** |
| Method 3 (pruning+avgpool) | 0.648 | 0.644 |

### n_classes R²

| Method | pool4x4 | pool8x8 |
|---|:---:|:---:|
| Method 1 (naive downsample) | 0.780 | 0.811 |
| **Method 2 (native+avgpool)** | 0.825 | **0.856** |
| Method 3 (pruning+avgpool) | 0.703 | 0.702 |

---

## Key Findings

1. **Method 2 (native 448px + avg pool) wins decisively across all targets and both resolutions.**
2. **pool8x8 consistently beats pool4x4** — the larger 64-token grid preserves spatial layout that
   pool4x4 discards.
3. **Method 3 (token pruning) is surprisingly weak** — pruning appears to discard layout-critical
   tokens, underperforming even naive downsampling on n_defs and n_classes.
4. **method2_pool8x8 (R²=0.802 on n_defs)** becomes the canonical feature set for Phase 1.6+,
   where an attention probe with 2D positional encoding further improves n_defs to R²=0.873.

---

## Experimental Setup

| Param | Value |
|---|:---:|
| Probe | PCA (whiten=True, max 1024 components) + Ridge (α=100) |
| CV | 5-fold, no data leakage |
| n_samples | 8,938 |
| Targets | line_count, n_defs, n_classes |

---

## Running

```bash
cd ~/CoderOCR/OCR-Coder

# Extract features (requires GPU — runs on Rosie)
sbatch MVV/Phase_1_5/scripts/extract_features_1_5.sh

# Run probe (CPU, ~5 min)
bash MVV/Phase_1_5/scripts/run_probe_1_5.sh
```

---

## File Structure

```
Phase_1_5/
├── README.md
├── scripts/
│   ├── extract_features_1_5.py   ← Feature extraction (SigLIP, all 3 methods)
│   ├── extract_features_1_5.sh   ← SLURM wrapper
│   ├── run_probe_1_5.py          ← PCA+Ridge probe for all conditions
│   └── run_probe_1_5.sh          ← Shell runner
├── data/
│   └── features/
│       ├── method2/pool8x8/      ← Committed (canonical features for Phase 1.6+)
│       └── method2/pool4x4/      ← gitignored (large, regeneratable)
└── results/
    └── probe_results_1_5.json    ← Full 5-fold results for all 6 conditions
```
