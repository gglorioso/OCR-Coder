# MVV Phase 1.3 — Nonlinear Encoding Probe + Geometric Domain Shift Analysis

**Core question:** Is structural information (`n_defs`, `n_classes`) actually absent at 256 tokens,
or does the Phase 1.2 Ridge failure reflect domain shift from the training distribution?

Phase 1.2 Exp2 showed Ridge R² for `n_defs` collapses from 0.851 (729 tokens) to 0.461 (256 tokens).
Phase 1.3 separates two competing explanations:

1. **Information loss:** 256-token features genuinely cannot encode function count.
2. **Domain shift:** Reducing the token budget rotates the SigLIP feature space, confounding the
   cross-resolution probe.

Both experiments were run and completed. Results support the **domain shift** explanation.

---

## Experiment 1: Nonlinear Encoding Probe (`nonlinear_probe/`)

**Status: COMPLETE**

Tested whether the apparent information loss at 256 tokens is a linear-probe artifact or a true
capacity failure. Used two evaluation modes:

- **Mode A (Resolution-as-Test):** Reproduces the Phase 1.2 Exp2 protocol exactly — PCA(1024) fit
  on `budget_729`, Ridge tested on 441 / 256 / 121 tokens. Confirms the degradation baseline.
- **Mode B (Native CV @ 256):** 5-fold cross-validation entirely within `budget_256` features,
  eliminating any cross-resolution domain shift. Probes: Ridge (PCA+linear) and Random Forest
  (300 trees, raw 18,432-dim features, no PCA).

**Configuration:**
- Pool: `pool4x4` (18,432-dim, fp16), 8,938 samples
- PCA: 1,024 components (56.8% variance explained), `whiten=True`, `randomized`
- Ridge: α=100, fit_intercept=True
- RandomForest: 300 estimators, n_jobs=-1
- Features from: `Phase_1_1/exp2_maxpool_comparison/data/features_maxpool/pool4x4/`
- Labels from: `Phase_1_2/exp2_spatial_regression/data/labels.jsonl`

### Mode A Results — Ridge Degradation Curve (train@729, test@N)

| Budget | Pixels | n_defs R² | n_classes R² |
|:------:|:------:|:---------:|:------------:|
| 729    | 378×378 | 0.851 (train) | 0.862 (train) |
| 441    | 294×294 | 0.758      | 0.784         |
| 256    | 224×224 | 0.461      | 0.675         |
| 121    | 154×154 | −0.138     | 0.110         |

Success threshold: R² ≥ 0.8. Neither target reaches threshold at 256 tokens.

### Mode B Results — Native 5-Fold CV at 256 Tokens (no domain shift)

| Probe | n_defs R² (mean ± std) | n_classes R² (mean ± std) |
|-------|:----------------------:|:-------------------------:|
| Ridge (PCA+linear) | 0.672 ± 0.014 | 0.780 ± 0.023 |
| Random Forest (raw) | 0.412 ± 0.018 | 0.660 ± 0.032 |

**Findings:**
- Ridge native CV R² for `n_defs` = 0.672 vs. Mode A cross-resolution R² = 0.461 — a gap of
  +0.211 attributable entirely to domain shift, not information loss.
- Random Forest underperforms Ridge in both targets, ruling out smooth nonlinearity as the cause
  of Mode A failure (the encoder's transformer MLP blocks already apply deep nonlinear transforms).
- The `n_classes` vs. `n_defs` gap persists in Mode B (0.780 vs. 0.672), consistent with the
  visual footprint hypothesis: class definitions have larger visual extent (~5–30 lines) than
  single-line `def` markers.
- Neither probe reaches the R² ≥ 0.8 success threshold at 256 tokens, but the native CV scores
  are substantially higher, indicating the Phase 1.2 result was partly a measurement artifact.

---

## Experiment 2: Geometric Domain Shift Analysis (`domain_shift/`)

**Status: COMPLETE**

Directly quantified how much the SigLIP `pool4x4` feature geometry changes when the token budget
drops from 729 to 256. Three metrics were computed on 8,938 matched image pairs after PCA(1024)
projection fit on `budget_729`.

**Configuration:**
- Features: same `pool4x4` .pt files, 18,432-dim fp16
- PCA: 1,024 components, `whiten=True`, fit on `budget_729`, frozen for `budget_256` projection
- CKA subsampled to 2,000 rows (O(N²) memory constraint)

### Metric 1 — Per-Image Cosine Similarity (729 vs. 256 in PCA space)

| Statistic | Value |
|-----------|-------|
| Mean cosine similarity | 0.292 |
| Std | 0.074 |
| Median | 0.298 |
| Min / Max | 0.011 / 0.615 |
| Fraction below 0.5 | 99.6% |

Verdict: **SEVERE directional drift.** The same image's 256-token feature vector points in a nearly
orthogonal direction to its 729-token counterpart after shared PCA projection.

### Metric 2 — Class-Conditioned Centroid Drift

Centroid drift is measured as the Euclidean displacement of the class centroid from budget_729 to
budget_256 in PCA space, normalized by the within-class spread at budget_729 (drift_ratio).
A ratio > 1.0 means the centroid moves farther than the natural within-class variance.

**n_defs groups:**

| Group | N | Centroid Drift | Within-Var (729) | Drift Ratio |
|-------|--:|:--------------:|:----------------:|:-----------:|
| 0     | 3,304 | 21.59 | 31.37 | 0.688 |
| 1     | 2,611 | 21.43 | 31.97 | 0.670 |
| 2     | 1,490 | 21.48 | 32.37 | 0.664 |
| 3     |   730 | 21.61 | 32.73 | 0.661 |
| 4+    |   803 | 21.74 | 31.48 | 0.691 |

All drift ratios are ~0.66–0.69. No group exceeds 1.0, meaning the centroid displacement
stays within the natural within-class spread. Centroid-level structure is partially preserved
even under severe per-image rotation.

**n_classes groups (4+ drift ratio = 0.814, highest across all groups):**

| Group | N | Drift Ratio |
|-------|--:|:-----------:|
| 0     | 4,502 | 0.667 |
| 1     | 3,317 | 0.684 |
| 2     |   604 | 0.668 |
| 3     |   184 | 0.656 |
| 4+    |   331 | 0.814 |

**line_count groups** (all drift ratios 0.668–0.688): stable across short (1–10 lines) and
long (31–40 lines) windows.

### Metric 3 — Linear CKA (Centered Kernel Alignment)

| Metric | Value |
|--------|-------|
| Linear CKA (P_729 vs. P_256) | 0.426 |
| N subsampled | 2,000 |
| Elapsed | 0.31 s |

CKA = 0.426, **below the 0.5 threshold** — substantial structural divergence between the
two budget representations at the global geometry level.

### Domain Shift Verdict: CONFIRMED

All three metrics point to the same conclusion:

- Cosine similarity: mean 0.292 (severe per-image rotation)
- CKA: 0.426 (global representational divergence)
- Centroid drift: 0.66–0.81× within-class spread (structure partially preserved at class level)

The SigLIP `pool4x4` encoder changes the geometry of its output substantially when the token
budget is halved. The cross-resolution evaluation in Phase 1.2 was measuring both information
capacity *and* this geometric domain shift conflated together.

---

## Combined Interpretation

| Finding | Implication |
|---------|-------------|
| Mode B Ridge n_defs R²=0.672 vs. Mode A R²=0.461 | 0.211 gap = domain shift artifact, not absent signal |
| Mode B Ridge n_classes R²=0.780 | n_classes encodes near the 0.8 threshold when shift is removed |
| RF < Ridge in Mode B | No additional nonlinear structure beyond what transformer MLPs already capture |
| CKA=0.426, mean cos_sim=0.292 | Feature space rotation is the primary cause of probe degradation |
| All centroid drift ratios < 1.0 | Class-level geometry is recoverable; per-image geometry is not |

**Next step:** Phase 1.4 should address domain shift directly — either via budget-aware PCA
(fit a separate projection per budget level) or by training the probe exclusively at the target
budget rather than generalizing across resolutions.

---

## File Structure

```
Phase_1_3/
├── README.md
├── nonlinear_probe/
│   ├── scripts/
│   │   └── run_probe.py          ← Mode A (Ridge degradation) + Mode B (native CV @ 256)
│   └── results/
│       ├── probe_results.json    ← all R² values, fold details
│       └── comparison_plot.png   ← Panel A: degradation curve | Panel B: CV bars
└── domain_shift/
    ├── scripts/
    │   ├── run_drift_analysis.py ← cosine sim, centroid drift, linear CKA, t-SNE
    │   └── plot_domain_shift.py  ← drift_ratio_bar.png, ndefs_displacement.png
    └── results/
        ├── drift_analysis.json   ← all three metrics + interpretation
        ├── drift_ratio_bar.png   ← grouped bar: drift ratio per label bucket × target
        ├── ndefs_displacement.png← centroid drift vs. within-class variance for n_defs
        └── tsne_drift.png        ← t-SNE: 729 vs 256 token features (400 pairs)
```

---

## Running

No SLURM needed. Both scripts are CPU-only sklearn, each runs in ~2–5 minutes.

```bash
cd ~/CoderOCR/OCR-Coder

# Nonlinear probe (Mode A + Mode B)
python MVV/Phase_1_3/nonlinear_probe/scripts/run_probe.py

# Domain shift analysis
python MVV/Phase_1_3/domain_shift/scripts/run_drift_analysis.py

# Regenerate drift visualisation plots from existing JSON
python MVV/Phase_1_3/domain_shift/scripts/plot_domain_shift.py
```
