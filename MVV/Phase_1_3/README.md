# MVV Phase 1.3 — Nonlinear Encoding Probe

**Core question:** Is `n_defs` (and `n_classes`) information nonlinearly encoded in the
256-token pool4x4 features, even though Ridge regression fails (R²=0.461)?

Phase 1.2 Exp2 established that a linear probe (Ridge) cannot extract function count from
256-token features. This does **not** mean the information is absent — only that it is not
linearly decodable. Phase 1.3 tests the nonlinear hypothesis using k-NN and MLP probes
on the identical features and PCA projection.

---

## Interpretation Key

| Outcome | Meaning |
|---|---|
| Both nonlinear probes fail (R² ≈ Ridge 0.461) | Information genuinely destroyed at 256 tokens |
| MLP passes, k-NN fails | Smooth nonlinear structure exists but is not locally consistent |
| Both pass | Info is nonlinearly encoded; projection adapter should use nonlinear heads |

---

## Experiment: `nonlinear_probe`

**Status: 🔲 NOT STARTED**

**Protocol** (identical to Phase 1.2 Exp2 up to the probe head):
- Pool: `pool4x4` only (winner from Exp2)
- PCA(1024, whiten=True, randomized), fit on budget_729, frozen for test budgets
- Resolution-as-Test: train on budget_729, evaluate on 441 / 256 / 121
- Targets: `n_defs`, `n_classes` (both failures from Phase 1.2)

**Probes:**

| Probe | Config | Why |
|---|---|---|
| Ridge | α=100 | Reproduced baseline from Exp2 (apples-to-apples) |
| k-NN | k=10, distance-weighted | Non-parametric; tests local geometric consistency |
| MLP | 1024→256→1, ReLU, early stopping | Tests for smooth nonlinear structure |

**No new data generation needed.** Reuses:
- Features: `Phase_1_1/exp2_maxpool_comparison/data/features_maxpool/pool4x4/`
- Labels: `Phase_1_2/exp2_spatial_regression/data/labels.jsonl`

---

## Running

No SLURM needed — pure sklearn, CPU only, runs in ~2 minutes.

```bash
cd ~/CoderOCR/OCR-Coder
python MVV/Phase_1_3/nonlinear_probe/scripts/run_probe.py
```

Optional overrides:
```bash
python MVV/Phase_1_3/nonlinear_probe/scripts/run_probe.py \
    --n_components 1024 \
    --alpha 100.0 \
    --k 10
```

---

## File Structure

```
Phase_1_3/
├── README.md
└── nonlinear_probe/
    ├── scripts/
    │   └── run_probe.py     ← Ridge + k-NN + MLP on pool4x4 PCA features
    └── results/
        ├── probe_results.json
        └── comparison_plot.png
```

---

## Baseline (from Phase 1.2 Exp2, pool4x4)

| Target | Ridge R² @ 729 | Ridge R² @ 441 | Ridge R² @ 256 | Ridge R² @ 121 |
|---|:---:|:---:|:---:|:---:|
| n_defs    | 0.851 | 0.758 | **0.461** | −0.138 |
| n_classes | 0.862 | 0.784 | **0.675** | 0.110 |

Success threshold: R² ≥ 0.8 at budget_256.
