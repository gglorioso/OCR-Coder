# MVV Phase 1.2 — Structural Regression Probes

**Status:** Complete (corrected 2026-03-05)

## Experiments

### Exp1 — Mean-Pool Structural Regression

**Status: DONE — Clean ablation isolating mean pooling as bottleneck**

Labels correctly windowed to the visible 40-line AST window (same labels as Exp2).
Only variable vs Exp2: **mean-pool features** instead of spatial max-pool.

| Budget | line_count R² | n_defs R² | n_classes R² |
|--------|:---:|:---:|:---:|
| 729 (train) | 0.970 | 0.833 | 0.855 |
| 441 | 0.921 | 0.720 | 0.773 |
| **256** | **0.855** | **0.364** | **0.568** |
| 121 | 0.041 | -0.479 | -0.405 |

**Finding:** Mean pooling is the architectural bottleneck — even with correct windowed labels,
n_defs and n_classes fail at 256 tokens. Fixing the labels alone is insufficient;
the feature representation must preserve spatial structure.

**Key files:** [exp1_structural_regression/results/regression_results.json](exp1_structural_regression/results/regression_results.json)

---

### Exp2 — Spatial Pooling + Native CV (Corrected)

**Status: DONE (corrected 2026-03-05)**

**Method:** pool4x4 spatial features, PCA(1024), Ridge(alpha=100), **5-fold native CV per budget**
on 8,821 clean MVV images (800x800, no distortion).

**Results (native CV, clean MVV images, 8,821 samples):**

| Target | tok=729 | tok=441 | tok=256 | tok=121 |
|---|---|---|---|---|
| line_count | 0.962±0.004 | 0.955±0.003 | 0.957±0.003 | 0.947±0.004 |
| n_defs | 0.804±0.020 | 0.777±0.015 | 0.672±0.014 | 0.498±0.021 |
| n_classes | 0.818±0.022 | 0.794±0.023 | 0.780±0.023 | 0.593±0.032 |

**Results by target at 256 tokens:**

| Target | Visual footprint | R²@256 (pool4x4, native CV) | Verdict |
|---|---|:---:|:---:|
| `line_count` | Global fill vs. bottom whitespace | **0.957** | PASS |
| `n_classes` | Large block (header + body, ~5-30 lines) | **0.780** | PASS |
| `n_defs` | Small marker, ~1-3 lines | **0.672** | PARTIAL |

**Key findings:**
- Spatial pooling **does** preserve structural information. The earlier failure was a measurement artifact.
- line_count is essentially flat (0.947-0.962) — survives even 121-token compression.
- n_defs drops below R²=0.5 between 256 and 121 tokens (0.672 -> 0.498).
- n_classes stays above 0.5 even at 121 tokens (0.593).
- True information loss is gradual, not the cliff seen in the original domain-shifted results.

**Correction note:** The original Exp2 protocol trained at budget_729 and tested at budget_256 — a
cross-budget paradigm that introduced a **domain shift confound** (confirmed by Phase 1.3:
cos sim=0.220 across budgets). Old figures (n_defs=0.461@256, n_classes=0.675@256) were measuring
domain shift artifact, not information loss. Native CV eliminates this confound entirely.

**Key files:**
- [exp2_spatial_regression/results/regression_results_v2.json](exp2_spatial_regression/results/regression_results_v2.json) — native CV results
- [exp2_spatial_regression/results/degradation_curve_v2.png](exp2_spatial_regression/results/degradation_curve_v2.png) — degradation curve plot
- [exp2_spatial_regression/results/regression_results.json](exp2_spatial_regression/results/regression_results.json) — original cross-budget results (domain-shifted, superseded)

---

## File Structure

```
Phase_1_2/
├── README.md
├── exp1_structural_regression/
│   ├── scripts/
│   │   ├── gen_labels.py       <- full-file AST counts (flawed baseline)
│   │   ├── run_regression.py   <- LinearRegression on mean-pool features
│   │   └── run_regression.sh
│   ├── data/labels.jsonl
│   └── results/
│       ├── regression_results.json
│       └── degradation_curve.png
└── exp2_spatial_regression/
    ├── scripts/
    │   ├── gen_labels.py           <- windowed counts (only visible 40-line window)
    │   ├── run_regression.py       <- PCA(1024) + Ridge on pool4x4/pool8x8 features (cross-budget, superseded)
    │   ├── run_regression.sh
    │   ├── run_regression_v2.py    <- native CV runner (corrected)
    │   └── plot_results_v2.py      <- degradation curve plotter
    ├── data/labels.jsonl
    └── results/
        ├── regression_results.json      <- original cross-budget results (superseded)
        ├── degradation_curve.png
        ├── regression_results_v2.json   <- corrected native CV results
        └── degradation_curve_v2.png     <- degradation curve (native CV)
```

## Key Files (Corrected Exp2)
- `exp2_spatial_regression/scripts/run_regression_v2.py` — native CV runner
- `exp2_spatial_regression/scripts/plot_results_v2.py` — degradation curve plotter
- `exp2_spatial_regression/results/regression_results_v2.json` — full results
- `exp2_spatial_regression/results/degradation_curve_v2.png` — degradation curve plot
