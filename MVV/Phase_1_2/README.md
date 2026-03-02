# MVV Phase 1.2 — Structural Regression

**Core question:** Does SigLIP use horizontal and vertical whitespace as structural anchors?
At 256 tokens (224×224 px), individual characters are unreadable — only spatial patterns survive.
Can a linear probe count functions and classes from geometry alone?

---

## Experiments

### Exp1 — Mean-Pool, Windowed Labels, LinearRegression

**Status: ✅ DONE — Clean ablation isolating mean pooling as bottleneck**

Labels correctly windowed to the visible 40-line AST window (same labels as Exp2).
Only variable vs Exp2: **mean-pool features** instead of spatial max-pool.

| Budget | line_count R² | n_defs R² | n_classes R² |
|--------|:---:|:---:|:---:|
| 729 (train) | 0.970 | 0.833 | 0.855 |
| 441 | 0.921 | 0.720 | 0.773 |
| **256** | **0.855** | **0.364** | **0.568** |
| 121 | 0.041 | −0.479 | −0.405 |

**Finding:** Mean pooling is the architectural bottleneck — even with correct windowed labels,
n_defs and n_classes fail at 256 tokens. Fixing the labels alone is insufficient;
the feature representation must preserve spatial structure.

**Key files:** [exp1_structural_regression/results/regression_results.json](exp1_structural_regression/results/regression_results.json)

---

### Exp2 — Pool4x4/Pool8x8, Windowed Labels, PCA + Ridge

**Status: ✅ DONE — Differentiated result**

**Pipeline:** Windowed labels (only AST nodes in the 40-line visible window) → spatial max-pool features → PCA(1024, whiten, randomized) → Ridge(alpha=100)

| Budget | line_count | n_defs | n_classes | line_count | n_defs | n_classes |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| | **pool4x4** | **pool4x4** | **pool4x4** | **pool8x8** | **pool8x8** | **pool8x8** |
| 729 (train) | 0.973 | 0.851 | 0.862 | 0.980 | 0.866 | 0.888 |
| 441 | 0.909 | 0.758 | 0.784 | 0.892 | 0.771 | **0.816** |
| **256** | **0.867** | 0.461 | 0.675 | **0.877** | 0.371 | 0.670 |
| 121 | 0.559 | −0.138 | 0.110 | 0.661 | −0.325 | 0.081 |

Success threshold: R² ≥ 0.8 at 256 tokens.

**Results by target:**

| Target | Visual footprint | R²@256 (pool4x4) | Verdict |
|---|---|:---:|:---:|
| `line_count` | Global fill vs. bottom whitespace | **0.867** | ✅ PASS |
| `n_classes` | Large block (header + body, ~5–30 lines) | 0.675 | ⚠️ PARTIAL |
| `n_defs` | Small marker, ~1–3 lines | 0.461 | ❌ FAIL |

**Key findings:**
- **Coarse spatial geometry confirmed:** The encoder retains line density information through the 256-token compression. It can tell you whether the bottom of the image is blank.
- **Fine-grained structure counting fails:** Individual `def` boundaries are below the visual resolution floor at 256 tokens. The 441→256 drop for n_defs is −0.30 (pool4x4).
- **pool4x4 beats pool8x8 for n_defs at low res:** Coarser pooling is more robust to resolution degradation — the 8×8 grid tries to capture fine-grained features that have been blurred away.
- **121-token floor confirmed:** n_defs R² goes negative at 154×154px — the probe's predictions become anti-correlated with truth as the feature distribution shifts past the training domain.

**Key files:** [exp2_spatial_regression/results/regression_results.json](exp2_spatial_regression/results/regression_results.json)

---

## File Structure

```
Phase_1_2/
├── README.md
├── exp1_structural_regression/
│   ├── scripts/
│   │   ├── gen_labels.py       ← full-file AST counts (flawed baseline)
│   │   ├── run_regression.py   ← LinearRegression on mean-pool features
│   │   └── run_regression.sh
│   ├── data/labels.jsonl
│   └── results/
│       ├── regression_results.json
│       └── degradation_curve.png
└── exp2_spatial_regression/
    ├── scripts/
    │   ├── gen_labels.py       ← windowed counts (only visible 40-line window)
    │   ├── run_regression.py   ← PCA(1024) + Ridge on pool4x4/pool8x8 features
    │   └── run_regression.sh   ← teaching partition, 64G RAM
    ├── data/labels.jsonl
    └── results/
        ├── regression_results.json
        └── degradation_curve.png
```

## Running Exp2 (features already exist, no GPU needed)

```bash
cd ~/CoderOCR/OCR-Coder
sbatch MVV/Phase_1_2/exp2_spatial_regression/scripts/run_regression.sh
```

Features read from: `MVV/Phase_1_1/exp2_maxpool_comparison/data/features_maxpool/`

## Next: Phase 1.2 Exp3

**Question:** Is `n_defs` information nonlinearly encoded in the pool4x4 features at 256 tokens?

A Ridge probe (linear) gets R²=0.46 at 256 tokens. An MLP or k-NN probe on the same PCA(1024) embeddings would test whether the structural information exists but requires nonlinear decoding. If the nonlinear probe also fails, the information is genuinely absent — not just linearly inaccessible.
