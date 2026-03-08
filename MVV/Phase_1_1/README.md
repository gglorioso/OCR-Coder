# MVV Phase 1.1 — Repository Classification Probes

**Status:** Complete (corrected 2026-03-05)

## Overview

**Test:** Can a frozen SigLIP vision encoder distinguish which Python source
file an image came from — purely from spatial structure, with no colour or
syntax highlighting?

**Why this matters:** Every file is Python, so keywords, syntax, and language
are identical across all 7,418 classes. The only discriminating signal is the
*spatial geometry* of each file: its indentation depth, block lengths,
whitespace gaps, and line density. If the encoder can fingerprint a file from
visual structure alone, it has a genuine "mental model" of code topology —
independent of the words.

---

## Experiments

### Exp1 — Mean-Pool Repository Probe

- **Method:** Mean-pool SigLIP features (1280D), LogisticRegression, cross-budget train/test (train @ 729, test @ 441/256/121)
- **Results:** Repo probe 76.3%@729 → 28.1%@121; knee at 256→121 tokens
- **Finding:** Strong repo signal at high res; collapses below 256 tokens. Per-file probe is null (cos sim=0.94 — probe collapses to mean).
- **Key files:** `exp1_meanpool_probe/scripts/run_probe.py`, `exp1_meanpool_probe/results/repo_probe_results.json`

### Exp2 — Spatial Max-Pool Comparison (Corrected 2026-03-05)

- **Method:** pool4x4 / pool8x8 / meanpool features, PCA(1024) dimensionality equalization, LogisticRegression(class_weight="balanced"), **native 5-fold stratified CV per budget** (eliminates cross-budget domain shift confound)
- **Dataset:** 8,980 samples, 15 repos, clean 800×800 MVV images
- **Results (balanced accuracy):**

| Pool | tok=729 | tok=441 | tok=256 | tok=121 |
|---|---|---|---|---|
| meanpool | **74.8%** | **72.5%** | **65.3%** | 43.4% |
| pool8x8 | 71.0% | **72.1%** | 64.8% | **45.8%** |
| pool4x4 | 68.3% | 67.3% | 61.2% | 40.0% |

- **Key finding:** meanpool wins at high resolution (729 tokens); pool8x8 is competitive at mid-range (441) and marginally better at the resolution floor (121 tokens). pool4x4 is consistently worst — wrong spatial granularity.
- **Correction note:** Original Exp2 result (pool8x8 69.8% vs meanpool 76.3%) was a curse-of-dimensionality artifact — pool8x8 had 73,728 dims vs meanpool's 1,152 with only ~9K samples. PCA(1024) equalization reverses the result. Cross-budget train/test also introduced domain shift confound (Phase 1.3 confirmed cos_sim=0.220 across budgets). Native CV eliminates both confounds.
- **Key files:**
  - `exp2_maxpool_comparison/scripts/run_probe_v2.py` — native CV + PCA + balanced probe runner
  - `exp2_maxpool_comparison/scripts/plot_probe_v2.py` — degradation curve plotter
  - `exp2_maxpool_comparison/results/probe_results_v2_balanced.json` — full results
  - `exp2_maxpool_comparison/results/probe_degradation_v2.png` — degradation curve plot

---

## Dataset: `data_mvv/`

**1 file → 1 class → 1 image.** 15 repos (black, click, cpython, django, fastapi, flask, httpx, numpy, pandas, poetry, pydantic, pytorch, requests, scikit-learn, transformers).

**Canvas spec:** Font DejaVu Sans Mono 16px, 10px char width, 20px line height, 80-char hard truncate, 40 lines per image, 800×800 px, PIL `L` (8-bit grayscale).

**AST-anchored start:** The 40-line window skips leading license/docstring headers, anchoring to the first real structural node (`import`, `class`, or `def`).

---

## Token Budget Sweep

| Budget (tokens) | Grid | Pixel dims | px / line | Perception state |
|---|---|---|---|---|
| **729** | 27×27 | 378×378 | 9.4 px | Readable — text and structure both visible |
| **441** | 21×21 | 294×294 | 7.3 px | Transition — semantic cliff edge |
| **256** | 16×16 | 224×224 | 5.6 px | Unreadable — structural blocks visible, text dead |
| **121** | 11×11 | 154×154 | 3.8 px | Topology floor — sub-symbolic noise |
