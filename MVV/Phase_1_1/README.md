# Phase 1.1 — Source-File Topological Fingerprinting

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

## The Resolution-as-Test Paradigm

This is **not** a standard train/test split. We isolate a single variable:
**pixel resolution**.

| Step | What happens |
|---|---|
| **Train** | Feed 729-token (378×378 px) images through SigLIP → extract [1280] vectors → train logistic regression to map vectors to 7,418 file IDs |
| **Test** | Feed the *exact same images* at 441, 256, and 121 tokens → run the frozen probe → measure accuracy drop |

The probe never sees new files. The only thing that changes between train and
test is how many pixels SigLIP receives. When accuracy drops, it is proof that
the feature vector collapsed due to spatial resolution loss — nothing else.

This is equivalent to giving a patient a vision test: you don't show them a
different chart to measure blindness; you show them the same chart from
further away.

---

## Dataset: `data_mvv/`

**1 file → 1 class → 1 image.** Perfect class balance, zero imbalance.

### Image Generation

Script: [`gen_mvv_images.py`](gen_mvv_images.py)
SLURM:  [`gen_mvv_images.sh`](gen_mvv_images.sh)

**Canvas spec:**

| Parameter | Value |
|---|---|
| Font | DejaVu Sans Mono, 16px |
| Char width | 10px |
| Line height | 20px (forced) |
| Max columns | 80 chars (hard truncate, no wrapping) |
| Lines per image | 40 |
| Canvas | 800 × 800 px |
| Mode | PIL `L` (8-bit grayscale, no colour) |

**AST-anchored start:** The 40-line window does not blindly start at line 0.
The script uses Python's `ast` module to skip any leading license header or
module docstring, anchoring instead to the first real structural node
(`import`, `class`, or `def`). This ensures every image starts with
high-density structural logic rather than a flat grey rectangle of comment text.

**Repos (15):** black, click, cpython, django, fastapi, flask, httpx, numpy,
pandas, poetry, pydantic, pytorch, requests, scikit-learn, transformers

**Total images:** ~7,400 (one per valid, non-trivial `.py` file)

---

## Token Budget Sweep

SigLIP-SO400M uses 14×14 px patches. Square grids map to these exact targets:

| Budget (tokens) | Grid | Pixel dims | px / line | Perception state |
|---|---|---|---|---|
| **729** | 27×27 | 378×378 | 9.4 px | Readable — text and structure both visible |
| **441** | 21×21 | 294×294 | 7.3 px | Transition — semantic cliff edge |
| **256** | 16×16 | 224×224 | 5.6 px | Unreadable — structural blocks visible, text dead |
| **121** | 11×11 | 154×154 | 3.8 px | Topology floor — sub-symbolic noise |

The resulting **degradation curve** (accuracy vs token budget) reveals the
exact knee where the encoder loses its structural map of the file.

---

## Feature Extraction

Script: [`extract_mvv_features.py`](extract_mvv_features.py) *(to be written)*
SLURM:  [`extract_mvv_features.sh`](extract_mvv_features.sh) *(to be written)*

For each token budget:
1. Bicubic-downsample the 800×800 PNG to target pixel dims
2. Pass through frozen SigLIP-SO400M
3. Extract patch tokens `h ∈ R^(N×1280)`
4. Mean-pool → `x ∈ R^1280`
5. Save to `data_mvv/features/budget_{N}/`

---

## Linear Probe

Script: [`run_probe.py`](run_probe.py) *(to be written)*

- Model: `LogisticRegression` (lbfgs, no hidden layers) — intentionally
  linear so accuracy reflects encoder quality, not probe capacity
- Train: 729-token feature vectors
- Test: 441 / 256 / 121-token feature vectors
- Metrics: Top-1 accuracy, Top-5 accuracy, random baseline (1/7418 ≈ 0.01%),
  lift over random
- Output: degradation curve plot + results JSON

---

## Success Criteria

| Metric | Target |
|---|---|
| Top-1 @ 729 tokens | > 50% (encoder memorises structure at high res) |
| Top-1 @ 256 tokens | Measurable drop (structural signal degrading) |
| Top-1 @ 121 tokens | Near-random (encoder blind at topology floor) |
| Knee location | Identifiable — sharp drop between two adjacent budgets |
