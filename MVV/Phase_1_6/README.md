# MVV Phase 1.6 — Attention Probe with 2D Positional Encoding

**Core question:** Can a LLaVA-style MHA projector with explicit 2D spatial positional encoding
decode function-boundary geometry (`n_defs`) from pool8x8 tokens, and does it surpass a
fully-optimized linear baseline?

Phase 1.5 established that Method 2 pool8x8 tokens (64 × 1152D) carry structural code layout
information. Phase 1.6 tests whether a learned attention head can exploit the spatial arrangement
of those tokens — something a linear probe (Ridge R²=0.802) physically cannot access.

---

## Interpretation Key

| Outcome | Meaning |
|---|---|
| Exp B ≈ Exp A (no pos enc) | Spatial geometry not the bottleneck; content alone drives prediction |
| Exp B >> Exp A | Tokens encode spatial structure; 2D positional signal is critical for decoding |
| Exp B > Ridge ceiling | Nonlinear spatial decoding exceeds linear baseline — projector design validated |

---

## Experiments

### Experiment A — AttentionProbe (No Positional Encoding)

**Status: COMPLETE**

| Metric | Value |
|---|:---:|
| Mean R² (5-fold) | 0.7244 |
| Std R² | ±0.0067 |
| Fold scores | 0.713 / 0.721 / 0.729 / 0.732 / 0.726 |

Architecture:
- MLP adapter: `Linear(1152→2048) → GELU → Linear(2048→2048)`
- Attention: `MHA(embed_dim=2048, num_heads=16, batch_first=True)`
- Positional encoding: **none**
- Regressor: `Linear(2048→1)` on CLS token output

---

### Experiment B — AttentionProbeRoPE (2D SinCos Positional Encoding)

**Status: COMPLETE**

| Metric | Value |
|---|:---:|
| Mean R² (5-fold) | **0.8729** |
| Std R² | ±0.0141 |
| Fold scores | 0.847 / 0.873 / 0.873 / 0.886 / 0.885 |

Architecture:
- MLP adapter: `Linear(1152→2048) → GELU → Linear(2048→2048)`
- Positional encoding: 2D SinCos injected into 64 spatial tokens (Y: dims 0–1023, X: dims 1024–2047)
- Learnable CLS token (no positional encoding on CLS)
- Attention: `MHA(embed_dim=2048, num_heads=16, batch_first=True)`
- Regressor: `Linear(2048→1)` on CLS token output

---

## Summary: A vs B vs Baseline

| Probe | R² (n_defs) | Notes |
|---|:---:|---|
| Ridge (linear ceiling) | 0.802 | From Phase 1.1/1.2, pool8x8 |
| Exp A — MHA, no pos enc | 0.724 | Below Ridge; spatial context missing |
| **Exp B — MHA + 2D SinCos** | **0.873** | +0.071 above Ridge ceiling |

**Key finding:** 2D positional encoding is doing almost all the work (+0.149 R² vs Exp A).
The attention head with 2D SinCos locates `def` boundaries in the 8×8 token grid and attends
to those regions directly, exceeding what any linear decoder can achieve.

**Implication for projector design:** The LLaVA-style MHA projector should always include
explicit 2D spatial positional encodings when operating on pooled vision tokens.

---

## Shared Hyperparameters (both experiments)

| Param | Value |
|---|:---:|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Epochs | 20 |
| Batch size | 64 |
| Loss | MSE |
| CV folds | 5 |
| n_samples | 8,938 |

---

## Features & Data

- **Input features:** Method 2 pool8x8 from Phase 1.5 (`MVV/Phase_1_5/data/features/method2/pool8x8/`)
- **Shape per file:** `[73728]` fp16 → reshaped to `[64, 1152]`
- **Labels source:** Phase 1.4 labels (`MVV/Phase_1_4/data/labels/`)
- **Target:** `n_defs` (number of function definitions per file)

---

## Running

No SLURM needed — GPU optional, runs on CPU in ~10 minutes.

```bash
cd ~/CoderOCR/OCR-Coder

# Experiment A (no positional encoding)
python MVV/Phase_1_6/scripts/run_attention_probe.py

# Experiment B (2D SinCos positional encoding)
python MVV/Phase_1_6/scripts/run_attention_probe_rope.py
```

---

## File Structure

```
Phase_1_6/
├── README.md
├── scripts/
│   ├── run_attention_probe.py        ← Exp A: MHA, no pos enc
│   ├── run_attention.sh              ← SLURM wrapper for Exp A
│   ├── run_attention_probe_rope.py   ← Exp B: MHA + 2D SinCos
│   └── run_attention_rope.sh         ← SLURM wrapper for Exp B
└── results/
    ├── attention_probe_results.json       ← Exp A results
    └── attention_probe_rope_results.json  ← Exp B results
```
