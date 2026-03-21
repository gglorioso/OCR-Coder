# Phase 1.4 — Figures

## Contents

| File | Script | Description |
|------|--------|-------------|
| `aspect_ratio_distortion.png` | `gen_aspect_ratio_distortion.py` | Side-by-side showing how squashing a code image to a square (448×448) compresses the indentation staircase horizontally, destroying the structural signal. |
| `topology_survival.png` | `gen_topology_survival.py` | Side-by-side showing that downsampling to 112×112 (the lowest SigLIP token budget, 121 tokens) blurs fine text completely yet the spatial topology of the indentation zone is still visible at the same position. |

## Sample Image

Both figures use the same source image:

```
MVV/Phase_1_1/data_mvv/images/cpython__Lib__argparse_py.png
```

**Dimensions:** 800 × 800 px (grayscale, PIL mode `L`)

**Why this image?** CPython's `argparse.py` contains deeply nested
`if/elif/else` chains and class hierarchies, producing a clear indentation
staircase across 5+ nesting levels (0, 4, 8, 12, 16 spaces). This makes it
ideal for demonstrating how compression affects structural topology.

## Running the Scripts

From the repo root:

```bash
python MVV/Phase_1_4/figures/gen_aspect_ratio_distortion.py
python MVV/Phase_1_4/figures/gen_topology_survival.py
```

Dependencies: `Pillow`, `matplotlib`, `numpy` (no PyTorch required).
