# MVV Phase 1.7 — Visual Enhancement Study

## Overview

Phase 1.7 evaluates whether adding visual enhancements to rendered code images improves the
structural information encoded by SigLIP features. Three rendering variants are compared against
the Phase 1.6 baseline (plain images, no enhancements) using the same probe architecture and
task: predicting `n_defs` (number of function/class definitions) from a 40-line code image.

All other variables are held constant — same model, same feature extraction pipeline, same probe,
same dataset.

## Experiment Design

### Rendering Variants

Images are 800×800 px, DejaVu Sans Mono 16px, 40 lines × 20 px/line. All variants use Monokai
dark background (`#272822`) and syntax highlighting via Pygments.

| Experiment | Rendering Flags | Description |
|------------|-----------------|-------------|
| **Exp A** | `--syntax-highlighting` | Monokai color-coded tokens only |
| **Exp B** | `--syntax-highlighting --line-numbers` | Adds a 40px left margin with line numbers and a vertical separator at x=43 |
| **Exp C** | `--syntax-highlighting --line-numbers --indent-guides` | Adds 1px vertical guides at every 4-space indent level |

Source files: Python files from scraped repos, filtered (no tests, no `__init__.py`, no vendored
code). Each image shows up to 40 lines starting at the first structural node (import/class/def),
skipping leading docstrings and comments.

### Feature Extraction

- Model: `google/siglip-so400m-patch14-384` (frozen, fp16)
- Input: 800×800 PNG resized to 448×448, converted to grayscale-as-RGB, normalized to [-1, 1]
- Extraction: Method 2 pool8x8
  - SigLIP produces a 32×32 patch grid (1024 tokens, 1152D each)
  - `avg_pool2d(kernel=2, stride=2)` → 16×16
  - `adaptive_max_pool2d(8×8)` → 64 tokens × 1152D = 73,728D fp16 per image
- Output: one `.pt` file per image in `data/features/exp_{A,B,C}/pool8x8/`

### Probe Architecture (AttentionProbeRoPE)

Identical to Phase 1.6 Experiment B — no changes to the probe between phases.

```
Input: [B, 64, 1152]
  MLP adapter: Linear(1152→2048) → GELU → Linear(2048→2048)
  2D SinCos positional encoding (8×8 grid; Y-axis: dims 0–1023, X-axis: dims 1024–2047)
  Prepend learnable CLS token → [B, 65, 2048]
  MHA(embed_dim=2048, num_heads=16)
  CLS output → Linear(2048→1)
```

Target: `n_defs` (count of `def`/`class` statements visible in the rendered 40-line window).

### Training Protocol

- 5-fold cross-validation, `KFold(shuffle=True, random_state=42)`
- Optimizer: AdamW, lr=1e-4
- Loss: MSE
- Epochs: 20 per fold
- Batch size: 64
- Dataset: 8,938 aligned samples (inner join of features with `Phase_1_2` labels)

## Results

### Per-Fold R² Scores

| Fold | Phase 1.6 Baseline | Exp A (Syntax) | Exp B (+LineNum) | Exp C (+Guides) |
|------|--------------------|----------------|------------------|-----------------|
| 1    | 0.8470             | 0.8306         | 0.8369           | 0.8630          |
| 2    | 0.8733             | 0.8396         | 0.8604           | 0.8689          |
| 3    | 0.8729             | 0.8485         | 0.8800           | 0.8559          |
| 4    | 0.8865             | 0.8720         | 0.8750           | 0.8632          |
| 5    | 0.8848             | 0.8691         | 0.8817           | 0.8714          |

### Summary

| Experiment | Mean R² | Std R² | Delta vs. Phase 1.6 |
|------------|---------|--------|---------------------|
| Phase 1.6 baseline | **0.8729** | 0.0141 | — |
| Exp A: Syntax only | 0.8519 | 0.0162 | -0.0210 |
| Exp B: +Line numbers | 0.8668 | 0.0167 | -0.0061 |
| Exp C: +Line numbers +Guides | 0.8645 | 0.0054 | -0.0084 |

## Interpretation

**None of the visual enhancements improve over the Phase 1.6 baseline.** All three variants score
below the baseline (0.8729 R²), with Exp A showing the largest regression (-0.021).

Key observations:

1. **Syntax highlighting alone (Exp A) hurts.** The colored images are slightly harder for SigLIP
   to read than the Phase 1.6 rendering, likely because the image content meaningfully changes
   but SigLIP was pretrained on natural images (not colorized code). The grayscale normalization
   step in feature extraction partially collapses the color signal before it reaches SigLIP.

2. **Line numbers partially recover performance (Exp B).** Adding explicit positional structure
   in the image brings Exp B to within 0.006 R² of the baseline. The spatial cue of the left
   margin may help the 2D positional encoding in the probe anchor spatial tokens more accurately.

3. **Indent guides offer no additional benefit (Exp C).** Exp C matches Exp B closely
   (~0.8645 vs ~0.8668) and halves the variance across folds (std 0.0054 vs 0.0167). The guides
   likely add low-signal 1px vertical lines that neither help nor harm average performance but
   do reduce fold-to-fold variance.

4. **The probe's 2D SinCos positional encoding is the primary spatial reasoning mechanism.** The
   Phase 1.6 baseline (plain monochrome images, no margin decoration) already achieved 0.8729 R².
   Visual chrome in the image is redundant with — or slightly degrades — the structural signal
   already captured by SigLIP patches and recovered by the attention probe.

**Conclusion:** Visual enhancements to the rendered image are not a productive direction for
improving SigLIP feature quality on code. The next phases should focus on architecture or
training changes rather than renderer improvements.

## File Structure

```
Phase_1_7/
  scripts/
    render_enhanced.py              # PIL-based renderer with --syntax-highlighting,
                                    #   --line-numbers, --indent-guides flags
    extract_features_1_7.py         # SigLIP Method 2 pool8x8 feature extractor
    run_attention_probe_rope_1_7.py  # AttentionProbeRoPE 5-fold CV trainer/evaluator
    run_phase_1_7.sh                 # SLURM batch script orchestrating all three experiments
  images/
    exp_A_syntax_only/              # Rendered PNGs for Exp A
    exp_B_syntax_linenum/           # Rendered PNGs for Exp B
    exp_C_syntax_linenum_guides/    # Rendered PNGs for Exp C
  data/features/
    exp_A/pool8x8/                  # 73728D fp16 .pt files for Exp A
    exp_B/pool8x8/                  # 73728D fp16 .pt files for Exp B
    exp_C/pool8x8/                  # 73728D fp16 .pt files for Exp C
  results/
    exp_A_results.json
    exp_B_results.json
    exp_C_results.json
```

## Cluster Config

- Partition: `teaching`, 1 GPU, 8 CPUs/GPU, 48 GB RAM, 8h wall time
- Submit: `sbatch MVV/Phase_1_7/scripts/run_phase_1_7.sh`
- Labels source: `MVV/Phase_1_2/exp2_spatial_regression/data/labels.jsonl`
