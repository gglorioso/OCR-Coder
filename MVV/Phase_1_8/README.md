# Phase 1.8 — AST-to-Patch-Grid Coordinate Mapping

## Purpose

Phase 1.8 produces **ground truth** for contrastive retrieval experiments: for every
Python function and class rendered in a Phase 1.1 MVV image, it records which rows
of the 8×8 SigLIP patch grid that node occupies.

This mapping enables:
- Supervised contrastive probing (does patch row _r_ activate more for functions vs.
  classes starting in that row?)
- Retrieval baselines (given a function name, predict the patch rows that should
  respond to a code-change query)
- Ablations over different grid sizes and scale factors

---

## Coordinate math

### Rendering pipeline

The original canvas is **800×800 px** (40 lines × 20 px/line). SigLIP rescales it to
**448×448 px** before extracting patch features.

```
Scale factor = 448 / 800 = 0.56
```

### From source line to patch-grid row

```
# 1. Effective line in rendered window (0-indexed)
effective_line = (ast_lineno - 1) - anchor_line
                 # ast_lineno is 1-indexed; anchor_line is the chunk offset

# 2. Canvas-space pixel bounds (800-px space)
y_start_800 = effective_start_line * 20
y_end_800   = effective_end_line   * 20 + 20     # exclusive bottom edge

# 3. Scale to SigLIP input space (448-px)
y_start_448 = y_start_800 * 0.56
y_end_448   = y_end_800   * 0.56

# 4. Map to patch-grid row
patch_height = 448 / 8 = 56 px
row_start = int(y_start_448 // 56)    # clamped to [0, 7]
row_end   = int((y_end_448 - 1) // 56)

grid_rows = list(range(row_start, row_end + 1))
```

**Example:** a function on lines 42–58 with `anchor_line=10`:
- effective_start = 41 − 10 = 31, effective_end = 57 − 10 = 47 → clamped to 39
- y_start_800 = 620, y_end_800 = 800
- y_start_448 = 347.2, y_end_448 = 448
- row_start = 6, row_end = 7 → `grid_rows = [6, 7]`

---

## Scripts

### `scripts/ast_extractor.py`

Walks a directory of Python source files, parses the AST of each file, and emits
one JSONL record per discovered `def` / `class` node with its patch-grid rows.

```bash
python scripts/ast_extractor.py \
    --py-dir    /path/to/scraped/repos \
    --manifest  MVV/Phase_1_1/data_mvv/manifest.jsonl \
    --out-path  MVV/Phase_1_8/data/ground_truth/ground_truth.jsonl \
    --line-height  20 \
    --canvas-size  800 \
    --output-size  448 \
    --grid-size    8
```

**Key flags:**

| Flag | Default | Meaning |
|------|---------|---------|
| `--py-dir` | (required) | Root directory of Python source files |
| `--manifest` | None | Phase 1.1 JSONL with `anchor_line` per file; if omitted, anchor=0 for all |
| `--out-path` | (required) | Output JSONL path |
| `--line-height` | 20 | px per line in 800-px canvas space |
| `--canvas-size` | 800 | Canvas side length before downscaling |
| `--output-size` | 448 | SigLIP input side length after downscaling |
| `--grid-size` | 8 | Patch-grid size (8 → 8×8 = 64 patches) |
| `--max-rows` | 40 | Lines per rendered image |

**Output record schema:**
```json
{
  "file":       "black__src__black__lines_py",
  "type":       "def",
  "name":       "calculate_loss",
  "lineno":     42,
  "end_lineno": 58,
  "anchor_line": 10,
  "grid_rows":  [6, 7]
}
```

Nodes outside the rendered window (`effective_start >= max_rows`) are skipped and
counted in the summary.

---

### `scripts/visual_validator.py`

Reads the JSONL produced by `ast_extractor.py`, finds the matching rendered image
for each file, and saves an annotated copy with:

- A faint 8×8 gray grid overlay so patch boundaries are visible
- A semi-transparent coloured band for each patch-grid row a node occupies
- A text label in each band: `def <name>  (L<start>–<end>)`

```bash
python scripts/visual_validator.py \
    --ground-truth MVV/Phase_1_8/data/ground_truth/ground_truth.jsonl \
    --image-dir    MVV/Phase_1_1/data_mvv/images \
    --out-dir      MVV/Phase_1_8/data/validation_checks \
    --max-samples  50 \
    --node-type    all
```

**Key flags:**

| Flag | Default | Meaning |
|------|---------|---------|
| `--ground-truth` | (required) | JSONL from ast_extractor.py |
| `--image-dir` | (required) | Directory of rendered `.png` images |
| `--out-dir` | (required) | Where to save annotated images |
| `--max-samples` | 50 | Cap on number of images to annotate |
| `--node-type` | `all` | Filter: `all`, `def`, or `class` |
| `--output-size` | 448 | Expected image side length (images are resized if needed) |
| `--grid-size` | 8 | Must match the value used in ast_extractor.py |

---

## What to look for in validation images

A correct mapping looks like:

- The coloured band covers the **vertical extent of the function/class body** in
  the rendered image, aligned to patch-row boundaries (±½ patch row is expected
  because the grid snaps to 56-px boundaries).
- The top of a function at line 0 of the window should fall in grid row 0
  (band at the very top).
- A function that spans lines 35–39 (the last five lines of a 40-line window)
  should highlight grid rows 6–7 (the bottom two patch rows).
- Multiple nodes whose lines overlap the same patch row will each draw a band
  in a different colour; stacked labels confirm which node is which.

**Common failure modes to watch for:**

| Symptom | Likely cause |
|---------|-------------|
| Band is one or two patch rows too high | `anchor_line` is wrong (manifest mismatch) |
| Band covers the whole image | `end_lineno` is unreliable (Python < 3.8 fallback) |
| No bands on images with obvious functions | Node type filter or file-stem mismatch |
| Band extends below the visible code | Function body extends beyond the 40-line window |

---

---

## Contrastive Adapter Training

Phase 1.8 also contains the first end-to-end contrastive adapter trained on this ground truth.

### Architecture

```
Text query ("def function_name(")
    → SigLIP Text Encoder (frozen, 1152D) → text_embeddings.pt

Vision features (pool8x8, [64, 1152])
    → 2D RoPE injection (8×8 grid coordinates)
    → ContrastiveAdapter MLP: Linear(1152,1152) → GELU → Linear(1152,1152)
    → [64, 1152] adapted tokens

Loss: BCEWithLogitsLoss — dot(text, token_i) → 1 for tokens in grid_rows, 0 otherwise
```

### Training Results

| Metric | Epoch 1 | Best (Ep 7) | Final (Ep 30) | Target |
|---|---|---|---|---|
| val_loss | 0.586 | 0.548 | 0.559 | ↓ |
| val_pos_sim | 0.425 | 0.498 | 0.488 | ↑ |
| val_neg_sim | 0.359 | 0.356 | 0.351 | ↓ |
| **val_gap** | 0.066 | **0.141** | 0.138 | **> 0.3** |

**Conclusion:** The adapter learns a genuine positive/negative separation (gap grows from 0.066 → 0.141) but plateaus at roughly half the target. Val_loss begins rising after epoch 7 while train_loss continues falling — mild overfitting. Root cause: function signatures alone (`"def calculate_loss("`) are too low-information to strongly predict spatial location. Text queries need richer semantic content (docstrings) or the objective needs redesigning. Temperature scaling and docstring-augmented queries are the planned Phase 1.9 improvements.

### Scripts

| Script | Purpose |
|---|---|
| `precompute_text_embeddings.py` | Encodes all unique function signatures via SigLIP text encoder → `text_embeddings.pt` |
| `dataset_1_8.py` | PyTorch Dataset: loads vision `.pt` + text embeddings, generates 64-dim multi-hot target masks |
| `model_1_8.py` | `ContrastiveAdapter` nn.Module with 2D RoPE + MLP |
| `train_1_8.py` | Training loop: dot-product similarity map, BCE spatial loss, val metrics |
| `run_phase_1_8.sh` | SLURM job (teaching partition): runs precompute → train sequentially |

---

## Data layout

```
MVV/Phase_1_8/
    scripts/
        ast_extractor.py              # AST parse + pixel math
        visual_validator.py           # Pillow overlay drawing
        precompute_text_embeddings.py # SigLIP text encoder pass
        dataset_1_8.py                # PyTorch Dataset
        model_1_8.py                  # ContrastiveAdapter (2D RoPE + MLP)
        train_1_8.py                  # Training loop
        run_phase_1_8.sh              # SLURM launcher
    data/
        ground_truth/         # ground_truth.jsonl (36,673 labeled nodes)
        text_embeddings/      # text_embeddings.pt — 22,511 × 1152 (gitignored)
        validation_checks/    # Annotated validation PNGs (gitignored)
    checkpoints/              # best.pt (gitignored)
    README.md
```
