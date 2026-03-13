#!/usr/bin/env python3
"""
label_generator_1_9.py — Phase 1.9: per-patch keyword multi-hot label generation

For each rendered code image, parses the corresponding Python source file with
the tokenize module and maps keyword token positions to the 32×32 SigLIP patch
grid, producing a [1024, 16] uint8 multi-hot label tensor.

Rendering assumptions (matching image generation pipeline):
  canvas_size  = 800 px
  line_height  = 20 px
  char_width   = 10 px
  output_size  = 448 px  (SigLIP input)
  grid_side    = 32      (32×32 patch grid)
  patch_px     = 14.0    (448 / 32)

Keywords detected (index order is fixed):
  0:def  1:class  2:import  3:return  4:if   5:for    6:while  7:else
  8:elif 9:try   10:except 11:with   12:pass 13:yield 14:lambda 15:raise

Output: MVV/Phase_1_9/data/labels/<stem>.pt  — shape [1024, 16] torch.uint8
Also writes: MVV/Phase_1_9/data/ground_truth.jsonl

Usage:
    python label_generator_1_9.py [options]
"""

import argparse
import io
import json
import tokenize
from pathlib import Path

import torch
from tqdm import tqdm


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parents[2]

KEYWORDS = [
    'def', 'class', 'import', 'return',
    'if',  'for',   'while',  'else',
    'elif','try',   'except', 'with',
    'pass','yield', 'lambda', 'raise',
]
VOCAB_SIZE = len(KEYWORDS)          # 16
KW_INDEX   = {kw: i for i, kw in enumerate(KEYWORDS)}

CANVAS_SIZE  = 800
LINE_HEIGHT  = 20
CHAR_WIDTH   = 10
OUTPUT_SIZE  = 448
GRID_SIDE    = 32
MAX_LINES    = CANVAS_SIZE // LINE_HEIGHT   # 40
SCALE        = OUTPUT_SIZE / CANVAS_SIZE    # 0.56
PATCH_PX     = OUTPUT_SIZE / GRID_SIDE      # 14.0


def token_to_patch(lineno: int, col: int, anchor_line: int):
    """
    Map a (1-indexed lineno, 0-indexed col) source position to a patch index
    in the 32×32 grid. Returns None if the position is outside the rendered
    window.
    """
    eff_line = (lineno - 1) - anchor_line
    if eff_line < 0 or eff_line >= MAX_LINES:
        return None

    pixel_y = eff_line * LINE_HEIGHT * SCALE
    pixel_x = col      * CHAR_WIDTH  * SCALE

    patch_row = int(pixel_y // PATCH_PX)
    patch_col = int(pixel_x // PATCH_PX)

    patch_row = max(0, min(GRID_SIDE - 1, patch_row))
    patch_col = max(0, min(GRID_SIDE - 1, patch_col))

    return patch_row * GRID_SIDE + patch_col


def generate_labels(source_path: Path, anchor_line: int) -> torch.Tensor:
    """
    Parse source_path and return [1024, 16] uint8 multi-hot keyword label tensor.
    """
    label = torch.zeros(GRID_SIDE * GRID_SIDE, VOCAB_SIZE, dtype=torch.uint8)

    try:
        source = source_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return label

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenError:
        return label

    for tok in tokens:
        if tok.type != tokenize.NAME:
            continue
        kw_idx = KW_INDEX.get(tok.string)
        if kw_idx is None:
            continue
        lineno, col = tok.start   # lineno is 1-indexed
        patch_idx = token_to_patch(lineno, col, anchor_line)
        if patch_idx is not None:
            label[patch_idx, kw_idx] = 1

    return label


def main():
    p = argparse.ArgumentParser(description="Phase 1.9 keyword label generation")
    p.add_argument("--data-dir",  default=str(_REPO_ROOT / "MVV" / "Phase_1_1" / "data_mvv"))
    p.add_argument("--repos-dir", default=str(_REPO_ROOT / "Scraped Repos"))
    p.add_argument("--feat-dir",  default=str(_REPO_ROOT / "MVV" / "Phase_1_9" / "data" / "features"))
    p.add_argument("--out-dir",   default=str(_REPO_ROOT / "MVV" / "Phase_1_9" / "data" / "labels"))
    args = p.parse_args()

    data_dir  = Path(args.data_dir)
    repos_dir = Path(args.repos_dir)
    feat_dir  = Path(args.feat_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_path = _REPO_ROOT / "MVV" / "Phase_1_9" / "data" / "ground_truth.jsonl"

    print("=" * 65)
    print("MVV Phase 1.9 — Keyword Label Generation")
    print(f"  Manifest : {data_dir / 'manifest.jsonl'}")
    print(f"  Repos    : {repos_dir}")
    print(f"  Features : {feat_dir}")
    print(f"  Labels   : {out_dir}")
    print(f"  GT JSONL : {gt_path}")
    print("=" * 65)

    entries = []
    with open(data_dir / "manifest.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    n_processed = n_skip_idempotent = n_skip_missing = 0
    kw_freq = [0] * VOCAB_SIZE
    gt_records = []

    for entry in tqdm(entries, desc="Labelling"):
        stem = Path(entry["image"]).stem
        source_file  = entry.get("source_file", "")
        anchor_line  = entry.get("anchor_line", 0)

        label_path = out_dir / f"{stem}.pt"
        feat_path  = feat_dir / f"{stem}.pt"

        if label_path.exists():
            n_skip_idempotent += 1
            # Still include in ground_truth if feat exists
            if feat_path.exists():
                gt_records.append({"stem": stem,
                                   "source_file": source_file,
                                   "anchor_line": anchor_line})
            continue

        src_path = repos_dir / source_file
        if not src_path.exists():
            n_skip_missing += 1
            continue

        label = generate_labels(src_path, anchor_line)
        torch.save(label, label_path)

        for kw_i in range(VOCAB_SIZE):
            kw_freq[kw_i] += int(label[:, kw_i].any())

        n_processed += 1

        if feat_path.exists():
            gt_records.append({"stem": stem,
                               "source_file": source_file,
                               "anchor_line": anchor_line})

    # Write ground_truth.jsonl
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gt_path, "w") as f:
        for rec in gt_records:
            f.write(json.dumps(rec) + "\n")

    print("\n" + "=" * 65)
    print("DONE")
    print(f"  Processed        : {n_processed:,}")
    print(f"  Skipped (exists) : {n_skip_idempotent:,}")
    print(f"  Skipped (no src) : {n_skip_missing:,}")
    print(f"  GT records       : {len(gt_records):,}  → {gt_path}")
    print()
    print("  Keyword frequency (images with ≥1 occurrence):")
    for kw, freq in zip(KEYWORDS, kw_freq):
        print(f"    {kw:<10s} {freq:5,}")
    print("=" * 65)


if __name__ == "__main__":
    main()
