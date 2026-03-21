#!/usr/bin/env python3
"""
gen_aspect_ratio_distortion.py
==============================
Figure: Aspect Ratio Distortion Destroys Indentation Structure

Shows a side-by-side comparison of:
  Left  — Original code image in grayscale at native (non-square) aspect ratio
  Right — Same image squashed to a square (448x448), destroying indentation

The source image is cropped from the full 800x800 MVV render to a
non-square 800x360 region (y=440..800 of argparse.py), which contains
a dense indentation staircase across 8 nesting levels (40→400px indent).
This lower half of argparse.py shows the deeply nested _ArgumentGroup /
_ActionsContainer if/elif/else chains — the richest staircase in the file.

Usage (from repo root):
    python MVV/Phase_1_4/figures/gen_aspect_ratio_distortion.py
"""

import sys
from pathlib import Path

from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Source image — cpython argparse.py (deeply nested if/else chains; rich
# indentation staircase across multiple nesting levels: 0→4→8→12→16 spaces)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_IMAGE = (
    REPO_ROOT
    / "MVV"
    / "Phase_1_1"
    / "data_mvv"
    / "images"
    / "cpython__Lib__argparse_py.png"
)

SQUARE_SIZE = 400   # target size for the squashed panel (400×400)
OUTPUT_PATH = Path(__file__).parent / "aspect_ratio_distortion.png"
DPI = 150


def load_and_prepare(path: Path):
    """
    Load the 800x800 grayscale MVV image and crop to the indentation-rich
    lower section: rows 440–800 (360px tall), giving an 800×360 region
    with ratio ≈ 2.22:1.

    This crop targets the densely nested portion of argparse.py which
    contains _ArgumentGroup / _ActionsContainer if/elif/else chains,
    spanning 8 distinct indent levels (40 → 400 px) — visible as a
    clear staircase in the left margin.

    Returns: (original_crop: PIL.Image, width, height)
    """
    img = Image.open(path).convert("L")  # ensure grayscale
    w, h = img.size                       # 800 x 800
    # Crop: full width, rows 440..800  → 800 x 360  (≈2.22:1 wide:tall)
    crop_y1, crop_y2 = 440, 800
    original = img.crop((0, crop_y1, w, crop_y2))
    crop_h = crop_y2 - crop_y1          # 360
    return original, w, crop_h


def make_squashed(original: Image.Image) -> Image.Image:
    """Squash 800×360 original to 400×400 using LANCZOS — destroys aspect ratio."""
    return original.resize((SQUARE_SIZE, SQUARE_SIZE), Image.LANCZOS)


def find_staircase_x(original: Image.Image) -> int:
    """
    Return approximate x-pixel where the deepest indentation staircase
    begins (used to place the annotation arrow on the right panel).

    Strategy: find the column where, across multiple rows in the lower half
    of the image, the pixel transitions from white background to text — i.e.
    the leftmost text column on the most-indented lines.
    """
    arr = np.array(original)
    h, w = arr.shape
    # Sample lines from the bottom half where deep nesting occurs
    max_leading = 0
    for row_i in range(h // 2, h, 20):
        if row_i >= h:
            break
        row = arr[row_i]
        leading = int(np.argmax(row < 200)) if np.any(row < 200) else w
        if leading > max_leading:
            max_leading = leading
    # The staircase starts around the deepest indent level we found
    return max_leading


def main():
    if not SOURCE_IMAGE.exists():
        print(f"ERROR: source image not found:\n  {SOURCE_IMAGE}", file=sys.stderr)
        print("Run from the repo root or ensure Phase_1_1 data exists.", file=sys.stderr)
        sys.exit(1)

    original, orig_w, orig_h = load_and_prepare(SOURCE_IMAGE)
    squashed = make_squashed(original)

    staircase_x = find_staircase_x(original)
    # Map staircase_x from original coords into squashed coords
    staircase_x_sq = int(staircase_x * SQUARE_SIZE / orig_w)

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 5.5),
        facecolor="white",
        gridspec_kw={"width_ratios": [orig_w / orig_h, 1.0]},
    )
    fig.subplots_adjust(top=0.84, bottom=0.04, left=0.04, right=0.96, wspace=0.14)

    fig.suptitle(
        "Aspect Ratio Distortion Destroys Indentation Structure",
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )

    # ── Left panel: original crop at native aspect ratio ──────────────────
    ax_orig = axes[0]
    ax_orig.imshow(np.array(original), cmap="gray", vmin=0, vmax=255,
                   aspect="equal", interpolation="nearest")
    ax_orig.set_title(f"Original\n(800×{orig_h}, {orig_w/orig_h:.2f}:1 ratio)", fontsize=11, pad=6)
    ax_orig.axis("off")

    # Dimension annotation
    ax_orig.annotate(
        f"{orig_w} × {orig_h} px",
        xy=(0.5, -0.03),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=8,
        color="#555555",
    )

    # ── Right panel: squashed to square ───────────────────────────────────
    ax_sq = axes[1]
    ax_sq.imshow(np.array(squashed), cmap="gray", vmin=0, vmax=255,
                 aspect="equal", interpolation="lanczos")
    ax_sq.set_title(
        "Squashed to Square\n(400×400, distorted)",
        fontsize=11,
        pad=6,
    )
    ax_sq.axis("off")

    # Dimension annotation
    ax_sq.annotate(
        f"{SQUARE_SIZE} × {SQUARE_SIZE} px",
        xy=(0.5, -0.03),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=8,
        color="#555555",
    )

    # Annotation: "Staircase\ncompressed" in red on the right panel
    # Place near the deepest indentation zone (right side of left margin)
    annot_x = max(staircase_x_sq - 20, 10)
    annot_y = SQUARE_SIZE * 0.55  # lower half where deep nesting lives
    ax_sq.annotate(
        "Staircase\ncompressed",
        xy=(annot_x, annot_y),
        xytext=(annot_x + 80, annot_y - 80),
        fontsize=9,
        color="red",
        fontweight="bold",
        ha="center",
        va="center",
        arrowprops=dict(
            arrowstyle="->",
            color="red",
            lw=1.8,
        ),
    )

    # Thin border around each panel for clarity
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#cccccc")
            spine.set_linewidth(0.8)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
