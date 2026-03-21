#!/usr/bin/env python3
"""
gen_topology_survival.py
========================
Figure: Spatial Topology Survives Extreme Compression

Shows that downsampling to 112x112 (the lowest SigLIP token budget tested,
corresponding to 121 tokens at patch size 14) preserves the spatial structure
of indentation even though all fine detail is lost.

Layout: 2 panels side by side
  Left  — Full resolution 800x800 with red staircase underlines marking each
           indent level along the left margin
  Right — Same image downsampled to 112x112 then upscaled back to 800x800
           (nearest-neighbor) to show pixelation, with the same staircase
           underlines

Usage (from repo root):
    python MVV/Phase_1_4/figures/gen_topology_survival.py
"""

import sys
from pathlib import Path

from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Source image — same file used in gen_aspect_ratio_distortion.py for
# consistency: cpython argparse.py (rich nested if/else structure)
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

LOW_RES = 112          # downsample target
DISPLAY_SIZE = 800     # upscale back to this for display
OUTPUT_PATH = Path(__file__).parent / "topology_survival.png"
DPI = 150

# ---------------------------------------------------------------------------
# Indent levels in pixels for a 800px-wide image rendered at ~10px per space.
# 4-space indentation at the MVV font size ≈ 40px per indent level.
# ---------------------------------------------------------------------------
INDENT_LEVELS_PX = [0, 40, 80, 120, 160]   # 0 through 4 levels of 4-space indent


def load_full(path: Path) -> Image.Image:
    """Load the 800x800 image as grayscale, return PIL Image."""
    return Image.open(path).convert("L")


def make_blurred(full: Image.Image) -> Image.Image:
    """
    Downsample to 112x112 with LANCZOS, then upscale to 800x800 with
    nearest-neighbor to show pixelation explicitly.
    """
    small = full.resize((LOW_RES, LOW_RES), Image.LANCZOS)
    return small.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.NEAREST)


def add_staircase_underlines(ax, img_w: int, img_h: int):
    """
    Draw short horizontal red lines that mark the indentation staircase —
    like underlines beneath each indent level on the left side of the image.

    For each indent level N, a horizontal red line spans from x=0 to
    x=indent_level_px, positioned at evenly spaced y positions across the
    image height.  This creates a staircase of red underlines cascading from
    left to right down the left margin.

    Parameters
    ----------
    ax           : matplotlib Axes (with imshow already called)
    img_w, img_h : pixel dimensions of the displayed image
    """
    n_levels = len(INDENT_LEVELS_PX)
    # Space levels evenly: place each at 1/(n+1) … n/(n+1) of image height
    y_positions = [
        int(img_h * (i + 1) / (n_levels + 1))
        for i in range(n_levels)
    ]

    for depth, (indent_px, y_pos) in enumerate(zip(INDENT_LEVELS_PX, y_positions)):
        # Scale indent_px from the 800px reference to the actual image width
        x_end = int(indent_px * img_w / 800)
        # For depth 0 (x_end == 0) draw a minimal 8px tick so it's visible
        x_end_draw = max(x_end, 8)

        ax.plot(
            [0, x_end_draw],
            [y_pos, y_pos],
            color="red",
            linewidth=3,
            solid_capstyle="butt",
            zorder=5,
        )

        # Label at the right end of the line
        label_x = x_end_draw + 4
        ax.text(
            label_x,
            y_pos,
            str(depth),
            color="red",
            fontsize=8,
            fontweight="bold",
            va="center",
            ha="left",
            zorder=6,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor="none",
                alpha=0.75,
            ),
        )


def main():
    if not SOURCE_IMAGE.exists():
        print(f"ERROR: source image not found:\n  {SOURCE_IMAGE}", file=sys.stderr)
        print("Run from the repo root or ensure Phase_1_1 data exists.", file=sys.stderr)
        sys.exit(1)

    full_img = load_full(SOURCE_IMAGE)
    blurred_img = make_blurred(full_img)

    full_arr = np.array(full_img)
    blurred_arr = np.array(blurred_img)

    h_full, w_full = full_arr.shape        # 800, 800
    h_blur, w_blur = blurred_arr.shape     # 800, 800 (upscaled)

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(
        1, 2,
        figsize=(13, 6),
        facecolor="white",
    )
    fig.subplots_adjust(top=0.86, bottom=0.04, left=0.03, right=0.97, wspace=0.10)

    fig.suptitle(
        "Spatial Topology Survives Extreme Compression",
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )

    # ── Left panel: full resolution 800x800 ───────────────────────────────
    ax_full = axes[0]
    ax_full.imshow(full_arr, cmap="gray", vmin=0, vmax=255,
                   aspect="equal", interpolation="nearest")
    ax_full.set_title("Full Resolution\n800×800", fontsize=11, pad=6)
    ax_full.axis("off")
    add_staircase_underlines(ax_full, w_full, h_full)

    ax_full.annotate(
        "800 × 800 px  |  729 tokens (SigLIP)",
        xy=(0.5, -0.03),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=8,
        color="#555555",
    )

    # ── Right panel: blurred + upscaled ───────────────────────────────────
    ax_blur = axes[1]
    ax_blur.imshow(blurred_arr, cmap="gray", vmin=0, vmax=255,
                   aspect="equal", interpolation="nearest")
    ax_blur.set_title(
        f"Blurred ({LOW_RES}×{LOW_RES})\nTopology Preserved",
        fontsize=11,
        pad=6,
    )
    ax_blur.axis("off")
    add_staircase_underlines(ax_blur, w_blur, h_blur)

    ax_blur.annotate(
        f"{LOW_RES} × {LOW_RES} px  |  121 tokens (SigLIP)",
        xy=(0.5, -0.03),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=8,
        color="#555555",
    )

    # Thin border
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
