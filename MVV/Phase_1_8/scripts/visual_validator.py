#!/usr/bin/env python3
"""
visual_validator.py — Phase 1.8: Visual sanity-check for AST → patch-grid mapping

Reads the ground-truth JSONL produced by ast_extractor.py, finds the matching
rendered image for each record, and draws:

  • A faint 8×8 grid overlay (gray lines at every patch boundary)
  • A semi-transparent red horizontal band for each patch-grid row that a node
    occupies, labelled with the node name and type

The annotated images are saved to --out-dir so a human can quickly verify that
the highlighted bands line up with the correct functions / classes in the image.

Usage:
    python visual_validator.py \\
        --ground-truth <path/to/ground_truth.jsonl> \\
        --image-dir    <path/to/rendered/images> \\
        --out-dir      <path/to/validation_checks/> \\
        --max-samples  50 \\
        --node-type    all   # or "def" or "class"
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Draw patch-grid overlays on rendered images to validate "
                    "the AST coordinate mapping."
    )
    p.add_argument(
        "--ground-truth", required=True, type=Path,
        help="JSONL file produced by ast_extractor.py.",
    )
    p.add_argument(
        "--image-dir", required=True, type=Path,
        help="Directory containing the rendered .png images.",
    )
    p.add_argument(
        "--out-dir", required=True, type=Path,
        help="Output directory for annotated validation images.",
    )
    p.add_argument(
        "--max-samples", type=int, default=50,
        help="Maximum number of unique images to annotate (default: 50).",
    )
    p.add_argument(
        "--node-type", default="all", choices=["all", "def", "class"],
        help="Which node types to highlight: 'all', 'def', or 'class' (default: all).",
    )
    p.add_argument(
        "--output-size", type=int, default=448,
        help="Expected image side length in pixels (default: 448).",
    )
    p.add_argument(
        "--grid-size", type=int, default=8,
        help="Number of patch-grid rows (and columns) per side (default: 8).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

# Band colours cycle so multiple nodes in the same row are distinguishable.
BAND_COLOURS = [
    (255,   0,   0),   # red
    (255, 140,   0),   # orange
    (  0, 160,  60),   # green
    (  0, 100, 220),   # blue
    (160,   0, 200),   # purple
]

BAND_ALPHA      = 80    # fill alpha (0–255)
OUTLINE_ALPHA   = 200   # border alpha
GRID_COLOUR     = (160, 160, 160, 100)   # faint gray grid lines


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _overlay_rgba(base, overlay):
    """Alpha-composite an RGBA overlay onto an RGBA base (in-place on base)."""
    from PIL import Image
    return Image.alpha_composite(base, overlay)


def draw_grid(draw, image_size: int, grid_size: int, colour) -> None:
    """Draw faint horizontal and vertical grid lines at every patch boundary."""
    patch_px = image_size // grid_size
    for i in range(1, grid_size):
        pos = i * patch_px
        # Horizontal
        draw.line([(0, pos), (image_size, pos)], fill=colour, width=1)
        # Vertical
        draw.line([(pos, 0), (pos, image_size)], fill=colour, width=1)


def draw_band(overlay_draw, row: int, patch_px: int,
              colour_rgb: tuple, label: str, label_y_offset: int) -> None:
    """
    Draw one semi-transparent band for a single patch-grid row.

    overlay_draw : ImageDraw on the RGBA overlay image
    row          : 0-indexed grid row
    patch_px     : pixel height of one patch row  (image_size // grid_size)
    colour_rgb   : (R, G, B) tuple
    label        : text to draw inside the band
    label_y_offset: vertical nudge so stacked labels don't overlap
    """
    y0 = row * patch_px
    y1 = (row + 1) * patch_px

    fill    = colour_rgb + (BAND_ALPHA,)
    outline = colour_rgb + (OUTLINE_ALPHA,)

    # Fill rectangle
    overlay_draw.rectangle([0, y0, patch_px * 8, y1], fill=fill)
    # Top border
    overlay_draw.line([(0, y0), (patch_px * 8, y0)], fill=outline, width=2)
    # Bottom border
    overlay_draw.line([(0, y1 - 1), (patch_px * 8, y1 - 1)], fill=outline, width=2)

    # Label — white text with black shadow for readability
    label_y = y0 + 2 + label_y_offset
    # Shadow
    overlay_draw.text((3, label_y + 1), label, fill=(0, 0, 0, 220))
    # Text
    overlay_draw.text((2, label_y), label, fill=(255, 255, 255, 240))


# ---------------------------------------------------------------------------
# Image annotation
# ---------------------------------------------------------------------------

def annotate_image(
    image_path: Path,
    records: list,
    out_path: Path,
    output_size: int,
    grid_size: int,
) -> bool:
    """
    Open `image_path`, draw patch-grid overlay and node bands, save to `out_path`.
    Returns True on success, False on error.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("ERROR: Pillow is required. Install with: pip install Pillow",
              file=sys.stderr)
        sys.exit(1)

    try:
        img = Image.open(image_path).convert("RGBA")
    except Exception as exc:
        print(f"  WARNING: could not open {image_path}: {exc}", file=sys.stderr)
        return False

    # Resize to expected output_size if necessary (images may be 800px originals)
    if img.size != (output_size, output_size):
        img = img.resize((output_size, output_size), Image.BICUBIC)

    patch_px = output_size // grid_size

    # --- Grid overlay ---
    grid_overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    grid_draw    = ImageDraw.Draw(grid_overlay)
    draw_grid(grid_draw, output_size, grid_size, GRID_COLOUR)
    img = _overlay_rgba(img, grid_overlay)

    # --- Node band overlays ---
    # Group records by row so we can stack labels vertically without overlap
    row_label_counts: dict = defaultdict(int)

    for idx, rec in enumerate(records):
        colour = BAND_COLOURS[idx % len(BAND_COLOURS)]
        label  = f"{rec['type']} {rec['name']}  (L{rec['lineno']}–{rec['end_lineno']})"

        band_overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        band_draw    = ImageDraw.Draw(band_overlay)

        # Each row this node spans gets its own band slice
        for row in rec["grid_rows"]:
            y_offset = row_label_counts[row] * 13   # ~12px per label line
            draw_band(band_draw, row, patch_px, colour, label, y_offset)
            row_label_counts[row] += 1

        img = _overlay_rgba(img, band_overlay)

    # Convert back to RGB for PNG saving (removes alpha, consistent with source images)
    final = img.convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.save(out_path, format="PNG")
    return True


# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------

def load_ground_truth(gt_path: Path, node_type_filter: str) -> dict:
    """
    Read ground_truth.jsonl and return a dict mapping file stem → list of records.
    Applies node_type_filter ('all' / 'def' / 'class').
    """
    file_records: dict = defaultdict(list)
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if node_type_filter != "all" and rec.get("type") != node_type_filter:
                continue
            file_records[rec["file"]].append(rec)
    return file_records


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

def find_image(stem: str, image_dir: Path) -> Path | None:
    """
    Locate the rendered .png for a given file stem.

    Tries:
      1. <image_dir>/<stem>.png
      2. Any .png in <image_dir> whose stem matches (recursive)
    """
    direct = image_dir / f"{stem}.png"
    if direct.exists():
        return direct

    # Recursive search (images may be in subdirectories)
    matches = list(image_dir.rglob(f"{stem}.png"))
    if matches:
        return matches[0]

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Validate inputs
    if not args.ground_truth.exists():
        print(f"ERROR: ground-truth file not found: {args.ground_truth}", file=sys.stderr)
        sys.exit(1)
    if not args.image_dir.exists():
        print(f"ERROR: image directory not found: {args.image_dir}", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    file_records = load_ground_truth(args.ground_truth, args.node_type)
    print(f"Loaded records for {len(file_records):,} unique files "
          f"(filter: node_type='{args.node_type}').")

    # Annotate up to max_samples images
    n_done    = 0
    n_missing = 0
    n_errors  = 0

    for stem, records in sorted(file_records.items()):
        if n_done >= args.max_samples:
            break

        img_path = find_image(stem, args.image_dir)
        if img_path is None:
            n_missing += 1
            print(f"  [MISSING]  {stem}.png")
            continue

        out_path = args.out_dir / f"{stem}_validated.png"
        print(f"  [{n_done + 1:>3}/{args.max_samples}] {stem}  "
              f"({len(records)} nodes) → {out_path.name}")

        ok = annotate_image(
            image_path  = img_path,
            records     = records,
            out_path    = out_path,
            output_size = args.output_size,
            grid_size   = args.grid_size,
        )

        if ok:
            n_done += 1
        else:
            n_errors += 1

    # Summary
    print()
    print("── visual_validator summary ────────────────────────────")
    print(f"  Images annotated     : {n_done}")
    print(f"  Images missing       : {n_missing}")
    print(f"  Images with errors   : {n_errors}")
    print(f"  Output directory     : {args.out_dir}")
    print("────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
