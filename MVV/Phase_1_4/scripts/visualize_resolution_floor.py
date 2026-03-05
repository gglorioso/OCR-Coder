#!/usr/bin/env python3
"""
visualize_resolution_floor.py

Visualize the resolution floor effect by downsampling a code image to 224x224
using three methods, then upsampling back with Nearest Neighbor to reveal
actual pixel blocks. Outputs a 2x2 comparison grid PNG.

Usage:
    python visualize_resolution_floor.py [input.png] [--zoom_keyword def] [--out resolution_comparison_grid.png]

Default input: MVV/Phase_1_1/data_mvv/images/black__action__main_py.png
"""

import argparse
import sys
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

DEFAULT_IMAGE = str(
    Path(__file__).resolve().parents[2]
    / "Phase_1_1" / "data_mvv" / "images" / "black__action__main_py.png"
)


def load_font(size=18):
    """Try to load a truetype font, fall back to PIL default."""
    candidate_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_label(img_rgba, text, font):
    """Draw white text with black shadow in top-left corner of image (modifies in place)."""
    draw = ImageDraw.Draw(img_rgba)
    x, y = 8, 6
    # Shadow (black, offset +1)
    draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0, 255))
    # Foreground (white)
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))


def draw_red_rect(img_rgba, left, top, width, height):
    """Draw a red rectangle on the image (modifies in place)."""
    draw = ImageDraw.Draw(img_rgba)
    right = left + width
    bottom = top + height
    draw.rectangle([left, top, right, bottom], outline=(255, 0, 0, 255), width=2)


def make_zoom_inset(img_rgba, left, top, width, height, scale=3):
    """Crop region, resize 3x with NN, return as RGBA image."""
    crop = img_rgba.crop((left, top, left + width, top + height))
    zoomed = crop.resize((width * scale, height * scale), Image.NEAREST)
    return zoomed


def paste_inset_with_border(canvas, inset, quadrant_x, quadrant_y, quad_w, quad_h, border=2):
    """Paste inset into bottom-right corner of a quadrant on the canvas, with a red border."""
    iw, ih = inset.size
    # Position: bottom-right with 4px margin
    margin = 4
    paste_x = quadrant_x + quad_w - iw - margin - border
    paste_y = quadrant_y + quad_h - ih - margin - border

    # Draw red border onto canvas first
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        [paste_x - border, paste_y - border,
         paste_x + iw + border - 1, paste_y + ih + border - 1],
        outline=(255, 0, 0, 255),
        width=border,
    )
    # Paste inset
    canvas.paste(inset, (paste_x, paste_y))


def to_rgba(img):
    """Convert any PIL image to RGBA."""
    return img.convert("RGBA")


def process_method(original_img, target_size, downsample_filter, orig_size):
    """
    Downsample original_img to target_size using downsample_filter,
    then upsample back to orig_size using Nearest Neighbor.
    Returns RGBA image at orig_size.
    """
    small = original_img.resize(target_size, downsample_filter)
    blown_up = small.resize(orig_size, Image.NEAREST)
    return to_rgba(blown_up)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the resolution floor effect on code images."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_IMAGE,
        help="Input PNG file path (default: black__action__main_py.png)",
    )
    parser.add_argument(
        "--zoom_keyword",
        default="def",
        help="Keyword to search for zoom region (not used in fixed-region mode, kept for CLI compat)",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "results" / "resolution_comparison_grid.png"),
        help="Output PNG file path",
    )
    args = parser.parse_args()

    # --- Load input ---
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    original = Image.open(args.input)
    orig_w, orig_h = original.size
    print(f"Input image: {args.input}")
    print(f"  Dimensions: {orig_w} x {orig_h} px, mode: {original.mode}")

    # --- Console: mean char height estimate ---
    target_size = (224, 224)
    assumed_lines = 40
    mean_char_height = target_size[1] / assumed_lines
    print(f"\nMean character height estimate at 224x224:")
    print(f"  Assumed lines per image: {assumed_lines}")
    print(f"  mean_char_height = 224 / {assumed_lines} = {mean_char_height:.1f} px")
    print(f"  (At 224px tall, each line of code is only ~{mean_char_height:.1f} pixels high)")

    # --- Zoom region (fixed, scaled proportionally) ---
    # Base region defined for a 'standard' image; scale if different size
    # Base: top=40, left=10, height=40, width=120 (for ~800px images)
    scale_x = orig_w / 800.0
    scale_y = orig_h / 800.0
    zoom_left = max(0, int(10 * scale_x))
    zoom_top = max(0, int(40 * scale_y))
    zoom_width = min(int(120 * scale_x), orig_w - zoom_left - 1)
    zoom_height = min(int(40 * scale_y), orig_h // 3 - zoom_top - 1)
    # Clamp minimums
    zoom_width = max(zoom_width, 20)
    zoom_height = max(zoom_height, 10)

    print(f"\nZoom inset region: left={zoom_left}, top={zoom_top}, "
          f"width={zoom_width}, height={zoom_height}")

    # --- Process three methods ---
    original_rgba = to_rgba(original)

    methods = [
        {
            "label": "Original (Ground Truth)",
            "image": original_rgba,
        },
        {
            "label": "Standard Bicubic — 'The Gray Blur'",
            "image": process_method(original, target_size, Image.BICUBIC, (orig_w, orig_h)),
        },
        {
            "label": "Nearest Neighbor — 'Aliased Skeleton'",
            "image": process_method(original, target_size, Image.NEAREST, (orig_w, orig_h)),
        },
        {
            "label": "Area Supersampled — 'Balanced Signal'",
            "image": process_method(original, target_size, Image.LANCZOS, (orig_w, orig_h)),
        },
    ]

    # --- Font ---
    font = load_font(size=18)

    # --- Build 2x2 grid ---
    # 4px black border between quadrants
    border_px = 4
    grid_w = orig_w * 2 + border_px
    grid_h = orig_h * 2 + border_px

    grid = Image.new("RGBA", (grid_w, grid_h), color=(0, 0, 0, 255))

    # Quadrant positions: (x_offset, y_offset)
    positions = [
        (0, 0),                          # top-left: Original
        (orig_w + border_px, 0),         # top-right: Bicubic
        (0, orig_h + border_px),         # bottom-left: Nearest Neighbor
        (orig_w + border_px, orig_h + border_px),  # bottom-right: Area
    ]

    for i, (method, pos) in enumerate(zip(methods, positions)):
        qx, qy = pos
        img = method["image"].copy()

        # Draw red rectangle around zoom region
        draw_red_rect(img, zoom_left, zoom_top, zoom_width, zoom_height)

        # Draw label
        draw_label(img, method["label"], font)

        # Make zoom inset (from the version with NO red rect yet — use original img data)
        inset_src = method["image"].copy()
        inset = make_zoom_inset(inset_src, zoom_left, zoom_top, zoom_width, zoom_height, scale=3)

        # Paste quadrant onto grid
        grid.paste(img, (qx, qy))

        # Paste inset with red border
        paste_inset_with_border(grid, inset, qx, qy, orig_w, orig_h, border=2)

    # --- Save ---
    grid_rgb = grid.convert("RGB")
    grid_rgb.save(args.out, format="PNG", optimize=False)
    print(f"\nSaved grid: {args.out}  ({grid_w} x {grid_h} px)")


if __name__ == "__main__":
    main()
