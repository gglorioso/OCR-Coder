"""
visualize_budgets.py — Two-row comparison of token budget downsampling.

Row 1: Images at their actual pixel size (padded to same canvas) — shows physical shrinkage.
Row 2: All images blown up to 800×800 (nearest-neighbor) — shows detail degradation.

Output: data_mvv/images/downsampled/budget_comparison.png
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

BUDGETS = [729, 441, 256, 121]
DIMS    = {729: (378, 378), 441: (294, 294), 256: (224, 224), 121: (154, 154)}
LABELS  = {729: "729 tokens\n378×378px\n(readable)",
           441: "441 tokens\n294×294px\n(transition)",
           256: "256 tokens\n224×224px\n(text dead)",
           121: "121 tokens\n154×154px\n(topology floor)"}

SRC_DIR = Path(__file__).parent.parent / "data_mvv" / "images" / "downsampled"
OUT     = SRC_DIR / "budget_comparison.png"

# ── Load the four downsampled images ──────────────────────────────────────────
imgs = {}
for b in BUDGETS:
    w, h = DIMS[b]
    p = SRC_DIR / f"black__action__main_py_budget{b}_{w}x{h}.png"
    imgs[b] = Image.open(p).convert("L")

# ── Layout constants ───────────────────────────────────────────────────────────
FULL      = 800          # display size for row 2
PADDING   = 20           # gap between cells
LABEL_H   = 55           # space for label below each image
ROW_GAP   = 40           # gap between the two rows
ROW1_H    = FULL         # row 1 canvas height = 800 (tallest image padded to this)
ROW2_H    = FULL
N         = len(BUDGETS)
TOTAL_W   = N * FULL + (N + 1) * PADDING
ROW_LABEL_W = 160        # left column for row labels

CANVAS_W = ROW_LABEL_W + TOTAL_W
CANVAS_H = LABEL_H + ROW1_H + LABEL_H + ROW_GAP + LABEL_H + ROW2_H + LABEL_H + PADDING

canvas = Image.new("L", (CANVAS_W, CANVAS_H), color=240)
draw   = ImageDraw.Draw(canvas)

# Try to load a font; fall back to default
try:
    font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    font_hd = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
except Exception:
    font_sm = ImageFont.load_default()
    font_hd = font_sm

def draw_label(text, cx, y, font):
    for i, line in enumerate(text.split("\n")):
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((cx - tw // 2, y + i * 18), line, fill=80, font=font)

# ── Row labels (left margin) ───────────────────────────────────────────────────
row1_y = LABEL_H
row2_y = LABEL_H + ROW1_H + LABEL_H + ROW_GAP + LABEL_H

draw_label("Actual size\n(shrinkage)", ROW_LABEL_W // 2, row1_y + ROW1_H // 2 - 18, font_hd)
draw_label("Blown up to\n800×800\n(degradation)", ROW_LABEL_W // 2, row2_y + ROW2_H // 2 - 27, font_hd)

# ── Draw both rows ─────────────────────────────────────────────────────────────
for i, b in enumerate(BUDGETS):
    w, h   = DIMS[b]
    cell_x = ROW_LABEL_W + PADDING + i * (FULL + PADDING)
    cx     = cell_x + FULL // 2

    # Column header label
    draw_label(LABELS[b], cx, 5, font_sm)

    # --- Row 1: actual size, centered in an 800×800 cell ---
    img    = imgs[b]
    off_x  = cell_x + (FULL - w) // 2
    off_y  = row1_y + (FULL - h) // 2
    canvas.paste(img, (off_x, off_y))
    # draw a thin border to show cell boundary
    draw.rectangle([cell_x, row1_y, cell_x + FULL - 1, row1_y + FULL - 1],
                   outline=180)

    # Row 1 size annotation below
    draw_label(f"{w}×{h}px", cx, row1_y + ROW1_H + 5, font_sm)

    # --- Row 2: nearest-neighbor upscale to 800×800 ---
    blown = img.resize((FULL, FULL), Image.NEAREST)
    canvas.paste(blown, (cell_x, row2_y))
    draw.rectangle([cell_x, row2_y, cell_x + FULL - 1, row2_y + FULL - 1],
                   outline=180)

    draw_label("↑ upscaled", cx, row2_y + ROW2_H + 5, font_sm)

# ── Save ───────────────────────────────────────────────────────────────────────
canvas.save(OUT, dpi=(150, 150))
print(f"Saved → {OUT}")
print(f"Canvas size: {CANVAS_W}×{CANVAS_H}px")
