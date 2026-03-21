"""
visualize_budgets_compact.py — Compact side-by-side comparison of token budgets.

Shows only the blown-up (800×800 nearest-neighbor) images in a single row,
with large labels showing token count and original pixel size.
A ylabel-style annotation on the left reads "Blown up to 800×800 scale".

Output: data_mvv/images/downsampled/resolution_comparison_compact.png
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BUDGETS = [729, 441, 256, 121]
DIMS    = {729: (378, 378), 441: (294, 294), 256: (224, 224), 121: (154, 154)}

# Large labels: token count + original pixel size
LABELS  = {
    729: "729 tokens | 378×378px",
    441: "441 tokens | 294×294px",
    256: "256 tokens | 224×224px",
    121: "121 tokens | 154×154px",
}

SRC_DIR = Path(__file__).parent.parent / "data_mvv" / "images" / "downsampled"
OUT     = SRC_DIR / "resolution_comparison_compact.png"

# ── Load the four downsampled images ──────────────────────────────────────────
imgs = {}
for b in BUDGETS:
    w, h = DIMS[b]
    p = SRC_DIR / f"black__action__main_py_budget{b}_{w}x{h}.png"
    imgs[b] = Image.open(p).convert("L")

# ── Layout constants ───────────────────────────────────────────────────────────
FULL        = 800    # display size for each blown-up image
PADDING     = 16     # gap between cells
LABEL_H     = 60     # space for label above each image (large font needs room)
LEFT_BAR_W  = 180    # left column for the "Blown up to 800×800 scale" annotation
N           = len(BUDGETS)

CANVAS_W = LEFT_BAR_W + N * FULL + (N + 1) * PADDING
CANVAS_H = LABEL_H + FULL + PADDING

canvas = Image.new("L", (CANVAS_W, CANVAS_H), color=250)
draw   = ImageDraw.Draw(canvas)

# ── Fonts ──────────────────────────────────────────────────────────────────────
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]

def load_font(size):
    for fp in FONT_PATHS:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            pass
    return ImageFont.load_default()

font_label   = load_font(26)   # large labels above each image (~24-28pt)
font_ylabel  = load_font(22)   # left-bar annotation

# ── Draw each blown-up image with its label ────────────────────────────────────
image_y = LABEL_H   # images start after label row

for i, b in enumerate(BUDGETS):
    cell_x = LEFT_BAR_W + PADDING + i * (FULL + PADDING)
    cx     = cell_x + FULL // 2

    # Nearest-neighbor upscale to FULL×FULL
    blown = imgs[b].resize((FULL, FULL), Image.NEAREST)
    canvas.paste(blown, (cell_x, image_y))

    # Thin border
    draw.rectangle(
        [cell_x, image_y, cell_x + FULL - 1, image_y + FULL - 1],
        outline=160
    )

    # Large label centered above the image
    label = LABELS[b]
    bbox  = draw.textbbox((0, 0), label, font=font_label)
    tw    = bbox[2] - bbox[0]
    th    = bbox[3] - bbox[1]
    tx    = cx - tw // 2
    ty    = (LABEL_H - th) // 2   # vertically center in label row
    draw.text((tx, ty), label, fill=30, font=font_label)

# ── Left-bar annotation: "Blown up to 800×800 scale" ─────────────────────────
# Render vertically rotated text by drawing on a temporary image then rotating.
bar_text  = "Blown up to 800×800 scale"
# Measure text in normal orientation
bbox_tmp  = draw.textbbox((0, 0), bar_text, font=font_ylabel)
txt_w     = bbox_tmp[2] - bbox_tmp[0]
txt_h     = bbox_tmp[3] - bbox_tmp[1]

# Create a temporary grayscale image wide enough for the text
tmp = Image.new("L", (txt_w + 20, txt_h + 10), color=250)
tmp_draw = ImageDraw.Draw(tmp)
tmp_draw.text((10, 5), bar_text, fill=60, font=font_ylabel)

# Rotate 90° counter-clockwise so it reads bottom-to-top
rotated = tmp.rotate(90, expand=True)

# Paste into the left bar, vertically centered over the image area
bar_img_cy = image_y + FULL // 2
paste_x    = (LEFT_BAR_W - rotated.width)  // 2
paste_y    = bar_img_cy - rotated.height // 2
canvas.paste(rotated, (paste_x, paste_y))

# ── Save ───────────────────────────────────────────────────────────────────────
canvas.save(OUT, dpi=(150, 150))
print(f"Saved → {OUT}")
print(f"Canvas size: {CANVAS_W}×{CANVAS_H}px")
