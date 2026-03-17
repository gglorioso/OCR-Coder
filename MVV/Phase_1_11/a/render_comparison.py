#!/usr/bin/env python3
"""
render_comparison.py — Phase 1.11a visual line-count comparison

Renders a single Python source file at 5 different line-count windows
(20, 40, 80, 160, 320 lines) all starting at the same AST anchor, using
the EXACT same rendering parameters as Phase 1.1.

Canvas spec (identical to Phase 1.1 gen_images.py):
  Font:        DejaVu Sans Mono, size 16
  Char width:  10px  (80 chars x 10px = 800px wide)
  Line height: 20px
  Canvas width: 800px, PIL 'L' mode (8-bit grayscale)
  Background:  white (255), text: black (0)
  Max cols:    80 chars (hard truncation, no wrapping)
  Tab size:    4 spaces

Output:
  results/sample_20lines.png   ... results/sample_320lines.png
  results/comparison_all.png   (all 5 images side-by-side, same height)
"""

import ast
import sys
from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFont

# ── Canvas constants — EXACT copy from Phase 1.1 gen_images.py ──────────────
FONT_PATH   = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_SIZE   = 16
LINE_HEIGHT = 20      # px
MAX_COLS    = 80      # chars before hard truncation
CANVAS_W    = MAX_COLS * 10   # 800px
BG_COLOR    = 255     # white
TEXT_COLOR  = 0       # black

# ── Source file (must have >= 320 lines) ────────────────────────────────────
REPO_ROOT = Path("/home/ad.msoe.edu/gloriosog/CoderOCR/OCR-Coder")
SOURCE_FILE = REPO_ROOT / "Scraped Repos" / "black" / "src" / "black" / "brackets.py"

# ── Line-count window sizes to compare ──────────────────────────────────────
LINE_COUNTS = [20, 40, 80, 160, 320]

# ── Comparison image layout ──────────────────────────────────────────────────
LABEL_HEIGHT   = 40    # px of label strip above each panel
PANEL_GAP      = 10    # px gap between panels
PANEL_TARGET_H = 800   # px — each panel is scaled to this height for comparison
LABEL_FONT_SIZE = 20


# ── AST anchor — EXACT logic from Phase 1.1 gen_images.py ───────────────────

STRUCTURAL_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.ClassDef,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
)


def find_topological_start(source: str) -> int:
    """
    Return the 0-indexed line number of the first real structural node,
    skipping any leading module docstring and comment-only lines.

    Falls back to line 0 on parse failure.

    Strategy:
      1. Parse the AST.
      2. If the first body node is a bare string expression (module docstring),
         skip it.
      3. Return the lineno of the first Import / ClassDef / FunctionDef node.
      4. If none found, return the line after the docstring (or 0).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0

    body = tree.body
    if not body:
        return 0

    # Detect and skip module-level docstring
    start_idx = 0
    first = body[0]
    if (isinstance(first, ast.Expr) and
            isinstance(first.value, (ast.Constant, ast.Str))):
        start_idx = 1

    # Find first structural node after the docstring
    for node in body[start_idx:]:
        if isinstance(node, STRUCTURAL_NODES):
            return node.lineno - 1  # ast lineno is 1-indexed

    # Fallback: start after docstring, or line 0
    if start_idx < len(body):
        return body[start_idx].lineno - 1
    return 0


# ── Rendering ────────────────────────────────────────────────────────────────

def render_chunk(lines: List[str], n_rows: int, font: ImageFont.FreeTypeFont) -> Image.Image:
    """
    Render exactly n_rows lines on a (CANVAS_W x n_rows*LINE_HEIGHT) grayscale canvas.
    Lines longer than MAX_COLS are hard-truncated (no wrapping).
    Fewer lines than n_rows → blank rows pad the bottom.

    Note: Phase 1.1 always used MAX_ROWS=40 rows → fixed 800x800 canvas.
    Here we vary n_rows so canvas height scales with line count, then the
    comparison function normalises height across panels.
    """
    canvas_h = n_rows * LINE_HEIGHT
    img  = Image.new("L", (CANVAS_W, canvas_h), color=BG_COLOR)
    draw = ImageDraw.Draw(img)
    for i in range(n_rows):
        if i < len(lines):
            line = lines[i].expandtabs(4)[:MAX_COLS]
        else:
            line = ""
        draw.text((0, i * LINE_HEIGHT), line, font=font, fill=TEXT_COLOR)
    return img


def build_comparison(images: List[Image.Image], labels: List[str]) -> Image.Image:
    """
    Scale each image to PANEL_TARGET_H preserving aspect ratio, then lay them
    out horizontally with LABEL_HEIGHT label strips and PANEL_GAP gaps.
    """
    try:
        label_font = ImageFont.truetype(FONT_PATH, LABEL_FONT_SIZE)
    except IOError:
        label_font = ImageFont.load_default()

    # Scale each panel to target height
    scaled = []
    for img in images:
        orig_w, orig_h = img.size
        scale = PANEL_TARGET_H / orig_h
        new_w = int(orig_w * scale)
        scaled.append(img.resize((new_w, PANEL_TARGET_H), Image.LANCZOS))

    total_w = sum(s.size[0] for s in scaled) + PANEL_GAP * (len(scaled) - 1)
    total_h = PANEL_TARGET_H + LABEL_HEIGHT

    comp = Image.new("L", (total_w, total_h), color=BG_COLOR)
    draw = ImageDraw.Draw(comp)

    x_offset = 0
    for img, label in zip(scaled, labels):
        # Paste panel below label strip
        comp.paste(img, (x_offset, LABEL_HEIGHT))

        # Draw label centred above the panel
        panel_w = img.size[0]
        try:
            bbox = draw.textbbox((0, 0), label, font=label_font)
            text_w = bbox[2] - bbox[0]
        except AttributeError:
            text_w = len(label) * LABEL_FONT_SIZE * 0.6  # rough fallback

        text_x = x_offset + (panel_w - text_w) // 2
        text_y = (LABEL_HEIGHT - LABEL_FONT_SIZE) // 2
        draw.text((text_x, text_y), label, font=label_font, fill=TEXT_COLOR)

        x_offset += panel_w + PANEL_GAP

    return comp


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load font — same as Phase 1.1
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except IOError:
        print(f"ERROR: font not found at {FONT_PATH}", file=sys.stderr)
        sys.exit(1)

    # Read source file
    source_path = SOURCE_FILE
    print(f"Source file : {source_path}")

    try:
        source = source_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        print(f"ERROR: source file not found: {source_path}", file=sys.stderr)
        sys.exit(1)

    all_lines = source.splitlines()
    n_total   = len(all_lines)
    print(f"Total lines : {n_total}")

    if n_total < 320:
        print(f"WARNING: file has only {n_total} lines; 320-line window will be padded.",
              file=sys.stderr)

    # Compute AST anchor — identical logic to Phase 1.1
    anchor = find_topological_start(source)
    print(f"Anchor line : {anchor}  (0-indexed, i.e. line {anchor + 1} in editor)")
    print()

    # Render each window and save individually
    rendered = []
    labels   = []
    for n_lines in LINE_COUNTS:
        chunk = all_lines[anchor: anchor + n_lines]
        img   = render_chunk(chunk, n_lines, font)

        out_path = results_dir / f"sample_{n_lines}lines.png"
        img.save(out_path, format="PNG", optimize=False)
        print(f"  Saved {n_lines:>3d}-line image  ({img.size[0]}x{img.size[1]}px) → {out_path}")

        rendered.append(img)
        labels.append(f"{n_lines} lines")

    # Build and save side-by-side comparison
    print()
    comp_path = results_dir / "comparison_all.png"
    comp = build_comparison(rendered, labels)
    comp.save(comp_path, format="PNG", optimize=False)
    print(f"Comparison  : {comp_path}  ({comp.size[0]}x{comp.size[1]}px)")


if __name__ == "__main__":
    main()
