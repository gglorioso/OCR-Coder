#!/usr/bin/env python3
"""
render_enhanced.py — MVV Phase 1.7 enhanced image renderer

Produces 800×800 RGB images from Python source files with optional visual
enhancements:
  --syntax-highlighting   Monokai color scheme via Pygments
  --line-numbers          Left margin with line numbers + separator
  --indent-guides         Vertical guide lines at 4-space indent stops

Canvas spec (matches Phase 1.1 geometry):
  Font:        DejaVu Sans Mono, size 16
  Char width:  10px
  Line height: 20px  (40 lines × 20px = 800px tall)
  Canvas:      800 × 800 px, PIL 'RGB' mode
  Background:  #272822 (Monokai dark) if --syntax-highlighting, else white

When --line-numbers is active:
  - Left margin: 40px (4 chars × 10px/char)
  - Separator line at x=43, color (80,80,80)
  - Code text starts at x=50
  - MAX_COLS for code: 75  ((800-50) // 10)

When --indent-guides is active:
  - After drawing all text, for each line draw a 1px vertical segment at
    x = code_start + k*4*10 for each 4-space indent level k ≥ 1.
  - Color: (80,80,80)

Usage:
    python render_enhanced.py \\
        --repos-dir "Scraped Repos" \\
        --output-dir MVV/Phase_1_7/images/exp_A \\
        --syntax-highlighting
"""

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# ── Pygments imports (optional — only used when --syntax-highlighting) ─────────
try:
    from pygments import lex
    from pygments.lexers import PythonLexer
    from pygments.token import Token
    _PYGMENTS_OK = True
except ImportError:
    _PYGMENTS_OK = False

# ── Canvas constants ───────────────────────────────────────────────────────────
FONT_PATH   = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_SIZE   = 16
LINE_HEIGHT = 20      # px
MAX_ROWS    = 40      # lines per image
CANVAS_W    = 800
CANVAS_H    = 800
CHAR_W      = 10      # px per char (DejaVu Mono 16px)

# Default MAX_COLS (no line numbers); overridden when --line-numbers is active
MAX_COLS_DEFAULT  = 80
MAX_COLS_LINENUM  = 75   # (800 - 50) // 10

# Line-number layout
LINENUM_MARGIN    = 40   # px reserved for digits
LINENUM_SEP_X     = 43   # px — vertical separator line x position
CODE_START_LINENUM = 50  # px — where code text begins (with line numbers)
CODE_START_PLAIN   = 0   # px — code text starts at left edge

# Colors
BG_DARK       = (39, 40, 34)    # #272822 Monokai dark
BG_LIGHT      = (255, 255, 255) # white
TEXT_DEFAULT  = (248, 248, 242) # #F8F8F2 Monokai default fg
TEXT_BLACK    = (0, 0, 0)
GUIDE_COLOR   = (80, 80, 80)
SEP_COLOR     = (80, 80, 80)
LINENUM_COLOR = (128, 128, 128)

# Monokai syntax color map (Pygments Token → hex string)
MONOKAI_MAP = {
    Token.Keyword:                 "#F92672",
    Token.Keyword.Declaration:     "#F92672",
    Token.Keyword.Namespace:       "#F92672",
    Token.Keyword.Type:            "#F92672",
    Token.Name.Function:           "#A6E22E",
    Token.Name.Class:              "#A6E22E",
    Token.Name.Decorator:          "#A6E22E",
    Token.Literal.String:          "#E6DB74",
    Token.Literal.Number:          "#AE81FF",
    Token.Comment:                 "#75715E",
    Token.Operator:                "#F92672",
    Token.Punctuation:             "#F92672",
    Token.Name.Builtin:            "#66D9EF",
}
MONOKAI_DEFAULT = "#F8F8F2"

# ── Same skip/slug/AST logic as gen_images.py ─────────────────────────────────
SKIP_PATTERNS = [
    "/test/", "/tests/", "test_", "_test.py", "conftest.py",
    "/vendor/", "/vendored/", "/third_party/", "/_vendor/",
    "/migrations/", "/generated/", "__pycache__",
]

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
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0

    body = tree.body
    if not body:
        return 0

    start_idx = 0
    first = body[0]
    if (isinstance(first, ast.Expr) and
            isinstance(first.value, (ast.Constant, ast.Str))):
        start_idx = 1

    for node in body[start_idx:]:
        if isinstance(node, STRUCTURAL_NODES):
            return node.lineno - 1

    if start_idx < len(body):
        return body[start_idx].lineno - 1
    return 0


def should_skip(rel_path: str) -> bool:
    p = rel_path.lower()
    if any(pat in p for pat in SKIP_PATTERNS):
        return True
    if Path(rel_path).name == "__init__.py":
        return True
    return False


def slug(rel_path: str) -> str:
    """Flatten a relative path to a filename-safe string."""
    return rel_path.replace("/", "__").replace("\\", "__").replace(".", "_")


# ── Color helpers ──────────────────────────────────────────────────────────────

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert '#RRGGBB' to (R, G, B) tuple."""
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def get_token_color(ttype, colormap: dict) -> str:
    """
    Walk up the Pygments token type hierarchy to find a matching color.
    Returns MONOKAI_DEFAULT if no match is found.
    """
    t = ttype
    while t is not None:
        if t in colormap:
            return colormap[t]
        # Move up to parent; Pygments token types support parent via attribute
        parent = t.parent if hasattr(t, "parent") else None
        if parent == t or parent is None:
            break
        t = parent
    return MONOKAI_DEFAULT


# ── Rendering ─────────────────────────────────────────────────────────────────

def tokenize_lines(lines: List[str]) -> List[List[Tuple[str, str]]]:
    """
    Tokenize a list of source lines with Pygments PythonLexer.
    Returns list of per-line span lists: [[(color_hex, text), ...], ...]
    Always returns exactly len(lines) sublists.
    """
    code_string = "\n".join(lines)
    token_stream = list(lex(code_string, PythonLexer()))

    # Rebuild per-line colored spans
    line_spans: List[List[Tuple[str, str]]] = [[] for _ in range(len(lines))]
    current_line = 0

    for ttype, value in token_stream:
        color = get_token_color(ttype, MONOKAI_MAP)
        # Split value on newlines; each newline advances current_line
        parts = value.split("\n")
        for part_idx, part in enumerate(parts):
            if current_line < len(line_spans) and part:
                line_spans[current_line].append((color, part))
            if part_idx < len(parts) - 1:
                current_line += 1

    return line_spans


def render_chunk_enhanced(
    lines: List[str],
    font: ImageFont.FreeTypeFont,
    args: argparse.Namespace,
) -> Image.Image:
    """
    Render up to MAX_ROWS lines on an 800×800 RGB canvas with optional
    syntax highlighting, line numbers, and indent guides.
    """
    use_syntax  = args.syntax_highlighting
    use_linenum = args.line_numbers
    use_guides  = args.indent_guides

    # Canvas background
    bg_color = BG_DARK if use_syntax else BG_LIGHT
    img  = Image.new("RGB", (CANVAS_W, CANVAS_H), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Layout parameters
    code_start = CODE_START_LINENUM if use_linenum else CODE_START_PLAIN
    max_cols   = MAX_COLS_LINENUM    if use_linenum else MAX_COLS_DEFAULT

    # Separator line (drawn once, spans full height)
    if use_linenum:
        draw.line(
            [(LINENUM_SEP_X, 0), (LINENUM_SEP_X, CANVAS_H - 1)],
            fill=SEP_COLOR,
            width=1,
        )

    # Pre-tokenize if syntax highlighting is on
    padded_lines: List[str] = []
    for i in range(MAX_ROWS):
        if i < len(lines):
            padded_lines.append(lines[i].expandtabs(4))
        else:
            padded_lines.append("")

    if use_syntax:
        if not _PYGMENTS_OK:
            print("ERROR: Pygments not installed — cannot use --syntax-highlighting",
                  file=sys.stderr)
            sys.exit(1)
        line_spans = tokenize_lines(padded_lines)
    else:
        # Plain text: single span per line in default color
        plain_color = MONOKAI_DEFAULT if use_syntax else "#000000"
        line_spans = [[(plain_color, line)] for line in padded_lines]

    # Draw each line
    for row_idx in range(MAX_ROWS):
        y = row_idx * LINE_HEIGHT

        # Draw line number
        if use_linenum:
            linenum_str = f"{row_idx + 1:>4}"
            draw.text((0, y), linenum_str, font=font, fill=LINENUM_COLOR)

        # Draw code spans, truncating at max_cols total chars
        x         = code_start
        chars_used = 0

        for color_hex, text_part in line_spans[row_idx]:
            if chars_used >= max_cols:
                break
            # How many chars can we still fit?
            remaining = max_cols - chars_used
            clipped   = text_part[:remaining]
            if not clipped:
                continue
            rgb = hex_to_rgb(color_hex)
            draw.text((x, y), clipped, font=font, fill=rgb)
            x         += len(clipped) * CHAR_W
            chars_used += len(clipped)

    # Draw indent guides (after all text)
    if use_guides:
        for row_idx in range(MAX_ROWS):
            line_text = padded_lines[row_idx]
            if not line_text:
                continue
            # Count leading spaces
            n_leading = len(line_text) - len(line_text.lstrip(" "))
            n_levels  = n_leading // 4  # number of 4-space indent levels
            y_top     = row_idx * LINE_HEIGHT
            y_bot     = y_top + LINE_HEIGHT - 1
            for level in range(1, n_levels + 1):
                guide_x = code_start + level * 4 * CHAR_W
                if guide_x >= CANVAS_W:
                    break
                draw.line([(guide_x, y_top), (guide_x, y_bot)],
                          fill=GUIDE_COLOR, width=1)

    return img


# ── Core pipeline ──────────────────────────────────────────────────────────────

def process_repo(
    repo_path: Path,
    output_dir: Path,
    font: ImageFont.FreeTypeFont,
    args: argparse.Namespace,
) -> List[dict]:
    """Walk one repo, produce one image per valid file, return manifest records."""
    records   = []
    repo_name = repo_path.name

    for py_file in sorted(repo_path.rglob("*.py")):
        rel_path = py_file.relative_to(repo_path).as_posix()

        if should_skip(rel_path):
            continue

        try:
            source = py_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        all_lines = source.splitlines()
        if len(all_lines) < 10:
            continue

        anchor = find_topological_start(source)
        chunk  = all_lines[anchor: anchor + MAX_ROWS]

        img_name = f"{repo_name}__{slug(rel_path)}.png"
        img_path = output_dir / img_name

        if not img_path.exists():
            img = render_chunk_enhanced(chunk, font, args)
            img.save(img_path, format="PNG", optimize=False)

        records.append({
            "image":          str(img_path),
            "repo":           repo_name,
            "source_file":    f"{repo_name}/{rel_path}",
            "anchor_line":    anchor,
            "n_source_lines": len(all_lines),
        })

    return records


def main():
    parser = argparse.ArgumentParser(
        description="MVV Phase 1.7 — Enhanced image renderer"
    )
    parser.add_argument("--repos-dir",           default="Scraped Repos",
                        help="Root directory containing per-repo subdirs")
    parser.add_argument("--output-dir",          required=True,
                        help="Directory to write output PNG images")
    parser.add_argument("--syntax-highlighting", action="store_true",
                        help="Apply Monokai syntax highlighting via Pygments")
    parser.add_argument("--line-numbers",        action="store_true",
                        help="Render line numbers in a left margin")
    parser.add_argument("--indent-guides",       action="store_true",
                        help="Draw vertical indent guide lines at 4-space stops")
    args = parser.parse_args()

    repos_dir  = Path(args.repos_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.syntax_highlighting and not _PYGMENTS_OK:
        print("ERROR: --syntax-highlighting requires Pygments (`pip install pygments`)",
              file=sys.stderr)
        sys.exit(1)

    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except IOError:
        print(f"ERROR: font not found at {FONT_PATH}", file=sys.stderr)
        sys.exit(1)

    print("=" * 65)
    print("MVV Phase 1.7 — Enhanced Image Renderer")
    print(f"  Repos dir:           {repos_dir}")
    print(f"  Output dir:          {output_dir}")
    print(f"  --syntax-highlighting: {args.syntax_highlighting}")
    print(f"  --line-numbers:        {args.line_numbers}")
    print(f"  --indent-guides:       {args.indent_guides}")
    print("=" * 65)

    if not repos_dir.exists():
        print(f"ERROR: repos-dir not found: {repos_dir}", file=sys.stderr)
        sys.exit(1)

    all_records = []
    repo_dirs = sorted(p for p in repos_dir.iterdir()
                       if p.is_dir() and not p.name.startswith("."))
    print(f"Found {len(repo_dirs)} repos in {repos_dir}")

    for repo_path in repo_dirs:
        records = process_repo(repo_path, output_dir, font, args)
        all_records.extend(records)
        print(f"  {repo_path.name:20s}: {len(records):5d} images")

    print(f"\nTotal images: {len(all_records):,}")
    print(f"Output dir:   {output_dir}")


if __name__ == "__main__":
    main()
