#!/usr/bin/env python3
"""
gen_mvv_images.py — Minimum Viable Vision (MVV) monochrome image generator

Produces a 1-file-1-class-1-image dataset for the MVV degradation study
(Test 1.1: Source-File Topological Fingerprinting).

Design:
  - One image per source file (no chunking, perfect class balance)
  - 40-line window anchored to the first real AST node (skips license headers
    and module docstrings so every image starts with structural logic)
  - Rendered in 800x800 grayscale — no syntax highlighting, no colour

Canvas spec (mathematically locked):
  Font:        DejaVu Sans Mono, size 16
  Char width:  10px  (80 chars x 10px = 800px wide)
  Line height: 20px  (40 lines x 20px = 800px tall)
  Canvas:      800 x 800 px, PIL 'L' mode (8-bit grayscale)
  Background:  white (255), text: black (0)

Rendering rules:
  - Lines truncated at col 80 (no wrapping) — preserves indentation topology
  - Chunks < 40 lines padded with blank rows at the bottom
  - Tabs expanded to 4 spaces before truncation

Resolution-as-Test paradigm:
  This script produces the single canonical image per file at full 800x800.
  Downstream, the feature extraction sweep bicubic-downsamples to the four
  SigLIP token budgets (729, 441, 256, 121). The linear probe is trained on
  729-token features and tested at each lower budget — resolution is the only
  variable. No train/test contamination, zero class imbalance.

Output:
  data_mvv/images/{repo}__{relpath_slug}.png  (one per file)
  data_mvv/manifest.jsonl                      (one record per image)

Usage:
    python "Data Crawling/gen_mvv_images.py" \\
        --repos-dir "Scraped Repos" \\
        --output-dir data_mvv
"""

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont

# ── Canvas constants ───────────────────────────────────────────────────────────
FONT_PATH   = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_SIZE   = 16
LINE_HEIGHT = 20     # px (glyph ~14px + 6px leading)
MAX_COLS    = 80     # chars before hard truncation
MAX_ROWS    = 40     # lines per image
CANVAS_W    = MAX_COLS * 10   # 800px  (DejaVu Mono 16px = 10px/char)
CANVAS_H    = MAX_ROWS * LINE_HEIGHT  # 800px
BG_COLOR    = 255   # white
TEXT_COLOR  = 0     # black

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


# ── AST anchor ────────────────────────────────────────────────────────────────

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


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def render_chunk(lines: List[str], font: ImageFont.FreeTypeFont) -> Image.Image:
    """
    Render up to MAX_ROWS lines on an 800x800 grayscale canvas.
    Lines longer than MAX_COLS are hard-truncated (no wrapping).
    Fewer than MAX_ROWS lines → blank rows pad the bottom.
    """
    img  = Image.new("L", (CANVAS_W, CANVAS_H), color=BG_COLOR)
    draw = ImageDraw.Draw(img)
    for i in range(MAX_ROWS):
        if i < len(lines):
            line = lines[i].expandtabs(4)[:MAX_COLS]
        else:
            line = ""
        draw.text((0, i * LINE_HEIGHT), line, font=font, fill=TEXT_COLOR)
    return img


# ── Core pipeline ─────────────────────────────────────────────────────────────

def process_repo(repo_path: Path, output_dir: Path,
                 font: ImageFont.FreeTypeFont) -> List[dict]:
    """Walk one repo, produce one image per valid file, return manifest records."""
    records = []
    images_dir = output_dir / "images"
    repo_name  = repo_path.name

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
            continue  # skip near-empty files

        # Find where real code begins
        anchor = find_topological_start(source)

        # Clamp anchor so we always have room for 40 lines if possible
        # If file is shorter than anchor+40, anchor stays but chunk is padded
        chunk = all_lines[anchor: anchor + MAX_ROWS]

        img_name = f"{repo_name}__{slug(rel_path)}.png"
        img_path = images_dir / img_name

        if not img_path.exists():
            img = render_chunk(chunk, font)
            img.save(img_path, format="PNG", optimize=False)

        records.append({
            "image":        str(img_path),
            "repo":         repo_name,
            "source_file":  f"{repo_name}/{rel_path}",
            "anchor_line":  anchor,
            "n_source_lines": len(all_lines),
        })

    return records


def main():
    parser = argparse.ArgumentParser(description="MVV monochrome image generator")
    parser.add_argument("--repos-dir",  default="Scraped Repos")
    parser.add_argument("--output-dir", default="data_mvv")
    args = parser.parse_args()

    repos_dir  = Path(args.repos_dir)
    output_dir = Path(args.output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)

    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except IOError:
        print(f"ERROR: font not found at {FONT_PATH}", file=sys.stderr)
        sys.exit(1)

    all_records = []
    repo_dirs = sorted(p for p in repos_dir.iterdir()
                       if p.is_dir() and not p.name.startswith("."))
    print(f"Found {len(repo_dirs)} repos in {repos_dir}")

    for repo_path in repo_dirs:
        records = process_repo(repo_path, output_dir, font)
        all_records.extend(records)
        print(f"  {repo_path.name:20s}: {len(records):5d} images")

    manifest_path = output_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nTotal images:  {len(all_records):,}")
    print(f"Unique classes: {len(set(r['source_file'] for r in all_records)):,}")
    print(f"Manifest:       {manifest_path}")

    # Anchor distribution — sanity check
    anchors = [r["anchor_line"] for r in all_records]
    at_zero = sum(1 for a in anchors if a == 0)
    print(f"\nAnchor at line 0 (no docstring skip needed): {at_zero:,} "
          f"({at_zero/len(anchors)*100:.1f}%)")
    print(f"Anchor > 0 (docstring/header skipped):        "
          f"{len(anchors)-at_zero:,} ({(len(anchors)-at_zero)/len(anchors)*100:.1f}%)")


if __name__ == "__main__":
    main()
