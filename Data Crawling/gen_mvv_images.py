#!/usr/bin/env python3
"""
gen_mvv_images.py — Minimum Viable Vision (MVV) monochrome image generator

Produces a controlled, scientifically clean dataset for Test 1.1 of the
MVV degradation study. Each image is a 40-line chunk of a Python source
file rendered in grayscale — no syntax highlighting, no color.

Canvas spec (mathematically locked):
  Font:        DejaVu Sans Mono, size 16
  Char width:  10px  (80 chars × 10px = 800px wide)
  Line height: 20px  (40 lines × 20px = 800px tall)
  Canvas:      800 × 800 px, PIL 'L' mode (8-bit grayscale)
  Background:  white (255), text: black (0)

Rendering rules:
  - Horizontal truncation: lines > 80 chars are sliced at char 80 (no wrap)
  - Vertical padding: chunks < 40 lines padded with blank lines at the bottom
  - Non-overlapping 40-line chunks per file (for multi-sample-per-class probe)

Output:
  data_mvv/images/{repo}__{relpath_slug}__c{chunk_idx}.png
  data_mvv/manifest.jsonl  — one record per image

Usage:
    python "Data Crawling/gen_mvv_images.py" \\
        --repos-dir "Scraped Repos" \\
        --output-dir data_mvv \\
        [--min-lines 40]   # skip files shorter than this (default: 40)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

from PIL import Image, ImageDraw, ImageFont

# ── Constants ─────────────────────────────────────────────────────────────────
FONT_PATH   = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_SIZE   = 16
CHAR_WIDTH  = 10     # px — DejaVu Mono size 16 measured
LINE_HEIGHT = 20     # px — forced (glyph ~14px + 6px leading)
MAX_COLS    = 80     # characters per line before truncation
MAX_ROWS    = 40     # lines per chunk
CANVAS_W    = MAX_COLS * CHAR_WIDTH   # 800
CANVAS_H    = MAX_ROWS * LINE_HEIGHT  # 800
BG_COLOR    = 255    # white
TEXT_COLOR  = 0      # black

SKIP_PATTERNS = [
    "/test/", "/tests/", "test_", "_test.py", "conftest.py",
    "/vendor/", "/vendored/", "/third_party/", "/_vendor/",
    "/migrations/", "/generated/", "__pycache__",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def should_skip(rel_path: str) -> bool:
    p = rel_path.lower()
    if any(pat in p for pat in SKIP_PATTERNS):
        return True
    if Path(rel_path).name == "__init__.py":
        return True
    return False


def slug(rel_path: str) -> str:
    """Convert relative path to a flat filename-safe string."""
    return rel_path.replace("/", "__").replace("\\", "__").replace(".", "_")


def render_chunk(lines: List[str], font: ImageFont.FreeTypeFont) -> Image.Image:
    """
    Render up to MAX_ROWS lines onto an 800×800 grayscale canvas.

    Rules:
      - Lines are truncated at MAX_COLS characters (no wrapping).
      - Fewer than MAX_ROWS lines → blank rows pad the bottom.
      - Output is PIL 'L' mode (8-bit grayscale).
    """
    img  = Image.new("L", (CANVAS_W, CANVAS_H), color=BG_COLOR)
    draw = ImageDraw.Draw(img)

    for i in range(MAX_ROWS):
        if i < len(lines):
            # Expand tabs, truncate, strip trailing whitespace only
            line = lines[i].expandtabs(4)[:MAX_COLS]
        else:
            line = ""
        y = i * LINE_HEIGHT
        draw.text((0, y), line, font=font, fill=TEXT_COLOR)

    return img


# ── Core pipeline ─────────────────────────────────────────────────────────────

def process_repo(repo_path: Path, output_dir: Path, font: ImageFont.FreeTypeFont,
                 min_lines: int) -> List[Dict]:
    """Walk one repo, render chunks, return manifest records."""
    records = []
    repo_name = repo_path.name
    images_dir = output_dir / "images"

    for py_file in sorted(repo_path.rglob("*.py")):
        rel_path = py_file.relative_to(repo_path).as_posix()

        if should_skip(rel_path):
            continue

        try:
            text = py_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        all_lines = text.splitlines()
        n_lines   = len(all_lines)

        if n_lines < min_lines:
            continue  # too short for even one meaningful chunk

        # Non-overlapping 40-line chunks
        chunks = [all_lines[i:i + MAX_ROWS]
                  for i in range(0, n_lines - MAX_ROWS + 1, MAX_ROWS)]

        if not chunks:
            continue

        file_slug = slug(rel_path)

        for chunk_idx, chunk_lines in enumerate(chunks):
            img_name = f"{repo_name}__{file_slug}__c{chunk_idx:03d}.png"
            img_path = images_dir / img_name

            if not img_path.exists():
                img = render_chunk(chunk_lines, font)
                img.save(img_path, format="PNG", optimize=False)

            records.append({
                "image":         str(img_path),
                "repo":          repo_name,
                "source_file":   f"{repo_name}/{rel_path}",
                "chunk_idx":     chunk_idx,
                "start_line":    chunk_idx * MAX_ROWS,
                "end_line":      chunk_idx * MAX_ROWS + len(chunk_lines) - 1,
                "n_source_lines": n_lines,
            })

    return records


def main():
    parser = argparse.ArgumentParser(description="MVV monochrome image generator")
    parser.add_argument("--repos-dir",  default="Scraped Repos",
                        help="Directory containing cloned repos")
    parser.add_argument("--output-dir", default="data_mvv",
                        help="Root output directory (images/ and manifest.jsonl go here)")
    parser.add_argument("--min-lines",  type=int, default=40,
                        help="Skip files with fewer than this many lines (default: 40)")
    args = parser.parse_args()

    repos_dir  = Path(args.repos_dir)
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Load font once
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except IOError:
        print(f"ERROR: Font not found at {FONT_PATH}", file=sys.stderr)
        sys.exit(1)

    manifest_path = output_dir / "manifest.jsonl"
    all_records: List[Dict] = []

    repo_dirs = sorted(p for p in repos_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    print(f"Found {len(repo_dirs)} repos in {repos_dir}")

    for repo_path in repo_dirs:
        records = process_repo(repo_path, output_dir, font, args.min_lines)
        all_records.extend(records)
        print(f"  {repo_path.name:20s}: {len(records):5d} chunks")

    with open(manifest_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nTotal chunks (images): {len(all_records)}")
    print(f"Manifest written to:   {manifest_path}")

    # Quick label-count summary
    from collections import Counter
    label_counts = Counter(r["source_file"] for r in all_records)
    n_classes    = len(label_counts)
    multi_sample = sum(1 for c in label_counts.values() if c >= 3)
    print(f"Unique source files:   {n_classes}")
    print(f"Files with ≥3 chunks:  {multi_sample}  (usable for linear probe)")


if __name__ == "__main__":
    main()
