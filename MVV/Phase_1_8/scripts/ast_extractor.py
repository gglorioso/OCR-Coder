#!/usr/bin/env python3
"""
ast_extractor.py — Phase 1.8: AST-to-patch-grid coordinate mapping

For each Python source file visible in a rendered MVV image, extract every
FunctionDef / AsyncFunctionDef / ClassDef node and compute which rows of the
8×8 SigLIP patch grid each node occupies.

Coordinate pipeline:
  1. AST lineno  (1-indexed, absolute in source file)
  2. Effective line in rendered window  (0-indexed, relative to anchor_line)
  3. Pixel coordinates in 800-px canvas space
  4. Pixel coordinates in 448-px SigLIP space  (scale = 448 / 800)
  5. Grid row indices  (patch row = pixel_y // (448 // grid_size))

Usage:
    python ast_extractor.py \\
        --py-dir  <path/to/python/files> \\
        --manifest <path/to/manifest.jsonl> \\
        --out-path <path/to/ground_truth.jsonl> \\
        --line-height 20 \\
        --canvas-size 800 \\
        --output-size 448 \\
        --grid-size 8
"""

import argparse
import ast
import json
import sys
import warnings
from pathlib import Path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract AST node → patch-grid row mappings for MVV images."
    )
    p.add_argument(
        "--py-dir", required=True, type=Path,
        help="Directory containing the Python source files (or a flat list).",
    )
    p.add_argument(
        "--manifest", default=None, type=Path,
        help="Optional Phase 1.1-style JSONL manifest with anchor_line per file.",
    )
    p.add_argument(
        "--out-path", required=True, type=Path,
        help="Output JSONL path for ground-truth records.",
    )
    p.add_argument(
        "--line-height", type=int, default=20,
        help="Line height in pixels in the 800-px canvas space (default: 20).",
    )
    p.add_argument(
        "--canvas-size", type=int, default=800,
        help="Canvas side length in pixels before downscaling (default: 800).",
    )
    p.add_argument(
        "--output-size", type=int, default=448,
        help="SigLIP input side length in pixels after downscaling (default: 448).",
    )
    p.add_argument(
        "--grid-size", type=int, default=8,
        help="Number of patch-grid rows (and columns) per side (default: 8).",
    )
    p.add_argument(
        "--max-rows", type=int, default=40,
        help="Maximum number of source lines rendered per image (default: 40).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: Path) -> dict:
    """
    Load a Phase 1.1 manifest JSONL and return a mapping from the image stem
    (e.g. 'black__src__black__lines_py') to anchor_line (int).

    The manifest records look like:
        {"image": "data_mvv/images/black__src__black__lines_py.png",
         "anchor_line": 10, ...}

    We key by the image stem (no directory, no extension) so we can match
    against Python source file slugs produced by the same naming scheme.
    """
    anchor_map: dict = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            image_field = rec.get("image", "")
            stem = Path(image_field).stem          # e.g. 'black__src__black__lines_py'
            anchor = rec.get("anchor_line", 0)
            anchor_map[stem] = anchor
    return anchor_map


# ---------------------------------------------------------------------------
# Slug helper (mirrors gen_images.py)
# ---------------------------------------------------------------------------

def path_to_slug(rel_path: str) -> str:
    """Convert a relative path to the same slug used in image filenames."""
    return rel_path.replace("/", "__").replace("\\", "__").replace(".", "_")


def file_to_stem(py_file: Path, py_dir: Path) -> str:
    """
    Derive the image stem for a .py file.

    Strategy: try to reconstruct '<repo>__<slug>' if the file is nested under a
    repo subdirectory; otherwise fall back to using the file's own slug relative
    to py_dir.

    The stem is used both as the output 'file' field and for manifest lookup.
    """
    try:
        rel = py_file.relative_to(py_dir)
    except ValueError:
        rel = Path(py_file.name)

    parts = rel.parts
    if len(parts) >= 2:
        # First part is the repo name, rest is the file path within the repo
        repo = parts[0]
        file_rel = "/".join(parts[1:])
        return f"{repo}__{path_to_slug(file_rel)}"
    else:
        return path_to_slug(str(rel))


# ---------------------------------------------------------------------------
# Coordinate math
# ---------------------------------------------------------------------------

def compute_grid_rows(
    effective_start: int,
    effective_end: int,
    line_height: int,
    canvas_size: int,
    output_size: int,
    grid_size: int,
) -> list:
    """
    Map a half-open line range [effective_start, effective_end] (0-indexed,
    within the rendered window) to a list of patch-grid row indices.

    All math is done exactly as specified:
      y_start_canvas = effective_start * line_height
      y_end_canvas   = effective_end   * line_height + line_height   (exclusive bottom)

      scale          = output_size / canvas_size
      y_start_out    = y_start_canvas * scale
      y_end_out      = y_end_canvas   * scale

      patch_height   = output_size / grid_size
      row_start      = int(y_start_out // patch_height), clamped [0, grid_size-1]
      row_end        = int((y_end_out - 1) // patch_height), clamped [0, grid_size-1]
    """
    scale = output_size / canvas_size
    patch_height = output_size / grid_size

    y_start_canvas = effective_start * line_height
    y_end_canvas   = effective_end   * line_height + line_height

    y_start_out = y_start_canvas * scale
    y_end_out   = y_end_canvas   * scale

    row_start = int(y_start_out // patch_height)
    row_end   = int((y_end_out - 1) // patch_height)

    row_start = max(0, min(grid_size - 1, row_start))
    row_end   = max(0, min(grid_size - 1, row_end))

    if row_end < row_start:
        return []
    return list(range(row_start, row_end + 1))


# ---------------------------------------------------------------------------
# AST processing
# ---------------------------------------------------------------------------

def extract_nodes(
    py_file: Path,
    stem: str,
    anchor_line: int,
    max_rows: int,
    line_height: int,
    canvas_size: int,
    output_size: int,
    grid_size: int,
) -> tuple:
    """
    Parse `py_file`, find all FunctionDef / AsyncFunctionDef / ClassDef nodes,
    and compute their patch-grid row ranges.

    Returns (records, n_skipped, had_syntax_error).
    """
    try:
        source = py_file.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        warnings.warn(f"Could not read {py_file}: {exc}")
        return [], 0, True

    try:
        tree = ast.parse(source, filename=str(py_file))
    except SyntaxError as exc:
        warnings.warn(f"SyntaxError in {py_file}: {exc}")
        return [], 0, True

    records = []
    n_skipped = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node_type = "def"
        elif isinstance(node, ast.ClassDef):
            node_type = "class"
        else:
            continue

        lineno     = node.lineno                       # 1-indexed, absolute
        end_lineno = getattr(node, "end_lineno", None)
        if end_lineno is None:
            end_lineno = lineno                        # fallback for older Python

        # Convert to 0-indexed, subtract anchor to get position in rendered window
        effective_start = (lineno - 1) - anchor_line
        effective_end   = (end_lineno - 1) - anchor_line

        # Clamp end to last rendered line
        effective_end = min(effective_end, max_rows - 1)

        # Skip nodes that start outside the rendered window
        if effective_start >= max_rows:
            n_skipped += 1
            continue

        # Skip nodes entirely above the rendered window
        if effective_end < 0:
            n_skipped += 1
            continue

        # Clamp start (node may begin before anchor but end inside)
        effective_start = max(0, effective_start)

        grid_rows = compute_grid_rows(
            effective_start, effective_end,
            line_height, canvas_size, output_size, grid_size,
        )

        if not grid_rows:
            n_skipped += 1
            continue

        records.append({
            "file":       stem,
            "type":       node_type,
            "name":       node.name,
            "lineno":     lineno,
            "end_lineno": end_lineno,
            "anchor_line": anchor_line,
            "grid_rows":  grid_rows,
        })

    return records, n_skipped, False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Ensure output directory exists
    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load optional manifest
    anchor_map: dict = {}
    if args.manifest is not None:
        if not args.manifest.exists():
            print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
            sys.exit(1)
        anchor_map = load_manifest(args.manifest)
        print(f"Loaded {len(anchor_map):,} anchor entries from manifest.")

    # Collect all .py files
    py_dir = args.py_dir
    if not py_dir.exists():
        print(f"ERROR: --py-dir does not exist: {py_dir}", file=sys.stderr)
        sys.exit(1)

    py_files = sorted(py_dir.rglob("*.py"))
    if not py_files:
        print(f"WARNING: no .py files found under {py_dir}", file=sys.stderr)

    # Process
    total_files     = 0
    total_nodes     = 0
    total_skipped   = 0
    total_errors    = 0

    with open(args.out_path, "w", encoding="utf-8") as out_f:
        for py_file in py_files:
            stem        = file_to_stem(py_file, py_dir)
            anchor_line = anchor_map.get(stem, 0)

            records, n_skipped, had_error = extract_nodes(
                py_file      = py_file,
                stem         = stem,
                anchor_line  = anchor_line,
                max_rows     = args.max_rows,
                line_height  = args.line_height,
                canvas_size  = args.canvas_size,
                output_size  = args.output_size,
                grid_size    = args.grid_size,
            )

            total_files   += 1
            total_nodes   += len(records)
            total_skipped += n_skipped
            if had_error:
                total_errors += 1

            for rec in records:
                out_f.write(json.dumps(rec) + "\n")

    # Summary
    print()
    print("── ast_extractor summary ──────────────────────────────")
    print(f"  Files processed      : {total_files:,}")
    print(f"  Nodes extracted      : {total_nodes:,}")
    print(f"  Nodes skipped (OOW)  : {total_skipped:,}  (out of rendered window)")
    print(f"  Files with errors    : {total_errors:,}  (SyntaxError / unreadable)")
    print(f"  Output written to    : {args.out_path}")
    print("───────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
