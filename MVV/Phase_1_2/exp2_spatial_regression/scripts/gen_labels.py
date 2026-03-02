#!/usr/bin/env python3
"""
gen_labels.py — Windowed ground-truth labels for MVV Phase 1.2 Exp2

Corrects the label misalignment from Exp1: the image shows exactly 40 lines
anchored at `anchor_line` (0-indexed), so we must only count AST nodes that
fall within that visible window. Counting the entire file (as Exp1 did) asks the
probe to predict structure it cannot possibly see.

Window definition:
  anchor_line       — 0-indexed first visible line (from manifest)
  visible range     — 1-indexed AST linenos: [anchor_line+1, anchor_line+40]

Targets written per file:
  line_count  — visible lines = min(40, n_source_lines - anchor_line)
  n_defs      — def/async def statements whose lineno is in the window
  n_classes   — class statements whose lineno is in the window

Output: exp2_spatial_regression/data/labels.jsonl
  {"stem": "black__action__main_py", "line_count": 40, "n_defs": 3, "n_classes": 1}

Files failing ast.parse() are logged and skipped.

Usage:
    python gen_labels.py [--manifest PATH] [--scraped_dir PATH] [--out PATH]
"""

import argparse
import ast
import json
import sys
from pathlib import Path


MAX_ROWS = 40   # lines rendered per image (must match gen_images.py)

_SCRIPT_DIR  = Path(__file__).parent
_EXP_DIR     = _SCRIPT_DIR.parent
_MVV_DIR     = _EXP_DIR.parent.parent
_REPO_ROOT   = _MVV_DIR.parent

DEFAULT_MANIFEST  = _MVV_DIR / "Phase_1_1" / "data_mvv" / "manifest.jsonl"
DEFAULT_SCRAPED   = _REPO_ROOT / "Scraped Repos"
DEFAULT_OUT       = _EXP_DIR / "data" / "labels.jsonl"


def _count_visible_nodes(tree: ast.AST, anchor_line: int) -> tuple[int, int]:
    """
    Count def/class nodes whose 1-indexed lineno falls in the visible window.
    anchor_line is 0-indexed, so the visible window (1-indexed) is:
        [anchor_line + 1, anchor_line + MAX_ROWS]
    """
    lo = anchor_line + 1
    hi = anchor_line + MAX_ROWS

    n_defs = sum(
        1 for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and lo <= node.lineno <= hi
    )
    n_classes = sum(
        1 for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and lo <= node.lineno <= hi
    )
    return n_defs, n_classes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest",    type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--scraped_dir", type=Path, default=DEFAULT_SCRAPED)
    parser.add_argument("--out",         type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.manifest.exists():
        sys.exit(f"ERROR: manifest not found at {args.manifest}")
    if not args.scraped_dir.exists():
        sys.exit(f"ERROR: scraped_dir not found at {args.scraped_dir}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ── Load manifest (need anchor_line and n_source_lines per stem) ──────────
    manifest_rows: dict[str, dict] = {}
    with open(args.manifest) as f:
        for line in f:
            row  = json.loads(line)
            stem = Path(row["image"]).stem
            manifest_rows[stem] = row
    print(f"Manifest entries : {len(manifest_rows):,}")

    # ── Parse each source file, count within visible window ───────────────────
    n_ok = n_skip_missing = n_skip_syntax = 0
    parse_errors: list[tuple[str, str]] = []

    with open(args.out, "w") as out_f:
        for stem, row in manifest_rows.items():
            source_path  = args.scraped_dir / row["source_file"]
            anchor_line  = int(row["anchor_line"])
            n_src_lines  = int(row["n_source_lines"])

            if not source_path.exists():
                n_skip_missing += 1
                if n_skip_missing <= 5:
                    print(f"  MISSING : {source_path}")
                elif n_skip_missing == 6:
                    print("  (further missing-file messages suppressed)")
                continue

            try:
                source_text = source_path.read_text(encoding="utf-8", errors="replace")
                tree        = ast.parse(source_text)
            except SyntaxError as exc:
                n_skip_syntax += 1
                parse_errors.append((stem, str(exc)))
                if n_skip_syntax <= 10:
                    print(f"  SYNTAX  : {stem}  ({exc})")
                elif n_skip_syntax == 11:
                    print("  (further syntax-error messages suppressed)")
                continue

            line_count        = min(MAX_ROWS, n_src_lines - anchor_line)
            n_defs, n_classes = _count_visible_nodes(tree, anchor_line)

            record = {
                "stem":       stem,
                "line_count": line_count,
                "n_defs":     n_defs,
                "n_classes":  n_classes,
            }
            out_f.write(json.dumps(record) + "\n")
            n_ok += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("WINDOWED LABEL GENERATION COMPLETE")
    print("=" * 50)
    print(f"  Written        : {n_ok:,} rows → {args.out}")
    print(f"  Skipped missing: {n_skip_missing}")
    print(f"  Skipped syntax : {n_skip_syntax}")
    if parse_errors:
        print("\n  First syntax errors:")
        for stem, msg in parse_errors[:5]:
            print(f"    {stem}: {msg}")


if __name__ == "__main__":
    main()
