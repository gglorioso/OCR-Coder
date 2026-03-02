#!/usr/bin/env python3
"""
gen_labels.py — Ground-truth label generation for MVV Phase 1.2

Reads the Phase 1.1 manifest (which maps image stems to .py source files),
AST-parses each source file, and extracts three integer structural metrics:
  - line_count : total lines in the file
  - n_defs     : number of `def` statements (at any nesting depth)
  - n_classes  : number of `class` statements (at any nesting depth)

Output: exp1_structural_regression/data/labels.jsonl
  One JSON object per successfully parsed file, keyed by the .pt stem so that
  joining with the feature tensors is a single dict lookup.

  {"stem": "black__action__main_py", "line_count": 197, "n_defs": 12, "n_classes": 3}

Files that fail ast.parse() are logged and skipped; their stems will simply be
absent from labels.jsonl. run_regression.py drops features with no matching label.

Usage:
    python gen_labels.py [--manifest PATH] [--scraped_dir PATH] [--out PATH]
"""

import argparse
import ast
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Defaults (resolved relative to this script's location)
# ---------------------------------------------------------------------------

_SCRIPT_DIR   = Path(__file__).parent                          # scripts/
_EXP_DIR      = _SCRIPT_DIR.parent                             # exp1_structural_regression/
_PHASE12_DIR  = _EXP_DIR.parent                                # Phase_1_2/
_MVV_DIR      = _PHASE12_DIR.parent                            # MVV/
_REPO_ROOT    = _MVV_DIR.parent                                # OCR-Coder/

DEFAULT_MANIFEST   = _MVV_DIR / "Phase_1_1" / "data_mvv" / "manifest.jsonl"
DEFAULT_SCRAPED    = _REPO_ROOT / "Scraped Repos"
DEFAULT_OUT        = _EXP_DIR / "data" / "labels.jsonl"


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _count_nodes(tree: ast.AST) -> tuple[int, int]:
    """Return (n_defs, n_classes) by walking the full AST."""
    n_defs    = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
    n_classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
    return n_defs, n_classes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest",    type=Path, default=DEFAULT_MANIFEST,
                        help="Path to Phase 1.1 manifest.jsonl")
    parser.add_argument("--scraped_dir", type=Path, default=DEFAULT_SCRAPED,
                        help="Root of scraped repositories (contains black/, flask/, …)")
    parser.add_argument("--out",         type=Path, default=DEFAULT_OUT,
                        help="Output labels.jsonl path")
    args = parser.parse_args()

    if not args.manifest.exists():
        sys.exit(f"ERROR: manifest not found at {args.manifest}")
    if not args.scraped_dir.exists():
        sys.exit(f"ERROR: scraped_dir not found at {args.scraped_dir}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ── Load manifest ────────────────────────────────────────────────────────
    entries = []
    with open(args.manifest) as f:
        for line in f:
            entries.append(json.loads(line))
    print(f"Manifest entries : {len(entries):,}")

    # ── Parse each source file ───────────────────────────────────────────────
    n_ok = n_skip_missing = n_skip_syntax = 0
    parse_errors: list[tuple[str, str]] = []

    with open(args.out, "w") as out_f:
        for row in entries:
            stem        = Path(row["image"]).stem           # "black__action__main_py"
            source_rel  = row["source_file"]                # "black/action/main.py"
            source_path = args.scraped_dir / source_rel

            # ── Missing file ─────────────────────────────────────────────────
            if not source_path.exists():
                n_skip_missing += 1
                if n_skip_missing <= 5:
                    print(f"  MISSING : {source_path}")
                elif n_skip_missing == 6:
                    print("  (further missing-file messages suppressed)")
                continue

            # ── Read & parse ─────────────────────────────────────────────────
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

            line_count        = len(source_text.splitlines())
            n_defs, n_classes = _count_nodes(tree)

            record = {
                "stem":       stem,
                "line_count": line_count,
                "n_defs":     n_defs,
                "n_classes":  n_classes,
            }
            out_f.write(json.dumps(record) + "\n")
            n_ok += 1

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("LABEL GENERATION COMPLETE")
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
