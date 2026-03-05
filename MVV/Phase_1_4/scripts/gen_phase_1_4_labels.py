#!/usr/bin/env python3
"""
gen_phase_1_4_labels.py — Generate syntax-style labels for MVV Phase 1.4

Reads an existing labels.jsonl (or a manifest.jsonl) that contains per-image
entries with enough information to locate the source .py file and the 40-line
window shown in the image.  Computes three labels for each window:

  nesting_depth (int 0/1/2)
    Uses ast.parse() on the window text; walks the AST to find the deepest
    block-nesting level.  Bins: 0 = depth 0-1, 1 = depth 2-3, 2 = depth 4+.
    On parse failure (partial window may be invalid syntax), falls back to a
    line-scan proxy: max(len(line) - len(line.lstrip())) // 4 over non-empty
    lines, then bins the same way.

  is_tabs (int 0/1)
    1 if any line in the window starts with a literal tab character, else 0.

  keyword_density (int)
    Count of keyword tokens {if, for, while, def, class, return, import}
    using the tokenize module so only real keyword tokens are counted (not
    substrings inside identifiers).  Falls back to a simple word-split count
    if tokenize raises an exception.

─────────────────────────────────────────────────────────────────────────────
Accepted input formats for --labels_jsonl
─────────────────────────────────────────────────────────────────────────────
Format A — data_v2b train/val manifest (PRIMARY, recommended):
  {"id": "black__file_monokai__task", "image": "...", "repo": "black",
   "source_file": "path/to/file.py", "start_line": 1, ...}
  • stem is derived from the image filename stem (repo__file__style).
  • anchor_line = start_line - 1  (start_line is 1-indexed in the manifest).
  • Full .py path = --py_dir / repo / source_file.
  • Multiple task_type entries per image are automatically deduplicated.

Format B — Phase 1.1 manifest:
  {"image": "data_mvv/images/black__action__main_py.png",
   "source_file": "black/action/main.py", "anchor_line": 0, ...}
  • stem is derived from the image filename stem.
  • Full .py path = --py_dir / source_file  (repo already prefixed).
  • anchor_line taken directly from the field.

Format C — Explicit labels format:
  {"stem": "black__file_monokai", "repo": "black",
   "source_file": "path/to/file.py", "anchor_line": 0}

─────────────────────────────────────────────────────────────────────────────
Output format per line:
  {"stem": "...", "nesting_depth": 0, "is_tabs": 0, "keyword_density": 5}

The stem matches .pt filenames in ./precomputed_features_tiled/ so that
run_probe_1_4.py can load features without any further mapping.

Usage:
    python gen_phase_1_4_labels.py \\
        --py_dir   "/path/to/Scraped Repos" \\
        --labels_jsonl  data_v2b/manifests/train.jsonl \\
        [--output  MVV/Phase_1_4/data/labels_1_4.jsonl]
"""

import argparse
import ast
import io
import json
import sys
import tokenize
from collections import Counter
from pathlib import Path
from statistics import mean


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_SIZE     = 40   # lines rendered per image
KEYWORD_TARGETS = frozenset({"if", "for", "while", "def", "class", "return", "import"})

_SCRIPT_DIR = Path(__file__).parent
_PHASE_DIR  = _SCRIPT_DIR.parent
_REPO_ROOT  = _PHASE_DIR.parent.parent

DEFAULT_OUTPUT = _PHASE_DIR / "data" / "labels_1_4.jsonl"


# ---------------------------------------------------------------------------
# Nesting-depth helpers
# ---------------------------------------------------------------------------

class _MaxDepthVisitor(ast.NodeVisitor):
    """AST visitor that records the maximum block-nesting depth seen."""

    # Node types that open a new indented block / scope
    _BLOCK_TYPES = (
        ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
        ast.For, ast.AsyncFor, ast.While, ast.If, ast.With,
        ast.AsyncWith, ast.Try, ast.ExceptHandler,
    )

    def __init__(self) -> None:
        self.max_depth = 0
        self._depth    = 0

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, self._BLOCK_TYPES):
            self._depth += 1
            if self._depth > self.max_depth:
                self.max_depth = self._depth
            super().generic_visit(node)
            self._depth -= 1
        else:
            super().generic_visit(node)


def _bin_depth(depth: int) -> int:
    """Map raw nesting depth to ordinal label 0/1/2."""
    if depth <= 1:
        return 0
    elif depth <= 3:
        return 1
    else:
        return 2


def compute_nesting_depth(window_text: str) -> int:
    """
    Returns binned nesting depth (0/1/2) for the code window.
    Falls back to indent-based scan when ast.parse fails on a partial window.
    """
    try:
        tree    = ast.parse(window_text)
        visitor = _MaxDepthVisitor()
        visitor.visit(tree)
        return _bin_depth(visitor.max_depth)
    except SyntaxError:
        # Fallback: approximate depth from indentation level
        max_indent = 0
        for line in window_text.splitlines():
            stripped = line.lstrip()
            if not stripped:
                continue
            leading  = line[: len(line) - len(stripped)]
            spaces   = leading.replace("\t", "    ")
            indent   = len(spaces) // 4
            if indent > max_indent:
                max_indent = indent
        return _bin_depth(max_indent)


# ---------------------------------------------------------------------------
# is_tabs
# ---------------------------------------------------------------------------

def compute_is_tabs(window_lines: list[str]) -> int:
    """Return 1 if any line starts with a tab character, else 0."""
    for line in window_lines:
        if line.startswith("\t"):
            return 1
    return 0


# ---------------------------------------------------------------------------
# keyword_density
# ---------------------------------------------------------------------------

def compute_keyword_density(window_text: str) -> int:
    """
    Count occurrences of KEYWORD_TARGETS using the tokenize module.
    Only genuine keyword tokens are counted (not substrings of identifiers).
    Falls back to word-split counting if tokenize raises an error.
    """
    count = 0
    try:
        tokens = tokenize.generate_tokens(io.StringIO(window_text).readline)
        for tok_type, tok_string, _, _, _ in tokens:
            if tok_type == tokenize.NAME and tok_string in KEYWORD_TARGETS:
                count += 1
    except (tokenize.TokenError, IndentationError):
        # Partial windows can confuse the tokenizer — fall back to word-split
        words = window_text.replace("\n", " ").replace("\t", " ").split()
        count = sum(1 for w in words if w in KEYWORD_TARGETS)
    return count


# ---------------------------------------------------------------------------
# Input parsing — detect and normalise all three input formats
# ---------------------------------------------------------------------------

def _detect_format(first_row: dict) -> str:
    """
    Return 'A' (data_v2b), 'B' (Phase 1.1 manifest), or 'C' (explicit).
    """
    if "repo" in first_row and "start_line" in first_row and "image" in first_row:
        return "A"
    if "image" in first_row and "anchor_line" in first_row and "repo" not in first_row:
        return "B"
    if "stem" in first_row and "anchor_line" in first_row:
        return "C"
    # Default: try B-like if image present, otherwise C-like
    if "image" in first_row:
        return "B"
    return "C"


def load_labels_jsonl(path: Path, py_dir: Path) -> list[dict]:
    """
    Load and normalise entries from any supported labels.jsonl format.
    Returns a deduplicated list of dicts with keys:
      stem, py_path (Path), anchor_line (int)
    """
    raw_rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                raw_rows.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                print(f"  WARNING line {lineno}: JSON error — {exc}", file=sys.stderr)

    if not raw_rows:
        print("ERROR: labels.jsonl is empty or unreadable.", file=sys.stderr)
        sys.exit(1)

    fmt = _detect_format(raw_rows[0])
    print(f"  Detected input format: {fmt!r}  ({len(raw_rows):,} raw rows)")

    seen: dict[str, dict] = {}  # stem → normalised entry (deduplication)

    for row in raw_rows:
        try:
            if fmt == "A":
                # data_v2b manifest
                stem       = Path(row["image"]).stem          # e.g. black__file_c0_monokai
                repo       = row["repo"]                       # e.g. "black"
                src_rel    = row["source_file"]                # relative to repo dir
                anchor     = int(row["start_line"]) - 1       # convert 1-indexed to 0-indexed
                full_path  = py_dir / repo / src_rel

            elif fmt == "B":
                # Phase 1.1 manifest
                stem       = Path(row["image"]).stem
                src_rel    = row["source_file"]                # already prefixed with repo
                anchor     = int(row["anchor_line"])
                full_path  = py_dir / src_rel

            else:
                # Explicit format C
                stem      = row["stem"]
                anchor    = int(row["anchor_line"])
                src_rel   = row.get("source_file", "")
                if not src_rel:
                    continue
                if "repo" in row:
                    full_path = py_dir / row["repo"] / src_rel
                else:
                    full_path = py_dir / src_rel

        except (KeyError, ValueError) as exc:
            print(f"  WARNING: skipping malformed row — {exc}", file=sys.stderr)
            continue

        # Deduplicate by stem (multiple task_type entries share the same stem)
        if stem not in seen:
            seen[stem] = {
                "stem":        stem,
                "py_path":     full_path,
                "anchor_line": anchor,
            }

    deduped = list(seen.values())
    print(f"  After deduplication: {len(deduped):,} unique image stems")
    return deduped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate syntax-style labels for MVV Phase 1.4."
    )
    parser.add_argument(
        "--py_dir",
        type=Path,
        required=True,
        help='Root directory of scraped .py repos (e.g. "Scraped Repos"). '
             'For Format A (data_v2b), files are at py_dir/repo/source_file. '
             'For Format B (Phase 1.1 manifest), files are at py_dir/source_file.',
    )
    parser.add_argument(
        "--labels_jsonl",
        type=Path,
        required=True,
        help="Input: data_v2b train.jsonl, Phase 1.1 manifest.jsonl, or "
             "a custom labels.jsonl with stem/anchor_line fields.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output path for Phase 1.4 labels (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    if not args.py_dir.exists():
        sys.exit(f"ERROR: --py_dir not found: {args.py_dir}")
    if not args.labels_jsonl.exists():
        sys.exit(f"ERROR: --labels_jsonl not found: {args.labels_jsonl}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # ── Load and normalise entries ─────────────────────────────────────────────
    print(f"Loading labels from: {args.labels_jsonl}")
    rows = load_labels_jsonl(args.labels_jsonl, args.py_dir)

    # ── Process each entry ────────────────────────────────────────────────────
    n_written       = 0
    n_skip_missing  = 0
    n_skip_empty    = 0
    skipped_missing = []

    nesting_counts  = Counter()   # for class distribution summary
    tabs_counts     = Counter()
    kw_densities    = []          # for mean/max stats

    with open(args.output, "w", encoding="utf-8") as out_f:
        for idx, row in enumerate(rows):
            if idx > 0 and idx % 1000 == 0:
                print(f"  Progress: {idx:,} / {len(rows):,}  "
                      f"(written={n_written:,}, skipped={n_skip_missing+n_skip_empty})")

            stem        = row["stem"]
            py_path     = row["py_path"]
            anchor_line = row["anchor_line"]

            # ── Locate and read the source file ───────────────────────────────
            if not py_path.exists():
                n_skip_missing += 1
                skipped_missing.append(str(stem))
                continue

            try:
                source_lines = py_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
            except OSError as exc:
                n_skip_missing += 1
                skipped_missing.append(str(stem))
                print(f"  WARN: could not read {py_path}: {exc}", file=sys.stderr)
                continue

            # ── Extract window ─────────────────────────────────────────────────
            window_lines = source_lines[anchor_line : anchor_line + WINDOW_SIZE]
            if not window_lines:
                # anchor_line is beyond end of file
                n_skip_empty += 1
                continue

            window_text = "\n".join(window_lines)

            # ── Compute labels ─────────────────────────────────────────────────
            nesting_depth   = compute_nesting_depth(window_text)
            is_tabs         = compute_is_tabs(window_lines)
            keyword_density = compute_keyword_density(window_text)

            record = {
                "stem":            stem,
                "nesting_depth":   nesting_depth,
                "is_tabs":         is_tabs,
                "keyword_density": keyword_density,
            }
            out_f.write(json.dumps(record) + "\n")
            n_written += 1

            # Accumulate stats
            nesting_counts[nesting_depth] += 1
            tabs_counts[is_tabs]          += 1
            kw_densities.append(keyword_density)

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("PHASE 1.4 LABEL GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Written        : {n_written:,} rows  →  {args.output}")
    print(f"  Skipped missing: {n_skip_missing}  (source file not found / unreadable)")
    print(f"  Skipped empty  : {n_skip_empty}  (anchor_line beyond EOF)")

    if skipped_missing:
        shown = skipped_missing[:5]
        extra = f"  ... +{len(skipped_missing)-5} more" if len(skipped_missing) > 5 else ""
        print(f"  First skipped  : {shown}{extra}")

    if n_written == 0:
        print("\n  WARNING: no records written — check --py_dir and --labels_jsonl paths.")
        return

    print()
    print("  nesting_depth class distribution:")
    labels_desc = {0: "0-1 (shallow)", 1: "2-3 (medium)", 2: "4+  (deep)"}
    for cls in (0, 1, 2):
        cnt = nesting_counts[cls]
        pct = 100.0 * cnt / n_written
        print(f"    class {cls} [{labels_desc[cls]}]: {cnt:,}  ({pct:.1f}%)")

    print()
    print("  is_tabs distribution:")
    for cls in (0, 1):
        cnt   = tabs_counts[cls]
        pct   = 100.0 * cnt / n_written
        label = "spaces" if cls == 0 else "tabs"
        print(f"    is_tabs={cls} [{label}]: {cnt:,}  ({pct:.1f}%)")

    print()
    print("  keyword_density stats:")
    print(f"    mean : {mean(kw_densities):.2f}")
    print(f"    max  : {max(kw_densities)}")
    print(f"    min  : {min(kw_densities)}")


if __name__ == "__main__":
    main()
