#!/usr/bin/env python3
"""
data_gen_2b.py — Phase 2b data pipeline

Processes ALL valid Python files from Scraped Repos:
1. Chunks files >500 lines into 500-line segments (one image per chunk)
2. Skips chunks whose image already exists (idempotent reruns)
3. Renders new monokai PNG images
4. Generates up to 6 AST label types per chunk
5. Repo-level 90/5/5 split (prevents data leakage across splits)
6. Writes JSONL manifests to --output-dir/manifests/

Key differences from Phase 2a (simple_data_gen.py):
- No sampling cap — uses all 6,954+ valid files
- Chunked rendering (500 lines/image instead of whole file)
- Unique naming: {repo}__{relpath}[_c{N}]_monokai (no collision risk)
- 6th task: function_explanation (using per-function docstrings)
- Repo-level split instead of random split
- Idempotent: skips images already on disk

Usage:
    python data_gen_2b.py \\
        --repos-dir ~/CoderOCR/OCR-Coder/Scraped\\ Repos \\
        --output-dir ~/CoderOCR/OCR-Coder/data_v2b \\
        --features-dir ~/CoderOCR/OCR-Coder/precomputed_features_tiled
"""

import argparse
import ast
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from code_to_image import convert_string_to_image

# ── Constants ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = 500       # lines per image
MIN_CHUNK_LINES = 50   # skip tiny tail chunks
STYLE = "monokai"
FONT_SIZE = 13


# ── File Discovery ─────────────────────────────────────────────────────────────

def should_skip_file(rel_path: str) -> bool:
    p = rel_path.lower()
    skip_patterns = [
        "/test/", "/tests/", "test_", "_test.py", "conftest.py",
        "/vendor/", "/vendored/", "/third_party/", "/_vendor/",
        "/migrations/", "/generated/", "__pycache__",
    ]
    if any(pat in p for pat in skip_patterns):
        return True
    if rel_path.endswith("__init__.py"):
        return True
    return False


def discover_files(repos_dir: Path) -> List[Dict]:
    """Walk all repos and return metadata for every valid unique Python file."""
    print(f"Discovering Python files in {repos_dir} ...")
    files: List[Dict] = []
    seen_hashes: set = set()

    for repo_path in sorted(repos_dir.iterdir()):
        if not repo_path.is_dir() or repo_path.name.startswith("."):
            continue
        repo_name = repo_path.name
        repo_count = 0

        for py_file in repo_path.rglob("*.py"):
            rel_path = str(py_file.relative_to(repo_path))
            if should_skip_file(rel_path):
                continue

            try:
                source = py_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            lines = source.splitlines()
            line_count = len(lines)
            if not (50 <= line_count <= 2500):
                continue
            if len(source) < 500:
                continue

            try:
                ast.parse(source)
            except SyntaxError:
                continue

            h = hashlib.sha256(source.encode()).hexdigest()[:16]
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            files.append({
                "repo": repo_name,
                "rel_path": rel_path,
                "abs_path": str(py_file),
                "line_count": line_count,
                "source": source,
            })
            repo_count += 1

        print(f"  {repo_name}: {repo_count} files")

    print(f"Total: {len(files)} valid unique Python files")
    return files


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_source(source: str, chunk_size: int = CHUNK_SIZE) -> List[Tuple[int, int, str]]:
    """
    Split source into chunks of at most chunk_size lines.
    Returns list of (start_line_1indexed, end_line_1indexed, code_str).
    """
    lines = source.splitlines(keepends=True)
    chunks = []
    i = 0
    while i < len(lines):
        chunk_lines = lines[i : i + chunk_size]
        chunks.append((i + 1, i + len(chunk_lines), "".join(chunk_lines)))
        i += chunk_size
    return chunks


def make_stem(repo: str, rel_path: str, chunk_idx: int, n_chunks: int) -> str:
    """Unique, filesystem-safe stem for image/feature files."""
    safe = rel_path.replace("/", "__").replace("\\", "__").replace(".py", "")
    base = f"{repo}__{safe}"
    if n_chunks == 1:
        return f"{base}_{STYLE}"
    return f"{base}_c{chunk_idx}_{STYLE}"


# ── AST Label Generation ───────────────────────────────────────────────────────

def extract_chunk_labels(
    source: str,
    start_line: int,
    end_line: int,
) -> List[Dict]:
    """
    Generate up to 6 Q&A pairs for the given line range using the file's AST.

    The full file is parsed (not just the chunk) so the AST is always valid.
    Only nodes whose lineno falls within [start_line, end_line] are used.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    def in_range(node) -> bool:
        return hasattr(node, "lineno") and start_line <= node.lineno <= end_line

    examples = []

    # Task 1: function_listing
    funcs = [
        n.name for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and in_range(n)
    ]
    if funcs:
        func_list = "\n".join(f"{i+1}. {name}" for i, name in enumerate(funcs[:20]))
        examples.append({
            "task": "function_listing",
            "question": "List all functions defined in this code.",
            "answer": f"This file defines the following functions:\n{func_list}",
        })

    # Task 2: class_listing
    classes = [
        n.name for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and in_range(n)
    ]
    if classes:
        class_list = "\n".join(f"{i+1}. {name}" for i, name in enumerate(classes[:20]))
        examples.append({
            "task": "class_listing",
            "question": "List all classes defined in this code.",
            "answer": f"This file defines the following classes:\n{class_list}",
        })

    # Task 3: import_listing (imports visible in this chunk)
    imports: set = set()
    for node in ast.walk(tree):
        if not in_range(node):
            continue
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    if imports:
        import_list = "\n".join(f"- {name}" for name in sorted(imports)[:30])
        examples.append({
            "task": "import_listing",
            "question": "What modules does this code import?",
            "answer": f"This code imports the following modules:\n{import_list}",
        })

    # Task 4: function_signatures
    sigs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and in_range(node):
            args = [arg.arg for arg in node.args.args]
            sigs.append(f"def {node.name}({', '.join(args)})")
            if len(sigs) >= 10:
                break
    if sigs:
        sig_list = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sigs))
        examples.append({
            "task": "function_signatures",
            "question": "What are the function signatures in this code?",
            "answer": f"Function signatures in this file:\n{sig_list}",
        })

    # Task 5: description (first meaningful docstring in chunk range)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and in_range(node):
            doc = ast.get_docstring(node)
            if doc and len(doc) > 30:
                examples.append({
                    "task": "description",
                    "question": "Describe what this code does.",
                    "answer": doc[:500],
                })
                break

    # Task 6: function_explanation (first function with a meaningful docstring)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and in_range(node):
            doc = ast.get_docstring(node)
            if doc and len(doc) > 50:
                examples.append({
                    "task": "function_explanation",
                    "question": f"Explain what the `{node.name}` function does.",
                    "answer": doc[:600],
                })
                break

    return examples


# ── Split Assignment ───────────────────────────────────────────────────────────

def repo_level_split(files: List[Dict], seed: int = 42) -> Dict[str, str]:
    """
    Assign each repo to a split (train / val / test) at the repo level.

    With N repos: 90% → train, 5% → val, 5% → test (at least 1 repo each).
    Returns dict mapping repo_name → split_name.
    """
    repos = sorted(set(f["repo"] for f in files))
    rng = random.Random(seed)
    rng.shuffle(repos)

    n = len(repos)
    n_val = max(1, round(n * 0.05))
    n_test = max(1, round(n * 0.05))
    # Ensure val + test don't eat all repos
    n_val = min(n_val, n // 5)
    n_test = min(n_test, n // 5)

    val_repos = set(repos[:n_val])
    test_repos = set(repos[n_val : n_val + n_test])
    train_repos = set(repos[n_val + n_test :])

    split_map: Dict[str, str] = {}
    for r in train_repos:
        split_map[r] = "train"
    for r in val_repos:
        split_map[r] = "val"
    for r in test_repos:
        split_map[r] = "test"
    return split_map


# ── Main Pipeline ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 2b data pipeline")
    parser.add_argument("--repos-dir", required=True,
                        help="Directory containing cloned repos (Scraped Repos/)")
    parser.add_argument("--output-dir", required=True,
                        help="Root output dir (images/ and manifests/ created here)")
    parser.add_argument("--features-dir",
                        default="./precomputed_features_tiled",
                        help="Dir to check for existing .pt files; images with .pt are skipped")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help="Lines per image chunk (default 500)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repos_dir = Path(args.repos_dir)
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    manifests_dir = output_dir / "manifests"
    features_dir = Path(args.features_dir)

    images_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Discover files ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 1: Discovering files")
    print("=" * 60)
    files = discover_files(repos_dir)

    # ── Step 2: Repo-level split assignment ───────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Repo-level split assignment")
    print("=" * 60)
    split_map = repo_level_split(files, seed=args.seed)
    for split_name in ("train", "val", "test"):
        repos_in_split = [r for r, s in split_map.items() if s == split_name]
        print(f"  {split_name}: {repos_in_split}")

    # ── Step 3: Render + label ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: Chunking, rendering, and labeling")
    print("=" * 60)

    split_examples: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}
    n_rendered = n_skipped = n_failed = n_examples = 0

    for file_idx, f in enumerate(files):
        if file_idx % 500 == 0:
            print(
                f"  [{file_idx}/{len(files)}] rendered={n_rendered} "
                f"skipped={n_skipped} examples={n_examples}"
            )

        source = f["source"]
        chunks = chunk_source(source, args.chunk_size)
        n_chunks = len(chunks)
        split = split_map[f["repo"]]

        for chunk_idx, (start_line, end_line, chunk_code) in enumerate(chunks):
            if (end_line - start_line + 1) < MIN_CHUNK_LINES:
                continue  # skip tiny tail chunks

            stem = make_stem(f["repo"], f["rel_path"], chunk_idx, n_chunks)
            img_path = images_dir / f"{stem}.png"
            pt_path = features_dir / f"{stem}.pt"

            # Skip if image already rendered (idempotent reruns)
            if img_path.exists():
                n_skipped += 1
                img_abs = str(img_path.resolve())
            else:
                try:
                    img_abs = convert_string_to_image(
                        code_str=chunk_code,
                        out_path=str(img_path),
                        style=STYLE,
                        font_size=FONT_SIZE,
                    )
                    n_rendered += 1
                except Exception as e:
                    print(
                        f"    WARN render failed {f['repo']}/{f['rel_path']} "
                        f"c{chunk_idx}: {e}"
                    )
                    n_failed += 1
                    continue

            # Generate AST labels for this chunk
            labels = extract_chunk_labels(source, start_line, end_line)
            if not labels:
                continue

            rel_stem = Path(f["rel_path"]).stem
            for label in labels:
                ex_id = f"{f['repo']}__{rel_stem}"
                if n_chunks > 1:
                    ex_id += f"_c{chunk_idx}"
                ex_id += f"__{label['task']}"

                split_examples[split].append({
                    "id": ex_id,
                    "image": img_abs,
                    "repo": f["repo"],
                    "source_file": f["rel_path"],
                    "chunk_idx": chunk_idx,
                    "start_line": start_line,
                    "end_line": end_line,
                    "task_type": label["task"],
                    "conversations": [
                        {
                            "role": "user",
                            "content": f"<img_start><image><img_end>\n{label['question']}",
                        },
                        {
                            "role": "assistant",
                            "content": label["answer"],
                        },
                    ],
                })
                n_examples += 1

    # ── Step 4: Write manifests ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 4: Writing manifests")
    print("=" * 60)
    for split_name, examples in split_examples.items():
        out_path = manifests_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as fh:
            for ex in examples:
                fh.write(json.dumps(ex) + "\n")
        print(f"  {split_name}: {len(examples)} examples → {out_path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Files processed : {len(files)}")
    print(f"  Images rendered : {n_rendered}")
    print(f"  Images skipped  : {n_skipped}  (already on disk)")
    print(f"  Render failures : {n_failed}")
    print(f"  Total examples  : {n_examples}")
    total = sum(len(v) for v in split_examples.values())
    for split_name, examples in split_examples.items():
        pct = 100 * len(examples) / max(total, 1)
        print(f"    {split_name}: {len(examples)} ({pct:.1f}%)")
    print(f"\nNext step:")
    print(f"  sbatch coder_vl/precompute_2b.sh")


if __name__ == "__main__":
    main()
