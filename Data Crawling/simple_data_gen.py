#!/usr/bin/env python3
"""
simple_data_gen.py — Lean MVP data generation for Phase 2a

Generates 10K training examples in one pass:
1. Walk repos, find .py files (50-2500 lines)
2. Filter out test/vendor/generated code (simple heuristics)
3. Render as images (reuse code_to_image.py)
4. Generate AST-based Q&A labels
5. Random 90/5/5 train/val/test split
6. Write manifests

Usage:
    python simple_data_gen.py \\
        --repos-dir /scratch/$USER/coder_vl_data/repos \\
        --output-dir /scratch/$USER/coder_vl_data \\
        --target 10000 \\
        --style monokai \\
        --font-size 13
"""

import argparse
import ast
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Import image rendering from existing script
sys.path.insert(0, str(Path(__file__).parent))
from code_to_image import convert_code_to_image


def should_skip_file(rel_path: str) -> Tuple[bool, str]:
    """Simple heuristics to filter out test/vendor/generated files."""
    path_lower = rel_path.lower()
    
    # Test files
    if any(x in path_lower for x in ['/test/', '/tests/', 'test_', '_test.py', 'conftest.py']):
        return True, "test"
    
    # Vendored code
    if any(x in path_lower for x in ['/vendor/', '/vendored/', '/third_party/', '/_vendor/']):
        return True, "vendored"
    
    # Generated/migration files
    if any(x in path_lower for x in ['/migrations/', '/generated/', '__pycache__']):
        return True, "generated"
    
    # Init files (usually trivial imports)
    if rel_path.endswith('__init__.py'):
        return True, "init"
    
    return False, ""


def extract_ast_labels(source_code: str, file_path: str) -> List[Dict]:
    """Generate 3-5 Q&A pairs from AST analysis."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []
    
    examples = []
    
    # Task 1: List functions
    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    if functions:
        func_list = "\n".join(f"{i+1}. {name}" for i, name in enumerate(functions[:20]))  # Cap at 20
        examples.append({
            "task": "function_listing",
            "question": "List all functions defined in this code.",
            "answer": f"This file defines the following functions:\n{func_list}"
        })
    
    # Task 2: List classes
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    if classes:
        class_list = "\n".join(f"{i+1}. {name}" for i, name in enumerate(classes[:20]))
        examples.append({
            "task": "class_listing",
            "question": "List all classes defined in this code.",
            "answer": f"This file defines the following classes:\n{class_list}"
        })
    
    # Task 3: List imports
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    
    if imports:
        import_list = "\n".join(f"- {name}" for name in sorted(set(imports))[:30])
        examples.append({
            "task": "import_listing",
            "question": "What modules does this code import?",
            "answer": f"This code imports the following modules:\n{import_list}"
        })
    
    # Task 4: Function signatures
    if functions:
        signatures = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Simple signature extraction (no type annotations for now)
                args = [arg.arg for arg in node.args.args]
                sig = f"def {node.name}({', '.join(args)})"
                signatures.append(sig)
                if len(signatures) >= 10:  # Cap at 10
                    break
        
        if signatures:
            sig_list = "\n".join(f"{i+1}. {sig}" for i, sig in enumerate(signatures))
            examples.append({
                "task": "function_signatures",
                "question": "What are the function signatures in this code?",
                "answer": f"Function signatures in this file:\n{sig_list}"
            })
    
    # Task 5: Docstrings
    docstrings = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            docstring = ast.get_docstring(node)
            if docstring:
                docstrings.append(docstring)
                break  # Just take the first meaningful docstring
    
    if docstrings:
        examples.append({
            "task": "description",
            "question": "Describe what this code does.",
            "answer": docstrings[0][:500]  # Cap at 500 chars
        })
    
    return examples


def discover_files(repos_dir: Path) -> List[Dict]:
    """Walk repos and collect metadata for all .py files."""
    print(f"🔍 Discovering Python files in {repos_dir}...")
    files = []
    
    for repo_path in repos_dir.iterdir():
        if not repo_path.is_dir() or repo_path.name.startswith('.'):
            continue
        
        repo_name = repo_path.name
        print(f"  Scanning {repo_name}...")
        
        for py_file in repo_path.rglob("*.py"):
            rel_path = py_file.relative_to(repo_path)
            
            # Skip filter
            skip, reason = should_skip_file(str(rel_path))
            if skip:
                continue
            
            try:
                source = py_file.read_text(encoding='utf-8', errors='ignore')
                line_count = source.count('\n') + 1
                
                # Size filter: 50-2500 lines
                if not (50 <= line_count <= 2500):
                    continue
                
                # Skip tiny files
                if len(source) < 500:
                    continue
                
                # Check AST validity
                try:
                    ast.parse(source)
                    ast_valid = True
                except SyntaxError:
                    ast_valid = False
                
                if not ast_valid:
                    continue
                
                # Compute hash for exact dedup
                content_hash = hashlib.sha256(source.encode()).hexdigest()[:16]
                
                files.append({
                    "repo": repo_name,
                    "rel_path": str(rel_path),
                    "abs_path": str(py_file),
                    "line_count": line_count,
                    "byte_size": len(source),
                    "content_hash": content_hash,
                })
            
            except Exception as e:
                # Skip files we can't read
                continue
    
    print(f"✅ Found {len(files)} valid Python files")
    return files


def deduplicate(files: List[Dict]) -> List[Dict]:
    """Remove exact duplicates by content hash."""
    print(f"🔄 Deduplicating files...")
    seen_hashes = set()
    unique = []
    
    for f in files:
        if f["content_hash"] not in seen_hashes:
            seen_hashes.add(f["content_hash"])
            unique.append(f)
    
    print(f"✅ Kept {len(unique)} unique files (removed {len(files) - len(unique)} duplicates)")
    return unique


def sample_by_size(files: List[Dict], target: int) -> List[Dict]:
    """Sample files to hit target count, preferring medium-sized files."""
    print(f"📊 Sampling {target} files by size distribution...")
    
    # Bucket by size
    small = [f for f in files if 50 <= f["line_count"] < 100]
    medium = [f for f in files if 100 <= f["line_count"] < 500]
    large = [f for f in files if 500 <= f["line_count"] < 1500]
    very_large = [f for f in files if 1500 <= f["line_count"] <= 2500]
    
    print(f"  Small (50-100): {len(small)} available")
    print(f"  Medium (100-500): {len(medium)} available")
    print(f"  Large (500-1500): {len(large)} available")
    print(f"  Very large (1500-2500): {len(very_large)} available")
    
    # Target distribution (rough)
    target_small = min(len(small), int(target * 0.10))
    target_medium = min(len(medium), int(target * 0.50))
    target_large = min(len(large), int(target * 0.30))
    target_very_large = min(len(very_large), int(target * 0.10))
    
    selected = []
    selected.extend(random.sample(small, target_small) if small else [])
    selected.extend(random.sample(medium, target_medium) if medium else [])
    selected.extend(random.sample(large, target_large) if large else [])
    selected.extend(random.sample(very_large, target_very_large) if very_large else [])
    
    # If we're short, fill from medium bucket
    if len(selected) < target and medium:
        remaining = target - len(selected)
        extra = [f for f in medium if f not in selected]
        selected.extend(random.sample(extra, min(remaining, len(extra))))
    
    print(f"✅ Selected {len(selected)} files")
    return selected


def render_images(files: List[Dict], output_dir: Path, style: str, font_size: int) -> List[Dict]:
    """Render each file as a PNG image."""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎨 Rendering {len(files)} images (style={style}, font_size={font_size})...")
    
    for i, f in enumerate(files):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(files)}")
        
        # Image filename: {repo}__{filename}.png
        safe_name = f["rel_path"].replace("/", "__").replace("\\", "__")
        image_name = f"{f['repo']}__{safe_name}.png"
        
        try:
            image_path = convert_code_to_image(
                code_file_path=f["abs_path"],
                output_dir=str(images_dir),
                style=style,
                font_size=font_size,
                line_numbers=False,  # No line numbers per PHASE2_PLAN
                image_pad=10
            )
            f["image_path"] = str(image_path)
        except Exception as e:
            print(f"    ⚠️  Failed to render {f['abs_path']}: {e}")
            f["image_path"] = None
    
    # Filter out files that failed to render
    files = [f for f in files if f.get("image_path")]
    print(f"✅ Rendered {len(files)} images successfully")
    return files


def generate_labels(files: List[Dict]) -> List[Dict]:
    """Generate AST-based Q&A labels for each file."""
    print(f"🏷️  Generating AST labels...")
    all_examples = []
    
    for i, f in enumerate(files):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(files)}")
        
        try:
            source = Path(f["abs_path"]).read_text(encoding='utf-8', errors='ignore')
            labels = extract_ast_labels(source, f["abs_path"])
            
            for j, label in enumerate(labels):
                example_id = f"{f['repo']}__{Path(f['rel_path']).stem}__{label['task']}"
                all_examples.append({
                    "id": example_id,
                    "image": f["image_path"],
                    "repo": f["repo"],
                    "source_file": f["rel_path"],
                    "line_count": f["line_count"],
                    "task_type": label["task"],
                    "conversations": [
                        {"role": "user", "content": f"<img_start><image><img_end>\n{label['question']}"},
                        {"role": "assistant", "content": label["answer"]}
                    ]
                })
        except Exception as e:
            continue
    
    print(f"✅ Generated {len(all_examples)} training examples")
    return all_examples


def split_examples(examples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Random 90/5/5 train/val/test split."""
    print(f"✂️  Splitting examples into train/val/test...")
    random.shuffle(examples)
    
    total = len(examples)
    train_size = int(total * 0.90)
    val_size = int(total * 0.05)
    
    train = examples[:train_size]
    val = examples[train_size:train_size + val_size]
    test = examples[train_size + val_size:]
    
    print(f"✅ Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test


def write_manifests(train, val, test, output_dir: Path):
    """Write final JSONL manifests."""
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Writing manifests to {manifests_dir}...")
    
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        manifest_path = manifests_dir / f"{split_name}.jsonl"
        with open(manifest_path, "w") as f:
            for example in split_data:
                f.write(json.dumps(example) + "\n")
        print(f"  Wrote {manifest_path} ({len(split_data)} examples)")
    
    print("✅ Manifests written successfully!")


def main():
    parser = argparse.ArgumentParser(description="Lean MVP data generation for Phase 2a")
    parser.add_argument("--repos-dir", type=str, required=True, help="Directory containing cloned repos")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for images and manifests")
    parser.add_argument("--target", type=int, default=10000, help="Target number of source files to process")
    parser.add_argument("--style", type=str, default="monokai", help="Pygments style for syntax highlighting")
    parser.add_argument("--font-size", type=int, default=13, help="Font size in points")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    repos_dir = Path(args.repos_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not repos_dir.exists():
        print(f"❌ Repos directory not found: {repos_dir}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"DeepSeek-Coder-VL Lean Data Generation Pipeline")
    print(f"{'='*60}\n")
    
    # Step 1: Discover files
    files = discover_files(repos_dir)
    
    # Step 2: Deduplicate
    files = deduplicate(files)
    
    # Step 3: Sample by size distribution
    # Target fewer source files (since each produces 3-5 examples)
    target_files = args.target // 4  # Each file → ~4 examples
    files = sample_by_size(files, target_files)
    
    # Step 4: Render images
    files = render_images(files, output_dir, args.style, args.font_size)
    
    # Step 5: Generate AST labels
    examples = generate_labels(files)
    
    # Step 6: Split
    train, val, test = split_examples(examples)
    
    # Step 7: Write manifests
    write_manifests(train, val, test, output_dir)
    
    print(f"\n{'='*60}")
    print(f"✅ Pipeline complete!")
    print(f"{'='*60}")
    print(f"Total examples: {len(examples)}")
    print(f"Output directory: {output_dir}")
    print(f"\nNext steps:")
    print(f"1. Verify images: ls {output_dir / 'images'} | head")
    print(f"2. Check manifests: head {output_dir / 'manifests' / 'train.jsonl'}")
    print(f"3. Start Phase 2a training!")


if __name__ == "__main__":
    main()

