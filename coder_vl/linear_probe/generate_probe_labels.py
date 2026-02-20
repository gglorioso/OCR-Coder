"""
Generate probe labels for linear probe test.

Parses ground truth answers from Phase 2a manifests to extract code properties
(has_class, num_functions, etc.) as labels for each unique image.

NOTE: Uses ONLY the train manifest (which has ~4.7 task entries per image)
and creates its own 80/20 split. The original val/test manifests only have
~1.2 entries per image, making label extraction unreliable.

Usage:
    python coder_vl/linear_probe/generate_probe_labels.py
"""

import json
import random
import re
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_DIR = PROJECT_ROOT / "Data Crawling" / "output" / "manifests"
OUTPUT_DIR = Path(__file__).resolve().parent / "probe_data"

SPLIT_SEED = 42
VAL_FRACTION = 0.20


def count_numbered_items(text):
    """Count '1. item' style entries in an answer."""
    return len(re.findall(r"^\d+\.", text, re.MULTILINE))


def count_bullet_items(text):
    """Count '- item' style entries in an answer."""
    return len(re.findall(r"^- ", text, re.MULTILINE))


def parse_manifest(manifest_path):
    """Parse manifest and group entries by image path."""
    image_data = defaultdict(lambda: {"tasks": {}, "metadata": {}})

    with open(manifest_path) as f:
        for line in f:
            entry = json.loads(line)
            image = entry["image"]
            task_type = entry["task_type"]
            answer = entry["conversations"][1]["content"]

            image_data[image]["tasks"][task_type] = answer
            image_data[image]["metadata"]["line_count"] = entry.get("line_count", 0)
            image_data[image]["metadata"]["source_file"] = entry.get("source_file", "")
            image_data[image]["metadata"]["repo"] = entry.get("repo", "")

    return image_data


def extract_labels(image_data):
    """Extract probe labels from task answers for a single image."""
    tasks = image_data["tasks"]
    meta = image_data["metadata"]

    # Count classes from class_listing answer
    if "class_listing" in tasks:
        num_classes = count_numbered_items(tasks["class_listing"])
    else:
        num_classes = 0

    # Count functions from function_listing answer
    if "function_listing" in tasks:
        num_functions = count_numbered_items(tasks["function_listing"])
    else:
        num_functions = 0

    # Count imports from import_listing answer
    if "import_listing" in tasks:
        num_imports = count_bullet_items(tasks["import_listing"])
    else:
        num_imports = 0

    line_count = meta.get("line_count", 0)

    return {
        # Binary labels
        "has_class": num_classes > 0,
        "has_function": num_functions > 0,
        "has_imports": num_imports > 0,
        "has_many_functions": num_functions > 5,
        "is_large_file": line_count > 500,
        # Count labels
        "num_classes": num_classes,
        "num_functions": num_functions,
        "num_imports": num_imports,
        # Bucketed labels
        "file_size_bucket": (
            0 if line_count < 200 else
            1 if line_count < 800 else
            2
        ),
        "function_count_bucket": (
            0 if num_functions == 0 else
            1 if num_functions <= 5 else
            2 if num_functions <= 15 else
            3
        ),
        # Metadata
        "line_count": line_count,
        "source_file": meta.get("source_file", ""),
    }


def print_stats(labels, split_name):
    """Print label distribution statistics."""
    print(f"\n{'='*60}")
    print(f"  {split_name}: {len(labels)} images")
    print(f"{'='*60}")

    for key in ["has_class", "has_function", "has_imports", "has_many_functions", "is_large_file"]:
        pos = sum(1 for l in labels if l[key])
        neg = len(labels) - pos
        pct = pos / len(labels) * 100
        print(f"  {key:<25} {pos:>5} pos / {neg:>5} neg  ({pct:5.1f}% positive)")

    for key in ["file_size_bucket", "function_count_bucket"]:
        counts = defaultdict(int)
        for l in labels:
            counts[l[key]] += 1
        dist = dict(sorted(counts.items()))
        print(f"  {key:<25} {dist}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Use ONLY the train manifest (good task coverage: ~4.7 entries/image)
    manifest_path = MANIFEST_DIR / "train.jsonl"
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found")
        return

    image_data = parse_manifest(manifest_path)

    # Build labels for all images
    all_labels = []
    for image_path, data in sorted(image_data.items()):
        label = extract_labels(data)
        label["image"] = image_path
        all_labels.append(label)

    print(f"Total images from train manifest: {len(all_labels)}")
    print(f"(Original val/test manifests skipped: only ~1.2 task entries/image)")

    # Shuffle and split 80/20
    rng = random.Random(SPLIT_SEED)
    shuffled = list(all_labels)
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * (1 - VAL_FRACTION))
    splits = {
        "train": shuffled[:split_idx],
        "val": shuffled[split_idx:],
    }

    for split_name, labels in splits.items():
        output_path = OUTPUT_DIR / f"probe_labels_{split_name}.jsonl"
        with open(output_path, "w") as f:
            for label in labels:
                f.write(json.dumps(label) + "\n")

        print_stats(labels, split_name)

    print(f"\nLabels saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
