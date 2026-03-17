"""
token_count.py
--------------
DeepSeek-Coder tokenizer baseline test.

Reads source code files referenced by the Phase_1_9/a ground_truth manifest,
tokenizes each with DeepSeek-Coder-V2-Lite-Instruct, and reports summary
statistics plus a simple ASCII histogram. Results are saved to
MVV/Token_Test/results/token_stats.json.

The manifest schema is:
  {"stem": "...", "source_file": "black/action/main.py", "anchor_line": 0}

source_file is a relative path; prepend "Scraped Repos/" to get the full path.
anchor_line is 0-indexed; always read exactly 40 lines from there.
"""

import json
import os
import random
import statistics
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]          # OCR-Coder/
SCRAPED_REPOS = REPO_ROOT / "Scraped Repos"
MANIFEST_PATH = REPO_ROOT / "MVV" / "Phase_1_9" / "a" / "data" / "ground_truth.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_PATH = RESULTS_DIR / "token_stats.json"

SAMPLE_SIZE = 500
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Load tokenizer
# ---------------------------------------------------------------------------
print("Loading tokenizer …")
tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    trust_remote_code=True,
)
print("Tokenizer loaded.\n")

# ---------------------------------------------------------------------------
# Load manifest and draw a random sample
# ---------------------------------------------------------------------------
print(f"Reading manifest: {MANIFEST_PATH}")
with open(MANIFEST_PATH, "r") as fh:
    all_entries = [json.loads(line) for line in fh if line.strip()]

print(f"Total entries in manifest: {len(all_entries)}")

random.seed(RANDOM_SEED)
sample = random.sample(all_entries, min(SAMPLE_SIZE, len(all_entries)))
print(f"Sampled {len(sample)} entries (seed={RANDOM_SEED})\n")

# ---------------------------------------------------------------------------
# Tokenize each source file
# ---------------------------------------------------------------------------
token_counts = []
skipped = 0

for entry in sample:
    source_file = entry["source_file"]

    # source_file is a repo-relative path like "black/action/main.py";
    # prepend "Scraped Repos/" to reach the file on disk.
    full_path = SCRAPED_REPOS / source_file

    if not full_path.exists():
        skipped += 1
        continue

    try:
        all_lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    except Exception as exc:
        print(f"  [WARN] Could not read {full_path}: {exc}")
        skipped += 1
        continue

    # Slice the 40-line snippet that was rendered as the image.
    # anchor_line is 0-indexed; always take exactly 40 lines from there.
    # Apply the same truncation as the image renderer: expandtabs(4)[:80]
    anchor = entry.get("anchor_line", 0)
    snippet_lines = all_lines[anchor : anchor + 40]
    code = "".join(line.expandtabs(4)[:80] for line in snippet_lines)

    ids = tokenizer.encode(code, add_special_tokens=False)
    token_counts.append(len(ids))

print(f"Tokenized: {len(token_counts)}  |  Skipped (file not found / unreadable): {skipped}\n")

if not token_counts:
    print("No files could be tokenized. Exiting.")
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
arr = np.array(token_counts)

mean_tokens   = float(np.mean(arr))
median_tokens = float(np.median(arr))
min_tokens    = int(np.min(arr))
max_tokens    = int(np.max(arr))
p95_tokens    = float(np.percentile(arr, 95))
std_tokens    = float(np.std(arr))

print("=== Token Count Statistics ===")
print(f"  Count  : {len(arr)}")
print(f"  Min    : {min_tokens:,}")
print(f"  Max    : {max_tokens:,}")
print(f"  Mean   : {mean_tokens:,.1f}")
print(f"  Median : {median_tokens:,.1f}")
print(f"  Std    : {std_tokens:,.1f}")
print(f"  P95    : {p95_tokens:,.1f}")
print()

# ---------------------------------------------------------------------------
# ASCII histogram
# ---------------------------------------------------------------------------
BUCKETS = [
    (0,     100,   "0–100"),
    (100,   200,   "100–200"),
    (200,   500,   "200–500"),
    (500,   1000,  "500–1k"),
    (1000,  2000,  "1k–2k"),
    (2000,  5000,  "2k–5k"),
    (5000,  10000, "5k–10k"),
    (10000, None,  "10k+"),
]

BAR_MAX_WIDTH = 40

print("=== Token Count Distribution ===")
counts_per_bucket = []
for lo, hi, label in BUCKETS:
    if hi is None:
        n = int(np.sum(arr >= lo))
    else:
        n = int(np.sum((arr >= lo) & (arr < hi)))
    counts_per_bucket.append(n)

max_count = max(counts_per_bucket) if counts_per_bucket else 1

for (lo, hi, label), n in zip(BUCKETS, counts_per_bucket):
    bar_len = int(round(n / max_count * BAR_MAX_WIDTH)) if max_count > 0 else 0
    bar = "#" * bar_len
    print(f"  {label:>10}  | {bar:<{BAR_MAX_WIDTH}} {n:>5}")

print()

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

results = {
    "manifest": str(MANIFEST_PATH),
    "sample_size": len(arr),
    "random_seed": RANDOM_SEED,
    "skipped": skipped,
    "stats": {
        "min": min_tokens,
        "max": max_tokens,
        "mean": round(mean_tokens, 2),
        "median": round(median_tokens, 2),
        "std": round(std_tokens, 2),
        "p95": round(p95_tokens, 2),
    },
    "histogram": [
        {"bucket": label, "count": n}
        for (_, _, label), n in zip(BUCKETS, counts_per_bucket)
    ],
}

with open(RESULTS_PATH, "w") as fh:
    json.dump(results, fh, indent=2)

print(f"Results saved to: {RESULTS_PATH}")
