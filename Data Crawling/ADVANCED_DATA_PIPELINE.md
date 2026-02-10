# Advanced Data Pipeline (Phase 2b+)

**Status:** Future enhancement - use after Phase 2a validates the basic approach

This document describes a production-grade 9-step pipeline for generating 50K+ high-quality training examples. It's overkill for Phase 2a validation but will be valuable for Phase 2b instruction tuning and Phase 3 SWE-bench specialization.

---

## When to Use This

**Use the lean MVP (`simple_data_gen.py`) for Phase 2a** (~10K examples)

**Use this advanced pipeline when:**
- Phase 2a gates pass (loss converged, ROUGE-L > 0.25)
- Scaling to 50K+ examples for Phase 2b
- Need rigorous deduplication (MinHash LSH)
- Need controlled size distribution sampling
- Need repo-level train/val/test splits to prevent leakage

**Estimated build time:** 3-5 days  
**Estimated run time:** 3-5 hours  
**Disk usage:** ~20-35 GB

---

## Directory Structure

```
/data/coder_vl_data/
├── repos/                          # Shallow git clones (~5-8 GB)
│   ├── flask/
│   ├── django/
│   ├── fastapi/
│   └── ...
├── stdlib/                         # Symlink to /usr/lib/python3.10/
├── metadata/
│   ├── repo_list.json              # Repo names, stars, clone URLs
│   ├── all_files.jsonl             # Every discovered .py file + metadata
│   ├── filtered_files.jsonl        # After quality filters
│   ├── deduped_files.jsonl         # After deduplication
│   ├── selected_files.jsonl        # After size-distribution sampling
│   └── splits.json                 # {train: [repo_names], val: [...], test: [...]}
├── images/                         # Rendered PNGs (~15-25 GB)
│   ├── flask__app.py__0.png        # {repo}__{filepath}__{chunk_idx}.png
│   └── ...
├── manifests/
│   ├── train.jsonl                 # Final training manifest
│   ├── val.jsonl                   # Final validation manifest
│   └── test.jsonl                  # Final test manifest
├── checkpoints/                    # Pipeline state for resume
│   └── pipeline_state.json
└── logs/
    └── pipeline.log
```

---

## Pipeline Steps

All scripts go in `/home/ad.msoe.edu/gloriosog/DS OCR/DS Coder/data_pipeline/`

### Step 1: `collect_repos.py` — Clone Repositories

**Purpose:** Get a curated list of top Python repos and shallow-clone them.

**Approach:**
- Use `gh api` (GitHub CLI) to search repos by stars: `gh api "search/repositories?q=language:python+stars:>5000&sort=stars&per_page=100"`
- Fallback: hardcoded seed list of ~80 repos (Django, Flask, FastAPI, requests, httpx, click, scikit-learn, pandas, numpy, rich, typer, pydantic, black, ruff, poetry, pytorch, transformers, langchain, etc.)
- Shallow clone: `git clone --depth 1 --single-branch` (saves ~90% disk vs full clone)
- Also symlink Python stdlib: `ln -s /usr/lib/python3.10 /data/coder_vl_data/stdlib`

**Outputs:** Cloned repos in `/data/coder_vl_data/repos/`, `repo_list.json`

**Checkpointing:** Track which repos are already cloned; skip on rerun.

**Estimated disk:** ~5-8 GB for 80 shallow-cloned repos  
**Estimated time:** ~20-40 min (network-bound; run from login node)

---

### Step 2: `discover_files.py` — Find All Python Files + Metadata

**Purpose:** Walk every cloned repo and stdlib, record metadata for every `.py` file.

**For each file, record:**
```json
{
  "repo": "django",
  "rel_path": "django/db/models/query.py",
  "abs_path": "/data/coder_vl_data/repos/django/django/db/models/query.py",
  "line_count": 1462,
  "byte_size": 52340,
  "has_docstring": true,
  "ast_valid": true,
  "is_test": false,
  "is_vendored": false,
  "is_generated": false,
  "is_migration": false,
  "content_hash": "sha256:abc123..."
}
```

**Detection heuristics:**
- **Test files:** path contains `/test/`, `/tests/`, `test_`, `_test.py`, `/conftest.py`, `/fixtures/`
- **Vendored:** path contains `/vendor/`, `/vendored/`, `/third_party/`, `/_vendor/`
- **Generated:** file starts with `# Generated`, `# Auto-generated`, `# DO NOT EDIT`, or path contains `/generated/`, `/migrations/`
- **Migrations:** path contains `/migrations/` (Django/Alembic)
- **AST valid:** `ast.parse(source)` succeeds
- **Has docstring:** first node in `ast.parse` body is an `ast.Expr` with `ast.Constant` (str)
- **Line count:** `source.count('\n') + 1` (faster than subprocess)

**Outputs:** `all_files.jsonl` (one JSON object per line)

**Estimated count:** ~150K-300K raw .py files  
**Estimated time:** ~5-10 min

---

### Step 3: `filter_files.py` — Apply Quality Filters

**Purpose:** Remove test files, vendored code, generated code, invalid Python, and files outside the 50-2500 line range.

**Filters (in order):**
1. `ast_valid == true`
2. `is_test == false`
3. `is_vendored == false`
4. `is_generated == false`
5. `is_migration == false`
6. `50 <= line_count <= 2500`
7. `byte_size > 500` (skip trivial `__init__.py`)

**Outputs:** `filtered_files.jsonl`

**Estimated yield:** ~30K-60K files  
**Estimated time:** <1 min

---

### Step 4: `deduplicate.py` — MinHash Near-Duplicate Removal

**Purpose:** Remove near-duplicate files (forks, copied utility code, common boilerplate).

**Approach:**
- **Exact duplicates:** group by `content_hash`, keep one per group (prefer repo with more stars)
- **Near-duplicates:** MinHash LSH using `datasketch` library
  - Tokenize: split source into 5-grams of lines (not characters)
  - MinHash with 128 permutations
  - LSH threshold: 0.7 (files >70% similar are considered duplicates)
  - From each cluster, keep the file from the highest-starred repo

**Fallback:** If `datasketch` not available, exact-hash-only dedup (still removes most duplicates)

**Outputs:** `deduped_files.jsonl`

**Estimated yield:** ~20K-40K files  
**Estimated time:** ~5-15 min

---

### Step 5: `sample_distribution.py` — Enforce Size Distribution

**Purpose:** Sample from deduplicated files to hit the target distribution.

**Target (from PHASE2_PLAN.md Section 7.1):**

| Bucket | Lines | Target % | Target count (15K total) |
|--------|-------|----------|--------------------------|
| Small | 50-100 | 10% | 1,500 |
| Medium | 100-500 | 50% | 7,500 |
| Large | 500-1500 | 30% | 4,500 |
| Very large | 1500-2500 | 10% | 1,500 |

**Strategy:**
- If a bucket has more files than needed: random sample (prefer files with docstrings)
- If a bucket has fewer files than needed: take all, log a warning, redistribute remaining budget to other buckets
- The "very large" bucket will likely be underrepresented — this is fine, take what's available

**Outputs:** `selected_files.jsonl` (the final 10K-20K files)

**Estimated time:** <1 min

---

### Step 6: `split_repos.py` — Train/Val/Test Split by Repository

**Purpose:** Assign entire repos to train/val/test splits to prevent data leakage.

**Approach:**
- Sort repos by number of selected files (descending)
- Assign to splits maintaining 90/5/5 ratio
- Use deterministic seed (42) for reproducibility
- Ensure val and test each have at least 3 repos for diversity
- `stdlib` always goes to train (it's canonical, not a leakage risk)

**Outputs:** `splits.json` mapping repo names → split assignment. Updates `selected_files.jsonl` with a `split` field.

**Estimated time:** <1 min

---

### Step 7: `render_images.py` — Batch Image Rendering

**Purpose:** Convert every selected Python file to a syntax-highlighted PNG image.

**Parameters (from PHASE2_PLAN.md Section 7.4):**
- Style: `monokai` (dark theme, high contrast)
- Font size: 13pt
- No line numbers
- PNG format
- Max 500 lines per image; files >500 lines split into 400-line chunks with 50-line overlap

**Implementation:**
- Import `convert_code_to_image` from existing `code_to_image.py`
- For files ≤500 lines: render as single image
- For files >500 lines: split into chunks, render each chunk
- Parallelize with `multiprocessing.Pool` (8-16 workers)
- Naming: `{repo}__{relpath_with_dots}__{chunk}.png`
  - Example: `django__django.db.models.query__0.png`

**Outputs:** PNG images in `/data/coder_vl_data/images/`, updates metadata JSONL with `image_paths` field

**Estimated disk:** ~15-25 GB for 15K-25K images  
**Estimated time:** ~2-4 hours on compute node with 16 CPUs (SLURM job)

---

### Step 8: `generate_labels.py` — AST-Based Training Labels

**Purpose:** Generate 5 Q&A pairs per file using Python's `ast` module.

**Task templates:**

| # | User prompt | AST source | Skip if |
|---|-------------|------------|---------|
| 1 | "List all functions defined in this code." | `ast.FunctionDef` nodes | No functions found |
| 2 | "List all classes defined in this code." | `ast.ClassDef` nodes | No classes found |
| 3 | "What modules does this code import?" | `ast.Import` / `ast.ImportFrom` | No imports found |
| 4 | "What are the function signatures?" | `ast.FunctionDef` + `ast.arguments` | No functions found |
| 5 | "Describe what this code does." | Module/class/function docstrings | No docstrings found |

**For each file, output 3-5 examples** (skip tasks with no relevant AST nodes)

**Answer formatting:**
- Task 1: "This file defines the following functions:\n1. parse_args\n2. validate_input\n..."
- Task 2: "This file defines the following classes:\n1. RequestHandler\n2. Response\n..."
- Task 3: "This code imports the following modules:\n- os\n- sys\n- json\n..."
- Task 4: "Function signatures in this file:\n1. def parse_args(argv: list[str]) -> Namespace\n..."
- Task 5: First docstring found, or concatenation of top-level docstrings

**For multi-chunk files:** Each chunk gets its own Q&A pairs based on AST nodes in that chunk's line range.

**Outputs:** `raw_examples.jsonl`

**Estimated yield:** ~50K-75K examples from 15K files (avg ~4 tasks per file)  
**Estimated time:** ~10-20 min

---

### Step 9: `build_manifests.py` — Final Training Manifests

**Purpose:** Combine labels + image paths + split assignments into final training manifests.

**Output format (per line of JSONL):**
```json
{
  "id": "django__django.db.models.query__0__task1",
  "image": "/data/coder_vl_data/images/django__django.db.models.query__0.png",
  "repo": "django",
  "source_file": "django/db/models/query.py",
  "chunk_idx": 0,
  "line_range": [1, 400],
  "task_type": "function_listing",
  "conversations": [
    {"role": "user", "content": "<img_start><image><img_end>\nList all functions defined in this code."},
    {"role": "assistant", "content": "This file defines the following functions:\n1. get_queryset(...)\n..."}
  ]
}
```

**Outputs:**
- `manifests/train.jsonl` (~90% of examples)
- `manifests/val.jsonl` (~5%)
- `manifests/test.jsonl` (~5%)

**Validation checks:**
- Every referenced image file exists
- No repo appears in multiple splits
- Distribution stats printed (task type breakdown, size bucket breakdown per split)

**Estimated time:** <1 min

---

## Orchestration

### `run_pipeline.py` — Master Orchestrator

A single script that runs steps 1-9 sequentially with checkpointing:

```bash
python run_pipeline.py --data-dir /data/coder_vl_data --target-files 15000
```

- Reads `checkpoints/pipeline_state.json` to know which steps completed
- On failure: logs error, saves state, exits. Rerun resumes from last completed step.
- `--skip-clone` flag to skip step 1 if repos already cloned
- `--step N` flag to rerun a specific step

### `render_images.sh` — SLURM Job for Step 7

```bash
#!/bin/bash
#SBATCH --job-name=render-images
#SBATCH --partition=teaching
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-render-%j.out
#SBATCH --error=slurm-render-%j.err

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
"$PYTHON" "$HOME/DS OCR/DS Coder/data_pipeline/render_images.py" \
    --data-dir /data/coder_vl_data \
    --workers 16 \
    --style monokai \
    --font-size 13 \
    --no-line-numbers
```

No GPU needed — uses `teaching` partition for CPU cores only.

---

## Dependencies

Check if already available, install if missing:

```bash
PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
"$PYTHON" -m pip install datasketch   # MinHash LSH deduplication
```

Already available in env: `pygments`, `Pillow`, `numpy`, `pandas`, `ast` (stdlib).

---

## Disk Usage Summary

| Component | Estimated Size |
|-----------|----------------|
| Shallow clones (80 repos) | 5-8 GB |
| Metadata JSONL files | <100 MB |
| Rendered images (15K-25K PNGs) | 15-25 GB |
| **Total** | **~20-35 GB** |

---

## Runtime Summary

| Step | Script | Where to Run | Est. Time |
|------|--------|--------------|-----------|
| 1 | `collect_repos.py` | Login node | 20-40 min |
| 2 | `discover_files.py` | Login node or compute | 5-10 min |
| 3 | `filter_files.py` | Login node | <1 min |
| 4 | `deduplicate.py` | Login node or compute | 5-15 min |
| 5 | `sample_distribution.py` | Login node | <1 min |
| 6 | `split_repos.py` | Login node | <1 min |
| 7 | `render_images.py` | SLURM (teaching, CPU) | 2-4 hours |
| 8 | `generate_labels.py` | Login node or compute | 10-20 min |
| 9 | `build_manifests.py` | Login node | <1 min |
| **Total** | | | **~3-5 hours** |

Only step 7 (image rendering) needs a SLURM job. Everything else runs fine on the login node.

---

## Files to Create

```
DS Coder/data_pipeline/
├── run_pipeline.py          # Master orchestrator
├── collect_repos.py         # Step 1: Clone repos
├── discover_files.py        # Step 2: Find and catalog .py files
├── filter_files.py          # Step 3: Quality filters
├── deduplicate.py           # Step 4: MinHash dedup
├── sample_distribution.py   # Step 5: Size distribution sampling
├── split_repos.py           # Step 6: Train/val/test split
├── render_images.py         # Step 7: Batch image rendering
├── generate_labels.py       # Step 8: AST label extraction
├── build_manifests.py       # Step 9: Final manifest assembly
├── render_images.sh         # SLURM job script for step 7
└── repo_seed_list.json      # Hardcoded fallback repo list
```

---

## Fallback Strategies

| Risk | Fallback |
|------|----------|
| `gh` CLI not available or not authenticated | Hardcoded seed list of 80 repos with clone URLs |
| `datasketch` can't be installed | Exact-hash-only dedup (SHA-256 of file content) |
| "Very large" bucket underrepresented | Accept fewer; redistribute budget to "large" bucket |
| Rendering takes too long | Reduce to 10K files; render in batches across multiple SLURM jobs |
| `/data/` quota exceeded | Use `/scratch/` for images (faster but temporary) |
| Repo clone fails (private, deleted) | Skip and log; seed list has 80 repos, only need ~60 |
| Compute nodes lack internet | Clone repos on login node first (step 1 on login node) |

---

## Verification

After the pipeline completes:

1. **Check manifest sizes:**
   ```bash
   wc -l /data/coder_vl_data/manifests/*.jsonl
   ```
   Expect ~45K-75K train, ~2.5K-4K val, ~2.5K-4K test

2. **Spot-check 10 random images:**
   Open PNGs, verify they're readable code with dark theme, no line numbers

3. **Spot-check 10 random labels:**
   Verify AST-extracted answers are correct against the source file

4. **Verify no repo leakage:**
   ```python
   import json
   d = json.load(open('splits.json'))
   assert not (set(d['train']) & set(d['val']) | set(d['train']) & set(d['test']))
   ```

5. **Print distribution stats:**
   From `build_manifests.py` output, compare to targets

---

*This pipeline is designed for Phase 2b+ when you need production-grade data quality and scale. For Phase 2a validation, use the lean MVP approach instead.*

