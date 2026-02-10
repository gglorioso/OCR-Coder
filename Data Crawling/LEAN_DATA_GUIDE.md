# Lean Data Generation Guide (Phase 2a MVP)

**Goal:** Generate ~10K training examples in one day to validate Phase 2a adapter training.

**Time to first training run:** ~6 hours (30 min setup + 2-4 hours data gen + validation)

---

## Quick Start

### 1. Clone Repos (30 min, run on login node)

```bash
# Create repos directory (using /scratch/ - no approval needed)
mkdir -p /scratch/$USER/coder_vl_data/repos
cd /scratch/$USER/coder_vl_data/repos

# Clone 15-20 top Python repos (shallow clones)
git clone --depth 1 https://github.com/django/django
git clone --depth 1 https://github.com/pallets/flask
git clone --depth 1 https://github.com/tiangolo/fastapi
git clone --depth 1 https://github.com/psf/requests
git clone --depth 1 https://github.com/encode/httpx
git clone --depth 1 https://github.com/pallets/click
git clone --depth 1 https://github.com/pydantic/pydantic
git clone --depth 1 https://github.com/python/cpython
git clone --depth 1 https://github.com/numpy/numpy
git clone --depth 1 https://github.com/pandas-dev/pandas
git clone --depth 1 https://github.com/scikit-learn/scikit-learn
git clone --depth 1 https://github.com/pytorch/pytorch
git clone --depth 1 https://github.com/huggingface/transformers
git clone --depth 1 https://github.com/psf/black
git clone --depth 1 https://github.com/python-poetry/poetry

# Check disk usage (should be ~2-4 GB)
du -sh .
```

**Pro tip:** Do this on the login node while you're working on other things. Network-bound, not compute-intensive.

---

### 2. Run Data Generation (2-4 hours on compute node)

```bash
cd /home/ad.msoe.edu/gloriosog/CoderOCR/OCR-Coder/Data\ Crawling

# Submit SLURM job
sbatch simple_data_gen.sh

# Monitor progress
tail -f slurm-data-gen-<jobid>.out
```

The script will:
1. Walk all repos and find `.py` files (50-2500 lines)
2. Filter out test/vendor/generated files
3. Deduplicate by content hash
4. Sample ~2,500 files (4 examples each = ~10K total)
5. Render each as a PNG image (monokai style, no line numbers)
6. Generate 3-5 AST-based Q&A labels per file
7. Split 90/5/5 train/val/test
8. Write manifests to `/scratch/$USER/coder_vl_data/manifests/`

---

### 3. Verify Output

```bash
# Check structure
ls -lh /scratch/$USER/coder_vl_data/

# Should see:
#   repos/        (~2-4 GB)
#   images/       (~2-3 GB, 2,000-3,000 PNGs)
#   manifests/    (train.jsonl, val.jsonl, test.jsonl)

# Count examples
wc -l /scratch/$USER/coder_vl_data/manifests/*.jsonl

# Expected output:
#   ~9,000 train.jsonl
#   ~500   val.jsonl
#   ~500   test.jsonl

# Spot-check a manifest entry
head -n 1 /scratch/$USER/coder_vl_data/manifests/train.jsonl | python -m json.tool

# Open a random image to verify rendering
# (Copy one to local machine or view on Rosie with image viewer)
ls /scratch/$USER/coder_vl_data/images/ | head -n 5
```

---

## What You Get

**Manifest format** (each line in `train.jsonl`, `val.jsonl`, `test.jsonl`):

```json
{
  "id": "django__query__function_listing",
  "image": "/scratch/$USER/coder_vl_data/images/django__django.db.models.query.py.png",
  "repo": "django",
  "source_file": "django/db/models/query.py",
  "line_count": 1462,
  "task_type": "function_listing",
  "conversations": [
    {
      "role": "user",
      "content": "<img_start><image><img_end>\nList all functions defined in this code."
    },
    {
      "role": "assistant",
      "content": "This file defines the following functions:\n1. get_queryset\n2. prefetch_related\n..."
    }
  ]
}
```

**Task types:**
- `function_listing` — List all function names
- `class_listing` — List all class names
- `import_listing` — List all imported modules
- `function_signatures` — Extract function signatures
- `description` — First docstring from the file

---

## Troubleshooting

### "Repos directory not found"

Clone repos first (see step 1 above). The script checks for `/scratch/$USER/coder_vl_data/repos/` and exits if missing.

### "Only got 3,000 examples, not 10,000"

This means:
- Not enough repos cloned (need 15-20 for 10K examples)
- Repos are small (clone larger ones like pytorch, transformers, cpython)
- Files filtered out as test/vendor code

**Fix:** Clone more repos or reduce `--target` to 5000.

### Image rendering fails

Check if Pygments/Pillow are installed:

```bash
$HOME/DS\ OCR/envs/deepseek-ocr/bin/python -c "import pygments; import PIL; print('OK')"
```

If missing, install:

```bash
$HOME/DS\ OCR/envs/deepseek-ocr/bin/pip install pygments Pillow
```

### "Permission denied" on `/scratch/`

This is unlikely, but if it happens, use your home directory:

```bash
# In simple_data_gen.sh, change:
REPOS_DIR="$HOME/coder_vl_data/repos"
OUTPUT_DIR="$HOME/coder_vl_data"
```

Just be mindful of your home quota (~50-100 GB). The MVP needs ~5-7 GB.

---

## Migrating to `/data/` (When Access is Approved)

Once you get `/data/` access, migrate your data:

```bash
# Simple move
mv /scratch/$USER/coder_vl_data /data/coder_vl_data

# Update simple_data_gen.sh:
# Change REPOS_DIR and OUTPUT_DIR to /data/coder_vl_data paths

# Update any training scripts to point to new paths
```

**Why migrate?**
- `/scratch/` files may be auto-deleted after 30-90 days
- `/data/` is permanent shared storage
- Better for long-term project storage

**For now:** `/scratch/` is perfect - large quota, no approval needed, fast!

---

## Next Steps After Data Generation

1. **Backup manifests to home:** `cp -r /scratch/$USER/coder_vl_data/manifests $HOME/backup/`
2. **Spot-check quality:** Open 10 random images, verify they're readable code with syntax highlighting
3. **Verify labels:** Check that AST-extracted answers match the source files
4. **Start Phase 2a training:** Use the manifests with your adapter training script

---

## When to Upgrade to Advanced Pipeline

After Phase 2a gates pass, use `ADVANCED_DATA_PIPELINE.md` for Phase 2b:
- 50K+ examples (not 10K)
- MinHash near-duplicate detection (not just exact hash)
- Controlled size distribution sampling
- Repo-level train/val/test splits (not random)
- Multi-chunk support for 500+ line files

The lean pipeline is **good enough for validation**. Don't overengineer until you know the approach works!

---

## Estimated Costs

| Resource | Amount | Notes |
|----------|--------|-------|
| Disk (repos) | ~2-4 GB | Shallow clones, 15-20 repos |
| Disk (images) | ~2-3 GB | 2,000-3,000 PNGs @ ~1 MB each |
| Disk (manifests) | <10 MB | JSONL text files |
| **Total disk** | **~5-7 GB** | All on `/scratch/$USER/` |
| Compute time | ~2-4 hours | 16 CPUs, `teaching` partition |
| Walltime | <6 hours | Includes buffer for slow I/O |

**No GPU needed** for data generation — only CPU for image rendering.

