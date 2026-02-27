# Sniper Method — Implementation Plan

**Last Updated:** 2026-02-26

---

## 1. What Is the Sniper Method?

The Sniper Method is a two-stage approach to SWE-bench bug localization using compressed code images:

- **Text-only baseline:** ~5-8 files fit in a 128K context window
- **Sniper Method:** 60-100+ files fit by encoding each file as a compressed image

Instead of reading all code as text, the model sees many files as visual tokens, narrows down
candidates, then zooms into the relevant files at higher resolution.

---

## 2. Resolution Problem — Why the Current Model May Not Be Learning

**Hypothesis (confirmed by data):**

The precomputed SigLIP features use a tiled approach capped at 1,120 tokens. For large files:

| File size | Scale at 1120 tokens | px/char | Readable? |
|---|---|---|---|
| ~100 lines (711×785px) | 1.00x | 8.9 | Yes |
| ~400 lines (711×2975px) | 0.67x | 5.6 | Marginal |
| ~600 lines (711×4415px) | 0.55x | 4.5 | Poor |
| ~1000 lines (711×7520px) | 0.40x | 3.6 | No |

**The misalignment:** Training descriptions were generated from raw code text (fully readable),
but the visual features encode blurry compressed images. For large files, the visual tokens cannot
carry the semantic content the text descriptions reference — so contrastive alignment fails.

**Why contrastive_v4 worked (val_cos=0.840):** It was trained on the full dataset, where small-
and medium-sized files (readable at compression) dominated the contrastive signal. val_cos reflects
the average, so a good score on small files masked the failure on large ones.

---

## 3. Two Training Tracks

### Track A — Current Model (SigLIP + DeepSeek-Coder-V2)

**Status:** Phase 3 contrastive training, job 227177 running (phase2b_v6).

**Fix applied:** Init from contrastive_v4 adapter (val_cos=0.840) instead of phase2a_v6.
This gives the adapter a starting point where visual embeddings are already aligned with text,
so contrastive gradients are non-trivial from step 1.

**Known limitation:** Resolution ceiling for large files. This architecture cannot be easily
scaled to higher token counts because the SigLIP features are precomputed at a fixed resolution.

**Success criteria:**
- val_pos_cos (description) > 0.5 at step 200 → contrastive is learning
- Retrieval Recall@5 (description) > 15%
- val_loss (generation) < 1.40

**If v6 still flat at step 200:** Consider Track B as primary.

---

### Track B — Qwen2.5-VL-7B Fine-Tuning (Parallel Track)

**Motivation:** Sidesteps the entire bootstrapping problem. The model already understands
images and code. Fine-tuning adapts existing capability rather than teaching alignment from zero.

**Architecture:**
```
Code Image → Qwen2.5-VL Vision Encoder (frozen) → Projector (frozen) → Qwen2.5-7B (LoRA) → Output
```

Only the LoRA weights in the LLM are trained (~40M params of 7B total).

**Analogy to classical CV tasks:**
| Task type | CV analog |
|---|---|
| `description` | Image captioning / scene understanding |
| `function_explanation` | Visual QA with reference entity |
| `function_listing` | Object detection |
| `import_listing` | Semantic segmentation |

**Training config:**
- Model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Tasks: `description` + `function_explanation` (11,725 samples)
  - Rationale: only tasks robust to image compression; others require character-level reading
- `max_pixels = 2048 * 28 * 28` → ~2048 visual tokens → ~8-10px/char (readable)
- LoRA: r=16, alpha=32, targets: q/k/v/o_proj + gate/up/down_proj
- Freeze: vision encoder + projector
- Precision: bfloat16 (H100 native)
- Partition: `dgxh100` (2× H100, 80GB each)
- Batch: 1 per GPU, grad_accum=8 → effective batch 16
- LR: 2e-4 (LoRA standard), cosine decay
- Epochs: 3
- Checkpoint dir: `./checkpoints/qwen_vl_v1`

**Scripts:**
- Training: `coder_vl/train_qwen_vl.py`
- SLURM job: `coder_vl/train_qwen_vl.sh`

**Dependency note:** Requires `transformers>=4.49` and `qwen-vl-utils`. The SLURM script
handles `pip install --user` before launching.

**Success criteria:**
- val_loss < 1.5 after epoch 1 (model is adapting to code domain)
- Qualitative: model describes a compressed code image with correct function names/purpose
- Retrieval: given bug description + 20 code images, correct file in top 5

---

## 4. Two-Stage Sniper Method Architecture

```
Bug Report (text)
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 1 — Sniper Scan  (~47K tokens)                   │
│  Input:  89 files × 500 tokens (very compressed)        │
│  Output: Top 10 candidate files                         │
│  Goal:   Structural fingerprinting, rough localization  │
│  Model:  Qwen2.5-VL zero-shot (no fine-tune needed)     │
└─────────────────────────────────────────────────────────┘
       │  Top 10 candidates
       ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 2 — Sniper Focus  (~22K tokens)                  │
│  Input:  10 candidates × 2048 tokens (readable)         │
│  Output: Suspected function/region + confidence         │
│  Goal:   Semantic code understanding, function-level    │
│  Model:  Fine-tuned Qwen2.5-VL (Track B)                │
└─────────────────────────────────────────────────────────┘
       │  Found?  ──Yes──► Pull raw text of 1-2 functions
       │  Not found?            │
       │       │                ▼
       │       ▼        ┌──────────────────────────┐
       │  Next 10        │  STAGE 3 — Raw Text      │
       │  candidates     │  Input: 1-2 files as text│
       │                 │  Output: Exact bug + fix  │
       └────────────────►└──────────────────────────┘
```

**Token budget (success case):**
- Stage 1: 89 × 500 + overhead = ~47K tokens
- Stage 2: 10 × 2048 + overhead = ~22K tokens
- Stage 3: 2 × 5K raw text = ~10K tokens
- **Total: ~79K tokens → fits in 128K context**

**Comparison to text-only:**
- Text-only covers 5-8 files per pass → needs ~15 passes for 89 files → ~90K tokens total
- Two-stage Sniper: covers all 89 files in one shot, similar total tokens, near-certain recall

**Stage 1 training requirement:** Model must rank files by relevance from compressed images.
Test zero-shot first — Qwen2.5-VL already understands "describe this image briefly."
Fine-tune Stage 1 only if zero-shot recall is < 50% (bug file in top 10).

---

## 5. Post-Validation: Adaptive Compression

After concept is validated, find the optimal compression ratio:

**The counterintuitive problem with fixed max_pixels:**
- Fixed budget means larger files get more compressed
- BUT readability requires constant WIDTH (px/char depends on width, not height)
- So large files need MORE tokens to maintain the same readability

**Adaptive approach:** Target constant px/char across all file sizes:
- Target: 8px/char → W_r ≈ 640px for typical 80-char code lines
- Small files (< 200 lines): tokens ≈ 640-1000 (below max budget)
- Medium files (200-500 lines): tokens ≈ 1500-2500
- Large files (500+ lines): tokens ≈ 4000+ (exceeds single-context budget → chunk)

**Chunking for very large files:**
Large files (500+ lines) may need to be split into chunks at the point of highest structural
interest (class/function boundaries), with each chunk processed independently.

**Experiments to run post-validation:**
1. Ablation: accuracy vs. tokens/image at [500, 1000, 1500, 2048] token budgets
2. Per-file-size analysis: does accuracy correlate with compression ratio?
3. Adaptive chunking: compare fixed-cap vs. adaptive-width approaches

---

## 6. Evaluation Plan

### Short-term (validate concept)
1. **Stage 2 generation quality:** val_loss on held-out `description` + `function_explanation`
2. **Qualitative test:** Give model a compressed code image, ask for description. Check if
   function names and purpose are correctly identified.
3. **Retrieval test:** 20-file batch, 1 file is the target. Does it rank correctly?

### Medium-term (prove the method)
4. **SWE-bench Lite subset:** Pick 20 issues, run two-stage Sniper, compare to text-only
   retrieval (BM25 or embedding-based) on file identification accuracy.
5. **Token efficiency:** Total tokens used vs. accuracy at locating the right file.

### Long-term (paper results)
6. **Full SWE-bench Lite:** Two-stage Sniper vs. text-only agent
7. **Compression ablation:** Report accuracy vs. tokens/file tradeoff curve
8. **Track A vs. Track B:** Which architecture achieves better code-image understanding?

---

## 7. Next Immediate Actions

1. **Monitor v6 (job 227177):** Watch `slurm-phase2b-v2-227177.out` at step 200.
   Look for val_pos_cos > 0.5.
2. **Download Qwen2.5-VL-7B:** Run on login node (one-time, ~15GB):
   ```bash
   python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct')"
   ```
3. **Submit Track B:** `sbatch coder_vl/train_qwen_vl.sh` (targets dgxh100)
4. **Zero-shot Stage 1 test:** After Track B downloads, give model a compressed image
   with no fine-tuning and check if it can broadly describe the code.
