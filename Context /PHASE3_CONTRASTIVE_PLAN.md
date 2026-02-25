# Phase 3: Contrastive Training for Sniper Method Localization

*Created: 2026-02-24 | Status: READY TO IMPLEMENT*

---

## Goal

Add InfoNCE contrastive loss to `train_phase2b.py` so the MLP adapter learns
visual embeddings that cluster near semantically related text descriptions.
This directly trains the retrieval pathway needed by the Sniper Method's
localization step:

```
Bug report (text)  →  text encoder  →  query vector [2048D]
                                              ↕ cosine similarity
code_file.py (image) → SigLIP → adapter → mean_pool → index vector [2048D]
```

---

## What Already Exists (Do Not Rebuild)

| Path | Description |
|------|-------------|
| `coder_vl/train_phase2b.py` | Training script to modify — QLoRA + generation loss only |
| `coder_vl/projector.py` | MLP adapter: 1280D → 4096D → 2048D, 13.6M params |
| `precomputed_features_tiled/` | 11,634 × [720, 1280] fp16 `.pt` files — do NOT recompute |
| `data_v2b/manifests/train.jsonl` | 40,083 examples (13 repos) |
| `data_v2b/manifests/val.jsonl` | 2,018 examples (pandas repo) |
| `checkpoints/phase2a_v6/best.pt` | Adapter init weights — load these, do not train from scratch |
| `checkpoints/contrastive_v4/best.pt` | Has good val_cos=0.840; may use adapter weights from here instead |

The data is already rendered and precomputed. **Do not re-run data_gen or precompute jobs.**

---

## Architecture: What Changes

### Training Objective

```
L_total = L_generation + 0.1 * L_InfoNCE(visual_emb, text_emb)
```

Where:
- `L_generation` = cross-entropy next-token loss (ALL task types, unchanged)
- `L_InfoNCE` = SigLIP contrastive loss (ONLY `description` and `function_explanation` tasks)
- `visual_emb` = `mean_pool(adapter(features))` → [B, 2048], L2-normalized
- `text_emb` = `mean_pool(embed_fn(answer_tokens))` → [B, 2048], L2-normalized, **no gradients**

### Why SigLIP+bias (not plain InfoNCE)

Plain InfoNCE failed at batch=64 (v2: val_cos=0.131). SigLIP without bias collapsed (v3: val_cos=-0.832). SigLIP+bias worked (v4: val_cos=0.840, bias=-9.005). Use SigLIP+bias.

### Why embed_fn for text embedding

- `embed_fn` = `coder.get_input_embeddings()` — the token embedding lookup table
- Already loaded in VRAM, zero extra memory
- Not modified by LoRA (LoRA targets q_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj)
- Output is 2048D — same space as visual_emb, no projection head needed
- Run with `torch.no_grad()` — text_emb is a fixed target, gradients only flow through adapter

### Which tasks feed InfoNCE

| Task | Generation loss | InfoNCE loss | Reason |
|------|----------------|--------------|--------|
| `description` | ✅ | ✅ | Semantically dense, good retrieval anchor |
| `function_explanation` | ✅ | ✅ | Semantically dense, good retrieval anchor |
| `function_listing` | ✅ | ❌ | Structurally accurate but semantically shallow; corrupts embedding space |
| `function_signatures` | ✅ | ❌ | Same reason |
| `class_listing` | ✅ | ❌ | Same reason |
| `import_listing` | ✅ | ❌ | Same reason |

---

## Exact Changes Required to `train_phase2b.py`

### 1. New imports (top of file)

```python
import torch.nn.functional as F
```

### 2. SigLIP loss function (add after imports, before Dataset class)

```python
def siglip_contrastive_loss(visual_emb, text_emb, log_temp, bias):
    """
    SigLIP contrastive loss with learnable temperature and bias.
    Avoids representation collapse at small batch sizes (proven in contrastive_v4).

    Args:
        visual_emb: [N, D] float32, L2-normalized
        text_emb:   [N, D] float32, L2-normalized, detached (no grad)
        log_temp:   scalar Parameter, init log(0.07) ≈ -2.659
        bias:       scalar Parameter, init -10.0

    Returns scalar loss. Returns 0.0 if N < 2.
    """
    N = visual_emb.size(0)
    if N < 2:
        return torch.tensor(0.0, device=visual_emb.device, requires_grad=True)
    logits = torch.matmul(visual_emb, text_emb.T) * log_temp.exp() + bias
    labels = torch.eye(N, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, labels)
```

### 3. `PrecomputedDataset.__getitem__` — return two new fields

Current return (line 126):
```python
return {"input_ids": input_ids, "labels": labels, "features": features}
```

Replace with:
```python
return {
    "input_ids":   input_ids,
    "labels":      labels,
    "features":    features,
    "task_type":   ex.get("task_type", ""),       # str — for InfoNCE gating
    "answer_text": conv[1]["content"],             # str — raw answer for text_emb
}
```

### 4. Add collate_fn (add as a top-level function, before main())

The default DataLoader collate cannot stack variable-length strings. Add:

```python
def collate_fn(batch):
    return {
        "input_ids":   torch.stack([b["input_ids"]  for b in batch]),
        "labels":      torch.stack([b["labels"]     for b in batch]),
        "features":    torch.stack([b["features"]   for b in batch]),
        "task_type":   [b["task_type"]   for b in batch],
        "answer_text": [b["answer_text"] for b in batch],
    }
```

Pass `collate_fn=collate_fn` to both DataLoader calls (train and val).

### 5. Learnable temperature and bias parameters (add in main(), after adapter is created)

After `adapter = adapter.to(device)`:

```python
# Learnable contrastive parameters (on same device as adapter)
log_temp = torch.nn.Parameter(
    torch.tensor([-2.659], device=device)   # exp(-2.659) ≈ 0.07
)
bias = torch.nn.Parameter(
    torch.tensor([-10.0], device=device)
)
```

### 6. Add contrast_params to optimizer (modify optimizer definition)

Current param groups (lines ~453-456):
```python
optimizer = AdamW([
    {"params": adapter_params, "lr": args.lr_adapter},
    {"params": lora_params,    "lr": args.lr_lora},
], weight_decay=0.0)
```

Replace with:
```python
contrast_params = [log_temp, bias]
optimizer = AdamW([
    {"params": adapter_params,  "lr": args.lr_adapter},
    {"params": lora_params,     "lr": args.lr_lora},
    {"params": contrast_params, "lr": args.lr_adapter},   # same lr as adapter
], weight_decay=0.0)
```

### 7. Add `--contrast_weight` CLI argument

Add with the other argparse arguments:
```python
parser.add_argument("--contrast_weight", type=float, default=0.1,
                    help="Weight for InfoNCE contrastive loss (default: 0.1)")
```

### 8. Training loop — add contrastive loss computation

Current forward pass (lines ~531-543):
```python
ids  = batch["input_ids"].to(device)
lbl  = batch["labels"].to(device)
feat = batch["features"].to(device)

projected  = adapter(feat.float()).half()
embeds, mask = replace_image_tokens(ids, projected, image_token_id, embed_fn)
adj_labels   = expand_labels(lbl, ids, image_token_id, feat.size(1))

loss = coder(inputs_embeds=embeds, attention_mask=mask, labels=adj_labels).loss
loss = loss / args.grad_accum
loss.backward()
```

Replace with:
```python
ids       = batch["input_ids"].to(device)
lbl       = batch["labels"].to(device)
feat      = batch["features"].to(device)
tasks     = batch["task_type"]      # list[str]
answers   = batch["answer_text"]    # list[str]

projected    = adapter(feat.float()).half()
embeds, mask = replace_image_tokens(ids, projected, image_token_id, embed_fn)
adj_labels   = expand_labels(lbl, ids, image_token_id, feat.size(1))

# Generation loss — all tasks
L_gen = coder(inputs_embeds=embeds, attention_mask=mask, labels=adj_labels).loss

# Contrastive loss — description and function_explanation only
CONTRASTIVE_TASKS = {"description", "function_explanation"}
c_mask = torch.tensor([t in CONTRASTIVE_TASKS for t in tasks], device=device)
n_contrast = c_mask.sum().item()

L_contrast = torch.tensor(0.0, device=device)
if n_contrast >= 2:
    # Visual embedding: mean-pool adapter output for contrastive examples
    vis_emb = projected[c_mask].float().mean(dim=1)          # [N, 2048]

    # Text embedding: encode answer text through frozen embed_fn
    contrast_answers = [answers[i] for i, c in enumerate(c_mask.tolist()) if c]
    ans_tok = tokenizer(
        contrast_answers,
        return_tensors="pt",
        max_length=256,
        truncation=True,
        padding=True,
    ).input_ids.to(device)
    with torch.no_grad():
        txt_emb = embed_fn(ans_tok).float().mean(dim=1)      # [N, 2048]

    vis_emb = F.normalize(vis_emb, dim=-1)
    txt_emb = F.normalize(txt_emb, dim=-1)
    L_contrast = siglip_contrastive_loss(vis_emb, txt_emb, log_temp, bias)

loss = (L_gen + args.contrast_weight * L_contrast) / args.grad_accum
loss.backward()
```

### 9. Update logging to show contrastive loss

In the log block (around line 554-559), change to:
```python
if is_main and global_step % args.log_steps == 0:
    lr_a = optimizer.param_groups[0]["lr"]
    lr_l = optimizer.param_groups[1]["lr"]
    print(f"  step={global_step}  "
          f"loss={loss.item() * args.grad_accum:.4f}  "
          f"L_gen={L_gen.item():.4f}  "
          f"L_contrast={L_contrast.item():.4f}  "
          f"temp={log_temp.exp().item():.3f}  bias={bias.item():.3f}  "
          f"lr_adapter={lr_a:.2e}  lr_lora={lr_l:.2e}")
```

### 10. Save log_temp and bias in checkpoint

In `save_checkpoint()`, add to the dict:
```python
"log_temp": log_temp.detach().cpu(),
"bias":     bias.detach().cpu(),
```

And restore them in the resume block:
```python
if "log_temp" in ckpt:
    log_temp.data = ckpt["log_temp"].to(device)
    bias.data     = ckpt["bias"].to(device)
```

### 11. Update evaluate() to also report mean positive cosine similarity

Add `log_temp` and `bias` as parameters, and compute pos_cos for contrastive tasks:

```python
@torch.no_grad()
def evaluate(adapter, coder, loader, image_token_id, embed_fn, device,
             log_temp, bias, tokenizer, contrast_weight):
    CONTRASTIVE_TASKS = {"description", "function_explanation"}
    adapter.eval(); coder.eval()
    total_loss, total_pos_cos, n, n_contrast = 0.0, 0.0, 0, 0

    for batch in loader:
        ids     = batch["input_ids"].to(device)
        lbl     = batch["labels"].to(device)
        feat    = batch["features"].to(device)
        tasks   = batch["task_type"]
        answers = batch["answer_text"]

        projected    = adapter(feat.float()).half()
        embeds, mask = replace_image_tokens(ids, projected, image_token_id, embed_fn)
        adj_labels   = expand_labels(lbl, ids, image_token_id, feat.size(1))
        L_gen = coder(inputs_embeds=embeds, attention_mask=mask, labels=adj_labels).loss

        c_mask = torch.tensor([t in CONTRASTIVE_TASKS for t in tasks], device=device)
        if c_mask.sum() >= 2:
            vis_emb = F.normalize(projected[c_mask].float().mean(dim=1), dim=-1)
            contrast_answers = [answers[i] for i, c in enumerate(c_mask.tolist()) if c]
            ans_tok = tokenizer(contrast_answers, return_tensors="pt",
                                max_length=256, truncation=True, padding=True).input_ids.to(device)
            txt_emb = F.normalize(embed_fn(ans_tok).float().mean(dim=1), dim=-1)
            L_contrast = siglip_contrastive_loss(vis_emb, txt_emb, log_temp, bias)
            pos_cos = (vis_emb * txt_emb).sum(dim=-1).mean().item()
            total_pos_cos += pos_cos
            n_contrast += 1
        else:
            L_contrast = torch.tensor(0.0)

        val_loss = (L_gen + contrast_weight * L_contrast).item()
        total_loss += val_loss
        n += 1

    adapter.train(); coder.train()
    return total_loss / max(n, 1), total_pos_cos / max(n_contrast, 1)
```

Update the call site to unpack the tuple:
```python
vl, pos_cos = evaluate(inner_adapter, inner_coder, val_loader,
                        image_token_id, embed_fn, device,
                        log_temp, bias, tokenizer, args.contrast_weight)
# ... (after all_reduce on vl, do same for pos_cos if distributed)
if is_main:
    print(f"  step={global_step}  val_loss={vl:.4f}  val_pos_cos={pos_cos:.3f}  "
          f"temp={log_temp.exp().item():.3f}  bias={bias.item():.3f}")
```

---

## New SLURM Job: `train_phase2b_v2.sh`

Create a new SLURM script (do not overwrite the original):

```bash
#!/bin/bash
#SBATCH --job-name=phase2b_v2
#SBATCH --partition=dgx
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-phase2b-v2-%j.out
#SBATCH --error=slurm-phase2b-v2-%j.err

PYTHON="$HOME/DS OCR/envs/deepseek-ocr/bin/python"
REPO="$HOME/CoderOCR/OCR-Coder"

cd "$REPO"

torchrun --nproc_per_node=2 coder_vl/train_phase2b.py \
    --features_dir   ./precomputed_features_tiled \
    --train_manifest data_v2b/manifests/train.jsonl \
    --val_manifest   data_v2b/manifests/val.jsonl \
    --checkpoint_dir ./checkpoints/phase2b_v2 \
    --init_from      ./checkpoints/phase2a_v6/best.pt \
    --batch_size     4 \
    --grad_accum     4 \
    --epochs         3 \
    --lr_adapter     1e-5 \
    --lr_lora        2e-5 \
    --contrast_weight 0.1 \
    --eval_steps     200 \
    --log_steps      10 \
    --ckpt_interval  1800
```

**Key parameters:**
- `lr_adapter=1e-5` — CRITICAL: 1e-4 catastrophically forgets contrastive alignment (proven in v4). Never go above 1e-5 for adapter.
- `batch_size=4` with 2 GPUs = effective batch of 32 (×4 grad_accum). Contrastive batch N is the subset of contrastive-task examples, typically ~1-2 per device batch.
- `epochs=3` — one more than Phase 2b v1 to account for the new objective

---

## New Evaluation Script: `coder_vl/eval_retrieval.py`

Create this script to measure Recall@k — the primary metric going forward (replacing ROUGE-L).

### What it does

For each image in the val set (description/function_explanation tasks only):
1. Compute `visual_emb = mean_pool(adapter(precomputed_features))` → [2048]
2. For each text description in the val set, compute `text_emb = mean_pool(embed_fn(answer_tokens))` → [2048]
3. Build similarity matrix: [N_images × N_texts]
4. For image i, rank all texts by cosine similarity; check if text_i appears in top-k
5. Report Recall@1, Recall@5, Recall@10

### Key implementation notes

- Load adapter weights from checkpoint: `adapter.load_state_dict(ckpt["adapter_state_dict"])`
- Load LoRA weights into coder: `coder.load_state_dict(ckpt["lora_state_dict"], strict=False)`
- `embed_fn = coder.get_input_embeddings()` — use for text embedding
- Filter val manifest to `task_type in {"description", "function_explanation"}` only
- Run in `@torch.no_grad()` mode throughout
- Report separately for `description` vs `function_explanation` tasks

### Spec

```python
"""
eval_retrieval.py — Retrieval Recall@k evaluation for Sniper Method localization.

Usage:
    python coder_vl/eval_retrieval.py \
        --checkpoint  ./checkpoints/phase2b_v2/best.pt \
        --val_manifest data_v2b/manifests/val.jsonl \
        --features_dir ./precomputed_features_tiled \
        --output       retrieval_results.json
"""
# CLI args: --checkpoint, --val_manifest, --features_dir, --output, --coder_model
# Loads: adapter (from checkpoint), coder (4-bit, for embed_fn only)
# Filters: task_type in {"description", "function_explanation"}
# Computes: full N×N cosine similarity matrix on CPU after batch encoding
# Reports: Recall@1, Recall@5, Recall@10 (overall + per task_type)
# Saves: JSON with all metrics + per-example results
```

---

## Success Criteria

| Metric | Baseline (Phase 2b v1) | Target |
|--------|----------------------|--------|
| val_pos_cos (description tasks) | ~0.0 (not measured) | > 0.5 |
| Retrieval Recall@5 (description) | 0.7% (near random) | > 15% |
| Retrieval Recall@5 (function_explanation) | 9.8% | > 25% |
| val_loss (generation) | 1.31 | < 1.40 (should not regress much) |

If val_pos_cos is rising but Recall@k is still low after training, the issue
is semantic diversity (13 repos too similar — need 50 repos).

---

## Critical Pitfalls — Do Not Repeat

From the training history:

1. **lr_adapter > 1e-5 destroys contrastive alignment** — Job 223660 at lr=1e-4 produced Chinese
   output loops. The generation loss gradient at high lr overwrites the contrastive init.
   Hard limit: lr_adapter ≤ 1e-5.

2. **SigLIP without bias collapses** — Job 223372 got val_cos=-0.832 because 63:1 negative:positive
   ratio at batch=64 made anti-alignment the loss minimum. The bias=-10.0 init breaks this.
   Always initialize bias=-10.0.

3. **NCCL timeout from uneven eval** — Job 223447 crashed at step 200 because rank 0 ran eval
   solo while ranks 1-3 waited. The fix (DistributedSampler on val_loader + dist.all_reduce)
   is already in train_phase2b.py. Do not remove it.

4. **Text embeddings must not receive gradients** — Use `torch.no_grad()` when computing
   `txt_emb = embed_fn(ans_tok)`. If gradients flow through text, the loss will move text
   embeddings toward visual ones, corrupting the LLM's embedding space.

5. **`projected` is fp16 — cast to float32 before mean_pool for contrastive** —
   `projected[c_mask].float().mean(dim=1)` not `.half().mean(dim=1)`. F.normalize and
   matmul in fp32 for numerical stability.

---

## File Checklist

Files to **create or modify**:

- [ ] `coder_vl/train_phase2b.py` — apply all 11 changes listed above
- [ ] `coder_vl/train_phase2b_v2.sh` — new SLURM job (do not overwrite existing .sh)
- [ ] `coder_vl/eval_retrieval.py` — new evaluation script
- [ ] `coder_vl/eval_retrieval.sh` — SLURM job for eval (dgx, 1 GPU, 2h)

Files to **not touch**:

- `data_v2b/` — data already generated
- `precomputed_features_tiled/` — features already computed
- `coder_vl/projector.py` — adapter architecture unchanged
- `checkpoints/phase2a_v6/best.pt` — init weights, read-only
