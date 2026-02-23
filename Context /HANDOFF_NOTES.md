# Phase 2b Training Script Spec (2026-02-23)

## Task
Write two files: `coder_vl/train_phase2b.py` and `coder_vl/train_phase2b.sh`.
The training script is the main deliverable. Base it heavily on `coder_vl/train_projector.py`
(Phase 2a) — read that file first. Most of the Dataset, token-replacement, and eval logic
is identical and should be reused directly.

---

## Key Decisions Made This Session

### Hardware & Memory
- **Partition:** `dgx` (V100, NOT dgxh100/H100)
- **GPUs:** 2× V100 (32 GB each)
- **Why V100 works:** 4-bit QLoRA cuts model from ~32 GB (fp16) to ~9 GB
- **Verified memory budget:** ~11.9 GB total, 20 GB headroom on V100
  - 4-bit model: 8.96 GB | LoRA weights: 28 MB | Adapter: 54 MB
  - Optimizer (AdamW on 20.7M trainable): 166 MB | Grads: 83 MB
  - Activations (grad_ckpt, seq=260): 115 MB | CUDA overhead: 2.5 GB

### LoRA Target Modules (VERIFIED from modeling_deepseek.py)
The PHASE2_PLAN.md doc was WRONG. It listed `q_a_proj`/`q_b_proj` but those don't
exist in the Lite model. From config: `q_lora_rank = None` → the model takes the
non-MLA branch → uses a single `q_proj`.

**Correct targets:** `["q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]`

Do NOT target `q_a_proj`, `q_b_proj` (don't exist in Lite).
Do NOT target FFN layers (`gate_proj`, `up_proj`, `down_proj`) — MoE explosion risk.

### Sequence Length
Verified from 2000 sampled examples: p99=220 tokens, max=358. Keep `max_seq_length=260`
(same as Phase 2a v6) — no need to increase.

### Text-Only Replay
Skip for now. G10 (text preservation) is a stretch goal; recoverable in v2 if needed.
The `--text_replay_manifest` arg can be added as a stub/optional arg but not used.

### Timing
Phase 2a v6 measured: ~7.75s per micro-batch on 1 V100 (batch=4, seq=260, 4-bit coder).
Phase 2b with LoRA adds ~20% overhead → ~9.3s per micro-batch.
2× V100, batch=4, accum=4: 1,250 gradient steps → ~13h. Fits in 24h.

---

## train_phase2b.py — Full Spec

### Imports (additions over train_projector.py)
```python
from peft import get_peft_model, LoraConfig, TaskType
import time
```

### Section 0: Argument Parser
Add these args on top of what train_projector.py has:
```
--lora_r          int,   default=16
--lora_alpha      int,   default=32
--lora_dropout    float, default=0.05
--lr_lora         float, default=2e-5   # LoRA param group lr
--lr_adapter      float, default=1e-5   # adapter param group lr (lower to protect contrastive init)
--ckpt_interval   int,   default=1800   # seconds between time-based checkpoints (30 min)
--resume          str,   default=None   # explicit checkpoint to resume from; if None, auto-detect latest
```
Remove `--lr` (replaced by `--lr_lora` and `--lr_adapter`).
Change `--init_from` meaning: now loads adapter weights from a Phase 2a checkpoint into
the adapter only (not the coder). Default: `./checkpoints/phase2a_v6/best.pt`.

### Section 1: Load Coder Model
Same BitsAndBytesConfig as Phase 2a (4-bit, bnb_4bit_compute_dtype=torch.float16).
Same `AutoModelForCausalLM.from_pretrained(...)` with `device_map={"": local_rank}`.
Same special token addition + `resize_token_embeddings`.

**New — apply LoRA immediately after loading:**
```python
lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    target_modules=["q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"],
    task_type=TaskType.CAUSAL_LM,
    bias="none",
)
coder = get_peft_model(coder, lora_config)
coder.print_trainable_parameters()   # log to confirm ~7.1M LoRA params
```

Do NOT call `coder.gradient_checkpointing_enable()` before `get_peft_model` —
peft has its own method. After get_peft_model:
```python
coder.enable_input_require_grads()   # required for grad_ckpt with peft
coder.gradient_checkpointing_enable()
coder.train()
```

### Section 2: Create Adapter
Identical to Phase 2a. Then load from `--init_from` checkpoint:
```python
ckpt = torch.load(args.init_from, map_location="cpu")
state = ckpt.get("adapter_state_dict", ckpt)
adapter.load_state_dict(state)
```

### Section 3: Auto-Resume Logic
**New section** — after adapter is created, before dataset loading:
```python
resume_path = _find_resume_checkpoint(args.checkpoint_dir, args.resume)
# _find_resume_checkpoint: if args.resume given, use it; else glob checkpoint_dir for
# "step_*.pt" files, sort by step number, return latest. Return None if none found.

start_epoch, start_step, best_val_loss = 0, 0, float("inf")
if resume_path:
    ckpt = torch.load(resume_path, map_location="cpu")
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    coder.load_state_dict(ckpt["lora_state_dict"], strict=False)  # LoRA weights only
    start_epoch = ckpt["epoch"]
    start_step  = ckpt["global_step"]
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print(f"  Resumed from {resume_path} at step {start_step}")
```

Note: load LoRA state with `strict=False` because the checkpoint contains only LoRA
delta weights, not the full coder state dict.

### Section 4: Datasets
Identical to Phase 2a — `PrecomputedDataset` class is unchanged.
Manifests: `data_v2b/manifests/train.jsonl` and `data_v2b/manifests/val.jsonl`.

### Section 5: Optimizer — TWO PARAM GROUPS
```python
lora_params    = [p for n, p in coder.named_parameters()  if p.requires_grad]
adapter_params = list(adapter.parameters())

optimizer = AdamW([
    {"params": adapter_params, "lr": args.lr_adapter},
    {"params": lora_params,    "lr": args.lr_lora},
], weight_decay=0.0)
```
Scheduler is built on total steps as before (cosine with warmup).

### Section 6: Checkpointing
**save_checkpoint** needs to save both adapter and LoRA:
```python
def save_checkpoint(adapter, coder, optimizer, scheduler, step, epoch, best_val, ckpt_dir, name):
    inner_adapter = adapter.module if is_distributed else adapter
    # Extract only LoRA state (not full 4-bit model)
    lora_state = {k: v for k, v in coder.state_dict().items()
                  if "lora_" in k}
    path = os.path.join(ckpt_dir, f"{name}.pt")
    torch.save({
        "adapter_state_dict":    inner_adapter.state_dict(),
        "lora_state_dict":       lora_state,
        "optimizer_state_dict":  optimizer.state_dict(),
        "scheduler_state_dict":  scheduler.state_dict(),
        "global_step":           step,
        "epoch":                 epoch,
        "best_val_loss":         best_val,
    }, path)
```

**Keep-last-3 helper:**
```python
def _cleanup_old_checkpoints(ckpt_dir, keep=3):
    # glob step_*.pt, sort by step number, delete all but last `keep`
```

**Time-based checkpointing** in training loop:
```python
last_ckpt_time = time.time()
# ... inside training loop, after optimizer.step():
if time.time() - last_ckpt_time >= args.ckpt_interval:
    save_checkpoint(..., name=f"step_{global_step}")
    _cleanup_old_checkpoints(args.checkpoint_dir)
    last_ckpt_time = time.time()
```

### Section 7: Training Loop
Largely identical to Phase 2a. Key differences:
- `adapter.train()` AND `coder.train()` at top of epoch (LoRA modules need train mode)
- `clip_grad_norm_` applied to `adapter.parameters() + lora_params` combined:
  ```python
  all_trainable = list(adapter.parameters()) + lora_params
  torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
  ```
- Skip batches before `start_step` if resuming mid-epoch (use a counter to fast-forward
  the dataloader, or simpler: just start from epoch 0 and skip already-done steps)
  Simplest approach: on resume, if `start_epoch > 0` skip to the right epoch; within
  an epoch, just let training re-run the early steps (small cost since steps are fast)

### Section 8: Evaluation
Identical to Phase 2a `evaluate()` function — no changes needed.
During eval, call `adapter.eval()` and `coder.eval()`. Restore both to `.train()` after.

### Section 9: DDP Wrapping
Same pattern as Phase 2a v6:
```python
if is_distributed:
    adapter = DDP(adapter, device_ids=[local_rank])
    # Do NOT wrap coder in DDP — peft handles parameter sync differently
    # Instead, broadcast LoRA params manually if needed, or rely on torchrun sync
```
Actually: wrapping a peft model in DDP is supported but requires care.
Recommended pattern:
```python
coder = DDP(coder, device_ids=[local_rank], find_unused_parameters=True)
adapter = DDP(adapter, device_ids=[local_rank])
```
`find_unused_parameters=True` is needed because not all coder parameters participate
in every forward pass (MoE routing means some experts are skipped).

---

## train_phase2b.sh — Full Spec

```bash
#!/bin/bash
#SBATCH --job-name=phase2b-train
#SBATCH --partition=dgx
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00
#SBATCH --output=slurm-phase2b-%j.out
#SBATCH --error=slurm-phase2b-%j.err
#SBATCH --mail-user=gloriosog@msoe.edu
#SBATCH --mail-type=END,FAIL
```

Commands:
1. Print job info header (Job ID, node, start time, nvidia-smi)
2. Set PYTHON and TORCHRUN paths (same as Phase 2a: `$HOME/DS OCR/envs/deepseek-ocr/bin/...`)
3. `cd "$HOME/CoderOCR/OCR-Coder" || exit 1`
4. Install peft if missing:
   ```bash
   "$PYTHON" -c "import peft" 2>/dev/null || "$PYTHON" -m pip install peft --quiet
   ```
5. Run with torchrun, 2 GPUs:
   ```bash
   "$TORCHRUN" --nproc_per_node=2 coder_vl/train_phase2b.py \
     --features_dir   ./precomputed_features_tiled \
     --train_manifest data_v2b/manifests/train.jsonl \
     --val_manifest   data_v2b/manifests/val.jsonl \
     --batch_size     4 \
     --lr_adapter     1e-5 \
     --lr_lora        2e-5 \
     --epochs         2 \
     --grad_accum     4 \
     --max_seq_length 260 \
     --checkpoint_dir ./checkpoints/phase2b \
     --eval_steps     200 \
     --log_steps      10 \
     --ckpt_interval  1800 \
     --init_from      ./checkpoints/phase2a_v6/best.pt \
     --lora_r         16 \
     --lora_alpha     32 \
     --lora_dropout   0.05
   ```
6. Capture EXIT_CODE, print footer with end time and exit.
7. Add a comment at the bottom showing the job-chain command:
   ```bash
   # To chain a continuation job:
   # sbatch --dependency=afterany:$SLURM_JOB_ID coder_vl/train_phase2b.sh
   ```

---

## Files to Read Before Writing
1. `coder_vl/train_projector.py` — Phase 2a training script (base for Phase 2b)
2. `coder_vl/train_phase2a.sh` — Phase 2a SLURM script (base for Phase 2b .sh)
3. `coder_vl/projector.py` — adapter module (unchanged, just imported)

## Pre-Submit Checklist
- [ ] `peft` installed: `"$HOME/DS OCR/envs/deepseek-ocr/bin/python" -m pip install peft`
- [ ] Verify peft + DDP works: check peft docs for torchrun compatibility with current version
- [ ] Confirm `./checkpoints/phase2a_v6/best.pt` exists (it does)
- [ ] Confirm `data_v2b/manifests/train.jsonl` has 40,083 lines (it does)

---

# Phase 2a Handoff Notes (2026-02-13)


## Immediate Status

**Phase 2a training completed but model is BROKEN - does not use visual features at all.**

### What Just Happened (Last 2 Hours)

1. ✅ Ran quick evaluation (15 examples) → all gates failed
2. 🔍 Deep debugging session to find root cause
3. ❌ Binary classification test (simplest possible task) → 0% accuracy
4. 📝 Updated all context files with findings

### Critical Discovery

**The model cannot use visual features in ANY capacity:**
- Not for complex tasks (listing functions) ❌
- Not for simple tasks (yes/no questions) ❌
- Model either repeats questions or refuses to answer
- **This is a fundamental architectural/training failure, not weak adapter**

---

## What We Know For Sure

### ✅ What's Working
1. **Vision encoder works** — features are diverse (cosine sim 0.19-0.66)
2. **Training converged** — loss decreased smoothly to 1.27
3. **Data format correct** — same format train/eval
4. **Token replacement logic correct** — verified identical between train/eval
5. **Precomputed features exist** — 2,165 feature files, all valid

### ❌ What's Broken
1. **Model ignores visual tokens completely**
2. **Generation fails even for yes/no questions**
3. **Model either repeats prompt or refuses to answer**
4. **Zero capability to decode visual information**

### 🤔 What We DON'T Know Yet
1. **Is the model attending to visual tokens?** (need attention visualization)
2. **Are embeddings properly aligned?** (need to check embedding space)
3. **Is the issue in training or architecture?** (likely both)

---

## Recommended Next Steps

### Option A: Diagnostic Test with Perfect Features (RECOMMENDED FIRST)
**Goal:** Verify token insertion mechanism works at all

Instead of using vision encoder features, use ground-truth text embeddings:
- Take the actual code text from the image
- Embed it with Coder-V2's embedding layer
- Use those as "visual" features
- If this works → problem is vision encoder / feature mapping
- If this fails → token insertion mechanism is broken

**Time:** 1-2 days | **Success probability:** 90% (for diagnosis)

### Option B: Swap Vision Encoder (If Option A works)
**Goal:** Use vision encoder designed for language alignment

- Replace DeepSeek-OCR-2 with SigLIP or CLIP
- These are pre-trained for vision-language tasks
- Features already in language-compatible space
- Keep same adapter architecture

**Time:** 3-5 days | **Success probability:** 60-75%

### Option C: Architectural Change
**Goal:** Use proven multimodal architecture

- Flamingo-style cross-attention (gated fusion)
- Or adapt Qwen2-VL's approach to Coder-V2
- Don't insert tokens, use cross-attention layers

**Time:** 1-2 weeks | **Success probability:** 70-85%

---

## Files for Next Session

### Read These First
- `Context /claude.md` — Current status (lines 1-75)
- `Context /PHASE2A_DEBUG_SUMMARY.md` — Complete debug report
- `Context /HANDOFF_NOTES.md` — This file

### Debug Outputs
- `2a_eval_quick.out` — Evaluation failure (all gates)
- `test_binary.out` — Binary test 0% accuracy
- `debug_single.out` — Single example trace

### Code to Review
- `coder_vl/train_projector.py` — Training script (check lines 143-186 for token replacement)
- `coder_vl/evaluate_phase2a.py` — Evaluation script (check lines 104-188)
- `coder_vl/projector.py` — Current adapter (2-layer MLP, 13.6M params)

---

## My Recommendation

**Do Option A first (diagnostic test), then decide:**

1. Create test where "visual features" are actually text embeddings
2. If model learns from those → vision encoder is the problem
3. Then try Option B (swap encoder)

**Do NOT spend weeks on Option A without this diagnostic.**

---

## Key Metrics to Watch

If you continue training with new approach:

1. **Binary classification accuracy** (during training)
   - Should reach >80% within first 100 steps
   - If stuck at 50% (random) → stop and debug

2. **Attention to visual tokens** (if possible to visualize)
   - Should be >0.1 average weight
   - If near zero → model ignoring vision

3. **Actual task performance** (not just loss)
   - Test on 10 examples every 50 steps
   - Manual inspection of outputs

## Questions to Answer

1. Can the model attend to inserted tokens at all?
2. Are visual embeddings in compatible space with text?
3. Should we pivot to different architecture entirely?
