# Claude Code Assistant Context

**Project:** DeepSeek-Coder-VL - Vision-Enabled Code Reasoning Model  
**Goal:** Build a multimodal code model by combining DeepSeek-VL2's vision encoder with DeepSeek-Coder-V2's code reasoning capabilities.

---

## Quick Status

**Current Phase:** Phase 3.4 — Stage 1 (Lossless Decoder) training ACTIVE on DGX (job 239379); H100 4-GPU scripts ready, need resubmission
**Last Updated:** 2026-04-01

- ✅ **DGX V100 Stage 1 running** — Job 239379 (4x V100, QLoRA nf4, LoRA r=32 α=64). Epoch 1 ~57% complete, loss 0.27–0.59. Continuation job queued (dependency=afterany:239379), will resume from `epoch_latest` for 2 more epochs.
- ✅ **H100 4-GPU scripts created** — `run_stage1_4h100.sh`, `run_stage2_4h100.sh` (5 epochs, 24h, 4x H100). dtype fix applied (`x = x.to(self.conv.weight.dtype)` in ConvRoPEProjector.forward). Job 239381 crashed before fix — needs resubmission.
- ✅ **DGX train_stage1.py overhauled** — `MVV/Phase_3/Phase_3_4/DGX_run/train_stage1.py`: manual grad-sync (no DDP), fp16 autocast, GradScaler(init_scale=2**10), has_valid_grad guard, --resume-from arg, mid-epoch checkpointing every 500 steps.
- ✅ **Inference LoRA loading fixed** — `run_inference_stage1.py` now uses `PeftModel.from_pretrained` (replaces fragile get_peft_model + load_adapter).
- ✅ **Reasoning dataset complete** — `MVV/Phase_3/Phase_3_4/reasoning_dataset.jsonl` (macro+micro Q&A pairs).
- ✅ **Phase 3.3 inference** — 5.7% line overlap on smoke test sample (undertrained baseline).

**Next Steps:**
1. Wait for DGX job 239379 + continuation to finish. Check `slurm-stage1-4gpu-*.out` for loss trends.
2. Resubmit H100 Stage 1: `sbatch MVV/Phase_3/Phase_3_4/run_stage1_4h100.sh` (dtype fix now applied).
3. Run inference on best checkpoint: `sbatch MVV/Phase_3/Phase_3_4/DGX_run/run_inference_stage1.sh`.
4. Once Stage 1 converges (val_loss < 1.40), submit Stage 2 reasoning fine-tune.

---

## Key Context

### The Problem
- SWE-bench requires reading 50-100+ files, but context windows are limited
- Text-only: ~5-10 files fit in 128K context
- **Our solution:** Code images → visual tokens (10-20x compression) → 50-100+ files fit

### The Architecture
```
Code Image → SigLIP Vision Encoder (1280D) → Projection Adapter (1280D→2048D) → DeepSeek-Coder-V2-Lite (2048D) → Patch
```

**Projection Adapter:**
- 2-layer MLP: `Linear(1280, 4096) → GELU → Linear(4096, 2048)`
- 13.6M parameters (5.2M + 8.4M)

### Key Finding (Phase 1)
- Visual tokens **capped at 1,120** (256 base + 6×144 patches)
- Compression ratios: 3.30x (medium files) to **20.16x** (large files)
- Small files (<100 lines) don't benefit (expected)

### Subagent Workflow (Context Management)

To prevent context window overflow, follow these rules each session:

| Task | Tool |
|---|---|
| Explore unfamiliar file structure | `Explore` agent (not direct Read) |
| Write scripts >100 lines | `general-purpose` agent (returns confirmation only) |
| Analyze results JSON | `general-purpose` agent (returns bullet summary only) |
| Read a specific small known file | Direct `Read` (fine) |
| Targeted search | Direct `Grep`/`Glob` (fine) |

- **Never re-read files already analyzed earlier in the same session**
- **End of session:** run `/pass` to create handoff note
- **Start of session:** run `/prime` to bootstrap from essential sections only

---

## Important Documents

- **`DEEPSEEK_CODER_VL_PLAN.md`** - Full technical plan, architecture, phases, references
- **`WORKSPACE_NOTES.md`** - Operational progress, execution details, technical challenges, next actions
- **`PHASE2_PLAN.md`** - Phase 2 execution runbook, experiment matrix, cross-instance decision log
- **`PROJECT_PLAN.md`** - Overall project tracking (broader scope than Coder-VL)

**When to reference:**
- Need detailed architecture? → `DEEPSEEK_CODER_VL_PLAN.md`
- Need current status/blockers? → `WORKSPACE_NOTES.md`
- Need Phase 2 execution decisions and A/B outcomes? → `PHASE2_PLAN.md`
- Need overall project context? → `PROJECT_PLAN.md`

### External Dependency Notes (Rosie HPC)
- **Primary docs:** `https://docs.hpc.msoe.edu/#/`
- **Known policy snapshot (verify against docs before long runs):**
  - `--time` default: 1 hour if not set
  - Standard `gpu`/`compute` max walltime: 24 hours
  - Use checkpoint + SLURM job chaining for multi-day training
- **Rule:** Do not rely on memory alone across AI instances; copy key policy values into `PHASE2_PLAN.md` with a verification date and source section name.

---

## Code Conventions & Patterns

### Rosie Supercomputer (SLURM)
- **User account:** `gloriosog` (filesystem username - use this for `/scratch/` and `/data/` paths, not the full email)
- **Python environment:** `$HOME/DS OCR/envs/deepseek-ocr/` (conda env)
  - **Python path:** `"$HOME/DS OCR/envs/deepseek-ocr/bin/python"` (use in scripts, not `conda activate`)
  - **Pip path:** `"$HOME/DS OCR/envs/deepseek-ocr/bin/pip"` (for installing packages)
  - Note: Quotes required due to space in "DS OCR"
- **Required packages for 8-bit:** `accelerate`, `bitsandbytes` (must be installed in above environment)
- **Partitions:** `teaching` (T4, 16GB), `dgx` (V100, 32GB), `dgxh100` (H100, 80GB)
- **Phase 2a uses `dgx` with 8-bit quantization** — reduces 32GB model to ~8GB per GPU
- **Storage:** Use `/scratch/gloriosog/` for data (no approval needed, large quota)
- **Job scripts:** Use `.sh` files with `sbatch` command
- **Max walltime:** 24 hours; use checkpoint + job chaining for longer runs
- **Output files:** `slurm-{jobid}.out` and `slurm-{jobid}.err`

### Model Loading Patterns
- **Memory constraints:** DeepSeek-Coder-V2-Lite (16B params) = ~32GB in bfloat16
- **Solution for Phase 2a:** Use 8-bit quantization (`load_in_8bit=True`, `device_map="auto"`)
  - Requires: `accelerate` and `bitsandbytes` packages
  - Reduces VRAM: 32GB → ~8GB (frozen coder model only)
  - Adapter trained in full precision (bfloat16)
- **Pro tip:** Use `AutoConfig.from_pretrained()` to get model dimensions without loading weights (instant, <1MB memory)

### File Structure
```
DS Coder/
├── claude.md                          # This file
├── DEEPSEEK_CODER_VL_PLAN.md         # Technical plan
├── WORKSPACE_NOTES.md                 # Progress tracking
├── inspect_*.py                       # Embedding inspection scripts
└── *.sh                               # SLURM job scripts
```

---

## Common Tasks & Patterns

### Creating a New Test Script
1. Create Python script (e.g., `test_xyz.py`)
2. Create corresponding SLURM script (e.g., `test_xyz.sh`)
3. Use direct Python path: `"$HOME/DS OCR/envs/deepseek-ocr/bin/python"`
4. Specify GPU resources: `#SBATCH --gres=gpu:1`
5. Update `WORKSPACE_NOTES.md` with execution details

### Debugging Memory Issues
- Check VRAM usage: `nvidia-smi` (if interactive session)
- Load models sequentially, clear GPU cache between loads
- Use smaller model variants for testing (e.g., Tiny instead of Lite)

### Updating Documentation
- **WORKSPACE_NOTES.md:** Add execution details, results, blockers, next steps
- **DEEPSEEK_CODER_VL_PLAN.md:** Update phase status, add findings to relevant sections
- **claude.md:** Update "Quick Status" and "Next Steps" when phase changes

---

## Key Technical Details

### Vision Encoder
- **Model:** SigLIP-SO400M from DeepSeek-VL2
- **Output dimension:** 1280D
- **Token cap:** 1,120 tokens per image (256 base + max 6×144 patches)
- **Status:** Frozen (use pre-trained weights)

### Coder Model
- **Model:** DeepSeek-Coder-V2-Lite-Instruct (16B params, 2.4B active MoE)
- **Embedding dimension:** 2048D (confirmed via config inspection)
- **Config type:** DeepseekV2Config
- **Status:** Will be LoRA fine-tuned (frozen weights, trainable adapters)

### Projection Adapter ✅ Designed
- **Input:** 1280D (vision encoder output)
- **Output:** 2048D (coder model input)
- **Architecture:** 2-layer MLP with GELU activation
  ```python
  nn.Sequential(
      nn.Linear(1280, 4096),  # Layer 1: 5.2M params
      nn.GELU(),
      nn.Linear(4096, 2048),  # Layer 2: 8.4M params
  )
  ```
- **Total size:** 13.6M parameters
- **Status:** Ready to implement

---

## When Helping

### What I Need
- **Code that follows existing patterns** (SLURM scripts, Python path conventions)
- **Updates to WORKSPACE_NOTES.md** when making progress or encountering issues
- **Clear explanations** of technical decisions and trade-offs

### What to Avoid
- Breaking existing conventions (e.g., using `conda activate` in SLURM)
- Creating files without updating relevant documentation
- Making assumptions about model dimensions (check WORKSPACE_NOTES.md first)

### Questions to Ask
- "Should I update WORKSPACE_NOTES.md with this progress?"
- "Does this follow the existing SLURM script pattern?"
- "Which document should I reference for [specific detail]?"

---

*This file helps AI assistants quickly understand the project context and conventions. For detailed information, see the referenced documents above.*

