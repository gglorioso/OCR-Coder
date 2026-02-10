# Claude Code Assistant Context

**Project:** DeepSeek-Coder-VL - Vision-Enabled Code Reasoning Model  
**Goal:** Build a multimodal code model by combining DeepSeek-VL2's vision encoder with DeepSeek-Coder-V2's code reasoning capabilities.

---

## Quick Status

**Current Phase:** 2a - Vision Encoder Extracted, Ready for Training
**Last Updated:** 2026-02-10 (late afternoon)

- ✅ **Phase 1 Complete:** Vision encoder compression validated (10-20x for large files)
- ✅ **Phase 1.5 Complete:** Embedding dimensions confirmed
  - Vision encoder: **1280D** (from Phase 1 testing)
  - Coder model: **2048D** (from config inspection)
  - Projection adapter: **13.6M parameters** (1280D → 4096D → 2048D)
- ✅ **Phase 2a Data (MVP) Complete:** ~11K code-image Q&A examples generated
  - 2,500 rendered code images (monokai, no line numbers)
  - 11,244 examples total → **10,119 train / 562 val / 563 test**
  - Manifests in `Data Crawling/output/manifests/{train,val,test}.jsonl`
- ✅ **Phase 2a Implementation Complete:** Full training infrastructure built
  - `coder_vl/projector.py` — Projection adapter (tested ✓)
  - `coder_vl/model.py` — Token integration (<image> → 1120 visual tokens)
  - `coder_vl/train_projector.py` — Training script (adapter-only, frozen models)
  - `coder_vl/train_phase2a.sh` — SLURM job for dgxh100
  - `coder_vl/extract_encoder.py` — Vision encoder extraction (in progress)

**Next Steps:**
1. Submit Phase 2a training: `sbatch coder_vl/train_phase2a.sh` (dgxh100, 1× H100, ~6-10 hours)
2. Monitor training: `tail -f slurm-phase2a-*.out`
   - Submit: `sbatch coder_vl/train_phase2a.sh`
   - Monitor: `tail -f slurm-phase2a-*.out`
3. Evaluate against Phase 2a gates (Section 8 in `PHASE2_PLAN.md`)
4. If gates pass → proceed to Phase 2b (adapter + LoRA)
5. If gates fail → debug, try architecture ablations (E04-E06)

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
- **Python path:** Use `$HOME/DS OCR/envs/deepseek-ocr/bin/python` (not `conda activate`)
- **Partitions:** `teaching` (T4, 16GB), `dgx` (V100, 32GB), `dgxh100` (H100, 80GB)
- **Phase 2 uses `dgxh100` exclusively** — V100 cannot hold both models simultaneously
- **Storage:** Use `/scratch/gloriosog/` for data (no approval needed, large quota)
- **Job scripts:** Use `.sh` files with `sbatch` command
- **Max walltime:** 24 hours; use checkpoint + job chaining for longer runs
- **Output files:** `slurm-{jobid}.out` and `slurm-{jobid}.err`

### Model Loading Patterns
- **Memory constraints:** Can't load both vision encoder (26GB) + coder model (30GB) in 32GB V100
- **Solution:** Load models separately, use `.cuda()` for GPU placement
- **Avoid:** `device_map="auto"` (requires `accelerate` package)
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

