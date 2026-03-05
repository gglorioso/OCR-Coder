# Claude Code Assistant Context

**Project:** DeepSeek-Coder-VL - Vision-Enabled Code Reasoning Model  
**Goal:** Build a multimodal code model by combining DeepSeek-VL2's vision encoder with DeepSeek-Coder-V2's code reasoning capabilities.

---

## Quick Status

**Current Phase:** Phase 3 — MVV Phase 1.3 complete (probe + domain shift analysis); Phase 1.4 syntactic texture next
**Last Updated:** 2026-03-04 (session 7)

- ✅ **Phase 1.3 nonlinear probe DONE** — Ridge native CV n_defs=0.672, n_classes=0.780; RF fails to beat Ridge → signal genuinely weak at 256 tokens
- ✅ **Domain shift script written** — smoke test confirms DOMAIN SHIFT (cos sim=0.220, CKA=0.47); full run pending
- ⚠️ **Contrastive track (phase2b_v7) and Qwen status still unknown** — check job logs

**Current Step:** Phase 1.4 — Syntactic Texture Probes (nesting depth, indentation style, keyword density)

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

