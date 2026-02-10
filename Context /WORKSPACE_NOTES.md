# DeepSeek-Coder-VL Workspace Notes

*Last updated: 2026-02-10*

## Phase 1: Validation Complete ✅

### Execution Details
- **Test Script:** `test_phase1_compression.py`
- **Job ID:** 221532
- **Node:** dh-dgx1-1
- **GPU:** Tesla V100-SXM2-32GB
- **Date:** 2026-02-06

### Key Results

**Visual Token Compression Achieved:**
- Visual tokens are **capped at 1,120** (256 base + 6×144 patches) for large files
- This is **well below** the Phase 1 success criteria of 2,000-3,000 tokens

**Compression Ratios by File Size:**
- **Small files (<100 lines):** 0.52x-0.55x (worse than text, expected)
- **Medium files (443 lines):** 3.30x ✅
- **Large files (1,462-2,677 lines):** 10.60x-20.16x ✅✅✅

**Specific Results:**
| File | Lines | Text Tokens | Visual Tokens | Compression |
|------|-------|------------|---------------|-------------|
| json_encoder.py | 443 | 3,699 | 1,120 | 3.30x |
| pathlib.py | 1,462 | 11,873 | 1,120 | 10.60x |
| http_client.py | 1,527 | 12,871 | 1,120 | 11.49x |
| ast.py | 1,710 | 14,148 | 1,120 | 12.63x |
| argparse.py | 2,595 | 21,002 | 1,120 | 18.75x |
| typing.py | 2,677 | 22,580 | 1,120 | **20.16x** |

**Context Window Impact:**
- With 100K tokens allocated for code:
  - **Text-only:** ~5-8 large files
  - **Visual:** ~89 large files
  - **Result:** 11-18x more files in context window

### Key Finding
Vision encoder uses dynamic tiling with max 6 patches, creating a hard cap that benefits large files. Compression improves dramatically with file size because text tokens scale linearly while visual tokens are capped.

### Status
✅ **Phase 1 PASSED** - All success criteria exceeded. Ready to proceed to Phase 2.

---

## Phase 1.5: Embedding Dimension Inspection ✅ COMPLETE

### Goal
Determine the exact embedding dimensions for both models to design the projection adapter architecture.

### Execution Details

**Date:** 2026-02-09
**Status:** ✅ Both models inspected successfully

#### Vision Encoder Inspection
- **Job ID:** 221778
- **Node:** dh-dgx1-1
- **GPU:** Tesla V100-SXM2-32GB
- **Script:** `inspect_embeddings_v2.py`

**Results:**
- ✅ **Vision encoder output dimension: 1280** (matches expected)
- Number of tokens per image: 256 (base view)
- Architecture confirmed:
  - `SAM (ImageEncoderViT)` → `[1, 896, 16, 16]`
  - `Qwen2Decoder2Encoder` → `[1, 256, 896]`
  - `MlpProjector` → `[1, 256, 1280]`

#### Coder Model Inspection
- **Job ID:** 221827
- **Node:** dh-dgx1-1
- **Script:** `inspect_coder_embeddings.py` (config-only, no GPU needed)
- **Execution time:** 4 seconds

**Results:**
- ✅ **Coder model input dimension: 2048**
- Config type: `DeepseekV2Config`
- Vocab size: 102,400
- Hidden layers: 27
- Attention heads: 16

### Technical Challenges Encountered

**Issue 1: Conda Environment Activation**
- Initial script tried to use `conda activate`, which failed (conda not initialized in SLURM)
- **Solution:** Matched Phase 1 approach - directly call Python binary at `$HOME/DS OCR/envs/deepseek-ocr/bin/python`

**Issue 2: Missing Accelerate Library**
- Initial script used `device_map="auto"` which requires the `accelerate` package
- **Solution:** Removed `device_map`, manually moved models to GPU with `.cuda()` (like Phase 1)

**Issue 3: GPU Memory Exhaustion**
- Vision encoder (DeepSeek-OCR-2) used ~26 GB VRAM
- Coder model (16B parameters) needs ~30+ GB VRAM
- Total needed: ~56 GB, but V100 only has 32 GB
- **Solution:** Created separate script that only loads the coder model (we already know vision dimension is 1280)

### Scripts Created

**`inspect_embeddings_v2.py` / `inspect_embeddings_v2.sh`**
- Loads both models sequentially with GPU memory cleanup between them
- Successfully extracted vision encoder dimension (1280)
- Failed on coder model due to memory constraints

**`inspect_coder_embeddings.py` / `inspect_coder_embeddings.sh`** ✅ Success
- Config-only loading (no model weights needed)
- Completed in 4 seconds
- Successfully determined: 2048D embedding dimension

### Projection Adapter Design (Finalized)

Based on inspection results:

**Dimensions:**
- Input: 1280D (from vision encoder)
- Output: 2048D (to coder model)
- Intermediate: 4096D (2× max of input/output)

**Architecture:**
```python
import torch.nn as nn

projector = nn.Sequential(
    nn.Linear(1280, 4096),  # Layer 1
    nn.GELU(),
    nn.Linear(4096, 2048),  # Layer 2
)
```

**Parameters:**
- Layer 1: 5.2M parameters (1280 × 4096 + 4096 bias)
- Layer 2: 8.4M parameters (4096 × 2048 + 2048 bias)
- **Total: 13.6M parameters**

Extremely lightweight compared to:
- Vision encoder: ~400M parameters
- Coder model: ~16B parameters (2.4B active with MoE)

---

## Phase 2: Prototype the Bridge (Implementation in Progress)

### Goal
Build and train the projection adapter that maps VL2 vision tokens → Coder-V2 embedding space.

### Tasks
- [x] Load DeepSeek-OCR-2 vision encoder on Rosie
- [x] Inspect vision encoder embedding dimensions → **1280D confirmed**
- [x] Load DeepSeek-Coder-V2-Lite config
- [x] Inspect coder model embedding dimensions → **2048D confirmed**
- [x] Design the MLP projector architecture → **1280D → 4096D → 2048D (13.6M params)**
- [x] Generate Phase 2a MVP training data (~11K examples total; 10,119 train / 562 val / 563 test) using `Data Crawling/simple_data_gen.sh` → manifests in `Data Crawling/output/manifests`
- [x] **Implement the projection adapter module** → `coder_vl/projector.py` (13.6M params, tested ✓)
- [x] **Implement model integration with token replacement** → `coder_vl/model.py` (LLaVA-style <image> → 1120 visual tokens)
- [x] **Implement Phase 2a training script** → `coder_vl/train_projector.py` (adapter-only training, frozen vision encoder + coder)
- [x] **Create SLURM job scripts** → `coder_vl/train_phase2a.sh` (dgxh100, 1× H100)
- [x] **Create vision encoder extraction script** → `coder_vl/extract_encoder.py` (extracts SAM + Qwen2Decoder2Encoder + MlpProjector from DeepSeek-OCR-2)
- [ ] **Extract vision encoder** → Run `coder_vl/extract_encoder.sh` to save standalone vision encoder (~1.5-2 GB) to `/data/gloriosog/models/vision_encoder.pt` (in progress)
- [ ] Scale training data to 50K–100K examples on `/data` using the advanced pipeline (`ADVANCED_DATA_PIPELINE.md`)
- [ ] Train projector Phase 2a (alignment pretraining)
- [ ] Evaluate against Phase 2a gates (Section 8 in PHASE2_PLAN.md)
- [ ] Train projector Phase 2b (instruction tuning with LoRA)

### Compute Requirements
- 2× H100 GPUs
- Estimated time: ~1 day for training

---

## "Sniper" Hybrid Method Evaluation

### Proposed Approach
A three-phase hybrid workflow combining vision and text:

1. **Wide Scan (Vision, 8x compression):** Feed 100 files as compressed images
   - Model uses "Block-Level Understanding" to locate buggy function
   - Prompt: "Which file and function contains the logic error?"

2. **Narrow Focus (Text):** Once file identified, retrieve raw text

3. **The Kill (Text):** Feed raw text to DeepSeek-Coder-V2 to generate exact patch

### Evaluation Summary

**Strengths:**
- ✅ Addresses token-level errors at high compression (8x+)
- ✅ Combines best of both: vision for coverage, text for fidelity
- ✅ Avoids "lost-in-the-middle" problem in text-only agents
- ✅ Matches research findings on graceful degradation hierarchy

**Considerations:**
- ⚠️ Requires two inference passes (vision scan + text generation)
- ⚠️ Adds latency but may improve accuracy
- ⚠️ Need to validate that vision can reliably identify file/function at 8x compression

**Recommendation:**
- Adopt as **Phase 4 enhancement** after initial Coder-VL validation
- Test both approaches:
  - Direct vision-to-patch (original plan)
  - Sniper hybrid (vision scan → text patch)
- Compare accuracy vs latency trade-offs on SWE-bench

### Additional Recommendations from Research

1. **Dynamic Highlighting:**
   - Low compression (1x-4x): Use syntax highlighting (visual cues help)
   - High compression (8x+): Use B&W/simple text (highlighting becomes noise)

2. **Graceful Degradation Metrics:**
   - Don't just check if model "sees" code
   - Test if it can repair fuzzy tokens (e.g., guess `user_id` from blurry `usr_id`)
   - This validates block-level understanding even at high compression

3. **Citations:**
   - Frame work as "First open-weights implementation of CodeOCR paradigm, optimized for DeepSeek-V2 MoE architecture"
   - Explicitly cite CodeOCR (Shi et al., 2026) and LongCodeOCR (2026) papers

---

## Research Context

### Key Papers
- **CodeOCR** (Shi et al., 2026): Code-as-vision paradigm, graceful degradation metrics, token-level error analysis at high compression
- **LongCodeOCR** (2026): Repository-scale code understanding via vision, maintains global semantic coverage and dependency closure

### Novel Contributions
1. First open-weights implementation of CodeOCR paradigm for code reasoning
2. Vision encoder transplant from OCR model to code model within the same family
3. Measured vision token compression ratios for code at scale (Phase 1 results)
4. Hybrid "Sniper" workflow combining vision and text
5. Practical application to SWE-bench with 11-18x more files in context

---

## Files Reference

### Test Scripts
- `test_phase1_compression.py` - Phase 1 validation script
- `test_vision_models.py` - VLM benchmark script
- `code_to_image.py` - Code → image converter
- `DS Coder/inspect_embeddings_v2.py` - Vision + Coder dimension inspector (sequential loading)
- `DS Coder/inspect_coder_embeddings.py` - Coder-only dimension inspector ✅ Current

### Output Files
- `slurm-221532.out` - Phase 1 compression test results
- `slurm-inspect-embeddings-221777.out` - First embedding inspection attempt (conda/accelerate issues)
- `slurm-inspect-embeddings-221778.out` - Vision encoder inspection (successful, 1280D confirmed)
- `slurm-inspect-coder-221827.out` - Coder model config inspection (successful, 2048D confirmed)

### Documentation
- `DEEPSEEK_CODER_VL_PLAN.md` - Main project plan (updated with Phase 1 results)
- `PROJECT_PLAN.md` - Overall project tracking
- `ROSIE_Commands_Reference.md` - Rosie supercomputer commands

---

## Next Actions

1. **✅ COMPLETED (2026-02-10):** Phase 2 implementation
   - ✅ Created `coder_vl/projector.py` — 13.6M parameter MLP (1280D→4096D→2048D), tested and verified
   - ✅ Created `coder_vl/model.py` — LLaVA-style token integration (<image> → 1120 visual tokens)
   - ✅ Created `coder_vl/train_projector.py` — Phase 2a training script (adapter-only, frozen encoder + coder)
   - ✅ Created `coder_vl/train_phase2a.sh` — SLURM job for dgxh100 (1× H100, 24h walltime)
   - ✅ Created `coder_vl/extract_encoder.py` — Vision encoder extraction from DeepSeek-OCR-2
   - ✅ Fixed extraction script with correct attribute names (model.sam_model, model.qwen2_model, model.projector)
   - ✅ Created `/pass` and `/prime` slash commands in `.claude/commands/`

2. **✅ COMPLETED (2026-02-10):** Vision encoder extraction
   - ✅ Fixed `/data` permission issues → all paths updated to project root (`./models/`, `./checkpoints/`)
   - ✅ Extracted vision encoder: `./models/vision_encoder.pt` (0.85 GB, fp16, 454M params)
   - ✅ Optimized `/prime` and `/pass` commands (~70-80% token reduction)

3. **Immediate (Next):** Run Phase 2a training
   - Submit training job: `sbatch coder_vl/train_phase2a.sh`
   - Monitor progress: `tail -f slurm-phase2a-*.out` (~6-10 hours on 1× H100)
   - Checkpoints saved to `/data/gloriosog/checkpoints/phase2a/`
   - Evaluate against Phase 2a gates (PHASE2_PLAN.md Section 8)

4. **Medium-term:** Scale data and Phase 2b
   - Scale training data to 50K–100K examples using advanced pipeline
   - If Phase 2a gates pass → proceed to Phase 2b (adapter + LoRA)
   - Phase 2b: Instruction fine-tuning (~12–18 hours on H100)

5. **Future:** Evaluate Sniper method in Phase 4
   - Compare direct vision-to-patch vs hybrid approach
   - Measure accuracy vs latency trade-offs

---

## Notes & Observations

### Phase 1
- Phase 1 validation exceeded all expectations - compression ratios of 10-20x for large files
- Visual token cap (1,120) is much lower than anticipated, which is excellent
- Small files don't benefit from compression, but that's expected and acceptable
- The Sniper method is a promising enhancement but should be validated after core model works
- Dynamic highlighting strategy should be implemented based on compression level

### Phase 1.5 (Embedding Inspection) ✅
- Vision encoder outputs 1280-dimensional embeddings (confirmed via test inference)
- Coder model uses 2048-dimensional embeddings (confirmed via config inspection)
- DeepSeek-OCR-2 + DeepSeek-Coder-V2-Lite cannot fit together in 32GB V100 VRAM (~56GB needed)
- Solution: Inspect models separately - config-only loading for coder model (no weights needed)
- Key learning: Don't load full model weights when you only need architecture info
- The projection adapter is very lightweight: 13.6M parameters (vs 400M vision encoder, 16B coder model)
- Completed in 4 seconds - much faster than loading full models

---

*This workspace file tracks progress, findings, and next steps for the DeepSeek-Coder-VL project.*

