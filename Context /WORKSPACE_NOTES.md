# DeepSeek-Coder-VL Workspace Notes

*Last updated: 2026-02-12*

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
- [x] **Implement model integration with token replacement** → `coder_vl/model.py` (LLaVA-style <image> → visual tokens)
- [x] **Create vision encoder extraction script** → `coder_vl/extract_encoder.py`
- [x] **Extract vision encoder** → `./models/vision_encoder.pt` (0.85 GB, fp16, 454M params)
- [x] **Simplify Phase 2a architecture** → Pre-computed features approach (2026-02-12, see notes below)
- [x] **Create pre-compute script** → `coder_vl/precompute_features.py` + `precompute_features.sh`
- [x] **Rewrite training script** → `coder_vl/train_projector.py` (simplified, no DDP/wandb/autocast)
- [x] **Update SLURM job** → `coder_vl/train_phase2a.sh` (dgx, 1× V100)
- [ ] **Run pre-compute job** → `sbatch coder_vl/precompute_features.sh` (saves [256, 1280] features per image)
- [ ] **Run Phase 2a training** → `sbatch coder_vl/train_phase2a.sh` (after pre-compute completes)
- [ ] Evaluate against Phase 2a gates (Section 8 in PHASE2_PLAN.md)
- [ ] Scale training data to 50K–100K examples using advanced pipeline
- [ ] Train projector Phase 2b (instruction tuning with LoRA)

### Compute Requirements (Updated 2026-02-12)
- **Pre-compute:** 1× V100 (dgx), ~10-30 min — one-shot job
- **Phase 2a training:** 1× V100 (dgx), ~8-12 hours — vision encoder NOT in VRAM
- **Phase 2b training:** 1× H100 (dgxh100), ~12-18 hours — LoRA needs more headroom

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

### Phase 2a Scripts (Current)
- `coder_vl/precompute_features.py` - Pre-compute vision features (one-shot) ✅ New
- `coder_vl/precompute_features.sh` - SLURM job for pre-computation ✅ New
- `coder_vl/train_projector.py` - Simplified adapter training (pre-computed features) ✅ Rewritten
- `coder_vl/train_phase2a.sh` - SLURM job for training ✅ Updated
- `coder_vl/projector.py` - Projection adapter module (unchanged)
- `coder_vl/model.py` - CoderVL model integration (not used by simplified training, kept for future)
- `coder_vl/extract_encoder.py` - Vision encoder extraction (already run)

### Earlier Scripts
- `test_phase1_compression.py` - Phase 1 validation script
- `test_vision_models.py` - VLM benchmark script
- `code_to_image.py` - Code → image converter
- `DS Coder/inspect_embeddings_v2.py` - Vision + Coder dimension inspector
- `DS Coder/inspect_coder_embeddings.py` - Coder-only dimension inspector

### Output Files
- `slurm-221532.out` - Phase 1 compression test results
- `slurm-inspect-embeddings-221778.out` - Vision encoder inspection (1280D confirmed)
- `slurm-inspect-coder-221827.out` - Coder model config inspection (2048D confirmed)
- `slurm-phase2a-222402.out` / `.err` - Phase 2a training attempt (dtype crash, see diagnosis above)

### Documentation
- `DEEPSEEK_CODER_VL_PLAN.md` - Main project plan (updated with Phase 1 results)
- `PHASE2_PLAN.md` - Phase 2 execution runbook
- `PROJECT_PLAN.md` - Overall project tracking
- `ROSIE_Commands_Reference.md` - Rosie supercomputer commands

---

## Next Actions

1. **✅ COMPLETED (2026-02-10):** Phase 2 initial implementation
   - ✅ Created `coder_vl/projector.py` — 13.6M parameter MLP (1280D→4096D→2048D), tested and verified
   - ✅ Created `coder_vl/model.py` — LLaVA-style token integration (<image> → visual tokens)
   - ✅ Created `coder_vl/extract_encoder.py` — Vision encoder extraction from DeepSeek-OCR-2
   - ✅ Extracted vision encoder: `./models/vision_encoder.pt` (0.85 GB, fp16, 454M params)

2. **✅ COMPLETED (2026-02-12):** Phase 2a architecture simplification
   - Job 222402 crashed on first batch — **not OOM, but a dtype mismatch** (see diagnosis below)
   - Root cause: stacking 8-bit quantization + bf16 autocast + gradient checkpointing caused Float vs BFloat16 conflict in MoE routing layers
   - **Decision: switch to pre-computed features approach**
     - Vision encoder runs once offline, saves `[num_tokens, 1280]` tensors per image (~6.4 GB total for 2175 images)
     - Training script only loads coder model (8-bit, ~8-10 GB) + adapter (55 MB) — no vision encoder in VRAM
     - This enables V100 (dgx) for Phase 2a instead of requiring H100
   - Removed: DDP, wandb, bf16 autocast, gradient checkpointing on frozen model
   - Added: `torch_dtype=torch.float16` for coder model (keeps non-quantized MoE params in fp16, consistent with bitsandbytes)
   - Added: `.half()` cast on adapter output before feeding to coder (prevents dtype mismatch)
   - New files: `coder_vl/precompute_features.py`, `coder_vl/precompute_features.sh`
   - Rewritten: `coder_vl/train_projector.py`, `coder_vl/train_phase2a.sh`

3. **✅ COMPLETED (2026-02-13):** Phase 2a training
   - Job 222458 completed successfully (~9.3 hours on V100)
   - Train loss: 1.40, Val loss: 1.27, Gap: 0.14 (all gates G1-G3 PASS)
   - Checkpoint: `./checkpoints/phase2a/best.pt` (step 550)

4. **❌ FAILED (2026-02-13):** Phase 2a evaluation — model not using visual features
   - Quick eval (15 examples, job 222733): All gates failed (G4=0.089, G5=0%, G6=0.136)
   - **Critical issue:** Model hallucinates instead of reading image content
     - Example: Asked to list functions → repeats same `__init__` signature 5+ times
     - Functions are hallucinated, not from actual code in image
   - **Debugging performed:**
     - ✅ Visual features ARE diverse (cosine similarity 0.19-0.66 between images)
     - ✅ Token replacement logic matches training/evaluation
     - ✅ Data format correct (`<img_start><image><img_end>`)
     - ✅ Base model (untrained) gives reasonable response ("need to see code")
     - ❌ Trained model learned text patterns but NOT visual decoding
   - **Root cause:** 2-layer MLP adapter insufficient to map OCR-2 (1280D) → Coder-V2 (2048D) representation spaces
     - Model minimizes loss by learning common text patterns from training distribution
     - Ignores visual tokens entirely (easier to hallucinate than decode)
   - **Files created for debugging:**
     - `coder_vl/debug_single_example.py` — traces single example through model
     - `coder_vl/test_no_image.py` — tests generation without visual features
     - `coder_vl/check_feature_diversity.py` — verifies features are distinct

5. **✅ COMPLETED (2026-02-14):** Test 1: Perfect Features Experiment
   - **Result:** Token insertion mechanism works correctly ✓
   - **Finding:** Projection adapter too weak to map OCR-2 → Coder space
   - **Conclusion:** Architecture is fine, problem is representation space mismatch
   - See detailed findings in "Test 1: Perfect Features Experiment" section below

6. **🔧 NEXT (2026-02-14):** Fix projection bottleneck
   - **Option 1 (RECOMMENDED):** Switch to SigLIP/CLIP vision encoder
     - Better language alignment than OCR-2 (document-focused)
     - Same 2-layer MLP might work with better source features
   - **Option 2:** Stronger adapter architecture
     - Add attention layers, deeper MLP, or cross-attention
     - More complex but keeps OCR-2 encoder
   - **Option 3:** Add contrastive loss during training
     - Force alignment between visual and text spaces

7. **Medium-term:** Scale data and Phase 2b
   - Scale training data to 50K–100K examples using advanced pipeline
   - If Phase 2a gates pass → proceed to Phase 2b (adapter + LoRA on H100)
   - Phase 2b: Instruction fine-tuning (~12–18 hours on H100)

5. **Future:** Evaluate Sniper method in Phase 4
   - Compare direct vision-to-patch vs hybrid approach
   - Measure accuracy vs latency trade-offs

---

## Notes & Observations

### Phase 2a — Job 222402 Diagnosis (2026-02-12)

**Job ID:** 222402 | **Node:** dh-dgx1-1 | **GPU:** V100-SXM2-32GB | **Partition:** dgx

**What happened:** Everything loaded successfully (vision encoder, coder model 8-bit, adapter 13.6M, datasets 10K). Training started but crashed on the **very first forward pass**.

**The error (NOT OOM):**
```
RuntimeError: Index put requires the source and destination dtypes match,
got Float for the destination and BFloat16 for the source.
```
Location: MoE routing in DeepSeek-Coder-V2-Lite (`modeling_deepseek.py` line 580):
```python
y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
```

**Root cause — three precision tricks conflicting:**
1. `load_in_8bit=True` — bitsandbytes casts weights to int8, activations to fp16/fp32
2. `torch.autocast(dtype=torch.bfloat16)` — forces ops into bf16
3. Gradient checkpointing on frozen model — recomputes forward in mixed precision

The MoE routing allocates `y` in float32, but experts under bf16 autocast return bfloat16. PyTorch refuses the assignment.

**Additional complexity problems identified:**
- Vision encoder loading from state_dict requires re-downloading full 26GB DeepSeek-OCR-2 model at training time (defeats the purpose of extraction)
- DDP/distributed code was dead weight (single-GPU run)
- WandB failed every time (no API key)
- Gradient checkpointing on frozen model had no benefit (only adapter trains)

**Solution: Pre-computed features approach**
- Separate vision encoding (one-shot) from adapter training
- Remove autocast, gradient checkpointing, DDP, wandb
- Load coder with `torch_dtype=torch.float16` (consistent precision for MoE)
- Cast adapter output to fp16 via `.half()` before coder model
- Use fixed 256 tokens per image (base view, no tiling) for Phase 2a simplicity
- V100 now viable: ~8-10 GB coder (8-bit) + ~55 MB adapter + ~3-5 GB activations = ~13-17 GB

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

### Phase 2a — Job 222453: Training Works, Eval Crashes on Missing Features (2026-02-12)

**Job ID:** 222453 | **Node:** dh-dgx1-1 | **Duration:** ~43 min before crash

**Good news:** Training ran successfully. Loss dropped from ~2.18 → ~1.81 over 199 batches (steps 1–50). 4-bit quantization + gradient checkpointing + `coder.train()` fixed prior OOM and dtype issues.

**Crash:** At step 50, the eval loop kicked in. A validation example references `convert_slow_tokenizer_monokai.png` — the decompression bomb image that failed during pre-computation (job 222445). The dataset has 2 missing features in val, 3 in train. When eval hit that example:
```
KeyError: '.../convert_slow_tokenizer_monokai.png'
```
at `PrecomputedDataset.__getitem__` line 87: `features = self._cache[ex["image"]]`.

**Fix needed:** Filter out examples with missing features during dataset initialization (in `PrecomputedDataset.__init__`), instead of crashing at lookup time. ~3-line change: after loading manifest, filter `self.examples` to only include examples whose `ex["image"]` has a corresponding `.pt` file in `features_dir`.

---

### Test 1: Perfect Features Experiment ✅ (2026-02-14)

**Goal:** Diagnose whether token insertion mechanism works when visual features are in the correct representation space.

**Approach:**
- Instead of using OCR-2 vision encoder features (1280D, document understanding space)
- Use Coder model's own text embeddings (2048D, code understanding space) as "visual" features
- If token insertion works → problem is the projection adapter
- If token insertion fails → problem is the architecture

**Implementation:**
- Created `coder_vl/test_perfect_features_quick.py` — inference-only test (no training)
- Tokenizes ground truth answer → gets embeddings → inserts as "visual" tokens
- Uses manual autoregressive generation (`.generate()` doesn't handle `inputs_embeds` well)
- Tests 5 examples in ~10 minutes on V100

**Results:** ✅ **Test PASSED** — Token insertion mechanism works correctly

Example outputs:
- **Example 2:** "What modules does this code import?"
  - Model correctly listed: itertools, math, matplotlib, numpy, scipy, sklearn.base, etc.
  - ✓ Model read the "visual" features and extracted correct information

- **Example 4:** "What are the function signatures?"
  - Model correctly listed: `def grep()`, `def walk_error()`, `def findfiles()`, etc.
  - ✓ Model used visual tokens to answer accurately

**Conclusion:**
- ✅ **Token insertion architecture is correct** — Model CAN use visual tokens when they're in the right space
- ❌ **Projection adapter is too weak** — 2-layer MLP cannot map OCR-2 space → Coder space effectively
- The problem is NOT the token replacement logic, attention masks, or integration mechanism
- The problem IS that OCR-2 features (document understanding) and Coder features (code understanding) live in fundamentally different semantic spaces, and the simple MLP cannot bridge them

**The gap:**
```
OCR-2 features:     "Text arranged in rows/columns, document layout"
                    ↓ [2-layer MLP, 13.6M params]
                    ↓ [Too weak to map semantic spaces]
Coder features:     "Python code, functions, imports, logic"
```

**Next steps (in priority order):**
1. **Better vision encoder** — Switch to SigLIP/CLIP (better language alignment than OCR-2)
2. **Stronger adapter** — Add attention layers, deeper MLP, or cross-attention mechanism
3. **Add supervision** — Contrastive loss to align visual/text spaces during training

**Files created:**
- `coder_vl/test_perfect_features_quick.py` — Quick inference test (10 min)
- `coder_vl/test_perfect_features_quick.sh` — SLURM script
- `coder_vl/test_perfect_features.py` — Full training script (if needed)
- `coder_vl/eval_perfect_features.py` — Evaluation script
- `Context/TEST1_PERFECT_FEATURES.md` — Full documentation

**Job IDs:**
- 222793 — Test 1 quick inference (successful, 3 min runtime)

---

*This workspace file tracks progress, findings, and next steps for the DeepSeek-Coder-VL project.*

