# DeepSeek-Coder-VL: A Vision-Enabled Code Reasoning Model

## Building a Multimodal Code Model via Vision Encoder Transplant

---

## 1. The Problem

Solving real-world GitHub issues (SWE-bench) requires an LLM to read **many files** from a codebase to understand context. But context windows are limited:

| Approach | Tokens per 100-line file | Files in 128K context | Effective context |
|----------|------------------------:|----------------------:|:-----------------:|
| Raw text | ~1,000 tokens | 5-10 files | ❌ Too few |
| Code images (Claude/Gemini) | ~7,000 tokens | 1-2 files | ❌ Even worse |
| Code images (DeepSeek VL encoder) | ~300-800 tokens | **50-100+ files** | ✅ Target |

The key insight: **DeepSeek's vision encoder compresses code images into far fewer tokens than both raw text AND other VLMs.** We measured this on Rosie:

| File | Lines | Text Tokens | DeepSeek Visual Tokens | Compression |
|------|------:|------------:|-----------------------:|:-----------:|
| fibonacci.py | 22 | ~111 | 256 (base only) | 0.4x ❌ |
| quicksort.py | 51 | ~355 | 832 (256 + 576) | 0.4x ❌ |
| api_handler.py | 113 | ~979 | 544 (256 + 288) | **1.8x** ✅ |

The compression improves with larger files because text tokens scale linearly but visual tokens scale sub-linearly. For real SWE-bench files (500-2000 lines), we project **5-10x+ compression**.

**The problem:** DeepSeek-OCR-2 has the efficient vision encoder, but it's an OCR specialist — it can't reason about code. DeepSeek-Coder-V2 is an excellent code reasoner, but it's text-only — it can't see images.

**The solution:** Combine them.

---

## 2. The Idea: Vision Encoder Transplant

Take the **eyes** from DeepSeek-VL2 (efficient vision encoder) and give them to **the brain** of DeepSeek-Coder-V2 (expert code reasoner). The result is a new model — **DeepSeek-Coder-VL** — that can:

1. **See** code as images (using VL2's efficient vision encoder)
2. **Understand** the code deeply (using Coder-V2's code reasoning)
3. **Generate** correct patches (using Coder-V2's code generation)
4. All while using **far fewer context tokens** than text

This is not a new technique. It's exactly how every modern VLM was built:

| Model | Vision Encoder | Language Model | Connection |
|-------|---------------|----------------|:----------:|
| LLaVA | CLIP ViT-L | LLaMA-2 | MLP projector |
| DeepSeek-VL2 | SigLIP-SO400M | DeepSeek-MoE | MLP projector |
| Qwen-VL | ViT-bigG | Qwen-2 | Cross-attention |
| **Ours (proposed)** | **SigLIP (from VL2)** | **DeepSeek-Coder-V2** | **Trained projector + LoRA** |

The difference: nobody has done this specifically optimized for **code understanding from images**.

---

## 3. Architecture

### 3.1 Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    DeepSeek-Coder-VL                             │
│                                                                  │
│   ┌──────────────┐    ┌──────────────┐    ┌───────────────────┐ │
│   │   SigLIP      │    │  Projection  │    │  DeepSeek-Coder   │ │
│   │   Vision      │───▶│  Adapter     │───▶│  V2 (MoE)         │ │
│   │   Encoder     │    │  (trainable) │    │                   │ │
│   │              │    │              │    │  Multi-head Latent │ │
│   │  FROM: VL2   │    │  NEW: train  │    │  Attention + MoE   │ │
│   │  STATUS: frozen│   │  this layer  │    │                   │ │
│   └──────────────┘    └──────────────┘    │  FROM: Coder-V2   │ │
│          ▲                                  │  STATUS: LoRA     │ │
│          │                                  │  fine-tuned       │ │
│   ┌──────┴──────┐                          └─────────┬─────────┘ │
│   │  Code Image │                                    │           │
│   │  (PNG)      │                              ┌─────▼─────┐    │
│   └─────────────┘                              │  Patch /   │    │
│                                                │  Fix Code  │    │
│   ┌─────────────┐                              └───────────┘    │
│   │  Bug Report │──────────────────── (text tokens) ───────▶     │
│   │  (text)     │                                                │
│   └─────────────┘                                                │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 The Three Components

#### Component A: SigLIP Vision Encoder (from DeepSeek-VL2)

- **What:** SigLIP-SO400M vision transformer (~400M parameters)
- **Input:** Code image (PNG, syntax-highlighted)
- **Output:** Visual token embeddings — shape `[N_tiles, tokens_per_tile, 1280]`
- **Token efficiency:** Uses dynamic tiling — images are split into tiles based on aspect ratio
  - Each tile produces a fixed set of visual tokens
  - A 113-line code image → 256 base + 288 patch = **544 tokens** (vs 979 text tokens)
- **Status in our model:** **Frozen** (use pre-trained weights as-is, no changes)
- **Source:** Extract directly from `deepseek-ai/DeepSeek-VL2` HuggingFace weights

#### Component B: Projection Adapter (NEW — train from scratch)

- **What:** A 2-layer MLP (Linear → GELU → Linear) that maps visual tokens from the vision encoder's embedding space into the code model's embedding space
- **Input:** Visual tokens from SigLIP — dimension 1280
- **Output:** Tokens compatible with Coder-V2's input — dimension 2048
- **Architecture:** `Linear(1280, 4096) → GELU → Linear(4096, 2048)` — 13.6M parameters
- **Status:** **Trained from scratch** — this is the bridge between the two models
- **Design rationale:** 2-layer MLP matches LLaVA-1.5's proven approach. See `PHASE2_PLAN.md` Section 9 for ablation alternatives (3-layer, LayerNorm, smaller hidden dim)

#### Component C: DeepSeek-Coder-V2 (code reasoning backbone)

- **What:** DeepSeek-Coder-V2-Lite-Instruct (16B total, 2.4B active MoE)
- **Architecture:** Multi-head Latent Attention (MLA) + Mixture of Experts (MoE)
- **Capabilities:** 338 programming languages, 128K context, strong code reasoning
- **Status:** **LoRA fine-tuned** — small adapters trained so the model learns to process visual token inputs alongside text
- **Why Lite (16B) not Full (236B):** Fits on 1 H100 GPU, faster iteration, same architecture
- **Source:** `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`

### 3.3 Why This Can Work

Both DeepSeek-VL2 and DeepSeek-Coder-V2 are built on the **same foundational architecture** (DeepSeek-V2):

| Feature | DeepSeek-VL2 | DeepSeek-Coder-V2-Lite | Compatible? |
|---------|:------------:|:----------------------:|:-----------:|
| Base architecture | DeepSeek-V2 MoE | DeepSeek-V2 MoE | ✅ Same family |
| Attention | Multi-head Latent (MLA) | Multi-head Latent (MLA) | ✅ Same |
| Tokenizer family | DeepSeek | DeepSeek | ✅ Same |
| Training data | Vision + text | Code + text | Different focus |

Because they share the same architectural DNA, the vision encoder's outputs should be **more compatible** with Coder-V2's embedding space than if we were mixing completely different model families. The projection adapter mainly needs to handle the dimensional mapping and fine-tune the representation.

---

## 4. How It Would Be Used (Inference)

### For SWE-bench:

```
1. Receive GitHub issue: "Bug in django/db/models/query.py line 234..."

2. Identify relevant files in the repo (50-100 files)

3. Convert ALL relevant code files to syntax-highlighted images:
   code_to_image.py → file1.png, file2.png, ..., file100.png

4. Feed to DeepSeek-Coder-VL:
   - 50-100 code images as visual tokens (~500 tokens each = ~30K tokens total)
   - Bug description as text (~500 tokens)
   - Instructions as text (~500 tokens)
   - Total: ~31K tokens
   
   vs text-only approach:
   - 50-100 files as text = 50K-200K tokens (doesn't fit!)
   - So you can only include 5-10 files = often miss the relevant one

5. Model reasons about ALL files visually + generates a patch

6. Apply patch, run tests, submit
```

### Token Budget Comparison (128K context):

| Approach | Files in context | Bug context | Reasoning room | Total |
|----------|:----------------:|:-----------:|:--------------:|:-----:|
| Text-only | 10 files × 3K = 30K | 1K | 97K | 128K |
| **Coder-VL** | **80 files × 500 = 40K** | 1K | 87K | 128K |

With Coder-VL, you get **8x more files** in context while still having plenty of room for reasoning.

---

## 4.5. Evaluation: The "Sniper" Hybrid Method

### Proposed Approach

Based on research findings from CodeOCR (Shi et al., 2026) and LongCodeOCR (2026), a hybrid "Sniper" workflow has been proposed:

**Three-Phase Workflow:**
1. **Wide Scan (Vision, 8x compression)**: Feed 100 files as compressed images
   - Model uses "Block-Level Understanding" to locate buggy function
   - Prompt: "Which file and function contains the logic error?"
2. **Narrow Focus (Text)**: Once file identified (e.g., `utils.py`), retrieve raw text
3. **The Kill (Text)**: Feed raw text to DeepSeek-Coder-V2 to generate exact patch

### Evaluation

**Strengths:**
- ✅ Addresses token-level errors at high compression (8x+)
- ✅ Combines best of both: vision for coverage, text for fidelity
- ✅ Avoids "lost-in-the-middle" problem in text-only agents
- ✅ Matches research findings on graceful degradation hierarchy
- ✅ Practical: leverages vision for what it's good at (locating), text for precision

**Considerations:**
- ⚠️ Requires two inference passes (vision scan + text generation)
- ⚠️ Adds latency but may improve accuracy
- ⚠️ Need to validate that vision can reliably identify file/function at 8x compression

**Recommendation:**
- **Adopt as Phase 4 enhancement** after initial Coder-VL validation
- Test both approaches:
  - **Direct vision-to-patch** (original plan)
  - **Sniper hybrid** (vision scan → text patch)
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
   - Explicitly cite CodeOCR and LongCodeOCR papers

---

## 5. Implementation Plan

### Phase 1: Validate the Vision Encoder (Week 1) ✅ **COMPLETE**

**Goal:** Confirm that DeepSeek-VL2's vision encoder produces efficient visual tokens for large code files.

**Tasks:**
- [x] Download DeepSeek-VL2-Tiny (3B) on Rosie — smallest variant for fast testing
- [x] Extract the SigLIP vision encoder module
- [x] Test on large code files (500-2000 lines) from real Python repos
- [x] Measure visual token counts and confirm sub-linear scaling
- [x] Compare compression ratios to our earlier measurements

**Compute:** 1× V100 or H100
**Key script:** `test_phase1_compression.py`

**Success criteria:**
- 500-line file: < 2,000 visual tokens (vs ~5,000 text tokens = 2.5x compression)
- 1000-line file: < 3,000 visual tokens (vs ~10,000 text tokens = 3.3x compression)

**Results (2026-02-06):**
- ✅ **All success criteria exceeded**
- Visual tokens are **capped at 1,120** (256 base + 6×144 patches) for large files
- **Compression ratios achieved:**
  - 443-line file: **3.30x** (1,120 visual vs 3,699 text tokens)
  - 1,462-line file: **10.60x** (1,120 visual vs 11,873 text tokens)
  - 2,677-line file: **20.16x** (1,120 visual vs 22,580 text tokens)
- **Scaling pattern:** Compression improves dramatically with file size
  - Small files (<100 lines): 0.52x-0.55x (worse than text, expected)
  - Medium files (400-500 lines): 3.30x ✅
  - Large files (1500+ lines): 10.60x-20.16x ✅✅✅
- **Context window impact:** With 100K tokens for code:
  - Text-only: ~5-8 large files
  - Visual: ~89 large files (**11-18x more files in context**)
- **Key finding:** Vision encoder uses dynamic tiling with max 6 patches, creating a hard cap that benefits large files

### Phase 2: Prototype the Bridge (Week 2)

**Goal:** Build and train the projection adapter that maps VL2 vision tokens → Coder-V2 embedding space.

**Tasks:**
- [ ] Load DeepSeek-Coder-V2-Lite-Instruct (16B) on Rosie
- [ ] Inspect embedding dimensions of both models
- [ ] Design the MLP projector architecture (likely 2-layer MLP: `Linear(1280, 4096) → GELU → Linear(4096, coder_embed_dim)`)
- [ ] Generate training data:
  - Take Python files from popular GitHub repos
  - Convert to images with `code_to_image.py`
  - Create (image, question, answer) triples:
    - "What does this code do?" → description
    - "List all functions defined" → function list
    - "Find the bug in this code" → bug description
    - "Fix this code to handle edge case X" → corrected code
- [ ] Train the projector on this dataset using standard cross-entropy loss

**Compute:** 2× H100 (one for each model during development)
**Key script:** `train_projector.py`

**Training approach:**
```
Phase 2a: Alignment pre-training (projector only, both models frozen)
  - Simple image captioning on code: "describe this code"
  - ~10K examples, a few hours of training
  
Phase 2b: Instruction fine-tuning (projector + LoRA on Coder-V2)
  - Code Q&A from images: "what does this function return?"
  - Bug finding: "is there a bug?" + image of buggy code
  - ~50K examples, ~1 day of training
```

### Phase 3: Code Reasoning Fine-tuning (Week 3)

**Goal:** Fine-tune the combined model specifically for SWE-bench-style tasks.

**Tasks:**
- [ ] Create SWE-bench-specific training data:
  - Take resolved GitHub issues from SWE-bench training set
  - Convert repo files to images
  - Create (images + issue text → patch) training examples
- [ ] LoRA fine-tune DeepSeek-Coder-V2 backbone with visual inputs
- [ ] Train on tasks:
  - Given code images + bug report → generate correct patch
  - Given code images + failing test → identify and fix the issue
- [ ] Evaluate on held-out SWE-bench instances

**Compute:** 4× H100 (LoRA fine-tuning with gradient accumulation)
**Key script:** `train_coder_vl.py`

**LoRA config:**
```python
# UPDATED (2026-02-09): MoE-safe config — attention-only, conservative rank.
# Applying LoRA to MoE FFN (gate/up/down_proj) creates params per-expert,
# causing memory explosion. Start with attention-only LoRA.
# DeepSeek-Coder-V2 uses MLA, so target modules differ from standard attention.
# See PHASE2_PLAN.md Section 12 for full rationale.
lora_config = {
    "r": 16,               # Conservative rank (MoE has many layers)
    "lora_alpha": 32,      # scaling factor = 2 × r
    "target_modules": [     # MLA attention modules ONLY (verify names on model)
        "q_a_proj", "q_b_proj",           # MLA query compression/decompression
        "kv_a_proj_with_mqa", "kv_b_proj", # MLA key-value compression
        "o_proj",                           # output projection
    ],
    "lora_dropout": 0.05,
    "task_type": "CAUSAL_LM",
}
```

### Phase 4: Evaluation & Integration (Week 4)

**Goal:** Evaluate on SWE-bench and integrate with the multi-agent system.

**Tasks:**
- [ ] Run DeepSeek-Coder-VL on SWE-bench Verified (500 instances)
- [ ] Compare to baselines:
  - DeepSeek-Coder-V2-Lite text-only (no vision)
  - DeepSeek-VL2 with code images (vision but weaker code reasoning)
  - Claude/Gemini with code images (strong reasoning but expensive vision tokens)
- [ ] Integrate into swarmpo as an additional agent
- [ ] Measure end-to-end metrics: accuracy, tokens used, time per instance

**Compute:** 4-8× H100 for parallel evaluation
**Key script:** Integration with `swarmpo/run_multi_agent.py`

---

## 6. Training Data Generation

### 6.1 Sources

| Source | Size | Use |
|--------|------|-----|
| GitHub top Python repos | ~100K files | General code image understanding |
| SWE-bench train split | ~19K instances | Bug finding and fixing from images |
| CodeSearchNet | ~500K functions | Function-level understanding |
| Self-generated | Unlimited | code_to_image.py → Q&A pairs |

### 6.2 Data Pipeline

```
1. Collect Python files from GitHub/SWE-bench
2. Convert to syntax-highlighted images (code_to_image.py)
3. Generate Q&A pairs using a teacher model (Claude/GPT-4):
   - "What does this code do?"
   - "List all classes, functions, and their signatures"
   - "This code has a bug: [description]. Generate a fix."
   - "Write a test for this function"
4. Format as instruction tuning data:
   {
     "image": "path/to/code_image.png",
     "conversations": [
       {"role": "user", "content": "<image>\nWhat does this code do?"},
       {"role": "assistant", "content": "This code implements..."}
     ]
   }
```

### 6.3 Estimated Data Needs

| Training phase | Examples needed | Generation method | Time |
|---------------|----------------:|:---------------:|:----:|
| Alignment pre-training | **50K–100K** | AST parsing (automated) | ~2 hours |
| Instruction fine-tuning | ~50K | Teacher model or self-distillation | ~1 day |
| SWE-bench specialization | ~5K | From SWE-bench dataset | ~4 hours |

**Update (2026-02-09):** Alignment data scaled from 10K→50K–100K using AST-based label generation (no API cost). See `PHASE2_PLAN.md` Section 7.1 for details.

---

## 7. Compute Requirements (Rosie)

### Training (Updated 2026-02-09 — matched to Rosie `dgxh100` partition)

| Phase | GPUs | Memory | Time | Partition |
|-------|:----:|:------:|:----:|:---------:|
| Phase 1: Encoder validation | 1× V100 | 32 GB | 1 day | `dgx` ✅ Done |
| Phase 2a: Adapter pretraining | 1× H100 | ~42 GB | 2–4 hours | `dgxh100` |
| Phase 2b: Adapter + LoRA | 1× H100 | ~43 GB | 12–18 hours | `dgxh100` |
| Phase 3: SWE-bench fine-tuning | 2–4× H100 | 160–320 GB | 2–3 days | `dgxh100` |
| Phase 4: Evaluation | 1–2× H100 | 80–160 GB | 1 day | `dgxh100` |

**Note:** V100 (32 GB) cannot hold both models simultaneously. Phase 2+ requires H100.  
**See `PHASE2_PLAN.md` Section 6 for detailed memory budgets.**

### Inference (per SWE-bench instance)

| Component | VRAM | Time |
|-----------|:----:|:----:|
| SigLIP encoder | ~1 GB | <1s per image |
| Projection adapter | <1 GB | <1s |
| Coder-V2-Lite | ~30 GB | 10-60s reasoning |
| **Total** | **~32 GB** | **~1 min/instance** |

Fits on a **single H100** (80 GB) for inference, with room to spare.

### Model Weights Storage

| Component | Size on Disk |
|-----------|:------------:|
| SigLIP vision encoder | ~1.5 GB |
| Projection adapter | ~200 MB |
| DeepSeek-Coder-V2-Lite | ~30 GB |
| LoRA adapters | ~500 MB |
| **Total** | **~32 GB** |

---

## 8. Risk Assessment

### High Risk
| Risk | Impact | Mitigation |
|------|--------|------------|
| Vision encoder can't capture code details at high compression | Model can't reason about code it sees | Test with various image sizes/resolutions in Phase 1 |
| Embedding space mismatch too large | Projection adapter can't bridge the gap | Try multiple adapter architectures; fallback to cross-attention |
| Not enough training data for code-from-images | Model doesn't learn to reason about visual code | Use teacher model (Claude) to generate more training data |

### Medium Risk
| Risk | Impact | Mitigation |
|------|--------|------------|
| LoRA fine-tuning degrades Coder-V2's text reasoning | Model gets worse at text-only tasks | Use low LoRA rank; evaluate on text benchmarks during training |
| Compute budget exceeded | Can't complete training | Start with smallest model variants; use gradient accumulation |

### Low Risk
| Risk | Impact | Mitigation |
|------|--------|------------|
| Image generation pipeline bottleneck | Slow inference | Batch image generation; cache images |
| Model doesn't outperform text-only | Research finding (negative result) | Still a valid contribution; publish findings |

---

## 9. What Makes This Novel

1. **First open-weights implementation of CodeOCR paradigm for code reasoning.** While CodeOCR and LongCodeOCR established the code-as-vision approach, this is the first open-weights implementation optimized specifically for DeepSeek-V2 MoE architecture and SWE-bench tasks.

2. **Vision encoder transplant from OCR model to code model within the same family.** The architectural compatibility between DeepSeek-VL2 and DeepSeek-Coder-V2 makes this transplant uniquely feasible, enabling efficient visual token compression (256-1120 tokens vs 7,000+ for generic VLMs).

3. **Measured vision token compression ratios for code at scale.** Our Phase 1 experiments on Rosie produced the first systematic measurements showing 10-20x compression for large files (1500+ lines), validating the approach.

4. **Hybrid "Sniper" workflow combining vision and text.** Leverages vision for wide scanning (coverage) and text for precise patching (fidelity), addressing the coverage-fidelity trade-off identified in CodeOCR research.

5. **Practical application to SWE-bench.** Demonstrated 11-18x more files in context window, fundamentally changing how code agents navigate large repositories.

---

## 10. Success Metrics

| Metric | Target | How to Measure |
|--------|:------:|:--------------|
| Visual token compression | >5x for 500+ line files | `count_tokens()` on images vs text |
| Code understanding accuracy | >90% on function listing task | Compare VL output to ground truth |
| SWE-bench solve rate | >20% (Coder-VL alone) | Run on SWE-bench Verified |
| Multi-agent improvement | >5% gain over text-only swarmpo | Add Coder-VL agent to ensemble |
| Inference speed | <2 min per SWE-bench instance | Wall clock time on H100 |

---

## 11. File Structure (Proposed)

```
DS OCR/
├── DEEPSEEK_CODER_VL_PLAN.md      # This document
├── PROJECT_PLAN.md                  # Overall project tracking
│
├── code_to_image.py                 # Code → image converter (existing)
├── test_vision_models.py            # VLM benchmark script (existing)
│
├── coder_vl/                        # NEW — DeepSeek-Coder-VL system
│   ├── extract_encoder.py           # Extract SigLIP from VL2 weights
│   ├── projector.py                 # Projection adapter module
│   ├── model.py                     # Combined Coder-VL model class
│   ├── train_projector.py           # Phase 2 training script
│   ├── train_lora.py                # Phase 3 LoRA fine-tuning
│   ├── generate_training_data.py    # Data generation pipeline
│   ├── evaluate_swebench.py         # SWE-bench evaluation
│   └── configs/
│       ├── projector_train.yaml
│       ├── lora_finetune.yaml
│       └── eval.yaml
│
├── swarmpo/                         # Multi-agent system (existing)
│   └── configs/
│       └── coder_vl_local.yaml      # NEW — config for Coder-VL agent
│
├── examples/                        # Test files (existing)
├── code_images/                     # Generated images (existing)
└── envs/                            # Conda environments
    └── deepseek-ocr/                # Existing env (extend for training)
```

---

## 12. References & Prior Art

- **LLaVA** (Liu et al., 2023) — Visual instruction tuning: connected CLIP to LLaMA via MLP projector
- **DeepSeek-VL2** (DeepSeek, 2024) — SigLIP + DeepSeek-MoE with dynamic tiling for efficient visual tokens
- **DeepSeek-Coder-V2** (DeepSeek, 2024) — 236B MoE code model, 338 languages, 128K context
- **SWE-bench** (Jimenez et al., 2024) — Real-world software engineering benchmark, 2294 instances
- **SWE-agent** (Yang et al., 2024) — Agent framework for solving SWE-bench problems
- **mini-swe-agent** — Lightweight SWE-bench solver achieving 74%+ on Verified
- **CodeOCR** (Shi et al., 2026) — Code-as-vision paradigm, graceful degradation metrics, token-level error analysis at high compression
- **LongCodeOCR** (2026) — Repository-scale code understanding via vision, maintains global semantic coverage and dependency closure

---

*Created: 2026-02-07*
*Status: Phase 1 Complete ✅ — Proceeding to Phase 2*
*Authors: Grant Glorioso, MSOE Rosie Supercomputer Challenge 2026*

