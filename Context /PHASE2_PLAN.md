# Phase 2 Plan - DeepSeek-Coder-VL Adapter

*Last updated: 2026-02-12*
*Owner: Grant Glorioso*
*Purpose: Single source of truth for Phase 2 execution decisions, constraints, experiments, and cross-instance comparisons.*

---

## 1) Objective

Build and train the projection adapter that maps frozen vision tokens to DeepSeek-Coder-V2 embedding space:

- Vision encoder output: `1280D` (frozen SigLIP-SO400M)
- Coder model input: `2048D` (frozen in Phase 2a, LoRA in Phase 2b)
- Adapter: `1280 → 4096 → 2048` (~13.6M params)

Success target:
- Enable robust code understanding from images with large-context compression advantages for SWE-bench workflows.

---

## 2) Rosie HPC Infrastructure (Confirmed)

Source: `ROSIE_Commands_Reference.md` + Phase 1 job outputs  
Docs: https://docs.hpc.msoe.edu/#/  

### Available Partitions

| Partition | GPU Type | VRAM/GPU | GPUs/Node | Nodes | CPUs/Node | Notes |
|-----------|----------|----------|-----------|-------|-----------|-------|
| `teaching` | Tesla T4 | 16 GB | 4 | — | 72 | Default partition. Too small for our models. |
| `dgx` | V100-SXM2 | 32 GB | 8 | 3 | — | DGX-1 nodes. Used in Phase 1 (`dh-dgx1-1`). |
| `dgxh100` | H100 | 80 GB | 8 | 2 | — | Newest. Use sparingly. **Required for Phase 2.** |

### Constraints

- Max walltime (all partitions): **24 hours** (jobs killed after this)
- Default walltime if `--time` omitted: **1 hour**
- Max CPUs per GPU: **16**
- Containers: Singularity available
- Python: use `$HOME/DS OCR/envs/deepseek-ocr/bin/python` (not `conda activate` in SLURM)
- Storage: `/data` and `/scratch` for large artifacts; avoid home quota pressure

### Partition Selection for Phase 2

| Phase | Minimum GPU | Partition | Request |
|-------|-------------|-----------|---------|
| Phase 2a pre-compute | 1× V100 (32 GB) | `dgx` | `--partition=dgx --gpus=1 --cpus-per-gpu=8` |
| Phase 2a training | 1× V100 (32 GB) | `dgx` | `--partition=dgx --gpus=1 --cpus-per-gpu=8` |
| Phase 2b (adapter + LoRA) | 1× H100 (80 GB) | `dgxh100` | `--partition=dgxh100 --gpus=1 --cpus-per-gpu=16` |

**V100 (dgx) IS viable for Phase 2a** (updated 2026-02-12) — with the pre-computed features approach, the vision encoder is NOT loaded during training. Only the 8-bit coder model (~8-10 GB) + adapter (~55 MB) + activations (~3-5 GB) are needed, totaling ~13-17 GB on a 32 GB V100.

**V100 is NOT viable for Phase 2b** — LoRA adds optimizer states and more activation memory. H100 required.

**T4 (teaching) is NOT viable** — 16 GB is insufficient for the 8-bit coder model plus training activations.

### Pre-Run Checklist (Verify Before Each New Run Window)

- [ ] `dgxh100` partition still available and not under maintenance
- [ ] Walltime/QoS policies unchanged (currently 24h max)
- [ ] GPU type confirmation: H100 80GB (not MIG-partitioned)
- [ ] Storage quotas on `/data` or `/scratch` sufficient for checkpoints (~5 GB per checkpoint)
- [ ] Model weights pre-cached on `/data` (compute nodes may lack internet access)

---

## 3) Token Integration Strategy

This section defines how visual tokens from the adapter are fed into the DeepSeek-Coder-V2 language model. The approach follows the **LLaVA/DeepSeek-VL2 placeholder-replacement pattern**, which is the proven standard for MLP-projection VLMs.

### 3.1 Special Tokens

Add to DeepSeek-Coder-V2's tokenizer:

| Token | ID | Purpose |
|-------|-----|---------|
| `<image>` | `vocab_size + 0` | Placeholder (replaced by projected visual tokens during forward pass) |
| `<img_start>` | `vocab_size + 1` | Marks beginning of visual token region |
| `<img_end>` | `vocab_size + 2` | Marks end of visual token region |

After adding tokens, resize the coder model's embedding table:
```python
tokenizer.add_special_tokens({
    "additional_special_tokens": ["<image>", "<img_start>", "<img_end>"]
})
model.resize_token_embeddings(len(tokenizer))
```

### 3.2 Input Format

Text input template (single image):
```
<img_start><image><img_end>
{user_instruction}
```

During tokenization, `<image>` is a single token. During the forward pass, it is **expanded** to N visual tokens (up to 1,120) after projection.

### 3.3 Forward Pass (Embedding Replacement)

```
1. Tokenize text input → input_ids (contains <image> placeholder)
2. Pass input_ids through coder model embedding layer → text_embeddings [seq_len, 2048]
3. Encode image through SigLIP vision encoder → visual_features [N_vis, 1280]
4. Project through adapter MLP → projected_visual [N_vis, 2048]
5. Find <image> token position in input_ids
6. Replace: splice projected_visual tokens into text_embeddings at that position
   Result: combined_embeddings [seq_len - 1 + N_vis, 2048]
7. Build new attention mask (all 1s, causal) and position_ids (sequential 0..new_len)
8. Forward combined_embeddings through coder model transformer layers
9. Compute next-token prediction loss on ASSISTANT TOKENS ONLY
```

### 3.4 Loss Function

- Standard **causal language modeling cross-entropy loss**
- Loss is **masked** — only computed on assistant response tokens
- Image tokens, `<img_start>`, `<img_end>`, and user prompt tokens are **excluded** from loss
- No loss on visual tokens (they are inputs, not prediction targets)

### 3.5 Position Encoding

- Use **sequential position IDs** for the entire combined sequence: `[0, 1, 2, ..., total_len-1]`
- The coder model's RoPE (Rotary Position Embedding) handles this naturally
- Visual tokens receive positions just like any other token — the model learns their meaning through training
- **No special position encoding** is needed for visual tokens with the MLA (Multi-head Latent Attention) architecture

### 3.6 Compatibility Note

DeepSeek-Coder-V2 uses MLA (Multi-head Latent Attention), which compresses KV pairs into a latent space. Visual tokens enter at the embedding level (before MLA) and flow through the same latent projection as text tokens. This is architecturally transparent — the MLA layer doesn't need modification.

---

## 4) Training Strategy

### Phase 2a — Alignment Pretraining (adapter only)

- **Trainable:** Adapter MLP only (13.6M params)
- **Frozen:** Vision encoder + coder model (both completely frozen)
- **Dataset:** 10K high-quality code-image captioning examples
- **Goal:** Teach the adapter to produce embeddings that the coder model can interpret as meaningful code descriptions

### Phase 2b — Instruction Fine-tuning (adapter + LoRA)

- **Trainable:** Adapter MLP + LoRA on coder model attention layers
- **Frozen:** Vision encoder + coder model base weights
- **Dataset:** 50K instruction-following examples (mixed tasks)
- **Include:** 20–30% text-only replay to reduce forgetting of text code skills
- **Goal:** Enable instruction-following on code images (localization, reasoning, patching)

### Transition Rule

Complete Phase 2a and validate against quantitative gates (Section 8) before starting Phase 2b. **Do not start Phase 2b until all Phase 2a gates pass.**

---

## 5) Training Hyperparameters

Based on the LLaVA-1.5 training recipe (Liu et al., 2023), adapted for our architecture and hardware.

### Phase 2a Config (Updated 2026-02-12 — Pre-computed Features Approach)

```yaml
# Phase 2a: Alignment Pretraining (adapter only)
# Uses pre-computed vision features — no vision encoder in VRAM during training
trainable_params: adapter_only  # 13.6M params
frozen: coder_model             # 8-bit quantized, fp16 non-quantized params
vision_features: pre-computed   # [256, 1280] per image, loaded from disk

optimizer: AdamW
learning_rate: 1e-3          # Aggressive LR is safe — only adapter weights update
lr_schedule: cosine
warmup_ratio: 0.03
weight_decay: 0.0            # LLaVA uses 0 for stage 1

batch_size_per_gpu: 4        # Per-GPU batch size (V100 memory budget)
gradient_accumulation_steps: 4  # Effective batch size = 16
epochs: 1                    # Single pass over 10K examples → ~632 steps

precision: fp16              # Coder loaded with torch_dtype=float16 + 8-bit quant
gradient_checkpointing: false # NOT USED — frozen model, no benefit; caused dtype conflicts
autocast: false              # NOT USED — 8-bit handles precision internally
max_seq_length: 2048         # 256 visual + ~1800 text tokens

eval_steps: 50               # Evaluate every 50 optimizer steps
log_steps: 10                # Print loss every 10 optimizer steps
```

**Key simplifications vs previous attempt (job 222402):**
- No `torch.autocast` — was causing Float vs BFloat16 mismatch in MoE routing
- No gradient checkpointing on frozen model — no benefit, contributed to dtype conflicts
- No DDP/distributed code — single GPU only
- No wandb — print-based logging only
- No vision encoder in VRAM — features pre-computed to disk
- Adapter output cast to fp16 via `.half()` before entering coder model

**Estimated time:** ~8–12 hours on 1× V100 (10K examples; fits in 24h walltime)

### Phase 2b Config

```yaml
# Phase 2b: Instruction Fine-tuning (adapter + LoRA)
trainable_params: [adapter, lora]
frozen: [vision_encoder, coder_model_base_weights]

# LoRA Configuration
lora_r: 16                   # Start conservative (not 64 — MoE has many layers)
lora_alpha: 32               # alpha = 2 × r
lora_dropout: 0.05
lora_target_modules:         # Attention only (NOT MoE FFN layers)
  - "q_a_proj"               # MLA query compression
  - "q_b_proj"               # MLA query decompression
  - "kv_a_proj_with_mqa"     # MLA key-value compression
  - "kv_b_proj"              # MLA key-value decompression
  - "o_proj"                 # Output projection

optimizer: AdamW
learning_rate: 2e-5          # Much lower than 2a — LoRA touches LLM weights
lr_schedule: cosine
warmup_ratio: 0.03
weight_decay: 0.0

batch_size_per_gpu: 4        # Smaller due to LoRA memory overhead
gradient_accumulation_steps: 8  # Effective batch size = 32
epochs: 1                    # Single pass over 50K examples → ~1,562 steps

precision: bf16
gradient_checkpointing: true
max_seq_length: 2048

checkpoint_interval_minutes: 30
eval_steps: 100
text_only_replay_ratio: 0.25 # 25% of batches are text-only (no images)
```

**Estimated time:** ~12–18 hours on 1× H100 (fits within 24h walltime)

### Why These Specific Values

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Phase 2a LR = 1e-3 | High | Only 13.6M adapter params train; LLaVA uses 1e-3 for stage 1 |
| Phase 2b LR = 2e-5 | Low | LoRA modifies LLM behavior; LLaVA uses 2e-5 for stage 2 |
| LoRA r = 16 | Conservative | MoE models have many attention layers; start small, scale up if needed |
| LoRA targets = attention only | Safe | Applying LoRA to MoE FFN (gate/up/down) would create params per expert × num_experts — memory explosion |
| Effective batch = 32 | Standard | Matches LLaVA; large enough for stable gradients on 10K–50K datasets |
| bf16 | Required | H100 native; halves memory vs fp32; no accuracy loss for training |
| Gradient checkpointing | Required | Reduces activation memory by ~40%; negligible speed cost on H100 |

---

## 6) Memory Budget

### Vision Encoder Extraction (Prerequisite)

**Critical:** DeepSeek-OCR-2 is a full VLM (~6.4 GB on disk). Phase 1 loaded the entire model, using ~26 GB VRAM in fp32. For Phase 2, we **extract only the vision encoder pipeline** and discard the language decoder:

```
DeepSeek-OCR-2 full model (~6.4 GB on disk, ~26 GB VRAM in fp32):
├── SAM (ImageEncoderViT)         ← EXTRACT (image → [1, 896, 16, 16])
├── Qwen2Decoder2Encoder          ← EXTRACT (→ [1, 256, 896])
├── MlpProjector                  ← EXTRACT (→ [1, 256, 1280])
└── Language decoder (Qwen2)      ← DISCARD
```

Extracted vision encoder in bf16: **~1.5–2 GB** (SAM ~0.8 GB + Qwen2Decoder2Encoder ~0.5 GB + MlpProjector ~0.1 GB).

**Implementation:** `coder_vl/extract_encoder.py` — loads full DeepSeek-OCR-2, saves only vision components as a standalone `torch.nn.Module`, verifies output shape matches Phase 1 results `[batch, 256, 1280]` per tile.

### Phase 2a: Adapter Only — Pre-computed Features (1× V100, 32 GB)

*Updated 2026-02-12: Switched to pre-computed features approach. Vision encoder is NOT loaded during training.*

| Component | VRAM | Notes |
|-----------|------|-------|
| Coder model (8-bit quantized, frozen) | ~8-10 GB | 16B params in int8 + overhead, `device_map="auto"` |
| Adapter weights (fp32) | ~0.055 GB | 13.6M params × 4 bytes (full precision for training stability) |
| Adapter optimizer states (AdamW) | ~0.22 GB | 2× fp32 copies (momentum + variance) |
| Adapter gradients | ~0.055 GB | Same size as weights |
| Activations (batch=4, seq≈2304) | ~3-5 GB | 27 layers, no grad checkpointing; grads flow through frozen coder for adapter backprop |
| Pre-computed features (CPU cache) | 0 GB VRAM | ~1.4 GB CPU RAM for 2175 images; batch loaded to GPU on demand |
| PyTorch overhead / fragmentation | ~2-3 GB | CUDA allocator overhead |
| **Total** | **~13-18 GB** | **Fits in 32 GB V100 with ~14-19 GB headroom** |

**Why V100 works now:** The vision encoder (~0.85 GB) is never loaded during training. Pre-computed features are cached in CPU memory and only a batch at a time moves to GPU. The 8-bit coder model is the dominant VRAM consumer.

### Phase 2b: Adapter + LoRA (1× H100, 80 GB)

| Component | VRAM (bf16) | Notes |
|-----------|-------------|-------|
| Vision encoder (extracted, frozen) | ~2 GB | Same as 2a |
| Coder model (base frozen + LoRA) | ~30.5 GB | Base weights + LoRA ~0.5 GB |
| LoRA optimizer states | ~2 GB | AdamW on ~100M LoRA params |
| LoRA gradients | ~0.2 GB | |
| Adapter (weights + optim + grads) | ~0.16 GB | Same as 2a |
| Activations (batch=4, seq=2048, grad ckpt) | ~6 GB | Smaller batch than 2a |
| PyTorch overhead / fragmentation | ~3 GB | |
| **Total** | **~44 GB** | **Fits in 80 GB with 36 GB headroom** |

### V100 Feasibility: VIABLE FOR PHASE 2a (Updated 2026-02-12)

With pre-computed features, the vision encoder is not loaded during training:

| Component | VRAM |
|-----------|------|
| Coder model (8-bit) | ~8-10 GB |
| Adapter + optimizer + gradients | ~0.33 GB |
| Activations | ~3-5 GB |
| Overhead | ~2-3 GB |
| **Total** | **~13-18 GB** |
| V100 capacity | 32 GB |
| **Remaining headroom** | **~14-19 GB ✅** |

**Decision (updated): Use `dgx` (V100) for Phase 2a. Reserve `dgxh100` (H100) for Phase 2b (LoRA).**

*Previous assessment (2026-02-09) that V100 was not viable assumed both vision encoder + coder model loaded simultaneously. The pre-computed features approach eliminates this constraint.*

---

## 7) Data Strategy

### 7.1 Phase 2a Data (Alignment — 50K–100K examples)

**Purpose:** Teach the adapter to produce embeddings the coder model recognizes as meaningful code representations. Larger alignment datasets produce more robust projectors (LLaVA used 558K).

**Label generation: AST-based (fully automated, no API needed)**

Python's `ast` module extracts ground-truth labels directly from source code — no teacher model required. Each source file produces multiple training examples:

| Task template | Ground truth source | Example output |
|---------------|-------------------|----------------|
| "List all functions defined in this code." | `ast.FunctionDef` nodes | `["parse_args", "validate_input", "run"]` |
| "List all classes defined in this code." | `ast.ClassDef` nodes | `["RequestHandler", "Response"]` |
| "What modules does this code import?" | `ast.Import` / `ast.ImportFrom` | `["os", "sys", "json", "typing.Optional"]` |
| "What are the function signatures?" | `ast.FunctionDef` + `ast.arguments` | `"def parse_args(argv: list[str]) -> Namespace"` |
| "Describe what this code does." | Module/class/function docstrings | First docstring found in file |

**Yield:** ~10K source files × 5 tasks each = **50K examples** (free, instant generation).
Scale to 100K by using 20K source files, or adding more task templates (decorators, type annotations, return types).

**Format:**

```json
{
  "image": "path/to/code_image.png",
  "conversations": [
    {"role": "user", "content": "<img_start><image><img_end>\nList all functions defined in this code."},
    {"role": "assistant", "content": "This file defines the following functions:\n1. parse_args(argv: list[str]) -> Namespace\n2. validate_input(data: dict) -> bool\n3. run(config: Config) -> None"}
  ]
}
```

**Source files:** Top 50–100 Python repositories by GitHub stars (Flask, Django, FastAPI, requests, click, httpx, scikit-learn, pandas, etc.) + Python standard library

**File size distribution:**

| Size range | Lines | % of data | Compression | Rationale |
|-----------|-------|-----------|-------------|-----------|
| Small | 50–100 | 10% | 0.5x (worse) | Baseline; model should handle these too |
| Medium | 100–500 | 50% | 1.5–3.3x | Best fidelity for alignment |
| Large | 500–1500 | 30% | 3.3–10x | Core use case; trains high-compression understanding |
| Very large | 1500–2500 | 10% | 10–20x | Stress test; validates the thesis |

**Estimated cost:** $0 (AST parsing is free)

### 7.2 Phase 2b Data (Instruction Tuning — 50K examples)

**Task mix (with rationale):**

| Task | % | Count | Why |
|------|---|-------|-----|
| Localization/extraction | 30% | 15K | Core capability: find functions, classes, signatures |
| Reasoning/bug analysis | 30% | 15K | Higher-order: explain logic, identify bugs |
| Targeted patching | 25% | 12.5K | End goal: generate correct code edits |
| Summarization/other | 15% | 7.5K | General understanding, avoid overspecialization |

**Rationale for mix:** Localization and reasoning are prerequisites for patching. Heavy localization+reasoning (60%) builds the foundation; patching (25%) is the end-goal task; summarization (15%) provides breadth. This mirrors LLaVA's finding that diverse instruction types outperform task-specific data.

**Text-only replay (25% of Phase 2b):**
- Source: Code Alpaca, CodeSearchNet, or filtered samples from The Stack
- Purpose: Prevent catastrophic forgetting of text-based code skills
- Format: Standard instruction-following without `<image>` tokens

### 7.3 Data Quality Controls

- **Deduplication:** MinHash at file level; remove near-duplicate repos (forks)
- **Repo-level splits:** Train/val/test split by repository (not by file) to prevent leakage
- **Split ratio:** 90% train / 5% val / 5% test
- **Teacher output validation:** Spot-check 2% of generated labels manually; reject if >5% error rate
- **Max response length:** 1024 tokens (truncate verbose teacher outputs)

### 7.4 Image Rendering Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Font | Monospace (Fira Code or Cascadia Code) | Standard code rendering |
| Font size | 13pt | Readable at 150 DPI without excessive image size |
| DPI | 150 | Balance of fidelity and file size |
| Theme | Dark (e.g., VS Code Dark+) | Higher contrast for syntax tokens |
| Syntax highlighting | Yes (for all Phase 2 files) | Files are ≤500 lines, compression ≤4x — highlighting helps |
| Format | PNG | Lossless — no JPEG artifacts on code text |
| Max lines per image | 500 | Beyond this, split into multiple images |
| Line numbers | **No** | Consistent with Phase 1 validation; saves visual tokens; see §7.5 |

**For files >500 lines:** Split into chunks of 400 lines with 50-line overlap. Each chunk becomes a separate image. Multi-image examples use multiple `<image>` placeholders:
```
<img_start><image><img_end>
<img_start><image><img_end>
{instruction}
```

### 7.5 Line Numbers Decision

**Decision: No line numbers** in training images.

**Rationale:**
1. **Consistency with Phase 1:** All compression measurements were done without line numbers. Adding them changes the image dimensions and invalidates our token counts.
2. **Compression cost:** Line numbers add ~3–4 characters per line. On a 500-line file, that's ~2000 extra characters of visual noise, which can push the image into a larger tile count and reduce compression.
3. **SWE-bench consistency:** At inference time, we render raw source code files from repos. The source files don't contain line numbers — we'd have to inject them. Keeping training and inference images identical avoids a distribution mismatch.
4. **High compression regime (10-20x):** At this compression, individual characters are barely distinguishable. Line numbers would be unreadable noise that wastes visual capacity.

**Future option (Phase 4 / Sniper method):** During the "narrow focus" phase where a single file is selected for patching, line numbers could optionally be added to a *low-compression* re-render of just that file. This is a rendering-time decision, not a training-time one — the model doesn't need line-number-specific training to benefit from them as visual anchors.

---

## 8) Pass/Fail Gates (Quantitative)

### Phase 2a Gates

All must pass before proceeding to Phase 2b.

| Gate | Metric | Threshold | How to Measure |
|------|--------|-----------|----------------|
| G1: Loss convergence | Training loss | < 3.0 (from initial ~8–10) | Monitor training curve |
| G2: Val loss | Validation loss | < 3.5 | Evaluate on held-out 500 examples |
| G3: No overfit | Train–val loss gap | < 0.5 | Compare at end of training |
| G4: Code description quality | ROUGE-L on val set | > 0.25 | Compare generated vs teacher descriptions |
| G5: Function listing | Exact-match accuracy | > 30% | "List all functions" task on val set |
| G6: Not degenerate | Distinct-1 (unigram diversity) | > 0.3 | Check outputs aren't repetitive/collapsed |

**Failure protocol:** If G1–G3 fail → debug adapter initialization, check gradient flow, try different LR. If G4–G6 fail → check image rendering quality, increase data to 15K, try 3-layer adapter.

### Phase 2b Gates

| Gate | Metric | Threshold | How to Measure |
|------|--------|-----------|----------------|
| G7: Improvement over 2a | Val loss | < 2.5 (down from 2a's < 3.5) | Evaluate on same val set |
| G8: Instruction following | Task accuracy (weighted avg) | > 50% | Across all 4 task types |
| G9: Localization accuracy | Function/class identification F1 | > 60% | "Find function X" tasks |
| G10: Text preservation | HumanEval pass@1 (text-only) | Degradation < 5% from baseline | Run coder model with LoRA on HumanEval |
| G11: Patching quality | Exact-match on simple patches | > 15% | Single-line fix tasks |
| G12: Bug localization | File + function identification | > 40% | "Where is the bug?" tasks |

**Failure protocol:** If G7–G9 fail → increase LoRA rank to 32, add data, check data quality. If G10 fails → increase text replay to 35%, reduce LoRA rank. If G11–G12 fail → these are stretch goals; continue to Phase 3 if G7–G10 pass.

---

## 9) Adapter Architecture & Ablations

### Baseline Architecture (E03)

```python
import torch.nn as nn

class ProjectionAdapter(nn.Module):
    def __init__(self, vision_dim=1280, hidden_dim=4096, coder_dim=2048):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),   # 5.2M params
            nn.GELU(),
            nn.Linear(hidden_dim, coder_dim),    # 8.4M params
        )
    
    def forward(self, visual_features):
        # visual_features: [batch, n_visual_tokens, 1280]
        return self.projector(visual_features)
        # output: [batch, n_visual_tokens, 2048]
```

Total: **13.6M parameters**

### Planned Ablations

| Exp ID | Architecture | Params | Hypothesis |
|--------|-------------|--------|------------|
| E03 | 2-layer MLP (1280→4096→2048) | 13.6M | Baseline (LLaVA-1.5 design) |
| E04 | 2-layer MLP + LayerNorm | 13.6M | LN after each linear may stabilize training |
| E05 | 3-layer MLP (1280→2048→4096→2048) | 14.9M | Extra layer for richer mapping |
| E06 | 2-layer MLP, smaller hidden (1280→2048→2048) | 6.8M | Test if bottleneck is sufficient |

**Run order:** E03 first (baseline). If Phase 2a gates pass → proceed to Phase 2b with E03. Run E04–E06 only if E03 fails gates or if time permits for optimization.

### Fallback Architecture

If all MLP variants fail Phase 2a gates after debugging:
- **Option A:** Add a learned query (Q-Former style) that compresses visual tokens before projection. Reduces N_vis from 1,120 to ~64 fixed queries. More complex but proven in BLIP-2.
- **Option B:** Use cross-attention layers instead of MLP. Higher capacity but more parameters (~50M+).

---

## 10) Reliability Pattern (24h Walltime Safe)

### Checkpoint Strategy

Training scripts must support:
- Auto-resume from latest checkpoint on job restart
- Atomic checkpoint save: write to `.tmp` then `os.rename()` to final path
- Persist per checkpoint: model weights, optimizer state, scheduler state, step count, epoch, RNG states (Python, NumPy, PyTorch, CUDA)
- Checkpoint cadence: every **30 minutes** (or every 50 steps, whichever is more frequent)
- Keep last 3 checkpoints; delete older ones to manage storage

### SLURM Job Chaining

```bash
# Submit initial job
JOB1=$(sbatch --parsable train_phase2a.sh)

# Chain continuation job (starts after JOB1 ends, regardless of exit code)
JOB2=$(sbatch --parsable --dependency=afterany:$JOB1 train_phase2a.sh)
JOB3=$(sbatch --parsable --dependency=afterany:$JOB2 train_phase2a.sh)
```

- Use `afterany` (not `afterok`) so the chain continues even if a job is killed by walltime
- Training script auto-detects and loads latest checkpoint on start
- Include early-exit logic: if training is complete, script exits immediately without wasting the slot

### Estimated Job Chain Length

| Phase | Est. Training Time | 24h Jobs Needed | Chain Length |
|-------|-------------------|-----------------|-------------|
| Phase 2a | 2–4 hours | 1 | No chaining needed |
| Phase 2b | 12–18 hours | 1 | No chaining needed |
| Phase 2b (conservative) | 20–28 hours | 2 | 1 chain |

Phase 2a and 2b should each fit within a single 24h job. Chain 2 jobs for Phase 2b as insurance.

---

## 11) Experiment Matrix (Cross-Instance Comparison)

Record each model/agent recommendation and outcome here.

| ID | Date | Source | Change | Hypothesis | Result | Keep? |
|----|------|--------|--------|------------|--------|-------|
| E01 | 2026-02-09 | Cursor (GPT-5.3) | Two-stage training (adapter→adapter+LoRA) | Improves stability vs joint start | Pending | TBD |
| E02 | 2026-02-09 | Gemini (site crawl) | 24h walltime mitigation via checkpoint chaining | Removes multi-day obstruction | Pending | TBD |
| E03 | 2026-02-09 | Claude (critique) | Baseline 2-layer MLP adapter (1280→4096→2048) | LLaVA-proven architecture | Pending | TBD |
| E04 | 2026-02-09 | Claude (critique) | 2-layer MLP + LayerNorm after each linear | May stabilize early training | Pending | TBD |
| E05 | 2026-02-09 | Claude (critique) | 3-layer MLP (1280→2048→4096→2048) | Richer mapping capacity | Pending | TBD |
| E06 | 2026-02-09 | Claude (critique) | Smaller hidden dim (1280→2048→2048, 6.8M params) | Test minimum viable capacity | Pending | TBD |
| E07 | 2026-02-09 | Claude (critique) | LoRA r=16, attention-only (not r=64 all-layers) | MoE-safe LoRA; avoid per-expert param explosion | Pending | TBD |
| E08 | 2026-02-09 | Claude (critique) | LLaVA-style placeholder token replacement | Standard VLM integration, compatible with MLA | Pending | TBD |
| E09 | 2026-02-12 | Cursor (Claude) | Pre-computed features: run vision encoder offline, train adapter with features from disk | Removes vision encoder from training VRAM; enables V100; fixes dtype mismatch | Pending | TBD |
| E10 | 2026-02-12 | Cursor (Claude) | Fixed 256 tokens/image (base view, no tiling) | Simplifies batching; sufficient for Phase 2a alignment; tiling for Phase 2b | Pending | TBD |

### Recording Results

When filling in the "Result" column, include:
- Final validation loss
- Key metric scores (ROUGE-L, exact-match, etc.)
- Training time (wall clock)
- Any anomalies observed
- Link to checkpoint path on `/data`

---

## 12) LoRA Design Notes (Phase 2b)

### Why Attention-Only LoRA (Not FFN)

DeepSeek-Coder-V2-Lite is a **Mixture of Experts** model. Each transformer layer has:
- **Shared attention** (MLA): `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj`
- **Multiple expert FFN layers**: each expert has `gate_proj`, `up_proj`, `down_proj`

If LoRA is applied to FFN layers in a MoE model:
- Trainable params = `num_experts × 3 FFN layers × 2 × lora_r × hidden_dim` per transformer layer
- This can be **10–50x more params** than attention-only LoRA
- Vastly increases optimizer memory and risks overfitting on 50K examples

**Decision:** Start with attention-only LoRA (r=16). Scale to FFN LoRA only if attention-only doesn't pass Phase 2b gates.

### MLA-Specific Target Modules

Standard attention LoRA targets `q_proj, k_proj, v_proj, o_proj`. But DeepSeek-Coder-V2 uses MLA (Multi-head Latent Attention) with compressed projections:

| Standard | DeepSeek MLA Equivalent | Apply LoRA? |
|----------|-------------------------|-------------|
| `q_proj` | `q_a_proj` + `q_b_proj` | ✅ Yes (both) |
| `k_proj` + `v_proj` | `kv_a_proj_with_mqa` + `kv_b_proj` | ✅ Yes (both) |
| `o_proj` | `o_proj` | ✅ Yes |

**Verify these module names** by running:
```python
for name, param in model.named_parameters():
    if "proj" in name:
        print(name, param.shape)
```

---

## 13) Open Questions (Remaining)

These are non-blocking but should be resolved before Phase 2b:

- [ ] **Exact MLA module names:** Run the parameter inspection script on the coder model to confirm LoRA target names match `q_a_proj`, `kv_a_proj_with_mqa`, etc.
- [ ] **`dgxh100` queue contention:** How long are typical queue waits? May need to run at off-peak times.
- [ ] **Internet on compute nodes:** Can compute nodes download models, or must all artifacts be pre-staged on `/data`?
- [ ] **Per-user GPU job limits:** Is there a cap on how many `dgxh100` GPUs one user can claim?
- [ ] **DeepSpeed availability:** Is DeepSpeed installed in the environment? (Needed if Phase 2b exceeds single-GPU memory.)

---

## 14) Implementation Order

1. **Pre-work:** ✅ COMPLETE
   - [x] Implement `coder_vl/extract_encoder.py` — extract vision encoder from DeepSeek-OCR-2
   - [x] Extract vision encoder → `./models/vision_encoder.pt` (0.85 GB, fp16, 454M params)
   - [x] Implement `coder_vl/projector.py` (adapter module, 13.6M params, tested)
   - [x] Implement `coder_vl/model.py` (combined forward pass with token integration)
   - [ ] Confirm MLA module names for LoRA targeting (needed for Phase 2b, not blocking)

2. **Data generation:** ✅ MVP COMPLETE (scaling pending)
   - [x] Clone top repos, collect Python files
   - [x] Render 2,500 code images (monokai, no line numbers)
   - [x] Generate Phase 2a labels via AST parsing → 11,244 examples (10,119 train / 562 val / 563 test)
   - [ ] Scale to 50K–100K examples using advanced pipeline (for Phase 2b)
   - [ ] Generate Phase 2b instruction-following labels

3. **Phase 2a training:** IN PROGRESS (updated 2026-02-12)
   - [x] First attempt (job 222402) — identified dtype mismatch crash in MoE routing
   - [x] Simplified architecture: pre-computed features approach
   - [x] Created `coder_vl/precompute_features.py` + `.sh` (pre-compute [256, 1280] per image)
   - [x] Rewrote `coder_vl/train_projector.py` (no DDP/wandb/autocast/grad-ckpt)
   - [x] Updated `coder_vl/train_phase2a.sh` (dgx V100 instead of dgxh100 H100)
   - [ ] **Run pre-compute:** `sbatch coder_vl/precompute_features.sh`
   - [ ] **Run training:** `sbatch coder_vl/train_phase2a.sh`
   - [ ] Evaluate against Phase 2a gates (Section 8)
   - [ ] If gates fail: debug, then try E04/E05/E06

4. **Phase 2b training (only after Phase 2a gates pass):**
   - [ ] Run Phase 2b with E03 adapter + LoRA r=16 on H100
   - [ ] Evaluate against Phase 2b gates (Section 8)
   - [ ] If G10 (text preservation) fails: increase replay ratio to 35%

---

## 15) Changelog

- 2026-02-24: **Architecture validated via probes; training tasks identified as bottleneck.** Linear probe (job 225927): Top-1=4.6% on 614-class source-file classification, 28× above random — frozen SigLIP features ARE informative. Semantic eval (job 225948): Retrieval Recall@5=2.6% overall, function_explanation=9.8%, description=0.7%. Conclusion: encoder works, decoder hallucinates due to wrong training tasks. Next: (1) fix probe bug (label mismatch in top_k_accuracy_score), (2) scrape 50 repos, (3) reweight tasks toward description/explanation, (4) run data_gen_v3 with 3 styles × 50 repos.
- 2026-02-24: **Phase 2b training and full eval complete.** Training job 225376 (~10h45m, 4× V100): best val_loss=1.3114 at step 800; checkpoint `./checkpoints/phase2b/best.pt`. Added `evaluate_phase2b.py`/`evaluate_phase2b.sh`; full eval on 2018 val examples: G4 ROUGE-L 0.3079 PASS, G5 exact-match 0% FAIL, G6 Distinct-1 0.20 FAIL. ROUGE strong on class/function/import listing; weak on description/explanation. Next: paper + optional localization-style probe.
- 2026-02-23: **Phase 2b data pipeline complete. Training plan finalized.** Precompute job 225264: 9,469/9,470 images processed, all [720,1280] fp16. Plan doc LoRA target module names were WRONG for Lite model (q_lora_rank=None → uses q_proj not q_a_proj/q_b_proj). Verified from modeling_deepseek.py. Correct targets: q_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj. Hardware changed: dgxh100 → dgx (2× V100), using 4-bit QLoRA (~11.9 GB VRAM, 20 GB headroom). Est. ~13h. Text-only replay deferred (G10 is stretch goal). seq_length=260 confirmed (p99=220 from actual data). peft not installed.
- 2026-02-22: **Phase 2b data pipeline planned. Home cleaned (134→95 GB).** Deleted Medical Imaging (27 GB), duplicate OCR-2 cache (6.4 GB), old precomputed_features/ (2.8 GB), pip cache (4.3 GB). Data plan: ~10K Python files from existing 15 repos + ~25 new shallow clones → render monokai images → precompute tiled features to /home (1.8 MB/img, fits at ~118 GB projected) → 6 AST task types incl. new function_explanation → 50K image-grounded + 12.5K Code Alpaca text-only. Repo scraping delegated to new instance.
- 2026-02-22: **Phase 2a DECLARED COMPLETE. Phase 2b greenlit.** Job 225091 (rep_penalty=1.3 re-eval) complete. Final gates: G4=0.2829 PASS, G5=0.000 FAIL (deferred), G6=PASS (collapse resolved — per-example d1=0.937; corpus d1=0.21 is metric artifact from inherently low-diversity tasks at 2000+ scale). No degenerate outputs. Checkpoint: `./checkpoints/phase2a_v6/best.pt`. Next: Phase 2b on H100 (LoRA r=16, lr=2e-5, attention layers, 50K data).
- 2026-02-22: **Phase 2a v6 eval complete.** Tiled features (720 tok/img, 5×144), job 224288 val_loss=1.3739. Full eval (job 224872): G4=0.2831 PASS, G5=0.011 FAIL, G6=0.0893 FAIL. G6 failure from `description` task `"""` loops; fix: repetition_penalty=1.3 (job 225091 running). G5 confirmed unfixable at Phase 2a (frozen LLM priors). Eval script upgraded: incremental saves, resume support. G5 deferred to Phase 2b (LoRA).

- 2026-02-09: **Major revision.** Added Rosie infrastructure specs (Section 2), token integration strategy (Section 3), concrete training hyperparameters (Section 5), memory budget tables (Section 6), quantitative pass/fail gates (Section 8), adapter ablation matrix (Section 9), LoRA design rationale (Section 12), and implementation order (Section 14). Resolved all cross-document inconsistencies. Removed vague language from gates. Confirmed V100 is not viable; H100 required.
- 2026-02-09: Created Phase 2 runbook for cross-instance continuity, added Rosie policy snapshot, and standardized experiment tracking format.
- 2026-02-09: **Data & encoder revision.** Scaled Phase 2a alignment from 10K→50K–100K examples using AST-based label generation (no API needed). Added vision encoder extraction step (extract from full DeepSeek-OCR-2, discard language decoder, ~2 GB vs ~26 GB). Fixed line numbers to "No" with rationale (Section 7.5). Expanded file size distribution to include large files (500–2500 lines). Updated memory budgets and training time estimates accordingly.
 - 2026-02-10: **Phase 2a MVP data generated.** Implemented lean data pipeline (`Data Crawling/simple_data_gen.py` + `simple_data_gen.sh`) that:
   - Samples 2,500 Python files from 15 top repos
   - Renders 2,500 code images (monokai, no line numbers) in `Data Crawling/output/images/`
   - Generates 11,244 AST-based Q&A examples (10,119 train / 562 val / 563 test) with manifests in `Data Crawling/output/manifests/`
   - Serves as Phase 2a alignment dataset; advanced `/data`-backed pipeline (`ADVANCED_DATA_PIPELINE.md`) reserved for scaling to 50K–100K examples in Phase 2b.
 - 2026-02-10: **Phase 2a implementation complete.** Built full training infrastructure for adapter alignment:
   - Implemented projection adapter (`coder_vl/projector.py`) — 2-layer MLP (1280D→4096D→2048D, 13.6M params), tested and verified with dummy inputs
   - Implemented CoderVL model integration (`coder_vl/model.py`) — LLaVA-style token replacement, handles <image> placeholder → 1120 visual tokens splicing
   - Implemented Phase 2a training script (`coder_vl/train_projector.py`) — adapter-only training with frozen vision encoder + coder, AdamW optimizer (LR=1e-3), cosine schedule, gradient checkpointing, wandb logging, checkpointing every 30 min
   - Created SLURM job script (`coder_vl/train_phase2a.sh`) — targets dgxh100 partition (1× H100, 24h walltime)
   - Created vision encoder extraction script (`coder_vl/extract_encoder.py`) — extracts SAM + Qwen2Decoder2Encoder + MlpProjector from DeepSeek-OCR-2, discards language decoder, saves standalone module (~1.5-2 GB vs ~26 GB full model)
   - Fixed extraction with correct DeepSeek-OCR-2 attribute paths (model.sam_model, model.qwen2_model, model.projector)
   - Created project-specific slash commands (`.claude/commands/pass.md`, `.claude/commands/prime.md`) for session handoff and context bootstrapping
   - Ready to run: vision encoder extraction in progress, Phase 2a training script ready for H100
 - 2026-02-12: **Phase 2a simplification — pre-computed features approach.** Job 222402 crashed on first batch due to dtype mismatch (Float vs BFloat16) in MoE routing layers caused by stacking 8-bit quantization + bf16 autocast + gradient checkpointing. Root cause analysis led to major architecture simplification:
   - **New approach:** Pre-compute vision features offline ([256, 1280] per image, ~6.4 GB total), train adapter using only coder model (8-bit) + adapter — no vision encoder in VRAM during training
   - **V100 now viable for Phase 2a** — updated partition from dgxh100 to dgx; memory budget ~13-18 GB out of 32 GB
   - Removed: DDP, wandb, bf16 autocast, gradient checkpointing on frozen model
   - Added: `torch_dtype=torch.float16` for consistent MoE precision, `.half()` cast on adapter output
   - New files: `coder_vl/precompute_features.py`, `coder_vl/precompute_features.sh`
   - Rewritten: `coder_vl/train_projector.py` (flat, self-contained, ~280 lines)
   - Updated: `coder_vl/train_phase2a.sh` (dgx, single V100)
   - Run order: (1) `sbatch precompute_features.sh` then (2) `sbatch train_phase2a.sh`
 - 2026-02-13: **Phase 2a training successful + evaluation infrastructure.** Job 222458 completed in 9.3 hrs with train loss 1.40, val loss 1.27 (gates G1-G3 PASS). Built evaluation script (`coder_vl/evaluate_phase2a.py`) for gates G4/G5/G6; fixed generation with KV caching (7.5x speedup, 42s/example). Job 222725 running (ETA ~6.5hrs, may need 8hr walltime).
- 2026-02-17: **Contrastive v4 complete + Phase 2a v2 submitted.** Job 223383 (contrastive v4): val_cos=0.840, val_loss=0.3356, bias=-9.005 — bias fix resolved v3 collapse. Phase 2a overhauled: `train_projector.py` now supports 4-GPU DDP (torchrun/NCCL, barriers around eval, rank-gated I/O); `train_phase2a.sh` updated to lr=1e-4, 2 epochs, 4×V100, combined 37K-example manifest (original + data_v2). Job submitted, ~17hr estimated.
- 2026-02-19: **Phase 2a v4/v5 eval complete; root causes identified; tiling plan.** v4 (lr=1e-4): all gates failed — LM fine-tuning catastrophically forgot contrastive alignment → Chinese loops (G6=0.09). v5 (lr=1e-5, job 223917): G6 PASSES (0.31), no Chinese loops, but G4=0.079/G5=0.000 still fail. Second root cause: 256-token base view = 88:1 compression for large files — visual tokens carry domain/structure but NOT fine-grained identifiers. Fix: enable tiling (1120 tokens, 20:1 compression). Plan: re-precompute features with tiling → retrain at lr=1e-5, 2 epochs → `phase2a_v6`.
