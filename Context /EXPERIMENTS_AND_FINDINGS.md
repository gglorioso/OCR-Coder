# DeepSeek-Coder-VL: Experiments, Findings, and Design Decisions

## 1. Project overview

- **Goal:** Build an open-weights **code-as-vision** system (“DeepSeek-Coder-VL”) that uses a frozen vision encoder (DeepSeek-OCR-2 / VL2) plus a small **projection adapter** to feed compressed visual tokens into a strong **code LLM** (DeepSeek-Coder-V2-Lite), enabling repository-scale context for SWE-bench–style tasks.
- **Key hypothesis:** Vision encoders can compress large code files into **≈10–20×** fewer tokens than text, making it possible to “see” 50–100 files at once. The main bottleneck is **aligning visual features to the code LLM’s embedding space**, not token budget.
- **Phases so far:**
  - **Phase 1:** Validate vision encoder compression on code.
  - **Phase 1.5:** Inspect embedding dimensions to design adapter.
  - **Phase 2a:** Train MLP adapter with frozen vision + frozen coder.
  - **Phase 2b:** Add QLoRA on coder to adapt to visual stream.
  - **Phase 2 diagnostics:** Test whether failures are due to encoder, adapter, data, or objective.
  - **Phase 3 (now):** Add **contrastive/retrieval objective** for Sniper-style localization.

---

## 2. Experiments and tests (chronological)

### 2.1 Phase 1 – Vision encoder compression test

- **Scripts / files**
  - `test_phase1_compression.py`
  - Notes in `WORKSPACE_NOTES.md` and `DEEPSEEK_CODER_VL_PLAN.md`
- **What was tested**
  - DeepSeek-VL2 / DeepSeek-OCR-2 **vision encoder** on real Python files (443–2,677 lines).
  - Compare **text tokens vs visual tokens**.
- **Key results**
  - Visual tokens **capped at 1,120** (256 base + up to 6×144 tiles).
  - Compression:
    - Medium files (~443 lines): **3.3×** compression.
    - Large files (~1.5–2.7K lines): **10.6–20.2×** compression.
  - With 100K code tokens:
    - Text-only: roughly **5–8 large files**.
    - Vision: roughly **89 large files** → **11–18× more files** in context.
- **Conclusion**
  - Vision encoder is **extremely efficient** on large code files; dynamic tiling creates a hard cap on visual tokens.
- **Design impact**
  - Justifies the **Sniper** idea (vision for repository-wide coverage, text for precision).
  - Confirms that token budget is not the main bottleneck; **alignment** becomes the central problem.

---

### 2.2 Phase 1.5 – Embedding dimension inspection

- **Scripts / files**
  - `DS Coder/inspect_embeddings_v2.py`
  - `DS Coder/inspect_coder_embeddings.py`
- **What was tested**
  - Extract the **output dimension of the vision encoder** and **input embedding dimension of DeepSeek-Coder-V2-Lite**.
- **Key results**
  - Vision encoder output: **1280D**.
  - Coder input embeddings: **2048D**.
- **Conclusion**
  - Projection adapter needs to map **1280 → 2048**; a simple 2-layer MLP is feasible.
- **Design impact**
  - Final adapter: `Linear(1280, 4096) → GELU → Linear(4096, 2048)` (~13.6M parameters), implemented in `coder_vl/projector.py`.

---

### 2.3 Phase 2a v1 – Initial adapter training & dtype crash

- **Scripts / files**
  - `coder_vl/train_projector.py` (early version)
  - `coder_vl/precompute_features.py` (later refactor)
  - Notes: “Phase 2a – Job 222402 Diagnosis”
- **What was tested**
  - Train adapter with:
    - **8-bit quantization**, `torch.autocast` to bfloat16,
    - gradient checkpointing, and
    - MoE coder model on a **single V100**.
- **Observed issues**
  - Crashed on first forward pass with:
    - `RuntimeError: Index put requires the source and destination dtypes match (Float vs BFloat16)` in MoE routing.
- **Conclusion**
  - Three precision tricks (8-bit, bf16 autocast, checkpointing) interacted badly with DeepSeek’s MoE implementation.
- **Design impact**
  - Switched to **precomputed visual features** and a much simpler training loop:
    - Remove autocast, gradient checkpointing, and DDP.
    - Load coder with `torch_dtype=torch.float16`.
    - Cast adapter outputs to `.half()` before feeding coder.
    - Precompute `[num_tokens, 1280]` features per image once.

---

### 2.4 Precompute features + dataset cleaning

- **Scripts / files**
  - `coder_vl/precompute_features.py`
  - `coder_vl/precompute_features.sh`
  - `precomputed_features_tiled/` (features)
- **What was tested**
  - One-shot pass over all code images to store **fp16 visual features**.
  - Handle **corrupted images** (decompression bombs) by skipping / filtering.
- **Key results**
  - Precompute succeeded for nearly all images; a few decompression bombs were detected and skipped.
- **Conclusion**
  - Precomputed features make training **lighter and more robust** on V100s.
- **Design impact**
  - Training scripts (`train_projector.py`, later `train_phase2a.sh`) now assume **precomputed vision features** instead of loading the vision encoder live.
  - Dataset loader filters out examples whose `.pt` features are missing.

---

### 2.5 Test 1 – Perfect Features Experiment

- **Scripts / files**
  - `coder_vl/test_perfect_features_quick.py`
  - `coder_vl/test_perfect_features.py`
  - `coder_vl/eval_perfect_features.py`
  - Notes: “Test 1: Perfect Features Experiment”
- **What was tested**
  - Replace vision features with **ground-truth text embeddings** from the coder itself:
    - Use `coder.embed(text_answer)` as “visual” embeddings.
    - Insert those as extra tokens via the same token-replacement mechanism.
- **Key results**
  - Model produced **correct imports and function signatures** when given perfect features.
  - Example: On “What modules does this code import?” it correctly listed all modules.
- **Conclusion**
  - **Token insertion architecture is correct.** When embeddings live in the right space, the coder can use them.
  - The problem is **NOT** the injection mechanism, but the **projection mapping** from OCR-2 space to coder space.
- **Design impact**
  - Focus shifted to **projection / alignment**, away from architecture-level worries.
  - Validated keeping the basic LLaVA-style injection design (`coder_vl/model.py`).

---

### 2.6 SigLIP vs OCR-2 alignment test

- **Scripts / files**
  - `coder_vl/siglip_test/extract_siglip.py`
  - `coder_vl/siglip_test/test_siglip_alignment.py`
  - `models/siglip_encoder.pt`
- **What was tested**
  - Compare random-adapter perplexity using:
    - OCR-2 vision features vs SigLIP features, both passed through simple adapters.
- **Key results**
  - OCR-2: lower loss / better perplexity than SigLIP on code images.
  - SigLIP, while better for general image–text, was **worse for code images** here.
- **Conclusion**
  - OCR-2 is **closer to coder embedding space** than SigLIP, despite SigLIP’s general strength.
- **Design impact**
  - **Stuck with OCR-2** as the primary encoder.
  - Treated SigLIP as an informative negative result: “better semantics ≠ easier alignment to coder.”

---

### 2.7 Linear probe tests (feature semantics)

- **Scripts / files**
  - `coder_vl/linear_probe/generate_probe_labels.py`
  - `coder_vl/linear_probe/extract_probe_features.py`
  - `coder_vl/linear_probe/train_linear_probe.py`
  - `coder_vl/test_linear_probe.py`
- **What was tested**
  - Train simple linear classifiers/regressors on **pooled visual features** to predict:
    - Binary properties (has class, has functions, etc.).
    - Regression targets (number of functions/classes, file size bucket).
- **Key results**
  - Both OCR-2 and SigLIP features achieved:
    - Binary accuracy **~+13.6% over baseline**.
    - Regression \(R^2 ≈ 0.44–0.50\).
- **Conclusion**
  - Visual features **do encode code semantics** (structure, counts, rough content).
  - The vision encoder is **not the main bottleneck**; alignment is.
- **Design impact**
  - Justified investing in **alignment objectives** (contrastive, embedding matching) rather than swapping encoders.

---

### 2.8 Diagnostic reconstruction test

- **Scripts / files**
  - `coder_vl/diagnostic_reconstruction.py`
- **What was tested**
  - Ask the trained adapter+coder to **reconstruct code text** from images and compare to ground truth with BLEU/ROUGE.
- **Key results**
  - BLEU ≈ **0.000**, ROUGE-L ≈ **0.011**.
  - Outputs degenerated into **Chinese conversational text / garbage**.
- **Conclusion**
  - With **LM loss alone**, the adapter drifts into **“easy” regions** of coder’s embedding space (Chinese conversational clusters), not code semantics.
- **Design impact**
  - Established the need for **explicit alignment supervision** (contrastive loss).
  - Highlighted that generation loss alone is **not sufficient** to align visual features.

---

### 2.9 Contrastive pretraining v1–v3

- **Scripts / files**
  - `coder_vl/contrastive_pretrain.py`
  - Result logs summarized in `WORKSPACE_NOTES.md`.
- **What was tested**
  - **v1 (InfoNCE, small dataset):**
    - ~2,165 unique images, batch=64, temp=0.07.
    - Loss improved but **val cosine saturated**; model memorized negative structure.
  - **v2 (InfoNCE, larger dataset):**
    - ~7.8K images; val_loss improved (4.16 → 3.25) but **val_cos only ~0.13**.
  - **v3 (SigLIP loss, no bias):**
    - SigLIP-style per-pair sigmoid with learnable temperature.
    - Achieved **very low loss** but **val_cos ≈ -0.83** → complete anti-alignment.
- **Conclusions**
  - **InfoNCE:** With small batch sizes, in-batch negatives are too few; model overfits batch structure.
  - **SigLIP without bias:** Extreme class imbalance (63 negatives, 1 positive) drives model toward anti-alignment.
- **Design impact**
  - Motivated adding a **bias term** and focusing on cosine alignment as a key signal.
  - Informed future choices for contrastive objectives in Phase 3.

---

### 2.10 Contrastive pretraining v4 – SigLIP + bias (success)

- **Scripts / files**
  - `coder_vl/contrastive_pretrain.py` (final config)
  - Checkpoint: `checkpoints/contrastive_v4/best.pt`
- **What was tested**
  - SigLIP-style loss **with learnable temperature and a bias parameter** initialized to -10.0.
- **Key results**
  - val_loss ≈ **0.336**, **val_cos ≈ 0.84**.
  - Train_cos also high positive; alignment is no longer collapsed.
- **Conclusion**
  - The **bias fix works**: avoids degenerate anti-alignment optimum.
  - Adapter can be trained to produce **well-aligned embeddings** relative to text.
- **Design impact**
  - This checkpoint becomes the **preferred initialization** for later adapter training.
  - Informs Phase 3 design: contrastive objective = **SigLIP + bias**, not plain InfoNCE.

---

### 2.11 Phase 2a v4–v6 – Adapter training with contrastive init

- **Scripts / files**
  - `coder_vl/train_projector.py` (overhauled)
  - `coder_vl/train_phase2a.sh`
  - `coder_vl/evaluate_phase2a.py`
  - Eval result JSONs: `eval_results_v6.json`
- **What was tested**
  - Multi-GPU DDP adapter training with:
    - Precomputed visual features.
    - Quantized coder with fp16 activations.
    - Contrastive-initialized adapter weights.
  - Various learning rates and tiling configurations.
- **Key results**
  - **v4 (lr=1e-4):** Low loss but **destroyed contrastive alignment**, produced Chinese loops; gates failed.
  - **v5 (lr=1e-5, base view 256 tokens):**
    - G6 (diversity) improved; outputs less collapsed.
    - Still weak on exact-match; 256-token compression (~88:1) loses symbol-level info.
  - **v6 (tiling, 720 tokens/image):**
    - Better structure, higher ROUGE; G4 passed.
    - G5 exact-match remained poor; G6 per-example diversity high but corpus metric low due to repeated structures.
- **Conclusions**
  - **Learning rate is critical**: lr_adapter must stay ≤1e-5 or generation loss overwrites alignment.
  - Adding tiling (720 tokens) improves **symbol capacity** but cannot fully fix exact-match with a frozen coder.
- **Design impact**
  - Calibrated final Phase 2a settings: **lr=1e-5**, ~720 tokens/image, multi-GPU DDP with distributed eval.
  - Validated using contrastive checkpoint as **initialization** but not relying on high-lr LM fine-tuning.

---

### 2.12 Phase 2b – Adapter + LoRA

- **Scripts / files**
  - `coder_vl/train_phase2b.py`
  - `coder_vl/evaluate_phase2b.py`
  - Eval JSON: `eval_results_2b.json`
- **What was tested**
  - Add **QLoRA** on specific attention-related modules of DeepSeek-Coder-V2-Lite while continuing to train the adapter.
  - Evaluate on multiple tasks (function/class/import listings, description, explanation).
- **Key results**
  - **G4 ROUGE-L:** ~0.31 **PASS**.
  - **G5 exact-match:** 0% **FAIL**.
  - **G6 Distinct-1:** 0.20 **FAIL** (but per-example analysis shows high diversity; corpus metric is misleading).
  - Task-level ROUGE shows strong structure (class/import/function listings) but weak free-form description/explanation.
- **Conclusions**
  - LoRA **improves structural fidelity** but still fails at symbol-perfect answers.
  - Model largely uses visual path for **structure/localization**, not fine-grained identifier decoding.
- **Design impact**
  - Motivated refocusing the research question toward **localization and retrieval**, not symbol-perfect generation.
  - Directly led toward **Sniper** framing and retrieval-first metrics.

---

### 2.13 Semantic evaluation & retrieval-style metrics

- **Scripts / files**
  - `coder_vl/eval_semantic.py`
  - `eval_semantic_results.json`
  - `coder_vl/test_image_sensitivity.py`
- **What was tested**
  - **Image sensitivity:** compare outputs with correct vs swapped images.
  - **Semantic retrieval:** retrieval-style tasks using DistilBERT embeddings and Recall@k.
- **Key results**
  - Image sensitivity: mean ROUGE-L correct vs swapped ≈ 0.335 → model does use images (not ignoring them).
  - Retrieval Recall@5: low overall (~2–3%), somewhat better for explanation tasks.
- **Conclusions**
  - Visual stream **does influence** outputs (images matter).
  - However, current training and metrics are optimized for LM-style generation, not retrieval.
- **Design impact**
  - Prompted the move to **InfoNCE-style contrastive loss** and **Recall@k** as primary metrics for Sniper localization (Phase 3).
  - Led to `coder_vl/eval_retrieval.py` for Retrieval Recall@k evaluation.

---

### 2.14 Data_gen_2b and extended training data

- **Scripts / files**
  - `code_to_image.py` (enhanced to render chunks).
  - `Data Crawling/data_gen_2b.py` / `.sh`
  - `coder_vl/precompute_2b.sh`
  - Manifests in `data_v2b/manifests/`
- **What was done**
  - Build **v2b dataset** (~45K examples) from ~6.9K Python files, chunked into 500-line images, six tasks per chunk (listings, descriptions, explanations).
  - Precompute tiled features (`precomputed_features_tiled/`) for all images.
- **Impact**
  - Provides the **larger, more diverse** dataset needed for:
    - Phase 2b QLoRA training.
    - Phase 3 contrastive + retrieval training on `description`/`function_explanation` tasks.

---

### 2.15 Phase 3 plan – Contrastive + generation (Sniper localization)

- **Scripts / files**
  - `coder_vl/train_phase2b.py` (to be modified with contrastive term).
  - `coder_vl/eval_retrieval.py`
  - Plan doc: `Context/PHASE3_CONTRASTIVE_PLAN.md`.
- **What will be tested**
  - Joint objective:
    - \( L_{\text{total}} = L_{\text{gen}} + 0.1 \cdot L_{\text{SigLIP+bias}}( \text{mean\_pool}(adapter(features)), \text{mean\_pool}(embed(answer)) ) \)
  - Only **description** and **function_explanation** examples feed the contrastive term.
- **Intended measurements**
  - **val_pos_cos** for contrastive pairs.
  - **Retrieval Recall@k** (overall and per task type).
- **Importance**
  - Directly trains the **localization pathway** needed in the Sniper method, decoupled from exact symbol reconstruction.

---

## 3. Design decisions and architecture changes (by theme)

### 3.1 Vision encoder & representation

- **Chose DeepSeek-OCR-2/VL2** over SigLIP after:
  - Compression tests (Phase 1) → excellent token efficiency.
  - Alignment tests → OCR-2 closer to coder’s space despite SigLIP’s general strength.
- **Kept vision encoder frozen**; all adaptation happens in a small adapter + LoRA.

### 3.2 Projection adapter

- **Architecture:** 2-layer MLP, 1280→4096→2048 (~13.6M params).
- **Status:**
  - Initially trained with **LM loss only** → collapsed to conversational cluster.
  - Later combined with **contrastive pretraining** (SigLIP+bias) → well-aligned embeddings.
- **Open direction:** Possible Q-Former / cross-attention adapter as an ablation, but not yet implemented.

### 3.3 Objectives

- **Generation-only (Phase 2a/2b):**
  - Good for structure-level alignment; insufficient for retrieval or symbol-perfect answers.
- **Contrastive (Phase 2 diagnostics, Phase 3 plan):**
  - **SigLIP + bias** chosen over InfoNCE due to small-batch behavior.
  - Text side stays **frozen** (no gradients through embeddings) to avoid corrupting coder space.
- **Final planned training:** Combine **LM loss** and **contrastive loss** with a small weight (0.1), only on semantic tasks.

### 3.4 Data & tasks

- Early tasks: heavy on **function/class/import listings** and **signatures**.
- Diagnostics showed:
  - Listings are good for **structure**, but misleading for embedding-level semantics.
  - **Descriptions and explanations** are better anchors for retrieval-style learning.
- Design shift:
  - Use **all tasks** for generation loss.
  - Use **only description/explanation** tasks for contrastive alignment and retrieval evaluation.

### 3.5 Metrics

- Early metrics:
  - **ROUGE-L** and **exact-match** on symbol lists.
  - **Distinct-1** to detect degenerate loops.
- Findings:
  - ROUGE can be high even when model hallucinates symbols.
  - Corpus-level Distinct-1 penalizes structurally repeated but healthy outputs.
- New metrics:
  - **val_pos_cos** (visual vs text embeddings).
  - **Retrieval Recall@k** on description/explanation tasks (via `eval_retrieval.py`).
- For Sniper:
  - Success = “do we retrieve the right file/function?” not “do we regenerate the exact list of functions?”

### 3.6 Compute / engineering choices

- **Precomputed features** to avoid loading vision encoder during training.
- **Multi-GPU DDP** with distributed eval to avoid NCCL timeouts (fixed val-loader).
- **Learning-rate constraint:** lr_adapter ≤1e-5 to preserve contrastive alignment.
- **No gradients through text embeddings** in contrastive objective.

---

## 4. Current status and open questions

- **Status now**
  - Robust **code-as-vision pipeline** is implemented end-to-end: rendering → precompute → adapter training → LoRA → generation and semantic evaluation.
  - Multiple diagnostics converge on the same story:
    - Vision encoder is good enough.
    - Adapter architecture is adequate in principle (per perfect-features test).
    - **Alignment + objective choice** are the actual bottlenecks.
- **Open questions (perfect for MICS framing)**
  1. How far can **contrastive + retrieval training** push localization quality without retraining the vision encoder?
  2. What is the right balance between **structural tasks** (listings) and **semantic tasks** (descriptions) for training a retrieval-friendly embedding space?
  3. Does a simple MLP adapter remain competitive vs a **Q-Former-style** adapter under the same training regime?

