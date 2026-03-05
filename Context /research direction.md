## Perceptual Map, Not a Checklist

To wrap your head around this research, you should stop viewing it as a checklist of tasks and instead treat it as a **Perceptual Map** of how a Vision-Language Model (VLM) "sees" code under compression.  
The likely cause of the previous failed "transplant" is that a **high-IQ brain (Coder-V2)** was fed a **low-fidelity signal**.  
These experiments are designed to find the **Minimum Viable Vision (MVV)** — the exact point where the visual signal becomes too degraded for the model to use.

## 🎯 Overall Research Goal

**Goal:** Identify the **Structural–Semantic Trade-off**:  
Quantify the specific **token budgets** and **resolution thresholds** at which a VLM can still perceive the **Topology** (layout, indentation, logical flow) of code even after it has lost the ability to read the **Semantics** (characters, keywords).

In other words, we want to map where **structure survives but text dies**.

---

## 🧪 Phase 1: The "Resolution Tiers" Test

**Core Idea:** Code intelligence is not a single scalar; it is a **hierarchy of perceptual tiers**.  
We are looking for the **"knee"** (failure point) for each tier to demonstrate that **Structure is more resilient than Text**.

### 1.1 Source-File Classification (Top-1, Top-5) ✅ COMPLETE

- **Intelligence Tier:** Macro-Level (**Global Taxonomy**).
- **Hypothesis:** This is the **most resilient** tier.
  The model can still distinguish, for example, **Python vs. Java** based on:
  - Block shapes
  - Density patterns
  - Overall layout
  even under extreme compression (e.g., **128 tokens**).
- **Test:**
  - Train a **614-class linear probe** on the hidden states of the **vision encoder**.
  - Task: **Predict the programming language** of the source file from the image tokens.

#### Results (2026-03-01)

- **Per-file probe:** Null result — cosine similarity = 0.94 (probe collapses to mean)
- **Repo probe:** 76.3% Top-1 @ 729 tokens → 28.1% @ 121 tokens; **knee at 256→121 tokens**
- **Exp2 (adaptive max-pool 8×8):** Beats mean-pool below 256 tokens (36% @ 121 vs. 28.1%); mean-pool wins at high resolution; crossover ~256 tokens
- **Key finding:** Macro-level taxonomy is resilient above 256 tokens; drops sharply below. Spatial pooling outperforms mean-pool in the low-budget regime.
- **Key files:** `Phase_1_1/exp1_meanpool_probe/results/repo_probe_results.json`, `Phase_1_1/exp2_maxpool_comparison/results/maxpool_repo_results.json`, `maxpool_comparison.png`

### 1.2 Structural Regression (Line / Function / Class Counts) ✅ COMPLETE

- **Intelligence Tier:** Meso-Level (**Spatial Geometry**).
- **Hypothesis:** The coefficient of determination \(R^2\) will remain **high (≥ 0.8)** even after character recognition fails.
  The model uses:
  - Horizontal line breaks
  - Vertical whitespace gaps
  as **structural anchors**, independent of semantics.
- **Test:**
  - Train a **regression probe** on image tokens to predict:
    - Total **line count**
    - Number of **functions**
    - Number of **classes**

#### Results (2026-03-01)

Two experiments run at 256 tokens — Exp1 (mean-pool, LinearRegression) and Exp2 (pool4x4/pool8x8, PCA 1024 + Ridge α=100):

| Target | Exp1 (mean-pool) R² | Exp2 (pool4x4) R² | Threshold | Status |
|---|---|---|---|---|
| line\_count | 0.855 | 0.867 | ≥ 0.8 | ✅ PASS |
| n\_defs | 0.364 | 0.461 | ≥ 0.8 | ❌ FAIL |
| n\_classes | 0.568 | 0.675 | ≥ 0.8 | ❌ FAIL (partial) |

- **Key finding:** Coarse line density survives 256-token compression regardless of pooling strategy. Fine-grained function and class boundaries do **not** survive — mean pooling is the architectural bottleneck, and spatial pooling (pool4x4) adds ~+0.10 R² but is insufficient to pass the gate.
- **Key files:** `Phase_1_2/exp1_structural_regression/results/regression_results.json`, `Phase_1_2/exp2_spatial_regression/results/regression_results.json`

### 1.3 Nonlinear Encoding Probe (n_defs @ 256 Tokens) 🔲 NEXT

- **Intelligence Tier:** Meso-Level (**Latent Structural Encoding**).
- **Motivation:** Phase 1.2 showed that linear probes (Ridge regression) fail to extract function count (n_defs R²=0.461) from 256-token pool4x4 features. This does not prove the information is absent — only that it is not **linearly** decodable.

- **Critical Methodological Insight (revised design):** The Phase 1.2 protocol — train at budget_729, test at budget_256 — introduces a **domain shift confound**. The k-NN collapse (R²=−5.5 when evaluated cross-budget) reveals that the pool4x4 feature manifold changes qualitatively across token budgets. An MLP probe trained at 729 tokens and tested at 256 cannot distinguish between "information absent" and "probe trained on wrong distribution." An MLP probe is also redundant with the encoder's own ~26 internal transformer MLP blocks, which have already applied far more powerful nonlinear transforms to the features.

- **Hypothesis:** If a probe trained and tested natively on budget_256 features (no cross-budget transfer) still fails to predict n_defs, the visual signal for function boundaries is **genuinely destroyed** at that resolution. If a Random Forest succeeds where Ridge fails, the information exists but requires nonlinear partitioning.

- **Test (two-mode design):**
  - **Mode A — Resolution-as-Test** (Ridge only, same as Exp2): provides degradation curve for context
  - **Mode B — Native 5-fold CV at 256 tokens** (new): eliminates domain shift entirely
    - **Ridge** (PCA 1024 + α=100): linear baseline, trained natively on 256-token features
    - **Random Forest** (300 trees, raw 18K-dim features, no PCA): axis-aligned splits orthogonal to the encoder's smooth internal nonlinearities; natural fit for discrete count targets
  - Targets: **n_defs** and **n_classes** (both failures from 1.2; footprint-size gap hypothesis)

- **Interpretation:**
  - Both probes fail in Mode B (R² ≈ 0.46) → signal is genuinely absent; higher token budgets required
  - RF passes, Ridge fails → nonlinear signal present; adapter design should include nonlinear heads
  - n_classes gap persists after domain shift removed → visual footprint hypothesis confirmed (class bodies are larger than def markers)

---

### 1.4 Syntactic Texture Probes (The Novel Contribution) ✅ COMPLETE

- **Intelligence Tier:** Micro-Level (**Syntactic Topology**).
- **High-Level Claim:** This is the **novel** part of the study.

**Phase 1.4 — Syntactic Texture Probes** (2026-03-05, session 8)

*Motivation:* Having established that SigLIP features encode coarse line density (Phase 1.2) but lack fine-grained function boundary signal (Phase 1.3), Phase 1.4 tests whether three syntactic surface properties — nesting depth, indentation style, and keyword density — are linearly decodable from 256-token mean-pooled SigLIP features. These targets were chosen to span a visual hierarchy:
- **Nesting depth** tests the "staircase" profile — the large-scale indentation gradient visible even at low resolution
- **Indentation style (tabs vs spaces)** tests the "pixel floor" — high-frequency sub-character spacing differences
- **Keyword density** serves as a semantic baseline — can the model count tokens it cannot fully resolve?

*Scripts:*
- `MVV/Phase_1_4/scripts/gen_phase_1_4_labels.py` — extracts nesting_depth (0/1/2 bins), is_tabs (binary), keyword_density (int count) from 40-line Python windows via `ast` + `tokenize`
- `MVV/Phase_1_4/scripts/run_probe_1_4.py` — RidgeClassifier / Ridge regression, native 5-fold CV on 1280-dim SigLIP features
- `MVV/Phase_1_4/scripts/visualize_resolution_floor.py` — 2×2 grid comparing Bicubic / Nearest Neighbor / Area downsampling at 224×224, blown back up with NN to reveal pixel blocks; used to explain the semantic probe failure in the paper

*Results (smoke test, 14,129 samples, 1280-dim mean-pooled features):*
- **nesting_depth:** accuracy = 0.538 ± 0.013, macro-F1 = 0.405 — above 33% chance baseline, but weak. Shallow code (class 0) nearly invisible to the model (F1 = 0.027); medium and deep nesting distinguishable (F1 ≈ 0.58 / 0.61). The staircase gradient is partially visible.
- **is_tabs:** accuracy = 0.9999 — **degenerate**. Only 1 tab-indented file in 14,129 samples. Dataset is 99.99% spaces; probe cannot be interpreted. Noted as data limitation.
- **keyword_density:** R² = +0.120 ± 0.006 — weak positive signal. SigLIP retains a coarse linear fingerprint of keyword density, but 88% of variance is lost. Consistent with Phase 1.3's finding that fine-grained semantic content does not survive 256-token compression.

*Visual fidelity insight:* At 224×224 px with 40 lines per image, mean character height = **5.6 px**. Individual glyphs are ~2–3 px wide — below reliable character recognition threshold. This explains keyword_density R² ≈ 0.12: the model encodes density texture (bright/dark horizontal banding) but cannot resolve letter identity.

*Interpretation:* SigLIP perceives syntactic structure primarily through coarse visual geometry (indentation gradient → nesting depth), not through glyph-level reading. The pixel floor for tab/space detection is untestable on this dataset. The semantic baseline (keyword density) confirms that token-level semantics are largely lost at 256-token budget — supporting the hypothesis that visual tokens compress layout information at the expense of lexical content.

*Next:* Run full label generation on 62,166-sample train set and execute probes at production scale.

---

## ⚔️ Phase 2: The "Compression Valve" Showdown 🔲 NOT STARTED

**Core Idea:** Compare compression strategies to find which method is **structurally smarter** at low token budgets.  
We are searching for the **Efficiency Frontier** in the **structure vs. budget** trade-off.

### 2.1 The Degradation Sweep (128 → 1120 Tokens)

- **Hypothesis:**
  - **Tiling + Pooling** will **outperform Downsampling at every token budget**.
  - **Downsampling** will hit a **sharp cliff** below approximately **432 tokens** because:
    - Characters fall below the **5-pixel height** threshold.
    - They effectively disappear due to **aliasing**, destroying semantic information.
  - **Tiling + Pooling** will degrade **gracefully** because:
    - It preserves the **aspect ratio** of the code.
    - It uses **pooled features** (best activations) instead of naïvely averaging pixels.

- **Test:**
  - Run **all Phase 1 probes** across **five matched token budgets** (e.g., 128, 256, 432, 720, 1120) for:
    - **Downsampling**
    - **Tiling**
    - **Tiling + Pooling**
  - Plot each probe’s performance across budgets on a **single graph** to:
    - Visualize the **degradation curves**.
    - Identify the **knee points** for each method and tier.

---

## 📏 Phase 3: The "Scale & Failure" Diagnostic 🔲 NOT STARTED

**Core Idea:** Demonstrate that current VLMs fail on large code files because they treat code like a **photo** instead of a **document**.  
The hypothesis is that naive resizing **blinds** the model at scale.

### 3.1 File-Size Stratification

- **Hypothesis:**
  - Large files (\(> 500\) lines) become **disproportionately unreadable** under downsampling.  
  - The **knee** of the degradation curve will occur:
    - **Earlier** for large files than for small files.
    - With a larger **performance gap** between Tiling-based methods and Downsampling.

- **Test:**
  - Split the dataset into **three file-size strata**:
    - **Small** files
    - **Medium** files
    - **Large** files
  - For each stratum, compute the **Degradation Curve** for all compression methods.  
  - Measure how the **performance gap between Tiling and Downsampling** **widens** as file size increases.

### 3.2 The "Hallucination" Floor (OCR-Fidelity Test)

- **Hypothesis:**
  - The **failed transplant** happens because the **language model "brain"** expects characters at around **20 px height**, but actually receives **3 px blurs**.  
  - At that point, the model **hallucinates** because its visual input is **sub-symbolic noise** while its priors assume clean text.

- **Test:**
  - Measure **Exact Match (EM)** for **keywords** at different image resolutions.
  - Identify the **Character Floor**:
    - The exact **pixel height** at which **Coder-V2 stops recognizing tokens** reliably.
  - Align this floor with:
    - Breakpoints in **keyword-density regression**.
    - Failure points in **downsampling vs. tiling** curves.

---

## Summary of the Contribution

- **MVV & Structural–Semantic Trade-off:** You are not just checking whether a VLM can "read" code, but **where structure outlives semantics** under compression.
- **Multi-Tier Probing:** By probing **macro**, **meso**, and **micro** tiers, you build a **perceptual map** of what the vision encoder truly attends to.
- **Methodological Showdown:** By contrasting **Downsampling** with **Tiling (+ Pooling)**, you aim to find the **most structure-preserving compression path** at low token budgets.
- **Scale & Failure Analysis:** By stratifying by file size and measuring the **Character Floor**, you diagnose **why large real-world codebases break current VLMs**.
