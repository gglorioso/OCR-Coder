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

### 1.1 Source-File Classification (Top-1, Top-5)

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

### 1.2 Structural Regression (Line / Function / Class Counts)

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

### 1.3 Syntactic Texture Probes (The Novel Contribution)

- **Intelligence Tier:** Micro-Level (**Syntactic Topology**).
- **High-Level Claim:** This is the **novel** part of the study.

- **Hypothesis:**  
  - **Nesting Depth** will be **more resilient** than **Keyword Density**.  
  - The model can still perceive the **"staircase" shape** of deeply nested code (loops, conditionals) even when it cannot reliably distinguish a `for` from a `while`.

- **Tests:**
  - **Nesting Depth**
    - Task: **3-class classification** — **Shallow / Medium / Deep**.
    - Measures whether the model perceives the **vertical indentation profile**.
  - **Indentation Style**
    - Task: **Binary classification** — **Tabs vs. Spaces**.
    - Probes whether the model is sensitive to **pixel-level gaps** that differentiate indentation styles.
  - **Keyword Density**
    - Task: **Regression**.
    - Role: **Semantic Baseline** to mark where **character-level recognition fails**.  
      When this collapses but structural probes remain strong, we have evidence for **structure–semantic decoupling**.

---

## ⚔️ Phase 2: The "Compression Valve" Showdown

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

## 📏 Phase 3: The "Scale & Failure" Diagnostic

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
