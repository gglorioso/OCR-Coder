## DeepSeek-Coder-VL: Vision-Enabled Code Reasoning

DeepSeek-Coder-VL is an experimental multimodal **code reasoning** project that combines:
- The **vision encoder** from DeepSeek-VL2 / DeepSeek-OCR-2 (SigLIP),
- With the **code-focused language model** DeepSeek-Coder-V2-Lite,
to let an LLM **read code as images** instead of plain text.

The core idea is simple:
- Large real-world repos (like those in SWE-bench) contain **hundreds of files**.
- Text-only LLMs can only fit **a handful of files** into their context window.
- The DeepSeek vision encoder can compress a 1500–2500 line code file into **≈1,120 visual tokens**, giving **5–20× better token efficiency** than text.
- By feeding these compressed visual tokens into a strong code LLM, we aim to:
  - See **50–100+ files at once**,
  - Still have room in context for bug reports, tests, and reasoning,
  - And ultimately generate **high-quality patches** for real GitHub issues.

At a high level, the target architecture is:

```text
Code images (from repo files)
   → SigLIP vision encoder (frozen, 1280D embeddings)
   → Projection adapter MLP (learned bridge: 1280D → 2048D)
   → DeepSeek-Coder-V2-Lite (code LLM with LoRA)
   → Bug localization, explanations, and code fixes
```

This repository contains:
- Experiments validating **visual token compression** on large Python files,
- Scripts for inspecting and matching the **embedding dimensions** of the vision encoder and code model,
- Planning documents for a projection adapter and training phases to turn this into a full **Coder-VL** system.

The long-term goal is to build a practical, open-weights agent that can tackle SWE-bench–style bugs by **seeing entire repositories at once** rather than peeking at a few files at a time.