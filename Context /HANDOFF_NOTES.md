# Phase 2a Handoff Notes (2026-02-13)

## Immediate Status

**Phase 2a training completed but model is BROKEN - does not use visual features at all.**

### What Just Happened (Last 2 Hours)

1. ✅ Ran quick evaluation (15 examples) → all gates failed
2. 🔍 Deep debugging session to find root cause
3. ❌ Binary classification test (simplest possible task) → 0% accuracy
4. 📝 Updated all context files with findings

### Critical Discovery

**The model cannot use visual features in ANY capacity:**
- Not for complex tasks (listing functions) ❌
- Not for simple tasks (yes/no questions) ❌
- Model either repeats questions or refuses to answer
- **This is a fundamental architectural/training failure, not weak adapter**

---

## What We Know For Sure

### ✅ What's Working
1. **Vision encoder works** — features are diverse (cosine sim 0.19-0.66)
2. **Training converged** — loss decreased smoothly to 1.27
3. **Data format correct** — same format train/eval
4. **Token replacement logic correct** — verified identical between train/eval
5. **Precomputed features exist** — 2,165 feature files, all valid

### ❌ What's Broken
1. **Model ignores visual tokens completely**
2. **Generation fails even for yes/no questions**
3. **Model either repeats prompt or refuses to answer**
4. **Zero capability to decode visual information**

### 🤔 What We DON'T Know Yet
1. **Is the model attending to visual tokens?** (need attention visualization)
2. **Are embeddings properly aligned?** (need to check embedding space)
3. **Is the issue in training or architecture?** (likely both)

---

## Recommended Next Steps

### Option A: Diagnostic Test with Perfect Features (RECOMMENDED FIRST)
**Goal:** Verify token insertion mechanism works at all

Instead of using vision encoder features, use ground-truth text embeddings:
- Take the actual code text from the image
- Embed it with Coder-V2's embedding layer
- Use those as "visual" features
- If this works → problem is vision encoder / feature mapping
- If this fails → token insertion mechanism is broken

**Time:** 1-2 days | **Success probability:** 90% (for diagnosis)

### Option B: Swap Vision Encoder (If Option A works)
**Goal:** Use vision encoder designed for language alignment

- Replace DeepSeek-OCR-2 with SigLIP or CLIP
- These are pre-trained for vision-language tasks
- Features already in language-compatible space
- Keep same adapter architecture

**Time:** 3-5 days | **Success probability:** 60-75%

### Option C: Architectural Change
**Goal:** Use proven multimodal architecture

- Flamingo-style cross-attention (gated fusion)
- Or adapt Qwen2-VL's approach to Coder-V2
- Don't insert tokens, use cross-attention layers

**Time:** 1-2 weeks | **Success probability:** 70-85%

---

## Files for Next Session

### Read These First
- `Context /claude.md` — Current status (lines 1-75)
- `Context /PHASE2A_DEBUG_SUMMARY.md` — Complete debug report
- `Context /HANDOFF_NOTES.md` — This file

### Debug Outputs
- `2a_eval_quick.out` — Evaluation failure (all gates)
- `test_binary.out` — Binary test 0% accuracy
- `debug_single.out` — Single example trace

### Code to Review
- `coder_vl/train_projector.py` — Training script (check lines 143-186 for token replacement)
- `coder_vl/evaluate_phase2a.py` — Evaluation script (check lines 104-188)
- `coder_vl/projector.py` — Current adapter (2-layer MLP, 13.6M params)

---

## My Recommendation

**Do Option A first (diagnostic test), then decide:**

1. Create test where "visual features" are actually text embeddings
2. If model learns from those → vision encoder is the problem
3. Then try Option B (swap encoder)

**Do NOT spend weeks on Option A without this diagnostic.**

---

## Key Metrics to Watch

If you continue training with new approach:

1. **Binary classification accuracy** (during training)
   - Should reach >80% within first 100 steps
   - If stuck at 50% (random) → stop and debug

2. **Attention to visual tokens** (if possible to visualize)
   - Should be >0.1 average weight
   - If near zero → model ignoring vision

3. **Actual task performance** (not just loss)
   - Test on 10 examples every 50 steps
   - Manual inspection of outputs

## Questions to Answer

1. Can the model attend to inserted tokens at all?
2. Are visual embeddings in compatible space with text?
3. Should we pivot to different architecture entirely?
