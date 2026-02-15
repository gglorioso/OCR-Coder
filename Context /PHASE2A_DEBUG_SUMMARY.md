# Phase 2a Debugging Summary (2026-02-13)

## TL;DR

**Training converged successfully (loss 1.40→1.27) but model IGNORES visual features and hallucinates outputs.**

- ❌ Evaluation failed: G4=0.089, G5=0%, G6=0.136 (all gates FAIL)
- ✅ Visual features ARE diverse (not the problem)
- ❌ Model learned text patterns, not visual decoding
- 🔧 Running binary classification test to verify pipeline can work

---

## What We Discovered

### 1. The Problem (Evaluation Failure)

**Quick eval (15 examples):**
- Asked: "List all functions defined in this code"
- Expected: List of 7 actual functions (\_\_init\_\_, forward, \_get\_name, etc.)
- Got: Repeats same hallucinated `__init__` signature 5+ times

**Example hallucination:**
```python
def __init__(self, name: str, value: Any, trace: bool = True, ...)
def __init__(self, name: str, value: Any, trace: bool = True, ...)
def __init__(self, name: str, value: Any, trace: bool = True, ...)
# ... (repeats, never lists actual functions from image)
```

**Base model (untrained):** Gives reasonable response ("I would need to see the actual code...")

**Trained model:** Hallucinates, completely ignoring image content.

### 2. What We Verified (Debugging)

#### ✅ Visual Features ARE Diverse
- Checked 20 precomputed feature files
- Cosine similarities: 0.19–0.66 (GOOD range)
- Different code images → different visual features
- **Conclusion:** Vision encoder IS working correctly

#### ✅ Token Replacement Logic Correct
- Training: `replace_image_tokens()` at line 143-186
- Evaluation: Same logic in `generate_one()`
- Both splice visual features at `<image>` token position
- **Conclusion:** No mismatch between train/eval

#### ✅ Data Format Correct
- Training data: `<img_start><image><img_end>\nQuestion`
- Evaluation: Same format
- Tokenization consistent
- **Conclusion:** Format is not the issue

#### ✅ Training Converged Properly
- No dtype errors, no crashes
- Loss decreased smoothly (1.6 → 1.4 train, 1.27 val)
- No overfitting (gap only 0.14)
- **Conclusion:** Training ran correctly

### 3. Root Cause Analysis

**The model learned to minimize loss WITHOUT using visual features.**

**How?**
1. Training data has strong patterns:
   - All examples start with "List all functions..."
   - All answers start with "This file defines the following functions:\n1. ..."
2. Model learns these text patterns from the **prompt alone**
3. Model hallucinates plausible-looking function signatures
4. Loss goes down because text generation is fluent
5. Model **never needs to decode visual features** to minimize loss

**Why the adapter didn't help:**
- Projection adapter (2-layer MLP, 13.6M params) maps 1280D → 2048D
- OCR-2 features are in a representation space optimized for OCR-2's decoder
- Coder-V2 embeddings are in a different representation space
- The MLP bridge might be too weak to make these compatible
- Or the model finds it easier to ignore visual tokens than decode them

### 4. Evidence

**Feature diversity check:**
```
3D_parallel_monokai  <-> _MixedMMH100_monokai  sim=0.3481
3D_parallel_monokai  <-> __main___monokai      sim=0.5684
__main___monokai     <-> _adafactor_monokai    sim=0.6523
```
→ Features ARE distinct (if they were identical, sim would be >0.9)

**Text-only generation (untrained model):**
```
User: List all functions defined in this code.
Assistant: To list all functions..., I would need to see the actual code.
```
→ Reasonable response when no image provided

**With image (trained model):**
```
def __init__(...repeated 5 times...)
```
→ Hallucinates instead of reading image

**Conclusion:** Training DID change behavior, but taught wrong thing (patterns, not vision).

---

## Current Status

### Files Created
- `coder_vl/debug_single_example.py` — Traces one example through model step-by-step
- `coder_vl/test_no_image.py` — Tests generation without visual features (control)
- `coder_vl/check_feature_diversity.py` — Verifies features are distinct
- `coder_vl/test_binary_classification.py` — **NEW**: Simple yes/no test
- `coder_vl/test_binary.sh` — SLURM job for binary test

### Binary Test Results (Job 222738)
**CRITICAL FAILURE: 0% accuracy (0/4 questions correct)**

The model CANNOT use visual features even for trivial yes/no questions:
- Q: "Does this code contain a class definition?"
  - Expected: "Yes"
  - Got: Repeats question back ("Does this code contain a class definition? Answer with...")
- Q: "Does this code contain a function definition?"
  - Expected: "Yes"
  - Got: "I'm sorry, but I can't" (refusal)

**This is not a weak adapter — this is a fundamental breakdown.**
The model isn't even attempting to use visual information.

---

## Next Steps (Decision Tree)

### If Binary Test Passes (accuracy >70%)
**Diagnosis:** Adapter works but is too weak for complex tasks
**Solutions:**
1. **Increase adapter capacity:**
   - Try 4-layer MLP instead of 2-layer
   - Add residual connections
   - Try transformer-based adapter (cross-attention)

2. **Add attention supervision:**
   - Explicitly penalize not attending to visual tokens
   - Use contrastive loss (correct image vs wrong image)

3. **More training:**
   - Scale to 50K-100K examples
   - Train for more epochs
   - Use larger batch size

### If Binary Test Fails (accuracy <50%)
**Diagnosis:** Fundamental issue with token integration or attention
**Solutions:**
1. **Check attention mechanism:**
   - Add debug prints to see if model attends to visual tokens
   - Verify attention masks are correct
   - Check if visual tokens are being masked out

2. **Try different vision encoder:**
   - Use SigLIP (vision-language pre-trained) instead of OCR-2
   - Use CLIP features (better aligned with language)

3. **Change architecture:**
   - Try prefix-tuning instead of token replacement
   - Try Flamingo-style cross-attention gating
   - Use visual prompts instead of token insertion

### If Binary Test Is Mixed (accuracy 50-70%)
**Diagnosis:** Pipeline works but signal is weak
**Solutions:**
1. **Strengthen supervision:**
   - Add auxiliary loss (e.g., predict if function exists in image)
   - Use curriculum learning (start simple, increase difficulty)

2. **Better feature extraction:**
   - Try using different OCR-2 layers
   - Ensemble multiple vision encoders
   - Add visual data augmentation

---

## Key Learnings

1. **Low training loss ≠ model learned correctly**
   - Loss can decrease by learning patterns, not understanding
   - Need evaluation metrics that test actual capability

2. **Vision-language alignment is hard**
   - Different models have incompatible representation spaces
   - Simple projection might not be enough

3. **Dataset bias is dangerous**
   - If prompts are too similar, model learns shortcuts
   - Need diverse prompts to force visual grounding

4. **Start simple for debugging**
   - Binary classification (yes/no) easier to diagnose than generation
   - Helps isolate where the pipeline breaks

---

## References

### Output Files
- `2a_eval_quick.out` — Failed evaluation results (15 examples)
- `debug_single.out` — Single example trace-through
- `test_no_image.out` — Control test (no visual features)
- `test_binary.out` — Binary classification test (pending)

### Code Files
- `coder_vl/train_projector.py:143-186` — Token replacement logic (training)
- `coder_vl/evaluate_phase2a.py:104-188` — Token replacement logic (eval)
- `coder_vl/projector.py` — Projection adapter (13.6M params, 2-layer MLP)
- `coder_vl/precompute_features.py:130` — Feature extraction pipeline

### Context Files
- `Context /claude.md` — Updated with failure diagnosis
- `Context /WORKSPACE_NOTES.md` — Updated with debug findings
- `Context /PHASE2_PLAN.md` — Original phase 2 plan (Section 8 = failure protocol)
