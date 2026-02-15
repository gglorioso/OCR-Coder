# Test 1: Perfect Features Experiment

**Created:** 2026-02-14
**Purpose:** Diagnose whether the token insertion mechanism works when visual features are in the correct representation space.

---

## The Core Hypothesis

We suspect the model is failing because:
1. **Representation space mismatch:** OCR-2 visual features live in a different space than Coder text embeddings
2. **Weak adapter:** The 2-layer MLP can't bridge the gap between these spaces
3. **Model learns to ignore visual tokens:** It's easier to hallucinate than decode misaligned features

**This test isolates the issue.**

---

## What Test 1 Does

### Normal Pipeline (Current)
```
Image → Vision Encoder (OCR-2) → Adapter → Coder Model
        [1280D, OCR space]       [2048D]   [2048D, Code space]
                                    ↑
                              MISMATCH - adapter struggles
```

### Test 1 Pipeline (Perfect Features)
```
Text → Coder Embeddings → Adapter → Coder Model
       [2048D, Code space] [2048D]  [2048D, Code space]
                              ↑
                    Should be EASY - features already aligned
```

**Key difference:** Instead of using visual features from the OCR-2 encoder, we:
1. Take the actual code content (ground truth answer)
2. Tokenize it using the Coder tokenizer
3. Get its embeddings from the Coder model's embedding layer
4. Use those as "visual" features

This gives the model **perfect information in the correct representation space**.

---

## Files Created

**Quick Test (RECOMMENDED - 5-10 minutes):**

1. **`coder_vl/test_perfect_features_quick.py`**
   - Inference-only test (NO training needed)
   - Uses text embeddings as "visual" features
   - Tests 5 examples in ~10 minutes

2. **`coder_vl/test_perfect_features_quick.sh`**
   - SLURM script for quick test
   - 1x V100 GPU, 30 minutes max (usually ~10 min)

**Full Training (if you want to train an adapter on perfect features):**

3. **`coder_vl/test_perfect_features.py`**
   - Training script (4 hours)
   - Only needed if quick test passes and you want to verify training works

4. **`coder_vl/eval_perfect_features.py`**
   - Evaluation for trained model

---

## How to Run

### Quick Test (START HERE)
```bash
cd /home/ad.msoe.edu/gloriosog/CoderOCR/OCR-Coder
sbatch coder_vl/test_perfect_features_quick.sh
```

Check progress (should finish in ~10 minutes):
```bash
tail -f perfect_quick.out
squeue -u gloriosog
```

View results:
```bash
cat perfect_quick.out
```

**This is all you need** - the quick test will tell you if token insertion works.

---

## Interpreting Results

### Scenario A: Low Loss + Good Answers ✓
```
Training: val_loss = 1.0-1.5
Evaluation: Model generates correct answers
```

**Interpretation:**
✓ Token insertion mechanism **WORKS**
✓ Model CAN learn from visual tokens when they're in the right space
✗ Current adapter is **TOO WEAK** to map OCR-2 → Coder space

**Next Steps:**
- Option 1: Stronger adapter (deeper MLP, attention layers, cross-attention)
- Option 2: Better vision encoder (SigLIP/CLIP with language alignment)
- Option 3: Add intermediate supervision (contrastive loss, OCR loss)

---

### Scenario B: High Loss + Bad Answers ✗
```
Training: val_loss = 3.0+
Evaluation: Model still hallucinates or ignores visual content
```

**Interpretation:**
✗ Token insertion mechanism is **FUNDAMENTALLY BROKEN**
✗ Even with perfect features, model can't use visual tokens
✗ Architecture issue in how we replace `<image>` tokens

**Next Steps:**
- Debug token replacement logic in `prepare_inputs_with_image()`
- Check if attention masks are correct (0s might be blocking visual tokens)
- Investigate gradient flow through visual token positions
- Consider alternative architecture (cross-attention instead of token insertion)

---

### Scenario C: Low Loss but Still Bad Answers ⚠️
```
Training: val_loss = 1.0-1.5
Evaluation: Model generates plausible but incorrect answers
```

**Interpretation:**
⚠️ Model learned to minimize loss, but not to use visual features
⚠️ Possible label leak or data issue
⚠️ Training objective doesn't align with generation task

**Next Steps:**
- Check if labels are leaking into input
- Verify `expand_labels()` correctly masks visual positions as -100
- Test with more diverse evaluation examples
- Add explicit visual grounding task (e.g., "Is this code Python or JavaScript?")

---

## Technical Details

### Dataset Construction
```python
# For each example:
question = "<img_start><image><img_end>\nWhat functions are defined?"
answer = "def foo():\n    ...\ndef bar():\n    ..."

# Normal: visual_features = vision_encoder(image)
# Test 1: visual_features = embed_fn(tokenize(answer[:500]))
#         ↑ Use first 500 chars of answer as "visual content"
```

### Adapter Behavior
In Test 1, the adapter receives **text embeddings** as input (already in Coder space).

It should learn:
- **Best case:** Identity mapping (output ≈ input)
- **Acceptable:** Simple linear transform
- **Bad:** Complex transformation (suggests overfitting)

### Why This Test is Diagnostic
```
If Test 1 PASSES → Projection is the bottleneck
If Test 1 FAILS  → Architecture/training is broken
```

This definitively tells us whether to:
1. Focus on improving the projection adapter
2. Redesign the token integration architecture

---

## Expected Timeline

- **Quick Test (recommended):** ~10 minutes (5 examples, inference only)
- **Full Training (if needed):** ~3-4 hours (10,119 examples, 1 epoch)
- **Total:** You'll have your answer in 10 minutes

---

## Notes

- This test uses the same training infrastructure as Phase 2a
- Checkpoint will be saved to `./checkpoints/perfect_features/best.pt`
- Uses 4-bit quantization to fit on V100 (16GB VRAM)
- Training curves should look similar to Phase 2a (loss decreasing)

The key difference is what we feed as "visual" features — and that will tell us everything.
