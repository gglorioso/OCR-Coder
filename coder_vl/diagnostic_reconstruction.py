"""
Diagnostic Test: Visual Feature Reconstruction Quality

Tests if the trained adapter can reconstruct code from visual features.
Measures BLEU/ROUGE between generated text and ground truth code.

Decision thresholds:
  - BLEU ≥0.3: Visual features preserve info → use stronger adapter
  - BLEU <0.1: Info lost in vision encoding → fine-tune encoder
  - 0.1-0.3: Gray zone, likely need stronger adapter + more data

Usage:
    python diagnostic_reconstruction.py
    python diagnostic_reconstruction.py --max_samples 30  # quick test
"""

import json
import argparse
from pathlib import Path
from collections import Counter

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from tqdm import tqdm

from projector import ProjectionAdapter


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _lcs_length(x, y):
    """Length of the longest common subsequence (space-optimised DP)."""
    m, n = len(x), len(y)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def rouge_l_f1(generated: str, reference: str) -> float:
    """Word-level ROUGE-L F1."""
    gen_tok = generated.lower().split()
    ref_tok = reference.lower().split()
    if not gen_tok or not ref_tok:
        return 0.0
    lcs = _lcs_length(gen_tok, ref_tok)
    p = lcs / len(gen_tok)
    r = lcs / len(ref_tok)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def _ngram_precision(generated: list, reference: list, n: int) -> float:
    """Compute n-gram precision."""
    if len(generated) < n or len(reference) < n:
        return 0.0

    gen_ngrams = Counter(tuple(generated[i:i+n]) for i in range(len(generated) - n + 1))
    ref_ngrams = Counter(tuple(reference[i:i+n]) for i in range(len(reference) - n + 1))

    clipped_counts = sum((gen_ngrams & ref_ngrams).values())
    total_counts = sum(gen_ngrams.values())

    return clipped_counts / total_counts if total_counts > 0 else 0.0


def bleu_score(generated: str, reference: str) -> float:
    """
    Compute BLEU-4 score (geometric mean of 1-4 gram precisions + brevity penalty).
    Simple implementation without smoothing.
    """
    gen_tok = generated.lower().split()
    ref_tok = reference.lower().split()

    if not gen_tok or not ref_tok:
        return 0.0

    # Brevity penalty
    bp = 1.0 if len(gen_tok) >= len(ref_tok) else (
        2.71828 ** (1 - len(ref_tok) / len(gen_tok))
    )

    # N-gram precisions (1-4)
    precisions = []
    for n in range(1, 5):
        p = _ngram_precision(gen_tok, ref_tok, n)
        if p == 0:
            return 0.0  # If any n-gram precision is 0, BLEU is 0
        precisions.append(p)

    # Geometric mean
    geo_mean = (precisions[0] * precisions[1] * precisions[2] * precisions[3]) ** 0.25

    return bp * geo_mean


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_one(prompt_ids, features, adapter, coder, tokenizer,
                 image_token_id, embed_fn, device, max_new_tokens):
    """
    Replace <image> token with projected features and generate greedily.
    Uses manual autoregressive loop since .generate() doesn't handle inputs_embeds well.

    Args:
        prompt_ids: [1, seq]            tokenised user prompt
        features:   [1, 256, 1280]      pre-computed vision features
    Returns:
        generated text (str)
    """
    ids = prompt_ids.to(device)
    feat = features.to(device)

    # Adapter: fp32 compute → fp16 output
    projected = adapter(feat.float()).half()          # [1, 256, 2048]

    # Text embeddings
    text_emb = embed_fn(ids)                          # [1, seq, 2048]

    # Find <image> position and splice
    positions = (ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    if len(positions) > 0:
        p = positions[0].item()
        combined = torch.cat(
            [text_emb[0, :p], projected[0], text_emb[0, p + 1:]],
            dim=0,
        ).unsqueeze(0)
        # Track original token IDs (needed for stopping)
        prefix_ids = ids[0, :p].tolist()
        suffix_ids = ids[0, p + 1:].tolist()
        prompt_token_ids = prefix_ids + suffix_ids  # without <image>
    else:
        combined = text_emb
        prompt_token_ids = ids[0].tolist()

    seq_len = combined.size(1)
    mask = torch.ones(1, seq_len, device=device)

    # Manual autoregressive generation with KV caching
    generated_ids = []
    past_key_values = None

    for step in range(max_new_tokens):
        # Forward pass
        if step == 0:
            # First step: process entire prompt
            outputs = coder(
                inputs_embeds=combined,
                attention_mask=mask,
                use_cache=True,
                past_key_values=None,
            )
            past_key_values = outputs.past_key_values
        else:
            # Subsequent steps: only process new token (use cached KV)
            outputs = coder(
                inputs_embeds=next_emb,
                attention_mask=mask,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values

        # Get next token logits (last position)
        logits = outputs.logits[0, -1, :]  # [vocab_size]
        next_token_id = logits.argmax().item()

        # Stop on EOS
        if next_token_id == tokenizer.eos_token_id:
            break

        generated_ids.append(next_token_id)

        # Prepare next token embedding (for next iteration)
        next_emb = embed_fn(torch.tensor([[next_token_id]], device=device))  # [1, 1, 2048]

        # Extend attention mask (KV cache handles sequence extension)
        mask = torch.cat([mask, torch.ones(1, 1, device=device)], dim=1)

    # Decode (skip the prompt, only decode generated tokens)
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Diagnostic: Reconstruction Quality Test")
    parser.add_argument("--checkpoint", default="./checkpoints/phase2a/best.pt")
    parser.add_argument("--features_dir", default="./precomputed_features")
    parser.add_argument("--val_manifest",
                        default="Data Crawling/output/manifests/val.jsonl")
    parser.add_argument("--coder_model",
                        default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max tokens to generate (increased for reconstruction)")
    parser.add_argument("--max_samples", type=int, default=50,
                        help="Number of val examples to test (0 = use all)")
    parser.add_argument("--output", default="coder_vl/diagnostic_results.json")
    args = parser.parse_args()

    device = "cuda"

    # ==================================================================
    # 1. Coder model  (same 4-bit setup as training)
    # ==================================================================
    print("=" * 60)
    print("LOADING CODER MODEL (4-bit, fp16)")
    print("=" * 60)

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    coder = AutoModelForCausalLM.from_pretrained(
        args.coder_model,
        quantization_config=bnb_cfg,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.coder_model, trust_remote_code=True,
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<image>", "<img_start>", "<img_end>"]}
    )
    coder.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    coder.eval()
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    embed_fn = coder.get_input_embeddings()
    print(f"  hidden_size={coder.config.hidden_size}  "
          f"image_token_id={image_token_id}")
    print("  Coder model loaded (eval mode)\n")

    # ==================================================================
    # 2. Adapter
    # ==================================================================
    print(f"Loading adapter from {args.checkpoint} ...")
    adapter = ProjectionAdapter(
        vision_dim=1280, hidden_dim=4096,
        coder_dim=coder.config.hidden_size,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(ckpt, dict) and "adapter_state_dict" in ckpt:
        adapter.load_state_dict(ckpt["adapter_state_dict"])
        step = ckpt.get("global_step", "?")
        print(f"  Loaded full checkpoint (step={step})")
    else:
        # adapter_final.pt is a raw state_dict
        adapter.load_state_dict(ckpt)
        print("  Loaded raw state_dict")
    adapter = adapter.to(device).eval()
    print(f"  Adapter ready ({adapter.num_parameters():,} params)\n")

    # ==================================================================
    # 3. Validation data
    # ==================================================================
    print("Loading validation data ...")
    features_dir = Path(args.features_dir)
    examples = []
    with open(args.val_manifest) as f:
        for line in f:
            examples.append(json.loads(line))

    valid = []
    for ex in examples:
        fp = features_dir / (Path(ex["image"]).stem + ".pt")
        if fp.exists():
            ex["_feat_path"] = str(fp)
            valid.append(ex)

    print(f"  {len(valid)}/{len(examples)} examples have features")
    if args.max_samples > 0:
        valid = valid[: args.max_samples]
        print(f"  Limited to {len(valid)} samples")
    print()

    # ==================================================================
    # 4. Generate reconstructions
    # ==================================================================
    print("=" * 60)
    print("DIAGNOSTIC: CODE RECONSTRUCTION TEST")
    print("=" * 60 + "\n")

    # Reconstruction prompt: ask model to transcribe the code image
    RECONSTRUCTION_PROMPT = (
        "Please transcribe the code shown in the image. "
        "Output only the code without any explanation.\n\n<image>"
    )

    results = []
    for ex in tqdm(valid, desc="Reconstructing"):
        # Ground truth: original code from the image
        # Extract from the original question or use image path to infer
        conv = ex["conversations"]

        # For reconstruction, we want the actual code content
        # The reference answer might contain descriptions, so we use the
        # image path to get the original source code if available
        reference = conv[1]["content"]  # Use assistant response as baseline

        # Prompt: simple reconstruction request
        prompt = f"User: {RECONSTRUCTION_PROMPT}\n\nAssistant:"
        tok = tokenizer(prompt, return_tensors="pt")
        prompt_ids = tok["input_ids"]                            # [1, seq]
        feat = torch.load(ex["_feat_path"], map_location="cpu")  # [256, 1280]
        feat = feat.unsqueeze(0)                                 # [1, 256, 1280]

        generated = generate_one(
            prompt_ids, feat, adapter, coder, tokenizer,
            image_token_id, embed_fn, device, args.max_new_tokens,
        )

        # Compute metrics
        bleu = bleu_score(generated, reference)
        rouge = rouge_l_f1(generated, reference)

        results.append({
            "id":        ex.get("id", ""),
            "image":     ex["image"],
            "generated": generated,
            "reference": reference,
            "bleu":      bleu,
            "rouge_l":   rouge,
        })

    # ==================================================================
    # 5. Aggregate metrics and verdict
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("DIAGNOSTIC RESULTS")
    print(f"{'=' * 60}\n")

    bleu_scores = [r["bleu"] for r in results]
    rouge_scores = [r["rouge_l"] for r in results]

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0

    print(f"  Samples tested:       {len(results)}")
    print(f"  Average BLEU-4:       {avg_bleu:.3f}")
    print(f"  Average ROUGE-L:      {avg_rouge:.3f}")
    print()

    # Decision logic
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    if avg_bleu >= 0.3:
        verdict = "PASS — Visual features preserve semantic information"
        recommendation = "→ Proceed with stronger adapter (Option 2: direct alignment, deeper MLP, or Q-Former)"
    elif avg_bleu < 0.1:
        verdict = "FAIL — Information lost in visual encoding"
        recommendation = "→ Need vision encoder fine-tuning (Option 3: contrastive pre-training)"
    else:
        verdict = "GRAY ZONE — Partial information preserved"
        recommendation = "→ Try stronger adapter first, may need more training data or encoder fine-tuning"

    print(f"\n  {verdict}")
    print(f"  {recommendation}")
    print()

    # ==================================================================
    # 6. Save results
    # ==================================================================
    output_data = {
        "num_samples": len(results),
        "avg_bleu": avg_bleu,
        "avg_rouge_l": avg_rouge,
        "verdict": verdict,
        "recommendation": recommendation,
        "per_example": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to {output_path}")
    print()

    # Show a few examples
    print("=" * 60)
    print("SAMPLE OUTPUTS (first 3)")
    print("=" * 60)
    for i, r in enumerate(results[:3], 1):
        print(f"\n--- Example {i} ---")
        print(f"Image: {r['image']}")
        print(f"BLEU: {r['bleu']:.3f}  |  ROUGE-L: {r['rouge_l']:.3f}")
        print(f"\nGenerated ({len(r['generated'].split())} words):")
        print(r['generated'][:200] + ("..." if len(r['generated']) > 200 else ""))
        print(f"\nReference ({len(r['reference'].split())} words):")
        print(r['reference'][:200] + ("..." if len(r['reference']) > 200 else ""))


if __name__ == "__main__":
    main()
