"""
Test 1: Image Sensitivity Check

Runs inference twice per example — once with the CORRECT feature file,
once with a RANDOMLY SWAPPED feature file from a different source image.

If the model is actually using visual tokens, outputs should differ
substantially between runs. If outputs are near-identical, the model is
ignoring the image and generating purely from language priors.

Reports:
  - Per-example ROUGE-L between correct-image output vs swapped-image output
  - Mean / median sentence-level similarity across all N examples
  - Example outputs so you can eyeball the difference

Usage (local smoke test):
    python coder_vl/test_image_sensitivity.py --max_samples 10 --no_model

Usage (full with model):
    python coder_vl/test_image_sensitivity.py --max_samples 50
"""

import json
import argparse
import random
import sys
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Minimal ROUGE-L (no external deps)
# ---------------------------------------------------------------------------

def rouge_l_f1(a: str, b: str) -> float:
    a_tok = a.lower().split()
    b_tok = b.lower().split()
    if not a_tok or not b_tok:
        return 0.0
    m, n = len(a_tok), len(b_tok)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a_tok[i - 1] == b_tok[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    lcs = prev[n]
    p = lcs / n
    r = lcs / m
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# Generation (copy of evaluate_phase2b generate_one)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_one(prompt_ids, features, adapter, coder, tokenizer,
                 image_token_id, embed_fn, device, max_new_tokens,
                 repetition_penalty=1.3):
    ids = prompt_ids.to(device)
    feat = features.to(device)

    projected = adapter(feat.float()).half()
    text_emb = embed_fn(ids)

    positions = (ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    if len(positions) > 0:
        p = positions[0].item()
        combined = torch.cat(
            [text_emb[0, :p], projected[0], text_emb[0, p + 1:]],
            dim=0,
        ).unsqueeze(0)
    else:
        combined = text_emb

    mask = torch.ones(1, combined.size(1), device=device)
    generated_ids = []
    past_key_values = None

    for step in range(max_new_tokens):
        if step == 0:
            outputs = coder(
                inputs_embeds=combined,
                attention_mask=mask,
                use_cache=True,
                past_key_values=None,
            )
        else:
            outputs = coder(
                inputs_embeds=next_emb,
                attention_mask=mask,
                use_cache=True,
                past_key_values=past_key_values,
            )
        past_key_values = outputs.past_key_values

        logits = outputs.logits[0, -1, :].clone()
        if repetition_penalty != 1.0 and generated_ids:
            for tok_id in set(generated_ids):
                logits[tok_id] = (logits[tok_id] / repetition_penalty
                                  if logits[tok_id] > 0
                                  else logits[tok_id] * repetition_penalty)

        next_token_id = logits.argmax().item()
        if next_token_id == tokenizer.eos_token_id:
            break
        generated_ids.append(next_token_id)
        next_emb = embed_fn(torch.tensor([[next_token_id]], device=device))
        mask = torch.cat([mask, torch.ones(1, 1, device=device)], dim=1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    default="./checkpoints/phase2b/best.pt")
    parser.add_argument("--features_dir",  default="./precomputed_features_tiled")
    parser.add_argument("--val_manifest",  default="data_v2b/manifests/val.jsonl")
    parser.add_argument("--coder_model",   default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct")
    parser.add_argument("--max_samples",   type=int, default=50,
                        help="Number of val examples to test (default 50)")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--save_file",     default="./sensitivity_results.json")
    parser.add_argument("--lora_r",        type=int, default=16)
    parser.add_argument("--lora_alpha",    type=int, default=32)
    parser.add_argument("--lora_dropout",  type=float, default=0.05)
    parser.add_argument("--no_model",      action="store_true",
                        help="Skip model loading — only check data/features are paired correctly")
    args = parser.parse_args()

    random.seed(args.seed)

    # ------------------------------------------------------------------
    # 1. Load val manifest + match features
    # ------------------------------------------------------------------
    features_dir = Path(args.features_dir)
    examples = []
    with open(args.val_manifest) as f:
        for line in f:
            examples.append(json.loads(line))

    # One feature file per image (multiple tasks share same image)
    valid = []
    seen_images = set()
    for ex in examples:
        fp = features_dir / (Path(ex["image"]).stem + ".pt")
        if fp.exists():
            ex["_feat_path"] = str(fp)
            # Keep only one task type per image to avoid redundant inference
            if fp not in seen_images:
                valid.append(ex)
                seen_images.add(fp)

    print(f"Unique images with features: {len(valid)}")
    random.shuffle(valid)
    valid = valid[: args.max_samples]
    print(f"Testing with {len(valid)} examples\n")

    if args.no_model:
        print("--no_model set: verifying feature file pairing only")
        for ex in valid[:5]:
            feat = torch.load(ex["_feat_path"], map_location="cpu")
            print(f"  {Path(ex['_feat_path']).name}: shape={feat.shape}")
        print("OK — features load correctly.")
        return

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig, TaskType
    sys.path.insert(0, str(Path(__file__).parent))
    from projector import ProjectionAdapter

    device = "cuda"

    print("Loading coder model (4-bit QLoRA) ...")
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
    tokenizer = AutoTokenizer.from_pretrained(args.coder_model, trust_remote_code=True)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<image>", "<img_start>", "<img_end>"]}
    )
    coder.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM, bias="none",
    )
    coder = get_peft_model(coder, lora_config)
    coder.eval()

    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    embed_fn = coder.get_input_embeddings()

    print("Loading checkpoint ...")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    adapter = ProjectionAdapter(
        vision_dim=1280, hidden_dim=4096, coder_dim=coder.config.hidden_size,
    )
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    coder.load_state_dict(ckpt["lora_state_dict"], strict=False)
    adapter = adapter.to(device).eval()
    print(f"  Checkpoint loaded (step={ckpt.get('global_step','?')})\n")

    # ------------------------------------------------------------------
    # 3. Run paired inference
    # ------------------------------------------------------------------
    feat_paths = [ex["_feat_path"] for ex in valid]
    results = []

    print(f"{'#':>4}  {'Similarity':>10}  {'Same?':>6}  ID")
    print("-" * 70)

    for i, ex in enumerate(valid):
        # Build prompt (take the user turn)
        conv = ex["conversations"]
        user_text = conv[0]["content"]  # contains <img_start><image><img_end>
        prompt = f"{user_text}\n"
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

        # Correct feature
        feat_correct = torch.load(ex["_feat_path"], map_location="cpu").unsqueeze(0)

        # Swapped feature: pick a different image
        other_paths = [p for p in feat_paths if p != ex["_feat_path"]]
        swap_path = random.choice(other_paths)
        feat_swapped = torch.load(swap_path, map_location="cpu").unsqueeze(0)

        out_correct = generate_one(
            prompt_ids, feat_correct, adapter, coder, tokenizer,
            image_token_id, embed_fn, device, args.max_new_tokens,
        )
        out_swapped = generate_one(
            prompt_ids, feat_swapped, adapter, coder, tokenizer,
            image_token_id, embed_fn, device, args.max_new_tokens,
        )

        sim = rouge_l_f1(out_correct, out_swapped)
        identical = (out_correct.strip() == out_swapped.strip())

        results.append({
            "id": ex["id"],
            "task_type": ex["task_type"],
            "feat_correct": ex["_feat_path"],
            "feat_swapped": swap_path,
            "out_correct": out_correct,
            "out_swapped": out_swapped,
            "similarity": round(sim, 4),
            "identical": identical,
        })

        print(f"{i+1:>4}  {sim:>10.4f}  {'YES' if identical else 'no':>6}  {ex['id']}")
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    sims = [r["similarity"] for r in results]
    n_identical = sum(r["identical"] for r in results)
    mean_sim = sum(sims) / len(sims)
    median_sim = sorted(sims)[len(sims) // 2]

    print()
    print("=" * 70)
    print("SENSITIVITY SUMMARY")
    print("=" * 70)
    print(f"  Examples tested   : {len(results)}")
    print(f"  Identical outputs : {n_identical}/{len(results)} ({n_identical/len(results):.1%})")
    print(f"  Mean similarity   : {mean_sim:.4f}")
    print(f"  Median similarity : {median_sim:.4f}")
    print()
    print("Interpretation:")
    if mean_sim > 0.7:
        print("  >> HIGH similarity — model is largely IGNORING the image.")
        print("     The visual tokens are not meaningfully altering generation.")
    elif mean_sim > 0.4:
        print("  >> MODERATE similarity — partial image use.")
        print("     Model uses some visual signal but language prior dominates.")
    else:
        print("  >> LOW similarity — model IS sensitive to image content.")
        print("     The visual tokens are influencing generation.")
    print()

    # Show a few examples
    print("Sample comparisons (first 3):")
    for r in results[:3]:
        print(f"\n  [{r['id']}]")
        print(f"  correct image : {r['out_correct'][:120]}")
        print(f"  swapped image : {r['out_swapped'][:120]}")
        print(f"  ROUGE-L sim   : {r['similarity']:.4f}")

    # Save
    with open(args.save_file, "w") as f:
        json.dump({"summary": {
            "n": len(results),
            "n_identical": n_identical,
            "mean_similarity": round(mean_sim, 4),
            "median_similarity": round(median_sim, 4),
        }, "results": results}, f, indent=2)
    print(f"\nSaved to {args.save_file}")


if __name__ == "__main__":
    main()
