"""
SigLIP vs OCR-2 Alignment Test

Compares how well each vision encoder's features align with the Coder model's
representation space WITHOUT any training.

For each encoder:
1. Initialize random adapter (same seed for fairness)
2. Insert visual tokens into prompt
3. Compute loss/perplexity on ground truth answers

Lower perplexity = better natural alignment = more likely to succeed with training.

Memory budget (~V100 32GB):
  Coder 4-bit: ~5 GB | SigLIP fp16: ~0.8 GB | overhead: ~3 GB = ~9 GB total
"""

import gc
import json
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    SiglipVisionModel,
    SiglipImageProcessor,
)


def load_ocr2_features(features_dir, image_path):
    """Load precomputed OCR-2 features for an image."""
    stem = Path(image_path).stem
    pt_path = Path(features_dir) / f"{stem}.pt"
    if pt_path.exists():
        return torch.load(pt_path, map_location="cpu")  # [256, 1280]
    return None


def create_adapter(input_dim, coder_dim=2048, hidden_dim=4096, seed=42):
    """Create a random projection adapter with fixed seed."""
    torch.manual_seed(seed)
    adapter = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, coder_dim),
    )
    # Freeze — we're testing alignment, not training
    for p in adapter.parameters():
        p.requires_grad = False
    return adapter


def compute_loss(
    coder, embed_fn, tokenizer, adapter, visual_features,
    question, ground_truth, image_token_id, device,
):
    """Compute cross-entropy loss on ground truth given visual features + random adapter."""
    # Tokenize
    q_ids = tokenizer(question, return_tensors="pt", add_special_tokens=True)["input_ids"].to(device)
    a_ids = tokenizer(ground_truth, return_tensors="pt", add_special_tokens=False,
                      max_length=256, truncation=True)["input_ids"].to(device)

    # Embeddings
    text_emb = embed_fn(q_ids)      # [1, seq, 2048]
    ans_emb = embed_fn(a_ids)       # [1, ans, 2048]

    # Project visual features
    vis = visual_features.unsqueeze(0).to(device).half()  # [1, N, vis_dim]
    projected = adapter.to(device).half()(vis)             # [1, N, 2048]

    # Find <image> token
    image_pos = (q_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    if len(image_pos) == 0:
        return float("inf")
    p = image_pos[0].item()

    # Build combined: [before_img | visual | after_img | answer]
    combined = torch.cat([
        text_emb[0, :p],
        projected[0],
        text_emb[0, p + 1:],
        ans_emb[0],
    ], dim=0).unsqueeze(0)

    # Labels: -100 for prompt positions, real ids for answer
    prompt_len = combined.size(1) - a_ids.size(1)
    labels = torch.cat([
        torch.full((1, prompt_len), -100, device=device),
        a_ids,
    ], dim=1)

    mask = torch.ones(combined.size(1), device=device).unsqueeze(0)

    with torch.no_grad():
        out = coder(inputs_embeds=combined.half(), attention_mask=mask, labels=labels)
    return out.loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_manifest", default="Data Crawling/output/manifests/val.jsonl")
    parser.add_argument("--ocr2_features_dir", default="./precomputed_features")
    parser.add_argument("--siglip_model", default="google/siglip-so400m-patch14-384")
    parser.add_argument("--coder_model", default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct")
    parser.add_argument("--num_examples", type=int, default=30)
    parser.add_argument("--num_seeds", type=int, default=3)
    args = parser.parse_args()

    device = "cuda"

    print("=" * 70)
    print("SIGLIP vs OCR-2 ALIGNMENT TEST")
    print("=" * 70)

    # --- [1/4] Coder model ---
    print("\n[1/4] Loading coder model (4-bit)...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    coder = AutoModelForCausalLM.from_pretrained(
        args.coder_model,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.coder_model, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<image>", "<img_start>", "<img_end>"]})
    coder.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    coder.eval()

    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    embed_fn = coder.get_input_embeddings()
    coder_dim = coder.config.hidden_size
    print(f"  Coder dim: {coder_dim}, Image token: {image_token_id}")

    # --- [2/4] SigLIP encoder ---
    print("\n[2/4] Loading SigLIP encoder...")
    siglip = SiglipVisionModel.from_pretrained(
        args.siglip_model, torch_dtype=torch.float16,
    ).to(device).eval()
    siglip_processor = SiglipImageProcessor.from_pretrained(args.siglip_model)
    siglip_dim = siglip.config.hidden_size
    print(f"  SigLIP dim: {siglip_dim}")

    # --- [3/4] Validation examples ---
    print(f"\n[3/4] Loading validation examples (need OCR-2 precomputed features)...")
    with open(args.val_manifest) as f:
        all_examples = [json.loads(line) for line in f]

    examples = []
    for ex in all_examples:
        if load_ocr2_features(args.ocr2_features_dir, ex["image"]) is not None:
            examples.append(ex)
        if len(examples) >= args.num_examples:
            break
    print(f"  Selected {len(examples)} examples with OCR-2 features available")

    # --- [4/4] Alignment comparison ---
    print(f"\n[4/4] Running alignment test ({args.num_seeds} seeds x {len(examples)} examples)...")
    ocr2_dim = 1280

    all_ocr2 = []
    all_siglip = []

    for seed in range(args.num_seeds):
        print(f"\n--- Seed {seed} ---")
        ocr2_adapter = create_adapter(ocr2_dim, coder_dim, seed=seed)
        siglip_adapter = create_adapter(siglip_dim, coder_dim, seed=seed)

        seed_ocr2, seed_siglip = [], []

        for i, ex in enumerate(examples):
            conv = ex["conversations"]
            question = conv[0]["content"]
            ground_truth = conv[1]["content"]

            # OCR-2: load precomputed
            ocr2_feat = load_ocr2_features(args.ocr2_features_dir, ex["image"])

            # SigLIP: encode on-the-fly
            try:
                image = Image.open(ex["image"]).convert("RGB")
                inputs = siglip_processor(images=image, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device).half()
                with torch.no_grad():
                    siglip_feat = siglip(pixel_values).last_hidden_state.squeeze(0).cpu()
            except Exception as e:
                print(f"  Skip {i}: {e}")
                continue

            # Compute losses
            loss_ocr2 = compute_loss(
                coder, embed_fn, tokenizer, ocr2_adapter, ocr2_feat,
                question, ground_truth, image_token_id, device,
            )
            loss_siglip = compute_loss(
                coder, embed_fn, tokenizer, siglip_adapter, siglip_feat,
                question, ground_truth, image_token_id, device,
            )

            seed_ocr2.append(loss_ocr2)
            seed_siglip.append(loss_siglip)

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(examples)}] OCR-2: {np.mean(seed_ocr2):.3f}  SigLIP: {np.mean(seed_siglip):.3f}")

        all_ocr2.extend(seed_ocr2)
        all_siglip.extend(seed_siglip)
        print(f"  Seed {seed} done — OCR-2: {np.mean(seed_ocr2):.3f}  SigLIP: {np.mean(seed_siglip):.3f}")

    # --- Results ---
    ocr2_mean = np.mean(all_ocr2)
    siglip_mean = np.mean(all_siglip)
    ocr2_ppl = np.exp(min(ocr2_mean, 20))   # cap to avoid overflow
    siglip_ppl = np.exp(min(siglip_mean, 20))

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  OCR-2   — Avg Loss: {ocr2_mean:.4f}   Perplexity: {ocr2_ppl:.1f}")
    print(f"  SigLIP  — Avg Loss: {siglip_mean:.4f}   Perplexity: {siglip_ppl:.1f}")
    print()

    if siglip_mean < ocr2_mean - 0.05:
        diff_pct = (ocr2_mean - siglip_mean) / ocr2_mean * 100
        print(f"  RESULT: SigLIP is {diff_pct:.1f}% better aligned")
        print(f"  RECOMMENDATION: Proceed with full SigLIP training")
    elif abs(siglip_mean - ocr2_mean) <= 0.05:
        print(f"  RESULT: No significant difference ({abs(siglip_mean - ocr2_mean):.4f})")
        print(f"  RECOMMENDATION: Feature source isn't the bottleneck — try stronger adapter")
    else:
        diff_pct = (siglip_mean - ocr2_mean) / siglip_mean * 100
        print(f"  RESULT: OCR-2 is {diff_pct:.1f}% better aligned")
        print(f"  RECOMMENDATION: Don't switch to SigLIP")

    print("=" * 70)

    # Per-seed breakdown
    print("\nPer-seed breakdown:")
    n = len(examples)
    for s in range(args.num_seeds):
        s_ocr2 = all_ocr2[s * n:(s + 1) * n]
        s_siglip = all_siglip[s * n:(s + 1) * n]
        print(f"  Seed {s}: OCR-2={np.mean(s_ocr2):.4f}  SigLIP={np.mean(s_siglip):.4f}  "
              f"delta={np.mean(s_ocr2) - np.mean(s_siglip):+.4f}")


if __name__ == "__main__":
    main()
