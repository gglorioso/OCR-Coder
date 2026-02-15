"""
Debug a single example to understand model behavior.
Shows the full prompt, features, generation process, and comparison with reference.
"""

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from projector import ProjectionAdapter


def main():
    device = "cuda"

    # Load validation example
    print("=" * 80)
    print("LOADING EXAMPLE")
    print("=" * 80)

    val_manifest = "Data Crawling/output/manifests/val.jsonl"
    with open(val_manifest) as f:
        example = json.loads(f.readline())

    print(f"\nExample ID: {example.get('id', 'unknown')}")
    print(f"Task type: {example.get('task_type', 'unknown')}")
    print(f"Image: {example['image']}")

    # Load conversations
    conv = example["conversations"]
    user_msg = conv[0]["content"]
    reference = conv[1]["content"]

    print(f"\n{'─' * 80}")
    print("USER MESSAGE:")
    print(f"{'─' * 80}")
    print(user_msg)
    print(f"\n{'─' * 80}")
    print("EXPECTED REFERENCE:")
    print(f"{'─' * 80}")
    print(reference)
    print()

    # Load precomputed features
    features_dir = Path("./precomputed_features")
    feat_path = features_dir / (Path(example["image"]).stem + ".pt")

    if not feat_path.exists():
        print(f"ERROR: Features not found at {feat_path}")
        return

    features = torch.load(feat_path, map_location="cpu")
    print(f"Features loaded: shape={features.shape}, dtype={features.dtype}")
    print(f"Feature stats: min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}")

    # Load model and adapter
    print(f"\n{'=' * 80}")
    print("LOADING MODEL")
    print("=" * 80)

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    coder = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        quantization_config=bnb_cfg,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        trust_remote_code=True,
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

    print(f"Model loaded: hidden_size={coder.config.hidden_size}")
    print(f"Image token ID: {image_token_id}")

    # Load adapter
    adapter = ProjectionAdapter(vision_dim=1280, hidden_dim=4096, coder_dim=2048)
    ckpt = torch.load("./checkpoints/phase2a/best.pt", map_location="cpu")
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    adapter = adapter.to(device).eval()
    print(f"Adapter loaded: {adapter.num_parameters():,} params")

    # Prepare prompt
    print(f"\n{'=' * 80}")
    print("GENERATING")
    print("=" * 80)

    prompt = f"User: {user_msg}\n\nAssistant:"
    print(f"\nFull prompt:\n{repr(prompt)}\n")

    tok = tokenizer(prompt, return_tensors="pt")
    prompt_ids = tok["input_ids"].to(device)

    print(f"Prompt tokens: {prompt_ids.shape[1]}")
    print(f"Decoded prompt: {tokenizer.decode(prompt_ids[0])}")
    print(f"<image> token present in prompt: {(prompt_ids[0] == image_token_id).any().item()}")

    # Project features
    feat = features.unsqueeze(0).to(device)  # [1, 256, 1280]
    projected = adapter(feat.float()).half()  # [1, 256, 2048]

    print(f"\nProjected features: shape={projected.shape}, dtype={projected.dtype}")
    print(f"Projected stats: min={projected.min():.4f}, max={projected.max():.4f}, mean={projected.mean():.4f}")

    # Text embeddings
    text_emb = embed_fn(prompt_ids)
    print(f"Text embeddings: shape={text_emb.shape}, dtype={text_emb.dtype}")

    # Find <image> and splice
    positions = (prompt_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    if len(positions) > 0:
        p = positions[0].item()
        print(f"\n<image> token found at position {p}")
        combined = torch.cat(
            [text_emb[0, :p], projected[0], text_emb[0, p + 1:]],
            dim=0,
        ).unsqueeze(0)
    else:
        print("\nWARNING: No <image> token found in prompt!")
        combined = text_emb

    print(f"Combined embeddings: shape={combined.shape}")

    # Generate
    max_new_tokens = 256
    generated_ids = []
    past_key_values = None
    mask = torch.ones(1, combined.size(1), device=device)

    print(f"\nGenerating (max {max_new_tokens} tokens)...")
    print("First 10 tokens:")

    with torch.no_grad():
        for step in range(max_new_tokens):
            if step == 0:
                outputs = coder(
                    inputs_embeds=combined,
                    attention_mask=mask,
                    use_cache=True,
                    past_key_values=None,
                )
                past_key_values = outputs.past_key_values
            else:
                outputs = coder(
                    inputs_embeds=next_emb,
                    attention_mask=mask,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                past_key_values = outputs.past_key_values

            logits = outputs.logits[0, -1, :]
            next_token_id = logits.argmax().item()

            if next_token_id == tokenizer.eos_token_id:
                print(f"\n  Step {step}: EOS token, stopping")
                break

            generated_ids.append(next_token_id)

            # Show first 10 tokens
            if step < 10:
                token_text = tokenizer.decode([next_token_id])
                print(f"  Step {step}: token_id={next_token_id}, text={repr(token_text)}")

            next_emb = embed_fn(torch.tensor([[next_token_id]], device=device))
            mask = torch.cat([mask, torch.ones(1, 1, device=device)], dim=1)

    # Decode
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    print(f"\n{'=' * 80}")
    print("RESULTS")
    print("=" * 80)
    print(f"\nGenerated ({len(generated_ids)} tokens):")
    print(f"{'─' * 80}")
    print(generated_text)
    print(f"\n{'─' * 80}")
    print("Reference:")
    print(f"{'─' * 80}")
    print(reference)
    print()


if __name__ == "__main__":
    main()
