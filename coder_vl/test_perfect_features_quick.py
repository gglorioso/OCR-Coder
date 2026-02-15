"""
Test 1: Perfect Features Experiment (Quick Inference Test)

No training needed. Just test if token insertion works when we give the model
PERFECT features (text embeddings from the coder model itself).

If this works → adapter is the problem
If this fails → token insertion mechanism is broken
"""

import torch
import argparse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def replace_image_with_perfect_features(
    input_ids, visual_text, tokenizer, coder, embed_fn, image_token_id, device
):
    """
    Replace <image> token with perfect visual features (text embeddings).

    Args:
        input_ids: [1, seq] - question tokens with <image>
        visual_text: str - the actual code content (ground truth)
        tokenizer: tokenizer
        coder: coder model
        embed_fn: embedding layer
        image_token_id: int
        device: device

    Returns:
        combined_embeds: [1, new_seq, dim]
        attention_mask: [1, new_seq]
    """
    # Tokenize visual content → these will become our "visual features"
    visual_tokens = tokenizer(
        visual_text,
        max_length=256,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        add_special_tokens=False,
    )
    visual_token_ids = visual_tokens["input_ids"].to(device)

    # Get embeddings for both text and "visual" content
    text_embeds = embed_fn(input_ids)  # [1, seq, dim]
    visual_embeds = embed_fn(visual_token_ids)  # [1, 256, dim]

    # Find <image> token position
    image_pos = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]

    if len(image_pos) == 0:
        # No image token - shouldn't happen
        return text_embeds, torch.ones(input_ids.size(1), device=device).unsqueeze(0)

    p = image_pos[0].item()

    # Concatenate: [before_image] + [visual_embeds] + [after_image]
    combined = torch.cat([
        text_embeds[0, :p],      # Before <image>
        visual_embeds[0],         # Replace <image> with 256 visual tokens
        text_embeds[0, p+1:]      # After <image>
    ], dim=0).unsqueeze(0)  # [1, new_seq, dim]

    # Create attention mask (all 1s - attend to everything)
    attention_mask = torch.ones(combined.size(1), device=device).unsqueeze(0)

    return combined, attention_mask


def test_example(example, tokenizer, coder, embed_fn, image_token_id, device, max_new_tokens=256):
    """Test a single example with perfect features."""
    conv = example["conversations"]
    question = conv[0]["content"]  # Has <img_start><image><img_end>\nQuestion?
    ground_truth = conv[1]["content"]  # The answer

    # Extract the actual question text (after the image tags)
    # Format: <img_start><image><img_end>\n{actual_question}
    if "\n" in question:
        question_text = question.split("\n", 1)[1]
    else:
        question_text = question

    # Use ground truth as "visual content" - this is our perfect feature
    # The model should be able to "see" this content through visual tokens
    visual_text = ground_truth[:1000]  # Use first 1000 chars

    print("=" * 80)
    print(f"Question: {question_text[:100]}...")
    print(f"\nVisual Content (first 200 chars): {visual_text[:200]}...")
    print(f"\nGround Truth: {ground_truth[:200]}...")

    # Tokenize question
    question_tokens = tokenizer(question, return_tensors="pt", add_special_tokens=True)
    input_ids = question_tokens["input_ids"].to(device)

    # Replace <image> with perfect features
    with torch.no_grad():
        combined_embeds, attention_mask = replace_image_with_perfect_features(
            input_ids, visual_text, tokenizer, coder, embed_fn, image_token_id, device
        )

        # Manual autoregressive generation (since .generate() doesn't handle inputs_embeds well)
        seq_len = combined_embeds.size(1)
        mask = torch.ones(1, seq_len, device=device)
        generated_ids = []
        past_key_values = None

        for step in range(max_new_tokens):
            # Forward pass
            if step == 0:
                # First step: process entire prompt
                outputs = coder(
                    inputs_embeds=combined_embeds,
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

            # Extend attention mask
            mask = torch.cat([mask, torch.ones(1, 1, device=device)], dim=1)

    # Decode the generated tokens
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Extract answer (after the question)
    if "\n\n" in generated_text:
        parts = generated_text.split("\n\n")
        answer = "\n\n".join(parts[1:]) if len(parts) > 1 else generated_text
    else:
        answer = generated_text

    print(f"\nGenerated Answer: {answer[:300]}...")
    print("=" * 80)

    return {
        "question": question_text,
        "ground_truth": ground_truth,
        "generated": answer,
        "visual_content": visual_text,
    }


def main():
    parser = argparse.ArgumentParser(description="Test 1: Perfect Features (Quick Test)")
    parser.add_argument("--test_manifest", default="Data Crawling/output/manifests/test.jsonl")
    parser.add_argument("--coder_model", default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct")
    parser.add_argument("--num_examples", type=int, default=5, help="Number of examples to test")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    device = "cuda"

    print("=" * 80)
    print("TEST 1: PERFECT FEATURES EXPERIMENT (QUICK INFERENCE TEST)")
    print("=" * 80)
    print("Testing if token insertion works with perfect features (text embeddings).")
    print("NO TRAINING - just inference with ground truth as 'visual' content.")
    print("=" * 80)
    print()

    # Load coder model
    print("Loading coder model (4-bit quantized)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    coder = AutoModelForCausalLM.from_pretrained(
        args.coder_model,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.coder_model, trust_remote_code=True)

    # Add special tokens
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<image>", "<img_start>", "<img_end>"]}
    )
    coder.resize_token_embeddings(len(tokenizer))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    coder.eval()

    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    embed_fn = coder.get_input_embeddings()

    print(f"✓ Model loaded (hidden_dim={coder.config.hidden_size})")
    print(f"✓ Image token ID: {image_token_id}\n")

    # Load test examples
    print(f"Loading test examples from {args.test_manifest}...")
    with open(args.test_manifest) as f:
        examples = [json.loads(line) for line in f]

    examples = examples[:args.num_examples]
    print(f"✓ Loaded {len(examples)} examples\n")

    # Test each example
    results = []
    for i, example in enumerate(examples):
        print(f"\n[Example {i+1}/{len(examples)}]")
        result = test_example(
            example, tokenizer, coder, embed_fn, image_token_id, device, args.max_new_tokens
        )
        results.append(result)

    # Summary
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print("\nRESULTS INTERPRETATION:")
    print("-" * 80)
    print("Look at the generated answers above and compare to ground truth.")
    print()
    print("✓ If answers are CORRECT or SIMILAR to ground truth:")
    print("  → Token insertion mechanism WORKS")
    print("  → Problem is the projection adapter (too weak to map OCR-2 → Coder space)")
    print("  → Solution: Stronger adapter, better vision encoder, or add supervision")
    print()
    print("✗ If answers are WRONG or HALLUCINATED (ignoring visual content):")
    print("  → Token insertion mechanism is BROKEN")
    print("  → Even with perfect features, model can't use visual tokens")
    print("  → Solution: Redesign architecture (cross-attention, debug attention masks)")
    print("=" * 80)


if __name__ == "__main__":
    main()
