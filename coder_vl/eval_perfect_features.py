"""
Evaluate the perfect features model.

This script tests if the model can correctly answer questions when given
"perfect" visual features (text embeddings from the actual code).
"""

import torch
import argparse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from projector import ProjectionAdapter


def load_model_and_adapter(checkpoint_path, coder_model_name, device="cuda"):
    """Load the trained adapter and coder model."""
    print("Loading coder model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    coder = AutoModelForCausalLM.from_pretrained(
        coder_model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(coder_model_name, trust_remote_code=True)

    # Add special tokens
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<image>", "<img_start>", "<img_end>"]}
    )
    coder.resize_token_embeddings(len(tokenizer))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    coder.eval()

    # Load adapter
    print(f"Loading adapter from {checkpoint_path}...")
    coder_dim = coder.config.hidden_size
    adapter = ProjectionAdapter(vision_dim=coder_dim, hidden_dim=4096, coder_dim=coder_dim)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    adapter = adapter.to(device)
    adapter.eval()

    print("✓ Model and adapter loaded\n")

    return coder, tokenizer, adapter


def generate_with_perfect_features(
    coder, tokenizer, adapter, question, visual_text, device="cuda", max_new_tokens=256
):
    """
    Generate answer using perfect visual features.

    Args:
        question: Question text with <img_start><image><img_end> placeholders
        visual_text: The actual code content (will be converted to embeddings)
    """
    # Tokenize question
    question_tokens = tokenizer(question, return_tensors="pt", add_special_tokens=True)
    input_ids = question_tokens["input_ids"].to(device)

    # Tokenize visual content (will become "perfect" visual features)
    visual_tokens = tokenizer(
        visual_text,
        max_length=256,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    visual_token_ids = visual_tokens["input_ids"].to(device)

    # Get embeddings
    embed_fn = coder.get_input_embeddings()

    with torch.no_grad():
        # Get visual embeddings from text
        visual_embeds = embed_fn(visual_token_ids).half()  # [1, 256, dim]

        # Pass through adapter
        projected = adapter(visual_embeds).half()  # [1, 256, dim]

        # Replace <image> token with projected features
        image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        text_embeds = embed_fn(input_ids)

        # Find image token position
        image_pos = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]

        if len(image_pos) > 0:
            p = image_pos[0].item()
            # Concatenate: before + visual + after
            combined = torch.cat([
                text_embeds[0, :p],
                projected[0],
                text_embeds[0, p+1:]
            ], dim=0).unsqueeze(0)  # [1, new_seq, dim]

            attention_mask = torch.ones(combined.size(1), device=device).unsqueeze(0)
        else:
            combined = text_embeds
            attention_mask = torch.ones(input_ids.size(1), device=device).unsqueeze(0)

        # Manual autoregressive generation (since .generate() doesn't handle inputs_embeds well)
        seq_len = combined.size(1)
        mask = torch.ones(1, seq_len, device=device)
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

            # Extend attention mask
            mask = torch.cat([mask, torch.ones(1, 1, device=device)], dim=1)

    # Decode the generated tokens
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Extract just the answer (after the question)
    if "\n\n" in generated_text:
        answer = generated_text.split("\n\n", 1)[1] if len(generated_text.split("\n\n")) > 1 else generated_text
    else:
        answer = generated_text

    return answer.strip()


def main():
    parser = argparse.ArgumentParser(description="Evaluate perfect features model")
    parser.add_argument("--checkpoint", required=True, help="Path to adapter checkpoint")
    parser.add_argument("--test_manifest", default="Data Crawling/output/manifests/test.jsonl")
    parser.add_argument("--coder_model", default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct")
    parser.add_argument("--num_examples", type=int, default=10, help="Number of examples to test")
    args = parser.parse_args()

    device = "cuda"

    # Load model
    coder, tokenizer, adapter = load_model_and_adapter(
        args.checkpoint, args.coder_model, device
    )

    # Load test examples
    print(f"Loading test examples from {args.test_manifest}...")
    with open(args.test_manifest) as f:
        examples = [json.loads(line) for line in f]

    examples = examples[:args.num_examples]
    print(f"Testing on {len(examples)} examples\n")
    print("=" * 70)

    # Evaluate each example
    for i, ex in enumerate(examples):
        conv = ex["conversations"]
        question = conv[0]["content"]
        ground_truth = conv[1]["content"]

        # Use ground truth as "visual" content (perfect features)
        visual_text = ground_truth[:500]

        print(f"\n[Example {i+1}/{len(examples)}]")
        print(f"Question: {question[:100]}...")
        print(f"\nGround Truth: {ground_truth[:200]}...")

        # Generate answer
        answer = generate_with_perfect_features(
            coder, tokenizer, adapter, question, visual_text, device
        )

        print(f"\nGenerated: {answer[:200]}...")
        print("-" * 70)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print("\nINTERPRETATION:")
    print("  - If answers are CORRECT: Token insertion works! Problem is projection.")
    print("  - If answers are WRONG: Token insertion mechanism may be broken.")
    print("=" * 70)


if __name__ == "__main__":
    main()
