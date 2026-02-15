"""
Binary Classification Test — Can the model use visual features at all?

Tests whether the trained model can answer simple yes/no questions about code images.
This is MUCH simpler than generating function lists, so if this fails, we know
the token integration is fundamentally broken.

Test cases:
1. "Does this code contain a class definition?" → Yes/No
2. "Does this code contain a function definition?" → Yes/No
3. "Does this code import numpy?" → Yes/No

If the model can't distinguish between images on these simple tasks, the visual
pathway is broken regardless of adapter capacity.
"""

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from projector import ProjectionAdapter


def load_model_and_adapter(device="cuda"):
    """Load coder model and trained adapter."""
    print("=" * 80)
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

    # Load adapter
    adapter = ProjectionAdapter(vision_dim=1280, hidden_dim=4096, coder_dim=2048)
    ckpt = torch.load("./checkpoints/phase2a/best.pt", map_location="cpu")
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    adapter = adapter.to(device).eval()
    print(f"Adapter loaded: {adapter.num_parameters():,} params\n")

    return coder, tokenizer, adapter, image_token_id, embed_fn


@torch.no_grad()
def generate_answer(question, features, adapter, coder, tokenizer, image_token_id, embed_fn, device):
    """Generate yes/no answer to a question about the image."""
    # Format prompt (expect "Yes" or "No" response)
    prompt = f"User: <img_start><image><img_end>\n{question} Answer with only 'Yes' or 'No'.\n\nAssistant:"

    tok = tokenizer(prompt, return_tensors="pt")
    prompt_ids = tok["input_ids"].to(device)

    # Project features
    feat = features.unsqueeze(0).to(device)
    projected = adapter(feat.float()).half()

    # Text embeddings
    text_emb = embed_fn(prompt_ids)

    # Replace <image> token
    positions = (prompt_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    if len(positions) > 0:
        p = positions[0].item()
        combined = torch.cat(
            [text_emb[0, :p], projected[0], text_emb[0, p + 1:]],
            dim=0,
        ).unsqueeze(0)
    else:
        combined = text_emb

    # Generate (expect very short answer)
    max_new_tokens = 10
    generated_ids = []
    past_key_values = None
    mask = torch.ones(1, combined.size(1), device=device)

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
            break

        generated_ids.append(next_token_id)
        next_emb = embed_fn(torch.tensor([[next_token_id]], device=device))
        mask = torch.cat([mask, torch.ones(1, 1, device=device)], dim=1)

    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return text


def main():
    device = "cuda"

    coder, tokenizer, adapter, image_token_id, embed_fn = load_model_and_adapter(device)

    # Define test cases (manually selected from validation set)
    # We know these files' content from the validation manifest
    test_cases = [
        {
            "image_stem": "linear_monokai",  # pytorch linear.py (has classes)
            "questions": [
                ("Does this code contain a class definition?", "Yes"),
                ("Does this code import tensorflow?", "No"),
            ],
        },
        {
            "image_stem": "ctx_manager_monokai",  # pytorch ctx_manager.py (has functions)
            "questions": [
                ("Does this code contain a function definition?", "Yes"),
                ("Does this code contain a class definition?", "Yes"),  # might have both
            ],
        },
    ]

    print("=" * 80)
    print("BINARY CLASSIFICATION TEST")
    print("=" * 80)
    print("Testing if the model can answer simple yes/no questions.\n")

    features_dir = Path("./precomputed_features")
    total_correct = 0
    total_questions = 0

    for case in test_cases:
        feat_path = features_dir / f"{case['image_stem']}.pt"

        if not feat_path.exists():
            print(f"⚠️  Features not found for {case['image_stem']}, skipping...")
            continue

        features = torch.load(feat_path, map_location="cpu")

        print(f"\n{'─' * 80}")
        print(f"Image: {case['image_stem']}")
        print(f"{'─' * 80}")

        for question, expected in case["questions"]:
            answer = generate_answer(
                question, features, adapter, coder, tokenizer,
                image_token_id, embed_fn, device
            )

            # Check if answer contains "Yes" or "No"
            answer_lower = answer.lower()
            if "yes" in answer_lower and "no" not in answer_lower:
                predicted = "Yes"
            elif "no" in answer_lower and "yes" not in answer_lower:
                predicted = "No"
            else:
                predicted = answer  # ambiguous

            correct = (predicted == expected)
            total_correct += int(correct)
            total_questions += 1

            status = "✓" if correct else "✗"
            print(f"\n  Q: {question}")
            print(f"  Expected: {expected}")
            print(f"  Got: {answer} → {predicted}")
            print(f"  {status} {'CORRECT' if correct else 'WRONG'}")

    print(f"\n{'=' * 80}")
    print("RESULTS")
    print("=" * 80)
    accuracy = total_correct / total_questions if total_questions > 0 else 0
    print(f"Accuracy: {total_correct}/{total_questions} = {accuracy:.1%}\n")

    if accuracy > 0.7:
        print("✓ GOOD: Model CAN use visual features for simple tasks")
        print("  → Adapter works but needs more capacity for complex tasks")
        print("  → Try: deeper adapter, more training data, or attention supervision")
    elif accuracy > 0.4:
        print("⚠️  MIXED: Model partially uses visual features")
        print("  → Token integration works but signal is weak")
        print("  → Try: stronger adapter or better feature extraction")
    else:
        print("✗ BAD: Model CANNOT use visual features")
        print("  → Fundamental issue with token integration or feature quality")
        print("  → Check: attention masks, embedding alignment, feature extraction")

    print("=" * 80)


if __name__ == "__main__":
    main()
