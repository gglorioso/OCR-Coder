"""
Test: Does the model generate the same output WITHOUT image features?
If yes, it proves the model isn't using the visual features.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from projector import ProjectionAdapter


def generate_text_only(prompt, coder, tokenizer, max_new_tokens=100):
    """Generate using text-only (no image)."""
    tok = tokenizer(prompt, return_tensors="pt")
    input_ids = tok["input_ids"].cuda()

    with torch.no_grad():
        outputs = coder.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0, input_ids.size(1):]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main():
    device = "cuda"

    # Load model
    print("Loading model...")
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    coder.eval()
    print("Model loaded.\n")

    # Test prompt (same as in training, but NO <image> token)
    prompt_with_placeholder = "User: <img_start><image><img_end>\nList all functions defined in this code.\n\nAssistant:"
    prompt_no_image = "User: List all functions defined in this code.\n\nAssistant:"

    print("=" * 80)
    print("TEST: Generate WITHOUT image")
    print("=" * 80)
    print(f"Prompt: {repr(prompt_no_image)}\n")

    generated = generate_text_only(prompt_no_image, coder, tokenizer, max_new_tokens=200)

    print("Generated (text-only, no training):")
    print("─" * 80)
    print(generated)
    print()

    print("=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    print("If this output looks similar to what we saw in the debug (repeating __init__),")
    print("it means the model is NOT using the visual features and is just following")
    print("learned patterns from the base model or training data distribution.")
    print()


if __name__ == "__main__":
    main()
