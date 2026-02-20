"""
Extract SigLIP-SO400M Vision Encoder

Downloads and saves the SigLIP-SO400M-patch14-384 vision encoder for reuse.
Output dim: 1152D, tokens: 729 per image (27x27 patches at 384px / 14px patch)
"""

import torch
import argparse
from pathlib import Path


def extract_siglip(
    model_name="google/siglip-so400m-patch14-384",
    output_path="./models/siglip_encoder.pt",
    device="cuda",
):
    print("=" * 70)
    print("EXTRACTING SIGLIP-SO400M VISION ENCODER")
    print("=" * 70)

    from transformers import SiglipVisionModel

    print(f"Loading SigLIP from {model_name}...")
    vision_model = SiglipVisionModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    ).to(device).eval()

    total_params = sum(p.numel() for p in vision_model.parameters())
    config = vision_model.config
    print(f"  Parameters: {total_params:,}")
    print(f"  Hidden dim: {config.hidden_size}")
    print(f"  Image size: {config.image_size}")
    print(f"  Patch size: {config.patch_size}")

    # Verify output shape
    print("\nVerifying output shape...")
    img_size = config.image_size
    dummy = torch.randn(1, 3, img_size, img_size, dtype=torch.float16, device=device)
    with torch.no_grad():
        output = vision_model(dummy)
        features = output.last_hidden_state
    print(f"  Output shape: {features.shape}")
    print(f"  Num tokens: {features.shape[1]}")
    print(f"  Feature dim: {features.shape[2]}")

    # Save
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": vision_model.cpu().state_dict(),
        "model_name": model_name,
        "hidden_size": config.hidden_size,
        "image_size": config.image_size,
        "patch_size": config.patch_size,
        "num_tokens": features.shape[1],
        "num_parameters": total_params,
    }
    torch.save(checkpoint, output_path)

    file_size = Path(output_path).stat().st_size / (1024**3)
    print(f"\nSaved to {output_path} ({file_size:.2f} GB)")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="google/siglip-so400m-patch14-384")
    parser.add_argument("--output_path", default="./models/siglip_encoder.pt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    extract_siglip(args.model_name, args.output_path, args.device)
