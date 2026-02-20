"""
Extract and pool visual features for linear probe test.

Supports two encoders:
  - ocr2:  Loads existing precomputed features from Phase 2a, mean-pools to [1280]
  - siglip: Loads SigLIP-SO400M encoder, processes images, mean-pools to [1152]

Usage:
    python coder_vl/linear_probe/extract_probe_features.py --encoder ocr2
    python coder_vl/linear_probe/extract_probe_features.py --encoder siglip
"""

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PRECOMPUTED_DIR = PROJECT_ROOT / "precomputed_features"
MODELS_DIR = PROJECT_ROOT / "models"
PROBE_DATA_DIR = Path(__file__).resolve().parent / "probe_data"


def load_ocr2_features(label_file):
    """Load existing precomputed OCR-2 features and mean-pool."""
    labels = []
    with open(label_file) as f:
        for line in f:
            labels.append(json.loads(line))

    features = []
    valid_labels = []
    missing = 0

    for label in tqdm(labels, desc="Loading OCR-2 features"):
        image_path = Path(label["image"])
        feature_file = PRECOMPUTED_DIR / f"{image_path.stem}.pt"

        if not feature_file.exists():
            missing += 1
            continue

        # [num_tokens, 1280] -> mean pool -> [1280]
        feat = torch.load(feature_file, map_location="cpu", weights_only=True)
        pooled = feat.float().mean(dim=0)
        features.append(pooled)
        valid_labels.append(label)

    if missing > 0:
        print(f"  Warning: {missing}/{len(labels)} images missing precomputed features")

    features = torch.stack(features)
    return features, valid_labels


def load_siglip_features(label_file, device="cuda"):
    """Extract SigLIP features from images and mean-pool."""
    from PIL import Image
    from torchvision import transforms
    from transformers import SiglipVisionModel

    # Disable PIL decompression bomb check
    Image.MAX_IMAGE_PIXELS = None

    # Load SigLIP encoder from extracted checkpoint
    ckpt_path = MODELS_DIR / "siglip_encoder.pt"
    print(f"  Loading SigLIP from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model_name = ckpt["model_name"]
    image_size = ckpt.get("image_size", 384)

    # Reconstruct model and load saved weights
    encoder = SiglipVisionModel.from_pretrained(model_name, torch_dtype=torch.float16)
    encoder.load_state_dict(ckpt["model_state_dict"])
    encoder = encoder.to(device).eval()
    print(f"  SigLIP loaded: {ckpt.get('hidden_size', '?')}D, {image_size}px input")

    # SigLIP uses simple normalization
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    labels = []
    with open(label_file) as f:
        for line in f:
            labels.append(json.loads(line))

    features = []
    valid_labels = []
    missing = 0

    with torch.no_grad():
        for label in tqdm(labels, desc="Extracting SigLIP features"):
            image_path = Path(label["image"])
            if not image_path.exists():
                missing += 1
                continue

            try:
                img = Image.open(image_path).convert("RGB")
                pixel_values = transform(img).unsqueeze(0).to(device).half()

                output = encoder(pixel_values)
                feat = output.last_hidden_state.squeeze(0)  # [num_tokens, 1152]
                pooled = feat.float().mean(dim=0).cpu()  # [1152]
                features.append(pooled)
                valid_labels.append(label)
            except Exception as e:
                print(f"\n  Skipping {image_path.name}: {e}")
                missing += 1
                continue

    if missing > 0:
        print(f"  Warning: {missing}/{len(labels)} images not found on disk")

    features = torch.stack(features)
    return features, valid_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=["ocr2", "siglip"], default="ocr2")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_dir = PROBE_DATA_DIR / args.encoder
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        label_file = PROBE_DATA_DIR / f"probe_labels_{split}.jsonl"
        if not label_file.exists():
            print(f"Skipping {split}: run generate_probe_labels.py first")
            continue

        print(f"\n--- {args.encoder} / {split} ---")

        if args.encoder == "ocr2":
            features, valid_labels = load_ocr2_features(label_file)
        else:
            features, valid_labels = load_siglip_features(label_file, args.device)

        print(f"  Features shape: {features.shape} ({features.dtype})")

        torch.save(features, output_dir / f"features_{split}.pt")
        with open(output_dir / f"labels_{split}.jsonl", "w") as f:
            for label in valid_labels:
                f.write(json.dumps(label) + "\n")

        print(f"  Saved to {output_dir}/")


if __name__ == "__main__":
    main()
