"""
Pre-compute Vision Features for Phase 2a Training

Runs the vision encoder (from DeepSeek-OCR-2) once over all unique training
images and saves the output tensors to disk.  This means the training script
never needs to load the vision encoder at all — saving ~1-2 GB VRAM and
removing a major source of complexity.

Base mode (--tiling not set):
    Each image → [256, 1280] tensor  (single 768x768 view, 88:1 compression)

Tiling mode (--tiling):
    Each image → [1280, 1280] tensor (2x2 grid + full thumbnail = 5 views,
    each 768x768 → 256 tokens, concatenated → 1280 tokens, ~20:1 compression)

Usage:
    python precompute_features.py --output_dir ./precomputed_features
    python precompute_features.py --output_dir ./precomputed_features_tiled --tiling
"""

import gc
import json
import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def load_vision_encoder(model_name="deepseek-ai/deepseek-ocr-2", device="cuda"):
    """
    Load vision encoder components from the full DeepSeek-OCR-2 model.

    Loads the entire model on CPU, extracts vision parts (SAM, Qwen2Decoder2Encoder,
    MlpProjector), moves them to GPU, and frees everything else.
    """
    from transformers import AutoModel

    print(f"Loading DeepSeek-OCR-2 from {model_name} ...")
    print("  (Loading full model to CPU, then moving vision parts to GPU)")

    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    print("  Full model loaded to CPU")

    # Extract vision components and move to GPU
    sam = model.model.sam_model.to(device).eval()
    d2e = model.model.qwen2_model.to(device).eval()
    proj = model.model.projector.to(device).eval()
    print(f"  Vision components moved to {device}")

    # Free the language model and everything else
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    print("  Freed remaining model memory")

    return sam, d2e, proj


def get_transform(image_size=768):
    """Standard image preprocessing — resize + ImageNet normalization."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def tile_image(pil_image, image_size=768):
    """
    Split image into a 2x2 grid of crops + the full image (thumbnail).

    Each of the 5 resulting PIL Images is sized image_size/2 x image_size/2
    (for the crops) or image_size x image_size (for the thumbnail).  The
    caller's transform will resize all of them to image_size x image_size
    before encoding, so each produces the same number of tokens (256) as a
    base-view image.

    Total tokens after encoding all 5 tiles: 5 x 256 = 1280.
    """
    img = pil_image.resize((image_size, image_size), Image.LANCZOS)
    half = image_size // 2
    return [
        img.crop((0,    0,    half,       half)),        # top-left
        img.crop((half, 0,    image_size, half)),        # top-right
        img.crop((0,    half, half,       image_size)),  # bottom-left
        img.crop((half, half, image_size, image_size)),  # bottom-right
        img,                                             # full thumbnail
    ]


def collect_unique_images(*manifest_paths):
    """Return sorted list of unique image paths across all manifest files."""
    image_paths = set()
    for path in manifest_paths:
        if not Path(path).exists():
            print(f"  Skipping missing manifest: {path}")
            continue
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                image_paths.add(data["image"])
    return sorted(image_paths)


def main():
    parser = argparse.ArgumentParser(description="Pre-compute vision features")
    parser.add_argument("--output_dir", type=str, default="./precomputed_features",
                        help="Directory to save feature .pt files")
    parser.add_argument("--manifest_dir", type=str,
                        default="Data Crawling/output/manifests",
                        help="Directory containing train/val/test .jsonl manifests")
    parser.add_argument("--image_size", type=int, default=768,
                        help="Resize images to this square size before encoding")
    parser.add_argument("--model_name", type=str,
                        default="deepseek-ai/deepseek-ocr-2",
                        help="HuggingFace model ID for DeepSeek-OCR-2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tiling", action="store_true",
                        help="Enable 2x2 tiling + thumbnail (5 views, 1280 tokens vs 256)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Collect unique images from all manifests ----
    manifest_dir = Path(args.manifest_dir)
    manifests = [
        str(manifest_dir / "train.jsonl"),
        str(manifest_dir / "val.jsonl"),
        str(manifest_dir / "test.jsonl"),
    ]
    image_paths = collect_unique_images(*manifests)
    print(f"Found {len(image_paths)} unique images across manifests\n")

    # ---- Load vision encoder ----
    sam, d2e, proj = load_vision_encoder(args.model_name, args.device)
    transform = get_transform(args.image_size)

    # ---- Process all images ----
    mode = "tiling (2x2 + thumbnail, ~1280 tokens)" if args.tiling else f"base ({args.image_size}x{args.image_size}, ~256 tokens)"
    print(f"\nEncoding images — mode: {mode}")
    token_counts = {}
    errors = []
    expected_shape = None

    for i, img_path in enumerate(tqdm(image_paths, desc="Encoding")):
        try:
            image = Image.open(img_path).convert("RGB")

            if args.tiling:
                tiles = tile_image(image, args.image_size)
                tile_features = []
                for tile in tiles:
                    tensor = transform(tile).unsqueeze(0).to(args.device).half()
                    with torch.no_grad():
                        feat = proj(d2e(sam(tensor)))  # [1, 256, 1280]
                    tile_features.append(feat.squeeze(0).cpu())
                features = torch.cat(tile_features, dim=0)  # [1280, 1280]
            else:
                tensor = transform(image).unsqueeze(0).to(args.device).half()
                with torch.no_grad():
                    features = proj(d2e(sam(tensor)))  # [1, num_tokens, 1280]
                features = features.squeeze(0).cpu()

            num_tokens = features.size(0)

            # Track token counts
            token_counts[num_tokens] = token_counts.get(num_tokens, 0) + 1

            # Verify shape on first image
            if i == 0:
                expected_shape = features.shape
                print(f"\n  First image: {Path(img_path).name}")
                print(f"  Output shape: {features.shape}  dtype: {features.dtype}")
                assert features.size(-1) == 1280, \
                    f"Expected 1280D features, got {features.size(-1)}"

            # Warn if shape changes (unexpected)
            if features.shape != expected_shape:
                print(f"\n  WARNING: shape mismatch on {Path(img_path).name}: "
                      f"{features.shape} vs expected {expected_shape}")

            # Save: image stem -> .pt file
            out_name = Path(img_path).stem + ".pt"
            torch.save(features, output_dir / out_name)

        except Exception as e:
            errors.append((img_path, str(e)))
            print(f"\n  ERROR on {Path(img_path).name}: {e}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("PRE-COMPUTATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Processed: {len(image_paths) - len(errors)} / {len(image_paths)}")
    print(f"  Errors:    {len(errors)}")
    print(f"  Token counts per image: {token_counts}")
    print(f"  Feature dtype: {expected_shape}")
    print(f"  Output dir: {output_dir}")
    if errors:
        print(f"\n  Failed images:")
        for path, err in errors[:20]:
            print(f"    {Path(path).name}: {err}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
