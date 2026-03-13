#!/usr/bin/env python3
"""
extract_features_1_7.py — GPU feature extraction for MVV Phase 1.7

Extracts Method 2 pool8x8 features from enhanced-rendered code images.

  Method 2 (Native + Spatial Pool):
    448×448 → SigLIP → 32×32 grid (1024 tokens)
    → avg_pool2d(kernel=2, stride=2) → 16×16 (256 tokens)
    → adaptive_max_pool2d(8×8) → flatten → save fp16 [73728]

Only pool8x8 is extracted (pool4x4 not needed for Phase 1.7).
Idempotent: skips stems where out_dir/STEM.pt already exists.

Usage:
    python extract_features_1_7.py \\
        --image-dir  MVV/Phase_1_7/images/exp_A_syntax_only \\
        --out-dir    MVV/Phase_1_7/data/features/exp_A/pool8x8 \\
        --device     cuda \\
        --batch-size 8
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import SiglipVisionModel


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "google/siglip-so400m-patch14-384"
IMAGE_SIZE = 448                 # 448/14 = 32 → 32×32 = 1024 tokens
GRID_SIDE  = IMAGE_SIZE // 14   # 32
N_TOKENS   = GRID_SIDE ** 2     # 1024
FEAT_DIM   = 1152               # SigLIP-SO400M hidden size
POOL_SIZE  = 8                  # pool8x8 only

SIGLIP_MEAN = [0.5, 0.5, 0.5]
SIGLIP_STD  = [0.5, 0.5, 0.5]


# ---------------------------------------------------------------------------
# Image transform
# ---------------------------------------------------------------------------

def make_transform(size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=SIGLIP_MEAN, std=SIGLIP_STD),
    ])


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: str):
    print(f"Loading SigLIP from '{model_name}' ...")
    model = SiglipVisionModel.from_pretrained(model_name, torch_dtype=torch.float16)
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded — {n_params:.0f}M params, dtype=fp16, device={device}")
    return model


# ---------------------------------------------------------------------------
# Extraction function
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_method2_pool8x8(patch_tokens: torch.Tensor) -> torch.Tensor:
    """
    Method 2: avg_pool 32×32 → 16×16, then adaptive_max_pool to 8×8.

    patch_tokens: [B, 1024, 1152] fp16
    Returns: flat fp16 tensor [B, 1152*8*8] = [B, 73728]
    """
    B, N, C = patch_tokens.shape
    assert N == N_TOKENS, f"Expected {N_TOKENS} tokens, got {N}"
    assert C == FEAT_DIM,  f"Expected {FEAT_DIM} dims, got {C}"

    # Reshape to spatial grid; cast to float for pooling numerics
    spatial = patch_tokens.reshape(B, GRID_SIDE, GRID_SIDE, C).permute(0, 3, 1, 2).float()
    # [B, 1152, 32, 32]

    # Compress 32×32 → 16×16 via avg_pool
    compressed = F.avg_pool2d(spatial, kernel_size=2, stride=2)
    # [B, 1152, 16, 16]

    # Pool to 8×8
    pooled = F.adaptive_max_pool2d(compressed, (POOL_SIZE, POOL_SIZE))
    # [B, 1152, 8, 8]

    flat = pooled.half().flatten(1)
    # [B, 73728]
    return flat


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_all(
    model,
    image_paths: list,
    out_dir: Path,
    batch_size: int,
    device: str,
):
    transform = make_transform(IMAGE_SIZE)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Idempotency check: skip stems where output already exists
    todo = []
    n_skip = 0
    for p in image_paths:
        stem    = Path(p).stem
        out_pt  = out_dir / f"{stem}.pt"
        if out_pt.exists():
            n_skip += 1
        else:
            todo.append(p)

    if n_skip:
        print(f"  Skipping {n_skip:,} already-extracted stems")

    print(f"  Processing {len(todo):,} images  "
          f"(IMAGE_SIZE={IMAGE_SIZE}, grid={GRID_SIDE}x{GRID_SIDE}={N_TOKENS} tokens, "
          f"FEAT_DIM={FEAT_DIM})")
    print(f"  Output dim: pool8x8 → {FEAT_DIM * POOL_SIZE * POOL_SIZE:,}d fp16")

    errors = []
    pbar = tqdm(range(0, len(todo), batch_size),
                desc=f"  {IMAGE_SIZE}x{IMAGE_SIZE} pool8x8",
                unit="batch")

    for batch_start in pbar:
        batch_paths = todo[batch_start: batch_start + batch_size]
        tensors, stems = [], []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(transform(img))
                stems.append(Path(p).stem)
            except Exception as e:
                errors.append((str(p), str(e)))
                if len(errors) <= 3:
                    print(f"\n  ERROR loading {Path(p).name}: {e}")

        if not tensors:
            continue

        batch = torch.stack(tensors).to(device).half()  # [B, 3, 448, 448]

        outputs = model(
            pixel_values=batch,
            interpolate_pos_encoding=True,
        )
        patch_tokens = outputs.last_hidden_state  # [B, 1024, 1152]

        flat = extract_method2_pool8x8(patch_tokens)  # [B, 73728]

        for feat, stem in zip(flat, stems):
            torch.save(feat.cpu(), out_dir / f"{stem}.pt")

    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1.7: extract Method 2 pool8x8 features from enhanced images"
    )
    parser.add_argument("--image-dir",  required=True,
                        help="Directory containing .png images to process")
    parser.add_argument("--out-dir",    required=True,
                        help="Directory to save .pt feature files (flat, no subdirs)")
    parser.add_argument("--model",      default=MODEL_NAME,
                        help="HuggingFace model ID for SigLIP")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Images per GPU batch (default: 8)")
    parser.add_argument("--device",     default="cuda",
                        help="Torch device (cuda or cpu)")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    out_dir   = Path(args.out_dir)

    print("=" * 65)
    print("MVV Phase 1.7 — Feature Extraction (Method 2, pool8x8)")
    print(f"  Image dir:  {image_dir}")
    print(f"  Output dir: {out_dir}")
    print(f"  Model:      {args.model}")
    print(f"  Device:     {args.device}")
    print(f"  Batch size: {args.batch_size}")
    print("=" * 65)

    if not image_dir.exists():
        raise FileNotFoundError(f"--image-dir not found: {image_dir}")

    image_paths = sorted(str(p) for p in image_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No .png files found in: {image_dir}")

    print(f"\nImages found: {len(image_paths):,}")

    model = load_model(args.model, args.device)

    errors = extract_all(
        model=model,
        image_paths=image_paths,
        out_dir=out_dir,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Summary
    n_saved = len(list(out_dir.glob("*.pt")))
    print("\n" + "=" * 65)
    print("EXTRACTION COMPLETE")
    print("=" * 65)
    print(f"  pool8x8: {n_saved:,} vectors  [{FEAT_DIM * POOL_SIZE * POOL_SIZE:,}d fp16]")
    if errors:
        print(f"\n  Total errors: {len(errors)}")
        for p, e in errors[:10]:
            print(f"    {Path(p).name}: {e}")
    print(f"\n  Output dir: {out_dir}")


if __name__ == "__main__":
    main()
