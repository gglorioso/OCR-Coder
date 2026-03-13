#!/usr/bin/env python3
"""
extract_features_1_5.py — GPU feature extraction for MVV Phase 1.5

Extracts two 256-token compression strategies from 448×448 SigLIP features:

  Method 2 (Native + Spatial Pool):
    448×448 → SigLIP → 32×32 grid (1024 tokens)
    → avg_pool2d(kernel=2, stride=2) → 16×16 (256 tokens)
    → adaptive_max_pool2d(4×4 or 8×8) → flatten → save fp16

  Method 3 (Token Pruning / Zero-Out):
    448×448 → SigLIP → 32×32 grid (1024 tokens)
    → per-token variance across 1152 dims → zero-out 768 lowest-variance tokens
    → avg_pool2d(kernel=2, stride=2) → 16×16 (256 tokens)
    → adaptive_max_pool2d(4×4 or 8×8) → flatten → save fp16

Method 1 features already exist — this script does NOT re-extract them.

Output structure:
  MVV/Phase_1_5/data/features/
    method2/
      pool4x4/   # [18432] fp16
      pool8x8/   # [73728] fp16
    method3/
      pool4x4/   # [18432] fp16
      pool8x8/   # [73728] fp16

Idempotent: skips stems where both pool4x4 and pool8x8 .pt files already exist
for a given method.

Usage:
    python extract_features_1_5.py \\
        --data-dir  MVV/Phase_1_1/data_mvv \\
        --device    cuda \\
        --batch-size 16
"""

import argparse
import json
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

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parents[3]          # OCR-Coder/

MODEL_NAME  = "google/siglip-so400m-patch14-384"
IMAGE_SIZE  = 448                # 448/14 = 32 → 32×32 = 1024 tokens
GRID_SIDE   = IMAGE_SIZE // 14   # 32
N_TOKENS    = GRID_SIDE ** 2     # 1024
N_KEEP      = 256                # tokens to keep for method3 (256 of 1024)
N_ZERO      = N_TOKENS - N_KEEP  # 768 tokens zeroed out
FEAT_DIM    = 1152               # SigLIP-SO400M hidden size
POOL_SIZES  = [4, 8]

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
    print(f"Loading SigLIP from '{model_name}' …")
    model = SiglipVisionModel.from_pretrained(model_name, torch_dtype=torch.float16)
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded — {n_params:.0f}M params, dtype=fp16, device={device}")
    return model


# ---------------------------------------------------------------------------
# Per-method extraction functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_method2(patch_tokens: torch.Tensor, device: str) -> dict:
    """
    Method 2: avg_pool 32×32 → 16×16, then adaptive_max_pool to each pool size.

    patch_tokens: [B, 1024, 1152] fp16 on device
    Returns: dict {ps: flat_fp16_tensor [B, 1152*ps*ps]}
    """
    B, N, C = patch_tokens.shape
    assert N == N_TOKENS, f"Expected {N_TOKENS} tokens, got {N}"
    assert C == FEAT_DIM, f"Expected {FEAT_DIM} dims, got {C}"

    # Reshape to spatial grid and cast to float for pooling numerics
    spatial = patch_tokens.reshape(B, GRID_SIDE, GRID_SIDE, C).permute(0, 3, 1, 2).float()
    # [B, 1152, 32, 32]

    # Compress 32×32 → 16×16 via avg_pool
    compressed = F.avg_pool2d(spatial, kernel_size=2, stride=2)
    # [B, 1152, 16, 16]

    result = {}
    for ps in POOL_SIZES:
        pooled = F.adaptive_max_pool2d(compressed, (ps, ps))  # [B, 1152, ps, ps]
        flat   = pooled.half().flatten(1)                      # [B, 1152*ps*ps]
        result[ps] = flat
    return result


@torch.no_grad()
def extract_method3(patch_tokens: torch.Tensor, device: str) -> dict:
    """
    Method 3: zero-out 768 lowest-variance tokens, then avg_pool 32×32 → 16×16,
    then adaptive_max_pool to each pool size.

    patch_tokens: [B, 1024, 1152] fp16 on device
    Returns: dict {ps: flat_fp16_tensor [B, 1152*ps*ps]}
    """
    B, N, C = patch_tokens.shape
    assert N == N_TOKENS, f"Expected {N_TOKENS} tokens, got {N}"
    assert C == FEAT_DIM, f"Expected {FEAT_DIM} dims, got {C}"

    # Cast to float for variance computation
    tokens_f = patch_tokens.float()  # [B, 1024, 1152]

    # Per-token variance across feature dim: [B, 1024]
    token_var = tokens_f.var(dim=2)

    # Keep top-256 highest-variance tokens per sample; zero out the rest
    topk_indices = token_var.topk(N_KEEP, dim=1).indices  # [B, 256]
    mask = torch.zeros(B, N_TOKENS, device=device, dtype=torch.float32)
    mask.scatter_(1, topk_indices, 1.0)  # 1 = keep, 0 = zero-out

    # Apply mask: zero out 768 lowest-variance tokens
    masked = tokens_f * mask.unsqueeze(2)  # [B, 1024, 1152]

    # Reshape to spatial grid — CRITICAL: zero-out BEFORE reshape so spatial
    # adjacency is preserved for avg_pool to work correctly
    spatial = masked.reshape(B, GRID_SIDE, GRID_SIDE, C).permute(0, 3, 1, 2)
    # [B, 1152, 32, 32]

    # Compress 32×32 → 16×16 via avg_pool
    compressed = F.avg_pool2d(spatial, kernel_size=2, stride=2)
    # [B, 1152, 16, 16]

    result = {}
    for ps in POOL_SIZES:
        pooled = F.adaptive_max_pool2d(compressed, (ps, ps))  # [B, 1152, ps, ps]
        flat   = pooled.half().flatten(1)                      # [B, 1152*ps*ps]
        result[ps] = flat
    return result


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_all_methods(
    model,
    image_paths: list,
    out_root: Path,
    batch_size: int,
    device: str,
):
    transform = make_transform(IMAGE_SIZE)

    # Output dirs for each method × pool size
    out_dirs = {
        "method2": {ps: out_root / "method2" / f"pool{ps}x{ps}" for ps in POOL_SIZES},
        "method3": {ps: out_root / "method3" / f"pool{ps}x{ps}" for ps in POOL_SIZES},
    }
    for method_dirs in out_dirs.values():
        for d in method_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    # Skip images where ALL outputs already exist (both methods, both pool sizes)
    todo = []
    n_skip = 0
    for p in image_paths:
        stem = Path(p).stem
        all_done = all(
            (out_dirs[m][ps] / f"{stem}.pt").exists()
            for m in ["method2", "method3"]
            for ps in POOL_SIZES
        )
        if all_done:
            n_skip += 1
        else:
            todo.append(p)
    if n_skip:
        print(f"  Skipping {n_skip} already-extracted stems (all 4 outputs exist)")

    print(f"  Processing {len(todo):,} stems  (IMAGE_SIZE={IMAGE_SIZE}, "
          f"grid={GRID_SIDE}×{GRID_SIDE}={N_TOKENS} tokens, FEAT_DIM={FEAT_DIM})")
    print(f"  Output dims: pool4x4 → {FEAT_DIM*4*4:,}d  |  pool8x8 → {FEAT_DIM*8*8:,}d")

    errors = []
    pbar = tqdm(range(0, len(todo), batch_size),
                desc=f"  {IMAGE_SIZE}×{IMAGE_SIZE} (grid {GRID_SIDE}×{GRID_SIDE})",
                unit="batch")

    for batch_start in pbar:
        batch_paths = todo[batch_start: batch_start + batch_size]
        tensors, stems = [], []

        for p in batch_paths:
            try:
                img = Image.open(p)
                tensors.append(transform(img))
                stems.append(Path(p).stem)
            except Exception as e:
                errors.append((str(p), str(e)))
                if len(errors) <= 3:
                    print(f"\n  ERROR on {Path(p).name}: {e}")

        if not tensors:
            continue

        batch = torch.stack(tensors).to(device).half()  # [B, 3, 448, 448]

        outputs = model(
            pixel_values=batch,
            interpolate_pos_encoding=True,
        )
        # patch_tokens: [B, 1024, 1152]
        patch_tokens = outputs.last_hidden_state

        # Method 2
        m2_feats = extract_method2(patch_tokens, device)
        for ps in POOL_SIZES:
            for feat, stem in zip(m2_feats[ps], stems):
                torch.save(feat.cpu(), out_dirs["method2"][ps] / f"{stem}.pt")

        # Method 3
        m3_feats = extract_method3(patch_tokens, device)
        for ps in POOL_SIZES:
            for feat, stem in zip(m3_feats[ps], stems):
                torch.save(feat.cpu(), out_dirs["method3"][ps] / f"{stem}.pt")

    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1.5: extract Method 2 and Method 3 features from 448×448 SigLIP"
    )
    parser.add_argument("--data-dir",   default=str(_REPO_ROOT / "MVV" / "Phase_1_1" / "data_mvv"),
                        help="Directory containing manifest.jsonl and images/")
    parser.add_argument("--model",      default=MODEL_NAME,
                        help="HuggingFace model ID for SigLIP")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Images per GPU batch (448px images; 16 recommended)")
    parser.add_argument("--device",     default="cuda",
                        help="Torch device (cuda or cpu)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_root = _REPO_ROOT / "MVV" / "Phase_1_5" / "data" / "features"

    print("=" * 65)
    print("MVV Phase 1.5 — Feature Extraction (Methods 2 & 3)")
    print(f"  Input:      {data_dir}")
    print(f"  Output:     {out_root}")
    print(f"  Model:      {args.model}")
    print(f"  Device:     {args.device}")
    print(f"  Batch size: {args.batch_size}")
    print("=" * 65)

    # Load image list from manifest
    manifest   = data_dir / "manifest.jsonl"
    images_dir = data_dir / "images"
    image_paths = []
    with open(manifest) as f:
        for line in f:
            raw      = json.loads(line)["image"]
            resolved = images_dir / Path(raw).name
            image_paths.append(str(resolved))

    print(f"\nImages to process: {len(image_paths):,}")
    first = Path(image_paths[0])
    print(f"  First path: {first}  exists={first.exists()}")

    model = load_model(args.model, args.device)

    errors = extract_all_methods(
        model=model,
        image_paths=image_paths,
        out_root=out_root,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Summary
    print("\n" + "=" * 65)
    print("EXTRACTION COMPLETE")
    print("=" * 65)
    for method in ["method2", "method3"]:
        for ps in POOL_SIZES:
            d = out_root / method / f"pool{ps}x{ps}"
            n = len(list(d.glob("*.pt"))) if d.exists() else 0
            dim = FEAT_DIM * ps * ps
            print(f"  {method}/pool{ps}x{ps}: {n:5,} vectors  [{dim:,}d fp16]")
    if errors:
        print(f"\n  Total errors: {len(errors)}")
        for p, e in errors[:10]:
            print(f"    {Path(p).name}: {e}")
    print(f"\n  Output root: {out_root}")


if __name__ == "__main__":
    main()
