#!/usr/bin/env python3
"""
extract_maxpool_features.py — Adaptive max-pool feature extraction for MVV Phase 1.1

Instead of mean-pooling the SigLIP patch tokens to a single [1152] vector,
this script preserves spatial structure by:
  1. Extracting raw patch tokens  [B, N, 1152]  from SigLIP
  2. Reshaping to the native grid [B, 1152, √N, √N]
  3. Applying adaptive_max_pool2d to two fixed output sizes: 4×4 and 8×8
  4. Flattening → [B, 18432] and [B, 73728]  (both fp16)

Both pool sizes are extracted in a single SigLIP forward pass per budget.

Token budgets → grids:
  729 tokens → 27×27 grid → 378×378 px
  441 tokens → 21×21 grid → 294×294 px
  256 tokens → 16×16 grid → 224×224 px
  121 tokens → 11×11 grid → 154×154 px

Output structure:
  data_mvv/features_maxpool/
    pool4x4/budget_729/{stem}.pt   — [18432] fp16
    pool4x4/budget_441/{stem}.pt
    ...
    pool8x8/budget_729/{stem}.pt   — [73728] fp16
    pool8x8/budget_441/{stem}.pt
    ...

Idempotent: skips stems where both pool4x4 and pool8x8 .pt files already exist.

Usage:
    python MVV/Phase_1_1/extract_maxpool_features.py \\
        --data-dir  MVV/Phase_1_1/data_mvv \\
        --model     google/siglip-so400m-patch14-384 \\
        --batch-size 32 \\
        --device    cuda
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


BUDGETS = {
    729: (378, 378),
    441: (294, 294),
    256: (224, 224),
    121: (154, 154),
}

POOL_SIZES = [4, 8]          # output grid side lengths
SIGLIP_MEAN = [0.5, 0.5, 0.5]
SIGLIP_STD  = [0.5, 0.5, 0.5]


def make_transform(h: int, w: int):
    return transforms.Compose([
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=SIGLIP_MEAN, std=SIGLIP_STD),
    ])


def load_model(model_name: str, device: str):
    print(f"Loading SigLIP from '{model_name}' ...")
    model = SiglipVisionModel.from_pretrained(model_name, torch_dtype=torch.float16)
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded — {n_params:.0f}M params, dtype=fp16, device={device}")
    return model


@torch.no_grad()
def extract_budget(
    model,
    image_paths: list,
    out_root: Path,
    budget: int,
    h: int, w: int,
    batch_size: int,
    device: str,
):
    """
    One full pass over all images at a single resolution.
    For each batch, extracts patch tokens, reshapes to spatial grid,
    applies adaptive max pool at each pool size, and saves both outputs.
    """
    transform = make_transform(h, w)
    g = int(budget ** 0.5)      # grid side length (27, 21, 16, or 11)

    # Output dirs: one per pool size
    out_dirs = {}
    for ps in POOL_SIZES:
        d = out_root / f"pool{ps}x{ps}" / f"budget_{budget}"
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[ps] = d

    # Skip images where ALL pool sizes are already done
    todo = []
    n_skip = 0
    for p in image_paths:
        stem = Path(p).stem
        if all((out_dirs[ps] / f"{stem}.pt").exists() for ps in POOL_SIZES):
            n_skip += 1
        else:
            todo.append(p)
    if n_skip:
        print(f"  Skipping {n_skip} already extracted")

    errors = []
    for batch_start in tqdm(range(0, len(todo), batch_size),
                            desc=f"  {h}×{w} (grid {g}×{g})", unit="batch"):
        batch_paths = todo[batch_start: batch_start + batch_size]
        tensors, stems = [], []

        for p in batch_paths:
            try:
                img = Image.open(p)
                tensors.append(transform(img))
                stems.append(Path(p).stem)
            except Exception as e:
                errors.append((p, str(e)))
                if len(errors) == 1:
                    print(f"\n  FIRST ERROR on {Path(p).name}: {e}")

        if not tensors:
            continue

        batch = torch.stack(tensors).to(device).half()   # [B, 3, H, W]

        outputs = model(
            pixel_values=batch,
            interpolate_pos_encoding=True,
        )

        # patch_tokens: [B, N, 1152]  — N = budget (e.g. 729)
        patch_tokens = outputs.last_hidden_state
        B, N, C = patch_tokens.shape

        # Reshape to spatial grid: [B, 1152, g, g]  (channels-first for pool)
        spatial = patch_tokens.reshape(B, g, g, C).permute(0, 3, 1, 2).contiguous()

        for ps in POOL_SIZES:
            # Adaptive max pool → [B, 1152, ps, ps]
            pooled = F.adaptive_max_pool2d(spatial.float(), (ps, ps))
            flat   = pooled.half().flatten(1)   # [B, 1152 * ps * ps]

            for feat, stem in zip(flat, stems):
                torch.save(feat.cpu(), out_dirs[ps] / f"{stem}.pt")

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   default="MVV/Phase_1_1/data_mvv")
    parser.add_argument("--model",      default="google/siglip-so400m-patch14-384")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--budgets",    type=int, nargs="+",
                        default=list(BUDGETS.keys()))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_root = data_dir / "features_maxpool"

    # Load image list from manifest
    manifest   = data_dir / "manifest.jsonl"
    images_dir = data_dir / "images"
    image_paths = []
    with open(manifest) as f:
        for line in f:
            raw      = json.loads(line)["image"]
            resolved = images_dir / Path(raw).name
            image_paths.append(str(resolved))
    print(f"Images to process: {len(image_paths)}")
    first = Path(image_paths[0])
    print(f"  First path: {first}  exists={first.exists()}")
    print(f"  Pool sizes: {[f'{ps}×{ps} → {1152*ps*ps:,}d' for ps in POOL_SIZES]}")

    model = load_model(args.model, args.device)

    all_errors = {}
    for budget in sorted(args.budgets, reverse=True):
        if budget not in BUDGETS:
            print(f"Unknown budget {budget}, skipping")
            continue
        h, w = BUDGETS[budget]
        print(f"\n[Budget {budget} tokens — {h}×{w} px — grid {int(budget**0.5)}×{int(budget**0.5)}]")
        errs = extract_budget(model, image_paths, out_root, budget,
                              h, w, args.batch_size, args.device)
        all_errors[budget] = errs

    # Summary
    print("\n" + "=" * 55)
    print("EXTRACTION COMPLETE")
    print("=" * 55)
    for budget in sorted(args.budgets, reverse=True):
        h, w = BUDGETS.get(budget, (0, 0))
        for ps in POOL_SIZES:
            d = out_root / f"pool{ps}x{ps}" / f"budget_{budget}"
            n = len(list(d.glob("*.pt"))) if d.exists() else 0
            dim = 1152 * ps * ps
            print(f"  pool{ps}x{ps}/budget_{budget:3d} ({h}×{w}): {n:5,} vectors  [{dim:,}d]")
    n_errs = sum(len(v) for v in all_errors.values())
    if n_errs:
        print(f"\n  Total errors: {n_errs}")
    print(f"  Output root: {out_root}")


if __name__ == "__main__":
    main()
