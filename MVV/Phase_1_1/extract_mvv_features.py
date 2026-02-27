#!/usr/bin/env python3
"""
extract_mvv_features.py — MVV feature extraction sweep

For each token budget, bicubic-downsamples every image in data_mvv/images/
to the exact SigLIP patch-grid pixel dimensions, passes it through frozen
SigLIP-SO400M, mean-pools the patch tokens, and saves a [1152] fp16 vector.

Token budgets and pixel targets (14px patches, square grid):
  729 tokens → 27×27 grid → 378×378 px   (native near-resolution, readable)
  441 tokens → 21×21 grid → 294×294 px   (transition zone)
  256 tokens → 16×16 grid → 224×224 px   (text unreadable, structure visible)
  121 tokens → 11×11 grid → 154×154 px   (topology floor, sub-symbolic)

Output structure:
  data_mvv/features/
    budget_729/{image_stem}.pt   — [1152] fp16 vector
    budget_441/{image_stem}.pt
    budget_256/{image_stem}.pt
    budget_121/{image_stem}.pt

Idempotent: skips .pt files that already exist.

Usage:
    python MVV/Phase_1_1/extract_mvv_features.py \\
        --data-dir MVV/Phase_1_1/data_mvv \\
        --model    google/siglip-so400m-patch14-384 \\
        --batch-size 32 \\
        --device cuda
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

# ── Token budgets: name → (H, W) pixel target ─────────────────────────────────
BUDGETS = {
    729: (378, 378),
    441: (294, 294),
    256: (224, 224),
    121: (154, 154),
}

# SigLIP-SO400M normalization constants
SIGLIP_MEAN = [0.5, 0.5, 0.5]
SIGLIP_STD  = [0.5, 0.5, 0.5]


def make_transform(h: int, w: int):
    """Bicubic resize to (h, w), convert L→RGB, normalize for SigLIP."""
    return transforms.Compose([
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Grayscale(num_output_channels=3),   # L → RGB (repeat channel)
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
    out_dir: Path,
    h: int, w: int,
    batch_size: int,
    device: str,
):
    """Run one full pass over all images at a single resolution."""
    transform = make_transform(h, w)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter already-done
    todo = [p for p in image_paths
            if not (out_dir / (Path(p).stem + ".pt")).exists()]
    n_skip = len(image_paths) - len(todo)
    if n_skip:
        print(f"  Skipping {n_skip} already extracted")

    errors = []
    for batch_start in tqdm(range(0, len(todo), batch_size),
                            desc=f"  {h}×{w}", unit="batch"):
        batch_paths = todo[batch_start: batch_start + batch_size]
        tensors, stems = [], []

        for p in batch_paths:
            try:
                img = Image.open(p)   # 'L' mode, 800×800
                tensors.append(transform(img))
                stems.append(Path(p).stem)
            except Exception as e:
                if not errors:
                    print(f"\n  FIRST ERROR on {Path(p).name}: {e}")
                errors.append((p, str(e)))

        if not tensors:
            continue

        batch = torch.stack(tensors).to(device).half()   # [B, 3, H, W]

        # SigLIP forward — interpolate_pos_encoding handles non-384 inputs
        outputs = model(
            pixel_values=batch,
            interpolate_pos_encoding=True,
        )

        # last_hidden_state: [B, N_tokens, 1152]
        patch_tokens = outputs.last_hidden_state

        # Mean-pool across tokens → [B, 1152]
        features = patch_tokens.mean(dim=1)   # fp16

        for feat, stem in zip(features, stems):
            torch.save(feat.cpu(), out_dir / f"{stem}.pt")

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   default="MVV/Phase_1_1/data_mvv",
                        help="Root data dir containing images/ and manifest.jsonl")
    parser.add_argument("--model",      default="google/siglip-so400m-patch14-384")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--budgets",    type=int, nargs="+",
                        default=list(BUDGETS.keys()),
                        help="Subset of token budgets to run (default: all four)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    feat_dir = data_dir / "features"

    # Load manifest — resolve paths relative to data_dir/images/ using stem
    # (handles stale absolute/relative paths after directory moves)
    manifest = data_dir / "manifest.jsonl"
    images_dir = data_dir / "images"
    image_paths = []
    with open(manifest) as f:
        for line in f:
            raw = json.loads(line)["image"]
            resolved = images_dir / Path(raw).name
            image_paths.append(str(resolved))
    print(f"Images to process: {len(image_paths)}")
    # Sanity check first path
    first = Path(image_paths[0])
    print(f"  First path: {first}  exists={first.exists()}")

    model = load_model(args.model, args.device)

    all_errors = {}
    for budget in sorted(args.budgets, reverse=True):
        if budget not in BUDGETS:
            print(f"Unknown budget {budget}, skipping")
            continue
        h, w = BUDGETS[budget]
        out_dir = feat_dir / f"budget_{budget}"
        print(f"\n[Budget {budget} tokens — {h}×{w} px → {out_dir.name}/]")
        errs = extract_budget(model, image_paths, out_dir, h, w,
                              args.batch_size, args.device)
        all_errors[budget] = errs

    # Summary
    print("\n" + "=" * 50)
    print("EXTRACTION COMPLETE")
    print("=" * 50)
    for budget in sorted(args.budgets, reverse=True):
        h, w = BUDGETS.get(budget, (0, 0))
        out_dir = feat_dir / f"budget_{budget}"
        n_saved = len(list(out_dir.glob("*.pt"))) if out_dir.exists() else 0
        n_err   = len(all_errors.get(budget, []))
        print(f"  budget_{budget:3d} ({h}×{w}): {n_saved:5,} vectors  "
              f"{f'| {n_err} errors' if n_err else ''}")
    print(f"  Features dir: {feat_dir}")


if __name__ == "__main__":
    main()
