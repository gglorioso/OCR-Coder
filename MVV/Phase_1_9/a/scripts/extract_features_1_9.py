#!/usr/bin/env python3
"""
extract_features_1_9.py — Phase 1.9: raw SigLIP patch feature extraction

Saves per-image [1024, 1152] fp16 tensors (the raw 32×32 patch grid from
SigLIP-SO400M at 448×448 input) — the pre-pool representation needed for
the convolutional compression test in Phase 1.9.

  448×448 → SigLIP → last_hidden_state → [1024, 1152] fp16 → save .pt

Output: MVV/Phase_1_9/data/features/<stem>.pt  — shape [1024, 1152] fp16
Idempotent: skips stems where .pt already exists.

Usage:
    python extract_features_1_9.py [--data-dir ...] [--batch-size 8] [--device cuda]
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import SiglipVisionModel


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parents[2]

MODEL_NAME = "google/siglip-so400m-patch14-384"
IMAGE_SIZE = 448          # 448 / 14 = 32  →  32×32 = 1024 tokens
GRID_SIDE  = IMAGE_SIZE // 14   # 32
N_TOKENS   = GRID_SIDE ** 2     # 1024
FEAT_DIM   = 1152

SIGLIP_MEAN = [0.5, 0.5, 0.5]
SIGLIP_STD  = [0.5, 0.5, 0.5]


def make_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=SIGLIP_MEAN, std=SIGLIP_STD),
    ])


def load_model(device: str):
    print(f"Loading SigLIP from '{MODEL_NAME}' …")
    model = SiglipVisionModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model = model.to(device).eval()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded — {n:.0f}M params, fp16, device={device}")
    return model


@torch.no_grad()
def extract(model, image_paths, out_dir: Path, batch_size: int, device: str):
    transform = make_transform()
    out_dir.mkdir(parents=True, exist_ok=True)

    todo, n_skip = [], 0
    for p in image_paths:
        stem = Path(p).stem
        if (out_dir / f"{stem}.pt").exists():
            n_skip += 1
        else:
            todo.append(p)
    if n_skip:
        print(f"  Skipping {n_skip} already-extracted stems")

    print(f"  Extracting {len(todo):,} images  (grid {GRID_SIDE}×{GRID_SIDE}={N_TOKENS} tokens)")

    errors = []
    for batch_start in tqdm(range(0, len(todo), batch_size), unit="batch"):
        batch_paths = todo[batch_start: batch_start + batch_size]
        tensors, stems = [], []
        for p in batch_paths:
            try:
                img = Image.open(p)
                tensors.append(transform(img))
                stems.append(Path(p).stem)
            except Exception as e:
                errors.append((str(p), str(e)))

        if not tensors:
            continue

        batch = torch.stack(tensors).to(device).half()          # [B, 3, 448, 448]
        outputs = model(pixel_values=batch, interpolate_pos_encoding=True)
        patch_tokens = outputs.last_hidden_state                 # [B, 1024, 1152] fp16

        for feat, stem in zip(patch_tokens, stems):
            torch.save(feat.cpu(), out_dir / f"{stem}.pt")      # [1024, 1152] fp16

    return errors, n_skip


def main():
    p = argparse.ArgumentParser(description="Phase 1.9 raw feature extraction")
    p.add_argument("--data-dir",   default=str(_REPO_ROOT / "MVV" / "Phase_1_1" / "data_mvv"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device",     default="cuda")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = _REPO_ROOT / "MVV" / "Phase_1_9" / "data" / "features"

    print("=" * 65)
    print("MVV Phase 1.9 — Raw SigLIP Feature Extraction")
    print(f"  Input:  {data_dir}")
    print(f"  Output: {out_dir}")
    print(f"  Device: {args.device}  |  Batch: {args.batch_size}")
    print("=" * 65)

    manifest = data_dir / "manifest.jsonl"
    images_dir = data_dir / "images"
    image_paths = []
    with open(manifest) as f:
        for line in f:
            raw = json.loads(line)["image"]
            image_paths.append(str(images_dir / Path(raw).name))

    print(f"\nImages in manifest: {len(image_paths):,}")

    model = load_model(args.device)
    errors, n_skip = extract(model, image_paths, out_dir, args.batch_size, args.device)

    n_out = len(list(out_dir.glob("*.pt")))
    print("\n" + "=" * 65)
    print("DONE")
    print(f"  Features written : {n_out:,}  (shape [1024, 1152] fp16)")
    print(f"  Skipped          : {n_skip:,}")
    if errors:
        print(f"  Errors           : {len(errors)}")
    print(f"  Output dir       : {out_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
