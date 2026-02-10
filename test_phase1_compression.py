#!/usr/bin/env python3
"""
test_phase1_compression.py — Phase 1: Vision Token Compression Scaling

Tests whether DeepSeek's vision encoder compresses code images efficiently
at scale (22 - 2600+ line files).

KEY INSIGHT from model source (modeling_deepseekocr2.py):
  - Vision tokens = BASE (256, from global 1024×1024 view)
                  + PATCHES (N × 144, from 768×768 tile crops)
  - Max patches = 6 (dynamic_preprocess max_num=6)
  - So max visual tokens = 256 + 6×144 = 1120, REGARDLESS of file size
  - This means compression ratio IMPROVES with larger files (more text
    tokens, capped visual tokens)

This script:
  1. Takes Python files of varying sizes (our examples + stdlib)
  2. Converts each to a syntax-highlighted image
  3. Calculates EXACT visual token count from image dimensions
     (replicates the model's dynamic_preprocess tiling logic)
  4. Runs vision encoder ONLY (no text generation — that was the bottleneck)
  5. Counts text tokens with the model's tokenizer
  6. Prints compression scaling report

Runtime: ~2-3 minutes (vs 20+ min with full text generation)

Usage:
    sbatch test_phase1_compression.sh
"""

import sys
import os
import math
import time
import torch
from pathlib import Path
from PIL import Image as PILImage, ImageOps

# Project root
PROJECT = Path(__file__).parent

# ── Replicate model's tiling logic (from modeling_deepseekocr2.py) ─────

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Exact copy from modeling_deepseekocr2.py line 156"""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def calculate_visual_tokens(image_width, image_height, base_size=1024, image_size=768,
                            min_num=2, max_num=6):
    """
    Calculate visual token count from image dimensions.
    Replicates the exact logic from model.infer() lines 763-826.

    Returns:
        dict with base_tokens, patch_count, patch_tokens, total_visual_tokens,
        crop_ratio, and aspect_ratio_match
    """
    aspect_ratio = image_width / image_height
    # Aspect ratio factor used for valid_img_tokens estimate
    ratio = 1 - ((max(image_width, image_height) - min(image_width, image_height))
                  / max(image_width, image_height))

    # Determine if cropping is needed
    if image_width <= image_size and image_height <= image_size:
        # Small image: no crops, just global view
        crop_ratio = (1, 1)
        num_patches = 0
    else:
        # Dynamic tiling: find best aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1)
            for i in range(1, n + 1) for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        crop_ratio = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, image_width, image_height, image_size
        )
        w_crops, h_crops = crop_ratio
        if w_crops > 1 or h_crops > 1:
            num_patches = w_crops * h_crops
        else:
            num_patches = 0

    # Token counts
    base_tokens = 256  # global view at base_size=1024
    patch_tokens_each = 144  # each 768×768 patch
    total_patch_tokens = num_patches * patch_tokens_each

    # The actual image tokens placed in the sequence:
    # From lines 819-825: base tokens = num_queries_base^2 + 1
    # Plus patch tokens = (num_queries * w_crops) * (num_queries * h_crops)
    patch_size = 16
    downsample_ratio = 4
    num_queries = math.ceil((image_size // patch_size) / downsample_ratio)  # 12
    num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)  # 16

    seq_base_tokens = num_queries_base * num_queries_base + 1  # 257
    if num_patches > 0:
        w_crops, h_crops = crop_ratio
        seq_patch_tokens = (num_queries * w_crops) * (num_queries * h_crops)
    else:
        seq_patch_tokens = 0
    seq_total = seq_base_tokens + seq_patch_tokens

    return {
        "base_tokens": base_tokens,
        "patch_count": num_patches,
        "patch_tokens": total_patch_tokens,
        "total_visual_tokens": base_tokens + total_patch_tokens,
        "crop_ratio": crop_ratio,
        "aspect_ratio": aspect_ratio,
        "seq_image_tokens": seq_total,  # actual tokens in model sequence
        "valid_img_tokens_estimate": int(base_tokens * ratio) + num_patches * patch_tokens_each,
    }


# ── Main ───────────────────────────────────────────────────────────────

print("=" * 70)
print("  Phase 1: Vision Token Compression Scaling Test (v2 — fast)")
print("=" * 70)
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("(No GPU — running token counting only, no vision encoder validation)")
print()

# ── Step 1: Collect test files ─────────────────────────────────────────

print("[1/4] Collecting test files...")

CONDA_LIB = PROJECT / "envs" / "deepseek-ocr" / "lib" / "python3.10"

test_files = []

# Our small examples
for f in sorted((PROJECT / "examples").glob("*.py")):
    test_files.append((f, f"examples/{f.name}"))

# Python stdlib — varying sizes
stdlib_files = [
    ("json/encoder.py", "stdlib/json_encoder.py"),
    ("collections/__init__.py", "stdlib/collections.py"),
    ("pathlib.py", "stdlib/pathlib.py"),
    ("http/client.py", "stdlib/http_client.py"),
    ("ast.py", "stdlib/ast.py"),
    ("argparse.py", "stdlib/argparse.py"),
    ("typing.py", "stdlib/typing.py"),
]

for rel, label in stdlib_files:
    path = CONDA_LIB / rel
    if path.exists():
        test_files.append((path, label))
    else:
        print(f"  [SKIP] {path} not found")

print(f"  Found {len(test_files)} files:")
for path, label in test_files:
    code = path.read_text()
    lines = code.count("\n") + 1
    print(f"    {label:<35s} {lines:>5d} lines  {len(code):>7d} chars")
print()

# ── Step 2: Convert to images ─────────────────────────────────────────

print("[2/4] Converting to syntax-highlighted images (no line numbers)...")

from code_to_image import convert_code_to_image

phase1_dir = PROJECT / "code_images" / "phase1"
phase1_dir.mkdir(parents=True, exist_ok=True)

image_data = []  # list of (label, src_path, img_path, width, height)
for path, label in test_files:
    try:
        img_path = convert_code_to_image(
            str(path),
            output_dir=str(phase1_dir),
            style="default",
            font_size=14,
            line_numbers=False,
        )
        img = PILImage.open(img_path)
        size_kb = Path(img_path).stat().st_size / 1024
        image_data.append((label, path, img_path, img.width, img.height))
        print(f"  {label:<35s} → {img.width:>4d}×{img.height:<5d}  {size_kb:>6.0f} KB")
    except Exception as e:
        print(f"  [ERROR] {label}: {e}")
print()

# ── Step 3: Calculate token counts ────────────────────────────────────

print("[3/4] Calculating visual token counts + text token counts...")
print()

# Load tokenizer (fast, no GPU needed)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-OCR-2", trust_remote_code=True)

results = []
for label, src_path, img_path, w, h in image_data:
    code = src_path.read_text()
    lines = code.count("\n") + 1
    chars = len(code)

    # Text tokens (using model's tokenizer — exact count)
    text_tokens = len(tokenizer.encode(code))

    # Visual tokens (from image dimensions — exact same logic as model)
    vt = calculate_visual_tokens(w, h)

    compression = text_tokens / vt["total_visual_tokens"] if vt["total_visual_tokens"] > 0 else 0

    result = {
        "label": label,
        "lines": lines,
        "chars": chars,
        "img_width": w,
        "img_height": h,
        "text_tokens": text_tokens,
        **vt,
        "compression_ratio": compression,
    }
    results.append(result)

    print(f"  {label}")
    print(f"    Source:  {lines:>5d} lines, {chars:>6d} chars")
    print(f"    Image:   {w}×{h}  (crop: {vt['crop_ratio'][0]}×{vt['crop_ratio'][1]} = {vt['patch_count']} patches)")
    print(f"    Text tokens:    {text_tokens:>6d}")
    print(f"    Visual tokens:  {vt['total_visual_tokens']:>6d}  (base={vt['base_tokens']} + {vt['patch_count']}×144={vt['patch_tokens']})")
    print(f"    Compression:    {compression:>6.2f}x")
    print()

# ── Step 4: Run vision encoder for validation (if GPU available) ──────

if torch.cuda.is_available():
    print("[4/4] Validating with actual vision encoder on GPU...")
    print("      (Running encoder only — NO text generation)")
    print()

    from transformers import AutoModel
    from torchvision import transforms

    model_name = "deepseek-ai/DeepSeek-OCR-2"
    t_load = time.time()

    try:
        model = AutoModel.from_pretrained(
            model_name,
            _attn_implementation="flash_attention_2",
            trust_remote_code=True,
            use_safetensors=True,
        )
        print(f"  Model loaded (Flash Attention 2) in {time.time() - t_load:.1f}s")
    except Exception:
        model = AutoModel.from_pretrained(
            model_name,
            _attn_implementation="eager",
            trust_remote_code=True,
            use_safetensors=True,
        )
        print(f"  Model loaded (eager attention) in {time.time() - t_load:.1f}s")

    model = model.eval().cuda().to(torch.bfloat16)

    # Access the inner model's vision components
    inner = model.model  # DeepseekOCR2Model
    sam = inner.sam_model
    qwen2 = inner.qwen2_model
    projector = inner.projector

    # Image transform (same as model uses)
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    from modeling_deepseekocr2_utils import dynamic_preprocess_local
    # ^ we'll define this inline since we can't import from cache

    print()
    print("  Validating vision encoder outputs for each file:")
    print()

    for i, (label, src_path, img_path, w, h) in enumerate(image_data):
        image = PILImage.open(img_path).convert("RGB")

        # Replicate model's preprocessing
        base_size = 1024
        image_size = 768

        # Global view
        global_view = ImageOps.pad(
            image, (base_size, base_size),
            color=(128, 128, 128)  # 0.5*255
        )
        global_tensor = image_transform(global_view).to(torch.bfloat16).unsqueeze(0).cuda()

        # Crop patches
        crop_tensors = None
        if w > image_size or h > image_size:
            from modeling_deepseekocr2_utils import dynamic_preprocess as _dp
        # Use inline dynamic_preprocess
            aspect_ratio = w / h
            target_ratios = set(
                (ii, jj) for n in range(2, 7)
                for ii in range(1, n + 1) for jj in range(1, n + 1)
                if 2 <= ii * jj <= 6
            )
            target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
            crop_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, w, h, image_size)

            if crop_ratio[0] > 1 or crop_ratio[1] > 1:
                target_w = image_size * crop_ratio[0]
                target_h = image_size * crop_ratio[1]
                resized = image.resize((target_w, target_h))

                patches = []
                blocks = crop_ratio[0] * crop_ratio[1]
                for b in range(blocks):
                    box = (
                        (b % (target_w // image_size)) * image_size,
                        (b // (target_w // image_size)) * image_size,
                        ((b % (target_w // image_size)) + 1) * image_size,
                        ((b // (target_w // image_size)) + 1) * image_size,
                    )
                    patches.append(image_transform(resized.crop(box)).to(torch.bfloat16))

                crop_tensors = torch.stack(patches, dim=0).cuda()

        # Run vision encoder ONLY
        t0 = time.time()
        with torch.no_grad():
            # Global features
            gf1 = sam(global_tensor)
            gf2 = qwen2(gf1)
            global_features = projector(gf2)

            # Patch features
            if crop_tensors is not None:
                lf1 = sam(crop_tensors)
                lf2 = qwen2(lf1)
                local_features = projector(lf2)
            else:
                local_features = None

        elapsed = time.time() - t0

        actual_base = global_features.shape[1]  # should be 256
        if local_features is not None:
            actual_patches = local_features.shape[0]
            actual_per_patch = local_features.shape[1]
            actual_total = actual_base + actual_patches * actual_per_patch
        else:
            actual_patches = 0
            actual_per_patch = 0
            actual_total = actual_base

        # Update result with actual values
        expected = results[i]["total_visual_tokens"]
        match = "✅" if actual_total == expected else f"❌ expected {expected}"

        print(f"  [{i+1}/{len(image_data)}] {label}")
        print(f"    Global: {global_features.shape}  Local: {local_features.shape if local_features is not None else 'None'}")
        print(f"    Actual visual tokens: {actual_total}  {match}")
        print(f"    Encoder time: {elapsed:.2f}s")
        print()

        results[i]["actual_visual_tokens"] = actual_total
        results[i]["encoder_time_s"] = elapsed

    # Clear GPU memory
    del model, sam, qwen2, projector
    torch.cuda.empty_cache()
else:
    print("[4/4] Skipping GPU validation (no GPU available)")
    print()


# ── Summary Report ─────────────────────────────────────────────────────

print()
print("=" * 78)
print("  📊 PHASE 1 RESULTS: Visual Token Compression Scaling")
print("=" * 78)
print()

header = f"  {'File':<35s} {'Lines':>6s} {'TxtTok':>7s} {'VisTok':>7s} {'Ratio':>7s} {'Patches':>8s} {'ImgSize':>12s}"
print(header)
print("  " + "─" * 76)

for r in sorted(results, key=lambda x: x["lines"]):
    actual = r.get("actual_visual_tokens")
    vt_str = str(r["total_visual_tokens"])
    if actual is not None and actual != r["total_visual_tokens"]:
        vt_str = f"{actual}*"  # mark mismatch
    print(
        f"  {r['label']:<35s} {r['lines']:>6d} {r['text_tokens']:>7d} "
        f"{vt_str:>7s} {r['compression_ratio']:>6.2f}x "
        f"{r['crop_ratio'][0]}×{r['crop_ratio'][1]}={r['patch_count']:>1d}  "
        f"{r['img_width']}×{r['img_height']}"
    )

print()
print("  " + "─" * 76)

# Averages by size category
categories = [
    ("Small (<100 lines)", lambda r: r["lines"] < 100),
    ("Medium (100-500)", lambda r: 100 <= r["lines"] < 500),
    ("Large (500-1500)", lambda r: 500 <= r["lines"] < 1500),
    ("XL (1500+)", lambda r: r["lines"] >= 1500),
]

for cat_label, pred in categories:
    group = [r for r in results if pred(r)]
    if group:
        avg_ratio = sum(r["compression_ratio"] for r in group) / len(group)
        avg_txt = sum(r["text_tokens"] for r in group) / len(group)
        avg_vis = sum(r["total_visual_tokens"] for r in group) / len(group)
        print(f"  {cat_label:<25s}  avg text={avg_txt:>7.0f}  avg visual={avg_vis:>6.0f}  avg ratio={avg_ratio:.2f}x")

print()

# Key findings
if results:
    best = max(results, key=lambda r: r["compression_ratio"])
    worst = min(results, key=lambda r: r["compression_ratio"])
    print(f"  Best compression:  {best['label']} ({best['lines']} lines) → {best['compression_ratio']:.2f}x")
    print(f"  Worst compression: {worst['label']} ({worst['lines']} lines) → {worst['compression_ratio']:.2f}x")
    print()

    # Scaling analysis
    small_ratios = [r["compression_ratio"] for r in results if r["lines"] < 200]
    large_ratios = [r["compression_ratio"] for r in results if r["lines"] > 1000]

    if small_ratios and large_ratios:
        avg_small = sum(small_ratios) / len(small_ratios)
        avg_large = sum(large_ratios) / len(large_ratios)
        print(f"  Avg compression for files <200 lines:   {avg_small:.2f}x")
        print(f"  Avg compression for files >1000 lines:  {avg_large:.2f}x")
        improvement = avg_large / avg_small if avg_small > 0 else 0
        print(f"  Scaling factor: {improvement:.1f}x better for large files")
        print()

        if avg_large > 1.0:
            print("  ✅ COMPRESSION ACHIEVED: Large files use fewer visual tokens than text tokens!")
            if avg_large >= 3.0:
                print(f"     → {avg_large:.1f}x compression is EXCELLENT. Coder-VL is very promising!")
            elif avg_large >= 1.5:
                print(f"     → {avg_large:.1f}x compression is GOOD. Worth pursuing with optimizations.")
            else:
                print(f"     → {avg_large:.1f}x compression is MODEST. May need denser image packing.")
        else:
            print("  ❌ NO NET COMPRESSION: Visual tokens still exceed text tokens for large files.")
            print("     → Consider: smaller font, multi-column layout, or different vision encoder.")

    print()
    print("  ─── Context Window Simulation (128K tokens, 100K for code) ───")
    print()

    for cat_label, pred in categories:
        group = [r for r in results if pred(r)]
        if group:
            avg_txt = sum(r["text_tokens"] for r in group) / len(group)
            avg_vis = sum(r["total_visual_tokens"] for r in group) / len(group)
            files_text = int(100_000 / avg_txt) if avg_txt > 0 else 0
            files_image = int(100_000 / avg_vis) if avg_vis > 0 else 0
            print(f"  {cat_label:<25s}  as text: ~{files_text:>3d} files  as images: ~{files_image:>3d} files  ({files_image/max(files_text,1):.1f}x more)")

    print()
    print("  ─── Key Observation ───")
    print()
    print(f"  Visual tokens are CAPPED at {256 + 6*144} (max 6 patches × 144 + 256 base).")
    print(f"  This means the vision encoder downscales very large files aggressively.")
    print(f"  Whether the model can still 'read' 2000+ line files at that resolution")
    print(f"  is a separate question — but the TOKEN MATH works in our favor.")

print()
print("=" * 78)
print("Done!")
