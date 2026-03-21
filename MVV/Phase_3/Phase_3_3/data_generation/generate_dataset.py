#!/usr/bin/env python3
"""
generate_dataset.py -- Phase 3.3 multimodal dataset generator

Reads the MVV manifest, renders each source file as sequential 40-line chunks
on 800x800 grayscale canvases, extracts SigLIP features via GPU, and writes
paired (tensor, text) files for downstream training.

Designed for SLURM array parallelism: each array task processes a disjoint
slice of the manifest. CPU workers render images in parallel; the main
process runs SigLIP on the GPU.

Outputs:
    MVV/Phase_3/full_data/tensors_and_texts/{file_id}_chunk{i}.pt   [1024, 1152] fp16
    MVV/Phase_3/full_data/tensors_and_texts/{file_id}_chunk{i}.txt  raw text
    MVV/Phase_3/full_data/manifest_out.jsonl                        per-chunk metadata
"""

import fcntl
import json
import logging
import math
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[4]  # OCR-Coder/
MANIFEST_PATH = REPO_ROOT / "MVV" / "Phase_1_1" / "data_mvv" / "manifest.jsonl"
SCRAPED_REPOS = REPO_ROOT / "Scraped Repos"
OUTPUT_DIR = REPO_ROOT / "MVV" / "Phase_3" / "full_data"
TENSORS_DIR = OUTPUT_DIR / "tensors_and_texts"
MANIFEST_OUT = OUTPUT_DIR / "manifest_out.jsonl"

# ── Canvas constants (locked to gen_images.py spec) ───────────────────────────
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_SIZE = 16
LINE_HEIGHT = 20       # px per line
MAX_COLS = 80          # chars before hard truncation
MAX_ROWS = 40          # lines per chunk
CANVAS_W = MAX_COLS * 10   # 800 px
CANVAS_H = MAX_ROWS * LINE_HEIGHT  # 800 px
BG_COLOR = 255         # white
TEXT_COLOR = 0          # black

# ── Filter thresholds ─────────────────────────────────────────────────────────
MAX_SOURCE_LINES = 2000

# ── SigLIP config ─────────────────────────────────────────────────────────────
SIGLIP_MODEL_ID = "google/siglip-so400m-patch14-384"
SIGLIP_INPUT_SIZE = 448        # 448/14 = 32 -> 32x32 = 1024 tokens natively
SIGLIP_TARGET_TOKENS = 1024   # target sequence length for adapter input
SIGLIP_DIM = 1152              # SO400M hidden dim
SIGLIP_BATCH_SIZE = 32        # images per GPU forward pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Manifest loading + SLURM slicing
# ═══════════════════════════════════════════════════════════════════════════════

def load_manifest(path: Path) -> List[dict]:
    """Load a JSONL manifest into a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def slurm_slice(items: list) -> list:
    """Return the subset of *items* assigned to the current SLURM array task."""
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))
    task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    # SLURM array IDs are 1-based; convert to 0-based index
    idx = task_id - 1
    per_task = math.ceil(len(items) / task_count)
    start = idx * per_task
    end = min(start + per_task, len(items))
    log.info(
        "SLURM array task %d/%d  ->  manifest[%d:%d]  (%d items)",
        task_id, task_count, start, end, end - start,
    )
    return items[start:end]


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Slug helper
# ═══════════════════════════════════════════════════════════════════════════════

def make_slug(source_file: str) -> str:
    """Convert a relative path like 'black/action/main.py' to 'black__action__main_py'."""
    return source_file.replace("/", "__").replace("\\", "__").replace(".", "_")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Multi-chunking
# ═══════════════════════════════════════════════════════════════════════════════

def chunk_lines(lines: List[str], chunk_size: int = MAX_ROWS) -> List[List[str]]:
    """Split *lines* into sequential chunks of *chunk_size*. Last chunk may be shorter."""
    return [lines[i : i + chunk_size] for i in range(0, len(lines), chunk_size)]


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Rendering (CPU, runs inside worker processes)
# ═══════════════════════════════════════════════════════════════════════════════

def _render_chunk_to_image(lines: List[str]) -> Image.Image:
    """
    Render up to MAX_ROWS lines on an 800x800 grayscale canvas.

    - Tabs expanded to 4 spaces.
    - Lines hard-truncated at MAX_COLS chars (no wrapping).
    - White background (255), black text (0), PIL 'L' mode.
    - If fewer than MAX_ROWS lines, remaining rows are blank (white).
    """
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    img = Image.new("L", (CANVAS_W, CANVAS_H), color=BG_COLOR)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines[:MAX_ROWS]):
        rendered = line.expandtabs(4)[:MAX_COLS]
        draw.text((0, i * LINE_HEIGHT), rendered, font=font, fill=TEXT_COLOR)
    return img


def _render_worker(args: Tuple[str, int, List[str]]) -> Tuple[str, int, Image.Image, str]:
    """
    Worker function for ProcessPoolExecutor.

    Args:
        args: (file_id, chunk_index, lines)

    Returns:
        (file_id, chunk_index, PIL.Image, raw_text)
    """
    file_id, chunk_idx, lines = args
    raw_text = "\n".join(lines)
    img = _render_chunk_to_image(lines)
    return (file_id, chunk_idx, img, raw_text)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  SigLIP feature extraction (GPU, main process only)
# ═══════════════════════════════════════════════════════════════════════════════

def load_siglip(device: torch.device):
    """Load the SigLIP vision encoder and image processor from HuggingFace cache."""
    from transformers import SiglipImageProcessor, SiglipVisionModel

    log.info("Loading SigLIP model: %s", SIGLIP_MODEL_ID)
    vision_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_ID)
    vision_model.eval().to(device)

    # --- Upscale SigLIP positional embeddings: 729 (27x27) -> 1024 (32x32) ---
    # Standard technique (used by LLaVA, Qwen-VL) to support higher-res input.
    # Done once at startup so the model natively accepts 448x448 images.
    old_pos = vision_model.vision_model.embeddings.position_embedding.weight.data
    hidden_size = old_pos.shape[-1]
    pos_grid = old_pos.view(1, 27, 27, hidden_size).permute(0, 3, 1, 2)
    new_pos_grid = F.interpolate(
        pos_grid.float(), size=(32, 32), mode="bicubic", align_corners=False
    )
    new_pos = new_pos_grid.permute(0, 2, 3, 1).reshape(1024, hidden_size)
    new_embedding = torch.nn.Embedding(1024, hidden_size)
    new_embedding.weight.data = new_pos
    model_dtype = vision_model.vision_model.embeddings.position_embedding.weight.dtype
    vision_model.vision_model.embeddings.position_embedding = new_embedding.to(
        device=device, dtype=model_dtype
    )
    vision_model.vision_model.embeddings.num_positions = 1024
    vision_model.vision_model.embeddings.register_buffer(
        "position_ids", torch.arange(1024).expand((1, -1)).to(device)
    )
    log.info("Interpolated SigLIP position embeddings: 729 -> 1024 tokens (27x27 -> 32x32)")

    processor = SiglipImageProcessor.from_pretrained(SIGLIP_MODEL_ID)
    log.info("SigLIP loaded on %s", device)
    return vision_model, processor


def extract_siglip_features(
    image: Image.Image,
    vision_model,
    processor,
    device: torch.device,
) -> torch.Tensor:
    """
    Extract SigLIP features from a grayscale PIL image.

    Returns tensor of shape [1024, 1152].

    By resizing the input to 448x448 before processing, SigLIP natively
    produces 1024 tokens (448/14 = 32 -> 32x32 grid) of 1152-dim,
    avoiding any lossy spatial interpolation.
    """
    # Convert grayscale (L) to RGB since SigLIP expects 3-channel input
    rgb_image = image.convert("RGB")

    # Resize to 448x448 so SigLIP produces 32x32 = 1024 tokens natively
    rgb_image = rgb_image.resize(
        (SIGLIP_INPUT_SIZE, SIGLIP_INPUT_SIZE), Image.LANCZOS
    )

    # Preprocess with the SigLIP image processor at 448x448
    inputs = processor(
        images=rgb_image, return_tensors="pt",
        size={"height": SIGLIP_INPUT_SIZE, "width": SIGLIP_INPUT_SIZE},
    )
    pixel_values = inputs["pixel_values"].to(device)  # [1, 3, 448, 448]

    with torch.no_grad():
        outputs = vision_model(pixel_values=pixel_values)
        hidden = outputs.last_hidden_state  # [1, 1024, 1152]

    return hidden.squeeze(0).half()  # [1024, 1152] fp16


def extract_siglip_batch(
    images: List[Image.Image],
    vision_model,
    processor,
    device: torch.device,
) -> List[torch.Tensor]:
    """
    Extract SigLIP features for a batch of grayscale PIL images.

    Returns a list of tensors, each [1024, 1152] fp16.
    """
    # Convert all to RGB and resize to 448x448 for native 1024-token output
    rgb_images = [
        img.convert("RGB").resize(
            (SIGLIP_INPUT_SIZE, SIGLIP_INPUT_SIZE), Image.LANCZOS
        )
        for img in images
    ]

    # Batch preprocess at 448x448
    inputs = processor(
        images=rgb_images, return_tensors="pt",
        size={"height": SIGLIP_INPUT_SIZE, "width": SIGLIP_INPUT_SIZE},
    )
    pixel_values = inputs["pixel_values"].to(device)  # [B, 3, 448, 448]

    with torch.no_grad():
        outputs = vision_model(pixel_values=pixel_values)
        hidden = outputs.last_hidden_state  # [B, 1024, 1152]

    return [hidden[i].half().cpu() for i in range(hidden.shape[0])]


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Manifest writing (file-locked for multi-process safety)
# ═══════════════════════════════════════════════════════════════════════════════

def append_manifest_record(record: dict, path: Path) -> None:
    """Append a single JSON record to *path* with advisory file locking."""
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(line)
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Setup ─────────────────────────────────────────────────────────────────
    TENSORS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        log.warning("No GPU detected -- running on CPU (will be very slow)")

    # ── Load manifest and SLURM-slice ─────────────────────────────────────────
    log.info("Loading manifest from %s", MANIFEST_PATH)
    manifest = load_manifest(MANIFEST_PATH)
    log.info("Total manifest entries: %d", len(manifest))

    manifest = slurm_slice(manifest)
    if not manifest:
        log.info("Nothing to process for this SLURM task. Exiting.")
        return

    # ── Hard filter: skip files > MAX_SOURCE_LINES ────────────────────────────
    filtered = []
    skipped = 0
    for rec in manifest:
        if rec.get("n_source_lines", 0) > MAX_SOURCE_LINES:
            skipped += 1
            log.debug("Skipped (too long, %d lines): %s",
                       rec["n_source_lines"], rec["source_file"])
        else:
            filtered.append(rec)
    log.info("Skipped %d files with n_source_lines > %d", skipped, MAX_SOURCE_LINES)
    log.info("Files to process after filter: %d", len(filtered))
    manifest = filtered

    # ── Load SigLIP ───────────────────────────────────────────────────────────
    vision_model, processor = load_siglip(device)

    # ── Prepare rendering tasks ───────────────────────────────────────────────
    # Read source files and build (file_id, chunk_idx, lines) tuples for workers.
    render_tasks: List[Tuple[str, int, List[str]]] = []
    # Parallel metadata for each task (same index)
    task_meta: List[dict] = []

    files_processed = 0
    files_skipped_read = 0

    for rec in manifest:
        source_file = rec["source_file"]
        src_path = SCRAPED_REPOS / source_file
        file_id = make_slug(source_file)

        try:
            source_text = src_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            log.warning("Cannot read %s: %s", src_path, exc)
            files_skipped_read += 1
            continue

        all_lines = source_text.splitlines()
        chunks = chunk_lines(all_lines, MAX_ROWS)
        total_chunks = len(chunks)

        for chunk_idx, chunk in enumerate(chunks):
            render_tasks.append((file_id, chunk_idx, chunk))
            task_meta.append({
                "file_id": file_id,
                "chunk_index": chunk_idx,
                "total_chunks": total_chunks,
                "source_file": source_file,
                "n_source_lines": len(all_lines),
            })

        files_processed += 1

    if files_skipped_read:
        log.info("Skipped %d files due to read errors", files_skipped_read)
    log.info("Total render tasks (chunks): %d from %d files", len(render_tasks), files_processed)

    if not render_tasks:
        log.info("No chunks to process. Exiting.")
        return

    # ── CPU rendering via ProcessPoolExecutor ─────────────────────────────────
    n_workers = os.cpu_count() or 16
    log.info("Rendering images with %d CPU workers", n_workers)

    # Collect rendered results: list of (file_id, chunk_idx, image, raw_text)
    rendered: List[Optional[Tuple[str, int, Image.Image, str]]] = [None] * len(render_tasks)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_render_worker, task): i
            for i, task in enumerate(render_tasks)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Rendering"):
            idx = futures[future]
            try:
                rendered[idx] = future.result()
            except Exception as exc:
                meta = task_meta[idx]
                log.error("Render failed for %s chunk %d: %s",
                          meta["file_id"], meta["chunk_index"], exc)

    # Drop failed renders
    valid_indices = [i for i, r in enumerate(rendered) if r is not None]
    log.info("Successfully rendered %d / %d chunks", len(valid_indices), len(render_tasks))

    # ── GPU SigLIP extraction in batches ──────────────────────────────────────
    log.info("Extracting SigLIP features (batch_size=%d)", SIGLIP_BATCH_SIZE)
    total_saved = 0

    for batch_start in tqdm(range(0, len(valid_indices), SIGLIP_BATCH_SIZE),
                            desc="SigLIP extraction"):
        batch_idx = valid_indices[batch_start : batch_start + SIGLIP_BATCH_SIZE]
        batch_images = [rendered[i][2] for i in batch_idx]  # PIL images

        try:
            features_list = extract_siglip_batch(
                batch_images, vision_model, processor, device
            )
        except Exception as exc:
            log.error("SigLIP batch failed at offset %d: %s\n%s",
                      batch_start, exc, traceback.format_exc())
            continue

        # Save each result
        for j, global_idx in enumerate(batch_idx):
            file_id, chunk_idx, _, raw_text = rendered[global_idx]
            meta = task_meta[global_idx]

            tensor_name = f"{file_id}_chunk{chunk_idx}.pt"
            text_name = f"{file_id}_chunk{chunk_idx}.txt"
            tensor_path = TENSORS_DIR / tensor_name
            text_path = TENSORS_DIR / text_name

            try:
                # Save tensor (fp16)
                torch.save(features_list[j], tensor_path)

                # Save raw text (no trailing blank-line padding)
                text_path.write_text(raw_text, encoding="utf-8")

                # Append to output manifest
                record = {
                    "file_id": file_id,
                    "chunk_index": chunk_idx,
                    "total_chunks": meta["total_chunks"],
                    "tensor_path": str(tensor_path.relative_to(REPO_ROOT)),
                    "text_path": str(text_path.relative_to(REPO_ROOT)),
                    "ground_truth_text": raw_text,
                    "source_file": meta["source_file"],
                    "n_source_lines": meta["n_source_lines"],
                }
                append_manifest_record(record, MANIFEST_OUT)
                total_saved += 1

            except Exception as exc:
                log.error("Failed to save %s chunk %d: %s", file_id, chunk_idx, exc)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("DONE")
    log.info("  Total files processed : %d", files_processed)
    log.info("  Total files skipped   : %d (too long) + %d (read error)",
             skipped, files_skipped_read)
    log.info("  Total chunks generated: %d", total_saved)
    log.info("  Output tensors dir    : %s", TENSORS_DIR)
    log.info("  Output manifest       : %s", MANIFEST_OUT)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
