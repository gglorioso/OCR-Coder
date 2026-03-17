"""
infer_1_9c.py — Phase 1.9c: Large-Scale Alignment Inference

Test whether DeepSeek-Coder-V2-Lite-Instruct can reconstruct Python source code
from injected ConvRoPEProjector vision embeddings after large-scale alignment
training (~8,980 samples, 5 epochs).  This is the evaluation step after train_1_9c.py.
We classify the KIND of failure rather than chasing high BLEU.
"""

import json
import os
import sys
import random
import difflib
import datetime
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # .../MVV/Phase_1_9/c
REPO_ROOT   = SCRIPT_DIR.parents[2]                   # .../OCR-Coder

MANIFEST_PATH   = REPO_ROOT / "MVV/Phase_1_1/data_mvv/manifest.jsonl"
FEATURES_DIR    = REPO_ROOT / "MVV/Phase_1_9/a/data/features"
SCRAPED_DIR     = REPO_ROOT / "Scraped Repos"
PROJECTOR_CKPT  = REPO_ROOT / "MVV/Phase_1_9/c/checkpoints/best.pt"
OUTPUT_DIR      = REPO_ROOT / "MVV/Phase_1_9/c/results"
REPORT_PATH     = OUTPUT_DIR / "reconstruction_report.md"

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/"
    "models--deepseek-ai--DeepSeek-Coder-V2-Lite-Instruct/"
    "snapshots/e434a23f91ba5b4923cf6c9d9a238eb4a08e3a11"
)

SYSTEM_PROMPT = (
    "You are an expert Python OCR assistant. I am injecting 256 high-resolution "
    "visual features representing a Python script. These features have been mapped "
    "to your embedding space but may be unaligned. Use your knowledge of Python "
    "patterns to reconstruct the code as accurately as possible."
)

USER_PROMPT = (
    "The following 256 embeddings represent a high-resolution image of a Python "
    "file. Using your knowledge of Python syntax and the visual structure provided, "
    "reconstruct the exact code content, including all indentation and keywords."
)

GHOSTING_KEYWORDS = ["def", "class", "import", "return", "if", "for"]


# ---------------------------------------------------------------------------
# Step 1: Data selection
# ---------------------------------------------------------------------------

def load_valid_entries(manifest_path: Path, features_dir: Path,
                       scraped_dir: Path) -> list:
    """Return manifest entries that have both a feature file and a source file."""
    valid = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            image_path = entry["image"]                        # e.g. data_mvv/images/black__action__main_py.png
            stem       = Path(image_path).stem                 # e.g. black__action__main_py

            feat_path   = features_dir / f"{stem}.pt"
            source_path = scraped_dir / entry["source_file"]  # Scraped Repos/black/action/main.py

            if feat_path.exists() and source_path.exists():
                entry["_stem"]        = stem
                entry["_feat_path"]   = feat_path
                entry["_source_path"] = source_path
                valid.append(entry)
    return valid


# ---------------------------------------------------------------------------
# Step 2: Failure classification
# ---------------------------------------------------------------------------

def classify_failure(output_text: str, edit_distance: float) -> str:
    """Tag the failure mode of a single LLM output."""
    # WORD_SALAD: >20% non-ASCII / non-printable chars
    non_printable = sum(
        1 for c in output_text
        if not c.isprintable() and c not in "\n\r\t"
    )
    if len(output_text) > 0 and non_printable / len(output_text) > 0.20:
        return "WORD_SALAD"

    # HALLUCINATION: valid Python but edit_distance > 0.8
    try:
        compile(output_text, "<string>", "exec")
        if edit_distance > 0.8:
            return "HALLUCINATION"
    except SyntaxError:
        pass

    # GHOSTING: >=3 Python keywords present AND edit_distance 0.3-0.8
    tokens    = output_text.split()
    kw_count  = sum(1 for kw in GHOSTING_KEYWORDS if kw in tokens)
    if kw_count >= 3 and 0.3 <= edit_distance <= 0.8:
        return "GHOSTING"

    return "OTHER"


# ---------------------------------------------------------------------------
# Step 3: Inference helpers
# ---------------------------------------------------------------------------

def run_inference(entry: dict, projector, llm, tokenizer,
                  device: torch.device) -> dict:
    """Run one forward pass; return a result dict."""
    stem        = entry["_stem"]
    feat_path   = entry["_feat_path"]
    source_path = entry["_source_path"]

    # Load reference source
    # Apply the same truncation as the image renderer: expandtabs(4)[:80]
    with open(source_path, encoding="utf-8", errors="replace") as f:
        ref_lines = f.readlines()
    anchor       = entry.get("anchor_line", 0)
    window       = [line.expandtabs(4)[:80] for line in ref_lines[anchor: anchor + 40]]
    ref_text     = "".join(window)
    ref_first_40 = ref_text

    # A. Vision features → projector → [1, 256, 2048]
    feats = torch.load(feat_path, map_location="cpu", weights_only=False).float().unsqueeze(0).to(device)
    # feats: [1, 1024, 1152]
    with torch.no_grad():
        visual_embeds = projector(feats)   # [1, 256, 2048]

    # B. Format prompt via DeepSeek chat template
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_PROMPT},
    ]
    text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")

    embed_fn = llm.get_input_embeddings()
    with torch.no_grad():
        text_embeds = embed_fn(inputs.input_ids.to(embed_fn.weight.device))
        # Move to same device as visual_embeds
        text_embeds = text_embeds.to(device)

    # C. Concatenate: vision tokens first, then text prompt
    full_embeds    = torch.cat([visual_embeds, text_embeds], dim=1).half()  # [1, 256+N, 2048] fp16
    seq_len        = full_embeds.shape[1]
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    position_ids   = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)

    # D. Manual greedy decode — avoids KV cache API version issues with DeepSeek V2 MLA
    MAX_NEW = 128
    embed_fn  = llm.get_input_embeddings()
    eos_id    = tokenizer.eos_token_id
    generated = []
    current_embeds = full_embeds  # [1, seq_len, 2048]

    with torch.no_grad():
        for _ in range(MAX_NEW):
            cur_len  = current_embeds.shape[1]
            attn_mask = torch.ones((1, cur_len), dtype=torch.long, device=device)
            outputs  = llm(
                inputs_embeds=current_embeds,
                attention_mask=attn_mask,
                use_cache=False,
                return_dict=True,
            )
            next_id = outputs.logits[:, -1, :].argmax(dim=-1)  # [1]
            tok_id  = next_id.item()
            if tok_id == eos_id:
                break
            generated.append(tok_id)
            new_embed = embed_fn(next_id.unsqueeze(0)).to(device)  # [1, 1, 2048]
            current_embeds = torch.cat([current_embeds, new_embed], dim=1)

    output_text = tokenizer.decode(generated, skip_special_tokens=True)

    # E. Metrics
    matcher    = difflib.SequenceMatcher(None, ref_text, output_text)
    char_ratio = matcher.ratio()
    edit_dist  = 1.0 - char_ratio

    failure_type = classify_failure(output_text, edit_dist)

    return {
        "stem":         stem,
        "image":        entry["image"],
        "ref_first_40": ref_first_40,
        "output_text":  output_text,
        "edit_dist":    edit_dist,
        "char_ratio":   char_ratio,
        "failure_type": failure_type,
    }


# ---------------------------------------------------------------------------
# Step 4: Report writing
# ---------------------------------------------------------------------------

def write_report(results: list, report_path: Path) -> None:
    date_str = datetime.date.today().isoformat()

    counts = {"WORD_SALAD": 0, "HALLUCINATION": 0, "GHOSTING": 0, "OTHER": 0}
    for r in results:
        counts[r["failure_type"]] += 1
    mean_edit = sum(r["edit_dist"] for r in results) / len(results)

    lines = [
        "# Phase 1.9c — LLM Reconstruction Report",
        f"**Date:** {date_str}  "
        f"**Model:** DeepSeek-Coder-V2-Lite-Instruct  "
        f"**Projector:** Phase 1.9c best.pt (large-scale alignment, ~8,980 samples, 5 epochs)",
        "",
        "## Summary",
        "| Metric | Value |",
        "|---|---|",
        f"| Samples | {len(results)} |",
        f"| Mean Edit Distance | {mean_edit:.3f} |",
        f"| Word Salad | {counts['WORD_SALAD']} |",
        f"| Hallucination | {counts['HALLUCINATION']} |",
        f"| Ghosting | {counts['GHOSTING']} |",
        f"| Other | {counts['OTHER']} |",
        "",
        "---",
        "",
    ]

    for i, r in enumerate(results, 1):
        lines += [
            f"## Sample {i}: {r['stem']}",
            f"**Image:** `{Path(r['image']).name}`  ",
            f"**Failure Type:** {r['failure_type']}",
            "",
            "### Reference Code",
            "```python",
            r["ref_first_40"].rstrip(),
            "```",
            "",
            "### LLM Output",
            "```",
            r["output_text"],
            "```",
            "",
            f"**Edit Distance:** {r['edit_dist']:.3f}  "
            f"**Char Match Ratio:** {r['char_ratio']:.3f}",
            "",
            "---",
            "",
        ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Data selection
    # ------------------------------------------------------------------
    print("Loading manifest and filtering valid entries ...")
    valid_entries = load_valid_entries(MANIFEST_PATH, FEATURES_DIR, SCRAPED_DIR)
    print(f"  Valid entries: {len(valid_entries)}")

    rng         = random.Random(42)
    sample_size = min(20, len(valid_entries))
    samples     = rng.sample(valid_entries, sample_size)
    print(f"  Sampled: {sample_size}")

    # ------------------------------------------------------------------
    # 2. Load models
    # ------------------------------------------------------------------
    # ConvRoPEProjector is defined in this same file (inlined), so no sys.path needed
    device = torch.device("cuda:0")

    print("Loading ConvRoPEProjector ...")
    projector = ConvRoPEProjector(feat_dim=1152, proj_dim=2048).to(device)

    ckpt       = torch.load(PROJECTOR_CKPT, map_location="cpu", weights_only=False)
    # Phase 1.9c checkpoint saves projector_state_dict directly (no prefix)
    proj_state = ckpt["projector_state_dict"]
    projector.load_state_dict(proj_state)
    projector.eval()
    print(f"  Projector loaded ({sum(p.numel() for p in projector.parameters()):,} params)")

    print("Loading DeepSeek-Coder-V2-Lite-Instruct ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map={"": "cuda:0"},
        load_in_8bit=True,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    llm.eval()
    print("  LLM loaded.")

    # ------------------------------------------------------------------
    # 3. Inference loop
    # ------------------------------------------------------------------
    results = []
    for idx, entry in enumerate(samples, 1):
        stem = entry["_stem"]
        print(f"[{idx}/{sample_size}] Running inference on {stem} ...", flush=True)
        r = run_inference(entry, projector, llm, tokenizer, device)
        results.append(r)
        print(
            f"[{idx}/{sample_size}] {stem} — "
            f"edit_dist={r['edit_dist']:.3f} ({r['failure_type']})"
        )

    # ------------------------------------------------------------------
    # 4. Write report
    # ------------------------------------------------------------------
    write_report(results, REPORT_PATH)


# ---------------------------------------------------------------------------
# ConvRoPEProjector (inlined — identical to MVV/Phase_1_9/scripts/model.py)
# ---------------------------------------------------------------------------

import torch.nn as nn


def _sinusoidal_freqs(seq_len: int, half_dim: int, device: torch.device):
    """Return (cos, sin) tables each [seq_len, half_dim]."""
    i      = torch.arange(half_dim, device=device, dtype=torch.float32)
    theta  = 1.0 / (10000.0 ** (i / half_dim))
    pos    = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(pos, theta)          # [seq_len, half_dim]
    return angles.cos(), angles.sin()


def _rope_rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE rotation. x: [..., seq_len, 2*half_dim]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def apply_2d_rope_16x16(x: torch.Tensor) -> torch.Tensor:
    """
    Inject 2D positional information into x via RoPE.

    The 256 tokens are laid out in a 16×16 row-major grid:
      token i  →  row = i // 16,  col = i % 16

    Dimension split (D=1152):
      dims [0:576]    carry Y-axis (row) RoPE
      dims [576:1152] carry X-axis (col) RoPE

    Args:
        x: [B, 256, 1152]
    Returns:
        [B, 256, 1152]
    """
    B, T, D = x.shape
    assert T == 256 and D == 1152, f"Expected [B,256,1152], got {x.shape}"

    half_D   = D // 2        # 576
    half_dim = half_D // 2   # 288 sin/cos pairs per axis
    device   = x.device

    rows = torch.arange(256, device=device) // 16   # [256]
    cols = torch.arange(256, device=device) % 16    # [256]

    cos_table, sin_table = _sinusoidal_freqs(16, half_dim, device)
    # [16, 288]

    cos_row = cos_table[rows]   # [256, 288]
    sin_row = sin_table[rows]
    cos_col = cos_table[cols]   # [256, 288]
    sin_col = sin_table[cols]

    x_row = x[:, :, :half_D]    # [B, 256, 576]
    x_col = x[:, :, half_D:]    # [B, 256, 576]

    x_row_rot = _rope_rotate(x_row, cos_row, sin_row)
    x_col_rot = _rope_rotate(x_col, cos_col, sin_col)

    return torch.cat([x_row_rot, x_col_rot], dim=-1)   # [B, 256, 1152]


class ConvRoPEProjector(nn.Module):
    """
    Compresses [B, 1024, 1152] raw SigLIP tokens to [B, 256, 2048] via:
      strided conv (32→16)  →  2D RoPE  →  MLP (1152→2048)
    """

    def __init__(self, feat_dim: int = 1152, proj_dim: int = 2048) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.grid_in  = 32
        self.grid_out = 16

        self.conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=2, stride=2)

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.conv.bias)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1024, 1152]  →  [B, 256, 2048]"""
        B, N, C = x.shape
        assert N == self.grid_in ** 2 and C == self.feat_dim, \
            f"Expected [B,{self.grid_in**2},{self.feat_dim}], got {x.shape}"

        # 1. Reshape to spatial grid
        x = x.reshape(B, self.grid_in, self.grid_in, C).permute(0, 3, 1, 2)
        # [B, 1152, 32, 32]

        # 2. Stride-2 conv: 32×32 → 16×16
        x = self.conv(x)
        # [B, 1152, 16, 16]

        # 3. Flatten spatial dims back to sequence
        x = x.flatten(2).transpose(1, 2)
        # [B, 256, 1152]

        # 4. Inject 2D positional information via RoPE
        x = apply_2d_rope_16x16(x)
        # [B, 256, 1152]

        # 5. Project to LLM embedding dimension
        x = self.mlp(x)
        # [B, 256, 2048]

        return x


if __name__ == "__main__":
    main()
