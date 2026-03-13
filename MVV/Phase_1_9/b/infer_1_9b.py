"""
infer_1_9b.py — Phase 1.9b: Honest Failure Baseline

Test whether DeepSeek-Coder-V2-Lite-Instruct can reconstruct Python source code
from injected ConvRoPEProjector vision embeddings.  This is an intentional failure
baseline; we classify the KIND of failure rather than chasing high BLEU.
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
SCRIPT_DIR = Path(__file__).resolve().parent          # .../MVV/Phase_1_9/b
REPO_ROOT   = SCRIPT_DIR.parents[2]                   # .../OCR-Coder

MANIFEST_PATH   = REPO_ROOT / "MVV/Phase_1_1/data_mvv/manifest.jsonl"
FEATURES_DIR    = REPO_ROOT / "MVV/Phase_1_9/data/features"
SCRAPED_DIR     = REPO_ROOT / "Scraped Repos"
PROJECTOR_CKPT  = REPO_ROOT / "MVV/Phase_1_9/checkpoints/best.pt"
OUTPUT_DIR      = REPO_ROOT / "MVV/Phase_1_9/b/results"
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
    with open(source_path, encoding="utf-8", errors="replace") as f:
        ref_lines = f.readlines()
    ref_text     = "".join(ref_lines)
    ref_first_40 = "".join(ref_lines[:40])

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
        "# Phase 1.9b — LLM Reconstruction Report",
        f"**Date:** {date_str}  "
        f"**Model:** DeepSeek-Coder-V2-Lite-Instruct  "
        f"**Projector:** Phase 1.9a best.pt (macro_F1=0.780)",
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
    # Add model.py location to path so ConvRoPEProjector is importable
    sys.path.insert(0, str(REPO_ROOT / "MVV/Phase_1_9/scripts"))
    from model import ConvRoPEProjector  # noqa: E402

    device = torch.device("cuda:0")

    print("Loading ConvRoPEProjector ...")
    projector = ConvRoPEProjector(feat_dim=1152, proj_dim=2048).to(device)

    ckpt       = torch.load(PROJECTOR_CKPT, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    # Checkpoint was saved from ConvRoPEKeywordDetector; strip "projector." prefix
    proj_state = {
        k[len("projector."):]: v
        for k, v in state_dict.items()
        if k.startswith("projector.")
    }
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


if __name__ == "__main__":
    main()
