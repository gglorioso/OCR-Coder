"""
Phase 2a Evaluation — Gates G4, G5, G6

Loads the best adapter checkpoint, generates responses on the validation set,
and computes:
  G4: ROUGE-L (code description quality)       threshold > 0.25
  G5: Exact-match accuracy (function listing)   threshold > 30%
  G6: Distinct-1 (unigram diversity)            threshold > 0.3

Uses best.pt (lowest validation loss) by default.
Runs on a single GPU (V100 32 GB).

Usage:
    python evaluate_phase2a.py
    python evaluate_phase2a.py --max_samples 20   # quick smoke test
"""

import json
import argparse
import re
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from tqdm import tqdm

from projector import ProjectionAdapter


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _lcs_length(x, y):
    """Length of the longest common subsequence (space-optimised DP)."""
    m, n = len(x), len(y)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def rouge_l_f1(generated: str, reference: str) -> float:
    """Word-level ROUGE-L F1."""
    gen_tok = generated.lower().split()
    ref_tok = reference.lower().split()
    if not gen_tok or not ref_tok:
        return 0.0
    lcs = _lcs_length(gen_tok, ref_tok)
    p = lcs / len(gen_tok)
    r = lcs / len(ref_tok)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def extract_function_names(text: str) -> set:
    """Pull function/method names from a listing response."""
    names = set()
    # "1. func_name" or "1. func_name(args) -> ..."
    for m in re.finditer(r"^\s*\d+\.\s*(\w+)", text, re.MULTILINE):
        names.add(m.group(1))
    # "- func_name"
    for m in re.finditer(r"^\s*[-•]\s*(\w+)", text, re.MULTILINE):
        names.add(m.group(1))
    # "def func_name"
    for m in re.finditer(r"\bdef\s+(\w+)", text):
        names.add(m.group(1))
    return names


def function_listing_exact_match(generated: str, reference: str) -> bool:
    gen = extract_function_names(generated)
    ref = extract_function_names(reference)
    if not ref:
        return False
    return gen == ref


def compute_distinct_1(texts: list) -> float:
    """Unique unigrams / total unigrams across all texts."""
    all_tok = []
    for t in texts:
        all_tok.extend(t.lower().split())
    if not all_tok:
        return 0.0
    return len(set(all_tok)) / len(all_tok)


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_one(prompt_ids, features, adapter, coder, tokenizer,
                 image_token_id, embed_fn, device, max_new_tokens,
                 repetition_penalty=1.0):
    """
    Replace <image> token with projected features and generate greedily.
    Uses manual autoregressive loop since .generate() doesn't handle inputs_embeds well.

    Args:
        prompt_ids: [1, seq]            tokenised user prompt
        features:   [1, 256, 1280]      pre-computed vision features
    Returns:
        generated text (str)
    """
    ids = prompt_ids.to(device)
    feat = features.to(device)

    # Adapter: fp32 compute → fp16 output
    projected = adapter(feat.float()).half()          # [1, 256, 2048]

    # Text embeddings
    text_emb = embed_fn(ids)                          # [1, seq, 2048]

    # Find <image> position and splice
    positions = (ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    if len(positions) > 0:
        p = positions[0].item()
        combined = torch.cat(
            [text_emb[0, :p], projected[0], text_emb[0, p + 1:]],
            dim=0,
        ).unsqueeze(0)
        # Track original token IDs (needed for stopping)
        prefix_ids = ids[0, :p].tolist()
        suffix_ids = ids[0, p + 1:].tolist()
        prompt_token_ids = prefix_ids + suffix_ids  # without <image>
    else:
        combined = text_emb
        prompt_token_ids = ids[0].tolist()

    seq_len = combined.size(1)
    mask = torch.ones(1, seq_len, device=device)

    # Manual autoregressive generation with KV caching
    generated_ids = []
    past_key_values = None

    for step in range(max_new_tokens):
        # Forward pass
        if step == 0:
            # First step: process entire prompt
            outputs = coder(
                inputs_embeds=combined,
                attention_mask=mask,
                use_cache=True,
                past_key_values=None,
            )
            past_key_values = outputs.past_key_values
        else:
            # Subsequent steps: only process new token (use cached KV)
            outputs = coder(
                inputs_embeds=next_emb,
                attention_mask=mask,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values

        # Get next token logits (last position)
        logits = outputs.logits[0, -1, :].clone()  # [vocab_size]

        # Repetition penalty: divide logit of already-generated tokens
        if repetition_penalty != 1.0 and generated_ids:
            for tok_id in set(generated_ids):
                if logits[tok_id] > 0:
                    logits[tok_id] /= repetition_penalty
                else:
                    logits[tok_id] *= repetition_penalty

        next_token_id = logits.argmax().item()

        # Stop on EOS
        if next_token_id == tokenizer.eos_token_id:
            break

        generated_ids.append(next_token_id)

        # Prepare next token embedding (for next iteration)
        next_emb = embed_fn(torch.tensor([[next_token_id]], device=device))  # [1, 1, 2048]

        # Extend attention mask (KV cache handles sequence extension)
        mask = torch.cat([mask, torch.ones(1, 1, device=device)], dim=1)

    # Decode (skip the prompt, only decode generated tokens)
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 2a Evaluation — G4/G5/G6")
    parser.add_argument("--checkpoint", default="./checkpoints/phase2a/best.pt")
    parser.add_argument("--features_dir", default="./precomputed_features")
    parser.add_argument("--val_manifest",
                        default="Data Crawling/output/manifests/val.jsonl")
    parser.add_argument("--coder_model",
                        default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Cap on val examples (0 = use all)")
    parser.add_argument("--save_file", default="",
                        help="Path to save/resume results JSON. If the file exists, "
                             "already-processed IDs are skipped (resume mode).")
    parser.add_argument("--save_every", type=int, default=100,
                        help="Flush results to --save_file every N examples (default 100)")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="Repetition penalty > 1.0 discourages repeated tokens (e.g. 1.3)")
    args = parser.parse_args()

    device = "cuda"

    # ==================================================================
    # 1. Coder model  (same 4-bit setup as training)
    # ==================================================================
    print("=" * 60)
    print("LOADING CODER MODEL (4-bit, fp16)")
    print("=" * 60)

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    coder = AutoModelForCausalLM.from_pretrained(
        args.coder_model,
        quantization_config=bnb_cfg,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.coder_model, trust_remote_code=True,
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<image>", "<img_start>", "<img_end>"]}
    )
    coder.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    coder.eval()
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    embed_fn = coder.get_input_embeddings()
    print(f"  hidden_size={coder.config.hidden_size}  "
          f"image_token_id={image_token_id}")
    print("  Coder model loaded (eval mode)\n")

    # ==================================================================
    # 2. Adapter
    # ==================================================================
    print(f"Loading adapter from {args.checkpoint} ...")
    adapter = ProjectionAdapter(
        vision_dim=1280, hidden_dim=4096,
        coder_dim=coder.config.hidden_size,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(ckpt, dict) and "adapter_state_dict" in ckpt:
        adapter.load_state_dict(ckpt["adapter_state_dict"])
        step = ckpt.get("global_step", "?")
        print(f"  Loaded full checkpoint (step={step})")
    else:
        # adapter_final.pt is a raw state_dict
        adapter.load_state_dict(ckpt)
        print("  Loaded raw state_dict")
    adapter = adapter.to(device).eval()
    print(f"  Adapter ready ({adapter.num_parameters():,} params)\n")

    # ==================================================================
    # 3. Validation data
    # ==================================================================
    print("Loading validation data ...")
    features_dir = Path(args.features_dir)
    examples = []
    with open(args.val_manifest) as f:
        for line in f:
            examples.append(json.loads(line))

    valid = []
    for ex in examples:
        fp = features_dir / (Path(ex["image"]).stem + ".pt")
        if fp.exists():
            ex["_feat_path"] = str(fp)
            valid.append(ex)

    print(f"  {len(valid)}/{len(examples)} examples have features")
    if args.max_samples > 0:
        valid = valid[: args.max_samples]
        print(f"  Limited to {len(valid)} samples")
    print()

    # ==================================================================
    # 4. Generate
    # ==================================================================
    print("=" * 60)
    print("RUNNING INFERENCE")
    print("=" * 60 + "\n")

    # Resume: load any existing partial results
    results = []
    processed_ids = set()
    if args.save_file and Path(args.save_file).exists():
        with open(args.save_file) as f:
            results = json.load(f)
        processed_ids = {r["id"] for r in results}
        print(f"  Resuming from {args.save_file}: {len(results)} examples already done\n")

    for ex in tqdm(valid, desc="Generating"):
        ex_id = ex.get("id", "")
        if ex_id in processed_ids:
            continue

        conv = ex["conversations"]
        user_msg = conv[0]["content"]
        reference = conv[1]["content"]
        task_type = ex.get("task_type", "unknown")

        # Prompt: user turn only, model completes from "Assistant:"
        prompt = f"User: {user_msg}\n\nAssistant:"
        tok = tokenizer(prompt, return_tensors="pt")
        prompt_ids = tok["input_ids"]                            # [1, seq]
        feat = torch.load(ex["_feat_path"], map_location="cpu")  # [num_tokens, 1280]
        feat = feat.unsqueeze(0)                                 # [1, num_tokens, 1280]

        generated = generate_one(
            prompt_ids, feat, adapter, coder, tokenizer,
            image_token_id, embed_fn, device, args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
        )

        results.append({
            "id":        ex_id,
            "task_type": task_type,
            "generated": generated,
            "reference": reference,
        })

        # Periodic save
        if args.save_file and len(results) % args.save_every == 0:
            with open(args.save_file, "w") as f:
                json.dump(results, f)
            tqdm.write(f"  [Saved {len(results)} results → {args.save_file}]")

    # Final save
    if args.save_file:
        with open(args.save_file, "w") as f:
            json.dump(results, f)
        print(f"\n  Results saved to {args.save_file} ({len(results)} total)")

    # ==================================================================
    # 5. Compute metrics
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("COMPUTING METRICS")
    print(f"{'=' * 60}\n")

    # --- G4: ROUGE-L (all examples) ---
    rouge_scores = []
    for r in results:
        s = rouge_l_f1(r["generated"], r["reference"])
        r["rouge_l"] = s
        rouge_scores.append(s)
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0

    # Per-task breakdown
    task_rouge = {}
    for r in results:
        task_rouge.setdefault(r["task_type"], []).append(r["rouge_l"])

    # --- G5: Exact-match on function_listing ---
    func_examples = [r for r in results if r["task_type"] == "function_listing"]
    func_matches = sum(
        1 for r in func_examples
        if function_listing_exact_match(r["generated"], r["reference"])
    )
    func_acc = func_matches / len(func_examples) if func_examples else 0

    # --- G6: Distinct-1 ---
    distinct1 = compute_distinct_1([r["generated"] for r in results])

    # ==================================================================
    # 6. Report
    # ==================================================================
    print("=" * 60)
    print("PHASE 2A GATE RESULTS")
    print("=" * 60)
    if len(results) < len(valid):
        print(f"  NOTE: partial results ({len(results)}/{len(valid)} examples)")
    print()

    g4_pass = avg_rouge > 0.25
    g5_pass = func_acc > 0.30
    g6_pass = distinct1 > 0.30

    print(f"  G4  ROUGE-L          {avg_rouge:.4f}   (threshold > 0.25)  "
          f"{'PASS' if g4_pass else 'FAIL'}")
    print(f"  G5  Func exact-match {func_acc:.4f}   (threshold > 0.30)  "
          f"{'PASS' if g5_pass else 'FAIL'}  "
          f"({func_matches}/{len(func_examples)} examples)")
    print(f"  G6  Distinct-1       {distinct1:.4f}   (threshold > 0.30)  "
          f"{'PASS' if g6_pass else 'FAIL'}")
    print()

    all_pass = g4_pass and g5_pass and g6_pass
    print(f"  Overall: {'ALL GATES PASS — proceed to Phase 2b' if all_pass else 'SOME GATES FAILED — see failure protocol in PHASE2_PLAN.md Section 8'}")
    print()

    # ROUGE-L by task type
    print("-" * 60)
    print("ROUGE-L by task type:")
    print("-" * 60)
    for task, scores in sorted(task_rouge.items()):
        avg = sum(scores) / len(scores)
        print(f"  {task:25s}  {avg:.4f}  (n={len(scores)})")
    print()

    # Sample outputs (5 examples from different task types)
    print("-" * 60)
    print("SAMPLE OUTPUTS (for manual inspection)")
    print("-" * 60)
    shown_tasks = set()
    n_shown = 0
    for r in results:
        if n_shown >= 5:
            break
        if r["task_type"] in shown_tasks:
            continue
        shown_tasks.add(r["task_type"])
        n_shown += 1
        print(f"\n  [{r['task_type']}]  id={r['id']}")
        print(f"  ROUGE-L: {r.get('rouge_l', 0):.4f}")
        ref_short = r["reference"][:200] + ("..." if len(r["reference"]) > 200 else "")
        gen_short = r["generated"][:200] + ("..." if len(r["generated"]) > 200 else "")
        print(f"  REF: {ref_short}")
        print(f"  GEN: {gen_short}")
    print()

    print("=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
