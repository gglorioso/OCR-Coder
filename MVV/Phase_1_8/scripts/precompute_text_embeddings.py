"""
Precompute SigLIP text embeddings for Phase 1.8 contrastive training.

For each unique query string "def {name}(" found in the ground-truth JSONL,
embed it with SiglipTextModel (EOS-pooled, float32) and save the result as a
single dict .pt file.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, SiglipTextModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_queries(jsonl_path: Path) -> list[str]:
    """Return deduplicated list of query strings from the ground-truth file."""
    seen: set[str] = set()
    queries: list[str] = []
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            q = f"def {entry['name']}("
            if q not in seen:
                seen.add(q)
                queries.append(q)
    return queries


def embed_queries(
    queries: list[str],
    model_name: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Embed every query string and return a {query: tensor_1152d} dict."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = SiglipTextModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    result: dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for start in range(0, len(queries), batch_size):
            batch_queries = queries[start : start + batch_size]

            inputs = tokenizer(
                batch_queries,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            # last_hidden_state: [B, seq_len, hidden]
            hidden = outputs.last_hidden_state  # [B, T, 1152]

            # Pool at EOS token position (last non-padding token).
            # SigLIP tokenizer pads on the right; the EOS token is the last
            # real token before padding.  We find it via attention_mask.
            attn_mask = inputs["attention_mask"]          # [B, T]
            seq_lens = attn_mask.sum(dim=1) - 1          # index of last real token
            eos_embeds = hidden[
                torch.arange(hidden.size(0), device=device), seq_lens
            ]  # [B, 1152]

            eos_embeds = eos_embeds.float().cpu()
            for q, emb in zip(batch_queries, eos_embeds):
                result[q] = emb

            if (start // batch_size) % 10 == 0:
                print(
                    f"  [{start + len(batch_queries)}/{len(queries)}] "
                    f"queries embedded"
                )

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-compute SigLIP text embeddings for Phase 1.8"
    )
    p.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("MVV/Phase_1_8/data/ground_truth/ground_truth.jsonl"),
        help="Path to ground_truth.jsonl",
    )
    p.add_argument(
        "--out-path",
        type=Path,
        default=Path("MVV/Phase_1_8/data/text_embeddings/text_embeddings.pt"),
        help="Output .pt file path",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading ground truth from: {args.ground_truth}")
    queries = load_queries(args.ground_truth)
    print(f"  {len(queries)} unique queries found")

    model_name = "google/siglip-so400m-patch14-384"
    print(f"Loading model: {model_name}")
    embeddings = embed_queries(queries, model_name, device, args.batch_size)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, args.out_path)
    print(f"Saved {len(embeddings)} embeddings → {args.out_path}")

    # Sanity check
    sample_key = next(iter(embeddings))
    sample_val = embeddings[sample_key]
    print(f"  Sample key : {sample_key!r}")
    print(f"  Sample shape: {sample_val.shape}, dtype: {sample_val.dtype}")


if __name__ == "__main__":
    main()
