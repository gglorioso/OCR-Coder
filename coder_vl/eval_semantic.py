"""
eval_semantic.py — Semantic evaluation of Phase 2b outputs

Two evaluations on the existing eval_results_2b.json:

1. Semantic similarity (proxy for BERTScore)
   For each example: cosine similarity between mean-pooled BERT embeddings of
   generated vs. reference text. Reported per task type.

2. Retrieval Recall@k (the actual downstream metric)
   For description + function_explanation tasks:
   - Encode all reference texts as the "file index"
   - Use each generated text as a query
   - Measure Recall@1, @3, @5: is the correct file in the top-k results?
   This tests whether the model's visual descriptions are useful for
   locating the right file given a query — the core Sniper Method claim.

Usage:
    python coder_vl/eval_semantic.py
    python coder_vl/eval_semantic.py --results eval_results_2b.json --model distilbert-base-uncased
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings, ignoring padding tokens."""
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


@torch.no_grad()
def encode(texts: list[str], tokenizer, model, batch_size: int = 64,
           max_length: int = 256, device: str = "cpu") -> torch.Tensor:
    """Encode a list of texts → L2-normalised sentence embeddings [N, D]."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
        out = model(**enc)
        emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
        emb = F.normalize(emb, p=2, dim=-1)
        all_embs.append(emb.cpu())
        if (i // batch_size) % 10 == 0:
            print(f"  Encoded {min(i + batch_size, len(texts))}/{len(texts)}", flush=True)
    return torch.cat(all_embs, dim=0)


# ---------------------------------------------------------------------------
# Semantic similarity
# ---------------------------------------------------------------------------

def eval_semantic_similarity(examples: list[dict], tokenizer, model) -> dict:
    """
    Cosine similarity between generated and reference embeddings.
    Returns per-task-type mean similarity.
    """
    print("\n" + "=" * 60)
    print("Evaluation 1: Semantic Similarity (cosine, BERT embeddings)")
    print("=" * 60)

    by_task = defaultdict(list)
    for ex in examples:
        by_task[ex["task_type"]].append(ex)

    results = {}
    for task, exs in sorted(by_task.items()):
        gens  = [e["generated"] for e in exs]
        refs  = [e["reference"]  for e in exs]
        print(f"\n  [{task}] encoding {len(exs)} pairs ...")
        gen_emb = encode(gens, tokenizer, model)
        ref_emb = encode(refs, tokenizer, model)
        # Pairwise cosine similarity (already L2-normalised → dot product)
        sims = (gen_emb * ref_emb).sum(dim=-1)
        mean_sim = sims.mean().item()
        median_sim = sims.median().item()
        print(f"  [{task}] mean cosine: {mean_sim:.4f}  median: {median_sim:.4f}")
        results[task] = {"mean_cosine": round(mean_sim, 4),
                         "median_cosine": round(median_sim, 4),
                         "n": len(exs)}

    overall = sum(r["mean_cosine"] * r["n"] for r in results.values()) / \
              sum(r["n"] for r in results.values())
    print(f"\n  Overall weighted mean cosine: {overall:.4f}")
    results["_overall"] = round(overall, 4)
    return results


# ---------------------------------------------------------------------------
# Retrieval Recall@k
# ---------------------------------------------------------------------------

def eval_retrieval(examples: list[dict], tokenizer, model,
                   ks: list[int] = (1, 3, 5, 10)) -> dict:
    """
    Retrieval evaluation for description + function_explanation tasks.

    Index  = reference texts  (one per unique image — ground-truth descriptions)
    Queries = generated texts (model output from visual features)

    For each query, rank all index entries by cosine similarity.
    Recall@k = fraction of queries where the correct entry appears in top-k.

    This directly tests: "can the model's visual description identify the right
    file from a pool of candidates?" — the Sniper Method localization claim.
    """
    print("\n" + "=" * 60)
    print("Evaluation 2: Retrieval Recall@k")
    print("=" * 60)

    retrieval_tasks = {"description", "function_explanation"}
    exs = [e for e in examples if e["task_type"] in retrieval_tasks]
    print(f"  Tasks: {retrieval_tasks}")
    print(f"  Pool size: {len(exs)} examples")

    if len(exs) < 10:
        print("  Not enough examples — skipping.")
        return {}

    # De-duplicate by id to get one query per example
    seen = set()
    deduped = []
    for e in exs:
        if e["id"] not in seen:
            seen.add(e["id"])
            deduped.append(e)

    gens = [e["generated"] for e in deduped]
    refs = [e["reference"]  for e in deduped]

    print(f"\n  Encoding {len(deduped)} generated texts (queries) ...")
    gen_emb = encode(gens, tokenizer, model)   # [N, D]
    print(f"  Encoding {len(deduped)} reference texts (index) ...")
    ref_emb = encode(refs, tokenizer, model)   # [N, D]

    # Similarity matrix: queries × index  [N, N]
    sim_matrix = gen_emb @ ref_emb.T           # [N, N]

    results = {}
    for k in ks:
        if k > len(deduped):
            continue
        topk_idx = sim_matrix.topk(k, dim=-1).indices  # [N, k]
        correct = torch.arange(len(deduped)).unsqueeze(1)  # [N, 1]
        hits = (topk_idx == correct).any(dim=-1).float()
        recall = hits.mean().item()
        print(f"  Recall@{k:2d}: {recall:.4f}  ({recall*100:.1f}%)")
        results[f"recall@{k}"] = round(recall, 4)

    # Random baseline
    for k in ks:
        if k > len(deduped):
            continue
        baseline = k / len(deduped)
        print(f"  Random@{k:2d}: {baseline:.4f}  ({baseline*100:.1f}%)")

    # Per-task breakdown
    print()
    for task in sorted(retrieval_tasks):
        task_idx = [i for i, e in enumerate(deduped) if e["task_type"] == task]
        if not task_idx:
            continue
        t_idx = torch.tensor(task_idx)
        task_sim = sim_matrix[t_idx][:, t_idx]   # sub-matrix for this task
        task_correct = torch.arange(len(task_idx)).unsqueeze(1)
        topk = task_sim.topk(min(5, len(task_idx)), dim=-1).indices
        r1 = (topk[:, :1] == task_correct).any(dim=-1).float().mean().item()
        r5 = (topk == task_correct).any(dim=-1).float().mean().item()
        print(f"  [{task}] n={len(task_idx)}  Recall@1={r1:.4f}  Recall@{min(5,len(task_idx))}={r5:.4f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",  default="./eval_results_2b.json")
    parser.add_argument("--model",    default="distilbert-base-uncased",
                        help="HuggingFace model for embeddings (default: distilbert-base-uncased)")
    parser.add_argument("--save",     default="./eval_semantic_results.json")
    args = parser.parse_args()

    with open(args.results) as f:
        examples = json.load(f)
    print(f"Loaded {len(examples)} examples from {args.results}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model:  {args.model}")

    print(f"\nLoading tokenizer and model ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    sim_results  = eval_semantic_similarity(examples, tokenizer, model)
    retr_results = eval_retrieval(examples, tokenizer, model)

    output = {
        "model": args.model,
        "n_examples": len(examples),
        "semantic_similarity": sim_results,
        "retrieval": retr_results,
    }

    with open(args.save, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.save}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Overall semantic similarity: {sim_results.get('_overall', 'N/A'):.4f}")
    if retr_results:
        for k in (1, 3, 5):
            key = f"recall@{k}"
            if key in retr_results:
                print(f"  Retrieval Recall@{k}: {retr_results[key]:.4f}  ({retr_results[key]*100:.1f}%)")


if __name__ == "__main__":
    main()
