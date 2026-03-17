"""
eval_retrieval.py — Phase 1.10: ColBERT-style retrieval evaluation

Tests whether the ConvRoPEProjector (Phase 1.9 / Phase 2 aligned weights) can
match natural-language repo descriptions to the correct code-image files via
MaxSim late interaction.

Pipeline:
  Query text → DeepSeek embed layer → L2-norm → [T_text, 2048]
  Code image feature → ConvRoPEProjector → L2-norm → [256, 2048]
  Score = sum of per-query-token max cosine similarities  (ColBERT MaxSim)
"""

import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# Paths (edit if you move things around)
# ---------------------------------------------------------------------------
REPO_ROOT      = Path("/home/ad.msoe.edu/gloriosog/CoderOCR/OCR-Coder")
FEATURES_DIR   = REPO_ROOT / "MVV/Phase_1_9/a/data/features"
GT_JSONL       = REPO_ROOT / "MVV/Phase_1_9/a/data/ground_truth.jsonl"
MANIFEST_JSONL = REPO_ROOT / "MVV/Phase_1_1/data_mvv/manifest.jsonl"
CHECKPOINT     = REPO_ROOT / "MVV/Phase_2/checkpoints/best_aligned.pt"
RESULTS_DIR    = REPO_ROOT / "MVV/Phase_1_10/results"
RESULTS_FILE   = RESULTS_DIR / "retrieval_results.json"

LLM_MODEL_ID   = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"

# Add repo root to sys.path so `MVV/Phase_1_9/a/scripts/model.py` is importable
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Editable query dict  — one entry per repo in the haystack
# ---------------------------------------------------------------------------
QUERIES = {
    "black":  "Python source code formatter with deeply nested AST traversal and recursive tree walking logic",
    "flask":  "Web framework with decorator-based routing, request context middleware, and HTTP handler chains",
    "django": "ORM model class definitions with multi-level class-based inheritance and database field declarations",
    "numpy":  "Dense low-level numerical computation with tightly packed array indexing and mathematical operations",
}

# Number of files randomly sampled per repo to form the balanced haystack
HAYSTACK_PER_REPO = 25
RANDOM_SEED       = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_manifests():
    """
    Returns a dict: source_file -> repo  by joining ground_truth.jsonl
    (which has stem + source_file) with manifest.jsonl (which has
    source_file + repo).
    """
    # Build source_file → repo from the main manifest
    sf2repo = {}
    with open(MANIFEST_JSONL) as f:
        for line in f:
            rec = json.loads(line)
            sf2repo[rec["source_file"]] = rec["repo"]

    # Build stem → repo via ground_truth.jsonl
    stem2repo = {}
    stem2sf   = {}
    with open(GT_JSONL) as f:
        for line in f:
            rec = json.loads(line)
            stem = rec["stem"]
            sf   = rec["source_file"]
            stem2sf[stem] = sf
            if sf in sf2repo:
                stem2repo[stem] = sf2repo[sf]
            # else: repo unknown — skip silently

    return stem2repo, stem2sf


def build_haystack(stem2repo: dict) -> dict:
    """
    For each repo key in QUERIES, sample exactly HAYSTACK_PER_REPO stems.
    Returns dict: repo -> list[stem]
    """
    rng = random.Random(RANDOM_SEED)

    # Group all known stems by repo
    repo2stems: dict[str, list] = {}
    for stem, repo in stem2repo.items():
        if repo in QUERIES:                           # only keep relevant repos
            repo2stems.setdefault(repo, []).append(stem)

    haystack: dict[str, list] = {}
    for repo in QUERIES:
        candidates = sorted(repo2stems.get(repo, []))  # sort for reproducibility
        if len(candidates) < HAYSTACK_PER_REPO:
            print(f"  [WARN] repo '{repo}' has only {len(candidates)} files "
                  f"(need {HAYSTACK_PER_REPO}) — skipping")
            continue
        haystack[repo] = rng.sample(candidates, HAYSTACK_PER_REPO)
        print(f"  repo '{repo}': sampled {HAYSTACK_PER_REPO}/{len(candidates)} files")

    return haystack


def load_projector() -> "ConvRoPEProjector":
    """Load ConvRoPEProjector with aligned weights, eval mode, cuda fp16."""
    from MVV.Phase_1_9.a.scripts.model import ConvRoPEProjector  # noqa: E402

    projector = ConvRoPEProjector(feat_dim=1152, proj_dim=2048)

    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    projector.load_state_dict(ckpt["projector_state_dict"])

    projector.eval().cuda()
    print(f"  Projector loaded from {CHECKPOINT} "
          f"(epoch={ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', '?'):.4f})")
    return projector


def load_llm():
    """
    Load DeepSeek-Coder-V2-Lite-Instruct in 8-bit with all weights frozen.
    Returns (tokenizer, embed_layer).
    """
    print(f"  Loading tokenizer from {LLM_MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True)

    print(f"  Loading LLM (8-bit) ...")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    llm.requires_grad_(False)
    llm.eval()

    embed_layer = llm.get_input_embeddings()
    print(f"  LLM loaded — embed weight shape: {embed_layer.weight.shape}")
    return tokenizer, embed_layer


@torch.no_grad()
def embed_query(text: str, tokenizer, embed_layer) -> torch.Tensor:
    """
    Tokenize `text`, pass through embedding layer, L2-normalise per token.
    Returns: [T_text, 2048] on cuda.
    """
    tokens = tokenizer(text, return_tensors="pt").input_ids.cuda()  # [1, T]
    emb = embed_layer(tokens).squeeze(0)          # [T, 2048]
    emb = F.normalize(emb, dim=-1)                # per-token L2 norm
    return emb                                    # [T_text, 2048]


@torch.no_grad()
def project_feature(feat_path: Path, projector) -> torch.Tensor:
    """
    Load a raw SigLIP feature file and project it.
    Returns: [256, 2048] on cuda, L2-normalised per token.
    """
    feat = torch.load(feat_path, map_location="cpu").unsqueeze(0).cuda().float()  # [1, 1024, 1152]
    proj = projector(feat).squeeze(0)                  # [256, 2048]
    proj = F.normalize(proj, dim=-1)                   # per-token L2 norm
    return proj                                        # [256, 2048]


@torch.no_grad()
def maxsim_score(query_emb: torch.Tensor, doc_emb: torch.Tensor) -> float:
    """
    ColBERT MaxSim late interaction score.

    query_emb : [T_text, 2048]  L2-normalised
    doc_emb   : [256,   2048]   L2-normalised

    Score = sum_{t in query} max_{d in doc} (q_t · d_d)
    """
    # [T_text, 256] similarity matrix
    sim = torch.matmul(query_emb.float(), doc_emb.float().T)
    # Max over document tokens per query token, then sum
    score = sim.max(dim=1)[0].sum().item()
    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Build haystack
    # -----------------------------------------------------------------------
    print("\n[1/4] Building haystack ...")
    stem2repo, _stem2sf = load_manifests()
    haystack = build_haystack(stem2repo)

    if not haystack:
        print("ERROR: no repos found in haystack — check manifest paths.")
        sys.exit(1)

    # Flat list of (stem, repo) for scoring
    all_files = [(stem, repo) for repo, stems in haystack.items() for stem in stems]
    print(f"  Total haystack size: {len(all_files)} files "
          f"across {len(haystack)} repos")

    # -----------------------------------------------------------------------
    # 2. Load models
    # -----------------------------------------------------------------------
    print("\n[2/4] Loading models ...")
    projector              = load_projector()
    tokenizer, embed_layer = load_llm()

    # -----------------------------------------------------------------------
    # 3. Run retrieval for each query
    # -----------------------------------------------------------------------
    print("\n[3/4] Running retrieval ...")
    all_results = {}

    recall1_hits = 0
    recall5_hits = 0

    for query_repo, query_text in QUERIES.items():
        if query_repo not in haystack:
            print(f"  [SKIP] '{query_repo}' not in haystack")
            continue

        print(f"\n  Query: '{query_repo}'")
        print(f"  Text : {query_text[:80]}...")

        # Embed query text
        q_emb = embed_query(query_text, tokenizer, embed_layer)  # [T, 2048]
        print(f"  Query tokens: {q_emb.shape[0]}")

        # Score every file in haystack
        scores = []
        for stem, repo in all_files:
            feat_path = FEATURES_DIR / f"{stem}.pt"
            if not feat_path.exists():
                continue                          # skip missing features silently
            doc_emb = project_feature(feat_path, projector)  # [256, 2048]
            score   = maxsim_score(q_emb, doc_emb)
            scores.append({"stem": stem, "repo": repo, "score": score})

        # Rank descending
        scores.sort(key=lambda x: x["score"], reverse=True)

        # Recall metrics
        r1 = int(scores[0]["repo"] == query_repo) if scores else 0
        r5 = int(any(s["repo"] == query_repo for s in scores[:5])) if scores else 0
        recall1_hits += r1
        recall5_hits += r5

        # Print top-10
        print(f"  --- Top 10 results (target repo: {query_repo}) ---")
        for rank, s in enumerate(scores[:10], 1):
            marker = "<< TARGET" if s["repo"] == query_repo else ""
            print(f"    {rank:2d}. {s['stem'][:55]:<55}  repo={s['repo']:<8}  "
                  f"score={s['score']:6.3f} {marker}")

        print(f"  Recall@1={r1}  Recall@5={r5}")

        all_results[query_repo] = {
            "query_text": query_text,
            "target_repo": query_repo,
            "recall_at_1": r1,
            "recall_at_5": r5,
            "ranked": scores,        # full ranked list saved to JSON
        }

    # -----------------------------------------------------------------------
    # 4. Summary table + save JSON
    # -----------------------------------------------------------------------
    print("\n[4/4] Summary")
    print()
    print(f"{'Query Repo':<12} | {'Recall@1':^8} | {'Recall@5':^8}")
    print(f"{'-'*12}-+-{'-'*8}-+-{'-'*8}")
    n_queries = len(all_results)
    for qr, res in all_results.items():
        r1 = res["recall_at_1"]
        r5 = res["recall_at_5"]
        print(f"{qr:<12} |   {r1}/1    |   {r5}/1   ")
    print(f"{'-'*12}-+-{'-'*8}-+-{'-'*8}")
    print(f"{'Overall':<12} |  {recall1_hits}/{n_queries}    |  {recall5_hits}/{n_queries}   ")
    print()
    print(f"Random baseline: {HAYSTACK_PER_REPO}/{len(all_files)} = "
          f"{HAYSTACK_PER_REPO / max(len(all_files), 1) * 100:.1f}%  "
          f"(expected ~25% for Recall@1 and ~5x for Recall@5)")

    # Save results JSON
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
