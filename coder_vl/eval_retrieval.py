"""
eval_retrieval.py — Retrieval Recall@k evaluation for Sniper Method localization.

For each image in the val set (description/function_explanation tasks only):
  1. Compute visual_emb = mean_pool(adapter(precomputed_features))  [2048D]
  2. Compute text_emb   = mean_pool(coder_last_hidden(answer_text)) [2048D]
  3. Build full N x N cosine similarity matrix on CPU
  4. For image i, rank all N texts by similarity; check if text_i is in top-k
  5. Report Recall@1, Recall@5, Recall@10 — overall and per task_type

Usage:
    python coder_vl/eval_retrieval.py \
        --checkpoint  ./checkpoints/phase2b_v2/best.pt \
        --val_manifest data_v2b/manifests/val.jsonl \
        --features_dir ./precomputed_features_tiled \
        --output       retrieval_results.json
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType

from projector import ProjectionAdapter


CONTRASTIVE_TASKS = {"description", "function_explanation"}


def recall_at_k(sim_matrix, k_values=(1, 5, 10)):
    """
    Compute Recall@k from a cosine similarity matrix.

    Args:
        sim_matrix: [N, N] tensor — sim_matrix[i, j] = cosine_sim(visual_i, text_j)
                    Diagonal is the positive pair (image i matched with its own text i).
        k_values:   tuple of k values to evaluate

    Returns:
        dict mapping k -> recall float in [0, 1]
    """
    N = sim_matrix.size(0)
    results = {}
    for k in k_values:
        k_eff = min(k, N)
        _, top_indices = sim_matrix.topk(k_eff, dim=1)
        correct = (top_indices == torch.arange(N, device=sim_matrix.device).unsqueeze(1))
        results[k] = correct.any(dim=1).float().mean().item()
    return results


@torch.no_grad()
def encode_visuals(examples, features_dir, adapter, device, batch_size=32):
    """Encode all examples through adapter -> mean_pool -> L2-normalize."""
    features_dir = Path(features_dir)
    all_embs = []

    for i in range(0, len(examples), batch_size):
        batch_ex = examples[i : i + batch_size]
        feats = []
        for ex in batch_ex:
            feat_file = features_dir / (Path(ex["image"]).stem + ".pt")
            feats.append(torch.load(feat_file, map_location="cpu"))
        feat_batch = torch.stack(feats).to(device)            # [B, T, 1280]
        projected  = adapter(feat_batch.float()).half()       # [B, T, 2048]
        emb = F.normalize(projected.float().mean(dim=1), dim=-1)  # [B, 2048]
        all_embs.append(emb.cpu())

    return torch.cat(all_embs, dim=0)  # [N, 2048]


@torch.no_grad()
def encode_texts(examples, tokenizer, coder, device, batch_size=16):
    """Encode answer texts through full coder forward -> last hidden state -> mean_pool -> L2-norm."""
    all_embs = []

    for i in range(0, len(examples), batch_size):
        batch_ex = examples[i : i + batch_size]
        answers  = [ex["conversations"][1]["content"] for ex in batch_ex]

        tok = tokenizer(
            answers,
            return_tensors="pt",
            max_length=256,
            truncation=True,
            padding=True,
        ).input_ids.to(device)

        hs  = coder(input_ids=tok, output_hidden_states=True).hidden_states[-1]  # [B, T, 2048]
        emb = F.normalize(hs.float().mean(dim=1), dim=-1)                        # [B, 2048]
        all_embs.append(emb.cpu())

    return torch.cat(all_embs, dim=0)  # [N, 2048]


def main():
    parser = argparse.ArgumentParser(description="Retrieval Recall@k evaluation")
    parser.add_argument("--checkpoint",   required=True,
                        help="Path to checkpoint (e.g. checkpoints/phase2b_v2/best.pt)")
    parser.add_argument("--val_manifest", default="data_v2b/manifests/val.jsonl")
    parser.add_argument("--features_dir", default="./precomputed_features_tiled")
    parser.add_argument("--output",       default="retrieval_results.json")
    parser.add_argument("--coder_model",  default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct")
    parser.add_argument("--batch_size",   type=int, default=16)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load val manifest — filter to contrastive tasks only
    # ------------------------------------------------------------------
    examples = []
    with open(args.val_manifest) as f:
        for line in f:
            ex = json.loads(line)
            if ex.get("task_type", "") in CONTRASTIVE_TASKS:
                examples.append(ex)
    print(f"Filtered to {len(examples)} examples ({', '.join(CONTRASTIVE_TASKS)})")

    if len(examples) < 2:
        print("ERROR: need at least 2 examples for retrieval evaluation.")
        return

    # ------------------------------------------------------------------
    # 2. Load coder (4-bit, for text encoding only)
    # ------------------------------------------------------------------
    print("Loading coder model (4-bit) ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    coder = AutoModelForCausalLM.from_pretrained(
        args.coder_model,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.coder_model, trust_remote_code=True)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<image>", "<img_start>", "<img_end>"]}
    )
    coder.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA config (needed to load LoRA weights from checkpoint)
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    coder = get_peft_model(coder, lora_config)

    # ------------------------------------------------------------------
    # 3. Load checkpoint weights
    # ------------------------------------------------------------------
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    coder_dim = coder.config.hidden_size
    adapter = ProjectionAdapter(vision_dim=1280, hidden_dim=4096, coder_dim=coder_dim)
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    adapter = adapter.to(device)
    adapter.eval()

    coder.load_state_dict(ckpt["lora_state_dict"], strict=False)
    coder.eval()

    if "log_temp" in ckpt:
        print(f"  Contrastive params: temp={ckpt['log_temp'].exp().item():.3f}  "
              f"bias={ckpt['bias'].item():.3f}")

    # ------------------------------------------------------------------
    # 4. Encode all examples
    # ------------------------------------------------------------------
    print(f"Encoding {len(examples)} visual embeddings ...")
    visual_embs = encode_visuals(examples, args.features_dir, adapter, device, args.batch_size)

    print(f"Encoding {len(examples)} text embeddings ...")
    text_embs = encode_texts(examples, tokenizer, coder, device, args.batch_size)

    # ------------------------------------------------------------------
    # 5. Build similarity matrix and compute Recall@k
    # ------------------------------------------------------------------
    print("Computing similarity matrix ...")
    # Both tensors on CPU for full-matrix computation
    sim_matrix = torch.matmul(visual_embs, text_embs.T)  # [N, N]

    k_values = (1, 5, 10)
    overall = recall_at_k(sim_matrix, k_values)

    # Per task_type breakdown
    task_types = [ex.get("task_type", "") for ex in examples]
    per_task = {}
    for task in CONTRASTIVE_TASKS:
        idx = [i for i, t in enumerate(task_types) if t == task]
        if len(idx) < 2:
            continue
        idx_t = torch.tensor(idx)
        sub_sim = sim_matrix[idx_t][:, idx_t]  # [M, M] — re-rank within task subset
        # Remap diagonal: position j in sub_sim is correct for row j
        per_task[task] = recall_at_k(sub_sim, k_values)

    # ------------------------------------------------------------------
    # 6. Report and save
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("RETRIEVAL RESULTS")
    print("=" * 50)
    print(f"N = {len(examples)} examples")
    print("\nOverall:")
    for k, r in overall.items():
        print(f"  Recall@{k:2d} = {r:.4f} ({r*100:.1f}%)")

    for task, task_results in per_task.items():
        n_task = sum(1 for t in task_types if t == task)
        print(f"\n{task} (n={n_task}):")
        for k, r in task_results.items():
            print(f"  Recall@{k:2d} = {r:.4f} ({r*100:.1f}%)")

    # Positive cosine similarity (diagonal of sim_matrix)
    pos_cos = sim_matrix.diagonal().mean().item()
    print(f"\nMean positive cosine similarity: {pos_cos:.4f}")

    results = {
        "checkpoint": args.checkpoint,
        "n_examples": len(examples),
        "overall": {f"recall@{k}": v for k, v in overall.items()},
        "per_task": {
            task: {f"recall@{k}": v for k, v in res.items()}
            for task, res in per_task.items()
        },
        "mean_pos_cos": pos_cos,
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
