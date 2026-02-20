"""
Contrastive Pre-training for DeepSeek-Coder-VL Projection Adapter (v3: SigLIP)

Stage 1 of 2-stage training:
  Stage 1 (this script): SigLIP loss aligns visual features with code text embeddings
  Stage 2: Re-run train_projector.py initialized from Stage 1 checkpoint

The adapter learns to map:
  proj(visual_features.mean()) ≈ coder.embed(code_text).mean()

Only the adapter trains (~13.6M params). Vision encoder and coder are frozen.
We only need coder's embedding lookup table — no transformer forward pass.
This makes each step very fast (~0.2-0.5s vs ~10s for generation training).

Loss: SigLIP (Zhai et al., 2023) instead of InfoNCE.
  - InfoNCE uses softmax over all in-batch pairs → quality degrades with small batch
  - SigLIP applies sigmoid to each pair independently → stable at any batch size
  - With batch=64, InfoNCE has 63 negatives; SigLIP treats all 4,032 pairs independently
  - Random-init SigLIP baseline: log(2) ≈ 0.693 (vs InfoNCE log(64) ≈ 4.16)
  - Target: val_loss < 0.3, avg positive-pair cosine > 0.5

Temperature: learnable nn.Parameter (initialized to 1.0, adapts during training).
  - Fixed temp=0.07 from step 1 is too aggressive for a randomly-initialized adapter
  - Learnable temp finds the right sharpness for the current alignment level

VRAM budget (V100 32 GB):
  Coder model (4-bit frozen):      ~8-10 GB
  Adapter (fp32, 13.6M params):    ~55 MB
  Batch 64 × [256, 1280] fp16:     ~42 MB
  Projected [64, 2048] fp32:       ~0.5 MB
  Text embeddings [64, 256, 2048]: ~100 MB
  Total:                           ~10-11 GB

Usage:
    python contrastive_pretrain.py --features_dir ./precomputed_features

After completion, run train_projector.py with:
    --init_from ./checkpoints/contrastive_v3/best.pt
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm

from projector import ProjectionAdapter


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ContrastiveDataset(Dataset):
    """
    One item per unique image: (visual_features [num_tokens, 1280], text_anchor).

    text_anchor = all assistant answers for that image concatenated.
    This gives the richest code-semantic signal available in the manifest
    without requiring access to the original source files.

    All feature tensors are cached in CPU RAM at init time.
    """

    def __init__(self, manifest_paths, features_dir):
        """
        Args:
            manifest_paths: str or list of str — one or more .jsonl manifest paths.
                            All are merged into a single dataset.
            features_dir:   str — directory containing precomputed .pt feature files.
        """
        self.features_dir = Path(features_dir)

        if isinstance(manifest_paths, str):
            manifest_paths = [manifest_paths]

        # Group ALL assistant answers by image path (across all manifests)
        answers_by_image = defaultdict(list)
        for manifest_path in manifest_paths:
            with open(manifest_path) as f:
                for line in f:
                    ex = json.loads(line)
                    img = ex["image"]
                    for turn in ex["conversations"]:
                        # Handle both role formats used across scripts
                        role = turn.get("role") or turn.get("from", "")
                        if role in ("assistant", "gpt"):
                            answers_by_image[img].append(turn["content"])

        # Load features and build item list (one entry per unique image)
        self._cache = {}
        self.items = []   # list of (image_path, text_anchor)

        missing = 0
        for img, answers in answers_by_image.items():
            feat_file = self.features_dir / (Path(img).stem + ".pt")
            if not feat_file.exists():
                missing += 1
                continue
            if img not in self._cache:
                self._cache[img] = torch.load(feat_file, map_location="cpu")
            text_anchor = " ".join(answers)
            self.items.append((img, text_anchor))

        print(f"  {len(self.items)} unique images loaded  ({missing} missing features)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img, text = self.items[idx]
        features = self._cache[img]   # [num_tokens, 1280] fp16
        return features, text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def masked_mean(emb, mask):
    """
    Padding-aware mean pooling.

    Args:
        emb:  [B, seq, dim] — token embeddings (may be fp16)
        mask: [B, seq]      — attention mask (1=real, 0=padding)

    Returns:
        [B, dim] fp32
    """
    mask_f = mask.unsqueeze(-1).float()
    return (emb.float() * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)


def siglip_loss(vis_pooled, txt_pooled, log_temp, bias):
    """
    SigLIP loss (Zhai et al., 2023) — sigmoid binary classification per pair.

    Args:
        vis_pooled: [B, dim] — L2-normalized visual embeddings
        txt_pooled: [B, dim] — L2-normalized text embeddings
        log_temp:   scalar nn.Parameter — log of temperature (learnable)
        bias:       scalar nn.Parameter — additive logit bias (learnable, init -10)

    Each (i, j) pair is classified independently:
      label[i,j] = +1 if i==j (positive pair), -1 otherwise (negative)
      logit[i,j] = vis[i]·txt[j] / temp + bias
      loss = -mean(logsigmoid(label * logit))

    The bias (init=-10) is critical to prevent representation collapse.
    Without it, the 63:1 negative:positive imbalance causes the model to push
    all embeddings anti-aligned (cosine≈-0.93), which correctly classifies 63/64
    pairs as negative while abandoning the 1 positive pair. With bias=-10, the
    model initialises to "predict negative for everything," so positive pairs
    produce large gradients from step 1, driving alignment.

    Random-init baseline: ~log(2) ≈ 0.693 (vs InfoNCE log(batch_size) ≈ 4.16)
    """
    B = vis_pooled.shape[0]
    temperature = log_temp.exp().clamp(max=100.0)
    logits = vis_pooled @ txt_pooled.T / temperature + bias   # [B, B]
    labels = 2 * torch.eye(B, device=logits.device) - 1      # +1 diagonal, -1 off-diag
    return -F.logsigmoid(labels * logits).mean()


def avg_pos_cosine(vis_pooled, txt_pooled):
    """
    Mean cosine similarity of positive pairs (diagonal entries).
    Both inputs should already be L2-normalized.
    Range [-1, 1]; target > 0.5 after training.
    """
    return (vis_pooled * txt_pooled).sum(dim=-1).mean().item()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(adapter, embed_fn, tokenizer, loader, log_temp, bias, max_text_tokens, device):
    adapter.eval()
    total_loss, total_cos, n = 0.0, 0.0, 0

    for feats, texts in loader:
        feats = feats.to(device)                  # [B, T, 1280]

        # Visual side
        vis_emb = adapter(feats.float())           # [B, T, 2048]
        vis_pooled = vis_emb.mean(dim=1)           # [B, 2048]
        vis_pooled = F.normalize(vis_pooled, dim=-1)

        # Text side
        tok = tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_text_tokens,
            add_special_tokens=True,
        ).to(device)
        txt_emb = embed_fn(tok.input_ids)          # [B, seq, 2048]
        txt_pooled = masked_mean(txt_emb, tok.attention_mask)   # [B, 2048]
        txt_pooled = F.normalize(txt_pooled, dim=-1)

        loss = siglip_loss(vis_pooled, txt_pooled, log_temp, bias)
        cos  = avg_pos_cosine(vis_pooled, txt_pooled)

        total_loss += loss.item()
        total_cos  += cos
        n += 1

    return total_loss / max(n, 1), total_cos / max(n, 1)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(adapter, log_temp, bias, optimizer, scheduler, epoch, step, val_loss, ckpt_dir, name):
    path = os.path.join(ckpt_dir, f"{name}.pt")
    torch.save({
        "adapter_state_dict":    adapter.state_dict(),
        "log_temp":              log_temp.data.cpu(),
        "bias":                  bias.data.cpu(),
        "optimizer_state_dict":  optimizer.state_dict(),
        "scheduler_state_dict":  scheduler.state_dict(),
        "epoch":                 epoch,
        "global_step":           step,
        "val_loss":              val_loss,
    }, path)
    print(f"  Saved checkpoint: {path}  (val_loss={val_loss:.4f}  temp={log_temp.exp().item():.4f}  bias={bias.item():.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Contrastive pre-training of projection adapter"
    )
    parser.add_argument("--features_dir",    default="./precomputed_features")
    parser.add_argument(
        "--train_manifest",
        nargs="+",
        default=["Data Crawling/output/manifests/train.jsonl"],
        help="One or more train manifest .jsonl files (space-separated). "
             "All are merged into a single training dataset.",
    )
    parser.add_argument(
        "--val_manifest",
        nargs="+",
        default=["Data Crawling/output/manifests/val.jsonl"],
        help="One or more val manifest .jsonl files (space-separated).",
    )
    parser.add_argument("--coder_model",     default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct")
    parser.add_argument("--batch_size",      type=int,   default=64)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--epochs",          type=int,   default=100)
    parser.add_argument("--temperature",     type=float, default=1.0,
                        help="Initial temperature for learnable log_temp parameter. "
                             "Will adapt during training. Default 1.0 (softer than "
                             "fixed 0.07 used in v1/v2, appropriate for random init).")
    parser.add_argument("--max_text_tokens", type=int,   default=256)
    parser.add_argument("--checkpoint_dir",  default="./checkpoints/contrastive_v3")
    parser.add_argument("--log_steps",       type=int,   default=10)
    parser.add_argument(
        "--init_from",
        default=None,
        help="Optional path to adapter checkpoint to initialize from "
             "(e.g. ./checkpoints/phase2a/best.pt). Default: random init.",
    )
    args = parser.parse_args()

    device = "cuda"
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ==================================================================
    # 1. Load coder model (4-bit, frozen) — only embedding layer needed
    # ==================================================================
    print("=" * 60)
    print("LOADING CODER MODEL (4-bit, fp16)")
    print("=" * 60)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    coder = AutoModelForCausalLM.from_pretrained(
        args.coder_model,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.coder_model, trust_remote_code=True,
    )

    # Keep vocab consistent with generation training
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<image>", "<img_start>", "<img_end>"]}
    )
    coder.resize_token_embeddings(len(tokenizer))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Freeze entire coder — Stage 1 only needs the embedding lookup
    for p in coder.parameters():
        p.requires_grad = False
    coder.eval()

    embed_fn = coder.get_input_embeddings()
    coder_dim = coder.config.hidden_size
    print(f"  hidden_size={coder_dim}")
    print(f"  Coder frozen, embed_fn extracted (no transformer forward needed)\n")

    # ==================================================================
    # 2. Create adapter
    # ==================================================================
    print("Creating adapter ...")
    adapter = ProjectionAdapter(vision_dim=1280, hidden_dim=4096, coder_dim=coder_dim)

    # Learnable temperature and bias (log scale / direct value).
    # NOTE: .to(device) must be on the raw tensor BEFORE nn.Parameter wrapping —
    # calling .to() on a Parameter returns a plain non-leaf Tensor (can't optimize).
    log_temp = nn.Parameter(
        torch.log(torch.tensor(args.temperature, dtype=torch.float32)).to(device)
    )
    # Bias initialized to -10: prevents representation collapse caused by 63:1
    # negative:positive class imbalance in SigLIP without bias.
    bias = nn.Parameter(torch.tensor(-10.0).to(device))

    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu")
        # Support both full checkpoint dicts and raw state_dicts
        state = ckpt.get("adapter_state_dict", ckpt)
        adapter.load_state_dict(state)
        if "log_temp" in ckpt:
            log_temp.data.copy_(ckpt["log_temp"].to(device))
            print(f"  Restored log_temp → temperature={log_temp.exp().item():.4f}")
        if "bias" in ckpt:
            bias.data.copy_(ckpt["bias"].to(device))
            print(f"  Restored bias={bias.item():.4f}")
        print(f"  Initialized from: {args.init_from}")
    else:
        print("  Initialized with random weights")

    adapter = adapter.to(device)
    print(f"  Parameters: {adapter.num_parameters():,}")
    print(f"  Initial temperature: {log_temp.exp().item():.4f} (learnable)")
    print(f"  Initial bias:        {bias.item():.4f} (learnable, -10 prevents collapse)\n")

    # ==================================================================
    # 3. Datasets (one item per unique image, features cached in CPU RAM)
    # ==================================================================
    print("Loading datasets ...")
    print(f"  Train manifests: {args.train_manifest}")
    print(f"  Val manifests:   {args.val_manifest}")
    train_ds = ContrastiveDataset(args.train_manifest, args.features_dir)
    val_ds   = ContrastiveDataset(args.val_manifest,   args.features_dir)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # ==================================================================
    # 4. Optimizer & scheduler
    # ==================================================================
    optimizer = AdamW(
        list(adapter.parameters()) + [log_temp, bias],
        lr=args.lr, weight_decay=1e-2,
    )
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = max(1, int(total_steps * 0.03))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    siglip_baseline = torch.log(torch.tensor(2.0)).item()   # ≈ 0.693

    print(f"\nTraining plan:")
    print(f"  Loss                 : SigLIP (sigmoid per-pair, batch-size independent)")
    print(f"  Unique train images  : {len(train_ds)}")
    print(f"  Unique val images    : {len(val_ds)}")
    print(f"  Batch size           : {args.batch_size}")
    print(f"  Steps/epoch          : {len(train_loader)}")
    print(f"  Epochs               : {args.epochs}")
    print(f"  Total steps          : {total_steps}")
    print(f"  Warmup steps         : {warmup_steps}")
    print(f"  Initial temperature  : {log_temp.exp().item():.4f} (learnable)")
    print(f"  Random-init baseline : {siglip_baseline:.3f}  (target < 0.3)\n")

    # ==================================================================
    # 5. Training loop
    # ==================================================================
    print("=" * 60)
    print("STARTING CONTRASTIVE PRE-TRAINING")
    print("=" * 60 + "\n")

    best_val_loss = float("inf")
    global_step   = 0

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        epoch_loss, epoch_cos, n_batches = 0.0, 0.0, 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{args.epochs}", leave=False)

        for feats, texts in progress:
            feats = feats.to(device)                  # [B, T, 1280] fp16

            optimizer.zero_grad()

            # --- Visual side ---
            vis_emb    = adapter(feats.float())        # [B, T, 2048] fp32
            vis_pooled = vis_emb.mean(dim=1)           # [B, 2048]
            vis_pooled = F.normalize(vis_pooled, dim=-1)

            # --- Text side (frozen embed lookup, no grad needed) ---
            with torch.no_grad():
                tok = tokenizer(
                    list(texts),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_text_tokens,
                    add_special_tokens=True,
                ).to(device)
                txt_emb    = embed_fn(tok.input_ids)   # [B, seq, 2048] fp16
                txt_pooled = masked_mean(txt_emb, tok.attention_mask)   # [B, 2048] fp32
                txt_pooled = F.normalize(txt_pooled, dim=-1)

            # --- SigLIP loss ---
            loss = siglip_loss(vis_pooled, txt_pooled, log_temp, bias)
            cos  = avg_pos_cosine(vis_pooled.detach(), txt_pooled)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss  += loss.item()
            epoch_cos   += cos
            n_batches   += 1

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                pos_cos=f"{cos:.3f}",
                temp=f"{log_temp.exp().item():.3f}",
                bias=f"{bias.item():.2f}",
            )

            if global_step % args.log_steps == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"  step={global_step:4d}  loss={loss.item():.4f}  "
                      f"pos_cos={cos:.3f}  temp={log_temp.exp().item():.3f}  "
                      f"bias={bias.item():.3f}  lr={lr:.2e}")

        # --- End of epoch: validate and report ---
        avg_train_loss = epoch_loss / max(n_batches, 1)
        avg_train_cos  = epoch_cos  / max(n_batches, 1)

        val_loss, val_cos = evaluate(
            adapter, embed_fn, tokenizer, val_loader,
            log_temp, bias, args.max_text_tokens, device,
        )

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={avg_train_loss:.4f}  train_cos={avg_train_cos:.3f}  "
            f"val_loss={val_loss:.4f}  val_cos={val_cos:.3f}  "
            f"temp={log_temp.exp().item():.4f}  bias={bias.item():.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                adapter, log_temp, bias, optimizer, scheduler,
                epoch, global_step, val_loss,
                args.checkpoint_dir, "best",
            )

    # Final save: adapter weights only (clean, for easy loading in train_projector.py)
    final_path = os.path.join(args.checkpoint_dir, "adapter_final.pt")
    torch.save(adapter.state_dict(), final_path)

    print(f"\nSaved final adapter: {final_path}")
    print(f"Best val loss: {best_val_loss:.4f}  (SigLIP random-init baseline: 0.693)")
    print(f"Final temperature: {log_temp.exp().item():.4f}")
    print(f"Final bias:        {bias.item():.4f}")
    print("Contrastive pre-training complete!")
    print(f"\nNext step:")
    print(f"  sbatch coder_vl/train_phase2a.sh")
    print(f"  (train_phase2a.sh uses --init_from {args.checkpoint_dir}/best.pt)")


if __name__ == "__main__":
    main()
