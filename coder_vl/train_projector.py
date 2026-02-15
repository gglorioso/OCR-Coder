"""
Phase 2a Training — Projection Adapter (Simplified)

Trains the projection adapter to map pre-computed vision features into the
coder model's embedding space.  Everything that isn't strictly needed is gone:
no DDP, no wandb, no gradient checkpointing, no vision encoder in VRAM.

What runs on GPU:
  - Coder model  (8-bit quantized, frozen)   ~8-10 GB
  - Adapter      (13.6M params, trainable)   ~55 MB
  - Activations  (for backprop through coder) ~3-5 GB
  Total: ~13-17 GB  →  fits on a single V100 (32 GB)

Usage:
    python train_projector.py --features_dir ./precomputed_features
"""

import os
import json
import argparse
from pathlib import Path

import torch
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

class PrecomputedDataset(Dataset):
    """
    Loads pre-computed vision features + tokenized text for each example.

    All unique feature tensors are cached in CPU memory at init time
    (~1.4 GB for 2175 images in fp16).  Individual __getitem__ calls
    are just dict lookups + tokenization.
    """

    def __init__(self, manifest_path, features_dir, tokenizer, max_seq_length=2048):
        self.features_dir = Path(features_dir)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # Load manifest
        self.examples = []
        with open(manifest_path) as f:
            for line in f:
                self.examples.append(json.loads(line))
        print(f"  Loaded {len(self.examples)} examples from {manifest_path}")

        # Cache all unique feature files in memory
        self._cache = {}
        self._load_features()

        # Drop examples whose features are missing (e.g. decompression bombs)
        before = len(self.examples)
        self.examples = [ex for ex in self.examples if ex["image"] in self._cache]
        if before != len(self.examples):
            print(f"  Filtered out {before - len(self.examples)} examples with missing features "
                  f"({len(self.examples)} remaining)")

    def _load_features(self):
        loaded, missing = 0, 0
        for ex in self.examples:
            img = ex["image"]
            if img in self._cache:
                continue
            feat_file = self.features_dir / (Path(img).stem + ".pt")
            if feat_file.exists():
                self._cache[img] = torch.load(feat_file, map_location="cpu")
                loaded += 1
            else:
                missing += 1
        print(f"  Cached {loaded} unique feature files  ({missing} missing)")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # Pre-computed vision features  [num_tokens, 1280]
        features = self._cache[ex["image"]]

        # Build conversation text
        conv = ex["conversations"]
        user_msg = conv[0]["content"]
        assistant_msg = conv[1]["content"]
        text = f"User: {user_msg}\n\nAssistant: {assistant_msg}"

        # Tokenize
        tok = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
        )
        input_ids = tok["input_ids"].squeeze(0)

        # Labels: mask everything up to and including "Assistant:" with -100
        labels = input_ids.clone()
        asst_tokens = self.tokenizer.encode("Assistant:", add_special_tokens=False)
        pos = _find_subseq(input_ids.tolist(), asst_tokens)
        if pos != -1:
            labels[: pos + len(asst_tokens)] = -100
        else:
            # Fallback: mask first half
            labels[: len(labels) // 2] = -100

        # Mask padding
        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        return {"input_ids": input_ids, "labels": labels, "features": features}


def _find_subseq(lst, sub):
    """Find the start index of *sub* inside *lst*, or -1."""
    n = len(sub)
    for i in range(len(lst) - n + 1):
        if lst[i : i + n] == sub:
            return i
    return -1


# ---------------------------------------------------------------------------
# Token-replacement helpers
# ---------------------------------------------------------------------------

def replace_image_tokens(input_ids, projected, image_token_id, embed_fn):
    """
    Replace <image> placeholder in each sequence with projected visual tokens.

    Args:
        input_ids:      [B, seq]           token ids (contains one <image> each)
        projected:      [B, vis_tok, dim]  adapter output in fp16
        image_token_id: int                id of <image> token
        embed_fn:       nn.Embedding       coder model's input embedding layer

    Returns:
        embeds:  [B, new_seq, dim]   combined text + visual embeddings
        mask:    [B, new_seq]        attention mask (1 = attend, 0 = pad)
    """
    batch = input_ids.size(0)
    text_embeds = embed_fn(input_ids)  # [B, seq, dim]

    out_embeds = []
    for i in range(batch):
        positions = (input_ids[i] == image_token_id).nonzero(as_tuple=True)[0]
        if len(positions) == 0:
            out_embeds.append(text_embeds[i])
        else:
            p = positions[0].item()
            combined = torch.cat(
                [text_embeds[i, :p], projected[i], text_embeds[i, p + 1:]],
                dim=0,
            )
            out_embeds.append(combined)

    # Pad to longest sequence in batch
    max_len = max(e.size(0) for e in out_embeds)
    padded, masks = [], []
    for e in out_embeds:
        pad = max_len - e.size(0)
        if pad > 0:
            e = torch.cat([e, torch.zeros(pad, e.size(1), device=e.device, dtype=e.dtype)])
        mask = torch.ones(max_len, device=e.device)
        if pad > 0:
            mask[-pad:] = 0
        padded.append(e)
        masks.append(mask)

    return torch.stack(padded), torch.stack(masks)


def expand_labels(labels, input_ids, image_token_id, num_visual_tokens):
    """
    Expand labels to match the longer sequence after <image> is replaced
    with *num_visual_tokens* visual tokens (all labelled -100).
    """
    batch = labels.size(0)
    out = []
    for i in range(batch):
        positions = (input_ids[i] == image_token_id).nonzero(as_tuple=True)[0]
        if len(positions) == 0:
            out.append(labels[i])
        else:
            p = positions[0].item()
            vis = torch.full((num_visual_tokens,), -100,
                             dtype=labels.dtype, device=labels.device)
            out.append(torch.cat([labels[i, :p], vis, labels[i, p + 1:]]))

    max_len = max(l.size(0) for l in out)
    padded = []
    for l in out:
        pad = max_len - l.size(0)
        if pad > 0:
            l = torch.cat([l, torch.full((pad,), -100, dtype=l.dtype, device=l.device)])
        padded.append(l)

    return torch.stack(padded)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(adapter, coder, loader, image_token_id, embed_fn, device):
    adapter.eval()
    total_loss, n = 0.0, 0
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        lbl  = batch["labels"].to(device)
        feat = batch["features"].to(device)

        projected = adapter(feat.float()).half()
        embeds, mask = replace_image_tokens(ids, projected, image_token_id, embed_fn)
        adj_labels = expand_labels(lbl, ids, image_token_id, feat.size(1))

        loss = coder(inputs_embeds=embeds, attention_mask=mask, labels=adj_labels).loss
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(adapter, optimizer, scheduler, step, epoch, ckpt_dir, name):
    path = os.path.join(ckpt_dir, f"{name}.pt")
    torch.save({
        "adapter_state_dict": adapter.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "global_step": step,
        "epoch": epoch,
    }, path)
    print(f"  Saved checkpoint: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 2a — Train projection adapter")
    parser.add_argument("--features_dir", default="./precomputed_features")
    parser.add_argument("--train_manifest",
                        default="Data Crawling/output/manifests/train.jsonl")
    parser.add_argument("--val_manifest",
                        default="Data Crawling/output/manifests/val.jsonl")
    parser.add_argument("--coder_model",
                        default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--checkpoint_dir", default="./checkpoints/phase2a")
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--log_steps", type=int, default=10)
    args = parser.parse_args()

    device = "cuda"

    # ==================================================================
    # 1.  Load coder model  (8-bit quantized, frozen, fp16 non-quant params)
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
        torch_dtype=torch.float16,       # keeps non-quantized params in fp16
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.coder_model, trust_remote_code=True,
    )

    # Add vision special tokens
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<image>", "<img_start>", "<img_end>"]}
    )
    coder.resize_token_embeddings(len(tokenizer))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Freeze entire coder model
    for p in coder.parameters():
        p.requires_grad = False

    # Gradient checkpointing: recompute activations during backward instead of
    # storing them, trading ~30% more compute for ~60-70% less activation memory.
    # Must set model to train mode — checkpointing only activates when self.training=True.
    # Safe because instruction-tuned models have dropout=0.0 (train/eval behave identically).
    coder.gradient_checkpointing_enable()
    coder.train()
    print("  Gradient checkpointing enabled (train mode for activation)")

    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    coder_dim = coder.config.hidden_size
    print(f"  hidden_size={coder_dim}  image_token_id={image_token_id}")
    print(f"  Coder model loaded and frozen\n")

    # ==================================================================
    # 2.  Create adapter  (trainable, fp32 weights)
    # ==================================================================
    print(f"Creating adapter (1280 -> {coder_dim}) ...")
    adapter = ProjectionAdapter(vision_dim=1280, hidden_dim=4096, coder_dim=coder_dim)
    adapter = adapter.to(device)
    print(f"  Parameters: {adapter.num_parameters():,}\n")

    # ==================================================================
    # 3.  Datasets  (pre-computed features, all cached in CPU RAM)
    # ==================================================================
    print("Loading datasets ...")
    train_ds = PrecomputedDataset(
        args.train_manifest, args.features_dir, tokenizer, args.max_seq_length,
    )
    val_ds = PrecomputedDataset(
        args.val_manifest, args.features_dir, tokenizer, args.max_seq_length,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # ==================================================================
    # 4.  Optimizer & scheduler
    # ==================================================================
    optimizer = AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.0)
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * 0.03)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    eff_batch = args.batch_size * args.grad_accum
    print(f"\nTraining plan:")
    print(f"  Steps: {total_steps}   Warmup: {warmup_steps}")
    print(f"  Effective batch size: {eff_batch}")
    print(f"  Epochs: {args.epochs}\n")

    # ==================================================================
    # 5.  Training loop
    # ==================================================================
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    embed_fn = coder.get_input_embeddings()
    global_step = 0
    best_val_loss = float("inf")

    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60 + "\n")

    for epoch in range(args.epochs):
        adapter.train()
        epoch_loss, n_batches = 0.0, 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch_idx, batch in enumerate(progress):
            ids  = batch["input_ids"].to(device)
            lbl  = batch["labels"].to(device)
            feat = batch["features"].to(device)

            # Features are fp16 from pre-compute; adapter weights are fp32
            # Cast to fp32 for adapter, then back to fp16 for coder model
            projected = adapter(feat.float()).half()

            # Replace <image> placeholder with visual tokens
            embeds, mask = replace_image_tokens(
                ids, projected, image_token_id, embed_fn,
            )
            adj_labels = expand_labels(lbl, ids, image_token_id, feat.size(1))

            # Forward through frozen coder — no autocast needed
            loss = coder(
                inputs_embeds=embeds,
                attention_mask=mask,
                labels=adj_labels,
            ).loss
            loss = loss / args.grad_accum
            loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Log
                if global_step % args.log_steps == 0:
                    lr = scheduler.get_last_lr()[0]
                    print(f"  step={global_step}  "
                          f"loss={loss.item() * args.grad_accum:.4f}  "
                          f"lr={lr:.2e}")

                # Validate
                if global_step % args.eval_steps == 0:
                    vl = evaluate(
                        adapter, coder, val_loader,
                        image_token_id, embed_fn, device,
                    )
                    print(f"  step={global_step}  val_loss={vl:.4f}")
                    if vl < best_val_loss:
                        best_val_loss = vl
                        save_checkpoint(
                            adapter, optimizer, scheduler,
                            global_step, epoch, args.checkpoint_dir, "best",
                        )
                    adapter.train()

            epoch_loss += loss.item() * args.grad_accum
            n_batches += 1
            progress.set_postfix(loss=f"{epoch_loss / n_batches:.4f}")

        avg = epoch_loss / max(n_batches, 1)
        print(f"\nEpoch {epoch + 1} — avg train loss: {avg:.4f}")
        save_checkpoint(
            adapter, optimizer, scheduler,
            global_step, epoch, args.checkpoint_dir, f"epoch{epoch + 1}",
        )

    # Final save (adapter weights only, for easy loading later)
    final_path = os.path.join(args.checkpoint_dir, "adapter_final.pt")
    torch.save(adapter.state_dict(), final_path)
    print(f"\nSaved final adapter weights: {final_path}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Training complete!")


if __name__ == "__main__":
    main()
