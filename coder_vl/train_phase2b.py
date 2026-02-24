"""
Phase 2b Training — QLoRA Fine-tuning of DeepSeek-Coder-V2-Lite

Trains both the projection adapter AND the DeepSeek-Coder-V2-Lite LLM
(via 4-bit QLoRA) using pre-computed tiled vision features.

Memory budget per V100 (32 GB):
  4-bit coder:  ~8.96 GB | LoRA delta weights (r=16): ~28 MB
  Adapter:        ~54 MB | Optimizer (AdamW, 20.7M trainable): ~166 MB
  Activations:   ~115 MB | CUDA overhead: ~2.5 GB
  Total: ~11.9 GB / 32 GB  → ample headroom on V100

LoRA target modules (verified from modeling_deepseek.py):
  q_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj
  (q_lora_rank=None in Lite → single q_proj, no q_a_proj/q_b_proj)

Usage (torchrun, 2 GPUs):
    torchrun --nproc_per_node=2 coder_vl/train_phase2b.py \\
        --features_dir   ./precomputed_features_tiled \\
        --train_manifest data_v2b/manifests/train.jsonl \\
        --val_manifest   data_v2b/manifests/val.jsonl
"""

import os
import json
import glob
import argparse
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm

from projector import ProjectionAdapter


# ---------------------------------------------------------------------------
# Dataset  (unchanged from Phase 2a)
# ---------------------------------------------------------------------------

class PrecomputedDataset(Dataset):
    """
    Loads pre-computed vision features + tokenized text for each example.
    All unique feature tensors are cached in CPU memory at init time.
    """

    def __init__(self, manifest_path, features_dir, tokenizer, max_seq_length=2048):
        self.features_dir = Path(features_dir)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.examples = []
        with open(manifest_path) as f:
            for line in f:
                self.examples.append(json.loads(line))
        print(f"  Loaded {len(self.examples)} examples from {manifest_path}")

        self._cache = {}
        self._load_features()

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

        features = self._cache[ex["image"]]

        conv = ex["conversations"]
        user_msg      = conv[0]["content"]
        assistant_msg = conv[1]["content"]
        text = f"User: {user_msg}\n\nAssistant: {assistant_msg}"

        tok = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
        )
        input_ids = tok["input_ids"].squeeze(0)

        labels = input_ids.clone()
        asst_tokens = self.tokenizer.encode("Assistant:", add_special_tokens=False)
        pos = _find_subseq(input_ids.tolist(), asst_tokens)
        if pos != -1:
            labels[: pos + len(asst_tokens)] = -100
        else:
            labels[: len(labels) // 2] = -100

        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        return {"input_ids": input_ids, "labels": labels, "features": features}


def _find_subseq(lst, sub):
    n = len(sub)
    for i in range(len(lst) - n + 1):
        if lst[i : i + n] == sub:
            return i
    return -1


# ---------------------------------------------------------------------------
# Token-replacement helpers  (unchanged from Phase 2a)
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
    with num_visual_tokens visual tokens (all labelled -100).
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
# Checkpointing helpers
# ---------------------------------------------------------------------------

def _find_resume_checkpoint(ckpt_dir, explicit=None):
    """Return explicit checkpoint path, or latest step_*.pt, or None."""
    if explicit:
        return explicit
    pattern = os.path.join(ckpt_dir, "step_*.pt")
    candidates = sorted(
        glob.glob(pattern),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[1]),
    )
    return candidates[-1] if candidates else None


def _cleanup_old_checkpoints(ckpt_dir, keep=3):
    """Delete all but the last `keep` step_*.pt files."""
    pattern = os.path.join(ckpt_dir, "step_*.pt")
    candidates = sorted(
        glob.glob(pattern),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[1]),
    )
    for old in candidates[:-keep]:
        os.remove(old)
        print(f"  Removed old checkpoint: {old}")


def save_checkpoint(adapter, coder, optimizer, scheduler,
                    step, epoch, best_val, ckpt_dir, name, is_distributed):
    """Save adapter weights, LoRA delta weights, and full optimizer state."""
    path = os.path.join(ckpt_dir, f"{name}.pt")
    inner_adapter = adapter.module if is_distributed else adapter
    inner_coder   = coder.module   if is_distributed else coder
    # Save only LoRA delta weights (not the full 4-bit model)
    lora_state = {k: v for k, v in inner_coder.state_dict().items() if "lora_" in k}
    torch.save({
        "adapter_state_dict":   inner_adapter.state_dict(),
        "lora_state_dict":      lora_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "global_step":          step,
        "epoch":                epoch,
        "best_val_loss":        best_val,
    }, path)
    print(f"  Saved checkpoint: {path}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(adapter, coder, loader, image_token_id, embed_fn, device):
    """
    Evaluate on loader.  adapter and coder must be unwrapped (not DDP wrappers).
    Sets eval mode, runs eval, restores train mode before returning.
    """
    adapter.eval()
    coder.eval()
    total_loss, n = 0.0, 0
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        lbl  = batch["labels"].to(device)
        feat = batch["features"].to(device)

        projected  = adapter(feat.float()).half()
        embeds, mask = replace_image_tokens(ids, projected, image_token_id, embed_fn)
        adj_labels = expand_labels(lbl, ids, image_token_id, feat.size(1))

        loss = coder(inputs_embeds=embeds, attention_mask=mask, labels=adj_labels).loss
        total_loss += loss.item()
        n += 1

    adapter.train()
    coder.train()
    return total_loss / max(n, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 2b — QLoRA fine-tuning")
    parser.add_argument("--features_dir",   default="./precomputed_features_tiled")
    parser.add_argument("--train_manifest", default="data_v2b/manifests/train.jsonl")
    parser.add_argument("--val_manifest",   default="data_v2b/manifests/val.jsonl")
    parser.add_argument("--coder_model",    default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct")
    parser.add_argument("--batch_size",     type=int,   default=4)
    parser.add_argument("--lr_lora",        type=float, default=2e-5,
                        help="Learning rate for LoRA parameters")
    parser.add_argument("--lr_adapter",     type=float, default=1e-5,
                        help="Learning rate for projection adapter (lower to protect contrastive init)")
    parser.add_argument("--epochs",         type=int,   default=2)
    parser.add_argument("--grad_accum",     type=int,   default=4)
    parser.add_argument("--max_seq_length", type=int,   default=260)
    parser.add_argument("--checkpoint_dir", default="./checkpoints/phase2b")
    parser.add_argument("--eval_steps",     type=int,   default=200)
    parser.add_argument("--log_steps",      type=int,   default=10)
    parser.add_argument("--lora_r",         type=int,   default=16)
    parser.add_argument("--lora_alpha",     type=int,   default=32)
    parser.add_argument("--lora_dropout",   type=float, default=0.05)
    parser.add_argument("--ckpt_interval",  type=int,   default=1800,
                        help="Seconds between time-based checkpoints (default: 1800 = 30 min)")
    parser.add_argument("--resume",         type=str,   default=None,
                        help="Explicit checkpoint to resume from; if None, auto-detects latest step_*.pt")
    parser.add_argument("--init_from",      default="./checkpoints/phase2a_v6/best.pt",
                        help="Phase 2a checkpoint to load adapter weights from (adapter only, not LoRA)")
    args = parser.parse_args()

    # ==================================================================
    # 0.  Distributed setup (works with torchrun; falls back to 1 GPU)
    # ==================================================================
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if is_distributed:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size  = dist.get_world_size()
        rank        = dist.get_rank()
    else:
        local_rank = 0
        world_size = 1
        rank       = 0

    torch.cuda.set_device(local_rank)
    device  = f"cuda:{local_rank}"
    is_main = (rank == 0)

    # ==================================================================
    # 1.  Load coder model (4-bit QLoRA) and apply LoRA
    # ==================================================================
    if is_main:
        print("=" * 60)
        print("LOADING CODER MODEL (4-bit QLoRA)")
        print("=" * 60)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    coder = AutoModelForCausalLM.from_pretrained(
        args.coder_model,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,       # non-quantized params (e.g. MoE gates) in fp16
        device_map={"": local_rank},     # full copy on this GPU (no pipeline conflict)
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

    # Apply LoRA — makes LoRA params trainable, freezes everything else
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    coder = get_peft_model(coder, lora_config)
    if is_main:
        coder.print_trainable_parameters()

    # Gradient checkpointing (peft-compatible order: must call these after get_peft_model)
    coder.enable_input_require_grads()   # required for grad_ckpt with peft
    coder.gradient_checkpointing_enable()
    coder.train()

    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    coder_dim = coder.config.hidden_size
    if is_main:
        print(f"  hidden_size={coder_dim}  image_token_id={image_token_id}\n")

    # ==================================================================
    # 2.  Create adapter and load Phase 2a weights
    # ==================================================================
    if is_main:
        print(f"Creating adapter (1280 -> {coder_dim}) ...")
    adapter = ProjectionAdapter(vision_dim=1280, hidden_dim=4096, coder_dim=coder_dim)

    if args.init_from and os.path.exists(args.init_from):
        ckpt = torch.load(args.init_from, map_location="cpu")
        state = ckpt.get("adapter_state_dict", ckpt)
        adapter.load_state_dict(state)
        if is_main:
            print(f"  Loaded adapter weights from: {args.init_from}")
    else:
        if is_main:
            print("  Initialized adapter with random weights")

    adapter = adapter.to(device)
    if is_main:
        print(f"  Adapter parameters: {adapter.num_parameters():,}\n")

    # Capture embed_fn before DDP wrapping — same object reference remains valid after wrapping
    embed_fn = coder.get_input_embeddings()

    # ==================================================================
    # 3.  Datasets
    # ==================================================================
    if is_main:
        print("Loading datasets ...")
    train_ds = PrecomputedDataset(
        args.train_manifest, args.features_dir, tokenizer, args.max_seq_length,
    )
    val_ds = PrecomputedDataset(
        args.val_manifest, args.features_dir, tokenizer, args.max_seq_length,
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True)  if is_distributed else None
    val_sampler   = DistributedSampler(val_ds,   shuffle=False) if is_distributed else None
    train_loader  = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        sampler=val_sampler, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # ==================================================================
    # 4.  Optimizer & scheduler — two param groups (different LRs)
    # ==================================================================
    lora_params    = [p for p in coder.parameters() if p.requires_grad]
    adapter_params = list(adapter.parameters())

    optimizer = AdamW([
        {"params": adapter_params, "lr": args.lr_adapter},
        {"params": lora_params,    "lr": args.lr_lora},
    ], weight_decay=0.0)

    total_steps  = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * 0.03)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    eff_batch = args.batch_size * args.grad_accum * world_size
    if is_main:
        print(f"\nTraining plan:")
        print(f"  Steps: {total_steps}   Warmup: {warmup_steps}")
        print(f"  Effective batch size: {eff_batch}  "
              f"(batch={args.batch_size} x accum={args.grad_accum} x gpus={world_size})")
        print(f"  Epochs: {args.epochs}  lr_adapter={args.lr_adapter}  lr_lora={args.lr_lora}\n")

    # ==================================================================
    # 5.  Auto-resume (restores adapter, LoRA, optimizer, scheduler)
    # ==================================================================
    resume_path = _find_resume_checkpoint(args.checkpoint_dir, args.resume)
    start_epoch, start_step, best_val_loss = 0, 0, float("inf")

    if resume_path and os.path.exists(resume_path):
        if is_main:
            print(f"Resuming from {resume_path} ...")
        ckpt = torch.load(resume_path, map_location="cpu")
        adapter.load_state_dict(ckpt["adapter_state_dict"])
        # LoRA delta weights only — strict=False because checkpoint lacks full coder state
        coder.load_state_dict(ckpt["lora_state_dict"], strict=False)
        start_epoch   = ckpt["epoch"]
        start_step    = ckpt["global_step"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if is_main:
            print(f"  Resumed: epoch={start_epoch}  step={start_step}  "
                  f"best_val={best_val_loss:.4f}\n")
    elif is_main:
        print("No resume checkpoint found — starting from scratch\n")

    # ==================================================================
    # 6.  DDP wrapping (after auto-resume so state loads into raw modules)
    # ==================================================================
    if is_distributed:
        # static_graph=True: required when using peft + gradient_checkpointing + DDP.
        # Gradient checkpointing recomputes activations during backward, which would
        # trigger DDP's "parameter ready" hook twice — static_graph bypasses that check.
        # Safe because LoRA params are in every attention layer (graph never changes).
        # Note: static_graph is incompatible with find_unused_parameters=True.
        coder   = DDP(coder,   device_ids=[local_rank], static_graph=True)
        adapter = DDP(adapter, device_ids=[local_rank])

    # ==================================================================
    # 7.  Training loop
    # ==================================================================
    if is_main:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        print("=" * 60)
        print("STARTING TRAINING")
        print("=" * 60 + "\n")

    global_step    = start_step
    last_ckpt_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        adapter.train()
        coder.train()
        epoch_loss, n_batches = 0.0, 0
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            disable=not is_main,
        )

        for batch_idx, batch in enumerate(progress):
            ids  = batch["input_ids"].to(device)
            lbl  = batch["labels"].to(device)
            feat = batch["features"].to(device)

            # Forward through DDP-wrapped adapter (syncs adapter grads across ranks)
            projected  = adapter(feat.float()).half()
            embeds, mask = replace_image_tokens(ids, projected, image_token_id, embed_fn)
            adj_labels   = expand_labels(lbl, ids, image_token_id, feat.size(1))

            # Forward through DDP-wrapped coder (syncs LoRA grads across ranks)
            loss = coder(inputs_embeds=embeds, attention_mask=mask, labels=adj_labels).loss
            loss = loss / args.grad_accum
            loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                all_trainable = list(adapter.parameters()) + lora_params
                torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if is_main and global_step % args.log_steps == 0:
                    lr_a = optimizer.param_groups[0]["lr"]
                    lr_l = optimizer.param_groups[1]["lr"]
                    print(f"  step={global_step}  "
                          f"loss={loss.item() * args.grad_accum:.4f}  "
                          f"lr_adapter={lr_a:.2e}  lr_lora={lr_l:.2e}")

                # Time-based checkpoint (rank 0 only, every ckpt_interval seconds)
                if is_main and time.time() - last_ckpt_time >= args.ckpt_interval:
                    save_checkpoint(
                        adapter, coder, optimizer, scheduler,
                        global_step, epoch, best_val_loss,
                        args.checkpoint_dir, f"step_{global_step}", is_distributed,
                    )
                    _cleanup_old_checkpoints(args.checkpoint_dir)
                    last_ckpt_time = time.time()

                # Validation — all ranks evaluate their shard, then all_reduce AVG
                if global_step % args.eval_steps == 0:
                    inner_adapter = adapter.module if is_distributed else adapter
                    inner_coder   = coder.module   if is_distributed else coder
                    vl = evaluate(inner_adapter, inner_coder, val_loader,
                                  image_token_id, embed_fn, device)
                    if is_distributed:
                        vl_t = torch.tensor(vl, device=device)
                        dist.all_reduce(vl_t, op=dist.ReduceOp.AVG)
                        vl = vl_t.item()
                    if is_main:
                        print(f"  step={global_step}  val_loss={vl:.4f}")
                        if vl < best_val_loss:
                            best_val_loss = vl
                            save_checkpoint(
                                adapter, coder, optimizer, scheduler,
                                global_step, epoch, best_val_loss,
                                args.checkpoint_dir, "best", is_distributed,
                            )
                    # Restore train mode (evaluate sets eval on inner modules)
                    adapter.train()
                    coder.train()

            epoch_loss += loss.item() * args.grad_accum
            n_batches  += 1
            progress.set_postfix(loss=f"{epoch_loss / n_batches:.4f}")

        avg = epoch_loss / max(n_batches, 1)
        if is_main:
            print(f"\nEpoch {epoch + 1} — avg train loss: {avg:.4f}")
            save_checkpoint(
                adapter, coder, optimizer, scheduler,
                global_step, epoch, best_val_loss,
                args.checkpoint_dir, f"epoch{epoch + 1}", is_distributed,
            )

    # Final save — weights only (no optimizer state, smaller file for inference)
    if is_main:
        inner_adapter = adapter.module if is_distributed else adapter
        inner_coder   = coder.module   if is_distributed else coder
        lora_state = {k: v for k, v in inner_coder.state_dict().items() if "lora_" in k}
        final_path = os.path.join(args.checkpoint_dir, "final.pt")
        torch.save({
            "adapter_state_dict": inner_adapter.state_dict(),
            "lora_state_dict":    lora_state,
        }, final_path)
        print(f"\nSaved final weights: {final_path}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print("Training complete!")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
