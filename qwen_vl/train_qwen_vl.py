#!/usr/bin/env python3
"""
Fine-tune Qwen2.5-VL-7B-Instruct for code image understanding (Sniper Method Stage 2).

Tasks:  description + function_explanation  (robust to image compression)
Tokens: ~2048 visual tokens per image  (~8-10px/char, readable monospace code)
LoRA:   r=16 on q/k/v/o/gate/up/down_proj  (LLM layers only, vision encoder frozen)
Loss:   causal LM on assistant tokens only
"""

import os
import json
import time
import argparse
import random

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_PIXELS     = 2048 * 28 * 28   # ~2048 visual tokens  (~8-10px/char)
TASK_FILTER    = {"description", "function_explanation"}
IGNORE_INDEX   = -100


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CodeImageDataset(Dataset):
    def __init__(self, manifest_path, task_filter=None, max_image_height=3500):
        """
        max_image_height: skip images taller than this (pixels).
          At 2048-token budget, H<=3500px → px/char>=6.5 (readable monospace).
          Eliminates decompression bomb images (122M px) that crash training.
        """
        self.samples = []
        skipped = 0
        with open(manifest_path) as f:
            for line in f:
                s = json.loads(line)
                if task_filter is not None and s.get("task_type") not in task_filter:
                    continue
                try:
                    with Image.open(s["image"]) as img:
                        _, H = img.size
                    if H > max_image_height:
                        skipped += 1
                        continue
                except Exception:
                    skipped += 1
                    continue
                self.samples.append(s)
        print(f"  Loaded {len(self.samples)} samples from {manifest_path} "
              f"(filtered {skipped} oversized/missing)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Collate: tokenize one sample at a time (batch_size=1 safe, works for N>1)
# ---------------------------------------------------------------------------
def make_collate_fn(processor, max_seq_len=4096):
    def collate_fn(batch):
        all_input_ids   = []
        all_attn_masks  = []
        all_labels      = []
        all_pixel_values = []
        all_image_grids  = []

        for sample in batch:
            img_path = sample["image"]
            convs    = sample["conversations"]
            assert len(convs) == 2, f"Expected [user, assistant], got {len(convs)} turns"

            # Strip DeepSeek VL image tokens from user text
            user_text = convs[0]["content"].replace("<img_start><image><img_end>\n", "").strip()
            asst_text = convs[1]["content"].strip()

            image = Image.open(img_path).convert("RGB")

            # Build full conversation (user + assistant)
            full_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text",  "text":  user_text},
                    ],
                },
                {
                    "role":    "assistant",
                    "content": asst_text,
                },
            ]

            # Build prompt-only (user + generation prompt, NO assistant text)
            prompt_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text",  "text":  user_text},
                    ],
                },
            ]

            # Tokenize full conversation
            full_text = processor.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )
            full_enc = processor(
                text=[full_text],
                images=[image],
                max_pixels=MAX_PIXELS,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len,
            )

            # Tokenize prompt only — to find where the assistant response starts
            prompt_text = processor.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_enc = processor(
                text=[prompt_text],
                images=[image],
                max_pixels=MAX_PIXELS,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len,
            )

            input_ids  = full_enc["input_ids"][0]          # [seq_len]
            attn_mask  = full_enc["attention_mask"][0]
            prefix_len = prompt_enc["input_ids"].shape[1]  # number of prompt tokens

            # Labels: IGNORE_INDEX for prompt, copy input_ids for assistant response
            labels = input_ids.clone()
            labels[:prefix_len] = IGNORE_INDEX

            all_input_ids.append(input_ids)
            all_attn_masks.append(attn_mask)
            all_labels.append(labels)

            # Pixel values (may differ in shape per image — only works for batch_size=1
            # unless images are padded; safe here because we pad sequences below)
            all_pixel_values.append(full_enc["pixel_values"])          # [1, C, H, W] or similar
            if "image_grid_thw" in full_enc:
                all_image_grids.append(full_enc["image_grid_thw"])

        # Pad sequences in the batch to the same length (left-pad is NOT standard for causal LM;
        # right-pad with pad_token_id, labels stay IGNORE_INDEX for padding)
        max_len = max(ids.shape[0] for ids in all_input_ids)
        pad_id  = processor.tokenizer.pad_token_id or 0

        padded_input_ids  = torch.full((len(batch), max_len), pad_id,      dtype=torch.long)
        padded_attn_masks = torch.zeros((len(batch), max_len),             dtype=torch.long)
        padded_labels     = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)

        for i, (ids, mask, lbl) in enumerate(zip(all_input_ids, all_attn_masks, all_labels)):
            L = ids.shape[0]
            padded_input_ids[i, :L]  = ids
            padded_attn_masks[i, :L] = mask
            padded_labels[i, :L]     = lbl

        result = {
            "input_ids":      padded_input_ids,
            "attention_mask": padded_attn_masks,
            "labels":         padded_labels,
        }

        # Stack pixel values (works for batch_size=1; for N>1 images must match shape)
        if all_pixel_values:
            try:
                result["pixel_values"] = torch.cat(all_pixel_values, dim=0)
            except RuntimeError:
                # Images have different visual token counts — fall back to first only
                # (This only happens with batch_size>1 and very different image sizes)
                result["pixel_values"] = all_pixel_values[0]

        if all_image_grids:
            try:
                result["image_grid_thw"] = torch.cat(all_image_grids, dim=0)
            except RuntimeError:
                result["image_grid_thw"] = all_image_grids[0]

        return result

    return collate_fn


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(model, optimizer, scheduler, step, epoch, best_val,
                    ckpt_dir, name, is_distributed):
    os.makedirs(ckpt_dir, exist_ok=True)
    inner = model.module if is_distributed else model
    # Save LoRA weights only (small — adapter_model.safetensors)
    inner.save_pretrained(os.path.join(ckpt_dir, name))
    torch.save({
        "optimizer_state_dict":  optimizer.state_dict(),
        "scheduler_state_dict":  scheduler.state_dict(),
        "step":                  step,
        "epoch":                 epoch,
        "best_val_loss":         best_val,
    }, os.path.join(ckpt_dir, f"{name}_optim.pt"))


def _cleanup_old_checkpoints(ckpt_dir, keep=3):
    """Keep only the most recent `keep` step_* checkpoints."""
    import glob
    dirs = sorted(glob.glob(os.path.join(ckpt_dir, "step_*")))
    for d in dirs[:-keep]:
        import shutil
        shutil.rmtree(d, ignore_errors=True)
        opt = d + "_optim.pt"
        if os.path.exists(opt):
            os.remove(opt)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, is_distributed, is_main, amp_dtype):
    inner = model.module if is_distributed else model
    inner.eval()

    total_loss, total_tokens = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            out = inner(**batch)
        # out.loss is already averaged over non-IGNORE tokens
        n_tokens = (batch["labels"] != IGNORE_INDEX).sum().item()
        total_loss   += out.loss.item() * n_tokens
        total_tokens += n_tokens

    if is_distributed:
        t = torch.tensor([total_loss, total_tokens], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_loss, total_tokens = t[0].item(), t[1].item()

    inner.train()
    return total_loss / max(total_tokens, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",     default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--val_manifest",   required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--resume",         default=None,
                        help="Path to a saved LoRA checkpoint dir to resume from")
    parser.add_argument("--batch_size",     type=int,   default=1)
    parser.add_argument("--grad_accum",     type=int,   default=8)
    parser.add_argument("--epochs",         type=int,   default=3)
    parser.add_argument("--lr",             type=float, default=2e-4)
    parser.add_argument("--lora_r",         type=int,   default=16)
    parser.add_argument("--lora_alpha",     type=int,   default=32)
    parser.add_argument("--lora_dropout",   type=float, default=0.05)
    parser.add_argument("--max_seq_len",    type=int,   default=4096)
    parser.add_argument("--eval_steps",     type=int,   default=200)
    parser.add_argument("--log_steps",      type=int,   default=10)
    parser.add_argument("--ckpt_interval",  type=int,   default=1800,
                        help="Seconds between time-based checkpoints")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Distributed setup
    # ------------------------------------------------------------------
    is_distributed = "LOCAL_RANK" in os.environ
    if is_distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main   = (local_rank == 0)
    # Auto-select dtype: bfloat16 on H100/A100 (compute capability >= 8.0)
    # V100 is compute 7.0 — is_bf16_supported() can return True via SW emulation
    # but training in bf16 is unstable on V100; use fp16 + GradScaler instead.
    cc = torch.cuda.get_device_capability(device)
    if cc[0] >= 8:   # Ampere (A100) or Hopper (H100)
        amp_dtype  = torch.bfloat16
        use_scaler = False
    else:            # V100 and older
        amp_dtype  = torch.float16
        use_scaler = True
    if is_main:
        print(f"  dtype={amp_dtype}  grad_scaler={use_scaler}")

    if is_main:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"  Qwen2.5-VL-7B  Code Image Fine-Tuning")
        print(f"  Tasks:      {TASK_FILTER}")
        print(f"  Max pixels: {MAX_PIXELS:,}  (~2048 visual tokens)")
        print(f"  Checkpoint: {args.checkpoint_dir}")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Model + processor
    # ------------------------------------------------------------------
    if is_main:
        print(f"Loading {args.model_name} ...")

    processor = AutoProcessor.from_pretrained(
        args.model_name,
        min_pixels=256  * 28 * 28,
        max_pixels=MAX_PIXELS,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=amp_dtype,
        device_map=None,   # we handle placement ourselves
    )

    # Freeze vision encoder and visual projector
    for name, param in model.named_parameters():
        if "visual" in name:
            param.requires_grad = False

    if is_main:
        total   = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Base model — trainable before LoRA: {trainable/1e6:.1f}M / {total/1e6:.1f}M")

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        # Target LLM attention + MLP layers; "model.layers" ensures we skip vision layers
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        # Only match modules inside the language model, not vision encoder
        modules_to_save=None,
    )

    if args.resume and os.path.isdir(args.resume):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.resume, is_trainable=True)
        if is_main:
            print(f"  Resumed LoRA from: {args.resume}")
    else:
        model = get_peft_model(model, lora_config)

    # Gradient checkpointing: recomputes activations on backward pass instead of
    # storing them — trades ~30% speed for ~40% activation memory reduction.
    # Critical for fitting 7B model + LoRA + batch on 32GB V100.
    model.enable_input_require_grads()   # required for grad ckpt with PEFT
    model.gradient_checkpointing_enable()

    model = model.to(device)

    if is_main:
        lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  LoRA parameters: {lora_params/1e6:.1f}M\n")

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_ds = CodeImageDataset(args.train_manifest, task_filter=TASK_FILTER)
    val_ds   = CodeImageDataset(args.val_manifest,   task_filter=TASK_FILTER)

    collate = make_collate_fn(processor, max_seq_len=args.max_seq_len)

    train_sampler = DistributedSampler(train_ds, shuffle=True)  if is_distributed else None
    val_sampler   = DistributedSampler(val_ds,   shuffle=False) if is_distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate,
    )

    # ------------------------------------------------------------------
    # Optimizer + scheduler
    # ------------------------------------------------------------------
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    total_steps = (len(train_loader) * args.epochs) // args.grad_accum
    scheduler   = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)
    scaler      = torch.cuda.amp.GradScaler() if use_scaler else None

    # ------------------------------------------------------------------
    # Resume optimizer state
    # ------------------------------------------------------------------
    start_epoch = 0
    start_step  = 0
    best_val    = float("inf")

    if args.resume:
        opt_path = args.resume.rstrip("/") + "_optim.pt"
        if os.path.exists(opt_path):
            ckpt = torch.load(opt_path, map_location="cpu")
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_step  = ckpt["step"]
            start_epoch = ckpt["epoch"]
            best_val    = ckpt["best_val_loss"]
            if is_main:
                print(f"  Resumed optimizer: step={start_step}  best_val={best_val:.4f}\n")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    global_step   = start_step
    last_ckpt_t   = time.time()
    accum_loss    = 0.0
    accum_tokens  = 0

    for epoch in range(start_epoch, args.epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(**batch)

            n_tokens = (batch["labels"] != IGNORE_INDEX).sum().item()
            loss     = out.loss / args.grad_accum

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss   += out.loss.item() * n_tokens
            accum_tokens += n_tokens

            if (global_step + 1) % args.grad_accum == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if is_main and (global_step + 1) % (args.log_steps * args.grad_accum) == 0:
                    avg_loss = accum_loss / max(accum_tokens, 1)
                    lr_now   = scheduler.get_last_lr()[0]
                    step_num = (global_step + 1) // args.grad_accum
                    print(f"  epoch={epoch}  step={step_num}  "
                          f"train_loss={avg_loss:.4f}  lr={lr_now:.2e}")
                    accum_loss   = 0.0
                    accum_tokens = 0

            global_step += 1

            # Time-based checkpoint
            if is_main and (time.time() - last_ckpt_t) > args.ckpt_interval:
                step_num = global_step // args.grad_accum
                save_checkpoint(model, optimizer, scheduler, global_step, epoch,
                                best_val, args.checkpoint_dir, f"step_{step_num}",
                                is_distributed)
                _cleanup_old_checkpoints(args.checkpoint_dir)
                last_ckpt_t = time.time()
                print(f"  [ckpt] saved step_{step_num}")

            # Eval checkpoint
            if global_step % (args.eval_steps * args.grad_accum) == 0:
                val_loss = evaluate(model, val_loader, device,
                                    is_distributed, is_main, amp_dtype)
                if is_main:
                    step_num = global_step // args.grad_accum
                    print(f"\n  *** step={step_num}  val_loss={val_loss:.4f} ***\n")
                    if val_loss < best_val:
                        best_val = val_loss
                        save_checkpoint(model, optimizer, scheduler, global_step, epoch,
                                        best_val, args.checkpoint_dir, "best",
                                        is_distributed)
                        print(f"  [best] new best val_loss={best_val:.4f}")

        # End of epoch checkpoint
        if is_main:
            save_checkpoint(model, optimizer, scheduler, global_step, epoch + 1,
                            best_val, args.checkpoint_dir, f"epoch{epoch + 1}",
                            is_distributed)
            print(f"\n  === epoch {epoch} done ===\n")

    # Final save
    if is_main:
        inner = model.module if is_distributed else model
        inner.save_pretrained(os.path.join(args.checkpoint_dir, "final"))
        print(f"\nTraining complete. Best val_loss={best_val:.4f}")
        print(f"Saved to: {args.checkpoint_dir}/final")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
