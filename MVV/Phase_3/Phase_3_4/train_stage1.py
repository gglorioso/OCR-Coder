"""
Phase 3.4 -- Stage 1 (Lossless Decoder) Training

Multi-GPU training with 4x H100s. Native bfloat16, no bitsandbytes.
DDP is used for parameter broadcasting at init, but gradient sync is
handled manually (all-reduce on trainable grads only) via no_sync() to
prevent NCCL deadlocks when individual ranks hit NaN/OOM.

Key details:
  - Native bfloat16 (no quantization) — H100 80GB has ample VRAM
  - bfloat16 autocast for consistent dtype handling
  - Manual gradient all-reduce (DDP no_sync + explicit sync)
  - broadcast_buffers=False to avoid non-trainable buffer sync overhead
  - Prefix-only (no teacher forcing): text tokens in input_ids are
    replaced with pad tokens so the model must predict from vision alone
  - Cosine LR schedule with linear warmup
  - Surgical LoRA targets (not "all-linear")
  - DistributedSampler for both train and val
  - Mid-epoch checkpointing every 500 optimizer steps
  - --resume-from support for crash recovery

Architecture unchanged:
  Code image -> SigLIP features [1024, 1152]
             -> ConvRoPEProjector [256, 2048]
             -> splice into token embeddings at placeholder positions
             -> DeepSeek-Coder-V2-Lite-Instruct (bfloat16 + LoRA)
             -> cross-entropy loss on text tokens only
"""

import os
os.environ["PYTHONNOUSERSITE"] = "1"

import argparse
import json
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_TEXT_TOKENS = 768
N_VISUAL_TOKENS = 256
GRAD_ACCUM_STEPS = 2       # 4 per GPU * 4 GPUs * 2 accum = effective 32
LR_PROJECTOR = 1e-5
LR_LORA = 5e-6
BATCH_SIZE = 4              # per GPU
VAL_SPLIT = 0.05
SEED = 42
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
WARMUP_RATIO = 0.03
EPOCHS = 5


# ---------------------------------------------------------------------------
# ConvRoPEProjector  (copied from Phase 3.3)
# ---------------------------------------------------------------------------
def _sinusoidal_freqs(seq_len: int, half_dim: int, device: torch.device):
    """Return (cos, sin) tables each [seq_len, half_dim]."""
    i      = torch.arange(half_dim, device=device, dtype=torch.float32)
    theta  = 1.0 / (10000.0 ** (i / half_dim))
    pos    = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(pos, theta)
    return angles.cos(), angles.sin()


def _rope_rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE rotation. x: [..., seq_len, 2*half_dim]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def apply_2d_rope_16x16(x: torch.Tensor) -> torch.Tensor:
    """Apply 2D RoPE on post-convolution 16x16 grid (256 tokens)."""
    B, T, D = x.shape
    assert T == 256 and D == 1152, f"Expected [B,256,1152], got {x.shape}"
    half_D   = D // 2
    half_dim = half_D // 2
    device   = x.device
    rows = torch.arange(256, device=device) // 16
    cols = torch.arange(256, device=device) % 16
    cos_table, sin_table = _sinusoidal_freqs(16, half_dim, device)
    cos_table = cos_table.to(x.dtype)
    sin_table = sin_table.to(x.dtype)
    cos_row = cos_table[rows]
    sin_row = sin_table[rows]
    cos_col = cos_table[cols]
    sin_col = sin_table[cols]
    x_row = x[:, :, :half_D]
    x_col = x[:, :, half_D:]
    x_row_rot = _rope_rotate(x_row, cos_row, sin_row)
    x_col_rot = _rope_rotate(x_col, cos_col, sin_col)
    return torch.cat([x_row_rot, x_col_rot], dim=-1)


class ConvRoPEProjector(nn.Module):
    """Compresses [B, 1024, 1152] raw SigLIP tokens to [B, 256, 2048]."""
    def __init__(self, feat_dim: int = 1152, proj_dim: int = 2048) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.grid_in  = 32
        self.grid_out = 16
        self.conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=2, stride=2)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.conv.bias)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        assert N == self.grid_in ** 2 and C == self.feat_dim
        x = x.to(self.conv.weight.dtype)
        x = x.reshape(B, self.grid_in, self.grid_in, C).permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        x = apply_2d_rope_16x16(x)
        x = self.mlp(x)
        return x


# ---------------------------------------------------------------------------
# CoderVLModel
# ---------------------------------------------------------------------------
class CoderVLModel(nn.Module):
    """Wraps ConvRoPEProjector + LoRA-adapted DeepSeek LLM for DDP training."""
    def __init__(self, projector: ConvRoPEProjector, llm: nn.Module) -> None:
        super().__init__()
        self.projector = projector
        self.llm = llm

    def forward(self, vision, input_ids, attention_mask, labels):
        B, S = input_ids.shape
        visual_embeds = self.projector(vision)
        with torch.no_grad():
            text_embeds = self.llm.get_input_embeddings()(input_ids)
        text_embeds = text_embeds.clone()
        visual_embeds = visual_embeds.to(dtype=text_embeds.dtype)
        text_embeds[:, :N_VISUAL_TOKENS, :] = visual_embeds
        position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        outputs = self.llm(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        return outputs.loss


# ---------------------------------------------------------------------------
# JointDataset
# ---------------------------------------------------------------------------
class JointDataset(Dataset):
    """Loads paired SigLIP .pt features and .txt ground-truth source files."""

    def __init__(self, data_dir: str, tokenizer, max_text_tokens: int = MAX_TEXT_TOKENS) -> None:
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_text_tokens = max_text_tokens

        pt_files = {p.stem: p for p in self.data_dir.glob("*.pt")}
        txt_files = {p.stem: p for p in self.data_dir.glob("*.txt")}
        common_stems = sorted(set(pt_files.keys()) & set(txt_files.keys()))
        self.samples = [(pt_files[s], txt_files[s]) for s in common_stems]
        print(f"[JointDataset] Found {len(self.samples)} paired samples in {data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[dict]:
        pt_path, txt_path = self.samples[idx]
        try:
            vision = torch.load(pt_path, map_location="cpu", weights_only=True).float()
            assert vision.shape == (1024, 1152), f"Bad shape: {vision.shape}"

            source_text = txt_path.read_text(encoding="utf-8")

            pad_id = self.tokenizer.pad_token_id
            eos_id = self.tokenizer.eos_token_id

            newline_ids = self.tokenizer.encode("\n", add_special_tokens=False)
            text_ids = self.tokenizer.encode(source_text, add_special_tokens=False)

            max_text_len = self.max_text_tokens + 1 - len(newline_ids)
            if len(text_ids) > max_text_len:
                text_ids = text_ids[:max_text_len]

            # Prefix-only training: text positions in input_ids are masked
            # (pad tokens) so the model must rely on vision tokens to predict
            # text.  Without this, teacher forcing lets the model copy the
            # previous ground-truth token and ignore vision entirely.
            placeholder = [pad_id] * N_VISUAL_TOKENS
            text_mask = [pad_id] * len(text_ids)   # mask text in input
            input_ids = placeholder + newline_ids + text_mask + [eos_id]

            labels = (
                [-100] * N_VISUAL_TOKENS
                + [-100] * len(newline_ids)
                + text_ids
                + [eos_id]
            )

            assert len(input_ids) == len(labels)

            return {
                "vision": vision,
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
        except Exception as e:
            print(f"[JointDataset] Error loading {pt_path.stem}: {e}")
            return None


# ---------------------------------------------------------------------------
# collate_fn
# ---------------------------------------------------------------------------
def collate_fn(batch: list) -> Optional[dict]:
    """Collate with None filtering and right-padding."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    pad_id = collate_fn.pad_token_id

    vision = torch.stack([b["vision"] for b in batch])

    max_len = max(b["input_ids"].size(0) for b in batch)

    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    for b in batch:
        seq_len = b["input_ids"].size(0)
        pad_len = max_len - seq_len

        ids = torch.cat([b["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)])
        lab = torch.cat([b["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        mask = torch.cat([torch.ones(seq_len, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])

        input_ids_list.append(ids)
        labels_list.append(lab)
        attention_mask_list.append(mask)

    return {
        "vision": vision,
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels": torch.stack(labels_list),
    }

collate_fn.pad_token_id = 0


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------
def setup_ddp():
    """Initialize distributed process group and set local device."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """Destroy the distributed process group."""
    dist.destroy_process_group()


def is_main_process():
    """True only on rank 0."""
    return int(os.environ.get("RANK", 0)) == 0


# ---------------------------------------------------------------------------
# Model loading (no bitsandbytes, native bfloat16)
# ---------------------------------------------------------------------------
def load_model(model_path: str, device: torch.device, resume_from: str = None):
    """Load tokenizer, bfloat16 LLM with LoRA, and ConvRoPEProjector.

    If resume_from is provided, loads projector weights and LoRA adapter from
    that checkpoint directory instead of starting from scratch.
    """
    if is_main_process():
        print(f"[load_model] Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_main_process():
        print(f"[load_model] Loading LLM in bfloat16 from {model_path}")
    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": device},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    if resume_from is not None:
        resume_dir = Path(resume_from)
        if is_main_process():
            print(f"[load_model] Resuming LoRA from {resume_dir / 'lora_adapter'}")
        llm = PeftModel.from_pretrained(llm, str(resume_dir / "lora_adapter"), is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=[
                "q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )
        llm = get_peft_model(llm, lora_config)

    if is_main_process():
        llm.print_trainable_parameters()

    projector = ConvRoPEProjector(feat_dim=1152, proj_dim=2048)
    projector.to(device=device, dtype=torch.bfloat16)

    if resume_from is not None:
        resume_dir = Path(resume_from)
        proj_ckpt_path = resume_dir / "projector.pth"
        if is_main_process():
            print(f"[load_model] Resuming projector from {proj_ckpt_path}")
        proj_ckpt = torch.load(str(proj_ckpt_path), map_location=device, weights_only=True)
        projector.load_state_dict(proj_ckpt["projector_state_dict"])

    model = CoderVLModel(projector, llm)

    # Print param counts
    if is_main_process():
        proj_trainable = sum(p.numel() for p in model.projector.parameters() if p.requires_grad)
        lora_trainable = sum(p.numel() for p in model.llm.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.llm.parameters() if not p.requires_grad)
        print(f"[load_model] Projector trainable params: {proj_trainable:,}")
        print(f"[load_model] LoRA trainable params:      {lora_trainable:,}")
        print(f"[load_model] Frozen LLM params:          {frozen:,}")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(model: nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    """Run validation on all ranks (NCCL timeout safety), return average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            vision = batch["vision"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            try:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    loss = model(vision, input_ids, attention_mask, labels)
                if not torch.isfinite(loss):
                    continue
                total_loss += loss.item()
                n_batches += 1
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    continue
                raise

    model.train()
    return total_loss / n_batches if n_batches > 0 else float("inf")


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(model: CoderVLModel, save_dir: Path, tag: str) -> None:
    """Save projector weights and LoRA adapter. Only call on rank 0."""
    ckpt_dir = save_dir / f"epoch_{tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {"projector_state_dict": model.projector.state_dict()},
        str(ckpt_dir / "projector.pth"),
    )
    model.llm.save_pretrained(str(ckpt_dir / "lora_adapter"))
    print(f"[save_checkpoint] Saved checkpoint to {ckpt_dir}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(model, tokenizer, data_dir, save_dir, epochs, batch_size, local_rank):
    """Main DDP training loop with cosine LR schedule, validation, and checkpointing."""
    device = torch.device(f"cuda:{local_rank}")
    save_dir = Path(save_dir)
    if is_main_process():
        save_dir.mkdir(parents=True, exist_ok=True)

    # Dataset and split
    full_dataset = JointDataset(data_dir, tokenizer)
    val_size = max(1, int(len(full_dataset) * VAL_SPLIT))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    if is_main_process():
        print(f"[train] Train: {train_size}, Val: {val_size}")

    collate_fn.pad_token_id = tokenizer.pad_token_id

    # DistributedSamplers
    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=SEED)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler,
        num_workers=2, pin_memory=True, collate_fn=collate_fn,
    )

    # Wrap in DDP — broadcast_buffers=False prevents allreduce on non-trainable
    # buffers (e.g. layernorm running stats), avoiding NCCL timeouts.
    # We use no_sync() everywhere and manually all-reduce trainable grads,
    # so DDP only handles parameter broadcasting at init.
    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=False, broadcast_buffers=False)

    # Optimizer -- two param groups (use unwrapped model)
    optimizer = torch.optim.AdamW([
        {"params": model.projector.parameters(), "lr": LR_PROJECTOR},
        {"params": [p for p in model.llm.parameters() if p.requires_grad], "lr": LR_LORA},
    ])

    # LR scheduler -- linear warmup + cosine decay
    # Account for grad accumulation in total steps calculation
    steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    if is_main_process():
        print(f"[train] Total optimizer steps: {total_steps}, warmup: {warmup_steps}")
        print(f"[train] Grad accum steps: {GRAD_ACCUM_STEPS}")
        print(f"[train] Effective batch size: {batch_size} * {dist.get_world_size()} * "
              f"{GRAD_ACCUM_STEPS} = {batch_size * dist.get_world_size() * GRAD_ACCUM_STEPS}")

    best_val_loss = float("inf")
    skipped_batches = 0

    # Collect trainable params once (for gradient sync + clipping)
    all_params = (
        list(model.projector.parameters())
        + [p for p in model.llm.parameters() if p.requires_grad]
    )

    world_size = dist.get_world_size()

    def sync_gradients():
        """All-reduce trainable gradients across ranks, average by world_size.
        Initializes missing grads to zero so ALL ranks participate in every
        all_reduce call, preventing NCCL deadlocks when some ranks skip backward."""
        for p in all_params:
            if p.grad is None:
                p.grad = torch.zeros_like(p.data)
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world_size)

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        ddp_model.train()

        epoch_loss = 0.0
        n_steps = 0
        accum_step = 0
        optimizer_steps = 0

        loader_iter = train_loader
        if is_main_process():
            loader_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(loader_iter):
            got_loss = False

            with ddp_model.no_sync():
                if batch is not None:
                    vision = batch["vision"].to(device)
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    try:
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            loss = ddp_model(vision, input_ids, attention_mask, labels)
                        if not torch.isfinite(loss):
                            print(f"[train] NaN/Inf loss at batch {batch_idx}, skipping")
                            optimizer.zero_grad()
                            skipped_batches += 1
                        else:
                            scaled_loss = loss / GRAD_ACCUM_STEPS
                            scaled_loss.backward()
                            got_loss = True
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"[train] OOM on batch {batch_idx}, skipping")
                            torch.cuda.empty_cache()
                            optimizer.zero_grad()
                            skipped_batches += 1
                        else:
                            raise
                else:
                    skipped_batches += 1

            accum_step += 1

            if accum_step % GRAD_ACCUM_STEPS == 0:
                sync_gradients()
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_steps += 1

            if got_loss:
                epoch_loss += loss.item()
                n_steps += 1

                # Mid-epoch checkpoint every 500 optimizer steps
                if optimizer_steps > 0 and optimizer_steps % 500 == 0 and is_main_process():
                    save_checkpoint(model, save_dir, f"step_{optimizer_steps}")
                    save_checkpoint(model, save_dir, "latest")
                    print(f"[train] Mid-epoch checkpoint at optimizer step {optimizer_steps}")

            if got_loss and is_main_process() and hasattr(loader_iter, "set_postfix"):
                loader_iter.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr_proj=f"{scheduler.get_last_lr()[0]:.2e}",
                    lr_lora=f"{scheduler.get_last_lr()[1]:.2e}",
                )

        # Handle leftover accumulated gradients
        if accum_step % GRAD_ACCUM_STEPS != 0:
            sync_gradients()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_train_loss = epoch_loss / n_steps if n_steps > 0 else float("inf")
        if is_main_process():
            print(f"[train] Epoch {epoch+1}/{epochs} -- train_loss: {avg_train_loss:.4f}")
            if skipped_batches > 0:
                print(f"[train] Skipped {skipped_batches} batches total so far")

        # Validation (all ranks to avoid NCCL timeout)
        val_loss = validate(ddp_model, val_loader, device)
        if is_main_process():
            print(f"[train] Epoch {epoch+1}/{epochs} -- val_loss:   {val_loss:.4f}")

        # Checkpoint (rank 0 only)
        if is_main_process():
            save_checkpoint(model, save_dir, str(epoch + 1))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, save_dir, "best")
                print(f"[train] New best val_loss: {best_val_loss:.4f}")

        # Barrier so all ranks wait for checkpointing to finish
        dist.barrier()

    if is_main_process():
        print(f"[train] Training complete. Best val_loss: {best_val_loss:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(SEED)

    parser = argparse.ArgumentParser(description="Phase 3.4 -- Stage 1 Lossless Decoder Training")
    parser.add_argument("--data-dir", type=str, default="MVV/Phase_3/full_data/tensors_and_texts",
                        help="Path to paired .pt/.txt data directory")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to DeepSeek-Coder-V2-Lite-Instruct model")
    parser.add_argument("--save-dir", type=str, default="MVV/Phase_3/checkpoints/stage1_run",
                        help="Directory for saving checkpoints")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to checkpoint dir (containing projector.pth and lora_adapter/) to resume from")
    args = parser.parse_args()

    if is_main_process():
        print(f"[main] Starting Stage 1 training")
        print(f"[main] Data: {args.data_dir}")
        print(f"[main] GPUs: {dist.get_world_size()}")
        if args.resume_from:
            print(f"[main] Resuming from checkpoint: {args.resume_from}")

    model, tokenizer = load_model(args.model_path, device, resume_from=args.resume_from)
    train(model, tokenizer, args.data_dir, args.save_dir, args.epochs, args.batch_size, local_rank)

    cleanup_ddp()


if __name__ == "__main__":
    main()
