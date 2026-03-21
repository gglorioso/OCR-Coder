"""
train_joint.py — Phase 3: Joint LoRA + Projector Training

Architecture:
  Code image → SigLIP features [1024, 1152]
             → ConvRoPEProjector [256, 2048]
             → splice into token embeddings at placeholder positions
             → DeepSeek-Coder-V2-Lite-Instruct (8-bit + LoRA)
             → cross-entropy loss on text tokens only

Key change from Phase 1.9c: The LLM is no longer frozen. LoRA adapters
are applied to all linear layers, allowing the backbone to learn to
interpret visual token prefixes. This addresses the RLHF override problem
identified in Phase 1.9c.

Training setup:
  - Projector: lr=1e-5 (conservative, prevents catastrophic forgetting)
  - LoRA: lr=2e-4, r=16, alpha=32, target_modules="all-linear"
  - QLoRA: 8-bit quantized backbone
  - Two parameter groups with separate learning rates
"""

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_TEXT_TOKENS = 768          # bumped from 512 — P95 for 40 lines is 529 tokens
N_VISUAL_TOKENS = 256
GRAD_ACCUM_STEPS = 4
LR_PROJECTOR = 1e-5            # CRITICAL: never 1e-4 — causes catastrophic forgetting
LR_LORA = 2e-4                 # standard LoRA learning rate
WARMUP_STEPS = 100
EPOCHS = 5
BATCH_SIZE = 1
VAL_SPLIT = 0.1
SEED = 42
CKPT_EVERY_STEPS = 500
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


# ---------------------------------------------------------------------------
# ConvRoPEProjector  (copied verbatim from spec)
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
    B, T, D = x.shape
    assert T == 256 and D == 1152, f"Expected [B,256,1152], got {x.shape}"
    half_D   = D // 2
    half_dim = half_D // 2
    device   = x.device
    rows = torch.arange(256, device=device) // 16
    cols = torch.arange(256, device=device) % 16
    cos_table, sin_table = _sinusoidal_freqs(16, half_dim, device)
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
        x = x.reshape(B, self.grid_in, self.grid_in, C).permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        x = apply_2d_rope_16x16(x)
        x = self.mlp(x)
        return x


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class JointDataset(Dataset):
    """Loads paired SigLIP .pt features and .txt ground-truth source files."""

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        max_text_tokens: int = MAX_TEXT_TOKENS,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_text_tokens = max_text_tokens

        # Discover samples by matching .pt and .txt stems
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
            # Load vision features
            vision = torch.load(pt_path, map_location="cpu", weights_only=True).float()
            assert vision.shape == (1024, 1152), f"Bad shape: {vision.shape}"

            # Load source text
            source_text = txt_path.read_text(encoding="utf-8")

            # Tokenize components
            pad_id = self.tokenizer.pad_token_id
            eos_id = self.tokenizer.eos_token_id

            newline_ids = self.tokenizer.encode("\n", add_special_tokens=False)
            text_ids = self.tokenizer.encode(source_text, add_special_tokens=False)

            # Budget: 256 (visual placeholders) + len(newline) + text + 1 (eos)
            max_text_len = self.max_text_tokens + 1 - len(newline_ids)  # +1 for eos
            if len(text_ids) > max_text_len:
                text_ids = text_ids[:max_text_len]

            # Build input_ids: [pad]*256 + newline + text + eos
            placeholder = [pad_id] * N_VISUAL_TOKENS
            input_ids = placeholder + newline_ids + text_ids + [eos_id]

            # Build labels: [-100]*256 + [-100] for newline + real text ids + eos
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
# Collate function
# ---------------------------------------------------------------------------
def collate_fn(batch: list) -> Optional[dict]:
    """Collate with None filtering and right-padding."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    pad_id = collate_fn.pad_token_id

    # Stack vision tensors
    vision = torch.stack([b["vision"] for b in batch])

    # Determine max sequence length
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


# Monkey-patch default; overwritten in main()
collate_fn.pad_token_id = 0


# ---------------------------------------------------------------------------
# CoderVLModel — the joint wrapper
# ---------------------------------------------------------------------------
class CoderVLModel(nn.Module):
    """Wraps ConvRoPEProjector + LoRA-adapted DeepSeek LLM."""

    def __init__(self, projector: ConvRoPEProjector, llm: nn.Module) -> None:
        super().__init__()
        self.projector = projector
        self.llm = llm

    def forward(
        self,
        vision: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        B, S = input_ids.shape

        # 1. Project vision features: [B, 1024, 1152] -> [B, 256, 2048]
        visual_embeds = self.projector(vision)

        # 2. Get text embeddings (with gradients — LoRA needs them)
        text_embeds = self.llm.get_input_embeddings()(input_ids).clone()

        # 3. Cast visual_embeds to match text_embeds dtype
        visual_embeds = visual_embeds.to(dtype=text_embeds.dtype)

        # 4. SPLICE: overwrite placeholder positions with visual embeddings
        text_embeds[:, :N_VISUAL_TOKENS, :] = visual_embeds

        # 5. Position IDs
        position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)

        # 6. Forward through LLM
        outputs = self.llm(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )

        # 7. Return loss
        return outputs.loss


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(
    model_path: str,
    projector_ckpt_path: Optional[str],
    device: str,
) -> tuple:
    """Load tokenizer, quantized LLM with LoRA, and projector."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[load_model] pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}")

    # 2. Load LLM in 8-bit
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map={"": device},
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # 3. Prepare for k-bit training and apply LoRA
    llm = prepare_model_for_kbit_training(llm)
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    llm = get_peft_model(llm, lora_config)
    llm.print_trainable_parameters()

    # 4. Load projector
    projector = ConvRoPEProjector(feat_dim=1152, proj_dim=2048)
    if projector_ckpt_path and Path(projector_ckpt_path).exists():
        ckpt = torch.load(projector_ckpt_path, map_location="cpu", weights_only=True)
        projector.load_state_dict(ckpt["projector_state_dict"])
        print(f"[load_model] Loaded projector from {projector_ckpt_path}")
    else:
        print("[load_model] Projector initialized from scratch")
    projector = projector.to(device)

    # 5. Create wrapper model
    model = CoderVLModel(projector, llm)

    # Print parameter counts
    proj_params = sum(p.numel() for p in model.projector.parameters() if p.requires_grad)
    lora_params = sum(p.numel() for p in model.llm.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.llm.parameters() if not p.requires_grad)
    total_trainable = proj_params + lora_params
    print(f"\n[Parameters]")
    print(f"  Projector (trainable):  {proj_params:>12,}")
    print(f"  LoRA (trainable):       {lora_params:>12,}")
    print(f"  Total trainable:        {total_trainable:>12,}")
    print(f"  LLM frozen:             {frozen_params:>12,}")
    print()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Mini-data preparation
# ---------------------------------------------------------------------------
def prepare_mini_data(repo_root: str, data_dir: str, n_samples: int = 100) -> None:
    """Create a small paired dataset from the Phase 1.1 manifest."""
    repo = Path(repo_root)
    data = Path(data_dir)
    data.mkdir(parents=True, exist_ok=True)

    manifest_path = repo / "MVV" / "Phase_1_1" / "data_mvv" / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    features_dir = repo / "MVV" / "Phase_1_9" / "a" / "data" / "features"
    scraped_dir = repo / "Scraped Repos"

    entries = []
    with open(manifest_path, "r") as f:
        for line in f:
            entries.append(json.loads(line.strip()))

    prepared = 0
    for entry in entries:
        if prepared >= n_samples:
            break

        stem = Path(entry.get("image", entry.get("image_path", ""))).stem
        if not stem:
            continue

        pt_src = features_dir / f"{stem}.pt"
        source_file = entry.get("source_file", "")
        txt_src = scraped_dir / source_file

        if not pt_src.exists() or not txt_src.exists():
            continue

        # Copy .pt file
        pt_dst = data / f"{stem}.pt"
        if not pt_dst.exists():
            shutil.copy2(pt_src, pt_dst)

        # Extract 40-line window and save .txt
        anchor_line = entry.get("anchor_line", 0)
        try:
            with open(txt_src, "r", encoding="utf-8", errors="replace") as sf:
                all_lines = sf.readlines()
        except Exception as e:
            print(f"  Skipping {stem}: {e}")
            continue

        start = max(0, anchor_line)
        window = all_lines[start : start + 40]
        # Pad to 40 lines if needed
        while len(window) < 40:
            window.append("\n")

        # Process lines: expand tabs, truncate to 80 chars
        processed = []
        for line in window:
            line = line.rstrip("\n").rstrip("\r")
            line = line.expandtabs(4)[:80]
            processed.append(line)

        txt_dst = data / f"{stem}.txt"
        with open(txt_dst, "w", encoding="utf-8") as tf:
            tf.write("\n".join(processed))

        prepared += 1

    print(f"[prepare_mini_data] Prepared {prepared}/{n_samples} samples in {data_dir}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(
    model: CoderVLModel,
    tokenizer,
    data_dir: str,
    output_dir: str,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    overfit: bool = False,
) -> None:
    """Main training loop with two-group optimizer."""
    device = next(model.projector.parameters()).device

    # Dataset
    full_dataset = JointDataset(data_dir, tokenizer, max_text_tokens=MAX_TEXT_TOKENS)
    if len(full_dataset) == 0:
        raise RuntimeError(f"No samples found in {data_dir}")

    if overfit:
        # Plumbing test: single sample, no val
        from torch.utils.data import Subset
        train_dataset = Subset(full_dataset, [0])
        val_dataset = None
        print(f"[OVERFIT MODE] Using 1 sample, no validation")
    else:
        # Train/val split
        n_val = max(1, int(len(full_dataset) * VAL_SPLIT))
        n_train = len(full_dataset) - n_val
        generator = torch.Generator().manual_seed(SEED)
        train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val], generator=generator)
        print(f"[train] Split: {n_train} train, {n_val} val")

    collate_fn.pad_token_id = tokenizer.pad_token_id

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=False,
    )
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

    # Optimizer: two parameter groups
    optimizer = torch.optim.AdamW([
        {"params": model.projector.parameters(), "lr": LR_PROJECTOR},
        {"params": [p for p in model.llm.parameters() if p.requires_grad], "lr": LR_LORA},
    ])

    # Cosine LR scheduler with warmup
    total_steps = (len(train_loader) * epochs + GRAD_ACCUM_STEPS - 1) // GRAD_ACCUM_STEPS

    def lr_lambda(step: int) -> float:
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Output dir
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Training
    global_step = 0
    model.projector.train()
    model.llm.train()

    print(f"\n{'='*60}")
    print(f"Starting training: {epochs} epochs, {len(train_loader)} steps/epoch")
    print(f"Grad accum: {GRAD_ACCUM_STEPS}, effective batch: {batch_size * GRAD_ACCUM_STEPS}")
    print(f"Total optimizer steps: ~{total_steps}")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue

            vision = batch["vision"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss = model(vision, input_ids, attention_mask, labels)
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()

            epoch_loss += loss.item() * GRAD_ACCUM_STEPS
            n_batches += 1

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                # Gradient clipping on all trainable parameters
                trainable_params = (
                    list(model.projector.parameters())
                    + [p for p in model.llm.parameters() if p.requires_grad]
                )
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if overfit or global_step % 10 == 0:
                    avg = epoch_loss / n_batches if n_batches > 0 else 0
                    lr_proj = optimizer.param_groups[0]["lr"] * lr_lambda(global_step)
                    lr_lo = optimizer.param_groups[1]["lr"] * lr_lambda(global_step)
                    print(
                        f"  [step {global_step:>5d}] loss={loss.item() * GRAD_ACCUM_STEPS:.4f}  "
                        f"avg={avg:.4f}  lr_proj={lr_proj:.2e}  lr_lora={lr_lo:.2e}"
                    )

                # Checkpoint
                if global_step % CKPT_EVERY_STEPS == 0:
                    save_checkpoint(model, out, global_step)

        # End of epoch
        avg_train_loss = epoch_loss / max(1, n_batches)
        print(f"\nEpoch {epoch+1}/{epochs}  train_loss={avg_train_loss:.4f}")

        # Validation
        if val_loader is not None:
            val_loss = validate(model, val_loader, device)
            print(f"  val_loss={val_loss:.4f}")

        # Save epoch checkpoint
        save_checkpoint(model, out, global_step, tag=f"epoch{epoch+1}")

    print("\nTraining complete.")
    save_checkpoint(model, out, global_step, tag="final")


def validate(model: CoderVLModel, val_loader: DataLoader, device: str) -> float:
    """Run validation and return average loss."""
    model.projector.eval()
    model.llm.eval()
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            vision = batch["vision"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss = model(vision, input_ids, attention_mask, labels)
            total_loss += loss.item()
            n += 1

    model.projector.train()
    model.llm.train()
    return total_loss / max(1, n)


def save_checkpoint(model: CoderVLModel, out_dir: Path, step: int, tag: str = "") -> None:
    """Save projector state dict and LoRA adapter."""
    suffix = f"_{tag}" if tag else f"_step{step}"
    ckpt_dir = out_dir / f"checkpoint{suffix}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save projector
    proj_path = ckpt_dir / "projector.pt"
    torch.save({"projector_state_dict": model.projector.state_dict()}, proj_path)

    # Save LoRA adapter
    model.llm.save_pretrained(str(ckpt_dir / "lora_adapter"))

    print(f"  [checkpoint] Saved to {ckpt_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: Joint LoRA + Projector Training")
    parser.add_argument("--data-dir", type=str, default="MVV/Phase_3/mini_data",
                        help="Directory with paired .pt/.txt files")
    parser.add_argument("--prepare-data", action="store_true",
                        help="Create mini dataset from manifest, then exit")
    parser.add_argument("--projector-ckpt", type=str,
                        default="MVV/Phase_2/checkpoints/best_aligned.pt",
                        help="Path to projector checkpoint to initialize from")
    parser.add_argument("--output-dir", type=str, default="MVV/Phase_3/checkpoints",
                        help="Checkpoint output directory")
    parser.add_argument("--overfit", action="store_true",
                        help="Plumbing test: 1 sample, no val, verbose loss")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    # Resolve paths relative to repo root
    repo_root = Path(__file__).resolve().parent.parent.parent
    data_dir = repo_root / args.data_dir
    output_dir = repo_root / args.output_dir
    projector_ckpt = repo_root / args.projector_ckpt

    if args.prepare_data:
        prepare_mini_data(str(repo_root), str(data_dir))
        return

    # Model path
    model_path = str(
        Path.home()
        / ".cache/huggingface/hub"
        / "models--deepseek-ai--DeepSeek-Coder-V2-Lite-Instruct"
        / "snapshots"
        / "e434a23f91ba5b4923cf6c9d9a238eb4a08e3a11"
    )

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[main] repo_root={repo_root}")
    print(f"[main] data_dir={data_dir}")
    print(f"[main] model_path={model_path}")
    print(f"[main] device={device}")

    model, tokenizer = load_model(
        model_path=model_path,
        projector_ckpt_path=str(projector_ckpt),
        device=device,
    )

    # Set collate_fn pad token
    collate_fn.pad_token_id = tokenizer.pad_token_id

    train(
        model=model,
        tokenizer=tokenizer,
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        overfit=args.overfit,
    )


if __name__ == "__main__":
    main()
