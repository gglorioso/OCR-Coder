"""
Phase 3.3 — Joint LoRA + Projector Training (Fixed LR)

Surgical patch of the Phase 3 training script with these fixes:
  - lr_lora: 2e-4 → 5e-6 (fixes val divergence from Phase 3 run)
  - LR schedule: cosine warmup → flat LR (stable for 100-sample diagnostic)
  - Progress: tqdm for epoch-level progress tracking
  - Checkpointing: per-epoch save of projector.pth + lora_adapter/
  - model_path: now a CLI arg for SLURM portability

Architecture unchanged:
  Code image → SigLIP features [1024, 1152]
             → ConvRoPEProjector [256, 2048]
             → splice into token embeddings at placeholder positions
             → DeepSeek-Coder-V2-Lite-Instruct (8-bit + LoRA)
             → cross-entropy loss on text tokens only
"""

import os
os.environ["PYTHONNOUSERSITE"] = "1"

import argparse
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm

torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_TEXT_TOKENS = 768
N_VISUAL_TOKENS = 256
GRAD_ACCUM_STEPS = 4
LR_PROJECTOR = 1e-5
LR_LORA = 5e-6
BATCH_SIZE = 1
VAL_SPLIT = 0.1
SEED = 42
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


# ---------------------------------------------------------------------------
# ConvRoPEProjector
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
# CoderVLModel
# ---------------------------------------------------------------------------
class CoderVLModel(nn.Module):
    """Wraps ConvRoPEProjector + LoRA-adapted DeepSeek LLM."""
    def __init__(self, projector: ConvRoPEProjector, llm: nn.Module) -> None:
        super().__init__()
        self.projector = projector
        self.llm = llm

    def forward(self, vision, input_ids, attention_mask, labels):
        B, S = input_ids.shape
        visual_embeds = self.projector(vision)
        text_embeds = self.llm.get_input_embeddings()(input_ids).clone()
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

            placeholder = [pad_id] * N_VISUAL_TOKENS
            input_ids = placeholder + newline_ids + text_ids + [eos_id]

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
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_path: str, projector_ckpt_path: Optional[str], device: str):
    """Load tokenizer, 8-bit LLM with LoRA, and ConvRoPEProjector."""
    print(f"[load_model] Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[load_model] Loading LLM in 8-bit from {model_path}")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map={"": device},
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
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

    # Projector
    projector = ConvRoPEProjector(feat_dim=1152, proj_dim=2048).to(device)

    if projector_ckpt_path and Path(projector_ckpt_path).exists():
        print(f"[load_model] Loading projector weights from {projector_ckpt_path}")
        ckpt = torch.load(projector_ckpt_path, map_location="cpu", weights_only=True)
        if "projector_state_dict" in ckpt:
            projector.load_state_dict(ckpt["projector_state_dict"])
        else:
            projector.load_state_dict(ckpt)
        print("[load_model] Projector weights loaded successfully.")
    else:
        print("[load_model] No projector checkpoint found, using random init.")

    model = CoderVLModel(projector, llm)

    # Print param counts
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
def validate(model: CoderVLModel, val_loader: DataLoader, device: str) -> float:
    """Run validation and return average loss."""
    model.projector.eval()
    model.llm.eval()

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

            loss = model(vision, input_ids, attention_mask, labels)
            total_loss += loss.item()
            n_batches += 1

    model.projector.train()
    model.llm.train()

    return total_loss / n_batches if n_batches > 0 else float("inf")


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(model: CoderVLModel, save_dir: Path, epoch: int) -> None:
    """Save projector weights and LoRA adapter for a given epoch."""
    ckpt_dir = save_dir / f"epoch_{epoch}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {"projector_state_dict": model.projector.state_dict()},
        str(ckpt_dir / "projector.pth"),
    )
    model.llm.save_pretrained(str(ckpt_dir / "lora_adapter"))
    print(f"[save_checkpoint] Saved epoch {epoch} checkpoint to {ckpt_dir}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(
    model: CoderVLModel,
    tokenizer,
    data_dir: str,
    save_dir: str,
    epochs: int,
    batch_size: int,
) -> None:
    """Main training loop with gradient accumulation, validation, and checkpointing."""
    device = next(model.projector.parameters()).device.type
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Dataset and split
    full_dataset = JointDataset(data_dir, tokenizer)
    val_size = max(1, int(len(full_dataset) * VAL_SPLIT))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    print(f"[train] Train: {train_size}, Val: {val_size}")

    collate_fn.pad_token_id = tokenizer.pad_token_id

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )

    # Optimizer — two param groups, flat LR
    optimizer = torch.optim.AdamW([
        {"params": model.projector.parameters(), "lr": LR_PROJECTOR},
        {"params": [p for p in model.llm.parameters() if p.requires_grad], "lr": LR_LORA},
    ])

    # Training
    model.projector.train()
    model.llm.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_samples = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(pbar):
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
            n_samples += 1

            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                all_params = (
                    list(model.projector.parameters())
                    + [p for p in model.llm.parameters() if p.requires_grad]
                )
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_postfix(loss=f"{loss.item() * GRAD_ACCUM_STEPS:.4f}")

        avg_train_loss = epoch_loss / n_samples if n_samples > 0 else float("inf")
        print(f"[train] Epoch {epoch+1}/{epochs} — train_loss: {avg_train_loss:.4f}")

        # Validation
        val_loss = validate(model, val_loader, device)
        print(f"[train] Epoch {epoch+1}/{epochs} — val_loss:   {val_loss:.4f}")

        # Checkpoint
        save_checkpoint(model, save_dir, epoch + 1)

    # Final checkpoint
    save_checkpoint(model, save_dir, "final")
    print("Training complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 3.3 — Joint LoRA + Projector Training")
    parser.add_argument("--data-dir", type=str, default="MVV/Phase_3/mini_data",
                        help="Path to paired .pt/.txt data directory")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to DeepSeek-Coder-V2-Lite-Instruct model")
    parser.add_argument("--projector-ckpt", type=str, default="MVV/Phase_2/checkpoints/best_aligned.pt",
                        help="Path to pretrained projector checkpoint")
    parser.add_argument("--save-dir", type=str, default="MVV/Phase_3/Phase_3_3/checkpoints",
                        help="Directory for saving checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[main] model_path: {args.model_path}")
    print(f"[main] data_dir:   {args.data_dir}")
    print(f"[main] device:     {device}")

    model, tokenizer = load_model(args.model_path, args.projector_ckpt, device)
    train(model, tokenizer, args.data_dir, args.save_dir, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
