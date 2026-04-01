"""
Smoke test -- single-GPU forward/backward validation for Stage 1 QLoRA pipeline.

Runs 50 training steps on a 200-sample subset, then exits.
No DDP, no torchrun, no checkpointing. Just validates the full code path works.
"""

import os
os.environ["PYTHONNOUSERSITE"] = "1"

import argparse
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from pathlib import Path
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---------------------------------------------------------------------------
# Constants (same as train_stage1.py)
# ---------------------------------------------------------------------------
MAX_TEXT_TOKENS = 768
N_VISUAL_TOKENS = 256
GRAD_ACCUM_STEPS = 2
LR_PROJECTOR = 1e-5
LR_LORA = 5e-6
BATCH_SIZE = 2
SEED = 42
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
WARMUP_RATIO = 0.05

# Smoke test limits
MAX_SAMPLES = 200
MAX_STEPS = 50


# ---------------------------------------------------------------------------
# ConvRoPEProjector  (copied from train_stage1.py exactly)
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
# CoderVLModel (copied from train_stage1.py exactly)
# ---------------------------------------------------------------------------
class CoderVLModel(nn.Module):
    """Wraps ConvRoPEProjector + QLoRA-adapted DeepSeek LLM for DDP training."""
    def __init__(self, projector: ConvRoPEProjector, llm: nn.Module) -> None:
        super().__init__()
        self.projector = projector
        self.llm = llm

    def forward(self, vision, input_ids, attention_mask, labels):
        B, S = input_ids.shape
        visual_embeds = self.projector(vision)

        # Text embeddings under no_grad to protect embedding space
        # NOTE: prepare_model_for_kbit_training casts embedding layer to fp32,
        # so text_embeds comes out as float32. We must cast back to fp16 to match
        # the LLM's compute dtype and avoid VRAM blow-up / dtype mismatches.
        with torch.no_grad():
            text_embeds = self.llm.get_input_embeddings()(input_ids)
        text_embeds = text_embeds.clone().to(torch.float16)

        # Cast projected features to match text embedding dtype (fp16)
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
# JointDataset (copied from train_stage1.py exactly)
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
# collate_fn (copied from train_stage1.py exactly)
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
# Model loading (same as train_stage1.py, but no DDP, device always cuda:0)
# ---------------------------------------------------------------------------
def load_model(model_path: str):
    """Load tokenizer, 4-bit quantized LLM with LoRA, and ConvRoPEProjector."""
    device = torch.device("cuda:0")

    print(f"[load_model] Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit quantization config for V100 (QLoRA / nf4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"[load_model] Loading LLM in 4-bit (nf4) from {model_path}")
    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": 0},
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
    )

    # Prepare for k-bit training (casts layernorm to fp32, enables gradient on input embeddings)
    llm = prepare_model_for_kbit_training(llm)

    # Enable gradient checkpointing to save VRAM
    llm.gradient_checkpointing_enable()

    # Surgical LoRA targets for DeepSeek-Coder-V2
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
    llm.print_trainable_parameters()

    # Projector -- random init, placed on device (fp32 master weights)
    projector = ConvRoPEProjector(feat_dim=1152, proj_dim=2048)
    projector.to(device=device)

    model = CoderVLModel(projector, llm)

    # Upcast all trainable params to fp32 (required for GradScaler.unscale_()).
    # Base model stays 4-bit quantized; only LoRA adapters + projector become fp32.
    # autocast handles the fp16 forward pass automatically.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    # Print param counts
    proj_trainable = sum(p.numel() for p in model.projector.parameters() if p.requires_grad)
    lora_trainable = sum(p.numel() for p in model.llm.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.llm.parameters() if not p.requires_grad)
    print(f"[load_model] Projector trainable params: {proj_trainable:,}")
    print(f"[load_model] LoRA trainable params:      {lora_trainable:,}")
    print(f"[load_model] Frozen LLM params:          {frozen:,}")
    vram_mb = torch.cuda.memory_allocated(device) / 1024**2
    print(f"[load_model] VRAM after model load: {vram_mb:.0f} MB")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Smoke test training loop
# ---------------------------------------------------------------------------
def smoke_test(model, tokenizer, data_dir, batch_size):
    """Run MAX_STEPS forward+backward steps on a subset, then exit."""
    device = torch.device("cuda:0")

    # Dataset -- full load, then take first MAX_SAMPLES via Subset
    full_dataset = JointDataset(data_dir, tokenizer)
    subset_size = min(MAX_SAMPLES, len(full_dataset))
    subset = Subset(full_dataset, list(range(subset_size)))
    print(f"[smoke_test] Using {subset_size} samples (subset of {len(full_dataset)})")

    collate_fn.pad_token_id = tokenizer.pad_token_id

    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collate_fn,
    )

    # Optimizer -- two param groups (same as train_stage1.py)
    optimizer = torch.optim.AdamW([
        {"params": model.projector.parameters(), "lr": LR_PROJECTOR},
        {"params": [p for p in model.llm.parameters() if p.requires_grad], "lr": LR_LORA},
    ])

    # LR scheduler
    total_steps = MAX_STEPS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # GradScaler for mixed-precision training -- prevents fp16 gradient overflow (NaN fix)
    scaler = torch.cuda.amp.GradScaler()

    print(f"[smoke_test] Starting {MAX_STEPS}-step smoke test (batch_size={batch_size}, grad_accum={GRAD_ACCUM_STEPS})")
    print(f"[smoke_test] VRAM before training: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")

    model.train()
    optimizer.zero_grad()

    global_step = 0
    accum_step = 0
    skipped = 0

    while global_step < MAX_STEPS:
        for batch in loader:
            if global_step >= MAX_STEPS:
                break

            if batch is None:
                skipped += 1
                continue

            vision = batch["vision"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            try:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    loss = model(vision, input_ids, attention_mask, labels)
                    scaled_loss = loss / GRAD_ACCUM_STEPS
                scaler.scale(scaled_loss).backward()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"[smoke_test] OOM at step {global_step} (seq_len={input_ids.shape[1]}), skipping")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    skipped += 1
                    continue
                else:
                    raise

            accum_step += 1

            if accum_step % GRAD_ACCUM_STEPS == 0:
                all_params = (
                    list(model.projector.parameters())
                    + [p for p in model.llm.parameters() if p.requires_grad]
                )
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            # Print loss every step
            print(f"  step {global_step:3d}/{MAX_STEPS} | loss={loss.item():.4f} | "
                  f"lr_proj={scheduler.get_last_lr()[0]:.2e} | lr_lora={scheduler.get_last_lr()[1]:.2e}")

            # Print VRAM every 10 steps
            if global_step % 10 == 0:
                alloc_gb = torch.cuda.memory_allocated(device) / 1024**3
                reserved_gb = torch.cuda.memory_reserved(device) / 1024**3
                print(f"  [VRAM] allocated={alloc_gb:.2f} GB | reserved={reserved_gb:.2f} GB")

        # If we exhausted the loader but haven't hit MAX_STEPS, loop again
        if global_step < MAX_STEPS:
            print(f"[smoke_test] Loader exhausted at step {global_step}, cycling data...")

    # Handle leftover accumulated gradients
    if accum_step % GRAD_ACCUM_STEPS != 0:
        all_params = (
            list(model.projector.parameters())
            + [p for p in model.llm.parameters() if p.requires_grad]
        )
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

    print()
    print(f"[smoke_test] Completed {MAX_STEPS} steps successfully.")
    print(f"[smoke_test] Skipped batches: {skipped}")
    alloc_gb = torch.cuda.memory_allocated(device) / 1024**3
    reserved_gb = torch.cuda.memory_reserved(device) / 1024**3
    print(f"[smoke_test] Final VRAM: allocated={alloc_gb:.2f} GB | reserved={reserved_gb:.2f} GB")
    print()
    print("=" * 60)
    print("  SMOKE TEST PASSED")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    torch.cuda.set_device(0)

    parser = argparse.ArgumentParser(description="Smoke test -- single-GPU Stage 1 QLoRA validation")
    parser.add_argument("--data-dir", type=str, default="MVV/Phase_3/full_data/tensors_and_texts",
                        help="Path to paired .pt/.txt data directory")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to DeepSeek-Coder-V2-Lite-Instruct model")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    print("=" * 60)
    print("  SMOKE TEST -- Single-GPU Stage 1 QLoRA Validation")
    print("=" * 60)
    print(f"[main] Data: {args.data_dir}")
    print(f"[main] Model: {args.model_path}")
    print(f"[main] Batch size: {args.batch_size}")
    print(f"[main] Max samples: {MAX_SAMPLES}")
    print(f"[main] Max steps: {MAX_STEPS}")
    print(f"[main] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[main] VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    model, tokenizer = load_model(args.model_path)
    smoke_test(model, tokenizer, args.data_dir, args.batch_size)


if __name__ == "__main__":
    main()
