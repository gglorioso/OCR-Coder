"""
train_1_9c.py — Phase 1.9c: Large-Scale Alignment Training

Scales Phase 2's alignment training from ~500 samples to the full ~8,980-sample
manifest (MVV/Phase_1_1/data_mvv/manifest.jsonl) for 5 epochs.

Architecture:
  Code image → SigLIP features [1024, 1152]
             → ConvRoPEProjector [256, 2048]
             → concat with tokenized source [T, 2048]
             → DeepSeek-Coder-V2-Lite-Instruct (frozen, 8-bit)
             → cross-entropy loss on text tokens only

Initialization: MVV/Phase_2/checkpoints/best_aligned.pt (val_loss=1.392)
Only the ConvRoPEProjector is trained. The LLM is strictly frozen.
"""

import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch.optim.lr_scheduler import LambdaLR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_TEXT_TOKENS = 512
GRAD_ACCUM_STEPS = 4
LR = 1e-5          # CRITICAL: never 1e-4 — causes catastrophic forgetting
WARMUP_STEPS = 100
EPOCHS = 1
BATCH_SIZE = 1
VAL_SPLIT = 0.1
SEED = 42
CKPT_EVERY_STEPS = 1000

# Number of visual tokens output by ConvRoPEProjector
N_VISUAL_TOKENS = 256

# ---------------------------------------------------------------------------
# ConvRoPEProjector (inlined from MVV/Phase_1_9/a/scripts/model.py)
# ---------------------------------------------------------------------------

def _sinusoidal_freqs(seq_len: int, half_dim: int, device: torch.device):
    """Return (cos, sin) tables each [seq_len, half_dim]."""
    i      = torch.arange(half_dim, device=device, dtype=torch.float32)
    theta  = 1.0 / (10000.0 ** (i / half_dim))
    pos    = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(pos, theta)          # [seq_len, half_dim]
    return angles.cos(), angles.sin()


def _rope_rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE rotation. x: [..., seq_len, 2*half_dim]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def apply_2d_rope_16x16(x: torch.Tensor) -> torch.Tensor:
    """
    Inject 2D positional information into x via RoPE.

    The 256 tokens are laid out in a 16×16 row-major grid:
      token i  →  row = i // 16,  col = i % 16

    Dimension split (D=1152):
      dims [0:576]    carry Y-axis (row) RoPE
      dims [576:1152] carry X-axis (col) RoPE

    Args:
        x: [B, 256, 1152]
    Returns:
        [B, 256, 1152]
    """
    B, T, D = x.shape
    assert T == 256 and D == 1152, f"Expected [B,256,1152], got {x.shape}"

    half_D   = D // 2        # 576
    half_dim = half_D // 2   # 288 sin/cos pairs per axis
    device   = x.device

    rows = torch.arange(256, device=device) // 16   # [256]
    cols = torch.arange(256, device=device) % 16    # [256]

    cos_table, sin_table = _sinusoidal_freqs(16, half_dim, device)
    # [16, 288]

    cos_row = cos_table[rows]   # [256, 288]
    sin_row = sin_table[rows]
    cos_col = cos_table[cols]   # [256, 288]
    sin_col = sin_table[cols]

    x_row = x[:, :, :half_D]    # [B, 256, 576]
    x_col = x[:, :, half_D:]    # [B, 256, 576]

    x_row_rot = _rope_rotate(x_row, cos_row, sin_row)
    x_col_rot = _rope_rotate(x_col, cos_col, sin_col)

    return torch.cat([x_row_rot, x_col_rot], dim=-1)   # [B, 256, 1152]


class ConvRoPEProjector(nn.Module):
    """
    Compresses [B, 1024, 1152] raw SigLIP tokens to [B, 256, 2048] via:
      strided conv (32→16)  →  2D RoPE  →  MLP (1152→2048)
    """

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
        """x: [B, 1024, 1152]  →  [B, 256, 2048]"""
        B, N, C = x.shape
        assert N == self.grid_in ** 2 and C == self.feat_dim, \
            f"Expected [B,{self.grid_in**2},{self.feat_dim}], got {x.shape}"

        # 1. Reshape to spatial grid
        x = x.reshape(B, self.grid_in, self.grid_in, C).permute(0, 3, 1, 2)
        # [B, 1152, 32, 32]

        # 2. Stride-2 conv: 32×32 → 16×16
        x = self.conv(x)
        # [B, 1152, 16, 16]

        # 3. Flatten spatial dims back to sequence
        x = x.flatten(2).transpose(1, 2)
        # [B, 256, 1152]

        # 4. Inject 2D positional information via RoPE
        x = apply_2d_rope_16x16(x)
        # [B, 256, 1152]

        # 5. Project to LLM embedding dimension
        x = self.mlp(x)
        # [B, 256, 2048]

        return x


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AlignmentDataset(Dataset):
    """
    Loads (vision_features, tokenized_source) pairs for alignment training.

    vision:    [1024, 1152] fp32 tensor  (SigLIP tokens)
    input_ids: [T]          long tensor  (T <= MAX_TEXT_TOKENS)
    """

    def __init__(self, entries, repo_root, tokenizer, max_text_tokens=MAX_TEXT_TOKENS):
        self.entries = entries
        self.repo_root = Path(repo_root)
        self.tokenizer = tokenizer
        self.max_text_tokens = max_text_tokens

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        try:
            stem = Path(entry["image"]).stem
            feat_path = self.repo_root / "MVV/Phase_1_9/a/data/features" / f"{stem}.pt"
            src_path  = self.repo_root / "Scraped Repos" / entry["source_file"]

            # Load vision features → [1024, 1152] fp32
            vision = torch.load(feat_path, weights_only=False).float()
            assert vision.shape == (1024, 1152), \
                f"Unexpected vision shape {vision.shape} for {feat_path}"

            # Load source text — only the 40-line window visible in the image
            # Apply the same truncation as the image renderer: expandtabs(4)[:80]
            anchor = entry.get("anchor_line", 0)
            with open(src_path, errors="replace") as _f:
                all_lines = _f.readlines()
            source_text = "".join(
                line.expandtabs(4)[:80] for line in all_lines[anchor: anchor + 40]
            )

            # Tokenize
            enc = self.tokenizer(
                source_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_text_tokens,
                add_special_tokens=False,
            )
            input_ids = enc["input_ids"].squeeze(0)   # [T]

            return {"vision": vision, "input_ids": input_ids}

        except Exception as e:
            # Missing file, load failure, etc. — filtered by collate_fn
            return None


# ---------------------------------------------------------------------------
# Collate function (module-level — required for DataLoader pickling)
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """
    Filters None items, stacks vision tensors, right-pads input_ids.

    Returns:
        dict with keys:
            "vision":             [B, 1024, 1152]
            "input_ids":          [B, T]
            "text_attention_mask":[B, T]
        or None if every item in the batch was None.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    vision = torch.stack([b["vision"] for b in batch])   # [B, 1024, 1152]

    # Determine max text length in this batch
    max_len = max(b["input_ids"].shape[0] for b in batch)

    # Right-pad input_ids with pad_token_id
    # NOTE: pad_token_id is monkey-patched onto collate_fn by main()
    pad_id = collate_fn.pad_token_id

    padded_ids  = []
    attn_masks  = []
    for b in batch:
        ids    = b["input_ids"]
        T      = ids.shape[0]
        pad_n  = max_len - T
        padded = torch.cat([ids, torch.full((pad_n,), pad_id, dtype=torch.long)])
        mask   = torch.cat([torch.ones(T, dtype=torch.long),
                            torch.zeros(pad_n, dtype=torch.long)])
        padded_ids.append(padded)
        attn_masks.append(mask)

    return {
        "vision":              vision,                           # [B, 1024, 1152]
        "input_ids":           torch.stack(padded_ids),          # [B, T]
        "text_attention_mask": torch.stack(attn_masks),          # [B, T]
    }

# Placeholder; overwritten in main() with the actual pad_token_id
collate_fn.pad_token_id = 0


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_projector(repo_root: Path, device: torch.device):
    """
    Loads ConvRoPEProjector from Phase 2's best_aligned.pt checkpoint.

    Phase 2's checkpoint saves projector_state_dict directly (no prefix).
    """
    projector = ConvRoPEProjector(feat_dim=1152, proj_dim=2048).to(device)

    ckpt_path = repo_root / "MVV/Phase_2/checkpoints/best_aligned.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    proj_state = ckpt["projector_state_dict"]
    projector.load_state_dict(proj_state)
    projector.requires_grad_(True)
    projector.train()

    n_params = sum(p.numel() for p in projector.parameters() if p.requires_grad)
    print(f"[projector] Loaded from {ckpt_path.name}  ({n_params:,} trainable params)")
    val_loss = ckpt.get("val_loss", "unknown")
    print(f"[projector] Phase 2 checkpoint val_loss={val_loss}")
    return projector


def load_llm(model_path: str):
    """
    Loads DeepSeek-Coder-V2-Lite-Instruct in 8-bit with all weights frozen.
    Returns (tokenizer, llm).
    """
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
    )
    # LLM strictly frozen — no gradients through any LLM parameter
    llm.requires_grad_(False)
    llm.eval()

    total_params = sum(p.numel() for p in llm.parameters())
    print(f"[llm] Loaded DeepSeek-Coder-V2-Lite-Instruct 8-bit  ({total_params / 1e9:.1f}B params, all frozen)")
    return tokenizer, llm


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def forward_batch(batch, projector, llm, embed_fn, tokenizer, device):
    """
    Single forward pass for one batch.

    Sequence layout:
        [visual_tokens (256)] [text_tokens (T)]

    Loss is computed only over text positions (first 256 positions masked to -100).

    Returns:
        scalar loss tensor
    """
    vision    = batch["vision"].to(device)       # [B, 1024, 1152]
    input_ids = batch["input_ids"].to(device)    # [B, T]
    B, T = input_ids.shape

    # 1. Vision → projector → [B, 256, 2048]
    visual_embeds = projector(vision)

    # 2. Text embeddings — no grad through embed_fn to protect LLM embedding space
    with torch.no_grad():
        text_embeds = embed_fn(input_ids)        # [B, T, 2048]

    # 3. Cast to match LLM's expected dtype
    target_dtype  = embed_fn.weight.dtype
    visual_embeds = visual_embeds.to(target_dtype)
    text_embeds   = text_embeds.to(target_dtype)

    # 4. Concatenate: [B, 256+T, 2048]
    full_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
    seq_len = full_embeds.shape[1]   # 256 + T

    # 5. Position IDs — explicit to bypass DeepSeek RoPE cache bug
    position_ids = (
        torch.arange(seq_len, dtype=torch.long, device=device)
        .unsqueeze(0)
        .expand(B, -1)
    )

    # 6. Attention mask — full attention over all (vision + text) tokens
    attention_mask = torch.ones((B, seq_len), dtype=torch.long, device=device)

    # 7. Labels — first 256 positions masked (-100), real token ids for text positions
    labels = torch.full((B, seq_len), -100, dtype=torch.long, device=device)
    labels[:, N_VISUAL_TOKENS:] = input_ids   # positions 256..256+T

    # Mask padding tokens in the text section only (positions 256 onwards)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    text_section = labels[:, N_VISUAL_TOKENS:]          # view into labels
    text_section[text_section == pad_id] = -100
    labels[:, N_VISUAL_TOKENS:] = text_section

    # 8. LLM forward (frozen)
    # NOTE: llm.requires_grad_(False) already blocks gradient accumulation
    # through frozen parameters. We do NOT wrap in torch.no_grad() so that
    # gradients flow back through full_embeds → visual_embeds → projector.
    outputs = llm(
        inputs_embeds=full_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        labels=labels,
        use_cache=False,
        return_dict=True,
    )
    return outputs.loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_training(train_loader, val_loader, projector, llm, embed_fn,
                 tokenizer, device, out_dir, log_path):
    optimizer = torch.optim.AdamW(projector.parameters(), lr=LR)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return max(1e-6, step / WARMUP_STEPS)
        return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda)

    global_step = 0
    best_val_loss = float("inf")
    optimizer.zero_grad()

    for epoch in range(1, EPOCHS + 1):
        projector.train()
        train_losses = []

        for i, batch in enumerate(train_loader):
            if batch is None:
                continue

            loss = (
                forward_batch(batch, projector, llm, embed_fn, tokenizer, device)
                / GRAD_ACCUM_STEPS
            )
            loss.backward()
            train_losses.append(loss.item() * GRAD_ACCUM_STEPS)

            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % CKPT_EVERY_STEPS == 0 and global_step > 0:
                    ckpt_path = out_dir / f"step_{global_step}.pt"
                    torch.save({
                        "epoch": epoch,
                        "global_step": global_step,
                        "projector_state_dict": projector.state_dict(),
                    }, ckpt_path)
                    print(f"  [ckpt] Saved mid-epoch checkpoint -> {ckpt_path}")

            if global_step % 50 == 0 and global_step > 0:
                print(
                    f"  step {global_step} | loss {train_losses[-1]:.4f}"
                    f" | lr {scheduler.get_last_lr()[0]:.2e}"
                )

        # --- Validation ---
        projector.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                loss = forward_batch(batch, projector, llm, embed_fn, tokenizer, device)
                val_losses.append(loss.item())

        avg_train = sum(train_losses) / max(1, len(train_losses))
        avg_val   = sum(val_losses)   / max(1, len(val_losses))
        print(
            f"\nEpoch {epoch}/{EPOCHS}"
            f" — train_loss={avg_train:.4f} | val_loss={avg_val:.4f}"
        )

        # --- Save every-epoch checkpoint ---
        epoch_ckpt = out_dir / f"epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "val_loss": avg_val,
                "train_loss": avg_train,
                "projector_state_dict": projector.state_dict(),
            },
            epoch_ckpt,
        )
        print(f"  Epoch checkpoint -> {epoch_ckpt}")

        # --- Save best checkpoint ---
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_ckpt = out_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "train_loss": avg_train,
                    "projector_state_dict": projector.state_dict(),
                },
                best_ckpt,
            )
            print(f"  New best val_loss={best_val_loss:.4f} -> {best_ckpt}")

        # --- Log to JSONL ---
        log_entry = {
            "epoch": epoch,
            "train_loss": avg_train,
            "val_loss": avg_val,
            "best_val_loss": best_val_loss,
            "global_step": global_step,
        }
        with open(log_path, "a") as lf:
            lf.write(json.dumps(log_entry) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    repo_root = Path(__file__).resolve().parents[3]  # OCR-Coder/
    out_dir   = repo_root / "MVV/Phase_1_9/c/checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = repo_root / "MVV/Phase_1_9/c/results"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "training_log.jsonl"

    device = torch.device("cuda:0")

    model_path = str(
        Path.home()
        / ".cache/huggingface/hub"
        / "models--deepseek-ai--DeepSeek-Coder-V2-Lite-Instruct"
        / "snapshots"
        / "e434a23f91ba5b4923cf6c9d9a238eb4a08e3a11"
    )

    # --- Tokenizer first (needed by Dataset) ---
    print("[setup] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    collate_fn.pad_token_id = pad_token_id

    # --- Manifest (full dataset: ~8,980 entries) ---
    manifest_path = repo_root / "MVV/Phase_1_1/data_mvv/manifest.jsonl"
    entries = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            stem      = Path(e["image"]).stem
            feat_path = repo_root / "MVV/Phase_1_9/a/data/features" / f"{stem}.pt"
            src_path  = repo_root / "Scraped Repos" / e["source_file"]
            if feat_path.exists() and src_path.exists():
                entries.append(e)

    print(f"[setup] Valid entries: {len(entries)}")

    # --- Fixed-seed 90/10 train/val split (seed=42) ---
    random.seed(SEED)
    random.shuffle(entries)
    n_val         = int(len(entries) * VAL_SPLIT)
    val_entries   = entries[:n_val]
    train_entries = entries[n_val:]   # use ALL remaining (full scale, no subsample)
    print(f"[setup] Train: {len(train_entries)} | Val: {len(val_entries)}")

    # --- Datasets and loaders ---
    train_ds = AlignmentDataset(train_entries, repo_root, tokenizer, MAX_TEXT_TOKENS)
    val_ds   = AlignmentDataset(val_entries,   repo_root, tokenizer, MAX_TEXT_TOKENS)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # --- Load models ---
    projector = load_projector(repo_root, device)
    print("[setup] Loading LLM (8-bit, this may take a minute) ...")
    tokenizer, llm = load_llm(model_path)
    embed_fn = llm.get_input_embeddings()
    print("[setup] Models loaded. Beginning training.\n")

    # --- Train ---
    run_training(
        train_loader, val_loader,
        projector, llm, embed_fn, tokenizer,
        device, out_dir, log_path,
    )

    print("\n[done] Training complete.")
    print(f"[done] Best checkpoint: {out_dir / 'best.pt'}")
    print(f"[done] Training log:    {log_path}")


if __name__ == "__main__":
    main()
