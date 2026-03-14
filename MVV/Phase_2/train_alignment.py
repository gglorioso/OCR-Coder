"""
train_alignment.py — Phase 2: Multimodal Alignment Training

Trains the ConvRoPEProjector to align vision features with the DeepSeek-Coder-V2
LLM embedding space using a next-token prediction (causal LM) objective.

Architecture:
  Code image → SigLIP features [1024, 1152]
             → ConvRoPEProjector [256, 2048]
             → concat with tokenized source [T, 2048]
             → DeepSeek-Coder-V2 (frozen, 8-bit)
             → cross-entropy loss on text tokens only

Only the projector is trained. The LLM is frozen.
"""

import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch.optim.lr_scheduler import LambdaLR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_TEXT_TOKENS = 512
GRAD_ACCUM_STEPS = 4
LR = 1e-5
WARMUP_STEPS = 100
EPOCHS = 2
BATCH_SIZE = 1
VAL_SPLIT = 0.1
SEED = 42

# Number of visual tokens output by ConvRoPEProjector
N_VISUAL_TOKENS = 256

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
            feat_path = self.repo_root / "MVV/Phase_1_9/data/features" / f"{stem}.pt"
            src_path  = self.repo_root / "Scraped Repos" / entry["source_file"]

            # Load vision features → [1024, 1152] fp32
            vision = torch.load(feat_path, weights_only=False).float()
            assert vision.shape == (1024, 1152), \
                f"Unexpected vision shape {vision.shape} for {feat_path}"

            # Load source text — only the 40-line window visible in the image
            anchor = entry.get("anchor_line", 0)
            with open(src_path, errors="replace") as _f:
                all_lines = _f.readlines()
            source_text = "".join(all_lines[anchor: anchor + 40])

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
    Loads ConvRoPEProjector from the Phase 1.9 checkpoint.

    The checkpoint was saved from ConvRoPEKeywordDetector, so all keys are
    prefixed with either 'projector.' or 'probe.'. We strip 'projector.'
    and discard 'probe.*' keys.
    """
    sys.path.insert(0, str(repo_root / "MVV/Phase_1_9/scripts"))
    from model import ConvRoPEProjector  # noqa: E402

    projector = ConvRoPEProjector(feat_dim=1152, proj_dim=2048).to(device)

    ckpt_path = repo_root / "MVV/Phase_1_9/checkpoints/best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]

    proj_state = {
        k[len("projector."):]: v
        for k, v in state.items()
        if k.startswith("projector.")
    }
    projector.load_state_dict(proj_state)
    projector.requires_grad_(True)
    projector.train()

    n_params = sum(p.numel() for p in projector.parameters() if p.requires_grad)
    print(f"[projector] Loaded from {ckpt_path.name}  ({n_params:,} trainable params)")
    return projector


def load_llm(model_path: str):
    """
    Loads the DeepSeek-Coder-V2 LLM in 8-bit with all weights frozen.
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
    llm.requires_grad_(False)
    llm.eval()

    total_params = sum(p.numel() for p in llm.parameters())
    print(f"[llm] Loaded DeepSeek-Coder-V2 8-bit  ({total_params / 1e9:.1f}B params, all frozen)")
    return tokenizer, llm


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def forward_batch(batch, projector, llm, embed_fn, tokenizer, device):
    """
    Single forward pass for one batch.

    Sequence layout:
        [visual_tokens (256)] [text_tokens (T)]

    Loss is computed only over text positions.

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

    # 5. Position IDs — explicit to bypass RoPE cache bugs
    position_ids = (
        torch.arange(seq_len, dtype=torch.long, device=device)
        .unsqueeze(0)
        .expand(B, -1)
    )

    # 6. Attention mask — full attention over all (vision + text) tokens
    attention_mask = torch.ones((B, seq_len), dtype=torch.long, device=device)

    # 7. Labels — -100 for visual tokens, real token ids for text tokens
    labels = torch.full((B, seq_len), -100, dtype=torch.long, device=device)
    labels[:, N_VISUAL_TOKENS:] = input_ids   # positions 256..256+T

    # Mask padding tokens in the text section only (positions 256 onwards)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    text_section = labels[:, N_VISUAL_TOKENS:]          # view into labels
    text_section[text_section == pad_id] = -100
    labels[:, N_VISUAL_TOKENS:] = text_section

    # 8. LLM forward (frozen)
    with torch.no_grad():
        # We cannot use torch.no_grad() for the whole call because we need
        # gradients w.r.t. visual_embeds → projector. Instead we use the
        # context only around the LLM's own parameters.
        pass  # intentional no-op — see note below

    # NOTE: llm.requires_grad_(False) already blocks gradient accumulation
    # through frozen parameters. We do NOT wrap in torch.no_grad() so that
    # gradients flow back through full_embeds → visual_embeds → projector.
    outputs = llm(
        inputs_embeds=full_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        labels=labels,
        return_dict=True,
    )
    return outputs.loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_training(train_loader, val_loader, projector, llm, embed_fn,
                 tokenizer, device, out_dir):
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

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            ckpt_path = out_dir / "best_aligned.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "projector_state_dict": projector.state_dict(),
                },
                ckpt_path,
            )
            print(f"  New best val_loss={best_val_loss:.4f} -> {ckpt_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    repo_root = Path(__file__).resolve().parents[2]
    out_dir   = repo_root / "MVV/Phase_2/checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # --- Manifest ---
    manifest_path = repo_root / "MVV/Phase_1_1/data_mvv/manifest.jsonl"
    entries = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            stem      = Path(e["image"]).stem
            feat_path = repo_root / "MVV/Phase_1_9/data/features" / f"{stem}.pt"
            src_path  = repo_root / "Scraped Repos" / e["source_file"]
            if feat_path.exists() and src_path.exists():
                entries.append(e)

    print(f"[setup] Valid entries: {len(entries)}")

    # --- 90/10 train/val split ---
    random.seed(SEED)
    random.shuffle(entries)
    n_val        = int(len(entries) * VAL_SPLIT)
    val_entries  = entries[:n_val]
    train_entries = entries[n_val:n_val + 500]  # subsample: ~8h at batch=1
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
        device, out_dir,
    )

    print("\n[done] Training complete.")
    print(f"[done] Best checkpoint: {out_dir / 'best_aligned.pt'}")


if __name__ == "__main__":
    main()
