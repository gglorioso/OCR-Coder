"""
Phase 1.8 training script — Spatial Contrastive Vision-Language Adapter.

Loss:    BCEWithLogitsLoss(sim, target_mask)
Metrics: val_loss, val_pos_sim, val_neg_sim, val_gap (key: want > 0.3)
"""

import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Local imports (same directory)
import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset_1_8 import SpatialContrastiveDataset
from model_1_8 import ContrastiveAdapter


# ---------------------------------------------------------------------------
# LR scheduler with linear warmup + cosine decay
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        progress = float(current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_pos_sim = 0.0
    total_neg_sim = 0.0
    total_pos_count = 0
    total_neg_count = 0
    n_batches = 0

    for batch in loader:
        if batch is None:
            continue
        vision = batch["vision"].to(device)         # [B, 64, 1152]
        text = batch["text"].to(device)             # [B, 1152]
        mask = batch["target_mask"].to(device)      # [B, 64]

        sim = model(vision, text)                   # [B, 64] — raw logits
        loss = criterion(sim, mask)
        total_loss += loss.item()
        n_batches += 1

        prob = torch.sigmoid(sim)                   # [B, 64]
        pos_mask = mask.bool()
        neg_mask = ~pos_mask

        if pos_mask.any():
            total_pos_sim += prob[pos_mask].sum().item()
            total_pos_count += pos_mask.sum().item()
        if neg_mask.any():
            total_neg_sim += prob[neg_mask].sum().item()
            total_neg_count += neg_mask.sum().item()

    val_loss = total_loss / max(1, n_batches)
    val_pos_sim = total_pos_sim / max(1, total_pos_count)
    val_neg_sim = total_neg_sim / max(1, total_neg_count)
    val_gap = val_pos_sim - val_neg_sim

    return {
        "val_loss": val_loss,
        "val_pos_sim": val_pos_sim,
        "val_neg_sim": val_neg_sim,
        "val_gap": val_gap,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Dataset ----
    print("Loading dataset …")
    full_dataset = SpatialContrastiveDataset(
        jsonl_path=args.ground_truth,
        feat_dir=args.feat_dir,
        text_emb_path=args.text_emb,
    )
    N = len(full_dataset)
    print(f"  Total entries: {N}")

    # 90/10 train/val split (deterministic)
    rng = random.Random(42)
    indices = list(range(N))
    rng.shuffle(indices)
    split = int(0.9 * N)
    train_indices = indices[:split]
    val_indices = indices[split:]
    print(f"  Train: {len(train_indices)}  |  Val: {len(val_indices)}")

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=SpatialContrastiveDataset.collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=SpatialContrastiveDataset.collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # ---- Model ----
    model = ContrastiveAdapter(hidden_dim=1152).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    # ---- Optimiser & scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-2
    )
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(100, total_steps // 10)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    criterion = nn.BCEWithLogitsLoss()

    # ---- Training loop ----
    best_val_loss = float("inf")
    global_step = 0

    print(f"\nStarting training: {args.epochs} epochs, "
          f"{steps_per_epoch} steps/epoch, "
          f"{warmup_steps} warmup steps")
    print(f"  Device: {device}")
    print(f"  Learning rate: {args.lr}")
    print()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        valid_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue

            vision = batch["vision"].to(device)       # [B, 64, 1152]
            text = batch["text"].to(device)           # [B, 1152]
            mask = batch["target_mask"].to(device)    # [B, 64]

            optimizer.zero_grad()
            sim = model(vision, text)                 # [B, 64]
            loss = criterion(sim, mask)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            valid_batches += 1
            global_step += 1

            # Per-step diagnostics every 50 steps
            if global_step % 50 == 0:
                with torch.no_grad():
                    prob = torch.sigmoid(sim)
                    pos_mask = mask.bool()
                    neg_mask = ~pos_mask
                    pos_sim = prob[pos_mask].mean().item() if pos_mask.any() else float("nan")
                    neg_sim = prob[neg_mask].mean().item() if neg_mask.any() else float("nan")

                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"  step {global_step:5d} | epoch {epoch:2d}/{args.epochs} "
                    f"| loss {loss.item():.4f} "
                    f"| pos_sim {pos_sim:.3f} "
                    f"| neg_sim {neg_sim:.3f} "
                    f"| lr {lr_now:.2e}"
                )

        avg_loss = epoch_loss / max(1, valid_batches)

        # Validation
        metrics = evaluate(model, val_loader, criterion, device)
        model.train()

        print(
            f"\nEpoch {epoch:2d}/{args.epochs} — "
            f"train_loss {avg_loss:.4f} | "
            f"val_loss {metrics['val_loss']:.4f} | "
            f"val_pos_sim {metrics['val_pos_sim']:.3f} | "
            f"val_neg_sim {metrics['val_neg_sim']:.3f} | "
            f"val_gap {metrics['val_gap']:.3f}"
        )

        # Checkpoint
        if metrics["val_loss"] < best_val_loss:
            best_val_loss = metrics["val_loss"]
            ckpt_path = out_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "val_gap": metrics["val_gap"],
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"  ✓ New best checkpoint saved → {ckpt_path} (val_loss={best_val_loss:.4f})")

        print()

    print(f"Training complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {out_dir / 'best.pt'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[3]   # …/OCR-Coder
    p = argparse.ArgumentParser(
        description="Phase 1.8 Spatial Contrastive Adapter training"
    )
    p.add_argument(
        "--feat-dir",
        type=Path,
        default=repo / "MVV/Phase_1_5/data/features/method2/pool8x8",
    )
    p.add_argument(
        "--ground-truth",
        type=Path,
        default=repo / "MVV/Phase_1_8/data/ground_truth/ground_truth.jsonl",
    )
    p.add_argument(
        "--text-emb",
        type=Path,
        default=repo / "MVV/Phase_1_8/data/text_embeddings/text_embeddings.pt",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=repo / "MVV/Phase_1_8/checkpoints",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
