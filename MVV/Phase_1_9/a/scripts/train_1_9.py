"""
Phase 1.9 training script — ConvRoPE Keyword Probe.

Loss    : BCEWithLogitsLoss  per token × keyword
Metrics : val_loss, per-keyword F1, macro-F1  (key: want macro-F1 > 0.3)
Output  : checkpoints/best.pt  +  results/grid_visualization.png
"""

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset

import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset_1_9 import KeywordPatchDataset
from model import ConvRoPEKeywordDetector, KEYWORDS, VOCAB_SIZE


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        progress = float(step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    all_preds, all_labels = [], []

    for batch in loader:
        if batch is None:
            continue
        vision = batch["vision"].to(device)    # [B, 1024, 1152]
        labels = batch["labels"].to(device)    # [B, 256, 16]

        logits = model(vision)                 # [B, 256, 16]
        loss   = criterion(logits, labels)
        total_loss += loss.item()
        n_batches  += 1

        probs = torch.sigmoid(logits).cpu().numpy()   # [B, 256, 16]
        lbls  = labels.cpu().numpy()                  # [B, 256, 16]

        # Flatten B×256 tokens → single pool per sample
        all_preds.append(probs.reshape(-1, VOCAB_SIZE))
        all_labels.append(lbls.reshape(-1, VOCAB_SIZE))

    val_loss = total_loss / max(1, n_batches)

    preds_flat  = np.concatenate(all_preds,  axis=0)   # [N, 16]
    labels_flat = np.concatenate(all_labels, axis=0)   # [N, 16]
    preds_bin   = (preds_flat >= 0.5).astype(int)
    labels_int  = labels_flat.astype(int)

    per_kw_f1 = f1_score(labels_int, preds_bin, average=None, zero_division=0)
    macro_f1  = float(per_kw_f1.mean())

    return {
        "val_loss":   val_loss,
        "macro_f1":   macro_f1,
        "per_kw_f1":  per_kw_f1.tolist(),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def make_grid_plot(model, dataset, device, out_path: Path):
    """
    For the first valid val sample, plot 2×VOCAB_SIZE heatmaps:
      Row 0 = ground truth   (16×16 grid per keyword)
      Row 1 = predicted prob  (16×16 grid per keyword)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping visualization")
        return

    # Find first valid sample
    sample = None
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample is not None:
            break
    if sample is None:
        return

    model.eval()
    with torch.no_grad():
        vision = sample["vision"].unsqueeze(0).to(device)   # [1, 1024, 1152]
        logits = model(vision)                               # [1, 256, 16]
        probs  = torch.sigmoid(logits).squeeze(0).cpu()     # [256, 16]

    gt = sample["labels"]   # [256, 16]

    fig, axes = plt.subplots(2, VOCAB_SIZE, figsize=(VOCAB_SIZE * 1.6, 3.5))
    for kw_i, kw in enumerate(KEYWORDS):
        gt_grid   = gt[:, kw_i].reshape(16, 16).numpy()
        pred_grid = probs[:, kw_i].reshape(16, 16).numpy()

        ax_gt   = axes[0, kw_i]
        ax_pred = axes[1, kw_i]

        ax_gt.imshow(gt_grid,   vmin=0, vmax=1, cmap="Blues", aspect="equal")
        ax_pred.imshow(pred_grid, vmin=0, vmax=1, cmap="Reds", aspect="equal")

        ax_gt.set_title(kw, fontsize=7)
        ax_pred.set_title(kw, fontsize=7)

        for ax in (ax_gt, ax_pred):
            ax.set_xticks([])
            ax.set_yticks([])

    axes[0, 0].set_ylabel("Ground Truth", fontsize=8)
    axes[1, 0].set_ylabel("Predicted",    fontsize=8)

    plt.suptitle("Phase 1.9 — Keyword Probe: Original vs Decoded Grid", fontsize=10)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Visualization saved → {out_path}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device  = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset …")
    full_ds = KeywordPatchDataset(
        ground_truth_jsonl = args.ground_truth,
        feat_dir           = args.feat_dir,
        labels_dir         = args.labels_dir,
    )
    N = len(full_ds)
    print(f"  Total entries: {N:,}")

    rng = random.Random(42)
    indices = list(range(N))
    rng.shuffle(indices)
    split = int(0.9 * N)
    train_ds = Subset(full_ds, indices[:split])
    val_ds   = Subset(full_ds, indices[split:])
    print(f"  Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, collate_fn=KeywordPatchDataset.collate_fn,
                              pin_memory=(device.type == "cuda"), drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=2, collate_fn=KeywordPatchDataset.collate_fn,
                              pin_memory=(device.type == "cuda"))

    model = ConvRoPEKeywordDetector().to(device)
    print(f"  Model params: {model.n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * args.epochs
    warmup_steps    = min(100, total_steps // 10)
    scheduler       = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion       = nn.BCEWithLogitsLoss()

    best_macro_f1 = -1.0
    global_step   = 0

    print(f"\nTraining: {args.epochs} epochs, {steps_per_epoch} steps/epoch, "
          f"lr={args.lr}, warmup={warmup_steps}")
    print(f"  Device: {device}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        valid_batches = 0

        for batch in train_loader:
            if batch is None:
                continue
            vision = batch["vision"].to(device)   # [B, 1024, 1152]
            labels = batch["labels"].to(device)   # [B, 256, 16]

            optimizer.zero_grad()
            logits = model(vision)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss    += loss.item()
            valid_batches += 1
            global_step   += 1

            if global_step % 50 == 0:
                with torch.no_grad():
                    probs    = torch.sigmoid(logits)
                    pos_mask = labels.bool()
                    pos_prob = probs[pos_mask].mean().item() if pos_mask.any() else float("nan")
                    neg_prob = probs[~pos_mask].mean().item() if (~pos_mask).any() else float("nan")
                lr_now = scheduler.get_last_lr()[0]
                print(f"  step {global_step:5d} | ep {epoch:2d}/{args.epochs} "
                      f"| loss {loss.item():.4f} "
                      f"| pos_prob {pos_prob:.3f} | neg_prob {neg_prob:.3f} "
                      f"| lr {lr_now:.2e}")

        avg_loss = epoch_loss / max(1, valid_batches)
        metrics  = evaluate(model, val_loader, criterion, device)
        model.train()

        print(f"\nEpoch {epoch:2d}/{args.epochs} — "
              f"train_loss {avg_loss:.4f} | "
              f"val_loss {metrics['val_loss']:.4f} | "
              f"macro_F1 {metrics['macro_f1']:.4f}")

        # Per-keyword F1 table
        print("  Per-keyword F1:")
        for kw, f1 in zip(KEYWORDS, metrics["per_kw_f1"]):
            print(f"    {kw:<10s} {f1:.4f}")

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            ckpt = out_dir / "best.pt"
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "macro_f1":         best_macro_f1,
                "per_kw_f1":        metrics["per_kw_f1"],
                "args":             vars(args),
            }, ckpt)
            print(f"  New best checkpoint (macro_F1={best_macro_f1:.4f}) → {ckpt}")

        print()

    print(f"Training complete. Best macro_F1: {best_macro_f1:.4f}")

    # Visualization
    print("Generating grid visualization …")
    vis_path = Path(args.out_dir).parent / "results" / "grid_visualization.png"
    make_grid_plot(model, full_ds, device, vis_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    repo = Path(__file__).resolve().parents[2]
    p    = argparse.ArgumentParser(description="Phase 1.9 keyword probe training")
    p.add_argument("--feat-dir",      type=Path,
                   default=repo / "MVV/Phase_1_9/data/features")
    p.add_argument("--labels-dir",    type=Path,
                   default=repo / "MVV/Phase_1_9/data/labels")
    p.add_argument("--ground-truth",  type=Path,
                   default=repo / "MVV/Phase_1_9/data/ground_truth.jsonl")
    p.add_argument("--out-dir",       type=Path,
                   default=repo / "MVV/Phase_1_9/checkpoints")
    p.add_argument("--epochs",        type=int,   default=20)
    p.add_argument("--batch-size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--device",        type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
