#!/usr/bin/env python3
"""
run_attention_probe_rope.py — Attention Probe for MVV Phase 1.6 (Experiment B)

Architecture: AttentionProbeRoPE
  - MLP adapter: vision space (1152D) → LLM space (2048D)
  - 2D SinCos positional encoding injected into spatial tokens (Y: dims 0-1023, X: dims 1024-2047)
  - Learnable CLS token (no positional encoding applied to CLS)
  - Standard MHA with 2D sinusoidal positional encodings (Experiment B)
  - Regression head: CLS output → scalar

Features: Method 2 pool8x8 from Phase 1.5
  - Each .pt file: [73728] fp16 = 64 tokens × 1152D
  - Reshaped to [64, 1152] before stacking

Target: n_defs only (single scalar regression)

Training: 5-fold CV, AdamW(lr=1e-4), MSE loss, 20 epochs per fold

Output:
  MVV/Phase_1_6/results/attention_probe_rope_results.json
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Paths  (derived from __file__ — no hardcoded absolute paths)
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT  = _SCRIPT_DIR.parents[2]   # OCR-Coder/

FEAT_DIR    = (_REPO_ROOT / "MVV" / "Phase_1_5" / "data" / "features" /
               "method2" / "pool8x8")

LABELS_PATH = (_REPO_ROOT / "MVV" / "Phase_1_2" /
               "exp2_spatial_regression" / "data" / "labels.jsonl")

OUT_PATH    = _REPO_ROOT / "MVV" / "Phase_1_6" / "results" / "attention_probe_rope_results.json"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

TARGET      = "n_defs"
N_FOLDS     = 5
LR          = 1e-4
EPOCHS      = 20
BATCH_SIZE  = 64
N_TOKENS    = 64
INPUT_DIM   = 1152
EMBED_DIM   = 2048
N_HEADS     = 16

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SinCos2DPositionalEncoding(nn.Module):
    """
    2D sinusoidal positional encoding for an H×W token grid.

    The 64 tokens represent an 8×8 physical grid in raster order:
      Token 0 → (x=0, y=0), Token 1 → (x=1, y=0), ..., Token 8 → (x=0, y=1), etc.

    Encoding splits embed_dim in half:
      First  embed_dim//2 dims encode the Y-axis (row)
      Last   embed_dim//2 dims encode the X-axis (col)

    Each half uses standard sinusoidal encoding over grid_h or grid_w positions.
    Result shape: [1, H*W, embed_dim] — broadcastable over batch.
    """
    def __init__(self, grid_h: int = 8, grid_w: int = 8, embed_dim: int = 2048):
        super().__init__()
        assert embed_dim % 2 == 0
        half = embed_dim // 2          # 1024 dims per axis

        # Build 1D sinusoidal encoding for each axis
        def make_1d_sincos(length, dim):
            pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)  # [L, 1]
            i   = torch.arange(dim // 2, dtype=torch.float32).unsqueeze(0)  # [1, dim//2]
            angle = pos / (10000.0 ** (2 * i / dim))  # [L, dim//2]
            enc = torch.zeros(length, dim)
            enc[:, 0::2] = torch.sin(angle)
            enc[:, 1::2] = torch.cos(angle[:, :dim//2 - (dim % 2)])  # handle odd dim
            return enc  # [L, dim]

        # Y encoding: shape [H, half] — one vector per row
        y_enc = make_1d_sincos(grid_h, half)   # [8, 1024]
        # X encoding: shape [W, half] — one vector per col
        x_enc = make_1d_sincos(grid_w, half)   # [8, 1024]

        # Expand to full 2D grid: each token gets [y_enc[row] | x_enc[col]]
        # Grid shape: [H, W, embed_dim]
        y_grid = y_enc.unsqueeze(1).expand(grid_h, grid_w, half)  # [8, 8, 1024]
        x_grid = x_enc.unsqueeze(0).expand(grid_h, grid_w, half)  # [8, 8, 1024]
        pe_2d  = torch.cat([y_grid, x_grid], dim=-1)               # [8, 8, 2048]

        # Flatten to [1, H*W, embed_dim] and register as buffer (not a parameter)
        pe_flat = pe_2d.reshape(grid_h * grid_w, embed_dim).unsqueeze(0)  # [1, 64, 2048]
        self.register_buffer("pe", pe_flat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N_spatial, embed_dim] — spatial tokens only (no CLS)."""
        return x + self.pe  # [B, 64, 2048]


class AttentionProbeRoPE(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter   = nn.Sequential(
            nn.Linear(1152, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
        )
        self.pos_enc   = SinCos2DPositionalEncoding(grid_h=8, grid_w=8, embed_dim=2048)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 2048))
        self.attention = nn.MultiheadAttention(embed_dim=2048, num_heads=16, batch_first=True)
        self.regressor = nn.Linear(2048, 1)

    def forward(self, x):
        # x: [B, 64, 1152]
        x = self.adapter(x)                            # [B, 64, 2048]
        x = self.pos_enc(x)                            # [B, 64, 2048] — 2D pos injected
        cls = self.cls_token.expand(x.size(0), -1, -1) # [B, 1, 2048]
        x = torch.cat([cls, x], dim=1)                 # [B, 65, 2048] — CLS first, no pos enc on CLS
        x, _ = self.attention(x, x, x)                 # [B, 65, 2048]
        cls_out = x[:, 0, :]                            # [B, 2048]
        return self.regressor(cls_out).squeeze(1)       # [B]


# ---------------------------------------------------------------------------
# Data loading helpers  (adapted from run_probe_1_5.py)
# ---------------------------------------------------------------------------

_TARGETS_FOR_LABELS = ["line_count", "n_defs", "n_classes"]


def load_labels(path: Path) -> dict:
    labels = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            labels[row["stem"]] = {t: row[t] for t in _TARGETS_FOR_LABELS}
    return labels


def load_features(feat_dir: Path) -> tuple:
    """
    Load all .pt files from feat_dir.
    Each file is [73728] fp16 = 64 tokens × 1152D.
    Returns (X [N, 64, 1152] float32 ndarray, stems [N] list of str).
    """
    paths = sorted(feat_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No .pt files found in: {feat_dir}")

    vecs, stems = [], []
    for p in paths:
        t = torch.load(p, map_location="cpu")
        if t.dtype == torch.float16:
            t = t.float()
        # Saved as [INPUT_DIM, N_TOKENS] (channels-first from flatten(1) on [1152,8,8]).
        # Transpose to [N_TOKENS, INPUT_DIM] so each row = one spatial token.
        vecs.append(t.reshape(INPUT_DIM, N_TOKENS).T.contiguous().numpy())
        stems.append(p.stem)

    return np.stack(vecs, axis=0), stems  # [N, 64, 1152]


def align(X: np.ndarray, stems: list, labels: dict) -> tuple:
    """Inner-join features with labels on stem, sorted by stem."""
    valid = [(s, i) for i, s in enumerate(stems) if s in labels]
    valid.sort(key=lambda x: x[0])
    if not valid:
        raise RuntimeError(
            "No overlap between feature stems and label stems.\n"
            f"  Feature stems sample: {stems[:3]}\n"
            f"  Label stems sample:   {list(labels.keys())[:3]}"
        )
    idxs       = [i for _, i in valid]
    kept_stems = [s for s, _ in valid]
    X_al = X[idxs]
    y_al = np.array(
        [labels[s][TARGET] for s in kept_stems], dtype=np.float32
    )
    return X_al, y_al, kept_stems


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("MVV Phase 1.6 — Attention Probe (Experiment B, 2D SinCos Positional Encoding)")
    print(f"  Features : {FEAT_DIR.relative_to(_REPO_ROOT)}")
    print(f"  Labels   : {LABELS_PATH.relative_to(_REPO_ROOT)}")
    print(f"  Target   : {TARGET}")
    print(f"  Folds    : {N_FOLDS}  |  Epochs: {EPOCHS}  |  LR: {LR}  |  Batch: {BATCH_SIZE}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load data
    labels = load_labels(LABELS_PATH)
    print(f"Labels loaded: {len(labels):,} stems")

    X_raw, stems = load_features(FEAT_DIR)
    print(f"Features loaded: {X_raw.shape[0]:,} samples, shape per sample: {X_raw.shape[1:]}")

    X, y, _ = align(X_raw, stems, labels)
    N = X.shape[0]
    print(f"Aligned: {N:,} samples")

    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)   # [N, 64, 1152]
    y_tensor = torch.tensor(y, dtype=torch.float32)   # [N]

    # 5-fold CV
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_r2s = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_tensor)):
        print(f"\n{'─'*60}")
        print(f"Fold {fold_idx + 1}/{N_FOLDS}  (train={len(train_idx)}, val={len(val_idx)})")

        # Build DataLoaders
        train_ds = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
        val_ds   = TensorDataset(X_tensor[val_idx],   y_tensor[val_idx])
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

        # Fresh model per fold
        model = AttentionProbeRoPE().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        criterion = nn.MSELoss()

        # Training
        model.train()
        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            n_batches  = 0
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = model(xb)
                loss  = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches  += 1
            avg_loss = epoch_loss / n_batches
            print(f"  Epoch {epoch + 1:2d}/{EPOCHS}  train_loss={avg_loss:.4f}")

        # Validation
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device)
                preds = model(xb).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(yb.numpy())

        all_preds   = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        fold_r2     = float(r2_score(all_targets, all_preds))
        fold_r2s.append(fold_r2)
        print(f"  Fold {fold_idx + 1} R² = {fold_r2:.4f}")

    # Summary
    mean_r2 = float(np.mean(fold_r2s))
    std_r2  = float(np.std(fold_r2s))

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for i, r2 in enumerate(fold_r2s):
        print(f"  Fold {i + 1}: R\u00b2 = {r2:.4f}")
    print(f"  Mean R\u00b2: {mean_r2:.4f} \u00b1 {std_r2:.4f}")
    print("=" * 70)

    # Save results
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "phase_1_6_attention_probe_experiment_B_2D_RoPE",
        "description": (
            "AttentionProbeRoPE (2D SinCos positional encoding) on method2_pool8x8 features, "
            "target=n_defs"
        ),
        "architecture": {
            "input_dim": INPUT_DIM,
            "n_tokens": N_TOKENS,
            "adapter": "Linear(1152,2048) -> GELU -> Linear(2048,2048)",
            "attention": "MHA(embed_dim=2048, num_heads=16, batch_first=True)",
            "positional_encoding": "2D SinCos (Y-axis: dims 0-1023, X-axis: dims 1024-2047)",
            "regressor": "Linear(2048, 1)",
        },
        "hyperparameters": {
            "lr": LR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "loss": "MSE",
        },
        "n_folds": N_FOLDS,
        "n_samples": N,
        "fold_r2": [round(r, 6) for r in fold_r2s],
        "mean_r2": round(mean_r2, 6),
        "std_r2":  round(std_r2,  6),
    }
    OUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved → {OUT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
