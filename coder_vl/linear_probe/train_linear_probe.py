"""
Train linear classifiers on visual features to test semantic content.

Tests whether vision encoder features contain code-semantic information
by training simple linear probes on binary, multi-class, and regression tasks.

Usage:
    python coder_vl/linear_probe/train_linear_probe.py --encoder ocr2
    python coder_vl/linear_probe/train_linear_probe.py --encoder siglip
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

PROBE_DATA_DIR = Path(__file__).resolve().parent / "probe_data"

BINARY_TASKS = [
    "has_class",
    "has_function",
    "has_imports",
    "has_many_functions",
    "is_large_file",
]

MULTICLASS_TASKS = {
    "file_size_bucket": 3,       # 0=small(<200), 1=medium(200-800), 2=large(>800)
    "function_count_bucket": 4,  # 0=none, 1=1-5, 2=6-15, 3=16+
}

REGRESSION_TASKS = ["num_functions", "num_classes", "num_imports"]


def load_data(encoder_name, split):
    """Load features and labels for a given encoder and split."""
    data_dir = PROBE_DATA_DIR / encoder_name
    features = torch.load(
        data_dir / f"features_{split}.pt", map_location="cpu", weights_only=True
    )
    labels = []
    with open(data_dir / f"labels_{split}.jsonl") as f:
        for line in f:
            labels.append(json.loads(line))
    return features.float(), labels


def train_binary_probe(train_X, train_labels, val_X, val_labels, task, lr=0.01, steps=500):
    """Train a binary linear probe and evaluate."""
    probe = nn.Linear(train_X.shape[1], 1)
    y_train = torch.tensor([float(l[task]) for l in train_labels])
    y_val = torch.tensor([float(l[task]) for l in val_labels])

    # Skip degenerate cases
    pos_rate = y_train.mean().item()
    if pos_rate == 0.0 or pos_rate == 1.0:
        return {
            "accuracy": max(pos_rate, 1 - pos_rate),
            "baseline": max(pos_rate, 1 - pos_rate),
            "f1": 0.0,
            "above_baseline": 0.0,
            "note": "degenerate",
        }

    optimizer = optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    probe.train()
    for _ in range(steps):
        logits = probe(train_X).squeeze(-1)
        loss = loss_fn(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    probe.eval()
    with torch.no_grad():
        preds = (probe(val_X).squeeze(-1) > 0).float()
        acc = (preds == y_val).float().mean().item()

        tp = ((preds == 1) & (y_val == 1)).sum().item()
        fp = ((preds == 1) & (y_val == 0)).sum().item()
        fn = ((preds == 0) & (y_val == 1)).sum().item()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        baseline = max(y_val.mean().item(), 1 - y_val.mean().item())

    return {
        "accuracy": round(acc, 4),
        "baseline": round(baseline, 4),
        "f1": round(f1, 4),
        "above_baseline": round(acc - baseline, 4),
    }


def train_multiclass_probe(train_X, train_labels, val_X, val_labels,
                           task, num_classes, lr=0.01, steps=500):
    """Train a multi-class linear probe and evaluate."""
    probe = nn.Linear(train_X.shape[1], num_classes)
    y_train = torch.tensor([l[task] for l in train_labels], dtype=torch.long)
    y_val = torch.tensor([l[task] for l in val_labels], dtype=torch.long)

    optimizer = optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    probe.train()
    for _ in range(steps):
        logits = probe(train_X)
        loss = loss_fn(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    probe.eval()
    with torch.no_grad():
        preds = probe(val_X).argmax(dim=-1)
        acc = (preds == y_val).float().mean().item()
        class_counts = torch.bincount(y_val, minlength=num_classes)
        baseline = class_counts.max().item() / len(y_val)

    return {
        "accuracy": round(acc, 4),
        "baseline": round(baseline, 4),
        "above_baseline": round(acc - baseline, 4),
    }


def train_regression_probe(train_X, train_labels, val_X, val_labels,
                           task, lr=0.01, steps=500):
    """Train a regression linear probe and evaluate with R^2."""
    probe = nn.Linear(train_X.shape[1], 1)
    y_train = torch.tensor([float(l[task]) for l in train_labels])
    y_val = torch.tensor([float(l[task]) for l in val_labels])

    # Normalize targets for stable training
    y_mean, y_std = y_train.mean(), y_train.std() + 1e-8
    y_train_n = (y_train - y_mean) / y_std
    y_val_n = (y_val - y_mean) / y_std

    optimizer = optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    probe.train()
    for _ in range(steps):
        preds = probe(train_X).squeeze(-1)
        loss = loss_fn(preds, y_train_n)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    probe.eval()
    with torch.no_grad():
        val_preds = probe(val_X).squeeze(-1)
        mse = loss_fn(val_preds, y_val_n).item()
        ss_res = ((val_preds - y_val_n) ** 2).sum().item()
        ss_tot = ((y_val_n - y_val_n.mean()) ** 2).sum().item()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {
        "mse": round(mse, 4),
        "r2": round(r2, 4),
        "mean_target": round(y_mean.item(), 2),
        "std_target": round(y_std.item(), 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=["ocr2", "siglip"], default="ocr2")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Linear Probe Test: {args.encoder.upper()}")
    print(f"  LR={args.lr}, Steps={args.steps}")
    print(f"{'='*60}")

    train_X, train_labels = load_data(args.encoder, "train")
    val_X, val_labels = load_data(args.encoder, "val")
    print(f"  Train: {train_X.shape}  Val: {val_X.shape}")

    results = {}

    # --- Binary classification ---
    print(f"\n--- Binary Classification ---")
    print(f"  {'Task':<25} {'Acc':>7} {'Base':>7} {'Delta':>7} {'F1':>7}")
    print(f"  {'-'*55}")
    for task in BINARY_TASKS:
        r = train_binary_probe(
            train_X, train_labels, val_X, val_labels,
            task, lr=args.lr, steps=args.steps,
        )
        results[task] = r
        delta = r["above_baseline"]
        icon = "pass" if delta > 0.05 else "weak" if delta > 0 else "FAIL"
        print(f"  {task:<25} {r['accuracy']:.3f}   {r['baseline']:.3f}   {delta:+.3f}   {r['f1']:.3f}  [{icon}]")

    # --- Multi-class classification ---
    print(f"\n--- Multi-class Classification ---")
    print(f"  {'Task':<25} {'Acc':>7} {'Base':>7} {'Delta':>7}")
    print(f"  {'-'*48}")
    for task, nc in MULTICLASS_TASKS.items():
        r = train_multiclass_probe(
            train_X, train_labels, val_X, val_labels,
            task, nc, lr=args.lr, steps=args.steps,
        )
        results[task] = r
        delta = r["above_baseline"]
        icon = "pass" if delta > 0.05 else "weak" if delta > 0 else "FAIL"
        print(f"  {task:<25} {r['accuracy']:.3f}   {r['baseline']:.3f}   {delta:+.3f}  [{icon}]")

    # --- Regression ---
    print(f"\n--- Regression ---")
    print(f"  {'Task':<25} {'R^2':>7} {'MSE':>7}")
    print(f"  {'-'*40}")
    for task in REGRESSION_TASKS:
        r = train_regression_probe(
            train_X, train_labels, val_X, val_labels,
            task, lr=args.lr, steps=args.steps,
        )
        results[task] = r
        icon = "pass" if r["r2"] > 0.1 else "weak" if r["r2"] > 0 else "FAIL"
        print(f"  {task:<25} {r['r2']:.3f}   {r['mse']:.4f}  [{icon}]")

    # --- Summary ---
    binary_deltas = [results[t]["above_baseline"] for t in BINARY_TASKS]
    avg_delta = sum(binary_deltas) / len(binary_deltas)
    regression_r2s = [results[t]["r2"] for t in REGRESSION_TASKS]
    avg_r2 = sum(regression_r2s) / len(regression_r2s)

    print(f"\n{'='*60}")
    print(f"  SUMMARY ({args.encoder.upper()})")
    print(f"{'='*60}")
    print(f"  Avg binary above-baseline:  {avg_delta:+.3f}")
    print(f"  Avg regression R^2:         {avg_r2:.3f}")

    if avg_delta > 0.10:
        print(f"\n  VERDICT: STRONG semantic content in visual features")
        print(f"  -> Alignment training (direct or contrastive) is feasible")
    elif avg_delta > 0.03:
        print(f"\n  VERDICT: WEAK semantic content in visual features")
        print(f"  -> Alignment may work but needs stronger adapter")
    else:
        print(f"\n  VERDICT: NO semantic content detected")
        print(f"  -> Visual features lack code semantics; need different encoder")
    print(f"{'='*60}")

    # Save results
    output_dir = PROBE_DATA_DIR / args.encoder
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "probe_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {output_dir / 'probe_results.json'}")


if __name__ == "__main__":
    main()
