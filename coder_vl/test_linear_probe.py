"""
Test 3: Linear Probe on Visual Features

Tests whether the precomputed visual features (720 × 1280) contain enough
information to predict code structure — WITHOUT the LLM.

Two probes:
  A) Source-file classification:  Can features distinguish which Python file
     is being shown? (N-way classification over unique source files)
     High accuracy = features contain file-specific visual content.

  B) Has-class probe (binary):     Does this image contain class definitions?
     Tests coarse structural awareness.

Both use average-pooled features [720, 1280] → [1280] and a logistic
regression (+ optional MLP) with 80/20 train/test split.

No GPU required — runs on CPU.

Usage:
    python coder_vl/test_linear_probe.py
    python coder_vl/test_linear_probe.py --mlp   # also train 2-layer MLP
"""

import json
import argparse
import random
from pathlib import Path
from collections import Counter, defaultdict

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def load_feature(path: str) -> np.ndarray:
    """Load [N_tok, 1280] tensor, mean-pool → [1280] float32 numpy array."""
    feat = torch.load(path, map_location="cpu")   # [720, 1280]
    return feat.float().mean(dim=0).numpy()        # [1280]


# ---------------------------------------------------------------------------
# Probe A: source-file classification
# ---------------------------------------------------------------------------

def probe_source_file(examples, features_dir, min_samples=5, seed=42):
    """
    Predict which source_file an image comes from.
    Only keeps source files with >= min_samples images (to have enough train data).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import accuracy_score, top_k_accuracy_score

    print("=" * 60)
    print("Probe A: Source-file classification")
    print("=" * 60)

    # One feature per unique image (multiple tasks share same image)
    seen = {}
    for ex in examples:
        img = ex["image"]
        if img not in seen:
            seen[img] = ex["source_file"]

    # Count per source_file
    file_counts = Counter(seen.values())
    valid_files = {f for f, n in file_counts.items() if n >= min_samples}
    print(f"  Unique images       : {len(seen)}")
    print(f"  Source files total  : {len(file_counts)}")
    print(f"  Files with >={min_samples} imgs : {len(valid_files)}")

    # Build X, y
    X, y, paths = [], [], []
    features_dir = Path(features_dir)
    for img_path, src_file in seen.items():
        if src_file not in valid_files:
            continue
        feat_path = features_dir / (Path(img_path).stem + ".pt")
        if not feat_path.exists():
            continue
        X.append(load_feature(str(feat_path)))
        y.append(src_file)
        paths.append(str(feat_path))

    if len(X) < 10:
        print("  Not enough data — skipping.")
        return

    X = np.stack(X)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)
    print(f"  Dataset size        : {len(X)} images × {X.shape[1]} features")
    print(f"  Classes             : {n_classes}")

    # Train/test split (stratified where possible)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    split = int(0.8 * len(idx))
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_enc[train_idx], y_enc[test_idx]

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"  Train/test split    : {len(X_train)} / {len(X_test)}")
    print()

    # Logistic regression
    print("  Training LogReg (max_iter=2000, lbfgs) ...")
    lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", n_jobs=-1,
                            random_state=seed, multi_class="multinomial")
    lr.fit(X_train, y_train)

    pred = lr.predict(X_test)
    acc_top1 = accuracy_score(y_test, pred)
    print(f"  Top-1 accuracy      : {acc_top1:.4f} ({acc_top1*100:.1f}%)")

    if n_classes >= 5:
        proba = lr.predict_proba(X_test)
        # Filter test set to classes seen during training (avoids label count mismatch)
        seen_mask = np.isin(y_test, lr.classes_)
        n_unseen = (~seen_mask).sum()
        if n_unseen > 0:
            print(f"  (Dropping {n_unseen} test samples with unseen classes for Top-k)")
        acc_top5 = top_k_accuracy_score(y_test[seen_mask], proba[seen_mask],
                                        k=min(5, len(lr.classes_)),
                                        labels=lr.classes_)
        print(f"  Top-5 accuracy      : {acc_top5:.4f} ({acc_top5*100:.1f}%)")

    # Baseline (random)
    baseline = 1.0 / n_classes
    print(f"  Random baseline     : {baseline:.4f} ({baseline*100:.1f}%)")
    lift = acc_top1 / baseline if baseline > 0 else 0
    print(f"  Lift over random    : {lift:.1f}×")
    print()

    # Interpretation
    if acc_top1 > 0.5:
        print("  >> STRONG signal: features carry file-specific information.")
    elif acc_top1 > baseline * 3:
        print("  >> MODERATE signal: features partially discriminate source files.")
    else:
        print("  >> WEAK signal: features don't reliably distinguish source files.")

    return {"probe": "source_file", "n_classes": n_classes, "n_samples": len(X),
            "top1": round(acc_top1, 4), "baseline": round(baseline, 4),
            "lift": round(lift, 2)}


# ---------------------------------------------------------------------------
# Probe B: has-class binary classification
# ---------------------------------------------------------------------------

def probe_has_class(examples, features_dir, seed=42):
    """
    Binary: does this image contain at least one class definition?
    Uses class_listing examples where reference != empty as positive label.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

    print("=" * 60)
    print("Probe B: Has-class binary classification")
    print("=" * 60)

    features_dir = Path(features_dir)

    # Collect one label per unique image
    # Positive = image has >=1 class (from class_listing reference)
    # Negative = image has no classes
    img_label = {}
    for ex in examples:
        img = ex["image"]
        if img in img_label:
            continue
        if ex["task_type"] == "class_listing":
            conv = ex["conversations"]
            ref = conv[-1]["content"] if conv else ""
            # Positive if reference mentions class names
            has_class = bool(ref.strip()) and "1." in ref
            img_label[img] = 1 if has_class else 0

    if len(img_label) < 20:
        # Fall back: use all images, label from whether ANY class_listing exists
        img_has_class_listing = set()
        for ex in examples:
            if ex["task_type"] == "class_listing":
                img_has_class_listing.add(ex["image"])
        for ex in examples:
            if ex["image"] not in img_label:
                img_label[ex["image"]] = 1 if ex["image"] in img_has_class_listing else 0

    X, y = [], []
    for img_path, label in img_label.items():
        feat_path = features_dir / (Path(img_path).stem + ".pt")
        if not feat_path.exists():
            continue
        X.append(load_feature(str(feat_path)))
        y.append(label)

    X = np.stack(X)
    y = np.array(y)
    pos = y.sum()
    print(f"  Dataset size   : {len(X)} images × {X.shape[1]} features")
    print(f"  Positive (has class): {int(pos)} ({pos/len(y):.1%})")
    print(f"  Negative (no class) : {len(y)-int(pos)} ({(len(y)-pos)/len(y):.1%})")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    split = int(0.8 * len(idx))
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"  Train/test split: {len(X_train)} / {len(X_test)}")
    print()

    lr = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs", random_state=seed)
    lr.fit(X_train, y_train)

    pred  = lr.predict(X_test)
    proba = lr.predict_proba(X_test)[:, 1]
    acc   = accuracy_score(y_test, pred)

    try:
        auc = roc_auc_score(y_test, proba)
    except Exception:
        auc = float("nan")

    majority_baseline = max(y_train.mean(), 1 - y_train.mean())

    print(f"  Accuracy        : {acc:.4f} ({acc*100:.1f}%)")
    print(f"  ROC-AUC         : {auc:.4f}")
    print(f"  Majority baseline: {majority_baseline:.4f}")
    print()
    print(classification_report(y_test, pred, target_names=["no-class", "has-class"],
                                 zero_division=0))

    if auc > 0.75:
        print("  >> STRONG signal: features distinguish class vs no-class.")
    elif auc > 0.6:
        print("  >> MODERATE signal: some class structure visible in features.")
    else:
        print("  >> WEAK signal: features don't distinguish class structure.")

    return {"probe": "has_class", "n_samples": len(X), "accuracy": round(acc, 4),
            "roc_auc": round(auc, 4) if not np.isnan(auc) else None,
            "majority_baseline": round(majority_baseline, 4)}


# ---------------------------------------------------------------------------
# Optional: 2-layer MLP probe
# ---------------------------------------------------------------------------

def probe_mlp_source_file(examples, features_dir, min_samples=5, seed=42):
    """Same as probe_source_file but with a 2-layer MLP (sklearn MLPClassifier)."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import accuracy_score, top_k_accuracy_score

    print("=" * 60)
    print("Probe A (MLP): Source-file classification")
    print("=" * 60)

    seen = {}
    for ex in examples:
        img = ex["image"]
        if img not in seen:
            seen[img] = ex["source_file"]

    file_counts = Counter(seen.values())
    valid_files = {f for f, n in file_counts.items() if n >= min_samples}

    X, y = [], []
    features_dir_p = Path(features_dir)
    for img_path, src_file in seen.items():
        if src_file not in valid_files:
            continue
        feat_path = features_dir_p / (Path(img_path).stem + ".pt")
        if not feat_path.exists():
            continue
        X.append(load_feature(str(feat_path)))
        y.append(src_file)

    if len(X) < 10:
        print("  Not enough data — skipping.")
        return

    X = np.stack(X)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    split = int(0.8 * len(idx))
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_enc[train_idx], y_enc[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"  n={len(X)}, classes={n_classes}, train={len(X_train)}, test={len(X_test)}")
    print("  Training MLP (hidden=512,256, max_iter=200) ...")

    mlp = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=200,
                        random_state=seed, early_stopping=True, n_iter_no_change=10)
    mlp.fit(X_train, y_train)

    pred = mlp.predict(X_test)
    acc_top1 = accuracy_score(y_test, pred)
    print(f"  Top-1 accuracy : {acc_top1:.4f} ({acc_top1*100:.1f}%)")

    if n_classes >= 5:
        proba = mlp.predict_proba(X_test)
        seen_mask = np.isin(y_test, mlp.classes_)
        acc_top5 = top_k_accuracy_score(y_test[seen_mask], proba[seen_mask],
                                        k=min(5, len(mlp.classes_)),
                                        labels=mlp.classes_)
        print(f"  Top-5 accuracy : {acc_top5:.4f} ({acc_top5*100:.1f}%)")

    baseline = 1.0 / n_classes
    print(f"  Baseline (rand): {baseline:.4f}")
    print()

    return {"probe": "source_file_mlp", "n_classes": n_classes, "top1": round(acc_top1, 4)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", default="./precomputed_features_tiled")
    parser.add_argument("--val_manifest", default="data_v2b/manifests/val.jsonl")
    parser.add_argument("--min_samples",  type=int, default=3,
                        help="Min images per source file for Probe A (default 3)")
    parser.add_argument("--mlp",          action="store_true",
                        help="Also run MLP probe (slower)")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--save_file",    default="./probe_results.json")
    args = parser.parse_args()

    try:
        import sklearn
        print(f"sklearn version: {sklearn.__version__}")
    except ImportError:
        print("ERROR: scikit-learn not found.")
        print("Install with: pip install scikit-learn")
        raise

    # Load train + val manifests for maximum coverage
    examples = []
    manifests = [
        args.val_manifest,
        args.val_manifest.replace("val.jsonl", "train.jsonl"),
    ]
    for mf in manifests:
        p = Path(mf)
        if p.exists():
            count = 0
            with open(p) as f:
                for line in f:
                    examples.append(json.loads(line))
                    count += 1
            print(f"  Loaded {count} examples from {p.name}")
        else:
            print(f"  Skipping {p.name} (not found)")
    print(f"Total: {len(examples)} examples\n")

    all_results = {}

    r_a = probe_source_file(examples, args.features_dir,
                             min_samples=args.min_samples, seed=args.seed)
    if r_a:
        all_results["probe_a"] = r_a
    print()

    r_b = probe_has_class(examples, args.features_dir, seed=args.seed)
    if r_b:
        all_results["probe_b"] = r_b
    print()

    if args.mlp:
        r_mlp = probe_mlp_source_file(examples, args.features_dir,
                                       min_samples=args.min_samples, seed=args.seed)
        if r_mlp:
            all_results["probe_a_mlp"] = r_mlp
        print()

    # Save
    with open(args.save_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {args.save_file}")


if __name__ == "__main__":
    main()
