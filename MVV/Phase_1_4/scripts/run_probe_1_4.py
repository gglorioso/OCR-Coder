#!/usr/bin/env python3
"""
run_probe_1_4.py — Syntax-style encoding probe for MVV Phase 1.4

Probes whether SigLIP features (mean-pooled, budget_256) encode low-level
syntax properties of code windows.  Uses native 5-fold CV entirely within
the 256-token mean-pool feature space.

Uses the clean MVV images (800x800, no distortion) from Phase 1.1.

Three probes:
  nesting_depth   (int 0/1/2)  — Ridge classifier, accuracy + per-class F1
  is_tabs         (int 0/1)    — Ridge classifier, accuracy + F1
  keyword_density (int)        — Ridge regression, R²

Why RidgeClassifier / Ridge regression:
  The features have already passed through ~26 SigLIP ViT transformer blocks +
  a VL2 MLP projector.  A linear probe is the most interpretable way to test
  whether a property is *linearly decodable* from the representation.
  A high R² / accuracy indicates the encoder preserves the property; a low
  score indicates it is not linearly encoded at 256-token resolution.

Feature loading:
  Each .pt file in Phase_1_1/data_mvv/features/budget_256/ has shape [1152] (fp16).
  These are already mean-pooled — loaded directly, no further pooling needed.

Output:
  MVV/Phase_1_4/results/probe_results.json

Usage:
    python run_probe_1_4.py [--labels PATH] [--features_dir PATH]
                            [--out_dir PATH] [--alpha FLOAT] [--n_folds INT]
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.model_selection import StratifiedKFold, KFold


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent
_PHASE_DIR  = _SCRIPT_DIR.parent
_REPO_ROOT  = _PHASE_DIR.parent.parent

DEFAULT_LABELS       = _PHASE_DIR / "data" / "labels_1_4.jsonl"
DEFAULT_FEATURES_DIR = _REPO_ROOT / "MVV" / "Phase_1_1" / "data_mvv" / "features" / "budget_256"
DEFAULT_OUT_DIR      = _PHASE_DIR / "results"

# Classification targets
CLF_TARGETS = ["nesting_depth", "is_tabs"]
# Regression targets
REG_TARGETS = ["keyword_density"]

FEATURE_DIM = 1152   # SigLIP-SO400M mean-pool output (budget_256)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_labels(path: Path) -> list[dict]:
    """Load Phase 1.4 labels.jsonl.  Each row: stem, nesting_depth, is_tabs, keyword_density."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_features(features_dir: Path, stems: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    Load clean MVV features for the given stems.

    Each .pt file has shape [1152] fp16 (already mean-pooled at budget_256).

    Returns:
      X     : np.ndarray of shape [N_found, 1152], float32
      found : list of stems for which a feature file existed
    """
    vecs  = []
    found = []
    missing = 0

    for stem in stems:
        pt_path = features_dir / f"{stem}.pt"
        if not pt_path.exists():
            missing += 1
            continue
        t = torch.load(pt_path, map_location="cpu")
        if t.dtype == torch.float16:
            t = t.float()
        # Shape: [1152] — already mean-pooled, load directly
        vec = t.numpy()
        vecs.append(vec)
        found.append(stem)

    if missing:
        print(f"  Feature files missing: {missing} / {len(stems)}")

    if not found:
        raise FileNotFoundError(
            f"No feature files found in {features_dir}. "
            "Check that the stems in labels match the .pt filenames."
        )

    return np.stack(vecs, axis=0), found


def align(
    labels: list[dict],
    X: np.ndarray,
    found_stems: list[str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Align X rows (indexed by found_stems) with Y values from labels.
    Returns (X_aligned, Y_dict) where Y_dict maps target_name → 1-D array.
    """
    stem_to_idx = {s: i for i, s in enumerate(found_stems)}
    stem_to_label = {row["stem"]: row for row in labels}

    common_stems = sorted(
        s for s in found_stems if s in stem_to_label
    )
    if not common_stems:
        raise RuntimeError(
            "No overlap between feature stems and label stems. "
            "Check that gen_phase_1_4_labels.py was run and stems match."
        )

    X_aligned = np.array([X[stem_to_idx[s]] for s in common_stems], dtype=np.float32)
    Y_dict = {
        "nesting_depth":   np.array([stem_to_label[s]["nesting_depth"]   for s in common_stems]),
        "is_tabs":         np.array([stem_to_label[s]["is_tabs"]         for s in common_stems]),
        "keyword_density": np.array([stem_to_label[s]["keyword_density"] for s in common_stems],
                                    dtype=np.float64),
    }

    print(f"  Aligned {len(common_stems):,} samples "
          f"(dropped {len(found_stems) - len(common_stems)} features with no label, "
          f"{len(labels) - len(common_stems)} labels with no feature)")

    return X_aligned, Y_dict


# ---------------------------------------------------------------------------
# Probe runners
# ---------------------------------------------------------------------------

def run_classifier_probe(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    n_folds: int,
) -> dict:
    """
    5-fold stratified CV with RidgeClassifier.
    Reports accuracy and per-class / macro F1.
    """
    print(f"\n{'='*55}")
    print(f"Probe: {name} (Ridge Classifier, alpha={alpha})")
    print("="*55)

    classes = sorted(np.unique(y).tolist())
    print(f"  Class distribution: " +
          ", ".join(f"{c}:{int((y==c).sum())}" for c in classes))

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_acc   = []
    fold_f1    = []        # macro F1
    fold_f1_per: dict[int, list[float]] = {c: [] for c in classes}

    header = f"  {'Fold':>5}  {'Acc':>7}  {'MacroF1':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        clf = RidgeClassifier(alpha=alpha, fit_intercept=True)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)

        acc     = accuracy_score(y_te, y_pred)
        macro_f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
        per_f1   = f1_score(y_te, y_pred, labels=classes, average=None,
                            zero_division=0)

        fold_acc.append(acc)
        fold_f1.append(macro_f1)
        for c, f in zip(classes, per_f1):
            fold_f1_per[c].append(f)

        print(f"  {fold+1:5d}  {acc:7.4f}  {macro_f1:9.4f}")

    mean_acc  = float(np.mean(fold_acc))
    std_acc   = float(np.std(fold_acc))
    mean_f1   = float(np.mean(fold_f1))
    std_f1    = float(np.std(fold_f1))

    print(f"\n  Mean accuracy   : {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Mean macro F1   : {mean_f1:.4f} ± {std_f1:.4f}")

    per_class_summary = {}
    print("  Per-class F1:")
    for c in classes:
        m = float(np.mean(fold_f1_per[c]))
        s = float(np.std(fold_f1_per[c]))
        per_class_summary[str(c)] = {"mean": round(m, 6), "std": round(s, 6)}
        print(f"    class {c}: {m:.4f} ± {s:.4f}")

    return {
        "probe_type":    "RidgeClassifier",
        "alpha":         alpha,
        "n_folds":       n_folds,
        "n_samples":     len(y),
        "classes":       classes,
        "accuracy":      {"mean": round(mean_acc, 6), "std": round(std_acc, 6)},
        "macro_f1":      {"mean": round(mean_f1, 6),  "std": round(std_f1,  6)},
        "per_class_f1":  per_class_summary,
        "fold_acc":      [round(v, 6) for v in fold_acc],
        "fold_macro_f1": [round(v, 6) for v in fold_f1],
    }


def run_regression_probe(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    n_folds: int,
) -> dict:
    """
    5-fold CV with Ridge regression.
    Reports R².
    """
    print(f"\n{'='*55}")
    print(f"Probe: {name} (Ridge Regression, alpha={alpha})")
    print("="*55)
    print(f"  y stats: mean={y.mean():.2f}  std={y.std():.2f}  "
          f"min={int(y.min())}  max={int(y.max())}")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_r2 = []
    print(f"  {'Fold':>5}  {'R²':>9}")
    print("  " + "-" * 18)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        reg = Ridge(alpha=alpha, fit_intercept=True)
        reg.fit(X_tr, y_tr)
        y_pred = reg.predict(X_te)

        r2 = r2_score(y_te, y_pred)
        fold_r2.append(r2)
        print(f"  {fold+1:5d}  {r2:+.6f}")

    mean_r2 = float(np.mean(fold_r2))
    std_r2  = float(np.std(fold_r2))

    print(f"\n  Mean R²: {mean_r2:+.4f} ± {std_r2:.4f}")

    return {
        "probe_type": "Ridge",
        "alpha":      alpha,
        "n_folds":    n_folds,
        "n_samples":  len(y),
        "r2":         {"mean": round(mean_r2, 6), "std": round(std_r2, 6)},
        "fold_r2":    [round(v, 6) for v in fold_r2],
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: dict) -> None:
    """Print a compact summary table of all probe results."""
    print()
    print("=" * 65)
    print("SUMMARY TABLE — MVV Phase 1.4 Syntax Probe")
    print("=" * 65)
    print(f"  {'Target':<20}  {'Type':<10}  {'Metric':<12}  {'Mean':>8}  {'Std':>7}")
    print("  " + "-" * 63)

    for name, res in results.items():
        ptype = res.get("probe_type", "?")
        if ptype == "RidgeClassifier":
            acc = res["accuracy"]
            f1  = res["macro_f1"]
            print(f"  {name:<20}  {ptype:<10}  {'Accuracy':<12}  {acc['mean']:>8.4f}  {acc['std']:>7.4f}")
            print(f"  {'':20}  {'':10}  {'Macro-F1':<12}  {f1['mean']:>8.4f}  {f1['std']:>7.4f}")
        elif ptype == "Ridge":
            r2 = res["r2"]
            print(f"  {name:<20}  {ptype:<10}  {'R²':<12}  {r2['mean']:>+8.4f}  {r2['std']:>7.4f}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MVV Phase 1.4 syntax-style encoding probe."
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS,
        help=f"Path to Phase 1.4 labels.jsonl (default: {DEFAULT_LABELS})",
    )
    parser.add_argument(
        "--features_dir",
        type=Path,
        default=DEFAULT_FEATURES_DIR,
        help=f"Directory of precomputed .pt feature files (default: {DEFAULT_FEATURES_DIR})",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for results (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=100.0,
        help="Ridge / RidgeClassifier L2 regularisation strength (default: 100.0)",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )
    args = parser.parse_args()

    # Validate paths
    if not args.labels.exists():
        raise FileNotFoundError(
            f"Labels file not found: {args.labels}\n"
            "Run gen_phase_1_4_labels.py first."
        )
    if not args.features_dir.exists():
        raise FileNotFoundError(f"Features directory not found: {args.features_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("MVV Phase 1.4 — Syntax-Style Encoding Probe")
    print(f"  Features : {args.features_dir}")
    print(f"  Labels   : {args.labels}")
    print(f"  Alpha    : {args.alpha}  |  Folds: {args.n_folds}")
    print("=" * 65)

    # ── Load labels ───────────────────────────────────────────────────────────
    t0     = time.time()
    labels = load_labels(args.labels)
    print(f"\nLoaded {len(labels):,} label rows  ({time.time()-t0:.1f}s)")

    # ── Load features (mean-pool over tiles) ──────────────────────────────────
    all_stems = [row["stem"] for row in labels]
    print(f"\nLoading features for {len(all_stems):,} stems from {args.features_dir} …")
    t0 = time.time()
    X_raw, found_stems = load_features(args.features_dir, all_stems)
    print(f"  Loaded {len(found_stems):,} feature vectors  ({time.time()-t0:.1f}s)  "
          f"shape={X_raw.shape}")

    # ── Align ─────────────────────────────────────────────────────────────────
    print("\nAligning features with labels …")
    X, Y_dict = align(labels, X_raw, found_stems)
    N, D = X.shape
    print(f"  Final dataset: {N:,} samples × {D:,} dims")

    # ── Run probes ────────────────────────────────────────────────────────────
    probe_results: dict[str, dict] = {}

    # Classification probes
    for target in CLF_TARGETS:
        probe_results[target] = run_classifier_probe(
            target, X, Y_dict[target], alpha=args.alpha, n_folds=args.n_folds
        )

    # Regression probe
    for target in REG_TARGETS:
        probe_results[target] = run_regression_probe(
            target, X, Y_dict[target], alpha=args.alpha, n_folds=args.n_folds
        )

    # ── Print summary ─────────────────────────────────────────────────────────
    print_summary(probe_results)

    # ── Save results ──────────────────────────────────────────────────────────
    summary = {
        "experiment":   "phase_1_4_syntax_probe",
        "feature_dim":  D,
        "n_samples":    N,
        "alpha":        args.alpha,
        "n_folds":      args.n_folds,
        "probes":       probe_results,
    }

    out_path = args.out_dir / "probe_results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Results saved → {out_path}")
    print("\nAll done.")


if __name__ == "__main__":
    main()
