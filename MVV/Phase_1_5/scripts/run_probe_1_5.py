#!/usr/bin/env python3
"""
run_probe_1_5.py — Ridge regression probe for MVV Phase 1.5

Compares three 256-token compression strategies (6 conditions total):

  Method 1 (Naive Downsampling, budget_256):
    224×224 → SigLIP → 16×16 grid → adaptive_max_pool2d → flatten
    Features already exist at Phase_1_1/exp2_maxpool_comparison/data/features_maxpool/

  Method 2 (Native + Spatial Pool):
    448×448 → SigLIP → 32×32 grid → avg_pool2d(2,2) → 16×16
    → adaptive_max_pool2d → flatten

  Method 3 (Token Pruning / Zero-Out):
    448×448 → SigLIP → 32×32 grid → zero-out 768 lowest-variance tokens
    → avg_pool2d(2,2) → 16×16 → adaptive_max_pool2d → flatten

For each of 6 conditions (3 methods × 2 pool sizes: 4×4, 8×8):
  5-fold CV: PCA(n_components=min(1024, n_train, n_features), whiten=True,
             svd_solver="randomized") → Ridge(alpha=100)
  Targets: line_count, n_defs, n_classes

Output:
  MVV/Phase_1_5/results/probe_results_1_5.json

Usage:
    python run_probe_1_5.py
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


# ---------------------------------------------------------------------------
# Paths  (derived from __file__ — no hardcoded absolute paths)
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT  = _SCRIPT_DIR.parents[2]   # OCR-Coder/

LABELS_PATH = (_REPO_ROOT / "MVV" / "Phase_1_2" /
               "exp2_spatial_regression" / "data" / "labels.jsonl")

# Method 1 features (already exist)
M1_FEAT_ROOT = (_REPO_ROOT / "MVV" / "Phase_1_1" /
                "exp2_maxpool_comparison" / "data" / "features_maxpool")

# Methods 2 & 3 features (extracted by extract_features_1_5.py)
M23_FEAT_ROOT = _REPO_ROOT / "MVV" / "Phase_1_5" / "data" / "features"

OUT_PATH = _REPO_ROOT / "MVV" / "Phase_1_5" / "results" / "probe_results_1_5.json"

# ---------------------------------------------------------------------------
# Probe configuration
# ---------------------------------------------------------------------------

TARGETS          = ["line_count", "n_defs", "n_classes"]
N_FOLDS          = 5
ALPHA            = 100.0
N_COMPONENTS_MAX = 1024

# Conditions: (label, feature_dir)
# Built dynamically below after path constants are defined.
POOL_SIZES = [4, 8]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_labels(path: Path) -> dict:
    labels = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            labels[row["stem"]] = {t: row[t] for t in TARGETS}
    return labels


def load_features(feat_dir: Path) -> tuple:
    """
    Load all .pt files from feat_dir.
    Returns (X [N, D] float64 ndarray, stems [N] list of str).
    """
    paths = sorted(feat_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No .pt files found in: {feat_dir}")

    vecs, stems = [], []
    for p in paths:
        t = torch.load(p, map_location="cpu")
        if t.dtype == torch.float16:
            t = t.float()
        vecs.append(t.numpy().ravel())
        stems.append(p.stem)

    return np.stack(vecs, axis=0), stems


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
    Y_al = np.array(
        [[labels[s][t] for t in TARGETS] for s in kept_stems], dtype=np.float64
    )
    return X_al, Y_al, kept_stems


# ---------------------------------------------------------------------------
# 5-fold CV probe (no data leakage: PCA fit inside fold on X_train only)
# ---------------------------------------------------------------------------

def run_cv(condition_label: str, feat_dir: Path, labels: dict) -> dict:
    t0 = time.time()
    print(f"\n  [{condition_label}] Loading features from: {feat_dir.relative_to(_REPO_ROOT)}")

    X_raw, stems = load_features(feat_dir)
    X, Y, _      = align(X_raw, stems, labels)
    N, D         = X.shape
    print(f"    {N:,} samples × {D:,} dims  ({time.time()-t0:.1f}s)")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_r2s = {t: [] for t in TARGETS}

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]

        n_tr = len(train_idx)
        k    = min(N_COMPONENTS_MAX, n_tr, D)

        # PCA fit on X_train ONLY — no leakage
        pca = PCA(n_components=k, whiten=True, svd_solver="randomized", random_state=42)
        X_tr_pca = pca.fit_transform(X_tr)
        X_te_pca = pca.transform(X_te)

        reg = Ridge(alpha=ALPHA, fit_intercept=True)
        reg.fit(X_tr_pca, Y_tr)
        Y_pred = reg.predict(X_te_pca)

        for j, target in enumerate(TARGETS):
            fold_r2s[target].append(float(r2_score(Y_te[:, j], Y_pred[:, j])))

        elapsed = time.time() - t0
        print(
            f"    fold {fold_idx+1}/{N_FOLDS}  (k={k})"
            + "  ".join(f"  {target}={fold_r2s[target][-1]:+.4f}" for target in TARGETS)
            + f"  [{elapsed:.1f}s]"
        )

    result = {"n_samples": N, "n_features": D}
    for target in TARGETS:
        folds = fold_r2s[target]
        result[target] = {
            "mean_r2": round(float(np.mean(folds)), 6),
            "std_r2":  round(float(np.std(folds)),  6),
            "fold_r2": [round(v, 6) for v in folds],
        }
    return result


# ---------------------------------------------------------------------------
# Build condition list
# ---------------------------------------------------------------------------

def build_conditions() -> list:
    """
    Returns list of (condition_label, feat_dir) tuples.
    Order: method1 (pool4x4, pool8x8), method2 (...), method3 (...)
    """
    conditions = []

    # Method 1 — budget_256 features from Phase 1.1
    for ps in POOL_SIZES:
        label    = f"method1_pool{ps}x{ps}"
        feat_dir = M1_FEAT_ROOT / f"pool{ps}x{ps}" / "budget_256"
        conditions.append((label, feat_dir))

    # Methods 2 & 3 — extracted by extract_features_1_5.py
    for method_idx in [2, 3]:
        for ps in POOL_SIZES:
            label    = f"method{method_idx}_pool{ps}x{ps}"
            feat_dir = M23_FEAT_ROOT / f"method{method_idx}" / f"pool{ps}x{ps}"
            conditions.append((label, feat_dir))

    return conditions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("MVV Phase 1.5 — 256-Token Compression Probe  (run_probe_1_5)")
    print("  3 methods × 2 pool sizes | 5-fold CV | PCA(whiten) + Ridge(alpha=100)")
    print(f"  Labels: {LABELS_PATH.relative_to(_REPO_ROOT)}")
    print("=" * 70)

    labels = load_labels(LABELS_PATH)
    print(f"\nLabels loaded: {len(labels):,} stems")

    conditions = build_conditions()
    print(f"Conditions to probe: {len(conditions)}")
    for label, feat_dir in conditions:
        exists = "OK" if feat_dir.exists() else "MISSING"
        n_files = len(list(feat_dir.glob("*.pt"))) if feat_dir.exists() else 0
        print(f"  {label:25s} {feat_dir.relative_to(_REPO_ROOT)}  [{exists}, {n_files} files]")

    all_results: dict = {}

    for condition_label, feat_dir in conditions:
        if not feat_dir.exists():
            print(f"\n  SKIPPING {condition_label} — directory not found: {feat_dir}")
            all_results[condition_label] = {"error": "feature directory not found"}
            continue

        n_files = len(list(feat_dir.glob("*.pt")))
        if n_files == 0:
            print(f"\n  SKIPPING {condition_label} — no .pt files in {feat_dir}")
            all_results[condition_label] = {"error": "no .pt files found"}
            continue

        all_results[condition_label] = run_cv(condition_label, feat_dir, labels)

    # ── Summary table ──────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("SUMMARY — Mean R² ± Std (5-fold CV)")
    print("=" * 70)
    header = f"{'Method':<20}  {'Pool':<8}  {'line_count R²':>16}  {'n_defs R²':>13}  {'n_classes R²':>14}"
    print(header)
    print("-" * 70)

    for condition_label, feat_dir in conditions:
        r = all_results.get(condition_label, {})
        if "error" in r:
            print(f"  {condition_label:<27}  ERROR: {r['error']}")
            continue

        # Parse method and pool from condition label, e.g. "method1_pool4x4"
        parts  = condition_label.split("_")
        method = parts[0]  # "method1"
        pool   = parts[1]  # "pool4x4"

        cols = []
        for target in TARGETS:
            tr = r.get(target, {})
            m  = tr.get("mean_r2", float("nan"))
            s  = tr.get("std_r2",  float("nan"))
            cols.append(f"{m:.3f}±{s:.3f}")

        print(f"  {method:<18}  {pool:<8}  {cols[0]:>16}  {cols[1]:>13}  {cols[2]:>14}")

    print("=" * 70)

    # ── Save JSON ──────────────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "description": (
            "Phase 1.5: 256-token compression comparison. "
            "Method1=naive_downsampling(budget_256), "
            "Method2=native_448px+avg_pool, "
            "Method3=token_pruning+avg_pool. "
            "5-fold CV, PCA(whiten=True)+Ridge(alpha=100), no data leakage."
        ),
        "targets":         TARGETS,
        "n_folds":         N_FOLDS,
        "alpha":           ALPHA,
        "n_components_max": N_COMPONENTS_MAX,
        "results":         all_results,
    }
    OUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved → {OUT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
