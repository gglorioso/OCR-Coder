#!/usr/bin/env python3
"""
run_regression.py — Spatial regression probe for MVV Phase 1.2 Exp2

Uses adaptive max-pool features (4×4 and 8×8) instead of mean-pooled features,
preserving the spatial arrangement of patch tokens. Handles the curse of
dimensionality (18k–74k dims, ~9k samples) with:

  PCA (n_components, whiten=True, svd_solver='randomized')
    — reduces to a safe dimensionality while preserving spatial variance
  Ridge (L2 regularization, alpha)
    — prevents the model from memorizing outlier spatial zones

Labels: windowed (only AST nodes visible in the 40-line image window).

Resolution-as-Test: train PCA+Ridge on budget_729, freeze, test on 441/256/121.

Output:
  results/regression_results.json  — R² per pool_size × target × budget
  results/degradation_curve.png    — R² vs token budget (2-panel: pool4x4, pool8x8)

Usage:
    python run_regression.py [--features_root PATH] [--labels PATH]
                             [--out_dir PATH] [--n_components INT] [--alpha FLOAT]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


BUDGETS      = [729, 441, 256, 121]
TRAIN_BUDGET = 729
TARGETS      = ["line_count", "n_defs", "n_classes"]
POOL_SIZES   = ["pool4x4", "pool8x8"]
SUCCESS_R2   = 0.8

TARGET_COLORS = {
    "line_count": "#2563eb",
    "n_defs":     "#16a34a",
    "n_classes":  "#dc2626",
}

_SCRIPT_DIR = Path(__file__).parent
_EXP_DIR    = _SCRIPT_DIR.parent
_MVV_DIR    = _EXP_DIR.parent.parent

DEFAULT_FEATURES_ROOT = _MVV_DIR / "Phase_1_1" / "exp2_maxpool_comparison" / "data" / "features_maxpool"
DEFAULT_LABELS        = _EXP_DIR / "data" / "labels.jsonl"
DEFAULT_OUT           = _EXP_DIR / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _px(budget: int) -> str:
    side = int(budget ** 0.5)
    return f"{side * 14}×{side * 14}"


def load_labels(path: Path) -> dict[str, dict]:
    labels = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            labels[row["stem"]] = {t: row[t] for t in TARGETS}
    return labels


def load_budget_features(features_root: Path, pool: str, budget: int) -> tuple[np.ndarray, list[str]]:
    d     = features_root / pool / f"budget_{budget}"
    paths = sorted(d.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No .pt files in {d}")

    vecs, names = [], []
    for p in paths:
        t = torch.load(p, map_location="cpu")
        if t.dtype == torch.float16:
            t = t.float()
        vecs.append(t.numpy())
        names.append(p.stem)

    return np.stack(vecs, axis=0), names


def align(X: np.ndarray, stems: list[str], labels: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Inner-join features with labels on stem, sort by stem for consistency."""
    valid = [(s, i) for i, s in enumerate(stems) if s in labels]
    valid.sort(key=lambda x: x[0])

    if not valid:
        raise RuntimeError("No overlap between feature stems and labels.")

    idxs       = [i for _, i in valid]
    kept_stems = [s for s, _ in valid]
    X_aligned  = X[idxs]
    Y_aligned  = np.array([[labels[s][t] for t in TARGETS] for s in kept_stems], dtype=np.float64)
    return X_aligned, Y_aligned, kept_stems


# ---------------------------------------------------------------------------
# Per-pool probe
# ---------------------------------------------------------------------------

def run_pool_probe(
    pool: str,
    features_root: Path,
    labels: dict,
    n_components: int,
    alpha: float,
) -> dict:
    print(f"\n{'='*60}")
    print(f"Pool: {pool}  (PCA n={n_components}, Ridge alpha={alpha})")
    print("="*60)

    # ── Load train features ────────────────────────────────────────────────────
    print(f"\nLoading {pool}/budget_{TRAIN_BUDGET} …")
    t0 = time.time()
    X_raw, train_stems = load_budget_features(features_root, pool, TRAIN_BUDGET)
    print(f"  {X_raw.shape[0]:,} samples × {X_raw.shape[1]:,} dims  ({time.time()-t0:.1f}s)")

    X_train, Y_train, train_kept = align(X_raw, train_stems, labels)
    N, D = X_train.shape
    n_dropped = len(train_stems) - N
    if n_dropped:
        print(f"  Dropped {n_dropped} stems with no label")
    print(f"  Training set: {N:,} samples")

    # ── Clamp n_components to valid range ─────────────────────────────────────
    max_k = min(N, D)
    k     = min(n_components, max_k)
    if k < n_components:
        print(f"  n_components clamped {n_components} → {k} (min(N,D)={max_k})")

    # ── PCA ───────────────────────────────────────────────────────────────────
    print(f"\nFitting PCA({k}, whiten=True, randomized) …")
    t0  = time.time()
    pca = PCA(n_components=k, whiten=True, svd_solver="randomized", random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  Done in {time.time()-t0:.1f}s  |  variance explained: {var_explained*100:.1f}%")
    print(f"  PCA output shape: {X_train_pca.shape}")

    # ── Ridge ─────────────────────────────────────────────────────────────────
    print(f"\nFitting Ridge(alpha={alpha}, multi-output) …")
    t0  = time.time()
    reg = Ridge(alpha=alpha, fit_intercept=True)
    reg.fit(X_train_pca, Y_train)
    print(f"  Done in {time.time()-t0:.2f}s")

    # ── Target statistics ──────────────────────────────────────────────────────
    print("\nTarget statistics (training window, windowed labels):")
    for j, t in enumerate(TARGETS):
        vals = Y_train[:, j]
        print(f"  {t:12s}: mean={vals.mean():.2f}  std={vals.std():.2f}  "
              f"min={int(vals.min())}  max={int(vals.max())}")

    # ── Evaluate across budgets ────────────────────────────────────────────────
    print(f"\n  {'Budget':>8}  {'Pixels':>10}  " + "  ".join(f"{t:>12}" for t in TARGETS))
    print("  " + "-" * (8 + 10 + 3 + 14 * len(TARGETS)))

    budget_results: dict[str, dict] = {}

    for budget in BUDGETS:
        t0 = time.time()
        X_raw_b, stems_b = load_budget_features(features_root, pool, budget)
        X_b, Y_b, _      = align(X_raw_b, stems_b, labels)
        X_b_pca           = pca.transform(X_b)

        Y_pred = reg.predict(X_b_pca)

        r2s = {}
        for j, target in enumerate(TARGETS):
            r2s[target] = round(float(r2_score(Y_b[:, j], Y_pred[:, j])), 6)

        budget_results[f"budget_{budget}"] = r2s

        tag  = " (TRAIN)" if budget == TRAIN_BUDGET else " (TEST) "
        vals = "  ".join(f"{r2s[t]:+.4f}" for t in TARGETS)
        print(f"  {tag} {budget:3d}  {_px(budget):>10}  {vals}  ({time.time()-t0:.1f}s)")

    # ── Hypothesis verdict ─────────────────────────────────────────────────────
    print(f"\nHypothesis check (R² ≥ {SUCCESS_R2} at budget_256):")
    for target in TARGETS:
        v      = budget_results["budget_256"][target]
        passed = v >= SUCCESS_R2
        print(f"  {target:12s}: R²={v:+.4f}  {'✓ PASS' if passed else '✗ FAIL'}")

    return {
        "pca_n_components":    k,
        "pca_variance_explained": round(float(var_explained), 6),
        "ridge_alpha":         alpha,
        "n_train":             N,
        "feature_dim":         D,
        "results":             budget_results,
        "target_stats": {
            t: {
                "mean": round(float(Y_train[:, j].mean()), 3),
                "std":  round(float(Y_train[:, j].std()),  3),
                "min":  int(Y_train[:, j].min()),
                "max":  int(Y_train[:, j].max()),
            }
            for j, t in enumerate(TARGETS)
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_regression(
    features_root: Path,
    labels_path: Path,
    out_dir: Path,
    n_components: int,
    alpha: float,
) -> None:
    print("=" * 60)
    print("MVV Phase 1.2 Exp2 — Spatial Regression (PCA + Ridge)")
    print("=" * 60)

    labels = load_labels(labels_path)
    print(f"Labels loaded: {len(labels):,} windowed files")

    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}
    for pool in POOL_SIZES:
        all_results[pool] = run_pool_probe(pool, features_root, labels, n_components, alpha)

    summary = {
        "probe":             "spatial_regression_pca_ridge",
        "train_budget":      TRAIN_BUDGET,
        "n_components":      n_components,
        "ridge_alpha":       alpha,
        "success_threshold": SUCCESS_R2,
        "targets":           TARGETS,
        "pool_sizes":        POOL_SIZES,
        "pools":             all_results,
    }
    out_path = out_dir / "regression_results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved → {out_path}")

    try:
        _plot(all_results, out_dir)
    except ImportError:
        print("matplotlib not available — skipping plot (pip install matplotlib to enable)")

    print("All done.")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot(all_results: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, pool in zip(axes, POOL_SIZES):
        pool_data = all_results[pool]["results"]
        k         = all_results[pool]["pca_n_components"]
        var       = all_results[pool]["pca_variance_explained"]

        for target in TARGETS:
            r2_vals = [pool_data[f"budget_{b}"][target] for b in BUDGETS]
            color   = TARGET_COLORS[target]
            ax.plot(BUDGETS, r2_vals, "o-", color=color, lw=2, markersize=7,
                    label=target.replace("_", " "))
            for b, v in zip(BUDGETS, r2_vals):
                ax.annotate(f"{v:.3f}", (b, v), textcoords="offset points",
                            xytext=(0, 9), ha="center", fontsize=7.5, color=color)

        ax.axhline(SUCCESS_R2, color="gray", lw=1.5, ls="--",
                   label=f"Success threshold (R²={SUCCESS_R2})")
        ax.axhline(0.0, color="black", lw=0.8, ls=":", alpha=0.5)

        # Shade unreadable zone
        ax.axvspan(121 - 20, 256 + 20, alpha=0.07, color="#f59e0b",
                   label="Unreadable (≤256 tokens)")

        # Train budget marker
        ax.axvspan(TRAIN_BUDGET - 25, TRAIN_BUDGET + 25, alpha=0.08,
                   color="#2563eb", label="Train budget")

        dim = {"pool4x4": 18432, "pool8x8": 73728}[pool]
        ax.set_title(
            f"{pool} ({dim:,}D → PCA {k} → Ridge)\n"
            f"Variance explained: {var*100:.1f}%",
            fontsize=10, fontweight="bold",
        )
        ax.set_xlabel("Token budget", fontsize=10)
        ax.set_ylabel("R²", fontsize=10)
        ax.set_xticks(BUDGETS)
        ax.set_xticklabels([f"{b}\n({_px(b)})" for b in BUDGETS], fontsize=8)
        ax.invert_xaxis()
        ax.set_ylim(-0.5, 1.1)
        ax.legend(fontsize=7.5, loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "MVV Phase 1.2 Exp2 — Spatial Regression (Windowed Labels)\n"
        "Resolution-as-Test: train=729, test=441/256/121",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    path = out_dir / "degradation_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {path}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MVV Phase 1.2 Exp2 spatial regression probe")
    parser.add_argument("--features_root", type=Path, default=DEFAULT_FEATURES_ROOT,
                        help="Root containing pool4x4/ and pool8x8/ subdirs")
    parser.add_argument("--labels",        type=Path, default=DEFAULT_LABELS,
                        help="Windowed labels.jsonl from gen_labels.py")
    parser.add_argument("--out_dir",       type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n_components",  type=int,  default=1024,
                        help="PCA components (default 1024)")
    parser.add_argument("--alpha",         type=float, default=100.0,
                        help="Ridge L2 regularization strength (default 100.0)")
    args = parser.parse_args()

    run_regression(args.features_root, args.labels, args.out_dir,
                   args.n_components, args.alpha)
