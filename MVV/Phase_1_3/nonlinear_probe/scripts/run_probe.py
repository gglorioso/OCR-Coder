#!/usr/bin/env python3
"""
run_probe.py — Nonlinear encoding probe for MVV Phase 1.3 (revised)

Two-mode design to isolate the information-capacity question from generalization:

  MODE A — Resolution-as-Test (inherited from Phase 1.2)
    Train PCA + Ridge on budget_729 features; evaluate on 441 / 256 / 121.
    Purpose: degradation curve for context, apples-to-apples with Exp2 baseline.
    Limitation: tests generalization across compression, not raw encoding capacity.

  MODE B — Native CV at 256 Tokens (new)
    5-fold cross-validation entirely within budget_256 features.
    Probes: Ridge (PCA+Ridge) and RandomForest (raw features, no PCA needed).
    Purpose: eliminates the domain-shift confound. If Ridge fails here, the
    visual signal for n_defs is genuinely absent at 256 tokens. If RF succeeds
    where Ridge fails, the signal exists but requires nonlinear partitioning.

Why RandomForest instead of MLP:
  - The pool4x4 features have already passed through ~26 transformer MLP blocks
    (SigLIP ViT) + VL2's own MlpProjector. An external MLP probe is redundant.
  - RF makes axis-aligned splits — orthogonal to the smooth nonlinearities
    already applied by the encoder. It is a genuinely different probe.
  - RF naturally handles discrete count targets (n_defs, n_classes) via step
    functions, unlike the smooth MLP approximation.
  - No gradient issues, no learning-rate tuning.

Visual footprint hypothesis:
  n_classes (R²=0.675) survives 256 tokens better than n_defs (R²=0.461).
  A class definition has a large visual footprint (header + body block, ~5–30
  lines). A def marker is a single indented line. Mode B results will indicate
  whether this gap persists when domain shift is eliminated.

Output:
  results/probe_results.json   — R² for both modes, all probes, all targets
  results/comparison_plot.png  — Panel A: degradation curve | Panel B: native CV bars

Usage:
    python run_probe.py [--features_root PATH] [--labels PATH] [--out_dir PATH]
                        [--n_components INT] [--alpha FLOAT]
                        [--n_estimators INT] [--n_folds INT]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


BUDGETS      = [729, 441, 256, 121]
TRAIN_BUDGET = 729
CV_BUDGET    = 256
TARGETS      = ["n_defs", "n_classes"]
POOL         = "pool4x4"
SUCCESS_R2   = 0.8

TARGET_COLORS = {
    "n_defs":    "#16a34a",
    "n_classes": "#dc2626",
}

_SCRIPT_DIR = Path(__file__).parent
_EXP_DIR    = _SCRIPT_DIR.parent
_MVV_DIR    = _EXP_DIR.parent.parent

DEFAULT_FEATURES_ROOT = _MVV_DIR / "Phase_1_1" / "exp2_maxpool_comparison" / "data" / "features_maxpool"
DEFAULT_LABELS        = _MVV_DIR / "Phase_1_2" / "exp2_spatial_regression" / "data" / "labels.jsonl"
DEFAULT_OUT           = _EXP_DIR / "results"


# ---------------------------------------------------------------------------
# Data helpers
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
# Mode A — Resolution-as-Test (Ridge baseline, degradation curve)
# ---------------------------------------------------------------------------

def run_mode_a(
    features_root: Path,
    labels: dict,
    n_components: int,
    alpha: float,
) -> dict:
    print(f"\n{'='*60}")
    print("MODE A — Resolution-as-Test (Ridge baseline, train@729)")
    print("="*60)

    print(f"\nLoading {POOL}/budget_{TRAIN_BUDGET} …")
    X_raw, train_stems = load_budget_features(features_root, POOL, TRAIN_BUDGET)
    X_train, Y_train, _ = align(X_raw, train_stems, labels)
    N, D = X_train.shape
    print(f"  {N:,} samples × {D:,} dims")

    k = min(n_components, N, D)
    print(f"\nFitting PCA({k}) …")
    t0  = time.time()
    pca = PCA(n_components=k, whiten=True, svd_solver="randomized", random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    var = pca.explained_variance_ratio_.sum()
    print(f"  Done {time.time()-t0:.1f}s | variance explained: {var*100:.1f}%")

    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(X_train_pca, Y_train)

    print(f"\n  {'Budget':>8}  {'Pixels':>10}  " + "  ".join(f"{t:>12}" for t in TARGETS))
    budget_results: dict[str, dict] = {}

    for budget in BUDGETS:
        X_raw_b, stems_b = load_budget_features(features_root, POOL, budget)
        X_b, Y_b, _      = align(X_raw_b, stems_b, labels)
        X_b_pca           = pca.transform(X_b)
        Y_pred            = ridge.predict(X_b_pca)

        r2s = {t: round(float(r2_score(Y_b[:, j], Y_pred[:, j])), 6)
               for j, t in enumerate(TARGETS)}
        budget_results[f"budget_{budget}"] = r2s

        tag  = "(TRAIN)" if budget == TRAIN_BUDGET else "(TEST) "
        vals = "  ".join(f"{r2s[t]:+.4f}" for t in TARGETS)
        print(f"  {tag} {budget:3d}  {_px(budget):>10}  {vals}")

    return {
        "pca_n_components":      k,
        "pca_variance_explained": round(float(var), 6),
        "ridge_alpha":           alpha,
        "n_train":               N,
        "feature_dim":           D,
        "results":               budget_results,
    }


# ---------------------------------------------------------------------------
# Mode B — Native CV at 256 tokens (Ridge + RandomForest)
# ---------------------------------------------------------------------------

def run_mode_b(
    features_root: Path,
    labels: dict,
    n_components: int,
    alpha: float,
    n_estimators: int,
    n_folds: int,
) -> dict:
    print(f"\n{'='*60}")
    print(f"MODE B — Native {n_folds}-Fold CV at budget_{CV_BUDGET}")
    print(f"  Ridge(PCA {n_components}, alpha={alpha})  |  "
          f"RandomForest({n_estimators} trees, raw features)")
    print("="*60)

    print(f"\nLoading {POOL}/budget_{CV_BUDGET} …")
    X_raw, stems = load_budget_features(features_root, POOL, CV_BUDGET)
    X, Y, _      = align(X_raw, stems, labels)
    N, D         = X.shape
    print(f"  {N:,} samples × {D:,} dims")

    print("\nTarget statistics at 256 tokens (windowed labels):")
    for j, t in enumerate(TARGETS):
        vals = Y[:, j]
        print(f"  {t:12s}: mean={vals.mean():.2f}  std={vals.std():.2f}  "
              f"min={int(vals.min())}  max={int(vals.max())}")

    kf           = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = {"ridge": [], "random_forest": []}   # list of {target: r2} per fold

    print(f"\n  {'Fold':>5}  " + "  ".join(
        f"{'Ridge '+t:>16}  {'RF '+t:>16}" for t in TARGETS))
    print("  " + "-" * (5 + (34 * len(TARGETS))))

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]

        # Ridge: PCA fit within fold (no data leakage)
        k   = min(n_components, X_tr.shape[0], X_tr.shape[1])
        pca = PCA(n_components=k, whiten=True, svd_solver="randomized", random_state=42)
        X_tr_pca = pca.fit_transform(X_tr)
        X_te_pca = pca.transform(X_te)

        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(X_tr_pca, Y_tr)
        Y_pred_ridge = ridge.predict(X_te_pca)

        # Random Forest: raw features, no PCA
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=-1,
            random_state=42 + fold,
        )
        rf.fit(X_tr, Y_tr)
        Y_pred_rf = rf.predict(X_te)

        r2_ridge = {t: round(float(r2_score(Y_te[:, j], Y_pred_ridge[:, j])), 6)
                    for j, t in enumerate(TARGETS)}
        r2_rf    = {t: round(float(r2_score(Y_te[:, j], Y_pred_rf[:, j])),    6)
                    for j, t in enumerate(TARGETS)}

        fold_results["ridge"].append(r2_ridge)
        fold_results["random_forest"].append(r2_rf)

        row = "  ".join(
            f"{r2_ridge[t]:+.4f}  {r2_rf[t]:+.4f}" for t in TARGETS
        )
        print(f"  {fold+1:5d}  {row}")

    # Aggregate across folds
    summary: dict[str, dict] = {}
    print(f"\n--- Summary (mean ± std over {n_folds} folds) ---")
    print(f"  {'Probe':>14}  " + "  ".join(f"{t:>14}" for t in TARGETS))
    for probe_name in ("ridge", "random_forest"):
        r2_mat = np.array([[fold_r2[t] for t in TARGETS]
                           for fold_r2 in fold_results[probe_name]])
        means  = r2_mat.mean(axis=0)
        stds   = r2_mat.std(axis=0)
        summary[probe_name] = {
            t: {"mean": round(float(means[j]), 6), "std": round(float(stds[j]), 6)}
            for j, t in enumerate(TARGETS)
        }
        vals = "  ".join(f"{means[j]:+.4f} ±{stds[j]:.4f}" for j in range(len(TARGETS)))
        print(f"  {probe_name:>14}  {vals}")

    print(f"\n--- Hypothesis check (R² ≥ {SUCCESS_R2}) ---")
    for probe_name in ("ridge", "random_forest"):
        for t in TARGETS:
            m = summary[probe_name][t]["mean"]
            passed = m >= SUCCESS_R2
            print(f"  {probe_name:>14} / {t:12s}: R²={m:+.4f}  "
                  f"{'✓ PASS' if passed else '✗ FAIL'}")

    print("\n--- Visual footprint gap (n_classes - n_defs) ---")
    for probe_name in ("ridge", "random_forest"):
        gap = (summary[probe_name]["n_classes"]["mean"]
               - summary[probe_name]["n_defs"]["mean"])
        print(f"  {probe_name:>14}: gap = {gap:+.4f}")

    return {
        "cv_budget":     CV_BUDGET,
        "n_folds":       n_folds,
        "n_samples":     N,
        "feature_dim":   D,
        "ridge_alpha":   alpha,
        "rf_n_estimators": n_estimators,
        "probes":        summary,
        "fold_details":  fold_results,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot(mode_a: dict, mode_b: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15, 5))

    # ── Panel A: degradation curve (Mode A, Ridge) ────────────────────────────
    for target in TARGETS:
        color  = TARGET_COLORS[target]
        r2_vals = [mode_a["results"][f"budget_{b}"][target] for b in BUDGETS]
        ax_a.plot(BUDGETS, r2_vals, "o--", color=color, lw=2, markersize=7,
                  label=target.replace("_", " "))
        for b, v in zip(BUDGETS, r2_vals):
            ax_a.annotate(f"{v:.3f}", (b, v), textcoords="offset points",
                          xytext=(0, 9), ha="center", fontsize=8, color=color)

    ax_a.axhline(SUCCESS_R2, color="gray", lw=1.5, ls="--",
                 label=f"Success R²={SUCCESS_R2}")
    ax_a.axhline(0, color="black", lw=0.8, ls=":", alpha=0.4)
    ax_a.axvspan(121 - 20, 256 + 20, alpha=0.07, color="#f59e0b",
                 label="Unreadable zone (≤256)")
    ax_a.set_title(
        "Mode A — Resolution-as-Test\n(Ridge, train@729, test@441/256/121)",
        fontsize=10, fontweight="bold",
    )
    ax_a.set_xlabel("Token budget", fontsize=10)
    ax_a.set_ylabel("R²", fontsize=10)
    ax_a.set_xticks(BUDGETS)
    ax_a.set_xticklabels([f"{b}\n({_px(b)})" for b in BUDGETS], fontsize=8)
    ax_a.invert_xaxis()
    ax_a.set_ylim(-0.3, 1.1)
    ax_a.legend(fontsize=8, loc="lower right")
    ax_a.grid(True, alpha=0.3)

    # ── Panel B: native CV bar chart (Mode B, Ridge vs RF) ────────────────────
    probe_names   = ["ridge", "random_forest"]
    probe_labels  = ["Ridge\n(PCA+linear)", "Random Forest\n(raw features)"]
    n_probes      = len(probe_names)
    n_targets     = len(TARGETS)
    bar_width     = 0.3
    group_gap     = 0.9
    x_positions   = np.arange(n_targets) * group_gap

    for pi, (pname, plabel) in enumerate(zip(probe_names, probe_labels)):
        offset = (pi - (n_probes - 1) / 2) * bar_width
        for ti, target in enumerate(TARGETS):
            mean = mode_b["probes"][pname][target]["mean"]
            std  = mode_b["probes"][pname][target]["std"]
            color = TARGET_COLORS[target]
            alpha_fill = 0.85 if pi == 0 else 0.45
            ax_b.bar(x_positions[ti] + offset, mean, bar_width,
                     color=color, alpha=alpha_fill,
                     label=f"{plabel}" if ti == 0 else "",
                     yerr=std, capsize=4, error_kw={"linewidth": 1.2})
            ax_b.text(x_positions[ti] + offset, mean + std + 0.02,
                      f"{mean:.3f}", ha="center", va="bottom",
                      fontsize=8, color=color)

    ax_b.axhline(SUCCESS_R2, color="gray", lw=1.5, ls="--",
                 label=f"Success R²={SUCCESS_R2}")
    ax_b.axhline(0, color="black", lw=0.8, ls=":", alpha=0.4)
    ax_b.set_title(
        f"Mode B — Native {mode_b['n_folds']}-Fold CV at {CV_BUDGET} tokens\n"
        "(domain shift eliminated)",
        fontsize=10, fontweight="bold",
    )
    ax_b.set_xlabel("Target", fontsize=10)
    ax_b.set_ylabel("R² (mean ± std)", fontsize=10)
    ax_b.set_xticks(x_positions)
    ax_b.set_xticklabels([t.replace("_", " ") for t in TARGETS], fontsize=10)
    ax_b.set_ylim(-0.1, 1.15)
    ax_b.legend(fontsize=8, loc="upper right")
    ax_b.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "MVV Phase 1.3 — Nonlinear Encoding Probe (pool4x4)\n"
        "Is n_defs / n_classes information destroyed at 256 tokens?",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    path = out_dir / "comparison_plot.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    features_root: Path,
    labels_path: Path,
    out_dir: Path,
    n_components: int,
    alpha: float,
    n_estimators: int,
    n_folds: int,
) -> None:
    print("=" * 60)
    print("MVV Phase 1.3 — Nonlinear Encoding Probe (revised)")
    print("Mode A: Resolution-as-Test | Mode B: Native CV @ 256")
    print("=" * 60)

    labels = load_labels(labels_path)
    print(f"Labels loaded: {len(labels):,} windowed files")

    out_dir.mkdir(parents=True, exist_ok=True)

    mode_a = run_mode_a(features_root, labels, n_components, alpha)
    mode_b = run_mode_b(features_root, labels, n_components, alpha, n_estimators, n_folds)

    summary = {
        "experiment":   "phase_1_3_nonlinear_encoding_probe_v2",
        "pool":         POOL,
        "targets":      TARGETS,
        "success_threshold": SUCCESS_R2,
        "mode_a_resolution_as_test": mode_a,
        "mode_b_native_cv_256":      mode_b,
    }
    out_path = out_dir / "probe_results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved → {out_path}")

    try:
        _plot(mode_a, mode_b, out_dir)
    except ImportError:
        print("matplotlib not available — skipping plot")

    print("\nAll done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MVV Phase 1.3 nonlinear encoding probe (v2)")
    parser.add_argument("--features_root", type=Path, default=DEFAULT_FEATURES_ROOT)
    parser.add_argument("--labels",        type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out_dir",       type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n_components",  type=int,  default=1024,
                        help="PCA components for Ridge (default 1024)")
    parser.add_argument("--alpha",         type=float, default=100.0,
                        help="Ridge L2 alpha (default 100.0)")
    parser.add_argument("--n_estimators",  type=int,  default=300,
                        help="RF trees (default 300)")
    parser.add_argument("--n_folds",       type=int,  default=5,
                        help="CV folds for Mode B (default 5)")
    args = parser.parse_args()

    run(args.features_root, args.labels, args.out_dir,
        args.n_components, args.alpha, args.n_estimators, args.n_folds)
