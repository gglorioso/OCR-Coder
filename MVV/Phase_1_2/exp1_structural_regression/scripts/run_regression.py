#!/usr/bin/env python3
"""
run_regression.py — Structural regression probe for MVV Phase 1.2

Resolution-as-Test paradigm (identical to Phase 1.1):
  Train: fit LinearRegression on budget_729 features.
  Test:  apply that frozen probe to budget_441, budget_256, budget_121.

Three regression targets (all integers, predicted independently):
  - line_count : total lines in the file
  - n_defs     : number of function definitions
  - n_classes  : number of class definitions

The key question: does R² remain high (≥ 0.8) at 256 tokens, where individual
characters are unreadable and only spatial whitespace geometry survives?

Outputs:
  results/regression_results.json  — R² per target × budget
  results/degradation_curve.png    — R² vs token budget plot

Usage:
    python run_regression.py [--features_dir PATH] [--labels PATH] [--out_dir PATH]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


BUDGETS      = [729, 441, 256, 121]
TRAIN_BUDGET = 729
TARGETS      = ["line_count", "n_defs", "n_classes"]
SUCCESS_R2   = 0.8    # hypothesis threshold

# Colours matched to Phase 1.1 palette
TARGET_COLORS = {
    "line_count": "#2563eb",
    "n_defs":     "#16a34a",
    "n_classes":  "#dc2626",
}

_SCRIPT_DIR  = Path(__file__).parent
_EXP_DIR     = _SCRIPT_DIR.parent
_MVV_DIR     = _EXP_DIR.parent.parent

DEFAULT_FEATURES = _MVV_DIR / "Phase_1_1" / "data_mvv" / "features"
DEFAULT_LABELS   = _EXP_DIR / "data" / "labels.jsonl"
DEFAULT_OUT      = _EXP_DIR / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _px(budget: int) -> str:
    """Token budget → pixel dim string. SigLIP patch size = 14px."""
    side = int(budget ** 0.5)
    return f"{side * 14}×{side * 14}"


def load_labels(path: Path) -> dict[str, dict]:
    """Return {stem: {line_count, n_defs, n_classes}} from labels.jsonl."""
    labels = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            labels[row["stem"]] = {t: row[t] for t in TARGETS}
    return labels


def load_budget_features(features_dir: Path, budget: int) -> tuple[np.ndarray, list[str]]:
    """Load all .pt files for one budget. Returns (X float32, sorted stems)."""
    d     = features_dir / f"budget_{budget}"
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
    """
    Inner-join features with labels on stem.
    Returns (X_aligned, Y_aligned, kept_stems) with rows sorted by stem.
    Y_aligned shape: (N, 3) — columns ordered as TARGETS.
    """
    valid = [(s, i) for i, s in enumerate(stems) if s in labels]
    valid.sort(key=lambda x: x[0])     # stable sort by stem

    if not valid:
        raise RuntimeError("No overlap between feature stems and labels — check gen_labels.py ran successfully.")

    idxs        = [i for _, i in valid]
    kept_stems  = [s for s, _ in valid]
    X_aligned   = X[idxs]
    Y_aligned   = np.array([[labels[s][t] for t in TARGETS] for s in kept_stems], dtype=np.float64)
    return X_aligned, Y_aligned, kept_stems


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def run_regression(features_dir: Path, labels_path: Path, out_dir: Path) -> None:
    print("=" * 60)
    print("MVV Phase 1.2 — Structural Regression Probe")
    print("=" * 60)

    # ── Load labels ─────────────────────────────────────────────────────────
    labels = load_labels(labels_path)
    print(f"\nLabels loaded    : {len(labels):,} files from {labels_path.name}")

    # ── Load train features (budget_729) ─────────────────────────────────────
    print(f"\nLoading budget_{TRAIN_BUDGET} features …")
    t0 = time.time()
    X_raw, train_stems = load_budget_features(features_dir, TRAIN_BUDGET)
    print(f"  {X_raw.shape[0]:,} samples × {X_raw.shape[1]} dims  ({time.time()-t0:.1f}s)")

    X_train, Y_train, train_kept = align(X_raw, train_stems, labels)
    N, D = X_train.shape
    print(f"  After label join : {N:,} samples  ({len(train_stems)-N} stems had no label)")

    # ── Print target statistics ───────────────────────────────────────────────
    print("\nTarget statistics (training set):")
    for j, t in enumerate(TARGETS):
        vals = Y_train[:, j]
        print(f"  {t:12s}: mean={vals.mean():.1f}  std={vals.std():.1f}  "
              f"min={vals.min():.0f}  max={vals.max():.0f}")

    # ── Fit probe ─────────────────────────────────────────────────────────────
    # Scale features to unit variance; targets left unscaled (R² is scale-invariant).
    scaler  = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print("\nFitting LinearRegression (multi-output) …")
    t0  = time.time()
    reg = LinearRegression(fit_intercept=True, n_jobs=-1)
    reg.fit(X_train_scaled, Y_train)
    print(f"  Done in {time.time()-t0:.2f}s")

    # ── Evaluate across budgets ───────────────────────────────────────────────
    print("\nEvaluating at each token budget:")
    print(f"  {'Budget':>8}  {'Tokens':>6}  {'Pixels':>10}  "
          + "  ".join(f"{t:>12}" for t in TARGETS))
    print("  " + "-" * (8 + 6 + 10 + 3 + 14 * len(TARGETS)))

    results: dict[str, dict] = {}

    for budget in BUDGETS:
        t0 = time.time()
        X_raw_b, stems_b = load_budget_features(features_dir, budget)
        X_b, Y_b, _      = align(X_raw_b, stems_b, labels)
        X_b_scaled        = scaler.transform(X_b)

        Y_pred = reg.predict(X_b_scaled)

        r2s = {}
        for j, target in enumerate(TARGETS):
            r2 = r2_score(Y_b[:, j], Y_pred[:, j])
            r2s[target] = round(float(r2), 6)

        results[f"budget_{budget}"] = r2s

        tag  = " (TRAIN)" if budget == TRAIN_BUDGET else " (TEST) "
        vals = "  ".join(f"{r2s[t]:+.4f}" for t in TARGETS)
        print(f"  {tag} {budget:3d}  {_px(budget):>10}  {vals}  ({time.time()-t0:.1f}s)")

    # ── Hypothesis verdict ────────────────────────────────────────────────────
    print("\nHypothesis check (R² ≥ {:.1f} at budget_256):".format(SUCCESS_R2))
    r2_256 = results["budget_256"]
    for target in TARGETS:
        passed = r2_256[target] >= SUCCESS_R2
        mark   = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {target:12s}: R²={r2_256[target]:+.4f}  {mark}")

    # ── Save results ──────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "probe":        "structural_regression",
        "train_budget": TRAIN_BUDGET,
        "n_train":      N,
        "feature_dim":  D,
        "targets":      TARGETS,
        "success_threshold_r2": SUCCESS_R2,
        "results":      results,
        "target_stats": {
            t: {
                "mean": round(float(Y_train[:, j].mean()), 2),
                "std":  round(float(Y_train[:, j].std()),  2),
                "min":  int(Y_train[:, j].min()),
                "max":  int(Y_train[:, j].max()),
            }
            for j, t in enumerate(TARGETS)
        },
    }
    out_path = out_dir / "regression_results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved → {out_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        _plot(results, out_dir)
    except ImportError:
        print("matplotlib not available — skipping plot (pip install matplotlib to enable)")
    print("All done.")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot(results: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))

    for target in TARGETS:
        r2_vals = [results[f"budget_{b}"][target] for b in BUDGETS]
        color   = TARGET_COLORS[target]
        ax.plot(BUDGETS, r2_vals, "o-", color=color, lw=2, markersize=7,
                label=target.replace("_", " "))
        for b, v in zip(BUDGETS, r2_vals):
            ax.annotate(f"{v:.3f}", (b, v), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8, color=color)

    # Shade the unreadable zone (≤ 256 tokens)
    ax.axvspan(121 - 20, 256 + 20, alpha=0.07, color="#f59e0b",
               label="Unreadable text (≤256 tokens)")

    ax.set_title(
        "Structural Regression: R² vs Token Budget",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("Token budget", fontsize=10)
    ax.set_ylabel("R² (coefficient of determination)", fontsize=10)
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([f"{b}\n({_px(b)})" for b in BUDGETS], fontsize=8)
    ax.invert_xaxis()
    ax.set_ylim(-0.1, 1.05)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "degradation_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {path}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MVV Phase 1.2 structural regression probe")
    parser.add_argument("--features_dir", type=Path, default=DEFAULT_FEATURES,
                        help="Path to data_mvv/features/ (contains budget_* subdirs)")
    parser.add_argument("--labels",       type=Path, default=DEFAULT_LABELS,
                        help="Path to labels.jsonl produced by gen_labels.py")
    parser.add_argument("--out_dir",      type=Path, default=DEFAULT_OUT,
                        help="Directory for regression_results.json and plots")
    args = parser.parse_args()

    run_regression(args.features_dir, args.labels, args.out_dir)
