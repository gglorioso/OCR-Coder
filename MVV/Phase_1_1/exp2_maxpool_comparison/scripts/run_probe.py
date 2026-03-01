"""
run_maxpool_probe.py — Repo-level linear probe on adaptive max-pool features.

Compares three pooling strategies across all token budgets:
  - mean-pool  (baseline, 1,152d)   from data_mvv/features/
  - pool 4×4   (16 zones, 18,432d)  from data_mvv/features_maxpool/pool4x4/
  - pool 8×8   (64 zones, 73,728d)  from data_mvv/features_maxpool/pool8x8/

For each strategy:
  - Train LogisticRegression on budget_729 features (15 repo classes)
  - Test on budget_441, 256, 121 (same images, lower resolution)
  - Report Top-1 and Top-3 accuracy + lift over random

Outputs:
  results/maxpool_repo_results.json
  results/maxpool_comparison_top1.png
  results/maxpool_comparison_top3.png

Usage:
    python MVV/Phase_1_1/run_maxpool_probe.py \\
        --features_dir  MVV/Phase_1_1/data_mvv/features \\
        --maxpool_dir   MVV/Phase_1_1/data_mvv/features_maxpool \\
        --out_dir       MVV/Phase_1_1/results
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BUDGETS     = [729, 441, 256, 121]
TRAIN_BUDGET = 729


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repo(stem: str) -> str:
    return stem.split("__")[0]


def _px(budget: int) -> str:
    side = int(budget ** 0.5)
    return f"{side * 14}×{side * 14}"


def load_budget_from_dir(feat_dir: Path, budget: int) -> tuple[np.ndarray, list[str]]:
    """Load all .pt files from feat_dir/budget_{budget}/. Returns (X float32, sorted stems)."""
    d = feat_dir / f"budget_{budget}"
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
    return np.stack(vecs), names


# ---------------------------------------------------------------------------
# Probe runner (repo-level, 15 classes)
# ---------------------------------------------------------------------------

def run_repo_probe(feat_dir: Path, label: str) -> dict:
    """
    Train LogReg on budget_729 features in feat_dir, test on all budgets.
    Returns results dict keyed by budget.
    """
    print(f"\n  [{label}] Loading budget_{TRAIN_BUDGET} …")
    t0 = time.time()
    X_raw, train_names = load_budget_from_dir(feat_dir, TRAIN_BUDGET)
    N, D = X_raw.shape
    print(f"    {N} samples × {D:,} dims  ({time.time()-t0:.1f}s)")

    repos      = sorted(set(_repo(n) for n in train_names))
    repo_to_id = {r: i for i, r in enumerate(repos)}
    n_repos    = len(repos)
    random_base = 1.0 / n_repos

    y_train = np.array([repo_to_id[_repo(n)] for n in train_names], dtype=np.int32)
    X_train = normalize(X_raw, norm="l2")

    print(f"    {n_repos} repos  |  random baseline = {random_base*100:.2f}%")
    print(f"    Training LogisticRegression (lbfgs, C=1, max_iter=2000) …")
    t0 = time.time()
    clf = LogisticRegression(
        solver="lbfgs", C=1.0, max_iter=2000,
        multi_class="multinomial", verbose=0,
    )
    clf.fit(X_train, y_train)
    print(f"    Done in {time.time()-t0:.1f}s")

    results = {"dim": int(D), "random_base": round(random_base, 6), "repos": repos, "budgets": {}}

    for budget in BUDGETS:
        t0 = time.time()
        X_raw_b, test_names = load_budget_from_dir(feat_dir, budget)
        X_test = normalize(X_raw_b, norm="l2")
        y_true = np.array([repo_to_id[_repo(n)] for n in test_names], dtype=np.int32)

        proba    = clf.predict_proba(X_test)
        top1_acc = float((proba.argmax(axis=1) == y_true).mean())

        top3_pred = np.argsort(proba, axis=1)[:, -3:]
        top3_acc  = float(np.array([y_true[i] in top3_pred[i]
                                     for i in range(len(y_true))]).mean())

        results["budgets"][budget] = {
            "top1_acc":  round(top1_acc, 6),
            "top3_acc":  round(top3_acc, 6),
            "lift_top1": round(top1_acc / random_base, 2),
            "lift_top3": round(top3_acc / (random_base * 3), 2),
        }
        tag = "(TRAIN)" if budget == TRAIN_BUDGET else "(TEST) "
        print(f"    {tag} budget_{budget:3d}: "
              f"Top-1={top1_acc*100:6.2f}%  Top-3={top3_acc*100:6.2f}%  "
              f"lift×{top1_acc/random_base:.1f}  ({time.time()-t0:.1f}s)")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

STRATEGY_STYLE = {
    "mean-pool": dict(color="#6b7280", lw=1.5, ls="--",  marker="D", ms=6, label="mean-pool (1,152d)"),
    "pool 4×4":  dict(color="#2563eb", lw=2.0, ls="-",   marker="o", ms=7, label="pool 4×4 (18,432d)"),
    "pool 8×8":  dict(color="#dc2626", lw=2.0, ls="-",   marker="s", ms=7, label="pool 8×8 (73,728d)"),
}


def _plot_metric(ax, all_results: dict, metric: str, metric_label: str):
    for strategy, res in all_results.items():
        vals = [res["budgets"][b][metric] * 100 for b in BUDGETS]
        style = STRATEGY_STYLE[strategy]
        ax.plot(BUDGETS, vals,
                color=style["color"], lw=style["lw"], ls=style["ls"],
                marker=style["marker"], markersize=style["ms"], label=style["label"])
        for b, v in zip(BUDGETS, vals):
            ax.annotate(f"{v:.1f}%", (b, v), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7,
                        color=style["color"])

    rand_pct = list(all_results.values())[0]["random_base"] * 100
    ax.axhline(rand_pct, color="gray", lw=1, ls=":", label=f"Random ({rand_pct:.1f}%)")
    ax.axvspan(TRAIN_BUDGET - 30, TRAIN_BUDGET + 30, alpha=0.07,
               color="#2563eb", label="Train budget")

    ax.set_xlabel("Token budget", fontsize=10)
    ax.set_ylabel(f"{metric_label} (%)", fontsize=10)
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([f"{b}\n({_px(b)})" for b in BUDGETS], fontsize=8)
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_comparison(all_results: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].set_title("Top-1 Accuracy — Pooling Strategy Comparison\n"
                       "Repo probe (15 classes), train @ 729 tokens", fontsize=10)
    _plot_metric(axes[0], all_results, "top1_acc", "Top-1 Accuracy")

    axes[1].set_title("Top-3 Accuracy — Pooling Strategy Comparison\n"
                       "Repo probe (15 classes), train @ 729 tokens", fontsize=10)
    _plot_metric(axes[1], all_results, "top3_acc", "Top-3 Accuracy")

    plt.suptitle("MVV Phase 1.1 — Adaptive Max-Pool vs Mean-Pool Degradation",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "maxpool_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {path}")

    # Per-budget lift table printed to console
    print("\n--- Lift over random (Top-1) ---")
    header = f"{'Budget':>8}  " + "  ".join(f"{s:>12}" for s in all_results)
    print(header)
    for b in BUDGETS:
        row = f"{b:>8}  "
        for res in all_results.values():
            lift = res["budgets"][b]["lift_top1"]
            row += f"  {lift:>12.1f}×"
        print(row)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=Path,
                        default=Path(__file__).parent.parent.parent / "data_mvv" / "features",
                        help="Dir containing budget_N/ subdirs with mean-pool features")
    parser.add_argument("--maxpool_dir",  type=Path,
                        default=Path(__file__).parent.parent / "data" / "features_maxpool",  # exp2/data/features_maxpool/
                        help="Dir containing pool4x4/ and pool8x8/ subdirs")
    parser.add_argument("--out_dir",      type=Path,
                        default=Path(__file__).parent.parent / "results")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    strategies = {
        "mean-pool": args.features_dir,
        "pool 4×4":  args.maxpool_dir / "pool4x4",
        "pool 8×8":  args.maxpool_dir / "pool8x8",
    }

    # Validate dirs exist
    missing = [name for name, d in strategies.items()
               if not (d / f"budget_{TRAIN_BUDGET}").exists()]
    if missing:
        print(f"ERROR: Missing feature dirs for: {missing}")
        print("  Run extract_maxpool_features.py first (see extract_maxpool_features.sh)")
        raise SystemExit(1)

    all_results = {}
    print("=" * 60)
    print("Repo-level probe — 15 classes, train on budget_729")
    print("=" * 60)

    for name, feat_dir in strategies.items():
        all_results[name] = run_repo_probe(feat_dir, name)

    # Save JSON
    json_path = args.out_dir / "maxpool_repo_results.json"
    json_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved → {json_path}")

    plot_comparison(all_results, args.out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
