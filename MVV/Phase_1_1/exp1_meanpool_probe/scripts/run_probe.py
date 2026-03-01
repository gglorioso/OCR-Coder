"""
run_probe.py — Linear probe for MVV Phase 1.1
Resolution-as-Test: train LogReg on 729-token features, test on 441/256/121.
All features are from the same images; only pixel resolution changes.

Runs two probes:
  1. Per-file probe  — 8,980 classes, 1 sample each (sanity check: should collapse)
  2. Repo probe      — 15 classes, ~600 samples each (real signal test)

Usage:
    python run_probe.py [--features_dir data_mvv/features] [--out_dir results]
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


BUDGETS = [729, 441, 256, 121]
TRAIN_BUDGET = 729


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repo_from_name(stem: str) -> str:
    """'black__action__main_py' → 'black'"""
    return stem.split("__")[0]


def _px(budget: int) -> str:
    """Token budget → pixel dim string. SigLIP patch size = 14px."""
    side = int(budget ** 0.5)
    px = side * 14
    return f"{px}×{px}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_budget(features_dir: Path, budget: int) -> tuple[np.ndarray, list[str]]:
    """Load all .pt files for one budget. Returns (X float32, sorted stems)."""
    d = features_dir / f"budget_{budget}"
    paths = sorted(d.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No .pt files found in {d}")

    vecs, names = [], []
    for p in paths:
        t = torch.load(p, map_location="cpu")
        if t.dtype == torch.float16:
            t = t.float()
        vecs.append(t.numpy())
        names.append(p.stem)

    return np.stack(vecs, axis=0), names


def _eval_budget(clf, features_dir, budget, name_to_idx, n_classes, top_k=5):
    """Load budget features, align to training order, return (top1, top5) accuracy."""
    X_raw, test_names = load_budget(features_dir, budget)
    X = normalize(X_raw, norm="l2")

    # Align: sort by training label index so row i has label i
    order = np.array([name_to_idx[n] for n in test_names])
    X_aligned = X[np.argsort(order)]
    y_true = np.arange(n_classes, dtype=np.int32)

    proba = clf.predict_proba(X_aligned)              # (N, n_classes)
    top1_acc = (proba.argmax(axis=1) == y_true).mean()

    top_k_pred = np.argsort(proba, axis=1)[:, -top_k:]
    topk_acc = np.array([y_true[i] in top_k_pred[i] for i in range(n_classes)]).mean()

    return float(top1_acc), float(topk_acc)


# ---------------------------------------------------------------------------
# Probe 1: Per-file (8,980 classes, 1 sample each)
# ---------------------------------------------------------------------------

def run_file_probe(features_dir: Path, out_dir: Path) -> dict:
    print("=" * 60)
    print("PROBE 1 — Per-file (8,980 classes, 1 sample each)")
    print("=" * 60)

    print(f"\nLoading budget_{TRAIN_BUDGET} features …")
    t0 = time.time()
    X_raw, train_names = load_budget(features_dir, TRAIN_BUDGET)
    N, D = X_raw.shape
    print(f"  {N} samples × {D} dims  ({time.time()-t0:.1f}s)")

    name_to_idx = {n: i for i, n in enumerate(train_names)}
    n_classes   = N
    random_base = 1.0 / n_classes
    print(f"  {n_classes} classes  |  random baseline = {random_base*100:.4f}%")

    X_train = normalize(X_raw, norm="l2")
    y_train = np.arange(n_classes, dtype=np.int32)

    print("\nTraining LogisticRegression (lbfgs, C=1e4, max_iter=5000) …")
    t0 = time.time()
    clf = LogisticRegression(
        solver="lbfgs", C=1e4, max_iter=5000,
        multi_class="multinomial", verbose=0,
    )
    clf.fit(X_train, y_train)
    print(f"  Done in {time.time()-t0:.1f}s")

    results = {}
    for budget in BUDGETS:
        t0 = time.time()
        top1, top5 = _eval_budget(clf, features_dir, budget, name_to_idx, n_classes)
        results[budget] = {
            "top1_acc":  round(top1, 6),
            "top5_acc":  round(top5, 6),
            "lift_top1": round(top1 / random_base, 2),
            "lift_top5": round(top5 / (random_base * 5), 2),
        }
        tag = "(TRAIN)" if budget == TRAIN_BUDGET else "(TEST) "
        print(f"  {tag} budget_{budget:3d}: Top-1={top1*100:6.2f}%  Top-5={top5*100:6.2f}%  "
              f"lift×{top1/random_base:.1f}  ({time.time()-t0:.1f}s)")

    summary = {
        "probe": "per_file",
        "n_classes": n_classes, "feature_dim": int(D),
        "random_baseline_top1": round(random_base, 8),
        "train_budget": TRAIN_BUDGET, "budgets": results,
    }
    path = out_dir / "file_probe_results.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved → {path}")
    return summary


# ---------------------------------------------------------------------------
# Probe 2: Repo-level (15 classes, ~600 samples each)
# ---------------------------------------------------------------------------

def run_repo_probe(features_dir: Path, out_dir: Path) -> dict:
    print("\n" + "=" * 60)
    print("PROBE 2 — Repo-level (15 classes, ~600 samples each)")
    print("=" * 60)

    print(f"\nLoading budget_{TRAIN_BUDGET} features …")
    t0 = time.time()
    X_raw, train_names = load_budget(features_dir, TRAIN_BUDGET)
    N, D = X_raw.shape
    print(f"  {N} samples × {D} dims  ({time.time()-t0:.1f}s)")

    # Build repo labels
    repos = sorted(set(_repo_from_name(n) for n in train_names))
    repo_to_idx = {r: i for i, r in enumerate(repos)}
    n_repos = len(repos)
    random_base = 1.0 / n_repos

    y_train = np.array([repo_to_idx[_repo_from_name(n)] for n in train_names], dtype=np.int32)
    counts = {r: int((y_train == i).sum()) for r, i in repo_to_idx.items()}
    print(f"  {n_repos} repos: " + ", ".join(f"{r}({c})" for r, c in sorted(counts.items())))
    print(f"  random baseline = {random_base*100:.2f}%")

    X_train = normalize(X_raw, norm="l2")

    print("\nTraining LogisticRegression (lbfgs, C=1, max_iter=2000) …")
    t0 = time.time()
    clf = LogisticRegression(
        solver="lbfgs", C=1.0, max_iter=2000,
        multi_class="multinomial", verbose=0,
    )
    clf.fit(X_train, y_train)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Evaluation helper (repo-level: labels per sample, not per-class)
    results = {}
    for budget in BUDGETS:
        t0 = time.time()
        X_raw_b, test_names = load_budget(features_dir, budget)
        X_test = normalize(X_raw_b, norm="l2")
        y_true = np.array([repo_to_idx[_repo_from_name(n)] for n in test_names], dtype=np.int32)

        proba = clf.predict_proba(X_test)             # (N, n_repos)
        top1_acc = (proba.argmax(axis=1) == y_true).mean()
        top3_pred = np.argsort(proba, axis=1)[:, -3:]
        top3_acc = np.array([y_true[i] in top3_pred[i] for i in range(len(y_true))]).mean()

        results[budget] = {
            "top1_acc":  round(float(top1_acc), 6),
            "top3_acc":  round(float(top3_acc), 6),
            "lift_top1": round(float(top1_acc) / random_base, 2),
            "lift_top3": round(float(top3_acc) / (random_base * 3), 2),
        }
        tag = "(TRAIN)" if budget == TRAIN_BUDGET else "(TEST) "
        print(f"  {tag} budget_{budget:3d}: Top-1={top1_acc*100:6.2f}%  Top-3={top3_acc*100:6.2f}%  "
              f"lift×{top1_acc/random_base:.1f}  ({time.time()-t0:.1f}s)")

    # Per-repo breakdown at 729 tokens (training budget)
    X_729 = normalize(X_raw, norm="l2")
    proba_729 = clf.predict_proba(X_729)
    pred_729  = proba_729.argmax(axis=1)
    per_repo  = {}
    for r, i in repo_to_idx.items():
        mask = y_train == i
        acc  = (pred_729[mask] == i).mean()
        per_repo[r] = round(float(acc), 4)

    summary = {
        "probe": "repo",
        "n_classes": n_repos, "feature_dim": int(D),
        "random_baseline_top1": round(random_base, 6),
        "train_budget": TRAIN_BUDGET,
        "repos": repos,
        "samples_per_repo": counts,
        "budgets": results,
        "per_repo_accuracy_at_729": per_repo,
    }
    path = out_dir / "repo_probe_results.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved → {path}")
    return summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_degradation(file_summary: dict, repo_summary: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, summary, title_tag, topk_key, topk_label in [
        (axes[0], file_summary, "Per-file (8,980 classes)", "top5_acc", "Top-5"),
        (axes[1], repo_summary, "Repo-level (15 classes)",  "top3_acc", "Top-3"),
    ]:
        budgets = BUDGETS
        top1_vals = [summary["budgets"][b]["top1_acc"] * 100 for b in budgets]
        topk_vals = [summary["budgets"][b][topk_key]   * 100 for b in budgets]
        rand_pct  = summary["random_baseline_top1"] * 100

        ax.plot(budgets, top1_vals, "o-",  color="#2563eb", lw=2, markersize=7, label="Top-1")
        ax.plot(budgets, topk_vals, "s--", color="#16a34a", lw=2, markersize=7, label=topk_label)
        ax.axhline(rand_pct, color="gray", lw=1, ls=":",
                   label=f"Random ({rand_pct:.2f}%)")

        for b, t1, tk in zip(budgets, top1_vals, topk_vals):
            ax.annotate(f"{t1:.1f}%", (b, t1), textcoords="offset points",
                        xytext=(0, 8),  ha="center", fontsize=8, color="#2563eb")
            ax.annotate(f"{tk:.1f}%", (b, tk), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=8, color="#16a34a")

        ax.axvspan(TRAIN_BUDGET - 30, TRAIN_BUDGET + 30, alpha=0.08,
                   color="#2563eb", label="Train budget")
        ax.set_title(f"SigLIP Degradation — {title_tag}\n"
                     f"Train: {TRAIN_BUDGET}-token, Test: lower budgets", fontsize=10)
        ax.set_xlabel("Token budget", fontsize=10)
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.set_xticks(budgets)
        ax.set_xticklabels([f"{b}\n({_px(b)})" for b in budgets], fontsize=8)
        ax.invert_xaxis()
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("MVV Phase 1.1 — Resolution-as-Test Probe", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "degradation_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {path}")

    # Per-repo bar chart (repo probe only)
    per_repo = repo_summary["per_repo_accuracy_at_729"]
    repos_sorted = sorted(per_repo, key=lambda r: per_repo[r], reverse=True)
    accs = [per_repo[r] * 100 for r in repos_sorted]
    rand_pct = repo_summary["random_baseline_top1"] * 100

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    bars = ax2.bar(repos_sorted, accs, color="#2563eb", alpha=0.8)
    ax2.axhline(rand_pct, color="gray", lw=1, ls=":", label=f"Random ({rand_pct:.1f}%)")
    ax2.axhline(100, color="#16a34a", lw=1, ls="--", label="Perfect (100%)")
    for bar, acc in zip(bars, accs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{acc:.0f}%", ha="center", va="bottom", fontsize=8)
    ax2.set_xlabel("Repository", fontsize=10)
    ax2.set_ylabel("Top-1 Accuracy @ budget_729 (%)", fontsize=10)
    ax2.set_title("Per-repo accuracy at training resolution (729 tokens)", fontsize=10)
    ax2.set_ylim(0, 115)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.tight_layout()
    path2 = out_dir / "per_repo_accuracy.png"
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)
    print(f"Plot saved → {path2}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=Path,
                        default=Path(__file__).parent.parent.parent / "data_mvv" / "features")
    parser.add_argument("--out_dir", type=Path,
                        default=Path(__file__).parent.parent / "results")
    parser.add_argument("--skip_file_probe", action="store_true",
                        help="Skip the per-file probe (slow, expected to fail)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_file_probe:
        file_summary = run_file_probe(args.features_dir, args.out_dir)
    else:
        # Load existing result if available
        p = args.out_dir / "file_probe_results.json"
        file_summary = json.loads(p.read_text()) if p.exists() else None

    repo_summary = run_repo_probe(args.features_dir, args.out_dir)

    if file_summary is not None:
        plot_degradation(file_summary, repo_summary, args.out_dir)

    print("\nAll done.")
