"""
run_probe_v2.py — Phase 1.1 Exp2 repo probe with native CV + PCA.

Replaces the cross-budget train/test split (domain shift) with proper
5-fold stratified CV within each (pool, budget) combination.
PCA to 1024 components equalizes dimensionality across pool types.

Pool types:
  - meanpool  (1152d)   from data_mvv/features/budget_{B}/
  - pool4x4  (18432d)  from exp2_maxpool_comparison/data/features_maxpool/pool4x4/budget_{B}/
  - pool8x8  (73728d)  from exp2_maxpool_comparison/data/features_maxpool/pool8x8/budget_{B}/

Outputs:
  results/probe_results_v2.json
"""

import json
import time
import glob
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent.parent.parent  # MVV/
PHASE_DIR = ROOT / "Phase_1_1"

MEANPOOL_DIR = PHASE_DIR / "data_mvv" / "features"
MAXPOOL_DIR  = PHASE_DIR / "exp2_maxpool_comparison" / "data" / "features_maxpool"
MANIFEST     = PHASE_DIR / "data_mvv" / "manifest.jsonl"
OUT_DIR      = PHASE_DIR / "exp2_maxpool_comparison" / "results"

BUDGETS      = [729, 441, 256, 121]
N_FOLDS      = 5
N_COMPONENTS = 1024

POOL_DIRS = {
    "meanpool": MEANPOOL_DIR,
    "pool4x4":  MAXPOOL_DIR / "pool4x4",
    "pool8x8":  MAXPOOL_DIR / "pool8x8",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stem_to_repo(stem: str) -> str:
    return stem.split("__")[0]


def load_features(feat_dir: Path, budget: int) -> tuple[np.ndarray, list[str]]:
    """Load all .pt files from feat_dir/budget_{budget}/. Returns (X float32, stems)."""
    d = feat_dir / f"budget_{budget}"
    paths = sorted(d.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No .pt files in {d}")
    vecs, names = [], []
    for p in paths:
        t = torch.load(p, map_location="cpu")
        if t.dtype == torch.float16:
            t = t.float()
        vecs.append(t.numpy().ravel())
        names.append(p.stem)
    return np.stack(vecs, axis=0), names


def build_labels(stems: list[str]) -> tuple[np.ndarray, list[str]]:
    """Map stems to integer repo labels. Returns (y, repo_names)."""
    repo_strs = [_stem_to_repo(s) for s in stems]
    repos = sorted(set(repo_strs))
    repo_to_id = {r: i for i, r in enumerate(repos)}
    y = np.array([repo_to_id[r] for r in repo_strs], dtype=np.int32)
    return y, repos


def topk_accuracy(proba: np.ndarray, y_true: np.ndarray, k: int) -> float:
    """Fraction where true label is in top-k predicted classes."""
    topk = np.argsort(proba, axis=1)[:, -k:]
    return float(np.mean([y_true[i] in topk[i] for i in range(len(y_true))]))


# ---------------------------------------------------------------------------
# Single (pool, budget) cross-validated probe
# ---------------------------------------------------------------------------

def run_cv_probe(feat_dir: Path, budget: int, pool_label: str) -> dict:
    t0 = time.time()
    X_raw, stems = load_features(feat_dir, budget)
    y, repos = build_labels(stems)
    n_samples, n_features = X_raw.shape
    n_classes = len(repos)

    # L2-normalize before PCA
    X = normalize(X_raw, norm="l2").astype(np.float32)

    n_comp = min(N_COMPONENTS, n_samples - 1, n_features)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_top1, fold_top5 = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # PCA: fit on train only, transform both
        pca = PCA(n_components=n_comp, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca  = pca.transform(X_test)

        clf = LogisticRegression(
            solver="lbfgs", C=1.0, max_iter=1000,
            multi_class="multinomial", class_weight="balanced", verbose=0,
        )
        clf.fit(X_train_pca, y_train)
        proba = clf.predict_proba(X_test_pca)

        top1 = topk_accuracy(proba, y_test, k=1)
        fold_top1.append(top1)

        if n_classes >= 5:
            top5 = topk_accuracy(proba, y_test, k=5)
            fold_top5.append(top5)

    elapsed = time.time() - t0
    result = {
        "n_samples":    int(n_samples),
        "n_features":   int(n_features),
        "n_components": int(n_comp),
        "n_classes":    int(n_classes),
        "top1_mean":    round(float(np.mean(fold_top1)), 6),
        "top1_std":     round(float(np.std(fold_top1)),  6),
        "fold_top1":    [round(v, 6) for v in fold_top1],
        "elapsed_s":    round(elapsed, 1),
    }
    if fold_top5:
        result["top5_mean"] = round(float(np.mean(fold_top5)), 6)
        result["top5_std"]  = round(float(np.std(fold_top5)),  6)
        result["fold_top5"] = [round(v, 6) for v in fold_top5]

    print(f"  {pool_label:10s} budget={budget:3d}: "
          f"Top-1={result['top1_mean']*100:6.2f}%±{result['top1_std']*100:.2f}%  "
          + (f"Top-5={result['top5_mean']*100:6.2f}%±{result['top5_std']*100:.2f}%" if fold_top5 else "")
          + f"  ({elapsed:.0f}s)")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Verify paths
    for label, d in POOL_DIRS.items():
        for b in BUDGETS:
            bd = d / f"budget_{b}"
            if not bd.exists():
                raise FileNotFoundError(f"Missing: {bd}")

    print("=" * 65)
    print("Phase 1.1 Exp2 — Native CV + PCA Repo Classification")
    print(f"  5-fold stratified CV  |  PCA n_components={N_COMPONENTS}")
    print(f"  15 repos, 8980 samples per (pool, budget) combo")
    print("=" * 65)

    all_results = {}

    for pool_label, feat_dir in POOL_DIRS.items():
        print(f"\n[{pool_label}]")
        all_results[pool_label] = {}
        for budget in BUDGETS:
            res = run_cv_probe(feat_dir, budget, pool_label)
            all_results[pool_label][str(budget)] = res

    # Save JSON
    out = {
        "mode":            "native_cv_pca",
        "description":     "5-fold native CV with PCA(1024) — domain shift eliminated, dimensionality equalized",
        "n_folds":         N_FOLDS,
        "n_components_pca": N_COMPONENTS,
        "results":         all_results,
    }
    json_path = OUT_DIR / "probe_results_v2_balanced.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults saved → {json_path}")

    # Summary table
    print()
    print("=" * 65)
    print("Phase 1.1 Exp2 — Native CV + PCA Repo Classification Summary")
    print("=" * 65)
    print(f"{'Pool':<12} {'Budget':>6}   {'Top-1 Acc':>12}   {'Top-5 Acc':>12}")
    print("-" * 65)
    for pool_label in POOL_DIRS:
        for budget in BUDGETS:
            r = all_results[pool_label][str(budget)]
            top1_str = f"{r['top1_mean']*100:.2f}%±{r['top1_std']*100:.2f}%"
            if "top5_mean" in r:
                top5_str = f"{r['top5_mean']*100:.2f}%±{r['top5_std']*100:.2f}%"
            else:
                top5_str = "N/A"
            print(f"{pool_label:<12} {budget:>6}   {top1_str:>12}   {top5_str:>12}")
    print("-" * 65)
    print("\nDone.")


if __name__ == "__main__":
    main()
