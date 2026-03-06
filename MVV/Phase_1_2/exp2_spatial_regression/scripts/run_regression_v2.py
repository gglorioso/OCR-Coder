#!/home/ad.msoe.edu/gloriosog/DS OCR/envs/deepseek-ocr/bin/python
"""
run_regression_v2.py — Native-resolution CV probe for MVV Phase 1.2 Exp2

For EACH budget independently, runs 5-fold CV (train on 4 folds, test on 1),
reporting mean R² ± std. This isolates pure information loss from domain shift
(unlike run_regression.py which trains on budget_729 and tests on lower budgets).

Pool: pool4x4 only (shape [18432] fp16 per sample)
Targets: line_count, n_defs, n_classes
Pipeline per fold: PCA(n_components=min(1024, n_train, n_features)) → Ridge(alpha=100)

Output:
  results/regression_results_v2.json  — mean/std R² per budget × target

Usage:
    python run_regression_v2.py
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


_REPO_ROOT   = Path(__file__).resolve().parents[4]          # OCR-Coder/
FEATURES_ROOT = _REPO_ROOT / "MVV" / "Phase_1_1" / "exp2_maxpool_comparison" / "data" / "features_maxpool"
POOL         = "pool4x4"
LABELS_PATH  = Path(__file__).resolve().parents[1] / "data" / "labels.jsonl"
OUT_PATH     = Path(__file__).resolve().parents[1] / "results" / "regression_results_v2.json"

BUDGETS  = [729, 441, 256, 121]
TARGETS  = ["line_count", "n_defs", "n_classes"]
N_FOLDS  = 5
ALPHA    = 100.0
N_COMPONENTS_MAX = 1024


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


def load_budget_features(pool: str, budget: int) -> tuple:
    """Return (X [N, D], stems [N]) for the given pool/budget."""
    d     = FEATURES_ROOT / pool / f"budget_{budget}"
    paths = sorted(d.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No .pt files in {d}")

    vecs, names = [], []
    for p in paths:
        t = torch.load(p, map_location="cpu")
        if t.dtype == torch.float16:
            t = t.float()
        vecs.append(t.numpy().ravel())     # flatten just in case
        names.append(p.stem)

    return np.stack(vecs, axis=0), names


def align(X: np.ndarray, stems: list, labels: dict) -> tuple:
    """Inner-join features with labels on stem, sort by stem."""
    valid = [(s, i) for i, s in enumerate(stems) if s in labels]
    valid.sort(key=lambda x: x[0])
    if not valid:
        raise RuntimeError("No overlap between feature stems and labels.")
    idxs       = [i for _, i in valid]
    kept_stems = [s for s, _ in valid]
    X_al = X[idxs]
    Y_al = np.array([[labels[s][t] for t in TARGETS] for s in kept_stems], dtype=np.float64)
    return X_al, Y_al, kept_stems


# ---------------------------------------------------------------------------
# Per-budget 5-fold CV
# ---------------------------------------------------------------------------

def run_budget_cv(budget: int, labels: dict) -> dict:
    t0 = time.time()
    print(f"\n  budget_{budget}: loading features …", end="", flush=True)

    X_raw, stems = load_budget_features(POOL, budget)
    X, Y, _      = align(X_raw, stems, labels)
    N, D         = X.shape
    print(f" {N:,} samples × {D:,} dims  ({time.time()-t0:.1f}s)")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # Per-target fold R² storage
    fold_r2s = {t: [] for t in TARGETS}

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]

        n_tr = len(train_idx)
        k    = min(N_COMPONENTS_MAX, n_tr, D)

        pca = PCA(n_components=k, whiten=True, svd_solver="randomized", random_state=42)
        X_tr_pca = pca.fit_transform(X_tr)
        X_te_pca = pca.transform(X_te)

        reg = Ridge(alpha=ALPHA, fit_intercept=True)
        reg.fit(X_tr_pca, Y_tr)
        Y_pred = reg.predict(X_te_pca)

        for j, target in enumerate(TARGETS):
            fold_r2s[target].append(float(r2_score(Y_te[:, j], Y_pred[:, j])))

        elapsed = time.time() - t0
        print(f"    fold {fold_idx+1}/{N_FOLDS}  (k={k}) "
              + "  ".join(f"{target}={fold_r2s[target][-1]:+.4f}" for target in TARGETS)
              + f"  [{elapsed:.1f}s]")

    # Aggregate
    result = {"n_samples": N}
    for target in TARGETS:
        folds = fold_r2s[target]
        result[target] = {
            "mean_r2": round(float(np.mean(folds)), 6),
            "std_r2":  round(float(np.std(folds)),  6),
            "fold_r2": [round(v, 6) for v in folds],
        }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("MVV Phase 1.2 Exp2 — Native-Resolution CV  (run_regression_v2)")
    print("  pool4x4 | 5-fold CV per budget | PCA + Ridge | alpha=100")
    print("=" * 65)

    labels = load_labels(LABELS_PATH)
    print(f"Labels loaded: {len(labels):,} stems")

    budget_results: dict = {}
    for budget in BUDGETS:
        budget_results[str(budget)] = run_budget_cv(budget, labels)

    # ── Summary table ──────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print(f"{'Budget':>8}   {'line_count R²':>16}   {'n_defs R²':>16}   {'n_classes R²':>16}")
    print("-" * 65)
    for budget in BUDGETS:
        r = budget_results[str(budget)]
        cols = []
        for target in TARGETS:
            m = r[target]["mean_r2"]
            s = r[target]["std_r2"]
            cols.append(f"{m:.2f}±{s:.2f}")
        print(f"  {budget:>4}       {cols[0]:>16}   {cols[1]:>16}   {cols[2]:>16}")
    print("=" * 65)

    # ── Save JSON ──────────────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "mode": "native_cv",
        "description": (
            "5-fold native CV per budget — no cross-budget train/test. "
            "Isolates information loss from domain shift."
        ),
        "pool":    POOL,
        "alpha":   ALPHA,
        "n_folds": N_FOLDS,
        "results": budget_results,
    }
    OUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved → {OUT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
