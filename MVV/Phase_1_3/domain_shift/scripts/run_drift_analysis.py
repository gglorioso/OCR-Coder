#!/usr/bin/env python3
"""
run_drift_analysis.py — Geometric domain shift analysis for SigLIP vision encoder features.

Answers three questions about whether reducing the token budget from 729 to 256
causes a meaningful geometric shift in the PCA-projected feature space:

  Q1. Per-image cosine similarity (budget_729 vs budget_256 after shared PCA):
      If the encoder compresses information consistently across budgets, projected
      vectors for the same image should point in the same direction (cos_sim ≈ 1).
      Low cosine similarity indicates per-image rotation / reprojection drift.

  Q2. Class-conditioned centroid drift:
      Even if individual vectors rotate, class centroids could remain stable.
      We measure whether centroid displacement (729→256) exceeds the natural
      within-class spread in the 729-token space (drift_ratio > 1 = bad).
      This reveals whether budget reduction distorts semantic groupings.

  Q3. Linear CKA (Centered Kernel Alignment):
      A rotation-invariant similarity metric for entire representation matrices.
      CKA = 1.0 means the two spaces are linearly equivalent; CKA < 0.5 means
      substantial structural divergence. Unlike cosine similarity it is insensitive
      to per-neuron rescaling and captures global geometry.

Data:
  Features: pool4x4 adaptive max-pool .pt files (18,432-dim, fp16)
  Labels:   labels.jsonl with line_count, n_defs, n_classes per stem

Usage:
    python run_drift_analysis.py [--features_root PATH] [--labels PATH]
                                  [--out_dir PATH] [--n_components INT]
                                  [--cka_subsample INT] [--tsne_subsample INT]
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ---------------------------------------------------------------------------
# Default paths (relative to this script)
# ---------------------------------------------------------------------------

_SCRIPT_DIR   = Path(__file__).parent
_PHASE_DIR    = _SCRIPT_DIR.parent.parent.parent        # MVV/
_FEATURES_ROOT_DEFAULT = (
    _PHASE_DIR / "Phase_1_1" / "exp2_maxpool_comparison" / "data" / "features_maxpool"
)
_LABELS_DEFAULT = (
    _PHASE_DIR / "Phase_1_2" / "exp2_spatial_regression" / "data" / "labels.jsonl"
)
_OUT_DEFAULT = _SCRIPT_DIR.parent / "results"

POOL_SUBDIR = "pool4x4"
TARGETS     = ["n_defs", "n_classes", "line_count"]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_labels(path: Path) -> dict[str, dict]:
    """Read labels.jsonl; return {stem: {line_count, n_defs, n_classes}}."""
    labels: dict[str, dict] = {}
    with open(path) as fh:
        for line in fh:
            row = json.loads(line)
            labels[row["stem"]] = {
                "line_count": row["line_count"],
                "n_defs":     row["n_defs"],
                "n_classes":  row["n_classes"],
            }
    return labels


def load_features_for_budget(features_root: Path, budget: int) -> tuple[np.ndarray, list[str]]:
    """
    Load all .pt files from features_root/pool4x4/budget_{budget}/.
    Returns (X float32 [N x D], sorted stems).
    fp16 tensors are cast to float32.
    """
    d     = features_root / POOL_SUBDIR / f"budget_{budget}"
    paths = sorted(d.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No .pt files found in {d}")

    vecs, stems = [], []
    for p in paths:
        t = torch.load(p, map_location="cpu")
        if t.dtype == torch.float16:
            t = t.float()
        vecs.append(t.numpy())
        stems.append(p.stem)

    return np.stack(vecs, axis=0), stems


def inner_join(
    X_729:  np.ndarray,
    stems_729: list[str],
    X_256:  np.ndarray,
    stems_256: list[str],
    labels: dict,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """
    Inner-join the two budget feature sets and labels on stem.
    Returns (X_729_aligned, X_256_aligned, common_stems, label_arrays).
    label_arrays maps target name → np.ndarray of length N.
    """
    set_256    = set(stems_256)
    set_labels = set(labels.keys())

    # Common stems present in all three sources
    common = sorted(s for s in stems_729 if s in set_256 and s in set_labels)
    if not common:
        raise RuntimeError("No stems shared across budget_729, budget_256, and labels.jsonl")

    idx_729 = {s: i for i, s in enumerate(stems_729)}
    idx_256 = {s: i for i, s in enumerate(stems_256)}

    rows_729 = np.array([idx_729[s] for s in common])
    rows_256 = np.array([idx_256[s] for s in common])

    label_arrays: dict[str, np.ndarray] = {}
    for target in TARGETS:
        label_arrays[target] = np.array([labels[s][target] for s in common], dtype=np.float64)

    return X_729[rows_729], X_256[rows_256], common, label_arrays


# ---------------------------------------------------------------------------
# Metric 1 — Global Cosine Similarity
# ---------------------------------------------------------------------------

def compute_cosine_similarity(P_729: np.ndarray, P_256: np.ndarray) -> dict:
    """
    Per-image cosine similarity between PCA-projected vectors.
    Both inputs are (N x K) float32/float64.
    """
    # L2-normalize each row
    norm_729 = np.linalg.norm(P_729, axis=1, keepdims=True) + 1e-12
    norm_256 = np.linalg.norm(P_256, axis=1, keepdims=True) + 1e-12
    cos_sim  = (P_729 / norm_729 * P_256 / norm_256).sum(axis=1)

    severe_frac = float((cos_sim < 0.5).mean())

    return {
        "mean":         float(np.mean(cos_sim)),
        "std":          float(np.std(cos_sim)),
        "median":       float(np.median(cos_sim)),
        "min":          float(np.min(cos_sim)),
        "max":          float(np.max(cos_sim)),
        "frac_below_0.5": severe_frac,
    }


# ---------------------------------------------------------------------------
# Metric 2 — Class-Conditioned Centroid Drift
# ---------------------------------------------------------------------------

def _within_variance(X_group: np.ndarray, centroid: np.ndarray) -> float:
    """Mean Euclidean distance from each point to the centroid (within-class spread)."""
    if len(X_group) < 2:
        return 0.0
    diffs = X_group - centroid[np.newaxis, :]
    dists = np.linalg.norm(diffs, axis=1)
    return float(dists.mean())


def _bucket_line_count(value: float) -> str:
    v = int(value)
    if v <= 10:
        return "1-10"
    elif v <= 20:
        return "11-20"
    elif v <= 30:
        return "21-30"
    else:
        return "31-40"


def _bucket_int_capped(value: float, cap: int = 4) -> str:
    """Group integers; values >= cap become '{cap}+'."""
    v = int(value)
    return f"{min(v, cap)}+" if v >= cap else str(v)


def compute_centroid_drift(
    P_729:        np.ndarray,
    P_256:        np.ndarray,
    label_arrays: dict[str, np.ndarray],
) -> dict:
    """
    For each target × group, measure centroid drift and within-variance ratio.
    Returns nested dict: {target: {group: {n, centroid_drift, within_var_729, drift_ratio}}}.
    """
    results: dict[str, dict] = {}

    for target in TARGETS:
        vals = label_arrays[target]

        # Choose grouping function
        if target == "line_count":
            grouper = _bucket_line_count
        else:
            grouper = lambda v: _bucket_int_capped(v, cap=4)

        groups: dict[str, list[int]] = {}
        for i, v in enumerate(vals):
            key = grouper(v)
            groups.setdefault(key, []).append(i)

        target_results: dict[str, dict] = {}
        for group_key, idxs in sorted(groups.items()):
            idxs_arr = np.array(idxs)
            g_729    = P_729[idxs_arr]
            g_256    = P_256[idxs_arr]

            c_729 = g_729.mean(axis=0)
            c_256 = g_256.mean(axis=0)

            c_drift    = float(np.linalg.norm(c_729 - c_256))
            within_var = _within_variance(g_729, c_729)
            drift_ratio = c_drift / (within_var + 1e-12)

            target_results[group_key] = {
                "n_samples":         len(idxs),
                "centroid_drift":    round(c_drift,    4),
                "within_var_729":    round(within_var, 4),
                "drift_ratio":       round(drift_ratio, 4),
            }

        results[target] = target_results

    return results


def print_centroid_table(centroid_results: dict) -> None:
    """Print a formatted centroid drift table for each target."""
    for target, groups in centroid_results.items():
        print(f"\n  Target: {target}")
        header = f"    {'Group':>10}  {'N':>6}  {'CentDrift':>12}  {'WithinVar729':>14}  {'DriftRatio':>11}"
        print(header)
        print("    " + "-" * (10 + 6 + 12 + 14 + 11 + 10))
        for group_key, stats in groups.items():
            flag = "  <<< >1.0" if stats["drift_ratio"] > 1.0 else ""
            print(
                f"    {group_key:>10}  {stats['n_samples']:>6}  "
                f"{stats['centroid_drift']:>12.4f}  "
                f"{stats['within_var_729']:>14.4f}  "
                f"{stats['drift_ratio']:>11.4f}"
                f"{flag}"
            )


# ---------------------------------------------------------------------------
# Metric 3 — Linear CKA (from scratch, no external dependency)
# ---------------------------------------------------------------------------

def _center_gram(G: np.ndarray) -> np.ndarray:
    """Double-center a Gram matrix: G_c = HGH where H = I - (1/n)11^T."""
    n     = G.shape[0]
    row_m = G.mean(axis=1, keepdims=True)
    col_m = G.mean(axis=0, keepdims=True)
    total = G.mean()
    return G - row_m - col_m + total


def _hsic(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Unbiased HSIC estimator (linear kernel).
    HSIC(X,Y) = <K_c, L_c>_F / (n-1)^2
    where K = X X^T, L = Y Y^T, both double-centered.
    """
    n   = X.shape[0]
    K_c = _center_gram(X @ X.T)
    L_c = _center_gram(Y @ Y.T)
    return float(np.sum(K_c * L_c)) / (n - 1) ** 2


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA between representation matrices X (n x d1) and Y (n x d2).
    CKA = HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
    Range [0, 1]; 1 = identical geometry, <0.5 = substantial divergence.
    """
    hsic_xy = _hsic(X, Y)
    hsic_xx = _hsic(X, X)
    hsic_yy = _hsic(Y, Y)
    denom   = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def compute_cka(
    P_729: np.ndarray,
    P_256: np.ndarray,
    subsample: int,
    rng: np.random.Generator,
) -> dict:
    """
    Subsample rows for CKA (O(N^2) memory), compute, and return results dict.
    """
    N      = P_729.shape[0]
    n_used = min(N, subsample)

    if n_used < N:
        idxs   = rng.choice(N, size=n_used, replace=False)
        X_sub  = P_729[idxs].astype(np.float64)
        Y_sub  = P_256[idxs].astype(np.float64)
    else:
        X_sub  = P_729.astype(np.float64)
        Y_sub  = P_256.astype(np.float64)

    t0  = time.time()
    cka = linear_cka(X_sub, Y_sub)
    elapsed = time.time() - t0

    return {
        "cka":            round(cka, 6),
        "n_subsample":    n_used,
        "n_total":        N,
        "elapsed_sec":    round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Metric 4 — t-SNE Visualization
# ---------------------------------------------------------------------------

def compute_and_plot_tsne(
    P_729:        np.ndarray,
    P_256:        np.ndarray,
    label_arrays: dict[str, np.ndarray],
    out_path:     Path,
    subsample:    int,
    rng:          np.random.Generator,
) -> None:
    """
    Subsample matched pairs, run t-SNE on the stacked matrix, plot with pair lines
    and n_defs color-coding. Save to out_path.
    """
    N      = P_729.shape[0]
    n_used = min(N, subsample)

    idxs   = rng.choice(N, size=n_used, replace=False)
    sub_729 = P_729[idxs]
    sub_256 = P_256[idxs]
    n_defs  = label_arrays["n_defs"][idxs]

    # Stack for joint t-SNE: first n_used rows = 729, last n_used = 256
    stacked = np.vstack([sub_729, sub_256]).astype(np.float32)

    print(f"  Running t-SNE on {stacked.shape[0]} points ({n_used} pairs) …")
    t0   = time.time()
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb  = tsne.fit_transform(stacked)
    print(f"  t-SNE done in {time.time()-t0:.1f}s")

    pts_729 = emb[:n_used]   # shape (n_used, 2)
    pts_256 = emb[n_used:]   # shape (n_used, 2)

    # Colormap for n_defs (0 = lightest, 3+ = darkest)
    n_defs_capped = np.clip(n_defs, 0, 4).astype(int)
    cmap_729 = plt.get_cmap("Blues")
    cmap_256 = plt.get_cmap("Oranges")

    # Normalize color values to [0.3, 1.0] so 0 is not invisible
    def _color_val(v, vmax=4):
        return 0.3 + 0.7 * (v / vmax)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw thin gray connector lines for each matched pair
    for i in range(n_used):
        ax.plot(
            [pts_729[i, 0], pts_256[i, 0]],
            [pts_729[i, 1], pts_256[i, 1]],
            color="gray", linewidth=0.35, alpha=0.4, zorder=1,
        )

    # Scatter 729 points (blue shades by n_defs)
    scatter_729 = ax.scatter(
        pts_729[:, 0], pts_729[:, 1],
        c=[_color_val(v) for v in n_defs_capped],
        cmap="Blues", vmin=0.0, vmax=1.0,
        s=18, marker="o", label="729 tokens",
        edgecolors="none", alpha=0.85, zorder=2,
    )

    # Scatter 256 points (orange shades by n_defs)
    scatter_256 = ax.scatter(
        pts_256[:, 0], pts_256[:, 1],
        c=[_color_val(v) for v in n_defs_capped],
        cmap="Oranges", vmin=0.0, vmax=1.0,
        s=18, marker="o", label="256 tokens",
        edgecolors="none", alpha=0.85, zorder=2,
    )

    # Legend: budget identity
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1d4ed8",
                   markersize=8, label="729 tokens (blue)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#c2410c",
                   markersize=8, label="256 tokens (orange)"),
        plt.Line2D([0], [0], color="gray", linewidth=1.0, label="matched pair"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9)

    # Color-bar representing n_defs intensity (shared across both)
    sm = plt.cm.ScalarMappable(cmap="Greys", norm=plt.Normalize(vmin=0, vmax=4))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("n_defs (color intensity; 0=light, 4+=dark)", fontsize=8)
    cbar.set_ticks([0, 1, 2, 3, 4])
    cbar.set_ticklabels(["0", "1", "2", "3", "4+"])

    ax.set_title(
        f"t-SNE: pool4x4 features — 729 vs 256 tokens (N={n_used} pairs)\n"
        "Color intensity = n_defs; gray lines connect same image at two budgets",
        fontsize=10, fontweight="bold",
    )
    ax.set_xlabel("t-SNE dim 1", fontsize=9)
    ax.set_ylabel("t-SNE dim 2", fontsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved t-SNE plot → {out_path}")


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def run_drift_analysis(
    features_root: Path,
    labels_path:   Path,
    out_dir:       Path,
    n_components:  int,
    cka_subsample: int,
    tsne_subsample: int,
) -> None:
    rng = np.random.default_rng(42)

    print("=" * 64)
    print("MVV Phase 1.3 — Geometric Domain Shift Analysis")
    print("pool4x4 features: budget_729 vs budget_256")
    print("=" * 64)

    # ── Load labels ───────────────────────────────────────────────────────────
    print(f"\nLoading labels from {labels_path} …")
    labels = load_labels(labels_path)
    print(f"  {len(labels):,} labeled stems")

    # ── Load features ─────────────────────────────────────────────────────────
    print(f"\nLoading budget_729 features …")
    t0 = time.time()
    X_729_raw, stems_729 = load_features_for_budget(features_root, 729)
    print(f"  {X_729_raw.shape[0]:,} samples × {X_729_raw.shape[1]:,} dims  ({time.time()-t0:.1f}s)")

    print(f"Loading budget_256 features …")
    t0 = time.time()
    X_256_raw, stems_256 = load_features_for_budget(features_root, 256)
    print(f"  {X_256_raw.shape[0]:,} samples × {X_256_raw.shape[1]:,} dims  ({time.time()-t0:.1f}s)")

    # ── Inner join ────────────────────────────────────────────────────────────
    print(f"\nInner-joining on stems …")
    X_729, X_256, common_stems, label_arrays = inner_join(
        X_729_raw, stems_729, X_256_raw, stems_256, labels
    )
    N, D = X_729.shape
    print(f"  {N:,} stems in common (dropped {len(stems_729) - N} with no match)")

    # ── PCA ───────────────────────────────────────────────────────────────────
    max_k = min(N, D)
    k     = min(n_components, max_k)
    if k < n_components:
        print(f"  n_components clamped {n_components} → {k} (min(N={N}, D={D})={max_k})")

    print(f"\nFitting PCA({k}, whiten=True, randomized) on X_729 …")
    t0  = time.time()
    pca = PCA(n_components=k, whiten=True, svd_solver="randomized", random_state=42)
    P_729 = pca.fit_transform(X_729)
    var_explained = float(pca.explained_variance_ratio_.sum())
    print(f"  Done in {time.time()-t0:.1f}s  |  variance explained: {var_explained*100:.1f}%")

    print(f"Projecting X_256 with frozen PCA …")
    t0    = time.time()
    P_256 = pca.transform(X_256)
    print(f"  Done in {time.time()-t0:.1f}s  |  P_729 shape: {P_729.shape}  P_256 shape: {P_256.shape}")

    # ── Metric 1: Cosine Similarity ───────────────────────────────────────────
    print("\n" + "=" * 64)
    print("METRIC 1 — Per-Image Cosine Similarity (PCA space)")
    print("=" * 64)
    t0  = time.time()
    cos = compute_cosine_similarity(P_729, P_256)
    print(f"  Mean:   {cos['mean']:.4f}")
    print(f"  Std:    {cos['std']:.4f}")
    print(f"  Median: {cos['median']:.4f}")
    print(f"  Min:    {cos['min']:.4f}")
    print(f"  Max:    {cos['max']:.4f}")
    print(f"  Frac(cos < 0.5):  {cos['frac_below_0.5']*100:.1f}%  (severe per-image rotation)")
    print(f"  ({time.time()-t0:.2f}s)")

    # ── Metric 2: Centroid Drift ───────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("METRIC 2 — Class-Conditioned Centroid Drift")
    print("=" * 64)
    t0              = time.time()
    centroid_results = compute_centroid_drift(P_729, P_256, label_arrays)
    print_centroid_table(centroid_results)
    print(f"\n  ({time.time()-t0:.2f}s)")

    # ── Metric 3: CKA ─────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("METRIC 3 — Linear CKA (rotation-invariant geometry similarity)")
    print("=" * 64)
    print(f"  Subsampling to min(N={N}, {cka_subsample}) = {min(N, cka_subsample)} rows …")
    t0         = time.time()
    cka_result = compute_cka(P_729, P_256, cka_subsample, rng)
    print(f"  CKA(P_729, P_256) = {cka_result['cka']:.4f}  "
          f"(n_subsample={cka_result['n_subsample']:,}, elapsed={cka_result['elapsed_sec']}s)")

    # ── Metric 4: t-SNE ───────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("METRIC 4 — t-SNE Visualization")
    print("=" * 64)
    tsne_path = out_dir / "tsne_drift.png"
    compute_and_plot_tsne(
        P_729, P_256, label_arrays,
        out_path    = tsne_path,
        subsample   = tsne_subsample,
        rng         = rng,
    )

    # ── Interpretation block ──────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("INTERPRETATION")
    print("=" * 64)

    cos_mean = cos["mean"]
    if cos_mean > 0.7:
        cos_verdict = f"HIGH (mean={cos_mean:.3f} > 0.7) — per-image directions are stable"
    elif cos_mean < 0.5:
        cos_verdict = f"LOW (mean={cos_mean:.3f} < 0.5) — severe per-image directional drift"
    else:
        cos_verdict = f"MODERATE (mean={cos_mean:.3f}, 0.5-0.7) — meaningful but not catastrophic drift"

    print(f"\n  Cosine Similarity: {cos_verdict}")

    # Count n_defs groups with drift_ratio > 1.0
    ndefs_groups  = centroid_results.get("n_defs", {})
    ndefs_over1   = [g for g, s in ndefs_groups.items() if s["drift_ratio"] > 1.0]
    ndefs_total   = len(ndefs_groups)
    if ndefs_over1:
        print(f"\n  Centroid Drift (n_defs): {len(ndefs_over1)}/{ndefs_total} groups have "
              f"drift_ratio > 1.0 — centroid displacement EXCEEDS natural within-class spread")
        print(f"    Groups with drift_ratio > 1.0: {ndefs_over1}")
        ndefs_verdict = "DRIFT EXCEEDS NATURAL SPREAD for n_defs"
    else:
        print(f"\n  Centroid Drift (n_defs): all {ndefs_total} groups have drift_ratio ≤ 1.0 "
              f"— centroid displacement is within natural spread")
        ndefs_verdict = "drift within natural spread for n_defs"

    cka_val = cka_result["cka"]
    if cka_val >= 0.5:
        cka_verdict = f"ABOVE 0.5 (CKA={cka_val:.3f}) — representational geometry is broadly preserved"
    else:
        cka_verdict = f"BELOW 0.5 (CKA={cka_val:.3f}) — substantial structural divergence between budgets"
    print(f"\n  Linear CKA: {cka_verdict}")

    # Overall verdict
    domain_shift_confirmed = (cos_mean < 0.7) or bool(ndefs_over1) or (cka_val < 0.5)
    print("\n" + "-" * 64)
    if domain_shift_confirmed:
        print("  OVERALL VERDICT: DOMAIN SHIFT CONFIRMED")
        print("  Reducing token budget from 729 to 256 introduces a measurable")
        print("  geometric shift in the SigLIP pool4x4 feature space.")
    else:
        print("  OVERALL VERDICT: DOMAIN SHIFT WEAK")
        print("  The two token budgets produce geometrically similar features.")
    print("-" * 64)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "analysis":        "geometric_domain_shift",
        "pool":            POOL_SUBDIR,
        "n_total":         N,
        "feature_dim_raw": D,
        "pca": {
            "n_components":       k,
            "variance_explained": round(var_explained, 6),
        },
        "metric1_cosine_similarity":  cos,
        "metric2_centroid_drift":     centroid_results,
        "metric3_linear_cka":         cka_result,
        "interpretation": {
            "cosine_verdict":    cos_verdict,
            "ndefs_drift":       ndefs_verdict,
            "cka_verdict":       cka_verdict,
            "domain_shift":      "CONFIRMED" if domain_shift_confirmed else "WEAK",
        },
    }
    json_path = out_dir / "drift_analysis.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {json_path}")
    print(f"t-SNE plot    → {tsne_path}")
    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Geometric domain shift analysis — pool4x4 SigLIP features (729 vs 256 tokens)"
    )
    parser.add_argument(
        "--features_root", type=Path, default=_FEATURES_ROOT_DEFAULT,
        help="Root dir containing pool4x4/budget_N/ subdirs with .pt feature files",
    )
    parser.add_argument(
        "--labels", type=Path, default=_LABELS_DEFAULT,
        help="labels.jsonl with stem, line_count, n_defs, n_classes per line",
    )
    parser.add_argument(
        "--out_dir", type=Path, default=_OUT_DEFAULT,
        help="Output directory for drift_analysis.json and tsne_drift.png",
    )
    parser.add_argument(
        "--n_components", type=int, default=1024,
        help="PCA components to retain (default 1024)",
    )
    parser.add_argument(
        "--cka_subsample", type=int, default=2000,
        help="Max rows for CKA computation — O(N^2) memory (default 2000)",
    )
    parser.add_argument(
        "--tsne_subsample", type=int, default=400,
        help="Number of matched pairs to visualize in t-SNE plot (default 400)",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_drift_analysis(
        features_root  = args.features_root,
        labels_path    = args.labels,
        out_dir        = args.out_dir,
        n_components   = args.n_components,
        cka_subsample  = args.cka_subsample,
        tsne_subsample = args.tsne_subsample,
    )
