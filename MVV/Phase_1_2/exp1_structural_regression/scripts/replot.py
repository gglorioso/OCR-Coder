#!/usr/bin/env python3
"""
replot.py — Standalone replot for MVV Phase 1.2 Structural Regression.

Loads pre-computed R² results from results/regression_results.json and
recreates the degradation-curve plot without re-running any regression.

Output: results/degradation_curve.png  (same location as run_regression.py)

Dependencies: matplotlib, numpy, json  (no sklearn required)
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Paths (relative to this script's location)
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent
_EXP_DIR    = _SCRIPT_DIR.parent
_RESULTS    = _EXP_DIR / "results"
_JSON_PATH  = _RESULTS / "regression_results.json"
_OUT_PATH   = _RESULTS / "degradation_curve.png"


# ---------------------------------------------------------------------------
# Constants (must match run_regression.py)
# ---------------------------------------------------------------------------

BUDGETS = [729, 441, 256, 121]
TARGETS = ["line_count", "n_defs", "n_classes"]

TARGET_COLORS = {
    "line_count": "#2563eb",
    "n_defs":     "#16a34a",
    "n_classes":  "#dc2626",
}


def _px(budget: int) -> str:
    """Token budget → pixel dim string. SigLIP patch size = 14px."""
    side = int(budget ** 0.5)
    return f"{side * 14}\u00d7{side * 14}"


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_results(json_path: Path) -> dict:
    with open(json_path) as f:
        data = json.load(f)
    return data["results"]


# ---------------------------------------------------------------------------
# Plot  (faithfully copied from the edited _plot function in run_regression.py)
# ---------------------------------------------------------------------------

def plot(results: dict, out_path: Path) -> None:
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
               label="Unreadable text (\u2264256 tokens)")

    ax.set_title(
        "Structural Regression: R\u00b2 vs Token Budget",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("Token budget", fontsize=10)
    ax.set_ylabel("R\u00b2 (coefficient of determination)", fontsize=10)
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([f"{b}\n({_px(b)})" for b in BUDGETS], fontsize=8)
    ax.invert_xaxis()
    ax.set_ylim(-0.1, 1.05)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {out_path}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Loading results from {_JSON_PATH}")
    results = load_results(_JSON_PATH)

    # Echo the values being plotted
    print(f"\n{'Budget':>8}  " + "  ".join(f"{t:>12}" for t in TARGETS))
    print("-" * (8 + 3 + 14 * len(TARGETS)))
    for b in BUDGETS:
        key = f"budget_{b}"
        vals = "  ".join(f"{results[key][t]:+.6f}" for t in TARGETS)
        print(f"{b:>8}  {vals}")

    plot(results, _OUT_PATH)
