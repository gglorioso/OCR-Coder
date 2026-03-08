"""
plot_probe_v2.py — Degradation curve for Phase 1.1 Exp2 native CV + PCA results.

Reads:  results/probe_results_v2.json
Writes: results/probe_degradation_v2.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent.parent / "results"
JSON_PATH   = RESULTS_DIR / "probe_results_v2_balanced.json"
PNG_PATH    = RESULTS_DIR / "probe_degradation_v2.png"

BUDGETS = [729, 441, 256, 121]

POOL_STYLE = {
    "pool4x4":  dict(color="#2563eb", label="pool 4×4"),
    "pool8x8":  dict(color="#f97316", label="pool 8×8"),
    "meanpool": dict(color="#16a34a", label="mean-pool"),
}

# ---------------------------------------------------------------------------

def main():
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"Results not found: {JSON_PATH}\nRun run_probe_v2.py first.")

    data = json.loads(JSON_PATH.read_text())
    results = data["results"]

    fig, ax = plt.subplots(figsize=(8, 5))

    x_pos = list(range(len(BUDGETS)))  # evenly spaced

    for pool_label, style in POOL_STYLE.items():
        pool_res = results.get(pool_label, {})
        top1_vals = []
        for b in BUDGETS:
            r = pool_res.get(str(b), {})
            top1_vals.append(r.get("top1_mean", float("nan")))

        ax.plot(
            x_pos, top1_vals,
            color=style["color"],
            linewidth=2,
            linestyle="-",
            marker="o",
            markersize=7,
            markeredgewidth=0,
            label=style["label"],
        )

    # 50% threshold line
    ax.axhline(0.5, color="#9ca3af", linewidth=1.2, linestyle="--", label="50% threshold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(b) for b in BUDGETS], fontsize=10)
    ax.set_xlabel("Token budget (high → low)", fontsize=11)
    ax.set_ylabel("Top-1 Accuracy", fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))

    ax.set_title(
        "Phase 1.1 Exp2 — Native CV Degradation Curve (PCA=1024, LogReg)",
        fontsize=12, fontweight="bold", pad=14,
    )
    ax.text(
        0.5, 1.01,
        "Domain shift eliminated — pure information loss per resolution",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=9, color="#6b7280",
    )

    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, color="#d1d5db", alpha=0.3)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {PNG_PATH}")


if __name__ == "__main__":
    main()
