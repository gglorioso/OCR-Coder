"""
Plot Phase 1.2 Exp2 native-resolution CV degradation curve.
Shows mean vs pool4x4 vs pool8x8 for each target in 3 subplots.
Saves: MVV/Phase_1_2/exp2_spatial_regression/results/degradation_curve_v2.png
"""

import json
import pathlib
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT    = pathlib.Path(__file__).resolve().parents[4]
RESULTS_FILE = REPO_ROOT / "MVV/Phase_1_2/exp2_spatial_regression/results/regression_results_v2.json"
OUT_FILE     = REPO_ROOT / "MVV/Phase_1_2/exp2_spatial_regression/results/degradation_curve_v2.png"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(RESULTS_FILE) as f:
    data = json.load(f)

results = data["results"]
POOLS    = data.get("pools", ["pool4x4"])   # backward-compat with single-pool JSON
BUDGETS  = [121, 256, 441, 729]             # low → high resolution
TARGETS  = ["line_count", "n_defs", "n_classes"]
LABELS   = {"line_count": "Line Count", "n_defs": "Num Definitions", "n_classes": "Num Classes"}

POOL_STYLES = {
    "mean":    {"linestyle": ":",  "marker": "^", "label_suffix": "mean"},
    "pool4x4": {"linestyle": "-",  "marker": "o", "label_suffix": "pool4x4"},
    "pool8x8": {"linestyle": "--", "marker": "s", "label_suffix": "pool8x8"},
}
POOL_COLORS = {
    "mean":    {"line_count": "grey",           "n_defs": "rosybrown",   "n_classes": "darkkhaki"},
    "pool4x4": {"line_count": "steelblue",      "n_defs": "darkorange",  "n_classes": "forestgreen"},
    "pool8x8": {"line_count": "cornflowerblue", "n_defs": "sandybrown",  "n_classes": "mediumseagreen"},
}

# ---------------------------------------------------------------------------
# Console summary table
# ---------------------------------------------------------------------------
for pool in POOLS:
    print("=" * 72)
    print(f"  [{pool}]  Phase 1.2 Exp2 — Native CV R² Summary")
    print(f"  (Ridge α=100, 5-fold native CV)")
    print("=" * 72)
    header = f"{'Target':<20}" + "".join(f"{'tok='+str(b):>12}" for b in BUDGETS)
    print(header)
    print("-" * 72)
    pool_data = results[pool]
    for t in TARGETS:
        label = LABELS[t]
        row = f"{label:<20}" + "".join(f"{pool_data[str(b)][t]['mean_r2']:>12.4f}" for b in BUDGETS)
        print(row)
    print("-" * 72)
    for t in TARGETS:
        label = LABELS[t]
        row = f"{label:<20}" + "".join(f"(±{pool_data[str(b)][t]['std_r2']:.4f}){'':>3}" for b in BUDGETS)
        print(row)
    print("=" * 72)
    print()

# ---------------------------------------------------------------------------
# Plot — 3-panel (one per target), 2 lines per panel (pool4x4 vs pool8x8)
# ---------------------------------------------------------------------------
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    pass

BUDGETS_DESC = list(reversed(BUDGETS))   # [729, 441, 256, 121]
x = np.arange(len(BUDGETS_DESC))

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for ax, target in zip(axes, TARGETS):
    for pool in POOLS:
        pool_data = results[pool]
        means = [pool_data[str(b)][target]["mean_r2"] for b in BUDGETS_DESC]
        stds  = [pool_data[str(b)][target]["std_r2"]  for b in BUDGETS_DESC]
        style = POOL_STYLES[pool]
        color = POOL_COLORS[pool][target]
        ax.plot(x, means,
                linestyle=style["linestyle"], marker=style["marker"],
                linewidth=2, markersize=7, color=color,
                label=style["label_suffix"], markeredgewidth=0)
        ax.fill_between(x,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.12, color=color)

    ax.axhline(0.5, linestyle=":", color="grey", linewidth=1.2, label="R²=0.5")
    ax.set_xlim(-0.4, len(BUDGETS_DESC) - 0.6)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in BUDGETS_DESC])
    ax.set_xlabel("Token Budget (resolution)", fontsize=11)
    ax.set_title(LABELS[target], fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, color="lightgrey", alpha=0.3)

axes[0].set_ylabel("R² (5-fold native CV)", fontsize=11)

fig.suptitle(
    "Phase 1.2 Exp2 — Native CV Degradation: mean vs pool4x4 vs pool8x8  (Ridge α=100)",
    fontsize=13, fontweight="bold", y=1.01
)

plt.tight_layout()
fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
print(f"PNG saved to: {OUT_FILE}")
