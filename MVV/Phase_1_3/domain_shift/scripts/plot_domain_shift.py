"""
Phase 1.3 Domain Shift Visualisation
Produces two publication-ready plots from drift_analysis.json.

Outputs (300 dpi):
  MVV/Phase_1_3/domain_shift/results/drift_ratio_bar.png
  MVV/Phase_1_3/domain_shift/results/ndefs_displacement.png
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = "/home/ad.msoe.edu/gloriosog/CoderOCR/OCR-Coder"
JSON_PATH = os.path.join(
    REPO_ROOT,
    "MVV/Phase_1_3/domain_shift/results/drift_analysis.json",
)
OUT_DIR = os.path.join(
    REPO_ROOT,
    "MVV/Phase_1_3/domain_shift/results",
)
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(JSON_PATH) as f:
    data = json.load(f)

centroid_drift = data["metric2_centroid_drift"]

# ---------------------------------------------------------------------------
# Plot 1 — "Signal vs. Noise" Grouped Bar Chart  (drift_ratio_bar.png)
# ---------------------------------------------------------------------------

TARGETS = ["line_count", "n_classes", "n_defs"]
DISPLAY_NAMES = {"line_count": "line_count", "n_classes": "n_classes", "n_defs": "n_defs"}

# Collect per-target label → drift_ratio mappings
target_data: dict[str, dict] = {}
for tgt in TARGETS:
    target_data[tgt] = {k: v["drift_ratio"] for k, v in centroid_drift[tgt].items()}

# Build a unified set of x-positions for a grouped bar chart.
# Each structural target occupies its own group; within a group each bar is a label bucket.
fig1, ax1 = plt.subplots(figsize=(11, 6))

PALETTE = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66"]
bar_width = 0.18
group_spacing = 1.5  # fixed distance between group centers

max_n_bars = max(len(target_data[t]) for t in TARGETS)
group_centers = [i * group_spacing for i in range(len(TARGETS))]

for gi, tgt in enumerate(TARGETS):
    label_buckets = list(target_data[tgt].keys())
    n_bars = len(label_buckets)
    offsets = np.arange(n_bars) * bar_width
    offsets -= offsets.mean()  # centre the group
    group_x = group_centers[gi] + offsets

    for bi, (lbl, off) in enumerate(zip(label_buckets, group_x)):
        ratio = target_data[tgt][lbl]
        color = PALETTE[bi % len(PALETTE)]
        ax1.bar(
            off, ratio, width=bar_width * 0.88,
            color=color, alpha=0.85,
            label=lbl if gi == 0 else "_nolegend_",
            edgecolor="white", linewidth=0.6,
        )
        # Value label on bar
        ax1.text(
            off, ratio + 0.01, f"{ratio:.3f}",
            ha="center", va="bottom", fontsize=7.5, color="#333333",
        )

# Reference line at drift_ratio = 1.0
ax1.axhline(1.0, color="#CC0000", linestyle="--", linewidth=1.6, zorder=5)
ax1.text(
    group_centers[-1] + bar_width * max_n_bars * 0.5,
    1.015,
    "Drift = Within-Cluster Variance",
    ha="right", va="bottom", color="#CC0000", fontsize=9, style="italic",
)
ax1.text(
    group_centers[len(group_centers) // 2],
    1.06,
    "Severe Domain Shift (Probe Blindness)",
    ha="center", va="bottom", color="#880000", fontsize=9.5, fontweight="bold",
)

# Axes
ax1.set_xticks(group_centers)
ax1.set_xticklabels([DISPLAY_NAMES[t] for t in TARGETS], fontsize=11)
ax1.set_ylabel("Drift Ratio  (centroid drift / within-cluster variance)", fontsize=10)
ax1.set_xlabel("Structural Target", fontsize=10)
ax1.set_ylim(0, 1.15)
ax1.set_xlim(group_centers[0] - bar_width * max_n_bars, group_centers[-1] + bar_width * max_n_bars)
ax1.yaxis.grid(True, alpha=0.3, color="grey", linestyle="-")
ax1.set_axisbelow(True)
ax1.spines[["top", "right"]].set_visible(False)

# Legend for label buckets (from first group's bars)
handles = [
    mpatches.Patch(color=PALETTE[i % len(PALETTE)], alpha=0.85, label=lbl)
    for i, lbl in enumerate(list(target_data["line_count"].keys()))
]
ax1.legend(
    handles=handles, title="Label bucket", fontsize=8.5,
    title_fontsize=9, loc="upper right", framealpha=0.85,
)

ax1.set_title(
    "Phase 1.3: Class-Conditioned Centroid Drift (729 \u2192 256 tokens)",
    fontsize=13, fontweight="bold", pad=14,
)

fig1.tight_layout()
out1 = os.path.join(OUT_DIR, "drift_ratio_bar.png")
fig1.savefig(out1, dpi=300, bbox_inches="tight")
plt.close(fig1)
print(f"Saved: {out1}")

# ---------------------------------------------------------------------------
# Plot 2 — n_defs Physical Displacement  (ndefs_displacement.png)
# ---------------------------------------------------------------------------

ndefs = centroid_drift["n_defs"]
label_order = list(ndefs.keys())          # ["0","1","2","3","4+"]
x_ticks = list(range(len(label_order)))

drift_vals   = [ndefs[k]["centroid_drift"] for k in label_order]
within_vals  = [ndefs[k]["within_var_729"] for k in label_order]

fig2, ax2 = plt.subplots(figsize=(9, 5.5))

# Fill between lines where drift > within_var (none here, but kept for generality)
ax2.fill_between(
    x_ticks,
    drift_vals,
    within_vals,
    where=[d > w for d, w in zip(drift_vals, within_vals)],
    color="#FFAAAA", alpha=0.45, label="Drift > Within-Var",
)

# Within-variance reference (dotted)
ax2.plot(
    x_ticks, within_vals,
    color="#888888", linestyle=":", linewidth=2.0,
    marker="s", markersize=6, markerfacecolor="white", markeredgewidth=1.5,
    label="Within-cluster variance @ 729 tokens",
    zorder=4,
)

# Drift distance (solid dark blue)
ax2.plot(
    x_ticks, drift_vals,
    color="#1A3A6B", linestyle="-", linewidth=2.5,
    marker="o", markersize=7, markerfacecolor="#1A3A6B",
    label="Centroid drift distance (729 → 256)",
    zorder=5,
)

# Annotate n_samples
for xi, lbl in zip(x_ticks, label_order):
    n = ndefs[lbl]["n_samples"]
    ax2.text(xi, drift_vals[xi] - 1.1, f"n={n:,}", ha="center", va="top",
             fontsize=7.5, color="#333333")

ax2.set_xticks(x_ticks)
ax2.set_xticklabels(label_order, fontsize=11)
ax2.set_xlabel("n_defs label (function-definition count)", fontsize=10)
ax2.set_ylabel("Geometric distance in PCA-1024 feature space", fontsize=10)
ax2.set_ylim(0, max(within_vals) * 1.15)

ax2.yaxis.grid(True, alpha=0.3, color="grey", linestyle="-")
ax2.set_axisbelow(True)
ax2.spines[["top", "right"]].set_visible(False)

ax2.legend(fontsize=9.5, loc="lower right", framealpha=0.85)
ax2.set_title(
    "Phase 1.3: n_defs Centroid Displacement by Label Count",
    fontsize=13, fontweight="bold", pad=14,
)

fig2.tight_layout()
out2 = os.path.join(OUT_DIR, "ndefs_displacement.png")
fig2.savefig(out2, dpi=300, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {out2}")
