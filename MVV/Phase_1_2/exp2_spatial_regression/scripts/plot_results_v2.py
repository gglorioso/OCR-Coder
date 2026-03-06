"""
Plot Phase 1.2 Exp2 native-resolution CV degradation curve.
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
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
RESULTS_FILE = REPO_ROOT / "MVV/Phase_1_2/exp2_spatial_regression/results/regression_results_v2.json"
OUT_FILE     = REPO_ROOT / "MVV/Phase_1_2/exp2_spatial_regression/results/degradation_curve_v2.png"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(RESULTS_FILE) as f:
    data = json.load(f)

results = data["results"]

BUDGETS  = [121, 256, 441, 729]   # low → high resolution
TARGETS  = ["line_count", "n_defs", "n_classes"]
LABELS   = {"line_count": "Line Count", "n_defs": "Num Definitions", "n_classes": "Num Classes"}
COLORS   = {"line_count": "steelblue", "n_defs": "darkorange", "n_classes": "forestgreen"}

# Extract mean and std per budget/target (sorted low→high)
means = {t: [] for t in TARGETS}
stds  = {t: [] for t in TARGETS}

for b in BUDGETS:
    bdata = results[str(b)]
    for t in TARGETS:
        means[t].append(bdata[t]["mean_r2"])
        stds[t].append(bdata[t]["std_r2"])

means = {t: np.array(v) for t, v in means.items()}
stds  = {t: np.array(v) for t, v in stds.items()}

# ---------------------------------------------------------------------------
# Console summary table
# ---------------------------------------------------------------------------
print("=" * 70)
print(f"{'Phase 1.2 Exp2 — Native CV R² Summary':^70}")
print(f"{'(pool4x4, Ridge α=100, 5-fold native CV)':^70}")
print("=" * 70)
header = f"{'Target':<20}" + "".join(f"{'tok='+str(b):>12}" for b in BUDGETS)
print(header)
print("-" * 70)
for t in TARGETS:
    label = LABELS[t]
    row = f"{label:<20}" + "".join(f"{means[t][i]:>12.4f}" for i in range(len(BUDGETS)))
    print(row)
print("-" * 70)
print(f"{'(std)':^70}")
for t in TARGETS:
    label = LABELS[t]
    row = f"{label:<20}" + "".join(f"(±{stds[t][i]:.4f}){'':>5}" for i in range(len(BUDGETS)))
    print(row)
print("=" * 70)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    pass  # fallback to matplotlib default

fig, ax = plt.subplots(figsize=(8, 5))

# Use evenly-spaced integer positions, high→low resolution (left→right)
BUDGETS_DESC = list(reversed(BUDGETS))   # [729, 441, 256, 121]
x = np.arange(len(BUDGETS_DESC))         # [0, 1, 2, 3] — evenly spaced

for t in TARGETS:
    # Reverse the arrays to match BUDGETS_DESC order
    m = means[t][::-1]
    s = stds[t][::-1]
    color = COLORS[t]
    label = LABELS[t]
    ax.plot(x, m, marker="o", linewidth=2, markersize=7, color=color, label=label,
            markeredgewidth=0)

# Threshold line
ax.axhline(0.5, linestyle="--", color="grey", linewidth=1.2, label="R²=0.5 threshold")

# Axes
ax.set_xlim(-0.4, len(BUDGETS_DESC) - 0.6)
ax.set_ylim(0.0, 1.05)
ax.set_xticks(x)
ax.set_xticklabels([str(b) for b in BUDGETS_DESC])
ax.set_xlabel("Token Budget (resolution)", fontsize=12)
ax.set_ylabel("R² (5-fold native CV)", fontsize=12)

# Title + subtitle
fig.suptitle(
    "Phase 1.2 Exp2 — Native CV Degradation Curve (pool4x4, Ridge α=100)",
    fontsize=13, fontweight="bold", y=0.98
)
ax.set_title(
    "Pure information loss at each resolution — domain shift eliminated",
    fontsize=9.5, color="dimgrey", pad=4
)

# Grid
ax.grid(True, color="lightgrey", alpha=0.3)

# Legend
ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

plt.tight_layout()
fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
print(f"\nPNG saved to: {OUT_FILE}")
