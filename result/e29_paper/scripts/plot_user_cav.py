"""Plot user-level CAV results: gap vs (X, Y)."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("/home/wlia0047/hj82_scratch2/wenyu/e29_paper")
data = json.load(open(OUT / "user_level_cav.json"))

# Filter to n_sampled >= 100
cells = []
for k, v in data.items():
    if v.get("n_sampled", 0) >= 100:
        cells.append(v)
cells.sort(key=lambda c: (c["X"], c["Y"]))

GRID_X = sorted(set(c["X"] for c in cells))
GRID_Y = sorted(set(c["Y"] for c in cells))

# Plot 1: gap vs Y, lines for each X
fig, ax = plt.subplots(figsize=(10, 6))
for X in GRID_X:
    xs, ys = [], []
    for c in cells:
        if c["X"] == X:
            xs.append(c["Y"])
            ys.append(c["gap"])
    ax.plot(xs, ys, marker='o', linewidth=2, label=f"X={X}")
ax.set_xlabel("Y (max sents per user)")
ax.set_ylabel("gap = self_sim - cross_sim")
ax.set_title("User-level CAV gap across (X, Y) — n=100 per cell")
ax.set_xscale('log')
ax.set_xticks(GRID_Y)
ax.set_xticklabels([str(y) for y in GRID_Y])
ax.grid(True, alpha=0.3)
ax.legend(title="X (min words)", loc='upper left', ncol=2)
plt.tight_layout()
out1 = OUT / "cav_gap_vs_Y.png"
plt.savefig(out1, dpi=120)
plt.close()
print(f"saved {out1}")

# Plot 2: gap vs X, lines for each Y
fig, ax = plt.subplots(figsize=(10, 6))
for Y in GRID_Y:
    xs, ys = [], []
    for c in cells:
        if c["Y"] == Y:
            xs.append(c["X"])
            ys.append(c["gap"])
    ax.plot(xs, ys, marker='o', linewidth=2, label=f"Y={Y}")
ax.set_xlabel("X (min words per sent)")
ax.set_ylabel("gap = self_sim - cross_sim")
ax.set_title("User-level CAV gap across (X, Y) — n=100 per cell")
ax.set_xticks(GRID_X)
ax.grid(True, alpha=0.3)
ax.legend(title="Y (max sents/user)", loc='upper left', ncol=2)
plt.tight_layout()
out2 = OUT / "cav_gap_vs_X.png"
plt.savefig(out2, dpi=120)
plt.close()
print(f"saved {out2}")

# Plot 3: AUC heatmap-style
import matplotlib.colors as mcolors
fig, ax = plt.subplots(figsize=(8, 6))
mat = np.full((len(GRID_X), len(GRID_Y)), np.nan)
for c in cells:
    mat[GRID_X.index(c["X"]), GRID_Y.index(c["Y"])] = c["auc"]
im = ax.imshow(mat, aspect='auto', cmap='viridis', vmin=0.85, vmax=1.0)
ax.set_xticks(range(len(GRID_Y)))
ax.set_xticklabels([str(y) for y in GRID_Y])
ax.set_yticks(range(len(GRID_X)))
ax.set_yticklabels([str(x) for x in GRID_X])
ax.set_xlabel("Y")
ax.set_ylabel("X")
ax.set_title("AUC heatmap")
for i in range(len(GRID_X)):
    for j in range(len(GRID_Y)):
        v = mat[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.3f}", ha='center', va='center',
                    color='white' if v < 0.95 else 'black', fontsize=9)
plt.colorbar(im, ax=ax)
plt.tight_layout()
out3 = OUT / "cav_auc_heatmap.png"
plt.savefig(out3, dpi=120)
plt.close()
print(f"saved {out3}")

# Plot 4: gap heatmap
fig, ax = plt.subplots(figsize=(8, 6))
mat = np.full((len(GRID_X), len(GRID_Y)), np.nan)
for c in cells:
    mat[GRID_X.index(c["X"]), GRID_Y.index(c["Y"])] = c["gap"]
im = ax.imshow(mat, aspect='auto', cmap='YlOrRd', vmin=0.20, vmax=0.45)
ax.set_xticks(range(len(GRID_Y)))
ax.set_xticklabels([str(y) for y in GRID_Y])
ax.set_yticks(range(len(GRID_X)))
ax.set_yticklabels([str(x) for x in GRID_X])
ax.set_xlabel("Y (max sents/user)")
ax.set_ylabel("X (min words/sent)")
ax.set_title("gap = self_sim - cross_sim heatmap")
for i in range(len(GRID_X)):
    for j in range(len(GRID_Y)):
        v = mat[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:+.3f}", ha='center', va='center',
                    color='black', fontsize=9)
plt.colorbar(im, ax=ax)
plt.tight_layout()
out4 = OUT / "cav_gap_heatmap.png"
plt.savefig(out4, dpi=120)
plt.close()
print(f"saved {out4}")
