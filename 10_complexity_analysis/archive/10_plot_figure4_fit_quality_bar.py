"""Figure 4 (alt, bar chart): per-user mean log p under each prior,
broken out by Amazon domain. Higher = better fit quality.

Layout: 1 row × 3 cols of subplots, one per domain. 2-GMM bar is
highlighted (domain color + dark border); the per-domain 2-GMM
value is shown in a stats panel below each subplot. The 3-domain μ
is shown as a suptitle.

Reads `global_mean` from each precomputed `per_user_fit_quality.json`.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

BASE = Path("/fs04/ar57/wenyu/result/personal_query/10_complexity_analysis_clause_features")
INTERP = BASE / "interpretability_paper"

DOMAINS = [
    {"key": "Baby_Products",            "label_short": "Baby",    "color": "#7DA0CC"},
    {"key": "Pet_Supplies",             "label_short": "Pet",     "color": "#E8A77D"},
    {"key": "Grocery_and_Gourmet_Food", "label_short": "Grocery", "color": "#7DBE8B"},
]

GROUPS = [
    {"key": "gmm_2",     "label": "GMM"},
    {"key": "student_t", "label": "t"},
    {"key": "laplace",   "label": "Lap"},
    {"key": "logistic",  "label": "Log"},
]


def _load_json(path: Path):
    with path.open() as f:
        return json.load(f)


# ============ Load global_mean per (domain, prior) ============
means = np.zeros((len(GROUPS), len(DOMAINS)))
for di, d in enumerate(DOMAINS):
    fq = _load_json(INTERP / d["key"] / "per_user_fit_quality.json")
    for gi, g in enumerate(GROUPS):
        means[gi, di] = float(fq["groups"][g["key"]]["global_mean"])

# ============ Plot: 1 row × 3 cols of subplots ============
# Target paper size: 240.23 pt × 65.66 pt = 3.337 in × 0.912 in (CIKM single-column banner)
sns.set_theme(style="darkgrid", context="paper", font_scale=0.5)
fig, axes = plt.subplots(1, 3, figsize=(3.337, 0.912), sharey=True)

for di, d in enumerate(DOMAINS):
    ax = axes[di]
    ax.set_facecolor("#EAEAF2")

    x = np.arange(len(GROUPS))
    width = 0.55
    # Non-2-GMM bars: light gray
    bars = ax.bar(x, means[:, di], width,
                  color="#C8C8C8", edgecolor="white", linewidth=0.3, zorder=3)
    # 2-GMM bar (index 0): domain color + dark border (the winner is visually highlighted)
    bars[0].set_color(d["color"])
    bars[0].set_edgecolor("#222222")
    bars[0].set_linewidth(1.0)

    # Value labels on top of bars
    for bar, val in zip(bars, means[:, di]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.010, f"{val:.2f}",
                ha="center", va="bottom", fontsize=5, fontweight="bold",
                color="#222222")

    # X-axis (prior abbreviations) - no rotation needed for short labels
    ax.set_xticks(x)
    ax.set_xticklabels([g["label"] for g in GROUPS], fontsize=6)

    # Subplot title (domain name)
    ax.set_title(d["label_short"], fontsize=8, fontweight="bold",
                 color=d["color"], pad=2)

    ax.tick_params(axis="y", labelsize=6, length=2, width=0.5)
    ax.tick_params(axis="x", length=2, width=0.5)
    # Force ALL y-tick values (matplotlib auto-prunes when figure is short)
    ax.set_yticks([-1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9])
    ax.grid(axis="y", alpha=0.6, linestyle="-", linewidth=0.4, color="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.set_ylim(bottom=-1.55, top=-0.85)

plt.tight_layout()
plt.subplots_adjust(top=0.85, bottom=0.22, wspace=0.10, left=0.06, right=0.98)

out_png = INTERP / "figure4_alt_fit_quality_bar.png"
out_pdf = INTERP / "figure4_alt_fit_quality_bar.pdf"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")

# ============ Summary table ============
print(f"\n=== Per-user mean log p (3 domains × 4 priors) ===")
print(f"{'Prior':<28}", end="")
for d in DOMAINS:
    print(f"{d['label_short']:>10}", end="")
print(f"{'3-dom μ':>10}")
print("-" * 75)
for gi, g in enumerate(GROUPS):
    print(f"{g['label']:<28}", end="")
    for di in range(len(DOMAINS)):
        print(f"{means[gi, di]:>10.4f}", end="")
    overall = float(means[gi].mean())
    print(f"{overall:>10.4f}")

print(f"\n=== Multivariate Gaussian Mixture 相对其他先验的领先 (nats, 3 域平均) ===")
gmm_idx = 0
for gi, g in enumerate(GROUPS[1:], 1):
    delta = float(means[gmm_idx].mean() - means[gi].mean())
    print(f"  vs {g['label']:<28}  +{delta:.3f} nats")
