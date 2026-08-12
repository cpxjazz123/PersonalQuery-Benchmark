"""Figure 4 alt (3-domain averaged): per-user encoder-based mean log p under each
prior vs per-user latent variance, with 3 Amazon domains averaged per prior.

For each (prior, domain):
  1. Load per-user (x = latent variance, y = mean log p) data
  2. Bin x into 25 equal-width bins covering all 3 domains
  3. Compute bin-mean y per (domain, prior, bin)
  4. Average across 3 domains → 1 cross-domain mean trend per prior
  5. Add ±1σ cross-domain std band

Style: VADES-paper darkgrid (light blue panel) with bold short title.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/fs04/ar57/wenyu/result/personal_query/10_complexity_analysis_clause_features")
INTERP = BASE / "interpretability_paper"

DOMAINS = [
    {"key": "Baby_Products",            "var_fname": "user_latent_variance_align_raw.json", "marker": "o", "label_short": "Baby"},
    {"key": "Pet_Supplies",             "var_fname": "user_latent_variance_gmm_raw.json",   "marker": "s", "label_short": "Pet"},
    {"key": "Grocery_and_Gourmet_Food", "var_fname": "user_latent_variance_gmm_raw.json",   "marker": "^", "label_short": "Grocery"},
]

GROUPS = [
    {"key": "gmm_2",            "label": "2-GMM (Gaussian)",     "color": "#4C72B0"},
    {"key": "student_t",        "label": "Student-t",            "color": "#DD8452"},
    {"key": "laplace",          "label": "Laplace",              "color": "#55A467"},
    {"key": "logistic",         "label": "Logistic",             "color": "#C44E52"},
    {"key": "student_t_gmm",    "label": "Student-t GMM (K=2)",  "color": "#8172B3"},
    {"key": "student_t_strict", "label": "Student-t strict VIB", "color": "#937860"},
]

N_BINS = 25


def _load_json(path: Path):
    with path.open() as f:
        return json.load(f)


# ============ Load per-(domain, prior) per-user data ============
per_domain = {}
for d in DOMAINS:
    fq = _load_json(INTERP / d["key"] / "per_user_fit_quality.json")
    var = _load_json(INTERP / d["key"] / d["var_fname"])
    per_domain[d["key"]] = {
        "fq": fq, "var": var,
        "n_users": len(fq["groups"]["gmm_2"]["per_user_mean_logp"]),
    }
    print(f"  {d['key']}: {per_domain[d['key']]['n_users']} users in fit_quality")

# ============ Bin by variance (one shared bin grid across 3 domains) ============
all_vars = []
for d in DOMAINS:
    all_vars.extend([v for v in per_domain[d["key"]]["var"].values() if v > 0])
all_vars = np.asarray(all_vars, dtype=np.float64)
X_MIN, X_MAX = 0.0, float(np.quantile(all_vars, 0.99)) * 1.05
bin_edges = np.linspace(X_MIN, X_MAX, N_BINS + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
print(f"\nShared X axis (q1-q99 across 3 domains): [{X_MIN:.2f}, {X_MAX:.2f}]")
print(f"  {N_BINS} equal-width bins, centers from {bin_centers[0]:.2f} to {bin_centers[-1]:.2f}")

# bin_mean_y[group_key] = (n_domains, N_BINS) — for std band
bin_mean_y = {g["key"]: np.full((len(DOMAINS), N_BINS), np.nan) for g in GROUPS}
bin_count = np.zeros((len(DOMAINS), N_BINS), dtype=np.int64)

for di, d in enumerate(DOMAINS):
    var_dict = per_domain[d["key"]]["var"]
    for gi, g in enumerate(GROUPS):
        per_user_logp = per_domain[d["key"]]["fq"]["groups"][g["key"]]["per_user_mean_logp"]
        xs, ys = [], []
        for uid, y in per_user_logp.items():
            x = var_dict.get(uid, 0.0)
            if x > 0 and X_MIN <= x <= X_MAX:
                xs.append(x)
                ys.append(y)
        if not xs:
            continue
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        bin_idx = np.clip(np.searchsorted(bin_edges, xs, side="right") - 1, 0, N_BINS - 1)
        for b in range(N_BINS):
            mask = bin_idx == b
            if mask.sum() >= 3:  # need at least 3 users to compute stable mean
                bin_mean_y[g["key"]][di, b] = float(np.mean(ys[mask]))
        bin_count[di] += np.bincount(bin_idx, minlength=N_BINS).astype(np.int64)

# Cross-domain mean & std (ignore NaN, ignore bins where ANY domain has <3 users)
y_min, y_max = float("inf"), float("-inf")
for g in GROUPS:
    arr = bin_mean_y[g["key"]]
    # bins where all 3 domains have valid mean
    valid = np.all(~np.isnan(arr), axis=0) & (np.sum(~np.isnan(arr), axis=0) == len(DOMAINS))
    valid_arr = arr[:, valid]
    y_min = min(y_min, float(np.nanmin(valid_arr)))
    y_max = max(y_max, float(np.nanmax(valid_arr)))
    print(f"  {g['label']}: {valid.sum()}/{N_BINS} bins have all 3 domains")
print(f"\nShared Y axis (across 6 priors × 3 domains): [{y_min:.2f}, {y_max:.2f}]")

Y_PAD = 0.05
YLIM = (y_min - Y_PAD, y_max + Y_PAD)

# ============ Plot ============
import seaborn as sns
sns.set_theme(style="darkgrid", context="paper", font_scale=1.1)

fig, ax = plt.subplots(figsize=(11, 7))
ax.set_facecolor("#EAEAF2")

# 1) Per-domain binned-mean markers (small, semi-transparent — visual analog of
#    single-domain's scatter, but binned to keep the figure readable across 3 doms)
trend_summary = []
for g in GROUPS:
    arr = bin_mean_y[g["key"]]  # (3, N_BINS)
    valid = ~np.isnan(arr).any(axis=0)
    xs = bin_centers[valid]
    if xs.size < 2:
        continue
    arr_valid = arr[:, valid]  # (3, n_valid) — slice first to avoid empty-slice warnings
    with np.errstate(invalid="ignore"):
        ys_mean = np.nanmean(arr_valid, axis=0)
        ys_std = np.nanstd(arr_valid, axis=0)
    color = g["color"]

    # 1a) Per-domain markers (one marker shape per domain, low alpha)
    for di, d in enumerate(DOMAINS):
        ax.scatter(xs, arr_valid[di],
                   s=10, alpha=0.45, c=color, marker=d["marker"],
                   edgecolors="none", rasterized=True, zorder=3)

    # 1b) Cross-domain averaged trend line (solid, opaque) — same linewidth as single-domain
    ax.plot(xs, ys_mean, c=color, linewidth=2.0, label=None, zorder=4)

    # 1c) ±1σ cross-domain band
    ax.fill_between(xs, ys_mean - ys_std, ys_mean + ys_std, color=color, alpha=0.12, linewidth=0, zorder=2)

    med_logp = float(np.median(ys_mean))
    if np.std(xs) > 1e-9:
        coef = np.polyfit(xs, ys_mean, 1)
        trend_summary.append((g, color, coef, med_logp))

# 2) Per-domain summary numbers (for console)
for di, d in enumerate(DOMAINS):
    grp_means = []
    for g in GROUPS:
        arr = bin_mean_y[g["key"]]
        v = arr[di, ~np.isnan(arr[di])]
        grp_means.append(np.nanmean(v) if v.size else np.nan)
    print(f"  {d['key']} ({d['marker']}): 3-domain mean of all priors = "
          f"{np.nanmean(grp_means):.4f}")

# ============ Legend (VADES-style: 2 cols at the bottom, ranked by mean log p) ============
from matplotlib.lines import Line2D

trend_summary.sort(key=lambda t: -t[3])  # highest (least negative) on top
legend_elems = []
for g, color, coef, med_logp in trend_summary:
    legend_elems.append(Line2D(
        [0], [0], color=color, linewidth=2.5,
        label=f"{g['label']}  (μ = {med_logp:.2f}, slope = {coef[0]:.3f})",
    ))
# Trailing domain legend entries (so the reader sees what marker = what domain)
for d in DOMAINS:
    legend_elems.append(Line2D(
        [0], [0], marker=d["marker"], color="#666666", linewidth=0,
        markersize=7, markerfacecolor="#666666", markeredgecolor="#666666",
        label=f"   {d['label_short']}  (binned mean)",
    ))

leg = ax.legend(handles=legend_elems,
                loc="upper center", bbox_to_anchor=(0.5, -0.18),
                ncol=2, fontsize=9, framealpha=0.95,
                title="Prior (ranked by mean log p, ↑ = more realistic) · per-domain bins",
                title_fontsize=9, labelspacing=0.8,
                handlelength=2.5, columnspacing=1.6)
leg.get_title().set_fontweight("bold")

ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(*YLIM)
ax.set_xlabel("per-user latent variance", fontsize=11)
ax.set_ylabel("per-user mean log p  (nats)", fontsize=11)
ax.set_title("Per-user fit quality vs. per-user latent variance",
             fontsize=12, fontweight="bold", pad=12)
ax.grid(alpha=0.6, linestyle="-", linewidth=0.7, color="white")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#888888")
ax.spines["bottom"].set_color("#888888")

plt.tight_layout()
plt.subplots_adjust(bottom=0.30)

out = INTERP / "figure4_alt_fit_quality_3domain.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.savefig(INTERP / "figure4_alt_fit_quality_3domain.pdf", bbox_inches="tight")
print(f"\nSaved: {out}")
print(f"Saved: {INTERP / 'figure4_alt_fit_quality_3domain.pdf'}")

# ============ Print summary table ============
print("\n=== 3-domain averaged per-user mean log p (Baby / Pet / Grocery) ===")
print(f"{'Prior':<25} | {'Baby μ':>8} {'Pet μ':>8} {'Grocery μ':>11} | {'3-dom μ':>9} {'3-dom σ':>9}")
print("-" * 80)
for g in GROUPS:
    arr = bin_mean_y[g["key"]]
    dom_means = []
    for di in range(len(DOMAINS)):
        v = arr[di, ~np.isnan(arr[di])]
        dom_means.append(np.nanmean(v) if v.size else np.nan)
    valid_mask = ~np.isnan(arr).any(axis=0)
    all_means = arr[:, valid_mask]
    flat = all_means.flatten()
    overall_mean = float(np.mean(flat))
    overall_std = float(np.std(flat))
    print(f"{g['label']:<25} | "
          f"{dom_means[0]:>8.4f} {dom_means[1]:>8.4f} {dom_means[2]:>11.4f} | "
          f"{overall_mean:>9.4f} {overall_std:>9.4f}")
