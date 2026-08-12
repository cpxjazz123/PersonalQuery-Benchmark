"""Figure 4 (alt, "realistic modeling"): per-user encoder-based mean log p under
each prior vs per-user latent variance.

The y-axis is the proper "fit quality" metric: for each user, score their
encoder-output sentence latents z_q under their learned prior, then average
per-dim. Higher (less negative) = better fit.

Difference from the original Figure 4 (alignment):
  Original  → disentanglement:   i-th latent dim correlates with i-th feature
  Alt       → realistic fit:     per-user log p(z_q | prior) vs per-user variance

Two modes (toggle MODE below):
  "single"   → one domain (Baby_Products), per-user scatter + trend line
  "3domain"  → 3 Amazon domains. Each prior is plotted as 3 per-domain scatters
               (one marker shape per domain: o=Baby, s=Pet, ^=Grocery) so each
               domain's band stays distinct. The trend line is fit on the
               cross-domain combined data.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ============ Configuration ============
MODE = "3domain"   # "single" or "3domain"

BASE = Path("/fs04/ar57/wenyu/result/personal_query/10_complexity_analysis_clause_features")
INTERP = BASE / "interpretability_paper"

CATEGORY = "Baby_Products"  # used only in "single" mode

DOMAINS = [
    {"key": "Baby_Products",            "var_fname": "user_latent_variance_align_raw.json", "label_short": "Baby",     "marker": "o"},
    {"key": "Pet_Supplies",             "var_fname": "user_latent_variance_gmm_raw.json",   "label_short": "Pet",      "marker": "s"},
    {"key": "Grocery_and_Gourmet_Food", "var_fname": "user_latent_variance_gmm_raw.json",   "label_short": "Grocery",  "marker": "^"},
]

GROUPS = [
    {"key": "gmm_2",            "label": "2-GMM (Gaussian)",     "color": "#4C72B0"},
    {"key": "student_t",        "label": "Student-t",            "color": "#DD8452"},
    {"key": "laplace",          "label": "Laplace",              "color": "#55A467"},
    {"key": "logistic",         "label": "Logistic",             "color": "#C44E52"},
]


# ============ Helpers ============
def _load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def _gather_per_user(domains: list[dict], g_key: str) -> tuple[np.ndarray, np.ndarray]:
    """For one prior, gather (var, logp) across the given domains. Drops users with var<=0."""
    xs_all, ys_all = [], []
    for d in domains:
        fq = _load_json(INTERP / d["key"] / "per_user_fit_quality.json")
        var = _load_json(INTERP / d["key"] / d["var_fname"])
        per_user_logp = fq["groups"][g_key]["per_user_mean_logp"]
        for uid, y in per_user_logp.items():
            if uid in var and var[uid] > 0:
                xs_all.append(var[uid])
                ys_all.append(y)
    return np.asarray(xs_all, dtype=np.float64), np.asarray(ys_all, dtype=np.float64)


# ============ Data loading (returns IDENTICAL format for both modes) ============
def _load(mode: str) -> tuple[dict, tuple[float, float], tuple[float, float], list[dict]]:
    """Returns (per_group, XLIM, YLIM, per_domain_data).
    per_group[key] = {var, logp, label, color} — combined across selected domains (used for axis range + trend line).
    per_domain_data[i] = {key, fq, var, marker, label_short} — for per-domain marker scatter.
    """
    if mode == "single":
        domains = [{"key": CATEGORY, "var_fname": "user_latent_variance_align_raw.json",
                    "label_short": CATEGORY, "marker": "o"}]
    else:
        domains = DOMAINS

    per_domain_data = []
    for d in domains:
        fq = _load_json(INTERP / d["key"] / "per_user_fit_quality.json")
        var = _load_json(INTERP / d["key"] / d["var_fname"])
        per_domain_data.append({"key": d["key"], "fq": fq, "var": var,
                                "marker": d["marker"], "label_short": d["label_short"]})

    per_group = {}
    all_logp_q01 = float("inf")
    all_logp_q99 = float("-inf")
    all_var_q99 = float("-inf")
    for g in GROUPS:
        var_arr, logp = _gather_per_user(domains, g["key"])
        per_group[g["key"]] = {"var": var_arr, "logp": logp, "label": g["label"], "color": g["color"]}
        all_logp_q01 = min(all_logp_q01, float(np.quantile(logp, 0.01)))
        all_logp_q99 = max(all_logp_q99, float(np.quantile(logp, 0.99)))
        all_var_q99 = max(all_var_q99, float(np.quantile(var_arr, 0.99)))
        print(f"  {g['label']}: n={len(logp)} (from {len(domains)} domain(s)), "
              f"median log p = {float(np.median(logp)):.4f}")

    XLIM = (0.0, all_var_q99 * 1.05)
    YLIM = (all_logp_q01 - 0.05, all_logp_q99 + 0.05)
    return per_group, XLIM, YLIM, per_domain_data


# ============ Plot (mode-agnostic) ============
import seaborn as sns
sns.set_theme(style="darkgrid", context="paper", font_scale=1.1)

per_group, XLIM, YLIM, per_domain_data = _load(MODE)
print(f"\nShared X axis (q1-q99 across groups): {XLIM}")
print(f"Shared Y axis (q1-q99 across groups): {YLIM}")

fig, ax = plt.subplots(figsize=(11, 7))
ax.set_facecolor("#EAEAF2")

# Sub-sample per (prior, domain) — high N + low alpha → soft VADES haze
N_SAMPLE = 5000
rng = np.random.default_rng(42)

trend_lines = []
for g in GROUPS:
    color = g["color"]
    xs_all, ys_all = [], []  # for the cross-domain trend line

    # Per-domain scatter (3 marker shapes in 3domain mode, 1 in single)
    for d_data in per_domain_data:
        var_dict = d_data["var"]
        per_user_logp = d_data["fq"]["groups"][g["key"]]["per_user_mean_logp"]
        xs_d, ys_d = [], []
        for uid, y in per_user_logp.items():
            if uid in var_dict and var_dict[uid] > 0:
                xs_d.append(var_dict[uid])
                ys_d.append(y)
        if not xs_d:
            continue
        xs_d = np.asarray(xs_d)
        ys_d = np.asarray(ys_d)
        if len(xs_d) > N_SAMPLE:
            idx = rng.choice(len(xs_d), N_SAMPLE, replace=False)
            xs_d, ys_d = xs_d[idx], ys_d[idx]
        ax.scatter(xs_d, ys_d, s=5, alpha=0.10, c=color, marker=d_data["marker"],
                   edgecolors="none", rasterized=True, label=None)
        xs_all.append(xs_d)
        ys_all.append(ys_d)

    if not xs_all:
        continue
    xs_combined = np.concatenate(xs_all)
    ys_combined = np.concatenate(ys_all)

    # Trend line — solid, opaque, fit on cross-domain combined data
    if np.std(xs_combined) > 1e-9 and np.std(ys_combined) > 1e-9:
        coef = np.polyfit(xs_combined, ys_combined, 1)
        xs = np.linspace(XLIM[0], XLIM[1], 100)
        ys = np.polyval(coef, xs)
        ax.plot(xs, ys, c=color, alpha=1.0, linewidth=2.0)
        med_logp = float(np.median(ys_combined))
        trend_lines.append((g, color, coef, med_logp))

# ============ Legend (top-right, 4 priors + per-domain markers in 3domain mode) ============
from matplotlib.lines import Line2D

trend_lines.sort(key=lambda t: -t[3])  # highest (least negative) on top
legend_elems = []
for g, color, _coef, _med_logp in trend_lines:
    legend_elems.append(Line2D(
        [0], [0], color=color, linewidth=2.5,
        label=g["label"],
    ))

if MODE == "3domain":
    # Add 3 domain marker entries (in light gray to visually separate from priors)
    for d_data in per_domain_data:
        legend_elems.append(Line2D(
            [0], [0], marker=d_data["marker"], color="#777777", linewidth=0,
            markersize=7, markerfacecolor="#777777", markeredgecolor="#777777",
            label=f"{d_data['label_short']}",
        ))

leg = ax.legend(handles=legend_elems,
                loc="upper right", fontsize=10, framealpha=0.95,
                labelspacing=0.7, handlelength=2.2,
                borderpad=0.6)
leg.get_title().set_fontweight("bold")

ax.set_xlim(*XLIM)
ax.set_ylim(*YLIM)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("")  # no title — axes + legend are self-explanatory

ax.grid(alpha=0.6, linestyle="-", linewidth=0.7, color="white")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#888888")
ax.spines["bottom"].set_color("#888888")

plt.tight_layout()

# ============ Save ============
if MODE == "single":
    out_png = INTERP / "figure4_alt_fit_quality_merged.png"
    out_pdf = INTERP / "figure4_alt_fit_quality_merged.pdf"
elif MODE == "3domain":
    out_png = INTERP / "figure4_alt_fit_quality_3domain.png"
    out_pdf = INTERP / "figure4_alt_fit_quality_3domain.pdf"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
print(f"\nSaved: {out_png}")
print(f"Saved: {out_pdf}")

# ============ Summary table ============
print(f"\n=== Per-user encoder-based mean log p under each prior ({MODE}) ===")
if MODE == "single":
    print(f"{'Group':<25} {'median':>10} {'q10':>10} {'q90':>10} {'mean':>10}")
    for g in GROUPS:
        logp = per_group[g["key"]]["logp"]
        print(f"{g['label']:<25} {np.median(logp):>10.4f} {np.quantile(logp, 0.1):>10.4f} "
              f"{np.quantile(logp, 0.9):>10.4f} {np.mean(logp):>10.4f}")
elif MODE == "3domain":
    # Per-domain breakdown (3 domains × 6 priors)
    print(f"{'Prior':<25} | {'Baby μ':>8} {'Pet μ':>8} {'Grocery μ':>11} | {'3-dom μ':>9} {'3-dom σ':>9}")
    print("-" * 80)
    for g in GROUPS:
        dom_means = []
        for d in DOMAINS:
            fq = _load_json(INTERP / d["key"] / "per_user_fit_quality.json")
            per_user_logp = fq["groups"][g["key"]]["per_user_mean_logp"]
            vals = np.array(list(per_user_logp.values()), dtype=np.float64)
            dom_means.append(float(np.mean(vals)))
        # 3-domain mean over combined pool
        logp = per_group[g["key"]]["logp"]
        var_arr = per_group[g["key"]]["var"]
        print(f"{g['label']:<25} | "
              f"{dom_means[0]:>8.4f} {dom_means[1]:>8.4f} {dom_means[2]:>11.4f} | "
              f"{float(np.mean(logp)):>9.4f} {float(np.std(logp)):>9.4f}")
