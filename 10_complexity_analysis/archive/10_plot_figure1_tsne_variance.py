"""Figure 1 (VADES-style): t-SNE 2D projection of user latent means with dot size ∝
per-user latent variance (trace of empirical per-sentence latent covariance).

Combines all 3 Amazon domains; color = domain. Mimics VADES paper Figure 1.

The empirical per-user variance is loaded from the precomputed
`interpretability_paper/<cat>/user_latent_variance_*_raw.json` files
(per-user trace of covariance over sentences, not the GMM's learned prior logvar).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

BASE = Path("/fs04/ar57/wenyu/result/personal_query/10_complexity_analysis_clause_features")
INTERP = BASE / "interpretability_paper"

DOMAINS = [
    ("Baby_Products",            "user_latent_variance_align_raw.json", "#1f77b4"),
    ("Pet_Supplies",             "user_latent_variance_gmm_raw.json",   "#d62728"),
    ("Grocery_and_Gourmet_Food", "user_latent_variance_gmm_raw.json",   "#2ca02c"),
]

TAG = "vades_lite_sentence_user_distribution_train10_holdout10_gmm"


def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_user_latent(cat: str) -> tuple[np.ndarray, list[str]]:
    """Latent = first GMM component's mu (matches interpretability convention)."""
    profile_path = BASE / cat / f"{TAG}_user_profiles.jsonl"
    rows = _load_jsonl(profile_path)
    latents, uids = [], []
    for r in rows:
        if "component_mus" in r:
            latents.append(np.asarray(r["component_mus"][0], dtype=np.float64))
            uids.append(r["user_id"])
        elif "user_mu" in r:
            latents.append(np.asarray(r["user_mu"], dtype=np.float64))
            uids.append(r["user_id"])
    return np.stack(latents, axis=0), uids


def _load_variance(cat: str, fname: str) -> dict[str, float]:
    p = INTERP / cat / fname
    with p.open() as f:
        return json.load(f)


# Per-domain subsample (t-SNE is O(n^2); 3k per domain is plenty)
MAX_PER_DOMAIN = 3000
rng = np.random.default_rng(42)

print("Loading per-domain latents and variances...")
all_latents, all_vars, all_colors, all_domains, all_uids = [], [], [], [], []

for cat, var_fname, color in DOMAINS:
    L, uids = _load_user_latent(cat)
    var_dict = _load_variance(cat, var_fname)
    v = np.asarray([var_dict.get(uid, 0.0) for uid in uids], dtype=np.float64)
    keep = v > 0
    L, v, uids = L[keep], v[keep], [u for u, k in zip(uids, keep) if k]
    if len(L) > MAX_PER_DOMAIN:
        idx = rng.choice(len(L), MAX_PER_DOMAIN, replace=False)
        L, v, uids = L[idx], v[idx], [uids[i] for i in idx]
    print(f"  {cat}: {len(L)} users, variance median={np.median(v):.2f}, "
          f"p99={np.quantile(v, 0.99):.2f}, max={v.max():.2f}, max/median={v.max()/np.median(v):.2f}")
    all_latents.append(L)
    all_vars.append(v)
    all_colors.extend([color] * len(L))
    all_domains.extend([cat] * len(L))
    all_uids.extend(uids)

Z = np.concatenate(all_latents, axis=0)
V_all = np.concatenate(all_vars, axis=0)
print(f"Combined: {Z.shape[0]} users, {Z.shape[1]} latent dims")

Xs = StandardScaler().fit_transform(Z)
print("Running t-SNE (perplexity=30)...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto")
XY = tsne.fit_transform(Xs)
print(f"  t-SNE done. XY shape: {XY.shape}")

# Log-scale the size: heavy-tail users (Pet's max ≈ 462) stand out
log_V = np.log10(V_all)
log_V_norm = (log_V - log_V.min()) / (log_V.max() - log_V.min() + 1e-9)
sizes = 15 + 180 * log_V_norm

fig, ax = plt.subplots(figsize=(11, 8))
for cat, _, color in DOMAINS:
    mask = np.array([d == cat for d in all_domains])
    ax.scatter(
        XY[mask, 0], XY[mask, 1],
        s=sizes[mask], c=color, alpha=0.55,
        edgecolors="none", label=cat, rasterized=True,
    )

top_k = 30
top_idx = np.argsort(V_all)[-top_k:]
ax.scatter(XY[top_idx, 0], XY[top_idx, 1],
           marker="*", s=160, facecolors="black", edgecolors="yellow", linewidth=1.0,
           label=f"top-{top_k} highest-variance users", zorder=5)

top1 = np.argsort(V_all)[-1]
ax.annotate(f"variance = {V_all[top1]:.1f} ({all_domains[top1]})",
            xy=(XY[top1, 0], XY[top1, 1]),
            xytext=(10, 10), textcoords="offset points",
            fontsize=9, color="black", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="black"))

info = (f"t-SNE perplexity=30, init=pca, seed=42\n"
        f"Dot size ∝ log10(per-user latent variance)\n"
        f"max subsample = {MAX_PER_DOMAIN} users/domain")
ax.text(0.02, 0.98, info, transform=ax.transAxes, ha="left", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray", alpha=0.95),
        family="monospace")

ax.set_title("Figure 1 (VADES-style): User Latent 2D Map with Variance-as-Size\n"
             "3 Amazon domains — Pet's top-variance user has 13.7× median (heavy tail)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("t-SNE dim 1", fontsize=10)
ax.set_ylabel("t-SNE dim 2", fontsize=10)
ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
ax.grid(alpha=0.3, linestyle="--")

plt.tight_layout()
out = INTERP / "figure1_tsne_variance.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.savefig(INTERP / "figure1_tsne_variance.pdf", bbox_inches="tight")
print(f"Saved: {out}")
print(f"Saved: {INTERP / 'figure1_tsne_variance.pdf'}")

print("\n=== Per-user latent variance (trace of empirical covariance) ===")
print(f"{'Domain':<25} {'median':>10} {'p99':>10} {'max':>10} {'max/median':>12}")
for cat, var_fname, _ in DOMAINS:
    v = _load_variance(cat, var_fname)
    arr = np.array(list(v.values()))
    print(f"{cat:<25} {np.median(arr):>10.2f} {np.quantile(arr,0.99):>10.2f} "
          f"{arr.max():>10.2f} {arr.max()/np.median(arr):>12.2f}")
