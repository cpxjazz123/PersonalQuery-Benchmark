#!/usr/bin/env python3
"""论文级三联可解释性图（VADES Fig 3-5 风格）。

输出 3 类图:
1. Before vs After 对比热力图（仅当两版本都有数据时）
2. 三联子图: A=相关系数热力图 / B=SVR 柱状图 / C=t-SNE 2D
3. 每维 Top-1 解读表（JSON + Markdown）

数据策略:
- Baby: gmm-only + gmm+align 两个版本都在，输出 before/after + align 版三联 + align 版 Top-1
- Pet / Grocery: 只有 gmm-only，输出 gmm-only 版三联 + Top-1
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


REPO_ROOT = Path("/fs04/ar57/wenyu")
CATEGORIES = ["Baby_Products", "Pet_Supplies", "Grocery_and_Gourmet_Food"]
FEATURE_BASE = REPO_ROOT / "result" / "personal_query" / "10_complexity_analysis_clause_features"
STAGE1_BASE = REPO_ROOT / "result" / "personal_query" / "01_preference_extraction"
OUTPUT_BASE = FEATURE_BASE / "interpretability_paper"

GMM_ONLY_TAG = "vades_lite_sentence_user_distribution_train10_holdout10_gmm"
GMM_ALIGN_TAG = "vades_lite_sentence_user_distribution_train10_holdout10_gmm_align"

FEATURE_NAMES = [
    "max_dependency_depth", "mean_dependency_depth", "dependency_tree_height",
    "depth_variance", "acl_count", "relcl_count", "ccomp_count", "xcomp_count",
    "advcl_count", "clause_nesting_depth", "mean_dependency_distance",
    "max_dependency_distance", "long_dependency_ratio", "amod_count",
    "advmod_count", "nmod_count", "compound_count", "modifier_density",
    "coordination_count", "max_branching_factor",
]


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_user_profiles(category: str, tag: str) -> tuple[list[str], np.ndarray]:
    path = FEATURE_BASE / category / f"{tag}_user_profiles.jsonl"
    rows = _load_jsonl(path)
    user_ids = [r["user_id"] for r in rows]
    latent_k0 = np.asarray([r["component_mus"][0] for r in rows], dtype=np.float64)
    return user_ids, latent_k0


def _load_user_feature_means(category: str, tag: str) -> dict[str, np.ndarray]:
    path = FEATURE_BASE / category / f"{tag}_sentences.jsonl"
    feat_buf: dict[str, list[np.ndarray]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "features" in row:
                feat_buf[row["user_id"]].append(
                    np.asarray([float(v) for v in row["features"].values()], dtype=np.float64)
                )
    return {uid: np.mean(np.stack(vs, axis=0), axis=0) for uid, vs in feat_buf.items()}


def _load_user_total_reviews(category: str) -> dict[str, int]:
    path = STAGE1_BASE / category / "stage1_filtered_users_reviews.json"
    with path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    return {u["user_id"]: int(u.get("total_products", 0)) for u in d["users"]}


def _load_user_accept_rate(category: str, tag: str) -> dict[str, float]:
    sel_path = FEATURE_BASE / category / f"{tag}_selected_query_records.jsonl"
    rej_path = FEATURE_BASE / category / f"{tag}_rejected_query_records.jsonl"
    sel = _load_jsonl(sel_path)
    rej = _load_jsonl(rej_path)
    sel_cnt: dict[str, int] = defaultdict(int)
    rej_cnt: dict[str, int] = defaultdict(int)
    for r in sel:
        sel_cnt[r["user_id"]] += 1
    for r in rej:
        rej_cnt[r["user_id"]] += 1
    out: dict[str, float] = {}
    for uid in set(sel_cnt) | set(rej_cnt):
        s = sel_cnt[uid]
        rj = rej_cnt[uid]
        out[uid] = s / (s + rj) if (s + rj) > 0 else float("nan")
    return out


def _compute_correlation_matrix(latent: np.ndarray, user_feat: dict, user_ids: list) -> np.ndarray:
    feat_mat = np.stack([user_feat[uid] for uid in user_ids], axis=0)
    corr = np.zeros((20, 20), dtype=np.float64)
    for i in range(20):
        for j in range(20):
            x = latent[:, i]
            y = feat_mat[:, j]
            if np.std(x) < 1e-9 or np.std(y) < 1e-9:
                corr[i, j] = 0.0
            else:
                corr[i, j] = float(np.corrcoef(x, y)[0, 1])
    return corr


def _run_svr(latent: np.ndarray, user_feat: dict, user_ids: list) -> dict:
    from sklearn.svm import SVR
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold

    feat_mat = np.stack([user_feat[uid] for uid in user_ids], axis=0)
    sx = StandardScaler()
    X_pqb = sx.fit_transform(latent)
    pca = PCA(n_components=20)
    X_pca = pca.fit_transform(feat_mat)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    pqb_mse, pca_mse = [], []
    for d in range(20):
        y = feat_mat[:, d]
        sy = StandardScaler()
        y_std = sy.fit_transform(y.reshape(-1, 1)).ravel()
        pqb_dim, pca_dim = [], []
        for tr, te in kf.split(X_pqb):
            svr = SVR(kernel="rbf", C=1.0, gamma="scale")
            svr.fit(X_pqb[tr], y_std[tr])
            pqb_dim.append(float(np.mean((svr.predict(X_pqb[te]) - y_std[te]) ** 2)))
            svr2 = SVR(kernel="rbf", C=1.0, gamma="scale")
            svr2.fit(X_pca[tr], y_std[tr])
            pca_dim.append(float(np.mean((svr2.predict(X_pca[te]) - y_std[te]) ** 2)))
        pqb_mse.append(float(np.mean(pqb_dim)))
        pca_mse.append(float(np.mean(pca_dim)))
    pqb_arr = np.asarray(pqb_mse)
    pca_arr = np.asarray(pca_mse)
    return {
        "pqb_mse": pqb_arr,
        "pca_mse": pca_arr,
        "pqb_wins": int(np.sum(pqb_arr < pca_arr)),
        "pca_wins": int(np.sum(pca_arr < pqb_arr)),
    }


def _run_tsne(latent: np.ndarray, user_ids: list, attr_vals: list) -> tuple:
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    sx = StandardScaler()
    X = sx.fit_transform(latent)
    n = min(3000, len(X))
    rng = np.random.default_rng(42)
    if len(X) > n:
        idx = rng.choice(len(X), n, replace=False)
        Xs = X[idx]
        uids = [user_ids[i] for i in idx]
        vals = [attr_vals[i] for i in idx]
    else:
        Xs = X
        uids = user_ids
        vals = attr_vals
    z = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto").fit_transform(Xs)
    return z, uids, vals, n


# ============ Figure 1: Before/After ============

def generate_before_after(category: str, tag1: str, tag2: str, label1: str, label2: str) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    log(f"[{category}] Loading data for before/after ({label1} vs {label2})...")
    user_ids_1, latent_1 = _load_user_profiles(category, tag1)
    user_ids_2, latent_2 = _load_user_profiles(category, tag2)
    user_feat_1 = _load_user_feature_means(category, tag1)
    user_feat_2 = _load_user_feature_means(category, tag2)

    log(f"  Computing correlation for {label1}...")
    corr1 = _compute_correlation_matrix(latent_1, user_feat_1, user_ids_1)
    log(f"  Computing correlation for {label2}...")
    corr2 = _compute_correlation_matrix(latent_2, user_feat_2, user_ids_2)

    diag1 = float(np.mean(np.diag(corr1)))
    diag2 = float(np.mean(np.diag(corr2)))
    off1 = float(np.mean(corr1 - np.diag(np.diag(corr1))))
    off2 = float(np.mean(corr2 - np.diag(np.diag(corr2))))

    out_dir = OUTPUT_BASE / category
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    for ax, corr, diag, off, label in [
        (axes[0], corr1, diag1, off1, label1),
        (axes[1], corr2, diag2, off2, label2),
    ]:
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            xticklabels=FEATURE_NAMES, yticklabels=[f"latent_{i}" for i in range(20)],
            ax=ax, cbar_kws={"label": "Pearson r"},
        )
        ax.set_xlabel("Feature dim (20-dim syntax style)")
        ax.set_ylabel("Latent dim (PQB)")
        ax.set_title(f"{label}\ndiagonal = {diag:.3f}, off-diag = {off:.3f}, ratio = {diag / (abs(off) + 1e-9):.2f}x",
                     fontsize=12)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.suptitle(f"Latent-Feature Correlation: effect of `latent_align_loss` — {category}",
                 fontsize=15, y=1.02)
    plt.tight_layout()
    out_png = out_dir / "before_after_heatmap.png"
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {out_png}")
    return {
        "category": category,
        "label1": label1, "label2": label2,
        "diag1": diag1, "diag2": diag2,
        "off1": off1, "off2": off2,
        "ratio1": diag1 / (abs(off1) + 1e-9),
        "ratio2": diag2 / (abs(off2) + 1e-9),
        "before_after_png": str(out_png),
    }


# ============ Figure 2: 3-panel composite ============

def generate_3panel(category: str, tag: str, label: str) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    log(f"[{category}/{label}] Loading data...")
    user_ids, latent = _load_user_profiles(category, tag)
    user_feat = _load_user_feature_means(category, tag)
    user_total_reviews = _load_user_total_reviews(category)
    user_accept_rate = _load_user_accept_rate(category, tag)

    log(f"  Panel A: correlation heatmap ({len(user_ids)} users)")
    corr = _compute_correlation_matrix(latent, user_feat, user_ids)
    diag = float(np.mean(np.diag(corr)))
    off = float(np.mean(corr - np.diag(np.diag(corr))))

    log(f"  Panel B: SVR per-dim")
    svr = _run_svr(latent, user_feat, user_ids)

    log(f"  Panel C: t-SNE 2D")
    attr_vals = [user_total_reviews.get(uid, 0) for uid in user_ids]
    z, uids, vals, n_tsne = _run_tsne(latent, user_ids, attr_vals)

    out_dir = OUTPUT_BASE / category
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.35, wspace=0.25)

    ax_a = fig.add_subplot(gs[0, 0])
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        xticklabels=FEATURE_NAMES, yticklabels=[f"latent_{i}" for i in range(20)],
        ax=ax_a, cbar_kws={"label": "Pearson r"},
    )
    ax_a.set_title(f"A. Latent-Feature Correlation\n(diagonal = {diag:.3f}, off-diag = {off:.3f}, ratio = {diag / (abs(off) + 1e-9):.1f}x)",
                   fontsize=12)
    ax_a.set_xlabel("Feature dim")
    ax_a.set_ylabel("Latent dim")
    plt.setp(ax_a.get_xticklabels(), rotation=45, ha="right")

    ax_b = fig.add_subplot(gs[0, 1])
    x = np.arange(20)
    ax_b.bar(x - 0.2, svr["pqb_mse"], 0.4, label=f"PQB (mean={np.mean(svr['pqb_mse']):.3f})", color="steelblue")
    ax_b.bar(x + 0.2, svr["pca_mse"], 0.4, label=f"PCA-20 (mean={np.mean(svr['pca_mse']):.3f})", color="darkorange")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right")
    ax_b.set_ylabel("5-fold CV MSE (std y)")
    ax_b.set_title(f"B. SVR Per-Dim Prediction\n(PQB wins {svr['pqb_wins']}/20, PCA wins {svr['pca_wins']}/20)",
                   fontsize=12)
    ax_b.legend()
    ax_b.grid(axis="y", alpha=0.3)

    ax_c = fig.add_subplot(gs[1, :])
    sc = ax_c.scatter(z[:, 0], z[:, 1], c=vals, cmap="viridis", s=10, alpha=0.7)
    plt.colorbar(sc, ax=ax_c, label="# reviews (stage1)")
    ax_c.set_xlabel("t-SNE 1")
    ax_c.set_ylabel("t-SNE 2")
    ax_c.set_title(f"C. t-SNE 2D Projection of User Latent ({n_tsne} users, colored by #reviews)",
                   fontsize=12)

    fig.suptitle(f"Interpretability Analysis — {category} ({label})",
                 fontsize=16, y=0.995)
    suffix = "align" if "align" in tag else "gmm"
    out_png = out_dir / f"composite_3panel_{suffix}.png"
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {out_png}")
    return {
        "category": category,
        "label": label,
        "n_users": len(user_ids),
        "diag": diag, "off": off,
        "pqb_mse_mean": float(np.mean(svr["pqb_mse"])),
        "pca_mse_mean": float(np.mean(svr["pca_mse"])),
        "pqb_wins": svr["pqb_wins"],
        "pca_wins": svr["pca_wins"],
        "n_users_tsne": n_tsne,
        "composite_png": str(out_png),
    }


# ============ Figure 3: Top-1 interpretation table ============

def generate_top1_table(category: str, tag: str, label: str) -> dict:
    from scipy import stats

    log(f"[{category}/{label}] Generating top-1 table...")
    user_ids, latent = _load_user_profiles(category, tag)
    user_feat = _load_user_feature_means(category, tag)
    feat_mat = np.stack([user_feat[uid] for uid in user_ids], axis=0)
    corr = _compute_correlation_matrix(latent, user_feat, user_ids)

    rows = []
    for i in range(20):
        off_diag = corr[i].copy()
        off_diag[i] = -np.inf
        best_off_idx = int(np.argmax(off_diag))
        r_diag, p_diag = stats.pearsonr(latent[:, i], feat_mat[:, i])
        rows.append({
            "latent_dim": i,
            "aligned_feature": FEATURE_NAMES[i],
            "r_diagonal": float(r_diag),
            "p_diagonal": float(p_diag),
            "offdiag_best_feature": FEATURE_NAMES[best_off_idx],
            "r_offdiag": float(corr[i, best_off_idx]),
            "dominance": float(corr[i, i] - corr[i, best_off_idx]),
        })

    n_diag_wins = int(np.sum([r["r_diagonal"] > r["r_offdiag"] for r in rows]))
    diag_mean = float(np.mean([r["r_diagonal"] for r in rows]))
    off_mean = float(np.mean([r["r_offdiag"] for r in rows]))

    out_dir = OUTPUT_BASE / category
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "align" if "align" in tag else "gmm"
    out_json = out_dir / f"top1_table_{suffix}.json"
    summary = {
        "category": category,
        "tag": tag,
        "label": label,
        "n_users": len(user_ids),
        "n_diagonal_wins": n_diag_wins,
        "diag_mean": diag_mean,
        "offdiag_best_mean": off_mean,
        "ratio": diag_mean / (abs(off_mean) + 1e-9),
        "rows": rows,
    }
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [f"# Top-1 Interpretation Table — {category} ({label})", ""]
    md.append(f"- **N users**: {len(user_ids)}")
    md.append(f"- **Diagonal mean r**: {diag_mean:.3f}")
    md.append(f"- **Off-diag best mean r**: {off_mean:.3f}")
    md.append(f"- **Diagonal wins**: {n_diag_wins}/20 dims")
    md.append(f"- **Ratio (diag / |off-diag|)**: {diag_mean / (abs(off_mean) + 1e-9):.2f}x")
    md.append("")
    md.append("| Latent dim | Aligned feature | r (diagonal) | p-value | Off-diag best | r (off-diag) | Dominance |")
    md.append("|---|---|---|---|---|---|---|")
    for r in rows:
        sig = "***" if r["p_diagonal"] < 0.001 else "**" if r["p_diagonal"] < 0.01 else "*" if r["p_diagonal"] < 0.05 else ""
        md.append(
            f"| latent_{r['latent_dim']} | `{r['aligned_feature']}` | "
            f"**{r['r_diagonal']:.3f}**{sig} | {r['p_diagonal']:.2e} | "
            f"`{r['offdiag_best_feature']}` | {r['r_offdiag']:.3f} | "
            f"{r['dominance']:+.3f} |"
        )
    out_md = out_dir / f"top1_table_{suffix}.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    log(f"  Saved: {out_json}")
    log(f"  Saved: {out_md}")
    return summary


def main() -> None:
    log("=" * 60)
    log("Paper-style figures for interpretability (VADES Fig 3-5)")
    log("=" * 60)

    all_results = {"before_after": [], "composite": [], "top1_table": []}

    for cat in CATEGORIES:
        log(f"\n{'='*60}\n>>> {cat}\n{'='*60}")
        has_only = (FEATURE_BASE / cat / f"{GMM_ONLY_TAG}_user_profiles.jsonl").exists()
        has_align = (FEATURE_BASE / cat / f"{GMM_ALIGN_TAG}_user_profiles.jsonl").exists()
        log(f"  has gmm-only: {has_only}, has gmm+align: {has_align}")

        if has_only and has_align:
            log("\n--- Figure 1: Before/After heatmap comparison ---")
            try:
                all_results["before_after"].append(
                    generate_before_after(cat, GMM_ONLY_TAG, GMM_ALIGN_TAG, "gmm-only", "gmm+align")
                )
            except Exception as e:
                log(f"  FAILED: {e}")

        primary_tag = GMM_ALIGN_TAG if has_align else (GMM_ONLY_TAG if has_only else None)
        primary_label = "gmm+align" if has_align else "gmm-only"
        if primary_tag is None:
            log(f"  No data found for {cat}, skipping")
            continue

        log(f"\n--- Figure 2: 3-panel composite ({primary_label}) ---")
        try:
            all_results["composite"].append(generate_3panel(cat, primary_tag, primary_label))
        except Exception as e:
            log(f"  FAILED: {e}")

        log(f"\n--- Figure 3: Top-1 interpretation table ({primary_label}) ---")
        try:
            all_results["top1_table"].append(generate_top1_table(cat, primary_tag, primary_label))
        except Exception as e:
            log(f"  FAILED: {e}")

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    summary_file = OUTPUT_BASE / "paper_figures_summary.json"
    summary_file.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\nSummary: {summary_file}")

    log("\n" + "=" * 90)
    log("=== Paper Figures Summary (3-panel metrics) ===")
    log("=" * 90)
    log(f"{'Category':<25} | {'Diag mean':>10} | {'Off-diag':>10} | {'Ratio':>8} | {'PQB MSE':>10} | {'PCA MSE':>10} | {'PQB wins':>9}")
    log("-" * 90)
    for r in all_results["composite"]:
        log(f"{r['category']:<25} | {r['diag']:>10.3f} | {r['off']:>10.3f} | {r['diag']/(abs(r['off'])+1e-9):>7.2f}x | {r['pqb_mse_mean']:>10.3f} | {r['pca_mse_mean']:>10.3f} | {r['pqb_wins']:>5}/20")
    log("=" * 90)


if __name__ == "__main__":
    main()
