#!/usr/bin/env python3
"""3 个可解释性实验 - 借鉴 VADES 论文 Figure 3-5 的设计。

实验 1: 嵌入-特征相关系数热力图 (VADES Figure 3)
实验 2: 风格特征预测 SVR 回归 (VADES Table 3)
实验 3: T-SNE 2D 投影 + 用户属性着色 (VADES Figure 5)

数据源:
- user_profiles.jsonl: 每用户 component_mus [2, 20] (GMM k=0)
- sentences.jsonl: 每用户的 20 维句法特征, 可聚合用户均值
- selected/rejected_query_records.jsonl: 用户 accept_rate
- stage1: 用户 total_reviews

输出:
- result/.../<CAT>/interpretability/correlation_heatmap.png
- result/.../<CAT>/interpretability/svr_per_dim_mse.json
- result/.../<CAT>/interpretability/tsne_2d_<attr>.png
- result/.../interpretability/interpretability_all_categories.json (聚合)
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
import os
GMM_TAG = os.environ.get("INTERP_GMM_TAG", "vades_lite_sentence_user_distribution_train10_holdout10_gmm")

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


def _load_user_profiles(category: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Returns (user_ids, latent_k0 [U, 20], mix_logits [U, 2])."""
    path = FEATURE_BASE / category / f"{GMM_TAG}_user_profiles.jsonl"
    rows = _load_jsonl(path)
    user_ids = [r["user_id"] for r in rows]
    latent_k0 = np.asarray([r["component_mus"][0] for r in rows], dtype=np.float64)  # [U, 20]
    mix = np.asarray([r["mix_logits"] for r in rows], dtype=np.float64)  # [U, 2]
    return user_ids, latent_k0, mix


def _load_user_feature_means(category: str) -> dict[str, np.ndarray]:
    """Aggregate per-sentence 20-dim features to per-user mean."""
    path = FEATURE_BASE / category / f"{GMM_TAG}_sentences.jsonl"
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
    """From stage1: count reviews per user_id."""
    path = STAGE1_BASE / category / "stage1_filtered_users_reviews.json"
    with path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    return {u["user_id"]: int(u.get("total_products", 0)) for u in d["users"]}


def _load_user_accept_rate(category: str) -> dict[str, float]:
    """Per-user accept_rate = selected / (selected + rejected)."""
    sel_path = FEATURE_BASE / category / f"{GMM_TAG}_selected_query_records.jsonl"
    rej_path = FEATURE_BASE / category / f"{GMM_TAG}_rejected_query_records.jsonl"
    sel = _load_jsonl(sel_path)
    rej = _load_jsonl(rej_path)
    cnt: dict[str, int] = defaultdict(int)
    for r in sel:
        cnt[r["user_id"]] += 1
    for r in rej:
        cnt[r["user_id"]] += 0  # placeholder; we need total
    out: dict[str, float] = {}
    sel_cnt: dict[str, int] = defaultdict(int)
    rej_cnt: dict[str, int] = defaultdict(int)
    for r in sel:
        sel_cnt[r["user_id"]] += 1
    for r in rej:
        rej_cnt[r["user_id"]] += 1
    for uid in set(sel_cnt) | set(rej_cnt):
        s = sel_cnt[uid]
        rj = rej_cnt[uid]
        out[uid] = s / (s + rj) if (s + rj) > 0 else float("nan")
    return out


def _experiment1_correlation_heatmap(
    category: str, latent: np.ndarray, user_feat: dict[str, np.ndarray], user_ids: list[str]
) -> dict:
    """Compute 20x20 correlation between latent dim and user-level 20-dim feature mean."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    feat_mat = np.stack([user_feat[uid] for uid in user_ids], axis=0)  # [U, 20]
    corr = np.zeros((20, 20), dtype=np.float64)
    for i in range(20):
        for j in range(20):
            x = latent[:, i]
            y = feat_mat[:, j]
            if np.std(x) < 1e-9 or np.std(y) < 1e-9:
                corr[i, j] = 0.0
            else:
                corr[i, j] = float(np.corrcoef(x, y)[0, 1])

    out_dir = FEATURE_BASE / category / f"interpretability{os.environ.get('INTERP_OUTPUT_SUFFIX', '')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        xticklabels=FEATURE_NAMES, yticklabels=[f"latent_{i}" for i in range(20)],
        ax=ax, cbar_kws={"label": "Pearson correlation"},
    )
    ax.set_xlabel("User-level feature mean (20-dim syntax style)")
    ax.set_ylabel("PQB latent dim")
    ax.set_title(f"Latent-feature correlation — {category} (GMM k=0)\n"
                 f"diagonal mean = {np.mean(np.diag(corr)):.3f}, off-diag mean = {np.mean(corr - np.diag(np.diag(corr))):.3f}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_png = out_dir / "correlation_heatmap.png"
    plt.savefig(out_png, dpi=120)
    plt.close(fig)

    return {
        "diagonal_mean": float(np.mean(np.diag(corr))),
        "offdiag_mean": float(np.mean(corr - np.diag(np.diag(corr)))),
        "offdiag_std": float(np.std(corr - np.diag(np.diag(corr)))),
        "max_offdiag": float(np.max(corr - np.diag(np.diag(corr)))),
        "diag_dominance_ratio": float(np.mean(np.diag(corr)) / (np.mean(np.abs(corr - np.diag(np.diag(corr)))) + 1e-9)),
        "heatmap_png": str(out_png),
    }


def _experiment2_svr_prediction(
    category: str, latent: np.ndarray, user_feat: dict[str, np.ndarray], user_ids: list[str]
) -> dict:
    """Per-dim SVR prediction of 20-dim user feature mean from PQB latent.

    Also run PCA-20 baseline (raw user features → PCA → SVR) to compare.
    """
    from sklearn.svm import SVR
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold

    feat_mat = np.stack([user_feat[uid] for uid in user_ids], axis=0)  # [U, 20]
    U, D = feat_mat.shape

    sx = StandardScaler()
    X_pqb = sx.fit_transform(latent)  # [U, 20]

    pca = PCA(n_components=20)
    X_pca = pca.fit_transform(feat_mat)  # [U, 20] raw features → PCA

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    pqb_mse = []
    pca_mse = []
    for d in range(D):
        y = feat_mat[:, d]
        sy = StandardScaler()
        y_std = sy.fit_transform(y.reshape(-1, 1)).ravel()

        pqb_dim = []
        pca_dim = []
        for tr, te in kf.split(X_pqb):
            svr = SVR(kernel="rbf", C=1.0, gamma="scale")
            svr.fit(X_pqb[tr], y_std[tr])
            pqb_dim.append(float(np.mean((svr.predict(X_pqb[te]) - y_std[te]) ** 2)))
            svr2 = SVR(kernel="rbf", C=1.0, gamma="scale")
            svr2.fit(X_pca[tr], y_std[tr])
            pca_dim.append(float(np.mean((svr2.predict(X_pca[te]) - y_std[te]) ** 2)))
        pqb_mse.append(float(np.mean(pqb_dim)))
        pca_mse.append(float(np.mean(pca_dim)))

    pqb_mse = np.asarray(pqb_mse)
    pca_mse = np.asarray(pca_mse)
    pqb_wins = int(np.sum(pqb_mse < pca_mse))
    pca_wins = int(np.sum(pca_mse < pqb_mse))

    out_dir = FEATURE_BASE / category / f"interpretability{os.environ.get('INTERP_OUTPUT_SUFFIX', '')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(20)
    ax.bar(x - 0.2, pqb_mse, 0.4, label=f"PQB latent (mean={np.mean(pqb_mse):.3f})", color="steelblue")
    ax.bar(x + 0.2, pca_mse, 0.4, label=f"PCA-20 baseline (mean={np.mean(pca_mse):.3f})", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right")
    ax.set_ylabel("5-fold CV MSE (std y)")
    ax.set_title(f"SVR per-dim prediction — {category} (GMM k=0)\n"
                 f"PQB wins {pqb_wins}/20 dims, PCA wins {pca_wins}/20 dims")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_png = out_dir / "svr_per_dim_mse.png"
    plt.savefig(out_png, dpi=120)
    plt.close(fig)

    return {
        "pqb_mean_mse": float(np.mean(pqb_mse)),
        "pca_mean_mse": float(np.mean(pca_mse)),
        "pqb_wins": pqb_wins,
        "pca_wins": pca_wins,
        "per_dim": {
            FEATURE_NAMES[d]: {"pqb_mse": float(pqb_mse[d]), "pca_mse": float(pca_mse[d])}
            for d in range(20)
        },
        "bar_png": str(out_png),
    }


def _experiment3_tsne_projection(
    category: str, latent: np.ndarray, user_ids: list[str],
    user_total_reviews: dict[str, int], user_accept_rate: dict[str, float],
) -> dict:
    """T-SNE 2D projection, colored by user attributes (2 figures)."""
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sx = StandardScaler()
    X = sx.fit_transform(latent)
    n = min(3000, len(X))
    rng = np.random.default_rng(42)
    if len(X) > n:
        idx = rng.choice(len(X), n, replace=False)
        Xs = X[idx]
        uids = [user_ids[i] for i in idx]
    else:
        Xs = X
        uids = user_ids

    log(f"[{category}] t-SNE on {n} users...")
    z = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto").fit_transform(Xs)

    out_dir = FEATURE_BASE / category / f"interpretability{os.environ.get('INTERP_OUTPUT_SUFFIX', '')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    figs = {}
    for attr_name, attr_vals, cmap, label in [
        ("total_reviews", [user_total_reviews.get(uid, 0) for uid in uids], "viridis", "# reviews (stage1)"),
        ("accept_rate", [user_accept_rate.get(uid, np.nan) for uid in uids], "RdYlGn", "query accept_rate (selected/selected+rejected)"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 7))
        sc = ax.scatter(z[:, 0], z[:, 1], c=attr_vals, cmap=cmap, s=8, alpha=0.7)
        plt.colorbar(sc, ax=ax, label=label)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title(f"t-SNE 2D — {category} (GMM k=0)\ncolor = {attr_name}")
        plt.tight_layout()
        out_png = out_dir / f"tsne_2d_{attr_name}.png"
        plt.savefig(out_png, dpi=120)
        plt.close(fig)
        figs[attr_name] = str(out_png)

    return {
        "n_users_projected": int(n),
        "tsne_pngs": figs,
    }


def run_for_category(category: str) -> dict:
    log(f"\n{'='*60}\n>>> {category}\n{'='*60}")
    user_ids, latent, mix = _load_user_profiles(category)
    log(f"用户数: {len(user_ids)}, latent shape: {latent.shape}, mix shape: {mix.shape}")

    user_feat = _load_user_feature_means(category)
    log(f"用户级 20-dim 特征均值: {len(user_feat)} 用户")

    user_total_reviews = _load_user_total_reviews(category)
    user_accept_rate = _load_user_accept_rate(category)

    log("实验 1: 相关系数热力图")
    e1 = _experiment1_correlation_heatmap(category, latent, user_feat, user_ids)
    log(f"  diagonal mean = {e1['diagonal_mean']:.3f}, off-diag mean = {e1['offdiag_mean']:.3f}")
    log(f"  heatmap: {e1['heatmap_png']}")

    log("实验 2: SVR per-dim 预测")
    e2 = _experiment2_svr_prediction(category, latent, user_feat, user_ids)
    log(f"  PQB mean MSE = {e2['pqb_mean_mse']:.4f}, PCA mean MSE = {e2['pca_mean_mse']:.4f}")
    log(f"  PQB wins {e2['pqb_wins']}/20 dims, PCA wins {e2['pca_wins']}/20 dims")
    log(f"  bar plot: {e2['bar_png']}")

    log("实验 3: T-SNE 2D 投影")
    e3 = _experiment3_tsne_projection(category, latent, user_ids, user_total_reviews, user_accept_rate)
    for k, v in e3["tsne_pngs"].items():
        log(f"  t-SNE ({k}): {v}")

    return {
        "category": category,
        "n_users": len(user_ids),
        "exp1_correlation": e1,
        "exp2_svr": e2,
        "exp3_tsne": e3,
    }


def main() -> None:
    log("=" * 60)
    log("3 个可解释性实验 (借鉴 VADES Fig 3-5) - 三域")
    log("=" * 60)

    all_results: dict = {"categories": {}}
    for cat in CATEGORIES:
        try:
            all_results["categories"][cat] = run_for_category(cat)
        except Exception as e:
            log(f"[{cat}] FAILED: {e}")
            all_results["categories"][cat] = {"error": str(e)}

    out_dir = FEATURE_BASE / "interpretability"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_suffix = os.environ.get("INTERP_OUTPUT_SUFFIX", "")
    out_file = out_dir / f"interpretability_all_categories{out_suffix}.json"
    summary: dict = {"categories": {}}
    for cat, res in all_results["categories"].items():
        if "error" in res:
            continue
        summary["categories"][cat] = {
            "n_users": res["n_users"],
            "exp1": {
                "diagonal_mean_corr": res["exp1_correlation"]["diagonal_mean"],
                "offdiag_mean_corr": res["exp1_correlation"]["offdiag_mean"],
                "diag_dominance_ratio": res["exp1_correlation"]["diag_dominance_ratio"],
                "heatmap_png": res["exp1_correlation"]["heatmap_png"],
            },
            "exp2": {
                "pqb_mean_mse": res["exp2_svr"]["pqb_mean_mse"],
                "pca_mean_mse": res["exp2_svr"]["pca_mean_mse"],
                "pqb_wins": res["exp2_svr"]["pqb_wins"],
                "pca_wins": res["exp2_svr"]["pca_wins"],
                "bar_png": res["exp2_svr"]["bar_png"],
            },
            "exp3": {
                "n_users_projected": res["exp3_tsne"]["n_users_projected"],
                "tsne_pngs": res["exp3_tsne"]["tsne_pngs"],
            },
        }
    summary["feature_names"] = FEATURE_NAMES
    out_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n聚合结果: {out_file}")

    log("\n" + "=" * 80)
    log("=== 三域可解释性指标对比 ===")
    log("=" * 80)
    log(f"{'指标':<40} | {'Baby':>10} | {'Pet':>10} | {'Grocery':>10}")
    log("-" * 80)
    for key, label in [
        ("exp1.diagonal_mean_corr", "相关系数热力图 diag mean"),
        ("exp1.diag_dominance_ratio", "diag/offdiag 比"),
        ("exp2.pqb_mean_mse", "SVR PQB mean MSE"),
        ("exp2.pca_mean_mse", "SVR PCA mean MSE"),
        ("exp2.pqb_wins", "SVR PQB wins"),
    ]:
        row = f"{label:<40}"
        for cat in CATEGORIES:
            d = summary["categories"].get(cat, {})
            cur = d
            for p in key.split("."):
                cur = cur.get(p) if isinstance(cur, dict) else None
                if cur is None:
                    break
            row += f" | {cur:>10}" if isinstance(cur, (int, float)) else f" | {'N/A':>10}"
        log(row)
    log("=" * 80)


if __name__ == "__main__":
    main()
