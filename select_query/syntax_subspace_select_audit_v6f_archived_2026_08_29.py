"""Stage 4 v6f-A — Posterior Exclusivity Audit.

读 Stage 4 v6f selection (posterior_max entries),重算每个 ASIN 的完整
Mahal²/log_p/posterior matrix,验证选出的 query 是否真的"最属于目标用户"。

用户指令 2026-08-28: 必须先回答这个,而不是直接看 retrieval flip。

4 个核心指标:
1. Posterior Rank@1: target user 在所有 asin user posterior 中是否 top-1
2. Posterior lift over chance: P(u|q*) / (1/K)
3. Posterior margin M_post = P(u|q*) - max_{v≠u} P(v|q*)
4. Exclusive coverage: P(u|q*) > {0.5, 0.25, 1/K} 的比例

附加 sanity: posterior 与 log_det(Σ_u) 相关性 — 看 discrimination
是否来自 Gaussian 宽度而非句法位置独特性。

输入:
  - SELECTION_IN (= stage8_5_selection.json, v6f posterior_max entries)
  - ASINS_IN, POOL_IN, GAUSSIANS_IN, FEAT_CACHE

输出:
  - result/select_query/posterior_audit.json
"""

from __future__ import annotations

import collections
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from syntax_subspace_utils import (  # noqa: E402
    ASINS_IN, FEAT_CACHE, GAUSSIANS_IN, POOL_IN, SELECTION_IN,
    PCA_DIM, log, feat_key,
)


def mahalanobis_sq(z: np.ndarray, mu: np.ndarray, sigma_diag: np.ndarray) -> float:
    diff = z - mu
    return float((diff * diff / sigma_diag).sum())


def main():
    log("=== Stage 4 v6f-A — Posterior Exclusivity Audit ===")

    log("\n=== 1. Loading PCA48 ===")
    from syntax_subspace_utils import _syntax_subspace_prepare
    P = _syntax_subspace_prepare()
    scaler = P["scaler"]
    fnames = P["feature_names_ordered"]

    from sklearn.decomposition import PCA
    pca = PCA(n_components=PCA_DIM, random_state=42)
    pca.fit(P["X_scaled"][P["train_idx"]])
    log(f"  PCA{PCA_DIM} ready")

    log("\n=== 2. Loading inputs ===")
    selection = json.load(open(SELECTION_IN))
    entries = selection["entries"]
    asin_data = json.load(open(ASINS_IN))["asins"]
    pool_data = json.load(open(POOL_IN))
    pools = pool_data["pools"]
    gauss_data = json.load(open(GAUSSIANS_IN))
    users_gauss = gauss_data["users"]
    log(f"  selection entries: {len(entries)}")
    log(f"  ASINs: {len(asin_data)}, pools: {len(pools)}, user Gaussians: {len(users_gauss)}")

    feat_map = {}
    with gzip.open(FEAT_CACHE, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            feat_map[rec["k"]] = rec["v"]
    log(f"  features: {len(feat_map)}")

    log("\n=== 3. Projecting pool queries ===")
    pool_z = {}
    miss = 0
    for asin, qs in pools.items():
        zs_for_asin = []
        for q in qs:
            k = feat_key(q["query"])
            feats = feat_map.get(k)
            if not feats:
                miss += 1
                continue
            vec = np.array([feats.get(n, 0.0) for n in fnames], dtype=np.float64)
            z = pca.transform(scaler.transform(vec[None, :]))[0]
            zs_for_asin.append((z, q))
        pool_z[asin] = zs_for_asin
    log(f"  ASINs with pool_z: {len(pool_z)}, missing features: {miss}")

    log("\n=== 4. Grouping entries by ASIN (posterior_max only) ===")
    by_asin = collections.defaultdict(list)
    n_posterior_max = 0
    n_other_method = 0
    for e in entries:
        if e["selection_method"] == "posterior_max":
            by_asin[e["asin"]].append(e)
            n_posterior_max += 1
        else:
            n_other_method += 1
    log(f"  posterior_max entries: {n_posterior_max}")
    log(f"  other method entries (skipped): {n_other_method}")
    log(f"  ASINs with posterior_max: {len(by_asin)}")

    log("\n=== 5. Computing exclusivity metrics per entry ===")
    metrics_per_entry = []
    n_audit_skip_no_pool = 0
    n_audit_skip_no_strict = 0
    n_audit_skip_no_gauss = 0
    n_audit_skip_query_miss = 0

    for asin, asin_entries in by_asin.items():
        zqs = pool_z.get(asin)
        if zqs is None:
            n_audit_skip_no_pool += 1
            continue
        strict_zqs = [(z, q) for z, q in zqs if q["strict"]]
        if not strict_zqs:
            n_audit_skip_no_strict += 1
            continue

        # 收集该 ASIN 所有用户 — 顺序: 按 entries 第一次出现顺序 (与 select.py 一致)
        asin_user_ids = []
        for e in asin_entries:
            uid = e["user_id"]
            if uid not in users_gauss:
                n_audit_skip_no_gauss += 1
                continue
            if uid not in asin_user_ids:
                asin_user_ids.append(uid)

        n_users = len(asin_user_ids)
        n_queries = len(strict_zqs)
        if n_users < 2:
            # 单用户没法算 posterior over users,跳过
            n_audit_skip_no_gauss += 1
            continue

        # Mahal² matrix + log_det
        mahal = np.zeros((n_users, n_queries))
        log_det = np.zeros(n_users)
        for ui, uid in enumerate(asin_user_ids):
            sigma = np.array(users_gauss[uid]["sigma_diag"])
            log_det[ui] = float(np.sum(np.log(sigma)))
            for qi, (z, _) in enumerate(strict_zqs):
                mahal[ui, qi] = mahalanobis_sq(z, np.array(users_gauss[uid]["mu"]), sigma)

        # Posterior
        log_p = -0.5 * mahal - 0.5 * log_det[:, None]
        log_p_max = log_p.max(axis=0, keepdims=True)
        log_p_shifted = log_p - log_p_max
        exp_shifted = np.exp(log_p_shifted)
        posterior = exp_shifted / exp_shifted.sum(axis=0, keepdims=True)

        K = n_users
        chance = 1.0 / K
        q_to_qi = {q["query"]: qi for qi, (z, q) in enumerate(strict_zqs)}
        uid_to_ui = {uid: ui for ui, uid in enumerate(asin_user_ids)}

        for e in asin_entries:
            uid = e["user_id"]
            selected = e["selected"]
            if selected is None:
                continue
            ui = uid_to_ui.get(uid)
            if ui is None:
                continue
            qi = q_to_qi.get(selected["query"])
            if qi is None:
                n_audit_skip_query_miss += 1
                continue

            target_post = float(posterior[ui, qi])
            sorted_post = np.sort(posterior[:, qi])[::-1]
            rank = int((sorted_post >= target_post - 1e-12).sum())
            other_post = np.delete(posterior[:, qi], ui)
            other_max_post = float(other_post.max())
            margin = target_post - other_max_post
            lift = target_post / chance
            log_det_u = float(log_det[ui])
            mahal_u_q = float(mahal[ui, qi])

            metrics_per_entry.append({
                "asin": asin,
                "user_id": uid,
                "K": K,
                "chance": chance,
                "target_posterior": target_post,
                "target_rank": rank,
                "other_max_posterior": other_max_post,
                "posterior_margin": margin,
                "posterior_lift": lift,
                "log_det": log_det_u,
                "mahal": mahal_u_q,
                "selected_query": selected["query"][:80],
                "above_0.5": target_post > 0.5,
                "above_0.25": target_post > 0.25,
                "above_chance": target_post > chance,
                "rank1": rank == 1,
                "margin_positive": margin > 0,
            })

    log(f"  entries audited: {len(metrics_per_entry)}")
    log(f"  audit-skip reasons: no_pool={n_audit_skip_no_pool}, no_strict={n_audit_skip_no_strict}, "
        f"no_gauss={n_audit_skip_no_gauss}, query_miss={n_audit_skip_query_miss}")

    if not metrics_per_entry:
        log("  no entries to audit")
        return

    log("\n=== 6. Aggregate metrics ===")
    post = np.array([m["target_posterior"] for m in metrics_per_entry])
    rank = np.array([m["target_rank"] for m in metrics_per_entry])
    margin = np.array([m["posterior_margin"] for m in metrics_per_entry])
    lift = np.array([m["posterior_lift"] for m in metrics_per_entry])
    log_det_arr = np.array([m["log_det"] for m in metrics_per_entry])
    K_arr = np.array([m["K"] for m in metrics_per_entry])
    chance_arr = np.array([m["chance"] for m in metrics_per_entry])
    rank1_arr = np.array([m["rank1"] for m in metrics_per_entry])
    margin_pos_arr = np.array([m["margin_positive"] for m in metrics_per_entry])
    above_05 = np.array([m["above_0.5"] for m in metrics_per_entry])
    above_025 = np.array([m["above_0.25"] for m in metrics_per_entry])
    above_chance = np.array([m["above_chance"] for m in metrics_per_entry])
    other_max_arr = np.array([m["other_max_posterior"] for m in metrics_per_entry])

    agg = {
        "n_total": int(len(metrics_per_entry)),
        "target_posterior": {
            "mean": float(post.mean()),
            "median": float(np.median(post)),
            "std": float(post.std()),
            "min": float(post.min()),
            "max": float(post.max()),
            "q25": float(np.percentile(post, 25)),
            "q75": float(np.percentile(post, 75)),
        },
        "target_rank": {
            "mean": float(rank.mean()),
            "median": float(np.median(rank)),
            "fraction_rank1": float(rank1_arr.mean()),
            "fraction_rank_le_2": float((rank <= 2).mean()),
            "fraction_rank_le_3": float((rank <= 3).mean()),
        },
        "posterior_margin": {
            "mean": float(margin.mean()),
            "median": float(np.median(margin)),
            "std": float(margin.std()),
            "fraction_positive": float(margin_pos_arr.mean()),
            "fraction_negative": float((margin < 0).mean()),
        },
        "other_max_posterior": {
            "mean": float(other_max_arr.mean()),
            "median": float(np.median(other_max_arr)),
        },
        "posterior_lift_over_chance": {
            "mean": float(lift.mean()),
            "median": float(np.median(lift)),
            "fraction_above_chance": float(above_chance.mean()),
        },
        "exclusive_coverage": {
            "P(u|q*) > 0.5": float(above_05.mean()),
            "P(u|q*) > 0.25": float(above_025.mean()),
            "P(u|q*) > 1/K (chance)": float(above_chance.mean()),
        },
        "K_per_asin": {
            "mean": float(K_arr.mean()),
            "median": float(np.median(K_arr)),
            "min": int(K_arr.min()),
            "max": int(K_arr.max()),
            "mean_chance": float(chance_arr.mean()),
        },
    }

    log(f"\n=== Target posterior P(u|q*) ===")
    log(f"  mean={agg['target_posterior']['mean']:.4f}, median={agg['target_posterior']['median']:.4f}, "
        f"std={agg['target_posterior']['std']:.4f}")
    log(f"  q25={agg['target_posterior']['q25']:.4f}, q75={agg['target_posterior']['q75']:.4f}")
    log(f"  min={agg['target_posterior']['min']:.4f}, max={agg['target_posterior']['max']:.4f}")
    log(f"  chance (mean 1/K) = {agg['K_per_asin']['mean_chance']:.4f}")

    log(f"\n=== Target-user Posterior Rank ===")
    log(f"  mean rank = {agg['target_rank']['mean']:.2f}, median rank = {agg['target_rank']['median']}")
    log(f"  Rank@1 (target IS top-1):        {agg['target_rank']['fraction_rank1']*100:.1f}%")
    log(f"  Rank ≤ 2 (top-2):                {agg['target_rank']['fraction_rank_le_2']*100:.1f}%")
    log(f"  Rank ≤ 3 (top-3):                {agg['target_rank']['fraction_rank_le_3']*100:.1f}%")

    log(f"\n=== Posterior margin M_post = P(u|q*) - max_{{v≠u}} P(v|q*) ===")
    log(f"  mean = {agg['posterior_margin']['mean']:.4f}, median = {agg['posterior_margin']['median']:.4f}, "
        f"std = {agg['posterior_margin']['std']:.4f}")
    log(f"  other_max_post mean = {agg['other_max_posterior']['mean']:.4f}, "
        f"median = {agg['other_max_posterior']['median']:.4f}")
    log(f"  M_post > 0 (target IS top-1):     {agg['posterior_margin']['fraction_positive']*100:.1f}%")
    log(f"  M_post < 0 (target NOT top-1):   {agg['posterior_margin']['fraction_negative']*100:.1f}%")

    log(f"\n=== Posterior lift over chance ===")
    log(f"  lift mean = {agg['posterior_lift_over_chance']['mean']:.2f}, "
        f"median = {agg['posterior_lift_over_chance']['median']:.2f}")
    log(f"  P(u|q*) > 1/K (chance):          {agg['posterior_lift_over_chance']['fraction_above_chance']*100:.1f}%")

    log(f"\n=== Exclusive coverage ===")
    log(f"  P(u|q*) > 0.50:  {agg['exclusive_coverage']['P(u|q*) > 0.5']*100:.1f}%")
    log(f"  P(u|q*) > 0.25:  {agg['exclusive_coverage']['P(u|q*) > 0.25']*100:.1f}%")
    log(f"  P(u|q*) > 1/K:   {agg['exclusive_coverage']['P(u|q*) > 1/K (chance)']*100:.1f}%")

    log(f"\n=== K (n_users per ASIN) ===")
    log(f"  mean = {agg['K_per_asin']['mean']:.2f}, median = {agg['K_per_asin']['median']}")
    log(f"  range [{agg['K_per_asin']['min']}, {agg['K_per_asin']['max']}]")

    log(f"\n=== Sanity: posterior vs log_det(Σ_u) ===")
    rho_pearson, p_pearson = pearsonr(post, log_det_arr)
    rho_spearman, p_spearman = spearmanr(post, log_det_arr)
    log(f"  log_det mean = {log_det_arr.mean():.3f}, std = {log_det_arr.std():.3f}, "
        f"range [{log_det_arr.min():.3f}, {log_det_arr.max():.3f}]")
    log(f"  ρ(P(u|q*), log_det(Σ_u)) Pearson  = {rho_pearson:+.3f}  (p = {p_pearson:.2e})")
    log(f"  ρ(P(u|q*), log_det(Σ_u)) Spearman = {rho_spearman:+.3f}  (p = {p_spearman:.2e})")
    if abs(rho_spearman) > 0.3:
        log(f"  ⚠ posterior 与 log_det 强相关: posterior discrimination 部分来自 σ 宽度,"
            f"而非纯句法位置 unique性")
    else:
        log(f"  ✓ posterior 与 log_det 相关性弱: discrimination 主要来自句法位置")

    agg["sanity_posterior_vs_log_det"] = {
        "pearson_r": float(rho_pearson),
        "pearson_p": float(p_pearson),
        "spearman_r": float(rho_spearman),
        "spearman_p": float(p_spearman),
        "log_det_mean": float(log_det_arr.mean()),
        "log_det_std": float(log_det_arr.std()),
        "log_det_min": float(log_det_arr.min()),
        "log_det_max": float(log_det_arr.max()),
    }

    OUT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/select_query/posterior_audit.json")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": ("Stage 4 v6f-A: Posterior Exclusivity Audit. Reads selection.json "
                                "(v6f posterior_max entries), recomputes full Mahal²/log_p/posterior "
                                "matrix per ASIN, computes 4 exclusivity metrics "
                                "(target_rank / lift / margin / coverage) + sanity check "
                                "(posterior vs log_det)."),
            },
            "aggregate": agg,
        }, f, ensure_ascii=False, indent=2)
    log(f"\nwrote → {OUT}")


if __name__ == "__main__":
    main()