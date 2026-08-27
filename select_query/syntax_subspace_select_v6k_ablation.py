"""Stage 4 v6k — Whitened Syntax Space Selection (用户指令 2026-08-28).

v6i/v6j G3 的 √χ²(0.95, 48)=8.073 漂亮结果来自一个错误的假设:
  PCA48 z-space 各维单位方差且同尺度 (z ~ N(0, I))
  → 实际 PCA 不 whitening,各 PC eigenvalue 不同,||z-μ|| 中位数=24.16
  → 8.073 是"极端严格中心筛选",5% pass rate 是 selection effect

v6k 正确做法:
  1. Whiten PCA48: z̃_j = z_j / √λ_j  → 各维单位方差
  2. 用真实用户历史评论 (298740 sentence features) 校准 shared R_95:
     r_u_i = ||z̃_u_i − μ̃_u||
     R_95 = pooled 95th percentile of r
  3. L2 margin score (无 per-user σ):
     M(q, u) = min_{v≠u} ||z̃_q − μ̃_v||_2 − ||z̃_q − μ̃_u||_2
  4. Select: q*_u = argmax_q M(q, u) subject to ||z̃_q − μ̃_u||_2 ≤ R_95

R_95 = "真实用户历史句法表达中约 95% 落入的 shared radius" —
代表"用户正常表达范围",与 candidate pool 无关。

输入: ASINS_IN / POOL_IN / GAUSSIANS_IN / FEAT_CACHE / sentences + rewrites
输出: result/select_query/v6k_ablation.json
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
    ASINS_IN, FEAT_CACHE, GAUSSIANS_IN, POOL_IN,
    PCA_DIM, log, feat_key,
)


def main():
    log("=== Stage 4 v6k — Whitened Syntax Space Selection ===")

    log("\n=== 1. Loading PCA48 ===")
    from syntax_subspace_utils import _syntax_subspace_prepare
    P = _syntax_subspace_prepare()
    scaler = P["scaler"]
    fnames = P["feature_names_ordered"]

    X_scaled = P["X_scaled"]
    user_ids_all = P["user_ids"]
    user_to_indices = P["user_to_indices"]
    log(f"  X_scaled shape: {X_scaled.shape}, {len(user_to_indices)} users")

    from sklearn.decomposition import PCA
    pca = PCA(n_components=PCA_DIM, random_state=42)
    pca.fit(P["X_scaled"][P["train_idx"]])
    explained_var = pca.explained_variance_
    sqrt_lambda = np.sqrt(explained_var)
    log(f"  PCA explained variance: λ range = [{explained_var.min():.3f}, "
        f"{explained_var.max():.3f}]")
    log(f"  sqrt(λ) range = [{sqrt_lambda.min():.3f}, {sqrt_lambda.max():.3f}]")

    log("\n=== 2. Project pool queries to PCA48 ===")
    asin_data = json.load(open(ASINS_IN))["asins"]
    pool_data = json.load(open(POOL_IN))
    pools = pool_data["pools"]
    gauss_data = json.load(open(GAUSSIANS_IN))
    users_gauss = gauss_data["users"]
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

    pool_z_raw = {}
    pool_z_white = {}
    miss = 0
    for asin, qs in pools.items():
        zs_raw = []
        zs_white = []
        for q in qs:
            k = feat_key(q["query"])
            feats = feat_map.get(k)
            if not feats:
                miss += 1
                continue
            vec = np.array([feats.get(n, 0.0) for n in fnames], dtype=np.float64)
            z_raw = pca.transform(scaler.transform(vec[None, :]))[0]
            z_white = z_raw / sqrt_lambda
            zs_raw.append((z_raw, q))
            zs_white.append((z_white, q))
        pool_z_raw[asin] = zs_raw
        pool_z_white[asin] = zs_white
    log(f"  ASINs with pool_z: {len(pool_z_raw)}, missing features: {miss}")

    log("\n=== 3. Compute whitened user centers μ̃_u from historical sentences ===")
    user_z_white_means = {}
    user_z_white_stds = {}
    for uid in user_to_indices:
        idx = user_to_indices[uid]
        z_user = pca.transform(X_scaled[idx]) / sqrt_lambda  # (n_sents, 48)
        user_z_white_means[uid] = z_user.mean(axis=0)
        user_z_white_stds[uid] = z_user.std(axis=0)

    log("\n=== 4. Compute R_95 from whitened historical residuals ===")
    all_residuals = []
    for uid in user_to_indices:
        idx = user_to_indices[uid]
        z_user = pca.transform(X_scaled[idx]) / sqrt_lambda
        mu = user_z_white_means[uid]
        residuals = np.linalg.norm(z_user - mu, axis=1)
        all_residuals.append(residuals)
    all_residuals = np.concatenate(all_residuals)
    log(f"  total (user, sentence) pairs: {len(all_residuals)}")
    log(f"  residual distribution (whitened L2 distance to μ̃_u):")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        log(f"    P{p} = {np.percentile(all_residuals, p):.3f}")

    R_95 = float(np.percentile(all_residuals, 95))
    R_90 = float(np.percentile(all_residuals, 90))
    R_99 = float(np.percentile(all_residuals, 99))
    R_median = float(np.percentile(all_residuals, 50))
    log(f"  R_95 = {R_95:.3f} (pooled 95th of whitened residual)")

    log("\n=== 5. Per-ASIN G3 (whitened L2 margin + R_95 gate) ===")
    by_asin_users = collections.OrderedDict()
    for entry in asin_data:
        by_asin_users[entry["asin"]] = list(entry["users_sampled"])

    variant_results = {
        "G3_white_R95": {"selected_entries": []},
        "G3_white_R90": {"selected_entries": []},
        "G3_white_R99": {"selected_entries": []},
        "G3_white_median": {"selected_entries": []},
    }
    n_asin_processed = 0
    n_skip = {"no_pool": 0, "no_strict": 0, "lt_2_users": 0}

    for asin_idx, (asin, user_ids) in enumerate(by_asin_users.items()):
        if asin_idx % 100 == 0:
            log(f"  ... ASIN {asin_idx}/{len(by_asin_users)}")
        zqs = pool_z_white.get(asin)
        if zqs is None:
            n_skip["no_pool"] += 1
            continue
        strict_zqs = [(z, q) for z, q in zqs if q["strict"]]
        if not strict_zqs:
            n_skip["no_strict"] += 1
            continue

        asin_users = []
        for uid in user_ids:
            if uid not in users_gauss or uid not in user_z_white_means:
                continue
            mu_white = user_z_white_means[uid]
            asin_users.append((uid, mu_white))
        if len(asin_users) < 2:
            n_skip["lt_2_users"] += 1
            continue

        n_users = len(asin_users)
        n_queries = len(strict_zqs)

        # Whitened L2 distance matrix
        l2_white = np.zeros((n_users, n_queries))
        for ui, (uid, mu) in enumerate(asin_users):
            for qi, (z, _) in enumerate(strict_zqs):
                l2_white[ui, qi] = float(np.linalg.norm(z - mu))

        # M_L2 = min_{v≠u} ||z-μ_v|| − ||z-μ_u||
        sorted_l2 = np.sort(l2_white, axis=0)
        argmin_l2_per_qi = np.argmin(l2_white, axis=0)
        is_self_argmin = (argmin_l2_per_qi[None, :] == np.arange(n_users)[:, None])
        d_other_l2 = np.where(is_self_argmin, sorted_l2[1], sorted_l2[0])
        M_L2 = d_other_l2 - l2_white

        # Per-user log_det from raw Mahal² Gaussian (for sanity check ρ)
        log_dets = np.zeros(n_users)
        for ui, (uid, _) in enumerate(asin_users):
            log_dets[ui] = float(np.sum(np.log(np.array(users_gauss[uid]["sigma_diag"]))))

        n_asin_processed += 1

        for ui, (uid, mu) in enumerate(asin_users):
            ld = float(log_dets[ui])
            score_M = M_L2[ui]

            for var_name, threshold in [
                ("G3_white_R95", R_95),
                ("G3_white_R90", R_90),
                ("G3_white_R99", R_99),
                ("G3_white_median", R_median),
            ]:
                in_dist = l2_white[ui] <= threshold
                if not in_dist.any():
                    continue
                avail_M = np.where(in_dist, score_M, -np.inf)
                best_qi = int(np.argmax(avail_M))
                target_M = float(score_M[best_qi])
                sorted_M = np.sort(M_L2[:, best_qi])[::-1]
                rank = int((sorted_M >= target_M - 1e-12).sum())
                other_M = np.delete(M_L2[:, best_qi], ui)
                margin = target_M - float(other_M.max())

                variant_results[var_name]["selected_entries"].append({
                    "asin": asin,
                    "user_id": uid,
                    "selected_query": strict_zqs[best_qi][1]["query"][:80],
                    "selected_score": target_M,
                    "target_rank": rank,
                    "score_margin": margin,
                    "log_det": ld,
                })

    log(f"\n  ASINs processed: {n_asin_processed}, skipped: {n_skip}")

    log("\n=== 6. Per-variant aggregate ===")
    summary = {}
    for var_name in ["G3_white_R95", "G3_white_R90", "G3_white_R99", "G3_white_median"]:
        entries = variant_results[var_name]["selected_entries"]
        if not entries:
            log(f"\n--- {var_name}: no entries ---")
            continue
        scores = np.array([e["selected_score"] for e in entries])
        log_dets = np.array([e["log_det"] for e in entries])
        ranks = np.array([e["target_rank"] for e in entries])
        margins = np.array([e["score_margin"] for e in entries])
        rank1 = float((ranks == 1).mean())
        margin_pos = float((margins > 0).mean())

        if np.std(scores) > 0 and np.std(log_dets) > 0:
            rho_p, p_p = pearsonr(scores, log_dets)
            rho_s, p_s = spearmanr(scores, log_dets)
        else:
            rho_p = rho_s = 0.0
            p_p = p_s = 1.0

        by_asin_q = collections.defaultdict(list)
        for e in entries:
            by_asin_q[e["asin"]].append(e["selected_query"])
        n_q_total = sum(len(qs) for qs in by_asin_q.values())
        n_unique_total = sum(len(set(qs)) for qs in by_asin_q.values())
        n_all_same = sum(1 for qs in by_asin_q.values() if len(set(qs)) == 1)
        unique_ratio = n_unique_total / max(n_q_total, 1)

        summary[var_name] = {
            "n_entries": len(entries),
            "score_mean": float(scores.mean()),
            "score_median": float(np.median(scores)),
            "score_std": float(scores.std()),
            "target_rank_mean": float(ranks.mean()),
            "target_rank_median": float(np.median(ranks)),
            "rank1_fraction": rank1,
            "rank_le_2_fraction": float((ranks <= 2).mean()),
            "score_margin_mean": float(margins.mean()),
            "score_margin_median": float(np.median(margins)),
            "margin_positive_fraction": margin_pos,
            "rho_pearson_score_vs_logdet": float(rho_p),
            "rho_spearman_score_vs_logdet": float(rho_s),
            "pearson_p": float(p_p),
            "spearman_p": float(p_s),
            "uniqueness": {
                "avg_unique_ratio": float(unique_ratio),
                "n_all_same_asins": n_all_same,
                "n_total_asins": len(by_asin_q),
            },
        }

        log(f"\n--- {var_name} ---")
        log(f"  n_entries = {len(entries)}")
        log(f"  score: mean={scores.mean():.3f}, median={np.median(scores):.3f}, "
            f"std={scores.std():.3f}")
        log(f"  target rank: mean={ranks.mean():.2f}, median={np.median(ranks)}, "
            f"Rank@1={rank1*100:.1f}%, Rank≤2={float((ranks<=2).mean())*100:.1f}%")
        log(f"  score margin: mean={margins.mean():.3f}, median={np.median(margins):.3f}, "
            f"M>0={margin_pos*100:.1f}%")
        log(f"  ρ(M, log_det): Pearson={rho_p:+.3f} (p={p_p:.2e}), "
            f"Spearman={rho_s:+.3f} (p={p_s:.2e})")
        log(f"  uniqueness: avg ratio={unique_ratio:.3f}, "
            f"all-same ASINs={n_all_same}/{len(by_asin_q)}")
        if abs(rho_s) > 0.3:
            log(f"  ⚠ 仍 σ-confounded (|ρ_s|>0.3)")
        else:
            log(f"  ✓ deconfounded (|ρ_s|<0.3)")

    log("\n=== 7. GO verdict ===")
    verdict = {}
    for var_name in summary:
        s = summary[var_name]
        deconfounded = abs(s["rho_spearman_score_vs_logdet"]) < 0.3
        rank1_ok = s["rank1_fraction"] >= 0.25
        margin_ok = s["margin_positive_fraction"] >= 0.25
        go = deconfounded and rank1_ok and margin_ok
        verdict[var_name] = {
            "deconfounded": deconfounded,
            "rank1_ok": rank1_ok,
            "margin_ok": margin_ok,
            "go": go,
        }
        log(f"  {var_name}: deconfounded={deconfounded}, rank1≥25%={rank1_ok}, "
            f"margin≥25%={margin_ok}, GO={go}")

    OUT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/select_query/v6k_ablation.json")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": ("Stage 4 v6k: Whitened Syntax Space Selection. "
                                "PCA48 z̃ = z / √λ, R_95 = pooled 95th percentile of "
                                "whitened L2 distance from user historical sentences "
                                "to μ̃_u. Gate: ||z̃_q - μ̃_u|| ≤ R_95. "
                                "Score: M(q,u) = min_{v≠u} L2 − self L2."),
                "R_95": R_95,
                "R_90": R_90,
                "R_99": R_99,
                "R_median": R_median,
                "n_historical_sentence_residuals": int(len(all_residuals)),
                "pca_explained_variance_range": [
                    float(explained_var.min()), float(explained_var.max())
                ],
            },
            "summary": summary,
            "verdict": verdict,
        }, f, ensure_ascii=False, indent=2)
    log(f"\nwrote → {OUT}")


if __name__ == "__main__":
    main()