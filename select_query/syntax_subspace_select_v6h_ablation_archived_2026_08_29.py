"""Stage 4 v6h — Gaussian Gate + L2 Margin (双层拆分).

用户指令 2026-08-28: 整个基于 per-user σ 的 Mahalanobis family 都不适合承担
"用户独特性"。把 style membership 和 style exclusivity 拆成两个不同东西:

  Layer 1: Mahal² ≤ χ²(0.95, 48) = 65.17 → "Query 落在目标用户 95% 表达范围内"
           (合法性 gate, 仍用 per-user σ)

  Layer 2: M_L2 = min_{v≠u} ||z_q − μ_v||_2 − ||z_q − μ_u||_2
           → "Query 离目标用户 μ 比离其他用户 μ 更近的程度"
           (exclusivity, 完全不依赖 σ)

  q*_u = argmax_q M_L2(q, u)  subject to  Mahal²(q, u) ≤ 65.17

GO 条件:
  1. |ρ(M_L2, log_det(Σ_u))| < 0.3  (deconfounded)
  2. Rank@1 > 34.2%  (vs v6f)
  3. M_L2 > 0 比例 > 34.2%

输入: ASINS_IN / POOL_IN / GAUSSIANS_IN / FEAT_CACHE
输出: result/select_query/v6h_ablation.json
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
    MAHAL_THRESHOLD_CHI2_PPF, MAHAL_THRESHOLD_DF,
    PCA_DIM, log, feat_key,
)
from scipy.stats import chi2


def mahalanobis_sq(z: np.ndarray, mu: np.ndarray, sigma_diag: np.ndarray) -> float:
    diff = z - mu
    return float((diff * diff / sigma_diag).sum())


def main():
    log("=== Stage 4 v6h — Gaussian Gate + L2 Margin (双层拆分) ===")

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

    log("\n=== 4. Per-ASIN: Mahal² (gate) + L2 (exclusivity) ===")
    mahal_threshold = chi2.ppf(MAHAL_THRESHOLD_CHI2_PPF, MAHAL_THRESHOLD_DF)
    log(f"  Mahal² gating threshold = χ²({MAHAL_THRESHOLD_CHI2_PPF}, "
        f"{MAHAL_THRESHOLD_DF}) = {mahal_threshold:.3f}")

    by_asin_users = collections.OrderedDict()
    for entry in asin_data:
        by_asin_users[entry["asin"]] = list(entry["users_sampled"])

    # Two ablation variants:
    #   D: v6h L2 margin
    #   A: v6f posterior (baseline for compare, 用同一 audit)
    variant_results = {
        "a_v6f_posterior_baseline": {"selected_entries": []},
        "d_v6h_L2_margin": {"selected_entries": []},
    }

    n_asin_processed = 0
    n_asin_skip = {"no_pool": 0, "no_strict": 0, "lt_2_users": 0}

    for asin_idx, (asin, user_ids) in enumerate(by_asin_users.items()):
        if asin_idx % 100 == 0:
            log(f"  ... ASIN {asin_idx}/{len(by_asin_users)}")
        zqs = pool_z.get(asin)
        if zqs is None:
            n_asin_skip["no_pool"] += 1
            continue
        strict_zqs = [(z, q) for z, q in zqs if q["strict"]]
        if not strict_zqs:
            n_asin_skip["no_strict"] += 1
            continue

        asin_users = []
        for uid in user_ids:
            if uid not in users_gauss:
                continue
            mu = np.array(users_gauss[uid]["mu"])
            sigma = np.array(users_gauss[uid]["sigma_diag"])
            asin_users.append((uid, mu, sigma))
        if len(asin_users) < 2:
            n_asin_skip["lt_2_users"] += 1
            continue

        n_users = len(asin_users)
        n_queries = len(strict_zqs)

        # Mahal² matrix (per-user Σ) — for gate + variant A
        mahal = np.zeros((n_users, n_queries))
        log_det = np.zeros(n_users)
        for ui, (uid, mu, sigma) in enumerate(asin_users):
            log_det[ui] = float(np.sum(np.log(sigma)))
            for qi, (z, _) in enumerate(strict_zqs):
                mahal[ui, qi] = mahalanobis_sq(z, mu, sigma)

        # L2 distance matrix (PCA48 standardized z, no per-user σ)
        l2 = np.zeros((n_users, n_queries))
        for ui, (uid, mu, _) in enumerate(asin_users):
            for qi, (z, _) in enumerate(strict_zqs):
                l2[ui, qi] = float(np.linalg.norm(z - mu))

        # d_other_L2 = min over v≠u of l2[v, qi]
        # sorted_l2[i, qi] = i-th smallest l2 over users
        sorted_l2 = np.sort(l2, axis=0)  # (n_users, n_queries)
        argmin_l2_per_qi = np.argmin(l2, axis=0)  # (n_queries,)
        # is_self_argmin[ui, qi] = (argmin_l2_per_qi[qi] == ui)
        is_self_argmin = (argmin_l2_per_qi[None, :] == np.arange(n_users)[:, None])
        d_other_l2 = np.where(is_self_argmin, sorted_l2[1], sorted_l2[0])

        # M_L2 = d_other_L2 − d_self_L2 (higher = more discriminative for u)
        M_L2 = d_other_l2 - l2  # (n_users, n_queries)

        # Variant A: v6f posterior
        log_p = -0.5 * mahal - 0.5 * log_det[:, None]
        log_p_max = log_p.max(axis=0, keepdims=True)
        log_p_shifted = log_p - log_p_max
        posterior = np.exp(log_p_shifted) / np.exp(log_p_shifted).sum(axis=0, keepdims=True)

        K = n_users
        chance = 1.0 / K
        n_asin_processed += 1

        for ui, (uid, mu, sigma) in enumerate(asin_users):
            in_dist_mask = mahal[ui] <= mahal_threshold
            if not in_dist_mask.any():
                continue

            ld = float(log_det[ui])

            # ---- Variant A: v6f posterior_max (baseline) ----
            avail_post = np.where(in_dist_mask, posterior[ui], -np.inf)
            best_qi_a = int(np.argmax(avail_post))
            target_post = float(posterior[ui][best_qi_a])
            sorted_post = np.sort(posterior[:, best_qi_a])[::-1]
            rank_a = int((sorted_post >= target_post - 1e-12).sum())
            other_post = np.delete(posterior[:, best_qi_a], ui)
            margin_a = target_post - float(other_post.max())
            best_mahal_a = float(mahal[ui][best_qi_a])

            variant_results["a_v6f_posterior_baseline"]["selected_entries"].append({
                "asin": asin,
                "user_id": uid,
                "selected_query": strict_zqs[best_qi_a][1]["query"][:80],
                "selected_distance": best_mahal_a,
                "selected_score": target_post,
                "target_rank": rank_a,
                "score_margin": margin_a,
                "log_det": ld,
                "K": K,
                "chance": chance,
            })

            # ---- Variant D: v6h L2 margin ----
            avail_M = np.where(in_dist_mask, M_L2[ui], -np.inf)
            best_qi_d = int(np.argmax(avail_M))
            target_M = float(M_L2[ui][best_qi_d])
            sorted_M = np.sort(M_L2[:, best_qi_d])[::-1]
            rank_d = int((sorted_M >= target_M - 1e-12).sum())
            other_M = np.delete(M_L2[:, best_qi_d], ui)
            margin_d = target_M - float(other_M.max())
            best_mahal_d = float(mahal[ui][best_qi_d])

            variant_results["d_v6h_L2_margin"]["selected_entries"].append({
                "asin": asin,
                "user_id": uid,
                "selected_query": strict_zqs[best_qi_d][1]["query"][:80],
                "selected_distance": best_mahal_d,
                "selected_score": target_M,
                "target_rank": rank_d,
                "score_margin": margin_d,
                "log_det": ld,
                "K": K,
            })

    log(f"\n  ASINs processed: {n_asin_processed}")
    log(f"  ASINs skipped: {n_asin_skip}")
    log(f"  entries per variant: {len(variant_results['a_v6f_posterior_baseline']['selected_entries'])}, "
        f"{len(variant_results['d_v6h_L2_margin']['selected_entries'])}")

    log("\n=== 5. Per-variant aggregate ===")
    summary = {}
    for var_name in ["a_v6f_posterior_baseline", "d_v6h_L2_margin"]:
        entries = variant_results[var_name]["selected_entries"]
        if not entries:
            continue
        scores = np.array([e["selected_score"] for e in entries])
        log_dets = np.array([e["log_det"] for e in entries])
        mahals = np.array([e["selected_distance"] for e in entries])
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

        # Per-ASIN uniqueness
        by_asin_q = collections.defaultdict(list)
        for e in entries:
            by_asin_q[e["asin"]].append(e["selected_query"])
        n_q_total = sum(len(qs) for qs in by_asin_q.values())
        n_unique_total = sum(len(set(qs)) for qs in by_asin_q.values())
        n_all_same = sum(1 for qs in by_asin_q.values() if len(set(qs)) == 1)
        unique_ratio = n_unique_total / max(n_q_total, 1)

        # Above chance (only for posterior variant)
        if "chance" in entries[0]:
            chances = np.array([e["chance"] for e in entries])
            above_chance = float((scores > chances).mean())
        else:
            above_chance = None

        summary[var_name] = {
            "n_entries": len(entries),
            "score_mean": float(scores.mean()),
            "score_median": float(np.median(scores)),
            "score_std": float(scores.std()),
            "selected_mahal_mean": float(mahals.mean()),
            "selected_mahal_median": float(np.median(mahals)),
            "selected_mahal_max": float(mahals.max()),
            "selected_mahal_p99": float(np.percentile(mahals, 99)),
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
        if above_chance is not None:
            summary[var_name]["above_chance_fraction"] = above_chance

        log(f"\n--- {var_name} ---")
        log(f"  n_entries = {len(entries)}")
        log(f"  selected Mahal²: mean={mahals.mean():.3f}, median={np.median(mahals):.3f}, "
            f"max={mahals.max():.3f}, p99={np.percentile(mahals, 99):.3f}")
        log(f"  score: mean={scores.mean():.3f}, median={np.median(scores):.3f}, "
            f"std={scores.std():.3f}")
        log(f"  target rank: mean={ranks.mean():.2f}, median={np.median(ranks)}, "
            f"Rank@1={rank1*100:.1f}%, Rank≤2={float((ranks<=2).mean())*100:.1f}%")
        log(f"  score margin: mean={margins.mean():.3f}, median={np.median(margins):.3f}, "
            f"M_disc>0={margin_pos*100:.1f}%")
        log(f"  ρ(score, log_det): Pearson={rho_p:+.3f} (p={p_p:.2e}), "
            f"Spearman={rho_s:+.3f} (p={p_s:.2e})")
        if above_chance is not None:
            log(f"  P(score > chance): {above_chance*100:.1f}%")
        log(f"  uniqueness: avg ratio={unique_ratio:.3f}, "
            f"all-same ASINs={n_all_same}/{len(by_asin_q)}")
        if abs(rho_s) > 0.3:
            log(f"  ⚠ 仍 σ-confounded (|ρ_s|>0.3)")
        else:
            log(f"  ✓ deconfounded (|ρ_s|<0.3)")

    log("\n=== 6. GO verdict ===")
    verdict = {}
    for var_name in ["a_v6f_posterior_baseline", "d_v6h_L2_margin"]:
        if var_name not in summary:
            continue
        s = summary[var_name]
        deconfounded = abs(s["rho_spearman_score_vs_logdet"]) < 0.3
        rank1_improved = s["rank1_fraction"] > 0.342
        margin_improved = s["margin_positive_fraction"] > 0.342
        go = deconfounded and rank1_improved and margin_improved
        verdict[var_name] = {
            "deconfounded": deconfounded,
            "rank1_improved_over_v6f": rank1_improved,
            "margin_improved_over_v6f": margin_improved,
            "go": go,
        }
        log(f"  {var_name}: deconfounded={deconfounded}, "
            f"rank1_improved={rank1_improved}, margin_improved={margin_improved}, "
            f"GO={go}")

    OUT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/select_query/v6h_ablation.json")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": ("Stage 4 v6h: Gaussian Gate (Mahal² ≤ 65.17) + L2 Margin "
                                "(M_L2 = min_other ||z-μ|| − ||z-μ_u||). 彻底拆分合法性 gate "
                                "(用 per-user σ) 与 exclusivity score (用 PCA48 z-space L2, "
                                "完全无 σ)。GO 条件: |ρ(M_L2, log_det)|<0.3 + Rank@1>34.2% + "
                                "M_L2>0>34.2%。"),
                "MAHAL_THRESHOLD": float(mahal_threshold),
                "MAHAL_THRESHOLD_CHI2_PPF": MAHAL_THRESHOLD_CHI2_PPF,
                "MAHAL_THRESHOLD_DF": MAHAL_THRESHOLD_DF,
                "l2_space": "PCA48 z (StandardScaler.transform(x) @ pca.components_[0:48])",
            },
            "summary": summary,
            "verdict": verdict,
        }, f, ensure_ascii=False, indent=2)
    log(f"\nwrote → {OUT}")


if __name__ == "__main__":
    main()