"""Stage 4 v6g — Covariance-deconfounded Discriminative Selection ablation.

用户指令 2026-08-28: Stage 4 v6f posterior 公式有 σ-confounded bug
(ρ(P(u|q), log|Σ_u|) = -0.735)。做 3 variant 并排 ablation:

Variant A: v6f current posterior (per-user Σ) — 失败 baseline
Variant B: contrastive Mahalanobis
            S(q, u) = mean_{v≠u} Mahal²(q, v) − Mahal²(q, u)
            argmax subject to Mahal²(q, u) ≤ χ²(0.95, 48)
Variant C: shared-Σ posterior
            pooled_σ_diag = mean(σ_diag over all 4324 users)
            用 pooled_Σ 算 Mahal² + softmax over users
            (log|Σ_u| 项是常数 → 取消 σ-confounding)

GO 条件:
  1. |ρ(score, log_det(Σ_u))| < 0.3  (deconfounded)
  2. Rank@1 > 34.2% (当前 v6f)
  3. M_disc > 0 比例 > 34.2%

**为什么 v6f posterior 与 log_det 相关**:
  log p(z|u) = -0.5 * Mahal²(z, u) − 0.5 * log_det(Σ_u)
  窄 Gaussian (small σ) → log p 系统性偏高 → softmax 吞掉 mass

**为什么 density ratio 不能 deconfound**:
  log p(z|u) − log mean_v p(z|v) 仍包含 log|Σ_u| 项
  (用户原命题修正: 必须 shared-Σ 才能彻底 deconfound)

输入:
  - ASINS_IN / POOL_IN / GAUSSIANS_IN / FEAT_CACHE
输出:
  - result/select_query/v6g_ablation.json
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
    log("=== Stage 4 v6g — Deconfounded Discriminative Selection ablation ===")

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

    # Compute global pooled_σ_diag (over all user Gaussians)
    pooled_sigma_diag = np.zeros(PCA_DIM)
    for uid, g in users_gauss.items():
        pooled_sigma_diag += np.array(g["sigma_diag"])
    pooled_sigma_diag /= len(users_gauss)
    log(f"\n=== Global pooled_σ_diag (mean over {len(users_gauss)} users) ===")
    log(f"  per-dim mean = {pooled_sigma_diag.mean():.4f}, "
        f"range = [{pooled_sigma_diag.min():.4f}, {pooled_sigma_diag.max():.4f}]")

    log("\n=== 4. Per-ASIN selection & audit ===")
    mahal_threshold = chi2.ppf(MAHAL_THRESHOLD_CHI2_PPF, MAHAL_THRESHOLD_DF)
    log(f"  Mahal² gating threshold = χ²({MAHAL_THRESHOLD_CHI2_PPF}, "
        f"{MAHAL_THRESHOLD_DF}) = {mahal_threshold:.3f}")

    # Group entries by asin (use asin_data which has users_sampled per asin)
    by_asin_users = collections.OrderedDict()
    for entry in asin_data:
        by_asin_users[entry["asin"]] = list(entry["users_sampled"])
    log(f"  total ASINs in asin_data: {len(by_asin_users)}")

    variant_results = {
        "a_posterior": {
            "selected_entries": [], "scores": [], "log_dets": [],
        },
        "b_contrastive_maha": {
            "selected_entries": [], "scores": [], "log_dets": [],
        },
        "c_shared_cov_posterior": {
            "selected_entries": [], "scores": [], "log_dets": [],
        },
    }

    n_asin_skip_no_pool = 0
    n_asin_skip_no_strict = 0
    n_asin_skip_lt_2_users = 0
    n_asin_processed = 0
    n_entries_selected_per_variant = 0

    for asin_idx, (asin, user_ids) in enumerate(by_asin_users.items()):
        if asin_idx % 100 == 0:
            log(f"  ... ASIN {asin_idx}/{len(by_asin_users)}")
        zqs = pool_z.get(asin)
        if zqs is None:
            n_asin_skip_no_pool += 1
            continue
        strict_zqs = [(z, q) for z, q in zqs if q["strict"]]
        if not strict_zqs:
            n_asin_skip_no_strict += 1
            continue

        # Filter user_ids that have Gaussians
        asin_users = []
        for uid in user_ids:
            if uid not in users_gauss:
                continue
            mu = np.array(users_gauss[uid]["mu"])
            sigma = np.array(users_gauss[uid]["sigma_diag"])
            asin_users.append((uid, mu, sigma))
        if len(asin_users) < 2:
            n_asin_skip_lt_2_users += 1
            continue

        n_users = len(asin_users)
        n_queries = len(strict_zqs)

        # Mahal² matrix (per-user Σ) — for variant A and gating
        mahal = np.zeros((n_users, n_queries))
        log_det = np.zeros(n_users)
        for ui, (uid, mu, sigma) in enumerate(asin_users):
            log_det[ui] = float(np.sum(np.log(sigma)))
            for qi, (z, _) in enumerate(strict_zqs):
                mahal[ui, qi] = mahalanobis_sq(z, mu, sigma)

        # Mahal² matrix (shared Σ) — for variant C
        mahal_shared = np.zeros((n_users, n_queries))
        for ui, (uid, mu, _) in enumerate(asin_users):
            for qi, (z, _) in enumerate(strict_zqs):
                mahal_shared[ui, qi] = mahalanobis_sq(z, mu, pooled_sigma_diag)

        # Variant A: current posterior (full Gaussian likelihood)
        log_p_a = -0.5 * mahal - 0.5 * log_det[:, None]
        log_p_a_max = log_p_a.max(axis=0, keepdims=True)
        log_p_a_shifted = log_p_a - log_p_a_max
        posterior_a = np.exp(log_p_a_shifted) / np.exp(log_p_a_shifted).sum(axis=0, keepdims=True)

        # Variant B: contrastive Mahalanobis score
        # S(q, u) = mean_{v≠u} Mahal²(q, v) − Mahal²(q, u)
        # Higher = more discriminative for u
        sum_all = mahal.sum(axis=0, keepdims=True)  # (1, n_queries)
        mean_other = (sum_all - mahal) / max(n_users - 1, 1)
        contrast_score = mean_other - mahal  # (n_users, n_queries)

        # Variant C: shared-Σ posterior
        # log p_shared(z|u) ∝ -0.5 * Mahal²_shared  (log_det_shared 是常数,softmax 抵消)
        log_p_c = -0.5 * mahal_shared
        log_p_c_max = log_p_c.max(axis=0, keepdims=True)
        log_p_c_shifted = log_p_c - log_p_c_max
        posterior_c = np.exp(log_p_c_shifted) / np.exp(log_p_c_shifted).sum(axis=0, keepdims=True)

        K = n_users
        chance = 1.0 / K
        n_asin_processed += 1

        for ui, (uid, mu, sigma) in enumerate(asin_users):
            in_dist_mask = mahal[ui] <= mahal_threshold
            if not in_dist_mask.any():
                continue

            # Variant A selection: argmax posterior_a subject to gate
            avail_post_a = np.where(in_dist_mask, posterior_a[ui], -np.inf)
            best_qi_a = int(np.argmax(avail_post_a))
            target_post_a = float(posterior_a[ui][best_qi_a])
            sorted_post_a = np.sort(posterior_a[:, best_qi_a])[::-1]
            rank_a = int((sorted_post_a >= target_post_a - 1e-12).sum())
            other_post_a = np.delete(posterior_a[:, best_qi_a], ui)
            margin_a = target_post_a - float(other_post_a.max())
            best_mahal_a = float(mahal[ui][best_qi_a])
            ld = float(log_det[ui])

            variant_results["a_posterior"]["selected_entries"].append({
                "asin": asin,
                "user_id": uid,
                "selected_query": strict_zqs[best_qi_a][1]["query"][:80],
                "selected_distance": best_mahal_a,
                "selected_score": target_post_a,
                "target_rank": rank_a,
                "score_margin": margin_a,
                "log_det": ld,
                "K": K,
                "chance": chance,
            })

            # Variant B selection: argmax contrast_score subject to gate
            avail_contrast = np.where(in_dist_mask, contrast_score[ui], -np.inf)
            best_qi_b = int(np.argmax(avail_contrast))
            target_contrast = float(contrast_score[ui][best_qi_b])
            sorted_contrast = np.sort(contrast_score[:, best_qi_b])[::-1]
            rank_b = int((sorted_contrast >= target_contrast - 1e-12).sum())
            other_contrast = np.delete(contrast_score[:, best_qi_b], ui)
            margin_b = target_contrast - float(other_contrast.max())
            best_mahal_b = float(mahal[ui][best_qi_b])

            variant_results["b_contrastive_maha"]["selected_entries"].append({
                "asin": asin,
                "user_id": uid,
                "selected_query": strict_zqs[best_qi_b][1]["query"][:80],
                "selected_distance": best_mahal_b,
                "selected_score": target_contrast,
                "target_rank": rank_b,
                "score_margin": margin_b,
                "log_det": ld,
                "K": K,
            })

            # Variant C selection: argmax posterior_c subject to gate
            avail_post_c = np.where(in_dist_mask, posterior_c[ui], -np.inf)
            best_qi_c = int(np.argmax(avail_post_c))
            target_post_c = float(posterior_c[ui][best_qi_c])
            sorted_post_c = np.sort(posterior_c[:, best_qi_c])[::-1]
            rank_c = int((sorted_post_c >= target_post_c - 1e-12).sum())
            other_post_c = np.delete(posterior_c[:, best_qi_c], ui)
            margin_c = target_post_c - float(other_post_c.max())
            best_mahal_c = float(mahal[ui][best_qi_c])

            variant_results["c_shared_cov_posterior"]["selected_entries"].append({
                "asin": asin,
                "user_id": uid,
                "selected_query": strict_zqs[best_qi_c][1]["query"][:80],
                "selected_distance": best_mahal_c,
                "selected_score": target_post_c,
                "target_rank": rank_c,
                "score_margin": margin_c,
                "log_det": ld,
                "K": K,
                "chance": chance,
            })
            n_entries_selected_per_variant += 1

    log(f"\n  ASINs processed (>=2 users with gauss, has strict pool): {n_asin_processed}")
    log(f"  ASINs skipped: no_pool={n_asin_skip_no_pool}, no_strict={n_asin_skip_no_strict}, "
        f"lt_2_users={n_asin_skip_lt_2_users}")
    log(f"  entries selected per variant: {n_entries_selected_per_variant}")

    log("\n=== 5. Per-variant aggregate ===")
    summary = {}
    for var_name in ["a_posterior", "b_contrastive_maha", "c_shared_cov_posterior"]:
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
            "selected_mahal_std": float(mahals.std()),
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
            f"std={mahals.std():.3f}")
        log(f"  score: mean={scores.mean():.3f}, median={np.median(scores):.3f}, std={scores.std():.3f}")
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

    log("\n=== 6. GO verdict per variant ===")
    verdict = {}
    for var_name in ["a_posterior", "b_contrastive_maha", "c_shared_cov_posterior"]:
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
            "margin_positive_improved_over_v6f": margin_improved,
            "go": go,
        }
        log(f"  {var_name}: deconfounded={deconfounded}, "
            f"rank1_improved={rank1_improved}, margin_improved={margin_improved}, "
            f"GO={go}")

    OUT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/select_query/v6g_ablation.json")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": ("Stage 4 v6g: 3-variant ablation — current posterior (a, "
                                "v6f baseline), contrastive Mahalanobis (b), shared-Σ "
                                "posterior (c). GO conditions: |ρ(score, log_det)|<0.3 + "
                                "Rank@1 > 34.2% + M_disc>0 > 34.2%."),
                "MAHAL_THRESHOLD": float(mahal_threshold),
                "MAHAL_THRESHOLD_CHI2_PPF": MAHAL_THRESHOLD_CHI2_PPF,
                "MAHAL_THRESHOLD_DF": MAHAL_THRESHOLD_DF,
                "pooled_sigma_diag_mean": float(pooled_sigma_diag.mean()),
                "pooled_sigma_diag_range": [float(pooled_sigma_diag.min()),
                                              float(pooled_sigma_diag.max())],
                "n_pooled_users": len(users_gauss),
            },
            "summary": summary,
            "verdict": verdict,
        }, f, ensure_ascii=False, indent=2)
    log(f"\nwrote → {OUT}")


if __name__ == "__main__":
    main()