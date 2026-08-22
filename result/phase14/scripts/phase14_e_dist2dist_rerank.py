#!/usr/bin/env python3
"""Phase 14.E: Distribution-to-distribution rerank (KL / Wasserstein-2 / Bhattacharyya).

Idea (user-provided): each user has prior N(μ_user, Σ_user_diag). For each (pair, cond),
the K=8 query candidates form a distribution N(μ_cand, Σ_cand_diag). Rerank by
distance between user prior and cand distribution (not point-to-point).

Why this is different from Phase 14.C/D:
  - Phase 14.C: point rerank — d(cand_i, μ_u) per single cand, best-of-K
  - Phase 14.D: point-to-distribution — log P(cand_i | N_u) per single cand
  - Phase 14.E: distribution-to-distribution — D(N_u, N_cand) over K=8 cand as one distribution

Distances (diagonal Gaussians):
  KL(N_p || N_q) = Σ_d [ log(σ_q/σ_p) + (σ_p² + (μ_p-μ_q)²)/(2σ_q²) - 0.5 ]
  W2²(N_p, N_q) = Σ_d [ (μ_p-μ_q)² + (σ_p - σ_q)² ]
  BC(N_p, N_q)  = Σ_d [ 1/8 * (σ_p²/σ_q² + σ_q²/σ_p²) + (μ_p-μ_q)²/(σ_p²+σ_q²) ] * log(2)

Pipeline:
  1. Reuse Phase 14.C candidates (720 records, 3 conds × 30 pairs × K=8)
  2. Load phase11_a_user_gaussians_768d.npz (876 × μ + σ_diag)
  3. Per (pair, cond): build cand distribution (shrunk toward target σ)
  4. Compute distances to all 876 users; rank target
  5. Per-cond aggregate (rank-1, top-10, top-100, mean rank)
  6. Paired bootstrap: KL/W2/BC vs pooled Maha (Phase 14.C baseline) vs per-user log-lik (Phase 14.D)
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

IN_JSONL = OUT_DIR / "phase14_b_layer_alpha_sweep.jsonl"
IN_USER_GAUSS = OUT_DIR / "phase11_a_user_gaussians_768d.npz"
IN_CAND_EMBS = OUT_DIR / "phase14_c_a22_cand_embs_768d.npy"

OUT_EVAL = OUT_DIR / "phase14_e_dist2dist_rerank_eval.json"
OUT_PER_PAIR = OUT_DIR / "phase14_e_dist2dist_rerank_per_pair.jsonl"
OUT_META = OUT_DIR / "phase14_e_dist2dist_rerank_meta.json"

CONDITIONS = ["A22_a0.5", "A14_a1.0", "D_off"]
SEED = 42
N_BOOTSTRAP = 2000
SIGMA_FLOOR = 1e-4
SHRINKAGE_ALPHA = 2.0  # σ_cand_shrunk = (K-2)/(K+α) * σ_cand + α/(K+α) * σ_user[target]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bootstrap_ci(values: list[float], n: int = N_BOOTSTRAP, seed: int = SEED):
    if not values:
        return 0.0, (0.0, 0.0)
    arr = np.array(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n_obs = len(arr)
    boot_means = []
    for _ in range(n):
        idx = rng.choice(n_obs, size=n_obs, replace=True)
        boot_means.append(float(arr[idx].mean()))
    bm = np.array(boot_means)
    return float(arr.mean()), (float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5)))


def main() -> None:
    log("=" * 70)
    log("Phase 14.E: distribution-to-distribution rerank (KL/W2/Bhattacharyya)")
    log("=" * 70)

    # === [1] Load candidates (same as Phase 14.C/D) ===
    log("[1] Loading candidates (A22_a0.5 + A14_a1.0 + D_off) ...")
    candidates: list[dict] = []
    with IN_JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["condition"] in CONDITIONS:
                candidates.append(r)
    log(f"  filtered candidates: {len(candidates)} (expected 720)")
    n_pairs = len({(c["user_id"], c["asin"]) for c in candidates})
    log(f"  pairs: {n_pairs}")

    # === [2] Load cand embs (cached) ===
    log("[2] Loading cached cand embs ...")
    cand_embs = np.load(IN_CAND_EMBS).astype(np.float32)
    log(f"  cand_embs: {cand_embs.shape}")
    if cand_embs.shape[0] != len(candidates):
        raise ValueError(f"cand embs rows {cand_embs.shape[0]} != len(candidates) {len(candidates)}")

    # === [3] Load user Gaussian prior ===
    log("[3] Loading per-user Gaussian (μ_768 + σ_diag) ...")
    npz = np.load(IN_USER_GAUSS, allow_pickle=True)
    user_ids_arr = list(npz["user_ids"])
    mu_user = npz["mu_768"].astype(np.float32)              # (876, 768)
    sigma_user = npz["sigma_diag"].astype(np.float32)       # (876, 768) variance per dim
    n_users = len(user_ids_arr)
    log(f"  users: {n_users}, emb_dim={mu_user.shape[1]}")
    log(f"  σ²_user stats: mean={sigma_user.mean():.4f}, min={sigma_user.min():.6f}, max={sigma_user.max():.4f}")

    uid_to_useridx = {u: i for i, u in enumerate(user_ids_arr)}

    # Floor user σ² to avoid degeneracy
    sigma_user_safe = np.maximum(sigma_user, SIGMA_FLOOR)

    # === [4] Pooled Mahalanobis (Phase 14.C baseline) ===
    log("[4] Pooled Mahalanobis (LW shrinkage) baseline ...")
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(mu_user)
    cov_shrunk = lw.covariance_
    shrinkage = float(lw.shrinkage_)
    eigvals = np.linalg.eigvalsh(cov_shrunk)
    inv_cov = np.linalg.inv(cov_shrunk + 1e-6 * np.eye(cov_shrunk.shape[0]))
    quad_user = np.einsum('ij,jk,ik->i', mu_user, inv_cov, mu_user)
    log(f"  shrinkage: {shrinkage:.4f}")

    # === [5] Group by (uid, asin, cond) ===
    log("[5] Grouping candidates by (uid, asin, cond) ...")
    cand_by_pair_cond = defaultdict(list)  # (uid, asin, cond) → list of cand indices
    for ci, c in enumerate(candidates):
        key = (c["user_id"], c["asin"], c["condition"])
        cand_by_pair_cond[key].append(ci)
    log(f"  (uid, asin, cond) groups: {len(cand_by_pair_cond)}")

    # === [6] Per-pair-per-cond: build cand distribution + compute distances + rank ===
    log("[6] Per-pair rerank (pooled Maha point + KL/W2/BC distribution) ...")
    per_pair_per_cond: dict[tuple[str, str, str], dict[str, int]] = {}

    n_skipped = 0
    for key, cands_idx in cand_by_pair_cond.items():
        uid, asin, cond = key
        target_idx = uid_to_useridx.get(uid)
        if target_idx is None:
            n_skipped += 1
            continue
        K_local = cand_embs[cands_idx]  # (K, 768)
        K = K_local.shape[0]

        # --- Cand distribution ---
        mu_cand = K_local.mean(axis=0)                  # (768,)
        sigma_cand = K_local.var(axis=0, ddof=0)        # (768,) variance per dim
        # Shrinkage towards target user σ
        shrink_w = SHRINKAGE_ALPHA / (K + SHRINKAGE_ALPHA)
        sigma_cand_shrunk = (1 - shrink_w) * sigma_cand + shrink_w * sigma_user_safe[target_idx]
        sigma_cand_safe = np.maximum(sigma_cand_shrunk, SIGMA_FLOOR)  # (768,)

        # --- Pooled Mahalanobis point rerank (best-of-K) ---
        # Phase 14.C baseline: best rank over K=8 individual cand embs
        best_maha_rank = 10**9
        for k in range(K):
            v = K_local[k]
            maha_d_k = v @ inv_cov @ mu_user.T - quad_user  # (876,)
            maha_d_full = quad_user * 0  # placeholder
            quad_v = np.einsum('i,ij,j->', v, inv_cov, v)
            maha_d = quad_v + quad_user - 2 * v @ inv_cov @ mu_user.T
            r = int((maha_d < maha_d[target_idx]).sum())
            if r < best_maha_rank:
                best_maha_rank = r

        # --- KL(N_u || N_cand): per user sum over 768 dims ---
        # KL = sum_d [ log(sigma_cand_d/sigma_user_u_d)
        #            + (sigma_user_u_d^2 + (mu_user_u_d - mu_cand_d)^2)/(2*sigma_cand_d^2)
        #            - 0.5 ]
        # shape (876, 768) for mu_user, broadcast with mu_cand (768,)
        diff_mu = mu_user - mu_cand[None, :]                         # (876, 768)
        diff_mu_sq = diff_mu ** 2                                    # (876, 768)
        # KL term 1: log(sigma_cand / sigma_user_safe)
        log_term = np.log(sigma_cand_safe[None, :]) - np.log(sigma_user_safe)  # (876, 768)
        # KL term 2: (sigma_user^2 + diff^2) / (2 * sigma_cand^2)
        sq_term = (sigma_user_safe ** 2 + diff_mu_sq) / (2 * sigma_cand_safe[None, :] ** 2)
        # KL term 3: -0.5 (constant)
        kl_per_dim = log_term + sq_term - 0.5                          # (876, 768)
        kl_per_user = kl_per_dim.sum(axis=1)                          # (876,)
        kl_target = kl_per_user[target_idx]
        kl_rank = int((kl_per_user < kl_target).sum())

        # --- W2²(N_u, N_cand): per-dim (μ diff)^2 + (σ diff)^2 ---
        sigma_user_sqrt = np.sqrt(sigma_user_safe)                      # (876, 768)
        sigma_cand_sqrt = np.sqrt(sigma_cand_safe)                     # (768,)
        w2_per_dim = diff_mu_sq + (sigma_user_sqrt - sigma_cand_sqrt[None, :]) ** 2  # (876, 768)
        w2_per_user = w2_per_dim.sum(axis=1)                            # (876,)
        w2_target = w2_per_user[target_idx]
        w2_rank = int((w2_per_user < w2_target).sum())

        # --- Bhattacharyya distance: exp(-BC) closer to 1 = more similar ---
        # BC_d = 1/8 * (σ_p²/σ_q² + σ_q²/σ_p²) + (μ_p-μ_q)²/(σ_p² + σ_q²)  * log(2)
        sigma_p_sq = sigma_user_safe                                    # (876, 768)
        sigma_q_sq = sigma_cand_safe[None, :]                           # (1, 768)
        ratio = sigma_p_sq / sigma_q_sq                                 # (876, 768)
        bc_ratio_term = 0.125 * (ratio + 1.0 / ratio)                   # (876, 768)
        bc_diff_term = diff_mu_sq / (sigma_p_sq + sigma_q_sq)           # (876, 768)
        bc_per_dim = (bc_ratio_term + bc_diff_term) * np.log(2.0)       # (876, 768)
        bc_per_user = bc_per_dim.sum(axis=1)                            # (876,)
        bc_target = bc_per_user[target_idx]
        # exp(-BC) closer to 1 = more similar → rank by -BC (lower BC better) → rank by BC asc
        bc_rank = int((bc_per_user < bc_target).sum())

        per_pair_per_cond[key] = {
            "maha_pooled": int(best_maha_rank),
            "kl_dist": int(kl_rank),
            "w2_dist": int(w2_rank),
            "bc_dist": int(bc_rank),
        }
    log(f"  done; skipped pairs: {n_skipped}")

    # === [7] Aggregate per (cond, metric) ===
    log("[7] Per-cond × per-metric aggregate ...")
    pair_keys = sorted({(uid, asin) for (uid, asin, _) in per_pair_per_cond.keys()})
    metrics = ["maha_pooled", "kl_dist", "w2_dist", "bc_dist"]

    per_cond_eval: dict[str, dict] = {}
    for cond in CONDITIONS:
        per_cond_eval[cond] = {}
        for metric in metrics:
            ranks = []
            for (uid, asin) in pair_keys:
                r = per_pair_per_cond.get((uid, asin, cond), {}).get(metric)
                if r is not None:
                    ranks.append(r)
            ranks_arr = np.array(ranks)
            mean_r, ci_r = bootstrap_ci(ranks)
            per_cond_eval[cond][metric] = {
                "n_pairs": len(ranks),
                "rank1_coverage": float((ranks_arr == 0).sum()) / max(1, len(ranks_arr)),
                "top10_coverage": float((ranks_arr < 10).sum()) / max(1, len(ranks_arr)),
                "top100_coverage": float((ranks_arr < 100).sum()) / max(1, len(ranks_arr)),
                "mean_best_rank": mean_r,
                "mean_best_rank_ci95": list(ci_r),
            }
            log(
                f"  {cond}/{metric:12}: n={len(ranks)}, "
                f"rank1={per_cond_eval[cond][metric]['rank1_coverage']*100:.1f}%, "
                f"top10={per_cond_eval[cond][metric]['top10_coverage']*100:.1f}%, "
                f"top100={per_cond_eval[cond][metric]['top100_coverage']*100:.1f}%, "
                f"mean_rank={mean_r:.1f} CI [{ci_r[0]:.1f},{ci_r[1]:.1f}]"
            )

    # === [8] Paired bootstrap: dist metrics vs pooled Maha baseline ===
    log("[8] Paired bootstrap: distribution metrics vs pooled Maha baseline ...")
    diffs_vs_maha: dict[str, dict] = {}
    for metric in ["kl_dist", "w2_dist", "bc_dist"]:
        diffs_vs_maha[metric] = {}
        for cond in CONDITIONS:
            d_list = []
            for (uid, asin) in pair_keys:
                r_dist = per_pair_per_cond.get((uid, asin, cond), {}).get(metric)
                r_maha = per_pair_per_cond.get((uid, asin, cond), {}).get("maha_pooled")
                if r_dist is not None and r_maha is not None:
                    d_list.append(int(r_maha) - int(r_dist))
            mean_d, ci_d = bootstrap_ci(d_list)
            diffs_vs_maha[metric][cond] = {
                "mean_rank_diff": mean_d,
                "ci95": list(ci_d),
                "ci_excludes_0": ci_d[0] > 0,
            }
            log(
                f"  {cond}/{metric}: diff_vs_maha={mean_d:+.1f} CI {ci_d[0]:+.1f}/{ci_d[1]:+.1f} "
                f"({'excl 0' if ci_d[0] > 0 else 'incl 0'})"
            )

    # === [9] Per-metric: A22_a0.5 vs A14_a1.0 vs D_off ===
    log("[9] Per-metric paired: A22 vs A14 vs D_off ...")
    cond_diffs: dict[str, dict] = {}
    for metric in metrics:
        cond_diffs[metric] = {}
        for cond_b in CONDITIONS:
            if cond_b == "A22_a0.5":
                continue
            d_list = []
            for (uid, asin) in pair_keys:
                r_a = per_pair_per_cond.get((uid, asin, "A22_a0.5"), {}).get(metric)
                r_b = per_pair_per_cond.get((uid, asin, cond_b), {}).get(metric)
                if r_a is not None and r_b is not None:
                    d_list.append(int(r_b) - int(r_a))
            mean_d, ci_d = bootstrap_ci(d_list)
            cond_diffs[metric][f"A22_vs_{cond_b}"] = {
                "mean_rank_diff": mean_d,
                "ci95": list(ci_d),
                "ci_excludes_0": ci_d[0] > 0,
            }

    # === [10] Verdict ===
    log("[10] Verdict ...")
    # Any cond's distribution metric beats pooled Maha (mean rank diff CI excludes 0)?
    winners = []
    for metric in ["kl_dist", "w2_dist", "bc_dist"]:
        for cond in CONDITIONS:
            if diffs_vs_maha[metric][cond]["ci_excludes_0"]:
                winners.append((metric, cond, diffs_vs_maha[metric][cond]["mean_rank_diff"]))
    if winners:
        verdict = "PARTIAL-GO"
        winners_str = "; ".join([f"{m}/{c}: diff={d:+.1f}" for m, c, d in winners])
        reason = f"Distribution-to-distribution rerank beats pooled Maha on: {winners_str}"
    else:
        verdict = "NO-GO"
        reason = (
            "Distribution-to-distribution rerank (KL/W2/Bhattacharyya) does NOT significantly beat "
            "pooled Mahalanobis on any cond. Point rerank remains best."
        )
    log(f"  VERDICT: {verdict}")
    log(f"  REASON: {reason}")

    # === [11] Save ===
    out = {
        "phase": "14.E",
        "encoder": "AnnaWegmann/Style-Embedding (768d)",
        "n_candidates": len(candidates),
        "n_pairs": len(pair_keys),
        "n_conditions": len(CONDITIONS),
        "conditions": CONDITIONS,
        "k_per_pair": 8,
        "metrics": metrics,
        "cov_pooled_method": "Ledoit-Wolf shrinkage",
        "cov_shrinkage": shrinkage,
        "sigma_floor": SIGMA_FLOOR,
        "shrinkage_alpha": SHRINKAGE_ALPHA,
        "per_cond_eval": per_cond_eval,
        "dists_vs_pooled_maha_diffs": diffs_vs_maha,
        "cond_diffs_per_metric": cond_diffs,
        "comparison_targets": {
            "Phase14_C_A14_a1.0_pooled_maha": {"top100": 0.533, "mean_rank": 164.0},
            "Phase14_C_A22_a0.5_pooled_maha": {"top100": 0.433, "mean_rank": 176.6},
            "Phase14_D_A14_a1.0_per_user_loglik": {"top100": 0.133, "mean_rank": 272.8},
        },
        "note": (
            "Per (pair, cond): K=8 query embs → N(μ_cand, σ_cand_diag_shrunk). "
            "Distance to each user prior N(μ_u, σ_u_diag): KL/W2/Bhattacharyya. "
            f"σ_cand shrunk towards σ_user[target] with α={SHRINKAGE_ALPHA} to reduce K=8 variance noise. "
            "Rank target = #{u | D(u, cand) < D(target, cand)}."
        ),
        "decision_logic": {
            "PARTIAL-GO": "distribution-dist beats pooled Mah > pooled Mah on any (metric, cond) (CI excludes 0)",
            "NO-GO": "no significant improvement on any (metric, cond)",
        },
        "verdict": verdict,
        "verdict_reason": reason,
    }
    OUT_EVAL.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log(f"  eval → {OUT_EVAL}")

    with OUT_PER_PAIR.open("w") as f:
        for (uid, asin) in pair_keys:
            row = {"user_id": uid, "asin": asin}
            for cond in CONDITIONS:
                for metric in metrics:
                    row[f"{cond}_{metric}_rank"] = per_pair_per_cond.get((uid, asin, cond), {}).get(metric)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"  per_pair → {OUT_PER_PAIR}")

    meta = {
        "phase": "14.E",
        "encoder": "AnnaWegmann/Style-Embedding (768d)",
        "n_pairs": len(pair_keys),
        "n_conditions": len(CONDITIONS),
        "conditions": CONDITIONS,
        "metrics": metrics,
        "cov_shrinkage": shrinkage,
        "sigma_floor": SIGMA_FLOOR,
        "shrinkage_alpha": SHRINKAGE_ALPHA,
        "verdict": verdict,
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log(f"  meta → {OUT_META}")

    log("=" * 70)
    log("PHASE 14.E COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()