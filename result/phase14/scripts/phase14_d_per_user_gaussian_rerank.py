#!/usr/bin/env python3
"""Phase 14.D: per-user Gaussian rerank (log-likelihood) vs pooled Mahalanobis.

Hypothesis: per-user σ²_diag captures user-specific style variance. A query
candidate landing inside target user's σ²-ellipsoid should be ranked higher
than a candidate inside some random user's ellipsoid.

Score per (cand x, user u):
    log P(x | N(μ_u, diag(σ²_u))) = -0.5 * Σ_d (x_d - μ_u,d)² / σ²_u,d - 0.5 * Σ_d log σ²_u,d
Lower score = higher likelihood = better match.

Compare against:
  - Pooled Mahalanobis (Phase 14 baseline)
  - Cosine (Phase 14 baseline)

Pipeline:
  1. Reuse Phase 14.C candidates (A22_a0.5 + A14_a1.0 + D_off on 30 pairs, 720 records)
  2. Load phase11_a_user_gaussians_768d.npz (876 × μ + σ²_diag)
  3. For each (pair, cond): best-of-K rerank under 3 metrics
       - cosine (Phase 14)
       - pooled Maha (Phase 14)
       - per-user log-likelihood (NEW)
  4. Aggregate per cond: rank-1 / top-10 / top-100 / mean rank
  5. Paired bootstrap diff: per-user log-likelihood vs pooled Maha vs cosine
  6. Verdict
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
IN_USER_EMBS = OUT_DIR / "phase10_user_embs_768d.npz"
IN_USER_GAUSS = OUT_DIR / "phase11_a_user_gaussians_768d.npz"
IN_CAND_EMBS = OUT_DIR / "phase14_c_a22_cand_embs_768d.npy"

OUT_EVAL = OUT_DIR / "phase14_d_per_user_gaussian_rerank_eval.json"
OUT_PER_PAIR = OUT_DIR / "phase14_d_per_user_gaussian_rerank_per_pair.jsonl"
OUT_META = OUT_DIR / "phase14_d_per_user_gaussian_rerank_meta.json"

CONDITIONS = ["A22_a0.5", "A14_a1.0", "D_off"]
SEED = 42
N_BOOTSTRAP = 2000
SIGMA_FLOOR = 1e-4  # avoid div-by-zero in degenerate dims


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


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
    log("Phase 14.D: per-user Gaussian log-likelihood rerank vs pooled Maha")
    log("=" * 70)

    # === [1] Load candidates (same as Phase 14.C) ===
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

    # === [2] Load cand embs (reuse Phase 14.C cache) ===
    log("[2] Loading cached cand embs ...")
    cand_embs = np.load(IN_CAND_EMBS).astype(np.float32)
    log(f"  cand_embs: {cand_embs.shape}")
    if cand_embs.shape[0] != len(candidates):
        raise ValueError(f"cand embs rows {cand_embs.shape[0]} != len(candidates) {len(candidates)}")

    # === [3] Load user 768d (μ_768 + σ_diag) ===
    log("[3] Loading per-user Gaussian (μ_768 + σ_diag) ...")
    npz = np.load(IN_USER_GAUSS, allow_pickle=True)
    user_ids_arr = list(npz["user_ids"])
    mu_user = npz["mu_768"].astype(np.float32)            # (876, 768)
    sigma_diag = npz["sigma_diag"].astype(np.float32)     # (876, 768) variance per dim
    valid_mask = npz["valid"].astype(bool) if "valid" in npz.files else np.ones(len(user_ids_arr), dtype=bool)
    n_users = len(user_ids_arr)
    log(f"  users: {n_users} (valid: {valid_mask.sum()}), emb_dim={mu_user.shape[1]}")
    log(f"  σ² stats: mean={sigma_diag.mean():.4f}, min={sigma_diag.min():.6f}, max={sigma_diag.max():.4f}")

    # Verify same order as Phase 14.C uid_to_useridx
    npz10 = np.load(IN_USER_EMBS, allow_pickle=True)
    user_ids_phase10 = list(npz10["user_ids"])
    if user_ids_arr != user_ids_phase10:
        raise ValueError("phase11.A user_ids differ from phase10 user_ids order — index alignment broken")
    uid_to_useridx = {u: i for i, u in enumerate(user_ids_arr)}

    # === [4] Pooled Mahalanobis baseline ===
    log("[4] Pooled Mahalanobis (LW shrinkage) for baseline ...")
    user_embs = mu_user  # point estimate
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(user_embs)
    cov_shrunk = lw.covariance_
    shrinkage = float(lw.shrinkage_)
    eigvals = np.linalg.eigvalsh(cov_shrunk)
    log(f"  shrinkage: {shrinkage:.4f}, eigvals min/max: {eigvals.min():.4f}/{eigvals.max():.4f}")
    inv_cov = np.linalg.inv(cov_shrunk + 1e-6 * np.eye(cov_shrunk.shape[0]))
    quad_user = np.einsum('ij,jk,ik->i', user_embs, inv_cov, user_embs)

    # === [5] Per-user Gaussian log-likelihood precompute ===
    log("[5] Precomputing per-user log-likelihood constants ...")
    sigma_floor = max(SIGMA_FLOOR, float(sigma_diag.min()) * 0.5)
    sigma_safe = np.maximum(sigma_diag, sigma_floor)  # (876, 768)
    log_sigma_sum = np.log(sigma_safe).sum(axis=1)     # (876,) per-user log|Σ_u|
    log(f"  σ²_floor={sigma_floor:.6f}, mean log|Σ|={log_sigma_sum.mean():.2f}")

    # === [6] Cosine precompute ===
    log("[6] Cosine precompute ...")
    user_embs_norm = l2_normalize(user_embs)
    cand_embs_norm = l2_normalize(cand_embs)

    # === [7] Group by (uid, asin, cond) ===
    log("[7] Grouping candidates by (uid, asin, cond) ...")
    cand_by_pair_cond = defaultdict(lambda: defaultdict(list))
    for ci, c in enumerate(candidates):
        key = (c["user_id"], c["asin"])
        cand_by_pair_cond[key][c["condition"]].append(ci)
    log(f"  pairs: {len(cand_by_pair_cond)}")

    # === [8] Per-pair-per-cond rerank (3 metrics) ===
    log("[8] Per-pair rerank (cosine + pooled Maha + per-user log-likelihood) ...")
    per_pair_per_cond: dict[tuple[str, str, str], dict[str, int]] = {}

    n_skipped = 0
    for (uid, asin), conds_local in cand_by_pair_cond.items():
        target_idx = uid_to_useridx.get(uid)
        if target_idx is None:
            n_skipped += 1
            continue
        for cond, cand_indices in conds_local.items():
            local_embs = cand_embs[cand_indices]   # (K, 768)
            local_norm = cand_embs_norm[cand_indices]

            # --- cosine: lower distance = better, so rank by distance asc ---
            cos_d = 1.0 - local_norm @ user_embs_norm.T
            target_d_cos = cos_d[:, target_idx:target_idx + 1]
            cos_rank_per_cand = (cos_d < target_d_cos).sum(axis=1)
            best_cos_rank = int(np.min(cos_rank_per_cand))

            # --- pooled Maha: same direction (lower = better) ---
            quad_cand = np.einsum('ij,jk,ik->i', local_embs, inv_cov, local_embs)
            cross = local_embs @ inv_cov @ user_embs.T
            maha_d = quad_cand[:, None] + quad_user[None, :] - 2 * cross
            target_d_maha = maha_d[:, target_idx:target_idx + 1]
            maha_rank_per_cand = (maha_d < target_d_maha).sum(axis=1)
            best_maha_rank = int(np.min(maha_rank_per_cand))

            # --- per-user log-likelihood (lower score = higher likelihood = better) ---
            # score_u = -0.5 * Σ [(x-μ)²/σ² + log σ²]; rank by score asc (lower better)
            # For each cand: compute scores against all 876 users
            # x_d - μ_u,d shape: (K, 768) - (876, 768) → need broadcasting
            diff = local_embs[:, None, :] - mu_user[None, :, :]  # (K, 876, 768)
            sq_term = (diff ** 2) / sigma_safe[None, :, :]      # (K, 876, 768)
            sq_per_user = sq_term.sum(axis=2)                     # (K, 876)
            score = -0.5 * sq_per_user - 0.5 * log_sigma_sum[None, :]  # (K, 876)
            target_score = score[:, target_idx:target_idx + 1]   # (K, 1)
            # rank: #{u | score(u) < score(target)} = better match → lower score
            ll_rank_per_cand = (score < target_score).sum(axis=1)
            best_ll_rank = int(np.min(ll_rank_per_cand))

            per_pair_per_cond[(uid, asin, cond)] = {
                "cosine": best_cos_rank,
                "maha_pooled": best_maha_rank,
                "loglik_per_user": best_ll_rank,
            }
    log(f"  done; skipped pairs: {n_skipped}")

    # === [9] Aggregate per (cond, metric) ===
    log("[9] Per-cond × per-metric aggregate ...")
    pair_keys = sorted({(uid, asin) for (uid, asin, _) in per_pair_per_cond.keys()})
    metrics = ["cosine", "maha_pooled", "loglik_per_user"]

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
                f"  {cond}/{metric}: n={len(ranks)}, "
                f"rank1={per_cond_eval[cond][metric]['rank1_coverage']*100:.1f}%, "
                f"top10={per_cond_eval[cond][metric]['top10_coverage']*100:.1f}%, "
                f"top100={per_cond_eval[cond][metric]['top100_coverage']*100:.1f}%, "
                f"mean_rank={mean_r:.1f} CI [{ci_r[0]:.1f},{ci_r[1]:.1f}]"
            )

    # === [10] Paired bootstrap diff (per-user loglik vs pooled Maha) ===
    log("[10] Paired bootstrap: loglik_per_user vs pooled Maha vs cosine ...")
    diffs: dict[str, dict] = {}
    for metric_b in metrics:
        diffs[f"loglik_vs_{metric_b}"] = {}
        for cond in CONDITIONS:
            # For each pair: loglik_rank - other_rank (lower loglik rank = loglik better)
            # Positive diff = other_rank > loglik_rank = loglik ranks target higher
            d_list = []
            for (uid, asin) in pair_keys:
                r_ll = per_pair_per_cond.get((uid, asin, cond), {}).get("loglik_per_user")
                r_b = per_pair_per_cond.get((uid, asin, cond), {}).get(metric_b)
                if r_ll is not None and r_b is not None:
                    d_list.append(int(r_b) - int(r_ll))
            mean_d, ci_d = bootstrap_ci(d_list)
            diffs[f"loglik_vs_{metric_b}"][cond] = {
                "mean_rank_diff": mean_d,
                "ci95": list(ci_d),
                "ci_excludes_0": ci_d[0] > 0,
            }
            log(
                f"  {cond}: loglik_vs_{metric_b} rank_diff={mean_d:+.1f} "
                f"CI {ci_d[0]:+.1f}/{ci_d[1]:+.1f} "
                f"({'excl 0' if ci_d[0] > 0 else 'incl 0'})"
            )

    # === [11] A22_a0.5 vs A14_a1.0 vs D_off paired (per metric) ===
    log("[11] Per-metric paired: A22_a0.5 vs A14_a1.0 vs D_off ...")
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

    # === [12] Verdict ===
    log("[12] Verdict ...")
    a22_loglik = per_cond_eval["A22_a0.5"]["loglik_per_user"]
    a22_maha = per_cond_eval["A22_a0.5"]["maha_pooled"]
    a14_loglik = per_cond_eval["A14_a1.0"]["loglik_per_user"]
    a14_maha = per_cond_eval["A14_a1.0"]["maha_pooled"]
    doff_loglik = per_cond_eval["D_off"]["loglik_per_user"]
    doff_maha = per_cond_eval["D_off"]["maha_pooled"]

    loglik_vs_maha = diffs["loglik_vs_maha_pooled"]
    loglik_vs_cosine = diffs["loglik_vs_cosine"]

    # Per-user log-likelihood must beat pooled Maha for any cond
    any_cond_better = any(loglik_vs_maha[c]["ci_excludes_0"] for c in CONDITIONS)
    if any_cond_better:
        verdict = "PARTIAL-GO"
        reason = (
            "per-user Gaussian log-likelihood beats pooled Mahalanobis on at least one cond "
            "(see ci_excludes_0 per cond)"
        )
    else:
        verdict = "NO-GO"
        reason = (
            "per-user Gaussian log-likelihood does NOT significantly beat pooled Mahalanobis on any cond "
            "(all CIs include 0). User-specific σ² does not help retrieval on this 30-pair subset."
        )
    log(f"  VERDICT: {verdict}")
    log(f"  REASON: {reason}")

    # === [13] Save ===
    out = {
        "phase": "14.D",
        "encoder": "AnnaWegmann/Style-Embedding (768d)",
        "n_candidates": len(candidates),
        "n_pairs": len(pair_keys),
        "n_conditions": len(CONDITIONS),
        "conditions": CONDITIONS,
        "k_per_cond": 8,
        "metrics": metrics,
        "cov_pooled_method": "Ledoit-Wolf shrinkage",
        "cov_shrinkage": shrinkage,
        "eigval_min": float(eigvals.min()),
        "eigval_max": float(eigvals.max()),
        "sigma_floor": sigma_floor,
        "per_cond_eval": per_cond_eval,
        "loglik_vs_other_metrics_diffs": diffs,
        "cond_diffs_per_metric": cond_diffs,
        "comparison": {
            "Phase14_A_sampled_A14_a1.0_cosine_top100_30pairs": "56.67%",
            "Phase14_A_sampled_A14_a1.0_maha_top100_30pairs": "53.3%",
            "Phase14_C_A14_a1.0_cosine_top100_30pairs_same_subset": "43.3%",
            "Phase14_C_A14_a1.0_maha_top100_30pairs_same_subset": "53.3%",
        },
        "note": (
            "Per-user Gaussian log-likelihood: score_u = -0.5 * Σ [(x-μ)²/σ² + log σ²]. "
            "Lower score = higher likelihood = better rank. σ² floor = "
            f"{sigma_floor:.6f} to avoid div-by-zero in degenerate dims."
        ),
        "decision_logic": {
            "PARTIAL-GO": "per-user loglik beats pooled Maha on at least one cond (CI excludes 0)",
            "NO-GO": "no significant improvement over pooled Maha on any cond",
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
        "phase": "14.D",
        "encoder": "AnnaWegmann/Style-Embedding (768d)",
        "n_pairs": len(pair_keys),
        "n_conditions": len(CONDITIONS),
        "conditions": CONDITIONS,
        "metrics": metrics,
        "cov_shrinkage": shrinkage,
        "sigma_floor": sigma_floor,
        "verdict": verdict,
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log(f"  meta → {OUT_META}")

    log("=" * 70)
    log("PHASE 14.D COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()