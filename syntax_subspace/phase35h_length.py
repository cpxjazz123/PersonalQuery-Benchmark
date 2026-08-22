#!/usr/bin/env python3
"""Phase 35.H: Length-stratified + length-matched evaluation.

目的:
  1. 长度和 coverage 的关系 (length vs full_cov rate)
  2. 长度和 residual score 的关系 (Spearman ρ)
  3. 长度和 Rank-1 的关系 (length-bucket rank-1)
  4. 长度匹配后 hybrid reranker 是否仍优于 baseline (length-matched subset)

分桶: L≤15, 16-25, 26-35, >35 words
Length-matched: |L(q_i) - L(q_j)| ≤ 3 words
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
PHASE35E_DIR = REPO_ROOT / "result/phase35e"
PHASE35G_DIR = REPO_ROOT / "result/phase35g"
OUT_DIR = REPO_ROOT / "result/phase35h"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/syntax_subspace")


def main():
    cands_data = json.load(open(SCRATCH / "phase35e_candidates_k16.json"))
    user_sents = json.load(open(SCRATCH / "user_sents_phase35b.json"))
    cache = __import__("torch").load(SCRATCH / "phase35e_user_qwen_residuals.pt", weights_only=False)
    all_resids = cache["residuals"]
    flat_meta = cache["meta"]
    syntax_row = {m: i for i, m in enumerate(flat_meta)}

    pca_data = np.load(PHASE35E_DIR / "pca_components.npz", allow_pickle=True)
    def _maybe_int(k):
        try: return int(k)
        except (ValueError, TypeError): return k
    pca_components = {_maybe_int(k): v for k, v in pca_data["components"].item().items()}
    pca_means = {_maybe_int(k): v for k, v in pca_data["means"].item().items()}
    PCA_DIMS = sorted(pca_components.keys())

    Z_all = {d: (all_resids - pca_means[d]) @ pca_components[d].T for d in PCA_DIMS}

    user_pca_mu = {}
    user_pca_inv_var_diag = {}
    pca_global_inv_sigma = {}
    for d in PCA_DIMS:
        pca_global_inv_sigma[d] = 1.0 / np.maximum(Z_all[d].var(axis=0), 1e-6)

    user_resids_dict = {}
    for i, m in enumerate(flat_meta):
        if m[0] == "user":
            user_resids_dict.setdefault(m[1], []).append(all_resids[i])
    for uid, vecs in user_resids_dict.items():
        V = np.stack(vecs, axis=0).astype(np.float64)
        for d in PCA_DIMS:
            Z = np.stack([Z_all[d][i] for i, mm in enumerate(flat_meta)
                          if mm[0] == "user" and mm[1] == uid], axis=0)
            user_pca_mu[(uid, d)] = Z.mean(axis=0)
            var = Z.var(axis=0) + 1e-3
            user_pca_inv_var_diag[(uid, d)] = 1.0 / np.maximum(var, 1e-6)

    # Build asin pool with length annotation
    asin_to_recs: dict[str, list[int]] = {}
    for i, r in enumerate(cands_data):
        asin_to_recs.setdefault(r["asin"], []).append(i)
    asin_pool: dict[str, list[dict]] = {}
    for asin, rec_idxs in asin_to_recs.items():
        pool = []
        for rec_i in rec_idxs:
            rec = cands_data[rec_i]
            n_attr = len(rec["attrs_used"])
            for k, q in enumerate(rec["candidates"]):
                cov = sum(1 for v in rec["attrs_used"].values() if v and str(v).strip().lower() in q.lower())
                L = len(q.split())  # word count
                pool.append({
                    "uid": rec["user_id"],
                    "rec_i": rec_i, "k": k,
                    "L": L, "cov": cov, "n_attr": n_attr, "q": q,
                    "z32": Z_all[32][syntax_row[("cand", rec_i, k)]],
                    "z64": Z_all[64][syntax_row[("cand", rec_i, k)]],
                })
        asin_pool[asin] = pool
    pool_sizes = sorted([(a, len(c)) for a, c in asin_pool.items()], key=lambda x: -x[1])
    print(f"[pool] sizes top 5: {pool_sizes[:5]}")

    # === Compute per-pool D matrices and softmax_g32_τ0.5 scores ===
    def maha_diag(cand_z, uid, d):
        mu = user_pca_mu[(uid, d)]
        inv_var = user_pca_inv_var_diag[(uid, d)]
        diffs = cand_z - mu
        return float(np.sqrt(np.maximum((diffs * diffs * inv_var).sum(), 1e-8)))

    def softmax(x, axis=-1):
        x_max = x.max(axis=axis, keepdims=True)
        e = np.exp(x - x_max)
        return e / e.sum(axis=axis, keepdims=True)

    # Per-record intra-product Rank-1 under different conditions:
    #  1) full pool (baseline, K=16)
    #  2) length-stratified (per length bucket, only consider cands in that bucket)
    #  3) length-matched (per cand, restrict competing cands to within |ΔL|≤3 of own length)
    length_buckets = [(1, 15), (16, 25), (26, 35), (36, 1000)]
    bucket_labels = ["L≤15", "16-25", "26-35", "L>35"]

    print("[length] bucket distribution:")
    all_L = [c["L"] for pool in asin_pool.values() for c in pool]
    all_cov = [c["cov"] / c["n_attr"] for pool in asin_pool.values() for c in pool]
    for (lo, hi), label in zip(length_buckets, bucket_labels):
        mask = [L for L in all_L if lo <= L <= hi]
        cands_in_bucket = [(L, cov) for L, cov in zip(all_L, all_cov) if lo <= L <= hi]
        if cands_in_bucket:
            full_cov_rate = sum(1 for L, cov in cands_in_bucket if cov >= 1.0) / len(cands_in_bucket)
            avg_L = sum(L for L, _ in cands_in_bucket) / len(cands_in_bucket)
            print(f"  {label}: n={len(cands_in_bucket)}, avg_L={avg_L:.1f}, full_cov={full_cov_rate:.3f}")

    # Per-asin-pool evaluation
    intra_results_full = []
    intra_results_strat = {label: [] for label in bucket_labels}
    intra_results_lm3 = []  # length-matched (|ΔL|≤3)

    for asin, pool in asin_pool.items():
        size = len(pool)
        if size < 2: continue
        uids_in_pool = list({c["uid"] for c in pool})
        if len(uids_in_pool) < 2: continue

        n_users = len(uids_in_pool)
        uid_to_idx = {u: i for i, u in enumerate(uids_in_pool)}

        # D_g32 matrix
        D_g32 = np.zeros((size, n_users))
        for j, v in enumerate(uids_in_pool):
            D_g32[:, j] = [maha_diag(c["z32"], v, 32) for c in pool]

        # L vector
        L_arr = np.array([c["L"] for c in pool])

        for uid in uids_in_pool:
            target_idx = uid_to_idx[uid]

            # === 1) Full pool, Phase 35.G SOTA scorer ===
            tau = 0.5
            logits = -D_g32 / tau
            probs = softmax(logits, axis=1)
            scores_full = probs[:, target_idx]

            # Find own cands
            own_cand_indices = [i for i, c in enumerate(pool) if c["uid"] == uid]
            if not own_cand_indices:
                continue
            # Best own cand (by score)
            best_own_idx = max(own_cand_indices, key=lambda i: scores_full[i])
            # Is best_own ranked #1?
            order_full = np.argsort(-scores_full)
            rank1_full_uid = pool[int(order_full[0])]["uid"]
            full_hit = rank1_full_uid == uid
            own_ranks_full = [int(np.where(order_full == i)[0][0]) for i in own_cand_indices]

            intra_results_full.append({
                "asin": asin, "uid": uid, "scorer": "softmax_g32_τ0.5_full_pool",
                "n_pool": size, "is_hit": full_hit,
                "mean_own_rank": float(np.mean(own_ranks_full)) if own_ranks_full else None,
                "best_own_L": pool[best_own_idx]["L"],
                "best_own_cov": pool[best_own_idx]["cov"],
            })

            # === 2) Length-stratified (per length bucket) ===
            for (lo, hi), label in zip(length_buckets, bucket_labels):
                bucket_mask = np.array([(lo <= L <= hi) for L in L_arr])
                bucket_indices = np.where(bucket_mask)[0]
                if len(bucket_indices) < 2:
                    continue
                # Pool within bucket
                D_bucket = D_g32[bucket_indices]  # (n_bucket, n_users)
                # Need at least one own cand in bucket
                own_in_bucket = [i for i in own_cand_indices if i in bucket_indices]
                if not own_in_bucket:
                    continue
                # Compute softmax within bucket
                logits_b = -D_bucket / tau
                probs_b = softmax(logits_b, axis=1)
                scores_b = probs_b[:, target_idx]
                order_b = np.argsort(-scores_b)
                rank1_uid_b = pool[bucket_indices[int(order_b[0])]]["uid"]
                hit_b = rank1_uid_b == uid
                own_in_bucket_pos = [int(np.where(order_b == np.where(bucket_indices == i)[0][0])[0][0])
                                      for i in own_in_bucket]
                intra_results_strat[label].append({
                    "asin": asin, "uid": uid, "scorer": f"softmax_g32_τ0.5_strat_{label}",
                    "n_pool": len(bucket_indices), "is_hit": hit_b,
                    "mean_own_rank": float(np.mean(own_in_bucket_pos)) if own_in_bucket_pos else None,
                })

            # === 3) Length-matched (|ΔL|≤3 from each own cand) ===
            for own_i in own_cand_indices:
                own_L = L_arr[own_i]
                match_mask = np.abs(L_arr - own_L) <= 3
                match_indices = np.where(match_mask)[0]
                if len(match_indices) < 2:
                    continue
                D_m = D_g32[match_indices]
                logits_m = -D_m / tau
                probs_m = softmax(logits_m, axis=1)
                scores_m = probs_m[:, target_idx]
                order_m = np.argsort(-scores_m)
                rank1_idx = int(order_m[0])
                rank1_uid_m = pool[match_indices[rank1_idx]]["uid"]
                hit_m = rank1_uid_m == uid
                # own cand rank in matched subset
                own_pos = int(np.where(order_m == np.where(match_indices == own_i)[0][0])[0][0])
                intra_results_lm3.append({
                    "asin": asin, "uid": uid, "scorer": "softmax_g32_τ0.5_lm3",
                    "n_pool": len(match_indices),
                    "own_cand_L": int(own_L),
                    "is_hit": hit_m,
                    "own_cand_rank": own_pos,
                })

    # Aggregate
    def aggregate(items):
        if not items:
            return {"n_users": 0, "n_hit": 0, "rank1_pct": 0, "mean_own_rank": None}
        n_hit = sum(1 for it in items if it["is_hit"])
        n = len(items)
        # accept either mean_own_rank (strat) or own_cand_rank (lm3)
        ranks = [it.get("mean_own_rank", it.get("own_cand_rank")) for it in items
                 if it.get("mean_own_rank") is not None or it.get("own_cand_rank") is not None]
        return {
            "n_users": n, "n_hit": n_hit,
            "rank1_pct": n_hit / max(n, 1) * 100,
            "mean_own_rank": float(np.mean(ranks)) if ranks else None,
            "median_own_rank": float(np.median(ranks)) if ranks else None,
            "p90_own_rank": float(np.percentile(ranks, 90)) if ranks else None,
        }

    summary = {
        "full_pool": aggregate(intra_results_full),
        "length_stratified": {label: aggregate(items) for label, items in intra_results_strat.items()},
        "length_matched_lm3": aggregate(intra_results_lm3),
    }

    print(f"\n=== Phase 35.H — Length Stratified + Length Matched ===")
    print(f"\n--- Full pool (Phase 35.G SOTA reference) ---")
    s = summary["full_pool"]
    print(f"  full_pool: n={s['n_users']}, rank1={s['n_hit']} ({s['rank1_pct']:.1f}%), mean={s.get('mean_own_rank', 0):.2f}")

    print(f"\n--- Length Stratified (rank-1 per bucket) ---")
    for label in bucket_labels:
        s = summary["length_stratified"][label]
        print(f"  {label}: n={s['n_users']}, rank1={s['n_hit']} ({s['rank1_pct']:.1f}%), mean={s.get('mean_own_rank', 0):.2f}")

    print(f"\n--- Length Matched (|ΔL|≤3) ---")
    s = summary["length_matched_lm3"]
    print(f"  lm3: n={s['n_users']}, rank1={s['n_hit']} ({s['rank1_pct']:.1f}%), mean={s.get('mean_own_rank', 0):.2f}")

    # === Spearman correlations ===
    # Length vs cov
    cov_full = [c["cov"] / c["n_attr"] for pool in asin_pool.values() for c in pool]
    rho_cov, p_cov = spearmanr(all_L, cov_full)
    print(f"\n--- Spearman Correlations ---")
    print(f"  ρ(L, cov_rate):       {rho_cov:+.3f} (p={p_cov:.2e})")

    # Length vs residual score (softmax_g32 probability of being owner)
    # We compute it from full results
    score_per_cand = []
    L_per_cand = []
    cov_per_cand = []
    n_attr_per_cand = []
    for asin, pool in asin_pool.items():
        size = len(pool)
        if size < 2: continue
        uids_in_pool = list({c["uid"] for c in pool})
        if len(uids_in_pool) < 2: continue
        n_users = len(uids_in_pool)
        uid_to_idx = {u: i for i, u in enumerate(uids_in_pool)}
        D_g32 = np.zeros((size, n_users))
        for j, v in enumerate(uids_in_pool):
            D_g32[:, j] = [maha_diag(c["z32"], v, 32) for c in pool]
        tau = 0.5
        logits = -D_g32 / tau
        probs = softmax(logits, axis=1)
        for i, c in enumerate(pool):
            uid = c["uid"]
            target_idx = uid_to_idx[uid]
            score_per_cand.append(probs[i, target_idx])
            L_per_cand.append(c["L"])
            cov_per_cand.append(c["cov"] / c["n_attr"])
            n_attr_per_cand.append(c["n_attr"])

    score_per_cand = np.array(score_per_cand)
    L_per_cand = np.array(L_per_cand)
    cov_per_cand = np.array(cov_per_cand)
    n_attr_per_cand = np.array(n_attr_per_cand)

    # ρ between L and residual score
    rho_score, p_score = spearmanr(L_per_cand, score_per_cand)
    print(f"  ρ(L, softmax_score):  {rho_score:+.3f} (p={p_score:.2e})")
    rho_score_cov, p_score_cov = spearmanr(cov_per_cand, score_per_cand)
    print(f"  ρ(cov, softmax_score):{rho_score_cov:+.3f} (p={p_score_cov:.2e})")
    # partial: score vs L controlling for cov
    rho_L_given_cov, _ = spearmanr(L_per_cand - 2 * cov_per_cand, score_per_cand)
    print(f"  ρ(L-2cov, score):     {rho_L_given_cov:+.3f} (rough partial)")

    print(f"\n  Mean residual score by length bucket:")
    for (lo, hi), label in zip(length_buckets, bucket_labels):
        m = (L_per_cand >= lo) & (L_per_cand <= hi)
        if m.sum() > 0:
            print(f"    {label}: mean_score={score_per_cand[m].mean():.4f}, n={m.sum()}")

    summary["spearman"] = {
        "rho_L_cov": float(rho_cov),
        "rho_L_score": float(rho_score),
        "rho_cov_score": float(rho_score_cov),
        "rho_partial_L_minus_cov": float(rho_L_given_cov),
    }

    json.dump(summary, open(OUT_DIR / "length_summary.json", "w"), indent=2, ensure_ascii=False)
    json.dump({
        "intra_full": intra_results_full,
        "intra_strat": intra_results_strat,
        "intra_lm3": intra_results_lm3,
    }, open(OUT_DIR / "length_intra.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n[save] → {OUT_DIR}/length_summary.json + length_intra.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
