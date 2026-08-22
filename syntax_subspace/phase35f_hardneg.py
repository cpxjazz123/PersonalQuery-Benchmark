#!/usr/bin/env python3
"""Phase 35.F: Hard-negative contrastive rerank.

对每个 asin pool 内的每个 candidate,计算:
  score(q, u) = -D(q, u) + log_sum_exp_{v≠u}(-D(q, v))
即: target user 越近越好 + 其他用户越远越好 (log-sum-exp 近似 max-pool)

Test on K=16 cands:
  HN_pca64_diag: hard-negative with pca64_diag Maha
  HN_pca32_global: hard-negative with pca32_global Maha
  HN_hybrid: HN(pca64_diag) + HN(pca32_global) hybrid
  margin_score: rank by margin = D_to_target - min_{v≠target} D_to_v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
PHASE35E_DIR = REPO_ROOT / "result/phase35e"
OUT_DIR = REPO_ROOT / "result/phase35f"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/syntax_subspace")


def main():
    # Load K=16 data
    cands_data = json.load(open(SCRATCH / "phase35e_candidates_k16.json"))
    user_sents = json.load(open(SCRATCH / "user_sents_phase35b.json"))
    cache = __import__("torch").load(SCRATCH / "phase35e_user_qwen_residuals.pt", weights_only=False)
    all_resids = cache["residuals"]
    flat_meta = cache["meta"]
    print(f"[load] {len(cands_data)} records × K=16, {len(user_sents)} users, {len(flat_meta)} texts")
    syntax_row = {m: i for i, m in enumerate(flat_meta)}

    # Load PCA components (K=16's PCA)
    pca_data = np.load(PHASE35E_DIR / "pca_components.npz", allow_pickle=True)
    def _maybe_int(k):
        try: return int(k)
        except (ValueError, TypeError): return k
    pca_components = {_maybe_int(k): v for k, v in pca_data["components"].item().items()}
    pca_means = {_maybe_int(k): v for k, v in pca_data["means"].item().items()}
    PCA_DIMS = sorted(pca_components.keys())

    # Project all to PCA
    Z_all = {d: (all_resids - pca_means[d]) @ pca_components[d].T for d in PCA_DIMS}

    # Build per-user PCA stats (K=16 user sents)
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

    # Per-cand PCA z (cached for all 1184 cands)
    cand_z: dict[tuple[int, int], dict[int, np.ndarray]] = {}
    for rec_i, rec in enumerate(cands_data):
        for k in range(len(rec["candidates"])):
            sr = syntax_row[("cand", rec_i, k)]
            cand_z[(rec_i, k)] = {d: Z_all[d][sr] for d in PCA_DIMS}

    # Build per-asin pool: for each (asin, user) pair, list of cands in pool
    asin_to_recs: dict[str, list[int]] = {}
    for i, r in enumerate(cands_data):
        asin_to_recs.setdefault(r["asin"], []).append(i)
    asin_pool: dict[str, list[dict]] = {}
    for asin, rec_idxs in asin_to_recs.items():
        pool = []
        for rec_i in rec_idxs:
            for k in range(len(cands_data[rec_i]["candidates"])):
                pool.append({
                    "uid": cands_data[rec_i]["user_id"],
                    "rec_i": rec_i, "k": k,
                    "z": cand_z[(rec_i, k)],
                })
        asin_pool[asin] = pool
    print(f"[pool] sizes top: {sorted([(a, len(c)) for a, c in asin_pool.items()], key=lambda x: -x[1])[:5]}")

    # === Compute hard-negative scoring ===
    def maha_score(cand_z_d: np.ndarray, target_uid: str, d: int, mode: str = "diag") -> float:
        if mode == "diag":
            mu_d = user_pca_mu[(target_uid, d)]
            inv_var = user_pca_inv_var_diag[(target_uid, d)]
            diffs = cand_z_d - mu_d
            return float(np.sqrt(np.maximum((diffs * diffs * inv_var).sum(), 1e-8)))
        if mode == "global":
            mu_d = user_pca_mu[(target_uid, d)]
            inv_sigma = pca_global_inv_sigma[d]
            diffs = cand_z_d - mu_d
            return float(np.sqrt(np.maximum((diffs * diffs * inv_sigma).sum(), 1e-8)))
        raise ValueError(mode)

    intra_results: dict[str, list] = {f"size>={s}": [] for s in [2, 3, 4, 5, 10]}
    for asin, pool in asin_pool.items():
        size = len(pool)
        if size < 2: continue
        uids_in_pool = list({c["uid"] for c in pool})

        for uid in uids_in_pool:
            # === Compute all D(cand, user) for all (cand, user) pairs ===
            # Pool size n_cands * n_users matrix
            n_cands = len(pool)
            n_users = len(uids_in_pool)
            other_uids = [v for v in uids_in_pool if v != uid]
            if len(other_uids) == 0:
                continue  # No negative candidates (single user in pool)

            # === D(cand, target_user) for all cands ===
            D_target_diag = np.array([maha_score(c["z"][64], uid, 64, "diag") for c in pool])
            D_target_g32 = np.array([maha_score(c["z"][32], uid, 32, "global") for c in pool])
            D_target_g64 = np.array([maha_score(c["z"][64], uid, 64, "global") for c in pool])

            # === D(cand, other_user) for all (cand, other_user) pairs ===
            D_other_diag = np.zeros((n_cands, len(other_uids)))
            D_other_g32 = np.zeros((n_cands, len(other_uids)))
            for j, v in enumerate(other_uids):
                D_other_diag[:, j] = [maha_score(c["z"][64], v, 64, "diag") for c in pool]
                D_other_g32[:, j] = [maha_score(c["z"][32], v, 32, "global") for c in pool]

            # === Hard-negative scoring ===
            # HN_diag: score = -D_target_diag + log_sum_exp(-D_other_diag per cand)
            # But to be a proper margin score, we want target lower than others:
            # Higher score = better; we use: score = -D_target_diag + alpha * log_sum_exp(-D_other_diag)
            # alpha=1 means: how much closer target is vs others (in log space)
            # alpha=0 means: just target distance (original Maha)
            alpha = 1.0
            HN_diag = -D_target_diag + alpha * (-np.log(np.sum(np.exp(-D_other_diag), axis=1) + 1e-12))
            # Note: log_sum_exp(-D_other) ≈ -min(D_other); negative score means lower is better
            # HN_diag HIGHER = target much closer than other users

            # Margin rank: rank by D_target - min D_other (want negative large)
            margin_diag = D_target_diag - D_other_diag.min(axis=1)  # negative = target closer than closest other

            # Hybrid HN
            HN_g32 = -D_target_g32 + alpha * (-np.log(np.sum(np.exp(-D_other_g32), axis=1) + 1e-12))
            hybrid_HN = 0.7 * (HN_diag - HN_diag.mean()) / (HN_diag.std() + 1e-8) + \
                        0.3 * (HN_g32 - HN_g32.mean()) / (HN_g32.std() + 1e-8)

            # Compare with regular Maha (Phase 35.E baselines)
            # pca64_diag alone = -D_target_diag
            # pca64_diag+pca64_global_07 = best Phase 35.E hybrid

            scorers = {
                "HN_pca64_diag_α1": HN_diag,
                "HN_pca64_diag_α05": -D_target_diag + 0.5 * (-np.log(np.sum(np.exp(-D_other_diag), axis=1) + 1e-12)),
                "margin_pca64_diag": -margin_diag,  # higher = better
                "hybrid_HN_diag_g32_07": hybrid_HN,
                "pca64_diag_alone": -D_target_diag,  # baseline for comparison
            }

            for scorer_name, scores in scorers.items():
                order = np.argsort(-scores)
                rank1_idx = int(order[0])
                rank1_uid = pool[rank1_idx]["uid"]
                hit = rank1_uid == uid
                own_ranks = [int(np.where(order == i)[0][0]) for i, c in enumerate(pool) if c["uid"] == uid]
                bucket = ("size>=10" if size >= 10 else
                          f"size>={'5' if size >= 5 else '4' if size >= 4 else '3' if size >= 3 else '2'}")
                intra_results[bucket].append({
                    "asin": asin, "uid": uid, "scorer": scorer_name,
                    "n_pool": size, "rank1_uid": rank1_uid, "is_hit": hit,
                    "mean_own_rank": float(np.mean(own_ranks)) if own_ranks else None,
                })

    # Aggregate
    intra_summary = {}
    for size_key, items in intra_results.items():
        per_scorer = {}
        for it in items:
            sn = it["scorer"]
            per_scorer.setdefault(sn, {"n_hit": 0, "n_users": 0, "ranks": []})
            per_scorer[sn]["n_users"] += 1
            if it["is_hit"]:
                per_scorer[sn]["n_hit"] += 1
            if it["mean_own_rank"] is not None:
                per_scorer[sn]["ranks"].append(it["mean_own_rank"])
        for sn, st in per_scorer.items():
            per_scorer[sn] = {
                "n_users": st["n_users"], "n_hit": st["n_hit"],
                "rank1_pct": st["n_hit"] / max(st["n_users"], 1) * 100,
                "mean_own_rank": float(np.mean(st["ranks"])) if st["ranks"] else None,
            }
        intra_summary[size_key] = per_scorer

    print(f"\n=== Phase 35.F — Hard-Negative Contrastive (K=16) ===")
    print(f"{'scorer':<35} {'rank1_size>=10':>20} {'mean_rank':>10}")
    size10 = intra_summary.get("size>=10", {})
    # Sort by rank1_pct
    sorted_scorers = sorted(size10.items(), key=lambda x: -x[1]["rank1_pct"])
    for sn, s in sorted_scorers:
        mr = s["mean_own_rank"] if s["mean_own_rank"] is not None else 0
        print(f"  {sn:<35} {s['n_hit']:>4}/{s['n_users']:<4} ({s['rank1_pct']:>5.1f}%) {mr:>9.2f}")

    json.dump({
        "intra_summary": intra_summary,
        "n_intra": {k: len(v) for k, v in intra_results.items()},
    }, open(OUT_DIR / "hn_intra.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n[save] HN intra → {OUT_DIR / 'hn_intra.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
