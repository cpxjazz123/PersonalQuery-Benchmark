#!/usr/bin/env python3
"""Phase 35.G: Softmax cross-entropy over all users in pool (listwise ranking).

Listwise ranking: 取代 margin/hard-negative (只对比 closest other),
                  让 score 反映"正确用户 vs 所有其他用户" 的 softmax 概率

For each (asin_pool, cand, target_user):
  logits_{v ∈ pool_users} = -D(cand, v) / τ
  prob_target = softmax(logits)[target_user]
  score = prob_target

Test:
  1. softmax-CE with various τ (0.1, 0.5, 1.0, 5.0, 10.0)
  2. softmax-CE on multiple D variants (pca64_diag, pca32_global, hybrid)
  3. Compare to margin (Phase 35.F SOTA 46.6%)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
PHASE35E_DIR = REPO_ROOT / "result/phase35e"
OUT_DIR = REPO_ROOT / "result/phase35g"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/syntax_subspace")


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = x.max(axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / e.sum(axis=axis, keepdims=True)


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

    # Build asin pool
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
                    "z64": Z_all[64][syntax_row[("cand", rec_i, k)]],
                    "z32": Z_all[32][syntax_row[("cand", rec_i, k)]],
                })
        asin_pool[asin] = pool
    print(f"[pool] sizes top: {sorted([(a, len(c)) for a, c in asin_pool.items()], key=lambda x: -x[1])[:5]}")

    def maha(cand_z, uid, d, mode="diag"):
        mu = user_pca_mu[(uid, d)]
        if mode == "diag":
            inv_var = user_pca_inv_var_diag[(uid, d)]
            diffs = cand_z - mu
            return float(np.sqrt(np.maximum((diffs * diffs * inv_var).sum(), 1e-8)))
        else:  # global
            inv_sigma = pca_global_inv_sigma[d]
            diffs = cand_z - mu
            return float(np.sqrt(np.maximum((diffs * diffs * inv_sigma).sum(), 1e-8)))

    intra_results: dict[str, list] = {f"size>={s}": [] for s in [2, 3, 4, 5, 10]}

    for asin, pool in asin_pool.items():
        size = len(pool)
        if size < 2: continue
        uids_in_pool = list({c["uid"] for c in pool})
        if len(uids_in_pool) < 2:
            continue  # Need at least 2 users for ranking

        # Precompute D matrix: (n_cands, n_users) for each variant
        # D_diag_64[c, v] = maha(c["z64"], v, 64, "diag")
        n_cands = len(pool)
        n_users = len(uids_in_pool)
        uid_to_idx = {u: i for i, u in enumerate(uids_in_pool)}

        D_diag64 = np.zeros((n_cands, n_users))
        D_g32 = np.zeros((n_cands, n_users))
        D_g64 = np.zeros((n_cands, n_users))
        for j, v in enumerate(uids_in_pool):
            D_diag64[:, j] = [maha(c["z64"], v, 64, "diag") for c in pool]
            D_g32[:, j] = [maha(c["z32"], v, 32, "global") for c in pool]
            D_g64[:, j] = [maha(c["z64"], v, 64, "global") for c in pool]

        for uid in uids_in_pool:
            target_idx = uid_to_idx[uid]

            # === Standard scorers (Phase 35.E baselines) ===
            s_diag64_alone = -D_diag64[:, target_idx]
            s_g32 = -D_g32[:, target_idx]
            s_g64 = -D_g64[:, target_idx]

            # === Phase 35.F margin / hybrid_HN ===
            other_mask = np.array([v != uid for v in uids_in_pool])
            D_other_diag = D_diag64[:, other_mask]
            D_other_g32 = D_g32[:, other_mask]
            margin_diag = D_diag64[:, target_idx] - D_other_diag.min(axis=1)
            HN_diag_a1 = -D_diag64[:, target_idx] - np.log(np.sum(np.exp(-D_other_diag), axis=1) + 1e-12)

            # === Phase 35.G softmax over all pool users ===
            # logits = -D / τ; prob_target = softmax(logits, axis=users)[target_idx]
            scorer_specs = []

            for tau in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]:
                # softmax-CE on D_diag64
                logits = -D_diag64 / tau
                probs = softmax(logits, axis=1)
                scorer_specs.append((f"softmax_diag64_τ{tau}", probs[:, target_idx]))

                # softmax-CE on D_g32
                logits = -D_g32 / tau
                probs = softmax(logits, axis=1)
                scorer_specs.append((f"softmax_g32_τ{tau}", probs[:, target_idx]))

            # Softmax + hybrid (combine D_diag64 + D_g32 in distance)
            for tau in [0.3, 1.0, 5.0]:
                # Hybrid distance = 0.7 * D_diag64 + 0.3 * D_g32 (Phase 35.E best weights)
                D_hybrid = 0.7 * D_diag64 + 0.3 * D_g32
                logits = -D_hybrid / tau
                probs = softmax(logits, axis=1)
                scorer_specs.append((f"softmax_hybrid_diag_g32_07_τ{tau}", probs[:, target_idx]))

                # 50/50
                D_hybrid50 = 0.5 * D_diag64 + 0.5 * D_g32
                logits = -D_hybrid50 / tau
                probs = softmax(logits, axis=1)
                scorer_specs.append((f"softmax_hybrid_50_τ{tau}", probs[:, target_idx]))

            # Margin-rank: lower score = closer (we want target to be closest)
            scorer_specs.append(("margin_diag64", -margin_diag))

            # Phase 35.F baselines
            scorer_specs.append(("HN_diag_a1", HN_diag_a1))

            # Standalone
            scorer_specs.append(("pca64_diag_alone", s_diag64_alone))

            for scorer_name, scores in scorer_specs:
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
                "median_own_rank": float(np.median(st["ranks"])) if st["ranks"] else None,
                "p90_own_rank": float(np.percentile(st["ranks"], 90)) if st["ranks"] else None,
            }
        intra_summary[size_key] = per_scorer

    print(f"\n=== Phase 35.G — Softmax Listwise Ranking (K=16) ===")
    print(f"{'scorer':<40} {'rank1':>12} {'mean':>8} {'median':>8} {'p90':>8}")
    size10 = intra_summary.get("size>=10", {})
    sorted_scorers = sorted(size10.items(), key=lambda x: (-x[1]["rank1_pct"], x[1]["mean_own_rank"] or 999))
    for sn, s in sorted_scorers:
        mr = s["mean_own_rank"] if s["mean_own_rank"] is not None else 0
        med = s["median_own_rank"] if s["median_own_rank"] is not None else 0
        p90 = s["p90_own_rank"] if s["p90_own_rank"] is not None else 0
        print(f"  {sn:<40} {s['n_hit']:>3}/{s['n_users']:<3} ({s['rank1_pct']:>5.1f}%) {mr:>7.2f} {med:>7.2f} {p90:>7.2f}")

    json.dump({
        "intra_summary": intra_summary,
        "n_intra": {k: len(v) for k, v in intra_results.items()},
    }, open(OUT_DIR / "softmax_intra.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n[save] softmax intra → {OUT_DIR / 'softmax_intra.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
