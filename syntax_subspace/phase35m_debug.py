#!/usr/bin/env python3
"""Debug: exact mirror of phase35i_v2_score.py N=7 logic to find why 35.8 vs 50.7."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
PHASE35E_DIR = REPO_ROOT / "result/phase35e"
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/syntax_subspace")
N_FIXED = 7


def softmax(x, axis=-1):
    x_max = x.max(axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / e.sum(axis=axis, keepdims=True)


def main():
    # === EXACT mirror of phase35i_v2_score.py ===
    cache = __import__("torch").load(SCRATCH / "phase35e_user_qwen_residuals.pt", weights_only=False)
    all_resids = cache["residuals"]
    flat_meta = cache["meta"]
    user_resids = {}
    for i, m in enumerate(flat_meta):
        if m[0] == "user":
            user_resids.setdefault(m[1], []).append(all_resids[i])
    print(f"[load] {len(user_resids)} users")

    pca_data = np.load(PHASE35E_DIR / "pca_components.npz", allow_pickle=True)
    def _maybe_int(k):
        try: return int(k)
        except (ValueError, TypeError): return k
    pca_components = {_maybe_int(k): v for k, v in pca_data["components"].item().items()}
    pca_means = {_maybe_int(k): v for k, v in pca_data["means"].item().items()}
    PCA_DIMS = sorted(pca_components.keys())

    user_z_per_d = {}
    for d in PCA_DIMS:
        user_z_per_d[d] = {}
        for uid, vecs in user_resids.items():
            V = np.stack(vecs, axis=0).astype(np.float32)
            Z = (V - pca_means[d]) @ pca_components[d].T
            user_z_per_d[d][uid] = Z

    user_pca_mu = {}
    pca_global_inv_sigma = {}
    for d in PCA_DIMS:
        var_global = np.stack([user_z_per_d[d][u].var(axis=0) for u in user_z_per_d[d]]).mean(axis=0)
        pca_global_inv_sigma[d] = 1.0 / np.maximum(var_global, 1e-6)
        for uid in user_z_per_d[d]:
            user_pca_mu[(uid, d)] = user_z_per_d[d][uid].mean(axis=0)

    def maha_global(cand_z, uid, d):
        mu = user_pca_mu[(uid, d)]
        inv_sigma = pca_global_inv_sigma[d]
        diffs = cand_z - mu
        return float(np.sqrt(np.maximum((diffs * diffs * inv_sigma).sum(), 1e-8)))

    # Load V2 N=7 cands + reuse residuals cache
    cands = json.load(open(SCRATCH / f"phase35i_v2_cands_N{N_FIXED}.json"))
    print(f"[load] {len(cands)} cands")

    cache2 = __import__("torch").load(SCRATCH / "phase35i_v2_cand_residuals.pt", weights_only=False)
    all_cand_residuals = cache2["residuals"]
    text_index = cache2["text_index"]
    cand_residuals_per_n = {}
    for k, (N, i) in enumerate(text_index):
        cand_residuals_per_n.setdefault(N, {})[i] = all_cand_residuals[k]
    cand_z_per_n = {}
    for N in [N_FIXED]:
        for i, c in enumerate(cands):
            r = cand_residuals_per_n[N][i]
            for d in PCA_DIMS:
                cand_z_per_n.setdefault(N, {}).setdefault(d, {})[i] = (r - pca_means[d]) @ pca_components[d].T
    print(f"[proj] cand_z_per_n[{N_FIXED}][32] count: {len(cand_z_per_n[N_FIXED][32])}")

    # === Per-(N, asin) intra-product softmax_g32 ===
    intra_results = []
    asin_to_recs = {}
    for i, c in enumerate(cands):
        asin_to_recs.setdefault(c["asin"], []).append(i)
    asin_pool = {}
    for asin, idxs in asin_to_recs.items():
        pool = []
        for i in idxs:
            c = cands[i]
            pool.append({"uid": c["user_id"], "z32": cand_z_per_n[N_FIXED][32][i]})
        asin_pool[asin] = pool

    for asin, pool in asin_pool.items():
        size = len(pool)
        if size < 2: continue
        uids_in_pool = list({p["uid"] for p in pool})
        if len(uids_in_pool) < 2: continue
        n_users = len(uids_in_pool)
        uid_to_idx = {u: i for i, u in enumerate(uids_in_pool)}
        D_g32 = np.zeros((size, n_users))
        for j, v in enumerate(uids_in_pool):
            D_g32[:, j] = [maha_global(p["z32"], v, 32) for p in pool]
        for uid in uids_in_pool:
            target_idx = uid_to_idx[uid]
            tau = 0.5
            logits = -D_g32 / tau
            probs = softmax(logits, axis=1)
            scores = probs[:, target_idx]
            order = np.argsort(-scores)
            rank1_uid = pool[int(order[0])]["uid"]
            hit = rank1_uid == uid
            intra_results.append({"asin": asin, "uid": uid, "n_pool": size, "is_hit": hit})

    n_hits = sum(1 for it in intra_results if it["is_hit"])
    n_total = len(intra_results)
    print(f"\n=== N=7 debug ===")
    print(f"  n_records={len(cands)}, n_users (size>=2)={n_total}, rank-1={n_hits}/{n_total} ({n_hits/max(n_total,1)*100:.1f}%)")
    # By asin
    from collections import defaultdict
    by_asin = defaultdict(lambda: {"hit": 0, "n": 0})
    for it in intra_results:
        by_asin[it["asin"]]["hit"] += 1 if it["is_hit"] else 0
        by_asin[it["asin"]]["n"] += 1
    print(f"  By asin:")
    for a in sorted(by_asin.keys()):
        st = by_asin[a]
        if st["n"] > 0:
            print(f"    {a}: {st['hit']}/{st['n']} = {st['hit']/st['n']*100:.1f}%")


if __name__ == "__main__":
    main()
