#!/usr/bin/env python3
"""Phase 35.J-3: Length-Controlled but candidate-pool unchanged.

User correction: Length-Matched (|ΔL|≤5) shrinks candidate pool,
making the task trivially easier. The real test is:
  - Pick queries with L in [lo, hi] (length controlled)
  - Let each query compete against ALL users in same asin (difficulty unchanged)
  - Compare Rank-1 to baseline

Plus report:
  - |C_matched| (length-matched candidate pool size)
  - Random baseline = 1 / |C|
  - True Rank-1 vs Random baseline ratio
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
PHASE35E_DIR = REPO_ROOT / "result/phase35e"
OUT_DIR = REPO_ROOT / "result/phase35j3"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/syntax_subspace")
N_ATTRS_LIST = [4, 5, 6, 7, 8, 9, 10]


def softmax(x, axis=-1):
    x_max = x.max(axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / e.sum(axis=axis, keepdims=True)


def count_attrs_covered(text, attrs):
    if not attrs: return 0
    return sum(1 for v in attrs.values() if v and str(v).strip().lower() in text.lower())


def main():
    cache = __import__("torch").load(SCRATCH / "phase35e_user_qwen_residuals.pt", weights_only=False)
    all_resids = cache["residuals"]
    flat_meta = cache["meta"]
    user_resids = {}
    for i, m in enumerate(flat_meta):
        if m[0] == "user":
            user_resids.setdefault(m[1], []).append(all_resids[i])

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

    # Load V2 cands + reuse residuals
    cand_residuals_per_n = {}
    cand_texts_per_n = {}
    for N in N_ATTRS_LIST:
        path = SCRATCH / f"phase35i_v2_cands_N{N}.json"
        if not path.exists(): continue
        cand_texts_per_n[N] = json.load(open(path))

    cache2 = __import__("torch").load(SCRATCH / "phase35i_v2_cand_residuals.pt", weights_only=False)
    all_cand_residuals = cache2["residuals"]
    text_index = cache2["text_index"]
    for k, (N, i) in enumerate(text_index):
        cand_residuals_per_n.setdefault(N, {})[i] = all_cand_residuals[k]

    cand_z_per_n = {N: {} for N in cand_texts_per_n}
    for N in cand_texts_per_n:
        for i, c in enumerate(cand_texts_per_n[N]):
            r = cand_residuals_per_n[N][i]
            for d in PCA_DIMS:
                cand_z_per_n[N].setdefault(d, {})[i] = (r - pca_means[d]) @ pca_components[d].T

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

    length_buckets = [(1, 15), (16, 25), (26, 35), (36, 50), (51, 80), (81, 1000)]
    bucket_labels = ["L≤15", "16-25", "26-35", "36-50", "51-80", "L>80"]

    # === Per-N, per-bucket: candidate-pool size + random baseline + actual Rank-1 ===
    print(f"\n{'='*100}")
    print("Phase 35.J-3: Length-Controlled Eval (cand-pool unchanged)")
    print(f"{'='*100}")
    print(f"{'N':<3} {'bucket':<10} {'n_test':<7} {'|C_pool|':<10} {'random_acc':<11} {'Rank-1':<8} {'lift':<10} {'mean_rank':<10} {'cov_avg':<7}")
    print('-' * 100)

    per_n_summary = {}
    for N in N_ATTRS_LIST:
        if N not in cand_texts_per_n: continue
        cands = cand_texts_per_n[N]
        asin_to_recs = {}
        for i, c in enumerate(cands):
            asin_to_recs.setdefault(c["asin"], []).append(i)
        asin_pool = {}
        for asin, idxs in asin_to_recs.items():
            pool = []
            for i in idxs:
                c = cands[i]
                cov = count_attrs_covered(c["query"], c["attrs_used"])
                L = len(c["query"].split())
                pool.append({
                    "uid": c["user_id"],
                    "cov": cov, "n_attr": N, "L": L,
                    "z32": cand_z_per_n[N][32][i],
                })
            asin_pool[asin] = pool

        # Per-bucket: query-length-controlled, full-pool competition
        bucket_results = {label: {"hit": 0, "n": 0, "ranks": [], "pool_sizes": [], "random_acc_sum": 0.0, "covs": []} for label in bucket_labels}

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

            # Test queries: cands with L in bucket (their own cands; restrict to own cands)
            for uid in uids_in_pool:
                target_idx = uid_to_idx[uid]
                logits = -D_g32 / 0.5
                probs = softmax(logits, axis=1)
                scores = probs[:, target_idx]
                order = np.argsort(-scores)

                own_indices = [i for i, p in enumerate(pool) if p["uid"] == uid]
                if not own_indices: continue

                # For each own cand, bucket it and check rank under FULL pool
                for own_i in own_indices:
                    own_L = pool[own_i]["L"]
                    for (lo, hi), label in zip(length_buckets, bucket_labels):
                        if lo <= own_L <= hi:
                            hit = pool[int(order[0])]["uid"] == uid
                            own_rank = int(np.where(order == own_i)[0][0])
                            bucket_results[label]["hit"] += 1 if hit else 0
                            bucket_results[label]["n"] += 1
                            bucket_results[label]["ranks"].append(own_rank)
                            bucket_results[label]["pool_sizes"].append(n_users)
                            bucket_results[label]["random_acc_sum"] += 1.0 / n_users
                            bucket_results[label]["covs"].append(pool[own_i]["cov"])
                            break

        per_n_summary[N] = {}
        for label in bucket_labels:
            st = bucket_results[label]
            if st["n"] == 0:
                per_n_summary[N][label] = {"n_test": 0}
                continue
            rank1 = st["hit"] / st["n"] * 100
            mean_rank = float(np.mean(st["ranks"]))
            avg_pool = float(np.mean(st["pool_sizes"]))
            avg_random = st["random_acc_sum"] / st["n"] * 100
            lift = rank1 / avg_random if avg_random > 0 else 0
            avg_cov = float(np.mean(st["covs"])) if st["covs"] else 0
            per_n_summary[N][label] = {
                "n_test": st["n"],
                "pool_size_avg": avg_pool,
                "random_acc": avg_random,
                "rank1_pct": rank1,
                "lift_vs_random": lift,
                "mean_rank": mean_rank,
                "avg_cov": avg_cov,
            }
            print(f"{N:<3} {label:<10} {st['n']:<7} {avg_pool:<10.2f} {avg_random:<11.2f} {rank1:<8.1f} {lift:<10.2f} {mean_rank:<10.2f} {avg_cov:<7.2f}")
        print()

    # Highlight N=7 L=36-50 specifically
    print(f"\n{'='*80}")
    print("KEY: N=7, L=36-50 (your suggested operating point)")
    print(f"{'='*80}")
    if 7 in per_n_summary and "36-50" in per_n_summary[7]:
        s = per_n_summary[7]["36-50"]
        if s["n_test"] > 0:
            print(f"  N=7, L=36-50:")
            print(f"    n_test (queries): {s['n_test']}")
            print(f"    pool_size_avg: {s['pool_size_avg']:.2f}")
            print(f"    random_acc: {s['random_acc']:.2f}%")
            print(f"    Rank-1: {s['rank1_pct']:.1f}%")
            print(f"    lift vs random: {s['lift_vs_random']:.2f}x")
            print(f"    mean_rank: {s['mean_rank']:.2f}")
            print(f"    avg_cov: {s['avg_cov']:.2f}")
        else:
            print("  No test queries in N=7, L=36-50 bucket")

    json.dump(per_n_summary, open(OUT_DIR / "length_controlled_summary.json", "w"),
              indent=2, ensure_ascii=False)
    print(f"\n[save] → {OUT_DIR}/length_controlled_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
