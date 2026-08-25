"""Stage 9E-P — Personalization Separability Diagnostic.

Determine root cause of Stage 9D's selection collapse (1-2 unique queries per ASIN).

Four checks (per-ASIN and aggregate):

CHECK 1: Pairwise distance between 10 user Gaussian mu vectors per ASIN
  - If users are far apart (high mean pairwise dist) → user style is separable
  - If users are close (low mean dist) → user style indistinguishable in PCA48

CHECK 2: Top-K candidate overlap rate across 10 users per ASIN
  - For K ∈ {1, 3, 5, 10}, compute mean Jaccard overlap of top-K candidates
  - High overlap (close to 1.0) → all users prefer same set of candidates
  - Low overlap → user preferences differentiate the candidate space

CHECK 3: Selected-query uniqueness + selection entropy per ASIN
  - N_unique = |{selected_query for all 10 users}| (range 1-10)
  - entropy = -Σ p_i log p_i over the selected-query distribution

CHECK 4: Pool-size sweep (sub-sample pool, redo selection, see if uniqueness grows)
  - Sizes: 10, 20, 30, 50, 75, 100, 150, 196 (capped at max per-ASIN)
  - For each size: for each ASIN with pool ≥ size, subsample + select + count unique

Decision logic:
- If uniqueness grows with pool size → pool coverage is bottleneck (need scale)
- If uniqueness stays at 1-2 even with pool 196 → user Gaussian / selection objective is bottleneck

Output:
- /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage9ep_check1_user_dist.json
- /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage9ep_check2_overlap.json
- /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage9ep_check3_uniqueness.json
- /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage9ep_check4_pool_sweep.json
- /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage9ep_summary.json

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
        gaussian/syntax_subspace_stage9ep_diagnostic.py \
        > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage9ep_run.log 2>&1 &
"""

from __future__ import annotations

import collections
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np


SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
ASINS_IN = SCRATCH / "stage8_5_asins.json"
POOL_9C_IN = SCRATCH / "stage9c_pool_pilot.json"
GAUSSIANS_IN = SCRATCH / "stage8_5_user_gaussians.json"

CHECK1_OUT = SCRATCH / "stage9ep_check1_user_dist.json"
CHECK2_OUT = SCRATCH / "stage9ep_check2_overlap.json"
CHECK3_OUT = SCRATCH / "stage9ep_check3_uniqueness.json"
CHECK4_OUT = SCRATCH / "stage9ep_check4_pool_sweep.json"
SUMMARY_OUT = SCRATCH / "stage9ep_summary.json"

SEED = 2024
POOL_SWEEP_SIZES = [10, 20, 30, 50, 75, 100, 150, 196]
TOP_K_VALUES = [1, 3, 5, 10]


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def mahalanobis_sq(z: np.ndarray, mu: np.ndarray, sigma_diag: np.ndarray) -> float:
    diff = z - mu
    return float((diff * diff / sigma_diag).sum())


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def selection_entropy(selected_qs: list) -> float:
    """Shannon entropy of selected-query distribution."""
    if not selected_qs:
        return 0.0
    counts = collections.Counter(selected_qs)
    n = len(selected_qs)
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * np.log2(p)
    return float(h)


def main():
    log("=== Stage 9E-P — Personalization Separability Diagnostic ===")

    # === Load ===
    log("\n=== 1. Loading inputs ===")
    sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
    sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/gaussian")
    from gaussian_vades import _syntax_subspace_prepare
    P = _syntax_subspace_prepare()
    scaler = P["scaler"]
    fnames = P["feature_names_ordered"]
    train_idx = P["train_idx"]
    from sklearn.decomposition import PCA
    pca = PCA(n_components=48, random_state=42)
    pca.fit(P["X_scaled"][train_idx])
    log(f"  PCA48 ready")

    pool_9C_data = json.load(open(POOL_9C_IN))
    pools_9C = pool_9C_data["pools"]
    gauss_data = json.load(open(GAUSSIANS_IN))
    users_gauss = gauss_data["users"]
    global_var = np.array(gauss_data["global_var"])

    asin_data_full = json.load(open(ASINS_IN))["asins"]
    asin_data = [a for a in asin_data_full if a["asin"] in pools_9C]
    log(f"  ASINs in 9C ∩ 8.5: {len(asin_data)}")

    # Project pool queries to PCA48
    import gzip as gz
    import hashlib
    feat_map = {}
    with gz.open(SCRATCH / "stage7b_query_features.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            feat_map[rec["k"]] = rec["v"]

    def feat_key(t):
        return hashlib.sha1(t.strip().lower().encode("utf-8")).hexdigest()

    pool_z = {}
    for asin, qs in pools_9C.items():
        zs_for_asin = []
        for q in qs:
            k = feat_key(q["query"])
            feats = feat_map.get(k)
            if not feats:
                continue
            vec = np.array([feats.get(n, 0.0) for n in fnames], dtype=np.float64)
            z = pca.transform(scaler.transform(vec[None, :]))[0]
            zs_for_asin.append((z, q))
        pool_z[asin] = zs_for_asin

    # === CHECK 1: User mu pairwise distance per ASIN ===
    log("\n=== CHECK 1: User mu pairwise distance per ASIN ===")
    check1 = {}
    for entry in asin_data:
        asin = entry["asin"]
        uids = [u for u in entry["users_sampled"] if u in users_gauss]
        if len(uids) < 2:
            check1[asin] = {"n_users": len(uids), "mean_pairwise_dist": None}
            continue
        mus = np.stack([np.array(users_gauss[u]["mu"]) for u in uids], axis=0)
        # Euclidean distance
        pairs = list(combinations(range(len(uids)), 2))
        dists = [float(np.linalg.norm(mus[i] - mus[j])) for i, j in pairs]
        # Mahalanobis distance (using global_var as proxy for within-population variance)
        maha_dists = []
        for i, j in pairs:
            diff = mus[i] - mus[j]
            m = float(np.sqrt((diff * diff / np.maximum(global_var, 1e-3)).sum()))
            maha_dists.append(m)
        # Also: mean L2 to ASIN centroid
        asin_centroid = np.stack([z for z, _ in pool_z[asin]], axis=0).mean(axis=0)
        d_to_centroid = [float(np.linalg.norm(m - asin_centroid)) for m in mus]

        check1[asin] = {
            "n_users": len(uids),
            "n_pairs": len(pairs),
            "mean_pairwise_l2": float(np.mean(dists)),
            "median_pairwise_l2": float(np.median(dists)),
            "max_pairwise_l2": float(np.max(dists)),
            "mean_pairwise_maha": float(np.mean(maha_dists)),
            "median_pairwise_maha": float(np.median(maha_dists)),
            "mean_user_to_centroid_l2": float(np.mean(d_to_centroid)),
        }
        log(f"  {asin}: {len(uids)} users, mean L2={check1[asin]['mean_pairwise_l2']:.3f}, "
            f"mean Maha={check1[asin]['mean_pairwise_maha']:.3f}, "
            f"user→centroid L2={check1[asin]['mean_user_to_centroid_l2']:.3f}")

    with open(CHECK1_OUT, "w", encoding="utf-8") as f:
        json.dump(check1, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {CHECK1_OUT}")

    # Aggregate
    all_l2 = [check1[a]["mean_pairwise_l2"] for a in check1 if check1[a]["mean_pairwise_l2"] is not None]
    all_maha = [check1[a]["mean_pairwise_maha"] for a in check1 if check1[a]["mean_pairwise_maha"] is not None]
    log(f"  AGGREGATE: mean L2 across ASINs = {np.mean(all_l2):.3f} ± {np.std(all_l2):.3f}")
    log(f"  AGGREGATE: mean Maha across ASINs = {np.mean(all_maha):.3f} ± {np.std(all_maha):.3f}")

    # === CHECK 2: Top-K candidate overlap rate ===
    log("\n=== CHECK 2: Top-K candidate overlap rate per ASIN ===")
    check2 = {}
    for entry in asin_data:
        asin = entry["asin"]
        uids = [u for u in entry["users_sampled"] if u in users_gauss]
        zqs = pool_z.get(asin, [])
        strict_zqs = [(z, q) for z, q in zqs if q.get("strict", True)]
        if len(strict_zqs) < max(TOP_K_VALUES):
            log(f"  {asin}: pool too small ({len(strict_zqs)}), skip")
            continue

        # Compute per-user top-K candidates (by Maha distance)
        user_topk = {}
        for uid in uids:
            mu = np.array(users_gauss[uid]["mu"])
            sigma = np.array(users_gauss[uid]["sigma_diag"])
            distances = np.array([mahalanobis_sq(z, mu, sigma) for z, _ in strict_zqs])
            # Index of candidate (use id(q) for stable identification)
            ids = [id(q) for z, q in strict_zqs]
            for K in TOP_K_VALUES:
                top_idx = np.argsort(distances)[:K]
                user_topk.setdefault(uid, {})[K] = set(ids[i] for i in top_idx)

        per_K = {}
        for K in TOP_K_VALUES:
            pair_jaccards = []
            for uid_a, uid_b in combinations(uids, 2):
                pair_jaccards.append(jaccard(user_topk[uid_a][K], user_topk[uid_b][K]))
            per_K[f"K={K}"] = {
                "n_pairs": len(pair_jaccards),
                "mean_jaccard": float(np.mean(pair_jaccards)) if pair_jaccards else 0.0,
                "median_jaccard": float(np.median(pair_jaccards)) if pair_jaccards else 0.0,
                "min_jaccard": float(np.min(pair_jaccards)) if pair_jaccards else 0.0,
                "max_jaccard": float(np.max(pair_jaccards)) if pair_jaccards else 0.0,
            }
        check2[asin] = {
            "n_users": len(uids),
            "n_pool": len(strict_zqs),
            **{k: v for k, v in per_K.items()},
        }
        log(f"  {asin}: pool={len(strict_zqs)}, "
            f"K=1 mean_jaccard={per_K['K=1']['mean_jaccard']:.3f}, "
            f"K=3 mean_jaccard={per_K['K=3']['mean_jaccard']:.3f}, "
            f"K=5 mean_jaccard={per_K['K=5']['mean_jaccard']:.3f}, "
            f"K=10 mean_jaccard={per_K['K=10']['mean_jaccard']:.3f}")

    with open(CHECK2_OUT, "w", encoding="utf-8") as f:
        json.dump(check2, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {CHECK2_OUT}")

    # Aggregate
    for K in TOP_K_VALUES:
        vals = [check2[a][f"K={K}"]["mean_jaccard"] for a in check2 if f"K={K}" in check2[a]]
        if vals:
            log(f"  AGGREGATE K={K}: mean Jaccard across ASINs = {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    # === CHECK 3: Selected-query uniqueness + entropy ===
    log("\n=== CHECK 3: Selected-query uniqueness + entropy per ASIN ===")
    check3 = {}
    for entry in asin_data:
        asin = entry["asin"]
        uids = [u for u in entry["users_sampled"] if u in users_gauss]
        zqs = pool_z.get(asin, [])
        strict_zqs = [(z, q) for z, q in zqs if q.get("strict", True)]
        if not strict_zqs:
            continue

        selected_qs = []
        for uid in uids:
            mu = np.array(users_gauss[uid]["mu"])
            sigma = np.array(users_gauss[uid]["sigma_diag"])
            distances = np.array([mahalanobis_sq(z, mu, sigma) for z, _ in strict_zqs])
            best_idx = int(np.argmin(distances))
            selected_qs.append(strict_zqs[best_idx][1]["query"])

        n_unique = len(set(selected_qs))
        ent = selection_entropy(selected_qs)
        family_counts = collections.Counter()
        for q_text in selected_qs:
            for z, q in strict_zqs:
                if q["query"] == q_text:
                    family_counts[q.get("family", "?")] += 1
                    break
        check3[asin] = {
            "n_users": len(uids),
            "n_pool": len(strict_zqs),
            "n_unique_selected": n_unique,
            "uniqueness_ratio": n_unique / len(uids),
            "selection_entropy_bits": ent,
            "max_entropy_bits": float(np.log2(len(uids))) if len(uids) > 1 else 0.0,
            "family_distribution": dict(family_counts),
            "selected_queries": selected_qs,
        }
        log(f"  {asin}: pool={len(strict_zqs)}, n_unique={n_unique}/{len(uids)}, "
            f"entropy={ent:.3f}/{check3[asin]['max_entropy_bits']:.3f} bits, "
            f"families={dict(family_counts)}")

    with open(CHECK3_OUT, "w", encoding="utf-8") as f:
        json.dump(check3, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {CHECK3_OUT}")

    # === CHECK 4: Pool-size sweep ===
    log("\n=== CHECK 4: Pool-size sweep (does uniqueness grow with pool?) ===")
    rng = np.random.RandomState(SEED)
    check4 = {size: {} for size in POOL_SWEEP_SIZES}
    for size in POOL_SWEEP_SIZES:
        log(f"\n  Pool size = {size}")
        for entry in asin_data:
            asin = entry["asin"]
            uids = [u for u in entry["users_sampled"] if u in users_gauss]
            zqs = pool_z.get(asin, [])
            strict_zqs = [(z, q) for z, q in zqs if q.get("strict", True)]
            if len(strict_zqs) < size:
                continue

            # Sub-sample
            idx = rng.choice(len(strict_zqs), size=size, replace=False)
            sub_zqs = [strict_zqs[i] for i in idx]

            selected_qs = []
            for uid in uids:
                mu = np.array(users_gauss[uid]["mu"])
                sigma = np.array(users_gauss[uid]["sigma_diag"])
                distances = np.array([mahalanobis_sq(z, mu, sigma) for z, _ in sub_zqs])
                best_idx = int(np.argmin(distances))
                selected_qs.append(sub_zqs[best_idx][1]["query"])

            n_unique = len(set(selected_qs))
            ent = selection_entropy(selected_qs)
            check4[size][asin] = {
                "n_users": len(uids),
                "n_pool_sub": size,
                "n_unique_selected": n_unique,
                "uniqueness_ratio": n_unique / len(uids),
                "selection_entropy_bits": ent,
            }
        # Aggregate
        all_unique = [check4[size][a]["n_unique_selected"] for a in check4[size] if "n_unique_selected" in check4[size][a]]
        all_ent = [check4[size][a]["selection_entropy_bits"] for a in check4[size] if "selection_entropy_bits" in check4[size][a]]
        if all_unique:
            log(f"    AGGREGATE size={size}: n_ASINs={len(all_unique)}, "
                f"mean unique={np.mean(all_unique):.2f}, "
                f"mean entropy={np.mean(all_ent):.3f} bits")

    with open(CHECK4_OUT, "w", encoding="utf-8") as f:
        json.dump(check4, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {CHECK4_OUT}")

    # === Summary ===
    summary = {
        "config": {
            "description": "Stage 9E-P — Personalization Separability Diagnostic",
            "n_asins": len(asin_data),
            "POOL_SWEEP_SIZES": POOL_SWEEP_SIZES,
            "TOP_K_VALUES": TOP_K_VALUES,
            "SEED": SEED,
        },
        "check1_user_dist": {
            "aggregate_mean_l2": float(np.mean(all_l2)),
            "aggregate_mean_maha": float(np.mean(all_maha)),
            "per_asin": check1,
        },
        "check2_top_k_overlap": {
            f"K={K}": {
                "aggregate_mean_jaccard": float(np.mean([
                    check2[a][f"K={K}"]["mean_jaccard"]
                    for a in check2 if f"K={K}" in check2[a]
                ])) if any(f"K={K}" in check2[a] for a in check2) else None,
            }
            for K in TOP_K_VALUES
        },
        "check3_uniqueness": check3,
        "check4_pool_sweep": {
            str(size): {
                "n_asins_tested": len(check4[size]),
                "mean_unique_selected": float(np.mean([
                    check4[size][a]["n_unique_selected"]
                    for a in check4[size] if "n_unique_selected" in check4[size][a]
                ])) if check4[size] else None,
            }
            for size in POOL_SWEEP_SIZES
        },
    }
    with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {SUMMARY_OUT}")

    log("\n=== Stage 9E-P complete ===")


if __name__ == "__main__":
    main()