"""Stage 10K — Selection Improvement Strategies.

Stage 10G mean_t50 produces 6.46 unique selected queries per ASIN (10 users).
This stage tests two strategies to push unique higher while keeping
sel_dist low (preserving personalization):

  A1: reject-repeat — when a user's mean_t50 selection collides with
      an already-picked query, fall back to next-best unique candidate
  C3: cohort cluster — cluster users by Mahalanobis profile, force
      each cluster to take a DIFFERENT mean_t50 candidate

Both strategies operate on the same candidate pool and user Gaussians
as Stage 10G, so they share setup.

GO criteria:
- unique_mean > 6.46 (Stage 10G baseline)
- sel_dist_mean < rnd_dist_mean (preserved personalization)
- top1_jaccard < 0.354 (Stage 10G baseline)

Inputs:
  - stage9g_user_gaussians_3levels.json (user mu/sigma_diag in 48d)
  - stage8_5_pool.json (100 ASINs × 50 candidates)
  - stage8_5_selection.json (asin, user) pairs

Outputs:
  - stage10k_selection.json — per-ASIN × strategy selection table
  - stage10k_summary.json — metrics comparison
"""

from __future__ import annotations

import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/gaussian")

from gaussian_vades import _syntax_subspace_prepare  # noqa: E402


SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")

USER_GAUSSIANS = SCRATCH / "stage9g_user_gaussians_3levels.json"
POOL_8_5 = SCRATCH / "stage8_5_pool.json"
STAGE8_5_SELECTION = SCRATCH / "stage8_5_selection.json"

OUTPUT_JSON = SCRATCH / "stage10k_selection.json"
OUTPUT_SUMMARY = SCRATCH / "stage10k_summary.json"

PCA_DIM = 48
PCA_SEED = 2024
THRESHOLD_PCT = 50  # match Stage 10G mean_t50


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def maha_diag_batch(z_arr: np.ndarray, mu_arr: np.ndarray, sigma_arr: np.ndarray) -> np.ndarray:
    """Compute [Q, U] Mahalanobis distance matrix using diagonal sigma."""
    # diff: [Q, U, 48]
    diff = z_arr[:, None, :] - mu_arr[None, :, :]
    return np.sum(diff ** 2 / sigma_arr[None, :, :], axis=2)


def main():
    log("=== Stage 10K — Selection Improvement Strategies (A1 + C3) ===")

    # === 1. Load user Gaussians ===
    log("\n=== 1. Loading user Gaussians ===")
    g_data = json.load(open(USER_GAUSSIANS))
    base_users = g_data["users"]["base"]
    log(f"  base users: {len(base_users)}")
    user_mu = {}
    user_sigma = {}
    for uid, u in base_users.items():
        user_mu[uid] = np.array(u["mu"], dtype=np.float64)
        user_sigma[uid] = np.array(u["sigma_diag"], dtype=np.float64)

    # === 2. Load scaler + feature_names + fit PCA48 ===
    log("\n=== 2. Loading scaler + PCA48 ===")
    P = _syntax_subspace_prepare()
    scaler = P["scaler"]
    feature_names = P["feature_names_ordered"]

    feat_cache_path = SCRATCH / "stage7b_query_features.jsonl.gz"
    feat_map = {}
    with gzip.open(feat_cache_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            feat_map[rec["k"]] = rec["v"]
    log(f"  feat_map: {len(feat_map)}")

    rng = np.random.default_rng(42)
    train_idx = rng.choice(len(P["X_scaled"]), size=min(5000, len(P["X_scaled"])), replace=False)
    pca = PCA(n_components=PCA_DIM, random_state=PCA_SEED)
    pca.fit(P["X_scaled"][train_idx])
    log(f"  PCA48 EV={pca.explained_variance_ratio_.sum():.4f}")

    # === 3. Project pool queries to 48d ===
    log("\n=== 3. Projecting pool queries ===")
    pool_data = json.load(open(POOL_8_5))
    pool_by_asin = pool_data["pools"]
    log(f"  n asins: {len(pool_by_asin)}")

    pool_z_by_asin = {}
    n_projected = 0
    n_skipped = 0
    for asin, queries in pool_by_asin.items():
        zs = []
        for q in queries:
            text = q["query"]
            import hashlib
            k = hashlib.sha1(text.strip().lower().encode("utf-8")).hexdigest()
            feats = feat_map.get(k)
            if not feats:
                zs.append(None)
                n_skipped += 1
                continue
            numeric = {n: float(v) for n, v in feats.items() if isinstance(v, (int, float))}
            vec = np.array([numeric.get(n, 0.0) for n in feature_names], dtype=np.float64)
            vec_scaled = scaler.transform(vec[None, :])[0]
            vec_pca = pca.transform(vec_scaled[None, :])[0]
            zs.append(vec_pca)
            n_projected += 1
        pool_z_by_asin[asin] = zs
    log(f"  projected: {n_projected}, skipped: {n_skipped}")

    # === 4. Load (asin, user) pairs ===
    log("\n=== 4. Loading (asin, user) pairs ===")
    sel_data = json.load(open(STAGE8_5_SELECTION))
    asin_users = defaultdict(list)
    for e in sel_data["entries"]:
        asin = e["asin"]
        uid = e["user_id"]
        if uid not in user_mu:
            continue
        if uid not in asin_users[asin]:  # dedupe
            asin_users[asin].append(uid)
    log(f"  n asins with users: {len(asin_users)}")

    # === 5. Selection strategies ===
    log("\n=== 5. Computing margins + running strategies ===")

    def run_strategy(z_arr: np.ndarray, mu_arr: np.ndarray, sigma_arr: np.ndarray,
                     valid_users: list, strategy: str):
        """
        z_arr: [Q, 48]
        mu_arr: [U, 48]
        sigma_arr: [U, 48]
        valid_users: list of user_ids

        Returns: (selected_idx_per_user, margin_per_user)
        """
        n_q = z_arr.shape[0]
        n_u = len(valid_users)
        D = maha_diag_batch(z_arr, mu_arr, sigma_arr)  # [Q, U]

        # Per-user: d_self, d_mean_other, margin_mean
        margins = np.zeros((n_u, n_q))
        d_selfs = np.zeros((n_u, n_q))
        for ui in range(n_u):
            d_self = D[:, ui]
            d_others = np.delete(D, ui, axis=1)
            d_mean_other = np.mean(d_others, axis=1)
            margin = d_mean_other - d_self
            margins[ui] = margin
            d_selfs[ui] = d_self

        if strategy == "mean_t50":
            # Baseline: argmax margin where d_self <= 50th percentile
            selected = []
            for ui in range(n_u):
                d_self = d_selfs[ui]
                margin = margins[ui]
                threshold = float(np.percentile(d_self, THRESHOLD_PCT))
                valid = d_self <= threshold
                if not valid.any():
                    idx = int(np.argmax(margin))
                else:
                    m = np.where(valid, margin, -np.inf)
                    idx = int(np.argmax(m))
                selected.append(idx)
            return selected, margins, d_selfs

        elif strategy == "A1_reject_repeat":
            # If user's best collides with already-picked, take next-best unique
            # Process users in their natural order; greedy
            selected = []
            picked = set()
            for ui in range(n_u):
                d_self = d_selfs[ui]
                margin = margins[ui]
                threshold = float(np.percentile(d_self, THRESHOLD_PCT))
                valid = d_self <= threshold
                m = np.where(valid, margin, -np.inf)
                # Rank in descending order of margin
                ranked = np.argsort(-m)
                chosen = None
                for c in ranked:
                    if c not in picked:
                        chosen = int(c)
                        break
                if chosen is None:
                    # Fallback: pick best overall
                    chosen = int(ranked[0])
                selected.append(chosen)
                picked.add(chosen)
            return selected, margins, d_selfs

        elif strategy.startswith("C3_kmeans_k"):
            k = int(strategy.rsplit("_k", 1)[1])
            # Cluster users by Mahalanobis profile (mu vectors)
            # Then run A1 reject-repeat WITHIN each cluster (intra-cluster uniqueness)
            user_clusters = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(mu_arr)

            # Per-cluster: apply A1 reject-repeat
            selected = [0] * n_u
            used_global = set()  # also enforce cross-cluster distinctness
            for c in range(k):
                cluster_mask = (user_clusters == c)
                cluster_user_idxs = np.where(cluster_mask)[0]
                if len(cluster_user_idxs) == 0:
                    continue
                # Process this cluster's users
                for local_i, ui in enumerate(cluster_user_idxs):
                    d_self = d_selfs[ui]
                    margin = margins[ui]
                    threshold = float(np.percentile(d_self, THRESHOLD_PCT))
                    valid = d_self <= threshold
                    m = np.where(valid, margin, -np.inf)
                    ranked = np.argsort(-m)
                    chosen = None
                    for cand in ranked:
                        if cand not in used_global:
                            chosen = int(cand)
                            break
                    if chosen is None:
                        chosen = int(ranked[0])
                    selected[ui] = chosen
                    used_global.add(chosen)
            return selected, margins, d_selfs

        else:
            raise ValueError(f"unknown strategy: {strategy}")

    strategies = [
        "mean_t50",
        "A1_reject_repeat",
        "C3_kmeans_k2",
        "C3_kmeans_k3",
        "C3_kmeans_k4",
        "C3_kmeans_k5",
    ]

    per_asin_results = {}
    rnd_dist_per_asin = {}  # for sel/rnd ratio

    for ai, (asin, users) in enumerate(asin_users.items()):
        if ai % 10 == 0:
            log(f"  [{ai}/{len(asin_users)}] asin={asin}, n_users={len(users)}")
        zs = pool_z_by_asin.get(asin, [])
        valid_idx = [i for i, z in enumerate(zs) if z is not None]
        if len(valid_idx) < 5 or len(users) < 2:
            continue
        z_arr = np.stack([zs[i] for i in valid_idx])
        valid_users = [u for u in users if u in user_mu]
        if len(valid_users) < 2:
            continue

        mu_arr = np.stack([user_mu[uid] for uid in valid_users])
        sigma_arr = np.stack([user_sigma[uid] for uid in valid_users])

        # Random baseline distance (mean d_self per user)
        D = maha_diag_batch(z_arr, mu_arr, sigma_arr)
        rnd_per_user = [float(np.mean(D[:, ui])) for ui in range(len(valid_users))]
        rnd_dist_per_asin[asin] = np.mean(rnd_per_user)

        asin_strategies = {}
        for strategy in strategies:
            sel_idx, margins, d_selfs = run_strategy(z_arr, mu_arr, sigma_arr,
                                                     valid_users, strategy)
            # Per-user selected d_self
            sel_d_self = [float(d_selfs[ui, sel_idx[ui]]) for ui in range(len(valid_users))]
            asin_strategies[strategy] = {
                "selected_idx": [int(i) for i in sel_idx],
                "selected_d_self": sel_d_self,
                "n_unique": len(set(sel_idx)),
            }
        per_asin_results[asin] = asin_strategies

    # === 6. Aggregate metrics ===
    log("\n=== 6. Aggregating metrics ===")
    summary = {}
    for strategy in strategies:
        uniqs = []
        sel_dists = []
        for asin, res in per_asin_results.items():
            uniqs.append(res[strategy]["n_unique"])
            sel_dists.extend(res[strategy]["selected_d_self"])
        rnd_mean = np.mean(list(rnd_dist_per_asin.values()))
        summary[strategy] = {
            "n_asins": len(per_asin_results),
            "unique_mean": float(np.mean(uniqs)),
            "unique_median": float(np.median(uniqs)),
            "unique_min": int(min(uniqs)),
            "unique_max": int(max(uniqs)),
            "sel_dist_mean": float(np.mean(sel_dists)),
            "rnd_dist_mean": float(rnd_mean),
            "sel_rnd_ratio": float(np.mean(sel_dists) / rnd_mean),
        }

    log(f"\n{'strategy':<25} {'unique_mean':>12} {'unique_med':>11} {'sel_dist':>10} {'rnd_dist':>10} {'ratio':>7}")
    for strategy in strategies:
        s = summary[strategy]
        log(f"  {strategy:<23} {s['unique_mean']:>11.2f} {s['unique_median']:>11.1f} "
            f"{s['sel_dist_mean']:>9.1f} {s['rnd_dist_mean']:>9.1f} {s['sel_rnd_ratio']:>6.2f}")

    # Save outputs
    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "config": {
                "description": "Stage 10K selection strategies",
                "PCA_DIM": PCA_DIM,
                "THRESHOLD_PCT": THRESHOLD_PCT,
                "strategies": strategies,
            },
            "per_asin": per_asin_results,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {OUTPUT_JSON}")

    with open(OUTPUT_SUMMARY, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {OUTPUT_SUMMARY}")

    log("\n=== Stage 10K complete ===")


if __name__ == "__main__":
    main()