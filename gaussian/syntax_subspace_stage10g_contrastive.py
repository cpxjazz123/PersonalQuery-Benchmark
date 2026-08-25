"""Stage 10G — Contrastive Mahalanobis Selection.

Stage 9G (commit 72a4f8b) confirmed feature enrichment does NOT break
personalization collapse (3.30 → 2.40 unique per ASIN). The bottleneck
is the absolute Maha argmin — it selects the same centroid query for
many users.

Stage 10G pivots the SELECTION mechanism:

1. Absolute Maha (baseline, Stage 8.5):
   q* = argmin_q D(q, u)

2. Nearest-other margin:
   M_nearest(q, u) = min_{v != u} D(q, v) - D(q, u)
   q* = argmax_q M_nearest(q, u)

3. Mean-other margin:
   M_mean(q, u) = mean_{v != u} D(q, v) - D(q, u)
   q* = argmax_q M_mean(q, u)

A query at the centroid of many users will have low margin even if its
absolute distance is small, so it gets deprioritized. A query uniquely
close to user u (vs other users) gets prioritized.

Self-distance threshold constraint:
- D(q, u) <= threshold_pct * (e.g., 25th/50th/75th percentile of d_self)
- Prevents selecting a query that is "different from everyone, including u"

Pool: Stage 8.5 strict pool (100 asins × ~30-50 queries each)
User Gaussians: Stage 9G base level (294 users, mu/sigma_diag in 48d PCA48 space)
Scaler + PCA48: from gaussian_vades._syntax_subspace_prepare()

GO criteria:
- Unique selected / 10 users: 3.3 → 5-6+
- Top-1 Jaccard: < 0.46
- Selected self-distance vs random: < 0.7x random distance (preserved)

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \\
        gaussian/syntax_subspace_stage10g_contrastive.py \\
        > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage10g_run.log 2>&1 &
"""

from __future__ import annotations

import gzip
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/gaussian")

# Reuse the SAME scaler + feature_names that Stage 8.5 / 9G used
from gaussian_vades import _syntax_subspace_prepare  # noqa: E402


SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")

USER_GAUSSIANS = SCRATCH / "stage9g_user_gaussians_3levels.json"
POOL_8_5 = SCRATCH / "stage8_5_pool.json"
STAGE8_5_SELECTION = SCRATCH / "stage8_5_selection.json"

OUTPUT_JSON = SCRATCH / "stage10g_contrastive_selection.json"

# Use a SEED different from Stage 9G to confirm robustness
PCA_DIM = 48
PCA_SEED = 2024
THRESHOLD_PCTS = [25, 50, 75]  # percentile thresholds for self-distance cap


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def project_pool_query(query_text: str, feat_map: dict, scaler, pca, feature_names) -> np.ndarray | None:
    """Project a single query to 48d PCA space."""
    import hashlib
    k = hashlib.sha1(query_text.strip().lower().encode("utf-8")).hexdigest()
    feats = feat_map.get(k)
    if not feats:
        return None
    # Numeric only, in feature_names order
    numeric = {n: float(v) for n, v in feats.items() if isinstance(v, (int, float))}
    if not numeric or len(numeric) < len(feature_names) * 0.5:
        return None
    vec = np.array([numeric.get(n, 0.0) for n in feature_names], dtype=np.float64)
    vec_scaled = scaler.transform(vec[None, :])[0]
    vec_pca = pca.transform(vec_scaled[None, :])[0]
    return vec_pca


def main():
    log("=== Stage 10G — Contrastive Mahalanobis Selection ===")

    # === 1. Load user Gaussians (base level) ===
    log("\n=== 1. Loading user Gaussians (base level) ===")
    g_data = __import__("json").load(open(USER_GAUSSIANS))
    base_users = g_data["users"]["base"]
    log(f"  base users: {len(base_users)}")

    # Build user_mu[uid] = 48d, user_sigma[uid] = 48d diagonal
    user_mu = {}
    user_sigma = {}
    for uid, u in base_users.items():
        user_mu[uid] = np.array(u["mu"], dtype=np.float64)
        user_sigma[uid] = np.array(u["sigma_diag"], dtype=np.float64)
    log(f"  loaded {len(user_mu)} user Gaussians (mu=48d, sigma_diag=48d)")

    # === 2. Get scaler + feature_names from _syntax_subspace_prepare ===
    log("\n=== 2. Loading scaler + feature_names ===")
    P = _syntax_subspace_prepare()
    scaler = P["scaler"]
    feature_names = P["feature_names_ordered"]
    log(f"  scaler loaded, feature_names: {len(feature_names)}")

    # Build a query→features map from stage7b_query_features.jsonl.gz
    feat_cache_path = SCRATCH / "stage7b_query_features.jsonl.gz"
    log(f"  loading feature cache: {feat_cache_path}")
    feat_map = {}
    with gzip.open(feat_cache_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            import json as _json
            rec = _json.loads(line)
            feat_map[rec["k"]] = rec["v"]
    log(f"  feat_map size: {len(feat_map)}")

    # Fit PCA48 (same seed as Stage 8.5 / 9G)
    log("\n=== 3. Fitting PCA48 (same scaler as Stage 8.5) ===")
    rng = np.random.default_rng(42)
    train_idx = rng.choice(len(P["X_scaled"]), size=min(5000, len(P["X_scaled"])), replace=False)
    pca = PCA(n_components=PCA_DIM, random_state=PCA_SEED)
    pca.fit(P["X_scaled"][train_idx])
    log(f"  PCA48 EV={pca.explained_variance_ratio_.sum():.4f}")

    # === 4. Load Stage 8.5 strict pool (100 asins) ===
    log("\n=== 4. Loading Stage 8.5 strict pool ===")
    import json as _json
    pool_data = _json.load(open(POOL_8_5))
    pool_by_asin = pool_data["pools"]
    log(f"  n asins: {len(pool_by_asin)}")
    for asin in list(pool_by_asin.keys())[:3]:
        log(f"    sample asin: {asin}, n_queries: {len(pool_by_asin[asin])}")

    # === 5. Project all pool queries to 48d ===
    log("\n=== 5. Projecting pool queries to 48d ===")
    pool_z_by_asin = {}  # asin -> [z_48d for each query], parallel to pool_by_asin[asin]
    n_skipped = 0
    n_projected = 0
    for asin, queries in pool_by_asin.items():
        zs = []
        for q in queries:
            text = q["query"]
            z = project_pool_query(text, feat_map, scaler, pca, feature_names)
            if z is None:
                zs.append(None)
                n_skipped += 1
            else:
                zs.append(z)
                n_projected += 1
        pool_z_by_asin[asin] = zs
    log(f"  projected: {n_projected}, skipped (no 182d cache): {n_skipped}")

    # === 6. Load Stage 8.5 selection (asin, user_id) pairs ===
    log("\n=== 6. Loading Stage 8.5 (asin, user_id) pairs ===")
    sel_data = _json.load(open(STAGE8_5_SELECTION))
    log(f"  n_entries: {sel_data['n_entries']}")

    # Build (asin -> [user_id]) with users in our Gaussian set
    asin_users = defaultdict(list)
    n_kept = 0
    n_skipped_user = 0
    for e in sel_data["entries"]:
        asin = e["asin"]
        uid = e["user_id"]
        if uid not in user_mu:
            n_skipped_user += 1
            continue
        asin_users[asin].append(uid)
        n_kept += 1
    log(f"  (asin, user) kept: {n_kept}, skipped user (no Gaussian): {n_skipped_user}")
    log(f"  n asins with users: {len(asin_users)}")

    # === 7. For each ASIN, compute D matrix [n_queries x n_users_in_asin] ===
    log("\n=== 7. Computing D matrices and selection variants ===")

    def maha_diag(z: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
        diff = z - mu
        return float(np.sum(diff ** 2 / sigma))

    # Per-asin results: asin -> { variant_name -> [selected_idx_per_user] }
    per_asin_results = defaultdict(dict)

    for asin, users in asin_users.items():
        zs = pool_z_by_asin.get(asin, [])
        valid_idx = [i for i, z in enumerate(zs) if z is not None]
        if len(valid_idx) < 5 or len(users) < 2:
            continue
        # Filter to valid queries
        z_arr = np.stack([zs[i] for i in valid_idx])  # [Q, 48]
        # Filter to users with valid Gaussian
        valid_users = [u for u in users if u in user_mu]
        if len(valid_users) < 2:
            continue

        # D matrix: [Q, U] — vectorized via numpy broadcasting
        n_q = len(valid_idx)
        n_u = len(valid_users)
        mu_arr = np.stack([user_mu[uid] for uid in valid_users])  # [U, 48]
        sigma_arr = np.stack([user_sigma[uid] for uid in valid_users])  # [U, 48]
        # diff: [Q, U, 48]
        diff = z_arr[:, None, :] - mu_arr[None, :, :]
        # D = sum((z - mu)^2 / sigma) per (q, u)
        D = np.sum(diff ** 2 / sigma_arr[None, :, :], axis=2)  # [Q, U]

        # Per-user selection variants
        variant_results = {  # variant -> [selected_idx_in_z_arr per user]
            "absolute": [],
            "nearest": [],
            "mean": [],
        }
        # With thresholds
        for pct in THRESHOLD_PCTS:
            variant_results[f"absolute_t{pct}"] = []
            variant_results[f"nearest_t{pct}"] = []
            variant_results[f"mean_t{pct}"] = []

        # Diagnostics — per-variant selected distance + margins
        diag = {
            "selected_dist": defaultdict(list),  # variant -> [per-user dist]
            "selected_margin_nearest": [],
            "selected_margin_mean": [],
            "selected_dist_random_mean": [],  # for sel/rnd ratio
        }

        for ui, uid in enumerate(valid_users):
            d_self = D[:, ui]  # [Q]
            d_others = np.delete(D, ui, axis=1)  # [Q, U-1]
            d_nearest_other = np.min(d_others, axis=1)
            d_mean_other = np.mean(d_others, axis=1)
            margin_nearest = d_nearest_other - d_self
            margin_mean = d_mean_other - d_self

            # 1. Absolute Maha (baseline)
            idx_abs = int(np.argmin(d_self))
            variant_results["absolute"].append(idx_abs)
            diag["selected_dist"]["absolute"].append(float(d_self[idx_abs]))

            # 2. Nearest-other margin (no threshold)
            idx_nearest = int(np.argmax(margin_nearest))
            variant_results["nearest"].append(idx_nearest)
            diag["selected_dist"]["nearest"].append(float(d_self[idx_nearest]))
            diag["selected_margin_nearest"].append(float(margin_nearest[idx_nearest]))

            # 3. Mean-other margin (no threshold)
            idx_mean = int(np.argmax(margin_mean))
            variant_results["mean"].append(idx_mean)
            diag["selected_dist"]["mean"].append(float(d_self[idx_mean]))
            diag["selected_margin_mean"].append(float(margin_mean[idx_mean]))

            # With thresholds: only consider queries with d_self <= pct
            for pct in THRESHOLD_PCTS:
                threshold = float(np.percentile(d_self, pct))
                valid_mask = d_self <= threshold

                # Absolute Maha with threshold
                d_abs = np.where(valid_mask, d_self, np.inf)
                idx_abs_t = int(np.argmin(d_abs))
                variant_results[f"absolute_t{pct}"].append(idx_abs_t)
                diag["selected_dist"][f"absolute_t{pct}"].append(float(d_self[idx_abs_t]))

                # Nearest margin with threshold
                m_n = np.where(valid_mask, margin_nearest, -np.inf)
                idx_nearest_t = int(np.argmax(m_n))
                variant_results[f"nearest_t{pct}"].append(idx_nearest_t)
                diag["selected_dist"][f"nearest_t{pct}"].append(float(d_self[idx_nearest_t]))

                # Mean margin with threshold
                m_m = np.where(valid_mask, margin_mean, -np.inf)
                idx_mean_t = int(np.argmax(m_m))
                variant_results[f"mean_t{pct}"].append(idx_mean_t)
                diag["selected_dist"][f"mean_t{pct}"].append(float(d_self[idx_mean_t]))

            # Random baseline (per-user mean)
            diag["selected_dist_random_mean"].append(float(np.mean(d_self)))

        # Store per-ASIN results
        per_asin_results[asin] = {
            "n_users": n_u,
            "n_queries": n_q,
            "variants": variant_results,
            "diag": diag,
        }

    log(f"  per_asin_results: {len(per_asin_results)}")

    # === 8. Aggregate metrics ===
    log("\n=== 8. Aggregating per-ASIN metrics ===")

    def jaccard_topk(sels, k_frac=0.1):
        """Top-K Jaccard: average pairwise Jaccard of top-K selections."""
        if len(sels) < 2:
            return 1.0
        k = max(1, int(len(sels) * k_frac))
        # For now, use unique/total as a proxy for Jaccard
        unique = len(set(sels))
        return 1.0 - unique / len(sels)

    summary = {}
    for variant in ["absolute", "nearest", "mean"] + [f"absolute_t{p}" for p in THRESHOLD_PCTS] + [f"nearest_t{p}" for p in THRESHOLD_PCTS] + [f"mean_t{p}" for p in THRESHOLD_PCTS]:
        unique_list = []
        jaccard_top1_list = []
        sel_dist_list = []
        rnd_dist_list = []

        for asin, r in per_asin_results.items():
            sels = r["variants"][variant]
            n_users = len(sels)
            if n_users < 2:
                continue
            unique = len(set(sels))
            unique_list.append(unique)
            jaccard_top1_list.append(1.0 - unique / n_users)

            # Mean selected distance for this variant
            sel_dist_list.append(np.mean(r["diag"]["selected_dist"][variant]))
            rnd_dist_list.append(np.mean(r["diag"]["selected_dist_random_mean"]))

        summary[variant] = {
            "n_asins": len(unique_list),
            "unique_mean": float(np.mean(unique_list)),
            "unique_median": float(np.median(unique_list)),
            "jaccard_top1_mean": float(np.mean(jaccard_top1_list)),
            "jaccard_top1_median": float(np.median(jaccard_top1_list)),
            "sel_dist_mean": float(np.mean(sel_dist_list)) if sel_dist_list else None,
            "rnd_dist_mean": float(np.mean(rnd_dist_list)) if rnd_dist_list else None,
            "sel_rnd_ratio": float(np.mean(rnd_dist_list) / max(np.mean(sel_dist_list), 1e-9)) if sel_dist_list and rnd_dist_list else None,
        }

    # === 9. Print comparison table ===
    log("\n=== 9. Variant Comparison ===")
    log(f"  {'Variant':<18} {'Unique':<8} {'top1 Jacc':<12} {'sel dist':<12} {'rnd dist':<12} {'sel/rnd':<10}")
    log(f"  {'-'*80}")
    for variant in ["absolute", "nearest", "mean"] + [f"nearest_t{p}" for p in THRESHOLD_PCTS] + [f"mean_t{p}" for p in THRESHOLD_PCTS]:
        s = summary[variant]
        log(f"  {variant:<18} {s['unique_mean']:<8.2f} {s['jaccard_top1_mean']:<12.3f} {s['sel_dist_mean'] or 0:<12.2f} {s['rnd_dist_mean'] or 0:<12.2f} {s['sel_rnd_ratio'] or 0:<10.2f}")

    # === 10. Save ===
    log("\n=== 10. Saving results ===")
    output = {
        "config": {
            "PCA_DIM": PCA_DIM,
            "PCA_SEED": PCA_SEED,
            "THRESHOLD_PCTS": THRESHOLD_PCTS,
            "n_users_in_gauss": len(user_mu),
            "n_asins_in_pool": len(pool_by_asin),
            "n_asins_in_results": len(per_asin_results),
        },
        "summary": summary,
        "per_asin_results": {asin: {k: v for k, v in r.items() if k != "diag"} for asin, r in per_asin_results.items()},
    }
    with open(OUTPUT_JSON, "w") as f:
        _json.dump(output, f, indent=2)
    log(f"  wrote → {OUTPUT_JSON}")

    log("\n=== Stage 10G complete ===")


if __name__ == "__main__":
    main()