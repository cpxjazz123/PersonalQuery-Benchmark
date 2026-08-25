"""Stage 10H-A — Contrastive Selection Stability.

Stage 10G's mean_t50 selection has a cohort dependency: the selected query
depends on WHICH other users are included as negatives. If we randomly sample
different negative-user subsets, does mean_t50 still produce the same query?

This stage verifies that mean_t50 is stable across negative-user sampling,
meaning the contrastive term rewards user-specific queries robustly, not
specific to any one batch of negative users.

Method:
- For each (asin, user), compute D matrix [Q × U].
- For each user, sample multiple random subsets of negative users of sizes
  {5, 10, 20, 50} from the OTHER users in the same ASIN.
- For each subset, run mean_t50 selection (mean margin + 50th pct threshold).
- Aggregate per-user metrics:
  - selection_mode: most common selected query across subsamples
  - selection_agreement: fraction of subsamples matching mode
  - unique_selected: number of distinct queries across subsamples
  - mean_self_distance: average D(q_sel, u) across subsamples
- Compare to absolute Maha (no negatives) selection.

GO criteria:
- selection_agreement > 0.5 for subset_size >= 5
  (i.e., >50% of subsamples produce the same query)
- unique_selected / n_subsamples < 0.3
  (most subsamples converge to same/close queries)
- mean_self_distance similar to absolute Maha baseline

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \\
        gaussian/syntax_subspace_stage10h_a_stability.py \\
        > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage10h_a_run.log 2>&1 &
"""

from __future__ import annotations

import gzip
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/gaussian")
from gaussian_vades import _syntax_subspace_prepare  # noqa: E402


SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
USER_GAUSSIANS = SCRATCH / "stage9g_user_gaussians_3levels.json"
POOL_8_5 = SCRATCH / "stage8_5_pool.json"
STAGE8_5_SELECTION = SCRATCH / "stage8_5_selection.json"

OUTPUT_JSON = SCRATCH / "stage10h_a_stability.json"

PCA_DIM = 48
PCA_SEED = 2024
N_REPLICATES = 20          # subsamples per user per subset size
SUBSET_SIZES = [5, 10, 20, 50]
THRESHOLD_PCT = 50         # mean_t50
SEED_BASE = 42


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    log("=== Stage 10H-A — Contrastive Selection Stability ===")

    # === 1. Load user Gaussians + scaler + PCA48 (same as Stage 10G) ===
    log("\n=== 1. Loading user Gaussians + scaler + PCA48 ===")
    import json as _json
    g_data = _json.load(open(USER_GAUSSIANS))
    base_users = g_data["users"]["base"]
    user_mu = {uid: np.array(u["mu"], dtype=np.float64) for uid, u in base_users.items()}
    user_sigma = {uid: np.array(u["sigma_diag"], dtype=np.float64) for uid, u in base_users.items()}
    log(f"  user Gaussians: {len(user_mu)}")

    P = _syntax_subspace_prepare()
    scaler = P["scaler"]
    feature_names = P["feature_names_ordered"]
    log(f"  scaler loaded, feature_names: {len(feature_names)}")

    feat_cache_path = SCRATCH / "stage7b_query_features.jsonl.gz"
    feat_map = {}
    with gzip.open(feat_cache_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = _json.loads(line)
            feat_map[rec["k"]] = rec["v"]
    log(f"  feat_map size: {len(feat_map)}")

    rng = np.random.default_rng(42)
    train_idx = rng.choice(len(P["X_scaled"]), size=min(5000, len(P["X_scaled"])), replace=False)
    pca = PCA(n_components=PCA_DIM, random_state=PCA_SEED)
    pca.fit(P["X_scaled"][train_idx])
    log(f"  PCA48 EV={pca.explained_variance_ratio_.sum():.4f}")

    # === 2. Project pool queries to 48d ===
    log("\n=== 2. Projecting pool queries ===")
    pool_data = _json.load(open(POOL_8_5))
    pool_by_asin = pool_data["pools"]

    def project_query(text):
        import hashlib
        k = hashlib.sha1(text.strip().lower().encode("utf-8")).hexdigest()
        feats = feat_map.get(k)
        if not feats:
            return None
        numeric = {n: float(v) for n, v in feats.items() if isinstance(v, (int, float))}
        if not numeric or len(numeric) < len(feature_names) * 0.5:
            return None
        vec = np.array([numeric.get(n, 0.0) for n in feature_names], dtype=np.float64)
        vec_scaled = scaler.transform(vec[None, :])[0]
        return pca.transform(vec_scaled[None, :])[0]

    pool_z_by_asin = {}
    n_skipped = 0
    for asin, queries in pool_by_asin.items():
        zs = []
        for q in queries:
            z = project_query(q["query"])
            if z is None:
                zs.append(None)
                n_skipped += 1
            else:
                zs.append(z)
        pool_z_by_asin[asin] = zs
    log(f"  pool projected, skipped: {n_skipped}")

    # === 3. Load (asin, user) pairs ===
    sel_data = _json.load(open(STAGE8_5_SELECTION))
    from collections import defaultdict
    asin_users = defaultdict(list)
    for e in sel_data["entries"]:
        asin = e["asin"]
        uid = e["user_id"]
        if uid not in user_mu:
            continue
        if uid not in asin_users[asin]:
            asin_users[asin].append(uid)
    log(f"  asins with valid users: {len(asin_users)}")

    # === 4. Per-user stability test ===
    log("\n=== 4. Running stability test (mean_t50 across N_REPLICATES × SUBSET_SIZES) ===")

    # Storage: per-user stability metrics per subset size
    stability_by_size = {sz: defaultdict(list) for sz in SUBSET_SIZES}
    abs_compare = []  # (asin, user, abs_idx, abs_dist)

    n_asins_processed = 0
    n_users_processed = 0
    sample_rng = np.random.default_rng(SEED_BASE)

    # Pool all user_mu/sigma into a global "other-user" pool for cross-ASIN sampling
    all_user_ids = sorted(user_mu.keys())
    all_mu_arr = np.stack([user_mu[uid] for uid in all_user_ids])  # [N_all, 48]
    all_sigma_arr = np.stack([user_sigma[uid] for uid in all_user_ids])  # [N_all, 48]
    user_to_global_idx = {uid: i for i, uid in enumerate(all_user_ids)}
    log(f"  global user pool: {len(all_user_ids)}")

    for asin, users in asin_users.items():
        zs = pool_z_by_asin.get(asin, [])
        valid_idx = [i for i, z in enumerate(zs) if z is not None]
        if len(valid_idx) < 5 or len(users) < 2:
            continue
        z_arr = np.stack([zs[i] for i in valid_idx])  # [Q, 48]
        n_q = len(valid_idx)
        valid_users = [u for u in users if u in user_mu]
        if len(valid_users) < 2:
            continue

        mu_arr = np.stack([user_mu[uid] for uid in valid_users])  # [U_asin, 48]
        sigma_arr = np.stack([user_sigma[uid] for uid in valid_users])  # [U_asin, 48]
        diff = z_arr[:, None, :] - mu_arr[None, :, :]
        D = np.sum(diff ** 2 / sigma_arr[None, :, :], axis=2)  # [Q, U_asin]

        n_asins_processed += 1

        for ui, uid in enumerate(valid_users):
            d_self = D[:, ui]
            target_global_idx = user_to_global_idx[uid]

            # Absolute Maha (no subset) for comparison
            abs_idx = int(np.argmin(d_self))
            abs_dist = float(d_self[abs_idx])
            abs_compare.append({
                "asin": asin,
                "user_id": uid,
                "abs_idx": abs_idx,
                "abs_dist": abs_dist,
            })

            # mean_t50 with FULL global negative set (Stage 10G selection)
            other_global_idx = np.array([i for i in range(len(all_user_ids)) if i != target_global_idx])
            d_others_full = np.zeros((n_q, len(other_global_idx)))
            for j, g_idx in enumerate(other_global_idx):
                mu_j = all_mu_arr[g_idx]
                sigma_j = all_sigma_arr[g_idx]
                diff_j = z_arr - mu_j[None, :]
                d_others_full[:, j] = np.sum(diff_j ** 2 / sigma_j[None, :], axis=1)
            full_margin = np.mean(d_others_full, axis=1) - d_self
            full_threshold = float(np.percentile(d_self, THRESHOLD_PCT))
            full_valid = d_self <= full_threshold
            full_m = np.where(full_valid, full_margin, -np.inf)
            full_idx = int(np.argmax(full_m))

            # Stability: for each subset size, sample N_REPLICATES negative subsets from GLOBAL pool
            for subset_size in SUBSET_SIZES:
                selections = []
                self_dists = []

                for rep in range(N_REPLICATES):
                    # Sample subset_size negative indices from other_global_idx
                    neg_sub = sample_rng.choice(other_global_idx, size=min(subset_size, len(other_global_idx)), replace=False)
                    # Compute D for these neg users
                    d_others_sub = np.zeros((n_q, len(neg_sub)))
                    for j, g_idx in enumerate(neg_sub):
                        mu_j = all_mu_arr[g_idx]
                        sigma_j = all_sigma_arr[g_idx]
                        diff_j = z_arr - mu_j[None, :]
                        d_others_sub[:, j] = np.sum(diff_j ** 2 / sigma_j[None, :], axis=1)
                    margin = np.mean(d_others_sub, axis=1) - d_self
                    threshold = float(np.percentile(d_self, THRESHOLD_PCT))
                    valid_mask = d_self <= threshold
                    m = np.where(valid_mask, margin, -np.inf)
                    sel_idx = int(np.argmax(m))
                    selections.append(sel_idx)
                    self_dists.append(float(d_self[sel_idx]))

                counter = Counter(selections)
                mode_idx, mode_count = counter.most_common(1)[0]
                agreement = mode_count / N_REPLICATES
                unique_count = len(set(selections))
                full_match_count = sum(1 for s in selections if s == full_idx)

                stability_by_size[subset_size]["agreement"].append(agreement)
                stability_by_size[subset_size]["unique_count"].append(unique_count)
                stability_by_size[subset_size]["mean_self_dist"].append(float(np.mean(self_dists)))
                stability_by_size[subset_size]["full_match_rate"].append(full_match_count / N_REPLICATES)
                stability_by_size[subset_size]["matches_abs"].append(sum(1 for s in selections if s == abs_idx) / N_REPLICATES)

            n_users_processed += 1

    log(f"  asins processed: {n_asins_processed}, users processed: {n_users_processed}")

    # === 5. Aggregate stability summary ===
    log("\n=== 5. Stability Summary ===")
    summary = {}
    for sz in SUBSET_SIZES:
        d = stability_by_size[sz]
        if not d["agreement"]:
            continue
        summary[f"subset_{sz}"] = {
            "n_users": len(d["agreement"]),
            "agreement_mean": float(np.mean(d["agreement"])),
            "agreement_median": float(np.median(d["agreement"])),
            "agreement_p25": float(np.percentile(d["agreement"], 25)),
            "agreement_p75": float(np.percentile(d["agreement"], 75)),
            "unique_count_mean": float(np.mean(d["unique_count"])),
            "unique_count_max": int(np.max(d["unique_count"])) if d["unique_count"] else 0,
            "unique_count_min": int(np.min(d["unique_count"])) if d["unique_count"] else 0,
            "mean_self_dist": float(np.mean(d["mean_self_dist"])),
            "mean_self_dist_std": float(np.std(d["mean_self_dist"])),
            "full_match_rate": float(np.mean(d["full_match_rate"])),
            "matches_abs_rate": float(np.mean(d["matches_abs"])),
        }

    log(f"  {'subset':<12} {'agreement':<10} {'full_match':<12} {'sel/rnd':<10} {'unique':<8}")
    log(f"  {'-'*60}")
    for sz in SUBSET_SIZES:
        s = summary.get(f"subset_{sz}")
        if s is None:
            continue
        log(f"  size={sz:<8} {s['agreement_mean']:.3f}      "
            f"{s['full_match_rate']:.3f}        "
            f"{s['mean_self_dist']:.2f}    "
            f"{s['unique_count_mean']:.2f}")

    # === 6. Comparison vs absolute Maha ===
    log("\n=== 6. Comparison vs Absolute Maha (baseline) ===")
    abs_dist_mean = float(np.mean([c["abs_dist"] for c in abs_compare]))
    log(f"  absolute Maha: sel_dist mean = {abs_dist_mean:.2f}")
    for sz in SUBSET_SIZES:
        s = summary.get(f"subset_{sz}")
        if s is None:
            continue
        diff = s["mean_self_dist"] - abs_dist_mean
        log(f"  subset_size={sz}: mean_self_dist = {s['mean_self_dist']:.2f} (diff = {diff:+.2f} vs absolute)")

    # === 7. Save ===
    log("\n=== 7. Saving results ===")
    output = {
        "config": {
            "PCA_DIM": PCA_DIM,
            "PCA_SEED": PCA_SEED,
            "N_REPLICATES": N_REPLICATES,
            "SUBSET_SIZES": SUBSET_SIZES,
            "THRESHOLD_PCT": THRESHOLD_PCT,
            "SEED_BASE": SEED_BASE,
            "n_users_total": n_users_processed,
            "n_asins_total": n_asins_processed,
        },
        "stability_summary": summary,
        "abs_dist_mean": abs_dist_mean,
    }
    with open(OUTPUT_JSON, "w") as f:
        _json.dump(output, f, indent=2)
    log(f"  wrote → {OUTPUT_JSON}")

    log("\n=== Stage 10H-A complete ===")


if __name__ == "__main__":
    main()