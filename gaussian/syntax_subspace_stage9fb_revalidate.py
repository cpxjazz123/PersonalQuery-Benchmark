"""Stage 9F-B revalidation — combine 9C + 9F pool, re-run selection + Stage 9E-P
diagnostic. Compare against 9C-only baseline.

Goal: verify that adding coverage queries actually improves per-user
personalization (unique selected 1-4 → 6-8 / 10, top-10 Jaccard 0.80 → 0.3-0.4).

Output:
- /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage9fb_selection_combined.json
- /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage9fb_diagnostic.json

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
        gaussian/syntax_subspace_stage9fb_revalidate.py \
        > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage9fb_reval_run.log 2>&1 &
"""

from __future__ import annotations

import collections
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
POOL_9C = SCRATCH / "stage9c_pool_pilot.json"
POOL_9F = SCRATCH / "stage9fb_pool.json"
ASINS = SCRATCH / "stage8_5_asins.json"
USER_GAUSSIANS = SCRATCH / "stage8_5_user_gaussians.json"
FEAT_CACHE = SCRATCH / "stage7b_query_features.jsonl.gz"
SELECTION_OUT = SCRATCH / "stage9fb_selection_combined.json"
DIAG_OUT = SCRATCH / "stage9fb_diagnostic.json"

SEED = 2024
PCA_DIM = 48
LAMBDA_SHRINK = 0.1


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def feat_key(t: str) -> str:
    return hashlib.sha1(t.strip().lower().encode("utf-8")).hexdigest()


def main():
    log("=== Stage 9F-B — Revalidation: combined pool selection + 9E-P diagnostic ===")

    # Load PCA48
    sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
    sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/gaussian")
    from gaussian_vades import _syntax_subspace_prepare
    P = _syntax_subspace_prepare()
    scaler = P["scaler"]
    fnames = P["feature_names_ordered"]
    train_idx = P["train_idx"]
    from sklearn.decomposition import PCA
    pca = PCA(n_components=PCA_DIM, random_state=SEED)
    pca.fit(P["X_scaled"][train_idx])

    # Load user Gaussians
    ug = json.load(open(USER_GAUSSIANS))
    users = ug["users"]

    # Load feature cache
    feat_map = {}
    with gzip.open(FEAT_CACHE, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            feat_map[rec["k"]] = rec["v"]
    log(f"  cache: {len(feat_map)} entries")

    # Load ASIN data
    asin_data = {a["asin"]: a for a in json.load(open(ASINS))["asins"]}

    # Load pools
    pools_9c = json.load(open(POOL_9C))["pools"]
    pools_9f = json.load(open(POOL_9F))["new_pool"]

    def build_combined_pool(asin):
        """Return list of dicts with 'query', 'source' (9c or 9f)."""
        out = []
        for q in pools_9c.get(asin, []):
            out.append({"query": q["query"], "source": "9c", "family": q.get("family", "?")})
        for q in pools_9f.get(asin, []):
            out.append({"query": q["query"], "source": "9f", "profile": q.get("profile", "?")})
        return out

    def project_pca48(query):
        feats = feat_map.get(feat_key(query))
        if not feats:
            return None
        vec = np.array([feats.get(n, 0.0) for n in fnames], dtype=np.float64)
        return pca.transform(scaler.transform(vec[None, :]))[0]

    def mahal_sq(z, mu, sigma_diag):
        """Mahalanobis squared distance (diagonal)."""
        diff = z - mu
        return float(np.sum(diff ** 2 / sigma_diag))

    # === Selection phase ===
    log("\n=== Selection: Mahalanobis per user on combined pool ===")
    selection_records = []
    asin_to_pool_z = {}
    for asin in pools_9c.keys():
        pool = build_combined_pool(asin)
        zs = []
        valid_pool = []
        for q in pool:
            z = project_pca48(q["query"])
            if z is not None:
                zs.append(z)
                valid_pool.append(q)
        if len(zs) == 0:
            continue
        asin_to_pool_z[asin] = (np.stack(zs), valid_pool)

    # For each (asin, user): compute Mahalanobis to all pool queries, take best
    n_processed = 0
    n_mahal_min = 0
    n_fallback = 0
    asin_centroids_data = ug.get("asin_centroids", {})
    global_mu = np.array([0.0] * PCA_DIM)  # zero mean fallback
    for asin in pools_9c.keys():
        if asin not in asin_to_pool_z:
            continue
        zs, valid_pool = asin_to_pool_z[asin]
        asin_info = asin_data.get(asin, {})
        # users_sampled is list of lists (groups of users)
        users_in_asin = []
        for grp in asin_info.get("users_sampled", []):
            if isinstance(grp, list):
                users_in_asin.extend(grp)
            else:
                users_in_asin.append(grp)

        for uid in users_in_asin:
            ug_info = users.get(uid)
            if not ug_info or ug_info.get("source") != "per_user":
                # Fall back to asin centroid or zero
                if asin in asin_centroids_data:
                    mu = np.array(asin_centroids_data[asin])
                else:
                    mu = global_mu
                sigma_diag = ug_info.get("sigma_diag") if ug_info else None
                if sigma_diag is None:
                    sigma_diag = np.ones(PCA_DIM)
                selection_method = "asin_fallback"
                n_fallback += 1
            else:
                mu = np.array(ug_info["mu"])
                sigma_diag = np.array(ug_info["sigma_diag"])
                selection_method = "mahal_min"
                n_mahal_min += 1

            distances = np.array([mahal_sq(z, mu, sigma_diag) for z in zs])
            best_idx = int(np.argmin(distances))
            selected = valid_pool[best_idx]
            selection_records.append({
                "asin": asin, "user_id": uid,
                "selection_method": selection_method,
                "selected_query": selected["query"],
                "selected_source": selected["source"],
                "selected_distance": float(distances[best_idx]),
                "random_distance": float(distances[np.random.randint(len(distances))]),
                "farthest_distance": float(distances.max()),
                "n_candidates": len(valid_pool),
            })
            n_processed += 1
    log(f"  selection_method: mahal_min={n_mahal_min}, asin_fallback={n_fallback}")

    log(f"  selection records: {n_processed}")
    with open(SELECTION_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"PCA_DIM": PCA_DIM, "LAMBDA_SHRINK": LAMBDA_SHRINK, "SEED": SEED},
            "records": selection_records,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {SELECTION_OUT}")

    # === Per-ASIN diagnostics (Stage 9E-P CHECK 2 + CHECK 3 + CHECK 4) ===
    log("\n=== Per-ASIN diagnostics (Stage 9E-P replication) ===")
    per_asin = collections.defaultdict(lambda: {
        "n_users": 0, "selected_unique": 0, "selected_source_counts": collections.Counter(),
        "top1_jaccard_sum": 0, "top5_jaccard_sum": 0, "top10_jaccard_sum": 0,
        "selected_distances": [],
        "random_distances": [],
        "selection_method_counts": collections.Counter(),
        "n_pool_9c": 0, "n_pool_9f": 0, "n_pool_combined": 0,
    })

    # Group by asin
    by_asin = collections.defaultdict(list)
    for r in selection_records:
        by_asin[r["asin"]].append(r)

    for asin, recs in by_asin.items():
        if asin not in asin_to_pool_z:
            continue
        zs, valid_pool = asin_to_pool_z[asin]
        n_pool_9c = sum(1 for q in valid_pool if q["source"] == "9c")
        n_pool_9f = sum(1 for q in valid_pool if q["source"] == "9f")
        per_asin[asin]["n_pool_9c"] = n_pool_9c
        per_asin[asin]["n_pool_9f"] = n_pool_9f
        per_asin[asin]["n_pool_combined"] = len(valid_pool)
        per_asin[asin]["n_users"] = len(recs)

        # CHECK 3: Selected uniqueness
        selected_queries = [r["selected_query"] for r in recs]
        per_asin[asin]["selected_unique"] = len(set(selected_queries))
        for r in recs:
            per_asin[asin]["selected_source_counts"][r["selected_source"]] += 1

        # Selected vs random distances
        for r in recs:
            per_asin[asin]["selected_distances"].append(r["selected_distance"])
            per_asin[asin]["random_distances"].append(r["random_distance"])
            per_asin[asin]["selection_method_counts"][r["selection_method"]] += 1

        # CHECK 2: Top-K Jaccard (between users)
        # Compute each user's top-K candidates
        ug_dict = ug
        user_topk = {}
        for r in recs:
            uid = r["user_id"]
            ug_info = ug_dict["users"].get(uid)
            if not ug_info or ug_info.get("source") != "per_user":
                if asin in asin_centroids_data:
                    mu = np.array(asin_centroids_data[asin])
                else:
                    mu = global_mu
                sigma_diag = np.ones(PCA_DIM)
            else:
                mu = np.array(ug_info["mu"])
                sigma_diag = np.array(ug_info["sigma_diag"])
            distances = np.array([mahal_sq(z, mu, sigma_diag) for z in zs])
            # Top-K by smallest distance
            for K in [1, 5, 10]:
                if K > len(distances):
                    continue
                top_idx = set(np.argsort(distances)[:K].tolist())
                user_topk.setdefault(K, {})[uid] = top_idx
        # Pairwise Jaccard
        for K in [1, 5, 10]:
            if K not in user_topk:
                continue
            topk_dict = user_topk[K]
            uids = list(topk_dict.keys())
            if len(uids) < 2:
                continue
            jaccards = []
            for i in range(len(uids)):
                for j in range(i + 1, len(uids)):
                    a = topk_dict[uids[i]]
                    b = topk_dict[uids[j]]
                    if len(a | b) == 0:
                        continue
                    j = len(a & b) / len(a | b)
                    jaccards.append(j)
            per_asin[asin][f"top{K}_jaccard_mean"] = float(np.mean(jaccards))

    # === Aggregate diagnostics ===
    log("\n=== Aggregate ===")
    diag = {}
    for asin, d in per_asin.items():
        d["selected_unique"] = d["selected_unique"]
        d["selected_source_counts"] = dict(d["selected_source_counts"])
        d["selection_method_counts"] = dict(d["selection_method_counts"])
        d["mean_selected_distance"] = float(np.mean(d["selected_distances"])) if d["selected_distances"] else None
        d["mean_random_distance"] = float(np.mean(d["random_distances"])) if d["random_distances"] else None

    # Compute summary stats
    selected_unique_per_asin = [d["selected_unique"] for d in per_asin.values()]
    top1_jaccards = [d.get("top1_jaccard_mean") for d in per_asin.values() if d.get("top1_jaccard_mean") is not None]
    top5_jaccards = [d.get("top5_jaccard_mean") for d in per_asin.values() if d.get("top5_jaccard_mean") is not None]
    top10_jaccards = [d.get("top10_jaccard_mean") for d in per_asin.values() if d.get("top10_jaccard_mean") is not None]
    selected_dists = [d["mean_selected_distance"] for d in per_asin.values() if d["mean_selected_distance"] is not None]
    random_dists = [d["mean_random_distance"] for d in per_asin.values() if d["mean_random_distance"] is not None]

    diag["per_asin"] = per_asin
    diag["summary"] = {
        "n_asins": len(per_asin),
        "selected_unique_mean": float(np.mean(selected_unique_per_asin)),
        "selected_unique_median": float(np.median(selected_unique_per_asin)),
        "selected_unique_min": int(np.min(selected_unique_per_asin)),
        "selected_unique_max": int(np.max(selected_unique_per_asin)),
        "top1_jaccard_mean": float(np.mean(top1_jaccards)) if top1_jaccards else None,
        "top5_jaccard_mean": float(np.mean(top5_jaccards)) if top5_jaccards else None,
        "top10_jaccard_mean": float(np.mean(top10_jaccards)) if top10_jaccards else None,
        "mean_selected_distance_across_asins": float(np.mean(selected_dists)) if selected_dists else None,
        "mean_random_distance_across_asins": float(np.mean(random_dists)) if random_dists else None,
    }

    log(f"  n_asins: {diag['summary']['n_asins']}")
    log(f"  selected_unique: mean={diag['summary']['selected_unique_mean']:.2f}, "
        f"median={diag['summary']['selected_unique_median']:.1f}, "
        f"min={diag['summary']['selected_unique_min']}, max={diag['summary']['selected_unique_max']}")
    log(f"  top-K Jaccard: top1={diag['summary']['top1_jaccard_mean']:.3f}, "
        f"top5={diag['summary']['top5_jaccard_mean']:.3f}, "
        f"top10={diag['summary']['top10_jaccard_mean']:.3f}")
    log(f"  selected vs random distance: "
        f"{diag['summary']['mean_selected_distance_across_asins']:.2f} vs "
        f"{diag['summary']['mean_random_distance_across_asins']:.2f}")

    # Selection method breakdown
    method_counts = collections.Counter()
    for d in per_asin.values():
        method_counts.update(d["selection_method_counts"])
    log(f"  selection method: {dict(method_counts)}")

    # Source breakdown (9C vs 9F)
    source_counts = collections.Counter()
    for d in per_asin.values():
        source_counts.update(d["selected_source_counts"])
    log(f"  selected source: {dict(source_counts)}")

    with open(DIAG_OUT, "w", encoding="utf-8") as f:
        json.dump(diag, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {DIAG_OUT}")

    # === Comparison to 9C baseline ===
    log("\n=== Comparison to 9C-only baseline ===")
    log(f"  9C baseline (Stage 9E-P CHECK 3): unique 1.60 mean (1-4 range)")
    log(f"  9C baseline (Stage 9E-P CHECK 2): top1=0.720, top10=0.796")
    log(f"  9F-B combined: unique={diag['summary']['selected_unique_mean']:.2f}, "
        f"top1={diag['summary']['top1_jaccard_mean']:.3f}, "
        f"top10={diag['summary']['top10_jaccard_mean']:.3f}")

    log("\n=== Stage 9F-B revalidation complete ===")


if __name__ == "__main__":
    main()