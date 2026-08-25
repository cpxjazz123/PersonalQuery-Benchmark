"""Stage 9G-2 — Controlled ablation: Base182 vs +Clause/Dependency vs +Connective-specific.

For each of 3 feature sets, compute:
1. **Representation quality (raw-space)**: family silhouette on 9C only (8 families)
2. **PCA dimension sweep**: silhouette at PCA dims ∈ {16, 24, 32, 48, 64, 96, 128, all}
3. **Content leakage (probe R²)**: how well do features predict brand/color/size attrs?
   - If probe R² jumps significantly with new features, content leakage is high
4. **Mahalanobis selection** on combined 9C+9F pool:
   - unique selected per ASIN
   - top-K Jaccard (1, 5, 10)
   - selected vs random distance ratio
5. **Stage 9E-P comparison**: 9C baseline (1.60 unique, 0.720 top1 Jaccard)

Output:
- /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage9g_ablation.json

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
        gaussian/syntax_subspace_stage9g_ablation.py \
        > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage9g_ablation_run.log 2>&1 &
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
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler


SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
POOL_9C = SCRATCH / "stage9c_pool_pilot.json"
POOL_9F = SCRATCH / "stage9fb_pool.json"
ASINS = SCRATCH / "stage8_5_asins.json"
USER_GAUSSIANS = SCRATCH / "stage9g_user_gaussians_3levels.json"
FEAT_ENRICHED = SCRATCH / "stage9g_enriched_features.jsonl.gz"
ABLATION_OUT = SCRATCH / "stage9g_ablation.json"

SEED = 2024
PCA_DIM_LIST = [16, 24, 32, 48, 64, 96, 128]
LAMBDA_SHRINK = 0.1


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def feat_key(t: str) -> str:
    return hashlib.sha1(t.strip().lower().encode("utf-8")).hexdigest()


def main():
    log("=== Stage 9G-2 — Controlled ablation ===")

    # Load enriched features
    log("\n=== 1. Loading enriched features ===")
    records = []
    with gzip.open(FEAT_ENRICHED, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec)
    log(f"  records: {len(records)}")

    # Filter to records with 182d features (essential)
    records_with_182d = [r for r in records if r["182d"]]
    log(f"  with 182d: {len(records_with_182d)}")

    # === Build feature matrices ===
    log("\n=== 2. Building feature matrices ===")

    # Get ordered feature names from first record
    all_182_names = sorted(records_with_182d[0]["182d"].keys())
    # Filter to numeric-only (182d includes string POS tags like 'NOUN')
    base_182_names = []
    for n in all_182_names:
        v = records_with_182d[0]["182d"][n]
        if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '').replace('-', '').isdigit()):
            base_182_names.append(n)
    clause_names = sorted(records_with_182d[0]["clause"].keys())
    conn_names = sorted(records_with_182d[0]["connective"].keys())
    log(f"  Base182 numeric: {len(base_182_names)} (filtered from {len(all_182_names)})")
    log(f"  +Clause/Dep: {len(clause_names)}")
    log(f"  +Connective: {len(conn_names)}")
    log(f"  Total enriched: {len(base_182_names) + len(clause_names) + len(conn_names)}")

    # Build X for each level
    def build_X(records, level):
        X = []
        for r in records:
            if level == "base":
                vec = [float(r["182d"].get(n, 0.0) or 0.0) for n in base_182_names]
            elif level == "clause":
                vec = [float(r["182d"].get(n, 0.0) or 0.0) for n in base_182_names]
                vec += [float(r["clause"].get(n, 0.0) or 0.0) for n in clause_names]
            elif level == "conn":
                vec = [float(r["182d"].get(n, 0.0) or 0.0) for n in base_182_names]
                vec += [float(r["clause"].get(n, 0.0) or 0.0) for n in clause_names]
                vec += [float(r["connective"].get(n, 0.0) or 0.0) for n in conn_names]
            X.append(vec)
        return np.array(X, dtype=np.float64)

    X_base = build_X(records_with_182d, "base")
    X_clause = build_X(records_with_182d, "clause")
    X_conn = build_X(records_with_182d, "conn")

    log(f"  X_base: {X_base.shape}")
    log(f"  X_clause: {X_clause.shape}")
    log(f"  X_conn: {X_conn.shape}")

    # Get family labels (only 9C queries have family)
    family_labels = []
    family_idx = []
    for i, r in enumerate(records_with_182d):
        if r["source"] == "9c" and "family" in r:
            family_labels.append(r["family"])
            family_idx.append(i)
    family_labels = np.array(family_labels)
    log(f"  9C records with family: {len(family_idx)}, families: {len(set(family_labels))}")

    # === Standardize each level (fit on full data) ===
    scaler_base = StandardScaler().fit(X_base)
    scaler_clause = StandardScaler().fit(X_clause)
    scaler_conn = StandardScaler().fit(X_conn)

    X_base_s = scaler_base.transform(X_base)
    X_clause_s = scaler_clause.transform(X_clause)
    X_conn_s = scaler_conn.transform(X_conn)

    # === 3. Raw-space family silhouette (Stage 9F-A replication) ===
    log("\n=== 3. Raw-space family silhouette (9C only) ===")
    base_9c = X_base_s[family_idx]
    clause_9c = X_clause_s[family_idx]
    conn_9c = X_conn_s[family_idx]

    sil_base_raw = float(silhouette_score(base_9c, family_labels, metric="euclidean")) if len(set(family_labels)) >= 2 else None
    sil_clause_raw = float(silhouette_score(clause_9c, family_labels, metric="euclidean"))
    sil_conn_raw = float(silhouette_score(conn_9c, family_labels, metric="euclidean"))
    log(f"  silhouette (raw, family labels):")
    log(f"    Base182:        {sil_base_raw:.4f}")
    log(f"    +Clause/Dep:    {sil_clause_raw:.4f}  Δ={sil_clause_raw - sil_base_raw:+.4f}")
    log(f"    +Connective:    {sil_conn_raw:.4f}  Δ={sil_conn_raw - sil_base_raw:+.4f}")

    # === 4. PCA dimension sweep ===
    log("\n=== 4. PCA dimension sweep (silhouette) ===")
    pca_sweep = {"base": {}, "clause": {}, "conn": {}}

    for level_name, X_full in [("base", X_base_s), ("clause", X_clause_s), ("conn", X_conn_s)]:
        log(f"\n  === Level: {level_name} ===")
        for n_dim in PCA_DIM_LIST:
            if n_dim > X_full.shape[1]:
                continue
            pca = PCA(n_components=n_dim, random_state=SEED)
            X_pca = pca.fit_transform(X_full)
            X_9c_pca = X_pca[family_idx]
            try:
                sil = float(silhouette_score(X_9c_pca, family_labels, metric="euclidean"))
            except Exception as e:
                sil = None
            pca_sweep[level_name][n_dim] = sil
            log(f"    PCA{n_dim}: silhouette = {sil:.4f}" if sil else f"    PCA{n_dim}: silhouette = None")
        # Also raw (no PCA)
        try:
            sil = float(silhouette_score(X_full[family_idx], family_labels, metric="euclidean"))
            pca_sweep[level_name]["raw"] = sil
            log(f"    raw: silhouette = {sil:.4f}")
        except Exception as e:
            pca_sweep[level_name]["raw"] = None

    # === 5. Content leakage (probe R²) ===
    # For each ASIN, predict 4 attrs (Brand/Color/Size/Age Range) from features
    log("\n=== 5. Content leakage probe (R² of features → attrs) ===")

    # Load ASIN data
    asin_data_full = json.load(open(ASINS))["asins"]
    asin_attrs = {}
    for a in asin_data_full:
        asin_attrs[a["asin"]] = a.get("attrs_used", {})

    # Build (query, brand, color, size, age) tuples
    leak_data = []
    for r in records_with_182d:
        attrs = asin_attrs.get(r["asin"], {})
        if not attrs:
            continue
        leak_data.append({
            "idx": records_with_182d.index(r),
            "brand": attrs.get("Brand", ""),
            "color": attrs.get("Color", ""),
            "age_range": attrs.get("Age Range (Description)", ""),
            "size": attrs.get("Size", ""),
            "main_cat": attrs.get("Main Category", ""),
        })
    log(f"  leak_data: {len(leak_data)}")

    # For each attr, compute probe R² per feature level
    attr_targets = ["brand", "color", "age_range", "size", "main_cat"]
    leak_results = {}
    for level_name, X_full in [("base", X_base_s), ("clause", X_clause_s), ("conn", X_conn_s)]:
        leak_results[level_name] = {}
        for attr in attr_targets:
            # Build y: one-hot encode (label encode) and run ridge regression
            attr_values = [d[attr] for d in leak_data if d[attr]]
            indices = [i for i, d in enumerate(leak_data) if d[attr]]
            if len(set(attr_values)) < 2:
                leak_results[level_name][attr] = {"n": 0, "r2_mean": 0.0}
                continue
            X_attr = X_full[indices]
            # Label encode
            label_map = {v: i for i, v in enumerate(sorted(set(attr_values)))}
            y = np.array([label_map[d[attr]] for d in leak_data if d[attr]])

            # Cross-validated Ridge R²
            try:
                from sklearn.linear_model import RidgeClassifier
                from sklearn.preprocessing import LabelBinarizer
                # One-vs-rest for multi-class
                lb = LabelBinarizer()
                y_oh = lb.fit_transform(y)
                if y_oh.shape[1] == 1:
                    y_oh = np.hstack([1 - y_oh, y_oh])
                # Ridge regression on each one-vs-rest column
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
                r2s = []
                for col in range(y_oh.shape[1]):
                    y_col = y_oh[:, col]
                    preds = cross_val_predict(Ridge(alpha=1.0), X_attr, y_col, cv=kf, n_jobs=1)
                    ss_res = np.sum((y_col - preds) ** 2)
                    ss_tot = np.sum((y_col - y_col.mean()) ** 2)
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
                    r2s.append(r2)
                leak_results[level_name][attr] = {
                    "n": len(y),
                    "r2_mean": float(np.mean(r2s)),
                    "r2_per_class": [float(r) for r in r2s],
                }
            except Exception as e:
                leak_results[level_name][attr] = {"n": len(y), "r2_mean": 0.0, "error": str(e)}
        log(f"  {level_name}:")
        for attr, info in leak_results[level_name].items():
            log(f"    {attr}: R² = {info['r2_mean']:.4f} (n={info['n']})")

    # === 6. Mahalanobis selection on combined 9C+9F pool ===
    log("\n=== 6. Mahalanobis selection on combined pool ===")
    ug_all = json.load(open(USER_GAUSSIANS))
    users_by_level = ug_all["users"]  # {"base": {uid: {...}}, "clause": {...}, "conn": {...}}
    asin_info = {a["asin"]: a for a in asin_data_full}

    # Get record index by asin and source
    records_by_asin = collections.defaultdict(list)
    for i, r in enumerate(records_with_182d):
        records_by_asin[r["asin"]].append({"idx": i, "r": r})

    # For each level: project pool via PCA(48) + run selection
    selection_results = {"base": {}, "clause": {}, "conn": {}}

    # Use fixed PCA48 for all levels (so the comparison is fair)
    PCA_DIM = 48

    for level_name, X_full in [("base", X_base_s), ("clause", X_clause_s), ("conn", X_conn_s)]:
        log(f"\n  === Level: {level_name} ===")
        if PCA_DIM > X_full.shape[1]:
            continue
        pca = PCA(n_components=PCA_DIM, random_state=SEED)
        Z = pca.fit_transform(X_full)
        log(f"    Z shape: {Z.shape}")

        # Use level-matched user Gaussians
        users = users_by_level.get(level_name, {})

        # Per ASIN: compute Mahalanobis selection per user
        per_asin_stats = []
        for asin, recs in records_by_asin.items():
            pool_idx = [x["idx"] for x in recs]
            Z_pool = Z[pool_idx]

            # User-level info
            asin_users = []
            for grp in asin_info.get(asin, {}).get("users_sampled", []):
                if isinstance(grp, list):
                    asin_users.extend(grp)
                else:
                    asin_users.append(grp)

            n_mahal_min = 0
            selected_distances = []
            random_distances = []
            selected_queries = []

            for uid in asin_users:
                ug_info = users.get(uid)
                if not ug_info or ug_info.get("source") not in ("per_user", "marginal"):
                    mu = np.zeros(PCA_DIM)
                    sigma_diag = np.ones(PCA_DIM)
                else:
                    mu = np.array(ug_info["mu"])
                    sigma_diag = np.array(ug_info["sigma_diag"])
                    n_mahal_min += 1

                diffs = Z_pool - mu
                distances = np.sum(diffs ** 2 / sigma_diag, axis=1)
                best_idx = int(np.argmin(distances))
                selected_queries.append(recs[best_idx]["r"]["query"])
                selected_distances.append(float(distances[best_idx]))
                random_distances.append(float(distances[np.random.randint(len(distances))]))

            per_asin_stats.append({
                "asin": asin,
                "n_users": len(asin_users),
                "n_mahal_min": n_mahal_min,
                "n_pool": len(recs),
                "selected_unique": len(set(selected_queries)),
                "mean_selected_dist": float(np.mean(selected_distances)) if selected_distances else 0,
                "mean_random_dist": float(np.mean(random_distances)) if random_distances else 0,
                "selected_queries": selected_queries,
            })

        # Aggregate
        unique_sel = [s["selected_unique"] for s in per_asin_stats]
        sel_dist_mean = np.mean([s["mean_selected_dist"] for s in per_asin_stats])
        rnd_dist_mean = np.mean([s["mean_random_dist"] for s in per_asin_stats])

        # Top-K Jaccard
        jaccard_means = {1: [], 5: [], 10: []}
        for stats in per_asin_stats:
            asin = stats["asin"]
            recs = records_by_asin[asin]
            pool_idx = [x["idx"] for x in recs]
            Z_pool = Z[pool_idx]
            asin_users = []
            for grp in asin_info.get(asin, {}).get("users_sampled", []):
                if isinstance(grp, list):
                    asin_users.extend(grp)
                else:
                    asin_users.append(grp)

            user_topk = {1: {}, 5: {}, 10: {}}
            for uid in asin_users:
                ug_info = users.get(uid)
                if not ug_info or ug_info.get("source") not in ("per_user", "marginal"):
                    continue
                mu = np.array(ug_info["mu"])
                sigma_diag = np.array(ug_info["sigma_diag"])
                diffs = Z_pool - mu
                distances = np.sum(diffs ** 2 / sigma_diag, axis=1)
                sorted_idx = np.argsort(distances)
                for K in [1, 5, 10]:
                    if K <= len(sorted_idx):
                        user_topk[K][uid] = set(sorted_idx[:K].tolist())
            for K in [1, 5, 10]:
                uids = list(user_topk[K].keys())
                if len(uids) < 2:
                    continue
                jaccards = []
                for i in range(len(uids)):
                    for j in range(i + 1, len(uids)):
                        a, b = user_topk[K][uids[i]], user_topk[K][uids[j]]
                        if len(a | b) > 0:
                            jaccards.append(len(a & b) / len(a | b))
                if jaccards:
                    jaccard_means[K].append(float(np.mean(jaccards)))

        selection_results[level_name] = {
            "n_asins": len(per_asin_stats),
            "unique_sel_mean": float(np.mean(unique_sel)),
            "unique_sel_median": float(np.median(unique_sel)),
            "unique_sel_min": int(np.min(unique_sel)),
            "unique_sel_max": int(np.max(unique_sel)),
            "mean_selected_dist_across_asins": float(sel_dist_mean),
            "mean_random_dist_across_asins": float(rnd_dist_mean),
            "selected_random_ratio": float(rnd_dist_mean / sel_dist_mean) if sel_dist_mean > 0 else 0,
            "top1_jaccard_mean": float(np.mean(jaccard_means[1])) if jaccard_means[1] else None,
            "top5_jaccard_mean": float(np.mean(jaccard_means[5])) if jaccard_means[5] else None,
            "top10_jaccard_mean": float(np.mean(jaccard_means[10])) if jaccard_means[10] else None,
            "per_asin_stats": per_asin_stats,
        }
        log(f"    unique_selected: mean={selection_results[level_name]['unique_sel_mean']:.2f}, "
            f"median={selection_results[level_name]['unique_sel_median']:.1f}, "
            f"min={selection_results[level_name]['unique_sel_min']}, "
            f"max={selection_results[level_name]['unique_sel_max']}")
        log(f"    top-K Jaccard: top1={selection_results[level_name]['top1_jaccard_mean']:.3f}, "
            f"top5={selection_results[level_name]['top5_jaccard_mean']:.3f}, "
            f"top10={selection_results[level_name]['top10_jaccard_mean']:.3f}")
        log(f"    selected vs random distance: {sel_dist_mean:.2f} vs {rnd_dist_mean:.2f} "
            f"(ratio {selection_results[level_name]['selected_random_ratio']:.2f}×)")

    # === Save results ===
    log("\n=== 7. Save results ===")
    ablation = {
        "config": {
            "SEED": SEED,
            "PCA_DIM": PCA_DIM,
            "LAMBDA_SHRINK": LAMBDA_SHRINK,
            "n_records": len(records_with_182d),
            "n_9c_with_family": len(family_idx),
            "n_families": len(set(family_labels)),
            "Base182_features": len(base_182_names),
            "Clause_Dep_features": len(clause_names),
            "Connective_features": len(conn_names),
        },
        "raw_silhouette_family": {
            "base": sil_base_raw,
            "clause": sil_clause_raw,
            "conn": sil_conn_raw,
        },
        "pca_silhouette_family": pca_sweep,
        "content_leakage_probe_r2": leak_results,
        "mahalanobis_selection": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_asin_stats"}
            for k, v in selection_results.items()
        },
        "comparison_vs_baseline": {
            "9C_baseline": {
                "unique_sel_mean": 1.60,
                "top1_jaccard": 0.720,
                "top10_jaccard": 0.796,
                "source": "Stage 9E-P",
            },
            "9F_B_baseline": {
                "unique_sel_mean": 2.60,
                "top1_jaccard": 0.629,
                "top10_jaccard": 0.754,
                "selected_random_ratio": 9.85,
                "source": "Stage 9F-B",
            },
        },
    }
    with open(ABLATION_OUT, "w", encoding="utf-8") as f:
        json.dump(ablation, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {ABLATION_OUT}")

    # === Summary table ===
    log("\n=== ABLATION SUMMARY ===")
    log(f"  {'Level':<15} {'Sil(raw)':<10} {'Sil(PCA48)':<12} {'Unique':<8} {'top1':<8} {'top10':<8} {'sel/rnd':<10}")
    log(f"  {'-'*75}")
    for level_name in ["base", "clause", "conn"]:
        sil_raw = ablation["raw_silhouette_family"][level_name]
        sil_pca48 = pca_sweep[level_name].get(48)
        unique = ablation["mahalanobis_selection"][level_name]["unique_sel_mean"]
        t1 = ablation["mahalanobis_selection"][level_name]["top1_jaccard_mean"]
        t10 = ablation["mahalanobis_selection"][level_name]["top10_jaccard_mean"]
        sr = ablation["mahalanobis_selection"][level_name]["selected_random_ratio"]
        log(f"  {level_name:<15} {sil_raw:<10.4f} {sil_pca48:<12.4f} {unique:<8.2f} {t1:<8.3f} {t10:<8.3f} {sr:<10.2f}")

    log("\n=== Content Leakage (probe R²): base vs enriched ===")
    log(f"  {'attr':<15} {'base':<10} {'clause':<10} {'conn':<10} {'Δ clause':<10} {'Δ conn':<10}")
    log(f"  {'-'*65}")
    for attr in attr_targets:
        r2_base = ablation["content_leakage_probe_r2"]["base"][attr]["r2_mean"]
        r2_clause = ablation["content_leakage_probe_r2"]["clause"][attr]["r2_mean"]
        r2_conn = ablation["content_leakage_probe_r2"]["conn"][attr]["r2_mean"]
        log(f"  {attr:<15} {r2_base:<10.4f} {r2_clause:<10.4f} {r2_conn:<10.4f} "
            f"{r2_clause - r2_base:<+10.4f} {r2_conn - r2_base:<+10.4f}")

    log("\n=== Stage 9G-2 ablation complete ===")


if __name__ == "__main__":
    main()