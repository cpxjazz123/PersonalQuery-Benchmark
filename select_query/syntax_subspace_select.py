"""Syntax Subspace — Stage 4 (Mahalanobis select) + Stage 7 (A1 reject-repeat).

归 select_query/: 基于 per-user Gaussian 从共享候选池选最佳查询。

Stage 4: 对每 (asin, user) 选 Mahalanobis 最小候选 (selected / random / farthest)
Stage 7: 对每 (asin, users) 跑多种 A1 策略(mean_t50 / A1_reject_repeat /
         C3_kmeans_k{2,3,4,5}),选出 unique 最多的策略。

用法:
  python select_query/syntax_subspace_select.py --stage select
  python select_query/syntax_subspace_select.py --stage a1_select

I/O 路径:
  输入: stage8_5_asins.json / stage8_5_pool.json / stage8_5_user_gaussians.json
        / stage8_5_selection.json (Stage 7)
  输出: stage8_5_selection.json + stage8_5_selection_stats.json  (Stage 4)
        stage10k_selection.json + stage10k_summary.json          (Stage 7)

共享工具 (log, feat_key, paths, hyperparams, _syntax_subspace_prepare) 来自:
  gaussian/syntax_subspace_utils.py
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gaussian"))
from syntax_subspace_utils import (  # noqa: E402
    ASINS_IN, FEAT_CACHE, GAUSSIANS_IN, POOL_IN, SELECTION_IN, SELECTION_OUT,
    SELECTION_STATS_OUT, A1_SELECTION_OUT, A1_SUMMARY_OUT,
    PCA_DIM, PCA_SEED, SEED, THRESHOLD_PCT, log, feat_key,
)


# ===========================================================================
# STAGE 4 — MAHA SELECT
# ===========================================================================

def mahalanobis_sq(z: np.ndarray, mu: np.ndarray, sigma_diag: np.ndarray) -> float:
    diff = z - mu
    return float((diff * diff / sigma_diag).sum())


def stage_select():
    log("=== STAGE 4 — MAHA SELECT ===")

    log("\n=== 1. Loading PCA48 ===")
    from syntax_subspace_utils import _syntax_subspace_prepare
    P = _syntax_subspace_prepare()
    scaler = P["scaler"]
    fnames = P["feature_names_ordered"]
    train_idx = P["train_idx"]

    from sklearn.decomposition import PCA
    pca = PCA(n_components=PCA_DIM, random_state=42)
    pca.fit(P["X_scaled"][train_idx])
    log(f"  PCA48 ready")

    log("\n=== 2. Loading inputs ===")
    asin_data = json.load(open(ASINS_IN))["asins"]
    pool_data = json.load(open(POOL_IN))
    pools = pool_data["pools"]
    gauss_data = json.load(open(GAUSSIANS_IN))
    users_gauss = gauss_data["users"]
    global_var = np.array(gauss_data["global_var"])
    log(f"  ASINs: {len(asin_data)}, pools: {len(pools)}")
    log(f"  user Gaussians: {len(users_gauss)}")

    feat_map = {}
    with gzip.open(FEAT_CACHE, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            feat_map[rec["k"]] = rec["v"]
    log(f"  features: {len(feat_map)}")

    log("\n=== 3. Projecting pool queries ===")
    pool_z = {}
    miss = 0
    for asin, qs in pools.items():
        zs_for_asin = []
        for q in qs:
            k = feat_key(q["query"])
            feats = feat_map.get(k)
            if not feats:
                miss += 1
                continue
            vec = np.array([feats.get(n, 0.0) for n in fnames], dtype=np.float64)
            z = pca.transform(scaler.transform(vec[None, :]))[0]
            zs_for_asin.append((z, q))
        pool_z[asin] = zs_for_asin
    log(f"  ASINs with pool_z: {len(pool_z)}, missing features: {miss}")

    log("\n=== 4. ASIN centroid fallback ===")
    asin_centroid = {}
    for asin, zqs in pool_z.items():
        zs = np.stack([z for z, _ in zqs], axis=0)
        asin_centroid[asin] = zs.mean(axis=0)
    log(f"  ASIN centroids: {len(asin_centroid)}")

    log("\n=== 5. Selection per (asin, user) ===")
    rng = random.Random(SEED)

    selection_entries = []
    n_mahal = 0
    n_asin_fallback = 0
    n_random_fallback = 0

    for entry in asin_data:
        asin = entry["asin"]
        attrs = entry["attrs_used"]
        n_input = len(attrs)
        zqs = pool_z.get(asin, [])
        n_pool_strict = sum(1 for z, q in zqs if q["strict"])

        c_asin = asin_centroid.get(asin)
        if c_asin is None:
            log(f"  WARNING: no centroid for {asin}, skipping users")
            continue

        for uid in entry["users_sampled"]:
            if uid in users_gauss:
                mu = np.array(users_gauss[uid]["mu"])
                sigma = np.array(users_gauss[uid]["sigma_diag"])
                source = users_gauss[uid]["source"]
                n_reviews = users_gauss[uid]["n_reviews"]
            else:
                mu = c_asin
                sigma = np.maximum(global_var, 1e-3)
                source = "asin_centroid_fallback"
                n_reviews = 0

            strict_zqs = [(z, q) for z, q in zqs if q["strict"]]
            if not strict_zqs:
                strict_zqs = zqs
            if not strict_zqs:
                selection_entries.append({
                    "asin": asin,
                    "user_id": uid,
                    "attrs_used": attrs,
                    "selection_method": "no_pool",
                    "selected": None,
                    "random": None,
                    "farthest": None,
                    "selected_distance": None,
                    "random_distance": None,
                    "farthest_distance": None,
                    "n_candidates": 0,
                    "user_source": source,
                    "n_reviews": n_reviews,
                })
                continue

            distances = np.array([mahalanobis_sq(z, mu, sigma) for z, _ in strict_zqs])
            best_idx = int(np.argmin(distances))
            worst_idx = int(np.argmax(distances))

            selected_q = strict_zqs[best_idx][1]
            farthest_q = strict_zqs[worst_idx][1]

            rng_u = random.Random(hash(uid) & 0xffffffff)
            random_idx = rng_u.randint(0, len(strict_zqs) - 1)
            random_q = strict_zqs[random_idx][1]

            if source == "per_user" or source == "global_var_fallback":
                method = "mahal_min"
                n_mahal += 1
            else:
                method = "asin_centroid_fallback"
                n_asin_fallback += 1

            selection_entries.append({
                "asin": asin,
                "user_id": uid,
                "attrs_used": attrs,
                "selection_method": method,
                "selected": selected_q,
                "random": random_q,
                "farthest": farthest_q,
                "selected_distance": float(distances[best_idx]),
                "random_distance": float(distances[random_idx]),
                "farthest_distance": float(distances[worst_idx]),
                "n_candidates": len(strict_zqs),
                "user_source": source,
                "n_reviews": n_reviews,
            })

    log(f"  total entries: {len(selection_entries)}")
    log(f"    mahal_min: {n_mahal}")
    log(f"    asin_centroid_fallback: {n_asin_fallback}")

    SELECTION_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(SELECTION_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Stage 8.5: Mahalanobis selection from shared pool",
                "SEED": SEED,
            },
            "n_entries": len(selection_entries),
            "n_mahal": n_mahal,
            "n_asin_fallback": n_asin_fallback,
            "n_random_fallback": n_random_fallback,
            "entries": selection_entries,
        }, f, ensure_ascii=False, indent=2)
    log(f"wrote → {SELECTION_OUT}")

    log("\n=== 7. Validation ===")
    from scipy.stats import wilcoxon
    mahal_entries = [e for e in selection_entries if e["selected_distance"] is not None]
    selected = np.array([e["selected_distance"] for e in mahal_entries])
    random_d = np.array([e["random_distance"] for e in mahal_entries])
    farthest = np.array([e["farthest_distance"] for e in mahal_entries])

    log(f"  N pairs: {len(mahal_entries)}")
    log(f"  selected: mean={selected.mean():.3f}, std={selected.std():.3f}, median={np.median(selected):.3f}")
    log(f"  random:   mean={random_d.mean():.3f}, std={random_d.std():.3f}, median={np.median(random_d):.3f}")
    log(f"  farthest: mean={farthest.mean():.3f}, std={farthest.std():.3f}, median={np.median(farthest):.3f}")

    w_sr, p_sr = wilcoxon(selected, random_d, alternative="less")
    log(f"  Wilcoxon selected < random: W={w_sr:.1f}, p={p_sr:.4g}")
    w_sf, p_sf = wilcoxon(selected, farthest, alternative="less")
    log(f"  Wilcoxon selected < farthest: W={w_sf:.1f}, p={p_sf:.4g}")
    w_rf, p_rf = wilcoxon(random_d, farthest, alternative="less")
    log(f"  Wilcoxon random < farthest: W={w_rf:.1f}, p={p_rf:.4g}")

    by_source = collections.defaultdict(lambda: {"selected": [], "random": [], "farthest": []})
    for e in mahal_entries:
        s = e["user_source"]
        by_source[s]["selected"].append(e["selected_distance"])
        by_source[s]["random"].append(e["random_distance"])
        by_source[s]["farthest"].append(e["farthest_distance"])

    log(f"\n=== Per-source breakdown ===")
    for src, d in by_source.items():
        sel = np.array(d["selected"])
        rnd = np.array(d["random"])
        log(f"  {src} (n={len(d['selected'])}):")
        log(f"    selected mean={sel.mean():.3f}, random mean={rnd.mean():.3f}, "
            f"diff={(sel - rnd).mean():+.3f}, %sel<rnd={100 * (sel < rnd).mean():.1f}%")

    with open(SELECTION_STATS_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "n_pairs": len(mahal_entries),
            "selected_mean": float(selected.mean()),
            "selected_std": float(selected.std()),
            "random_mean": float(random_d.mean()),
            "random_std": float(random_d.std()),
            "farthest_mean": float(farthest.mean()),
            "farthest_std": float(farthest.std()),
            "wilcoxon_selected_vs_random": {"W": float(w_sr), "p_value": float(p_sr), "alternative": "less"},
            "wilcoxon_selected_vs_farthest": {"W": float(w_sf), "p_value": float(p_sf), "alternative": "less"},
            "wilcoxon_random_vs_farthest": {"W": float(w_rf), "p_value": float(p_rf), "alternative": "less"},
            "per_source": {
                src: {
                    "n": len(d["selected"]),
                    "selected_mean": float(np.mean(d["selected"])),
                    "random_mean": float(np.mean(d["random"])),
                    "farthest_mean": float(np.mean(d["farthest"])),
                    "pct_selected_lt_random": float(100 * (np.array(d["selected"]) < np.array(d["random"])).mean()),
                }
                for src, d in by_source.items()
            },
        }, f, ensure_ascii=False, indent=2)
    log(f"wrote → {SELECTION_STATS_OUT}")


# ===========================================================================
# STAGE 7 — A1 SELECTION
# ===========================================================================

def maha_diag_batch(z_arr: np.ndarray, mu_arr: np.ndarray, sigma_arr: np.ndarray) -> np.ndarray:
    diff = z_arr[:, None, :] - mu_arr[None, :, :]
    return np.sum(diff ** 2 / sigma_arr[None, :, :], axis=2)


def stage_a1_select():
    log("=== STAGE 7 — A1 SELECTION ===")

    log("\n=== 1. Loading user Gaussians ===")
    g_data = json.load(open(GAUSSIANS_IN))
    base_users = g_data["users"]
    log(f"  base users: {len(base_users)}")
    user_mu = {}
    user_sigma = {}
    for uid, u in base_users.items():
        user_mu[uid] = np.array(u["mu"], dtype=np.float64)
        user_sigma[uid] = np.array(u["sigma_diag"], dtype=np.float64)

    log("\n=== 2. Loading scaler + PCA48 ===")
    from syntax_subspace_utils import _syntax_subspace_prepare
    P = _syntax_subspace_prepare()
    scaler = P["scaler"]
    feature_names = P["feature_names_ordered"]

    feat_cache_path = FEAT_CACHE
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
    from sklearn.decomposition import PCA
    pca = PCA(n_components=PCA_DIM, random_state=PCA_SEED)
    pca.fit(P["X_scaled"][train_idx])
    log(f"  PCA48 EV={pca.explained_variance_ratio_.sum():.4f}")

    log("\n=== 3. Projecting pool queries ===")
    pool_data = json.load(open(POOL_IN))
    pool_by_asin = pool_data["pools"]
    log(f"  n asins: {len(pool_by_asin)}")

    pool_z_by_asin = {}
    n_projected = 0
    n_skipped = 0
    for asin, queries in pool_by_asin.items():
        zs = []
        for q in queries:
            text = q["query"]
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

    log("\n=== 4. Loading (asin, user) pairs ===")
    sel_data = json.load(open(SELECTION_IN))
    asin_users = collections.defaultdict(list)
    for e in sel_data["entries"]:
        asin = e["asin"]
        uid = e["user_id"]
        if uid not in user_mu:
            continue
        if uid not in asin_users[asin]:
            asin_users[asin].append(uid)
    log(f"  n asins with users: {len(asin_users)}")

    log("\n=== 5. Computing margins + running strategies ===")

    from sklearn.cluster import KMeans

    def run_strategy(z_arr: np.ndarray, mu_arr: np.ndarray, sigma_arr: np.ndarray,
                     valid_users: list, strategy: str):
        n_q = z_arr.shape[0]
        n_u = len(valid_users)
        D = maha_diag_batch(z_arr, mu_arr, sigma_arr)
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
            selected = []
            picked = set()
            for ui in range(n_u):
                d_self = d_selfs[ui]
                margin = margins[ui]
                threshold = float(np.percentile(d_self, THRESHOLD_PCT))
                valid = d_self <= threshold
                m = np.where(valid, margin, -np.inf)
                ranked = np.argsort(-m)
                chosen = None
                for c in ranked:
                    if c not in picked:
                        chosen = int(c)
                        break
                if chosen is None:
                    chosen = int(ranked[0])
                selected.append(chosen)
                picked.add(chosen)
            return selected, margins, d_selfs

        elif strategy.startswith("C3_kmeans_k"):
            k = int(strategy.rsplit("_k", 1)[1])
            user_clusters = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(mu_arr)
            selected = [0] * n_u
            used_global = set()
            for c in range(k):
                cluster_mask = (user_clusters == c)
                cluster_user_idxs = np.where(cluster_mask)[0]
                if len(cluster_user_idxs) == 0:
                    continue
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
    rnd_dist_per_asin = {}

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

        D = maha_diag_batch(z_arr, mu_arr, sigma_arr)
        rnd_per_user = [float(np.mean(D[:, ui])) for ui in range(len(valid_users))]
        rnd_dist_per_asin[asin] = np.mean(rnd_per_user)

        asin_strategies = {}
        for strategy in strategies:
            sel_idx, margins, d_selfs = run_strategy(z_arr, mu_arr, sigma_arr,
                                                     valid_users, strategy)
            sel_d_self = [float(d_selfs[ui, sel_idx[ui]]) for ui in range(len(valid_users))]
            asin_strategies[strategy] = {
                "selected_idx": [int(i) for i in sel_idx],
                "selected_d_self": sel_d_self,
                "n_unique": len(set(sel_idx)),
            }
        per_asin_results[asin] = asin_strategies

    log("\n=== 6. Aggregating metrics ===")
    summary = {}
    for strategy in strategies:
        uniqs = []
        sel_dists = []
        for asin, res in per_asin_results.items():
            uniqs.append(res[strategy]["n_unique"])
            sel_dists.extend(res[strategy]["selected_d_self"])
        rnd_mean = np.mean(list(rnd_dist_per_asin.values())) if rnd_dist_per_asin else 0.0
        summary[strategy] = {
            "n_asins": len(per_asin_results),
            "unique_mean": float(np.mean(uniqs)) if uniqs else 0.0,
            "unique_median": float(np.median(uniqs)) if uniqs else 0.0,
            "unique_min": int(min(uniqs)) if uniqs else 0,
            "unique_max": int(max(uniqs)) if uniqs else 0,
            "sel_dist_mean": float(np.mean(sel_dists)) if sel_dists else 0.0,
            "rnd_dist_mean": float(rnd_mean),
            "sel_rnd_ratio": float(np.mean(sel_dists) / rnd_mean) if rnd_mean > 0 else None,
        }

    log(f"\n{'strategy':<25} {'unique_mean':>12} {'unique_med':>11} {'sel_dist':>10} {'rnd_dist':>10} {'ratio':>7}")
    for strategy in strategies:
        s = summary[strategy]
        log(f"  {strategy:<23} {s['unique_mean']:>11.2f} {s['unique_median']:>11.1f} "
            f"{s['sel_dist_mean']:>9.1f} {s['rnd_dist_mean']:>9.1f} {s['sel_rnd_ratio']:>6.2f}")

    with open(A1_SELECTION_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Stage 10K selection strategies",
                "PCA_DIM": PCA_DIM,
                "THRESHOLD_PCT": THRESHOLD_PCT,
                "strategies": strategies,
            },
            "per_asin": per_asin_results,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {A1_SELECTION_OUT}")

    with open(A1_SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {A1_SUMMARY_OUT}")

    log("\n=== Stage 10K complete ===")


# ===========================================================================
# MAIN
# ===========================================================================

STAGE_FUNCTIONS = {
    "select": stage_select,
    "a1_select": stage_a1_select,
}


def main():
    parser = argparse.ArgumentParser(description="Syntax Subspace — select_query (Mahalanobis + A1)")
    parser.add_argument(
        "--stage",
        required=True,
        choices=list(STAGE_FUNCTIONS.keys()) + ["all"],
        help="Which stage to run",
    )
    args = parser.parse_args()

    log(f"=== syntax_subspace_select.py — stage={args.stage} ===")

    if args.stage == "all":
        for stage_name in STAGE_FUNCTIONS:
            log(f"\n>>> Running stage: {stage_name}")
            STAGE_FUNCTIONS[stage_name]()
    else:
        STAGE_FUNCTIONS[args.stage]()


if __name__ == "__main__":
    main()