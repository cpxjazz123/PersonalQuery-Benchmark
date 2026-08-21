#!/usr/bin/env python3
"""Phase 14.Q9-2 LOPO: Combine Q_NarrativeFirstPerson + cached + Q-8.

Test if first-person narrative generation breaks the 18/30 ceiling.
"""
from __future__ import annotations
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

IN_PAIRS_14B = OUT_DIR / "phase14_b_layer_alpha_sweep.jsonl"
IN_PAIRS_Q8 = OUT_DIR / "phase14_q8_generated.jsonl"
IN_PAIRS_Q9 = OUT_DIR / "phase14_q9_generated.jsonl"
IN_USER_HIDDENS = OUT_DIR / "phase14_b_user_hiddens_5layers.npz"
IN_NEUTRAL_HIDDENS = OUT_DIR / "phase13_a_neutral_hiddens.npz"
IN_CAND_RESID_11 = OUT_DIR / "phase14_p_cand_residuals_qwen.npy"
IN_CAND_RESID_Q8 = OUT_DIR / "phase14_q8_cand_residuals_qwen.npy"
IN_CAND_RESID_Q9 = OUT_DIR / "phase14_q9_cand_residuals_qwen.npy"

OUT_EVAL = OUT_DIR / "phase14_q9_lopo_eval.json"
OUT_PER_PAIR = OUT_DIR / "phase14_q9_lopo_per_pair.jsonl"

LAYERS_5 = [8, 14, 18, 22, 26]
LAYER_FOCUS = 26
N_USERS = 198
HIDDEN = 3584

PCA_DIMS_ENSEMBLE = [300, 500]
ALPHA = 0.25
MAX_PCA = max(PCA_DIMS_ENSEMBLE)
LW_SHRINK_USER = 0.1
LW_SHRINK_POOLED = 0.1
MIN_VAR = 1e-4
SEED = 42
N_BOOTSTRAP = 2000

CACHED_CONDITIONS = ["D_off", "A8_a0.5", "A8_a1.0", "A14_a0.5", "A14_a1.0",
                     "A18_a0.5", "A18_a1.0", "A22_a0.5", "A22_a1.0", "A26_a0.5", "A26_a1.0"]
Q8_CONDITIONS = [f"Q_DynamicExemplar4_s{i}" for i in range(1, 9)]
Q9_CONDITIONS = ["Q_NarrativeFirstPerson", "Q_NarrativeFirstPerson_s1", "Q_NarrativeFirstPerson_s2",
                 "Q_NarrativeFirstPerson_s3", "Q_NarrativeFirstPerson_s4"]
SELECTED_CONDITIONS = CACHED_CONDITIONS + Q8_CONDITIONS + Q9_CONDITIONS


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bootstrap_ci(values, n=N_BOOTSTRAP, seed=SEED):
    if not values:
        return 0.0, (0.0, 0.0)
    arr = np.array(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n_obs = len(arr)
    boot_means = []
    for _ in range(n):
        idx = rng.choice(n_obs, size=n_obs, replace=True)
        boot_means.append(float(arr[idx].mean()))
    bm = np.array(boot_means)
    return float(arr.mean()), (float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5)))


def aggregate(ranks):
    if not ranks:
        return None
    arr = np.array(ranks)
    mean, ci = bootstrap_ci(ranks)
    return {
        "n": len(ranks),
        "rank1": int((arr == 0).sum()),
        "rank1_pct": float((arr == 0).sum()) / max(1, len(arr)),
        "top10": int((arr < 10).sum()),
        "top10_pct": float((arr < 10).sum()) / max(1, len(arr)),
        "top100": int((arr < 100).sum()),
        "top100_pct": float((arr < 100).sum()) / max(1, len(arr)),
        "mean_rank": mean,
        "mean_rank_ci95": list(ci),
    }


def main() -> None:
    log("=" * 70)
    log("Phase 14.Q9-2 LOPO: Q_NarrativeFirstPerson + cached + Q-8")
    log("=" * 70)

    # === [1] Load candidates ===
    log("[1] Loading candidates ...")
    candidates = []
    cand_resid_list = []

    with IN_PAIRS_14B.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
    cand_resid_11 = np.load(IN_CAND_RESID_11)
    cand_resid_list.append(cand_resid_11)
    log(f"  cached: {len(candidates)}")

    for in_path, in_resid, name in [
        (IN_PAIRS_Q8, IN_CAND_RESID_Q8, "Q-8"),
        (IN_PAIRS_Q9, IN_CAND_RESID_Q9, "Q-9"),
    ]:
        n = 0
        with in_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                c = json.loads(line)
                if c["condition"] not in (Q8_CONDITIONS if name == "Q-8" else Q9_CONDITIONS):
                    continue
                candidates.append(c)
                n += 1
        log(f"  + {name}: {n}")
        cand_resid_list.append(np.load(in_resid))

    cand_residuals_all = np.concatenate(cand_resid_list, axis=0)
    log(f"  total: {len(candidates)}")

    pairs = sorted({(c["user_id"], c["asin"]) for c in candidates})
    cand_by_pair_cond = defaultdict(lambda: defaultdict(list))
    for ci, c in enumerate(candidates):
        key = (c["user_id"], c["asin"])
        cand_by_pair_cond[key][c["condition"]].append(ci)

    # === [2] Load hiddens + neutral ===
    log("[2] Loading hiddens ...")
    npz = np.load(IN_USER_HIDDENS, allow_pickle=True)
    user_ids_cache = list(npz["user_ids"])
    user_hiddens = npz["hiddens"].astype(np.float32)
    uid_to_useridx = {u: i for i, u in enumerate(user_ids_cache)}

    npz_n = np.load(IN_NEUTRAL_HIDDENS, allow_pickle=True)
    neutral_vecs = npz_n["vecs"].astype(np.float32)
    global_neutral = neutral_vecs[:, LAYERS_5, :].mean(axis=0)

    layer_idx = LAYERS_5.index(LAYER_FOCUS)
    user_residuals = user_hiddens[:, :, layer_idx, :] - global_neutral[layer_idx][None, None, :]
    cand_residuals = cand_residuals_all[:, layer_idx, :]
    cand_residuals_cached_only = cand_resid_11[:, layer_idx, :]

    # === [3] Fit PCA-500 ===
    log(f"[3] Fitting Randomized PCA-{MAX_PCA} on (user + 2640 cached) ...")
    user_residuals_flat = user_residuals.reshape(-1, HIDDEN)
    train_pool = np.concatenate([user_residuals_flat, cand_residuals_cached_only], axis=0)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=MAX_PCA, svd_solver='randomized', random_state=SEED)
    pca.fit(train_pool)
    components_full = pca.components_.astype(np.float32)
    pca_mean = pca.mean_.astype(np.float32)
    log(f"  PCA-{MAX_PCA} fitted: explained var {pca.explained_variance_ratio_.sum():.4f}")

    user_pca_full = (user_residuals.reshape(-1, HIDDEN) - pca_mean) @ components_full.T
    user_pca_full_3d = user_pca_full.reshape(N_USERS, 30, MAX_PCA)
    cand_pca_full = (cand_residuals - pca_mean) @ components_full.T

    user_mu_full = user_pca_full_3d.mean(axis=1)
    user_var_full = user_pca_full_3d.var(axis=1)
    pooled_var_full = user_var_full.mean(axis=0, keepdims=True)
    user_var_full = (1 - LW_SHRINK_USER) * user_var_full + LW_SHRINK_USER * pooled_var_full
    user_var_full = np.maximum(user_var_full, MIN_VAR)
    pooled_var_diag_full = (1 - LW_SHRINK_POOLED) * pooled_var_full[0] + LW_SHRINK_POOLED * np.ones(MAX_PCA)

    # === [4] LOPO ===
    log(f"[4] LOPO over {len(pairs)} folds ...")
    pair_keys_valid = [(uid, asin) for (uid, asin) in pairs if uid_to_useridx.get(uid) is not None]
    pair_keys = sorted(pair_keys_valid)

    fold_records = []
    t0 = time.time()

    for fold_idx, (uid, asin) in enumerate(pair_keys):
        target_idx = uid_to_useridx[uid]
        cond_to_cand_idx = cand_by_pair_cond[(uid, asin)]

        all_cand_idx = []
        for cond in SELECTED_CONDITIONS:
            if cond in cond_to_cand_idx:
                all_cand_idx.extend(cond_to_cand_idx[cond])
        cand_residuals_per_pair = cand_residuals[np.array(all_cand_idx)]

        if len(all_cand_idx) == 0:
            fold_records.append({"fold": fold_idx, "user_id": uid, "asin": asin, "best_rank": 200, "n_cands": 0})
            continue

        maha_list = []
        for pca_dim in PCA_DIMS_ENSEMBLE:
            comp = components_full[:pca_dim]
            cand_pca = (cand_residuals_per_pair - pca_mean) @ comp.T
            inv_var = 1.0 / (pooled_var_diag_full[:pca_dim] + 1e-6)
            user_mu = user_mu_full[:, :pca_dim]
            delta = cand_pca[:, None, :] - user_mu[None, :, :]
            weighted = delta * inv_var[None, None, :]
            maha = (delta * weighted).sum(axis=-1)
            maha_list.append(maha)

        maha_300 = maha_list[0]
        maha_500 = maha_list[1]
        maha_300_norm = maha_300 / (maha_300.max() + 1e-9)
        maha_500_norm = maha_500 / (maha_500.max() + 1e-9)
        score = ALPHA * maha_300_norm + (1 - ALPHA) * maha_500_norm

        target_score = score[:, target_idx: target_idx + 1]
        rank_per_cand = (score < target_score).sum(axis=1)

        cond_to_local = defaultdict(list)
        for local_i, ci in enumerate(all_cand_idx):
            cond_to_local[candidates[ci]["condition"]].append(local_i)

        record = {
            "fold": fold_idx,
            "user_id": uid,
            "asin": asin,
            "n_cands": len(all_cand_idx),
        }

        # Per-condition / sub-pool ranks
        sub_pools = {
            "Q9_default_alone": ["Q_NarrativeFirstPerson"],
            "Q9_5subsets_alone": Q9_CONDITIONS,
            "Q8_8subsets_alone": Q8_CONDITIONS,
            "cached_88": CACHED_CONDITIONS,
            "BoK_88_plus_Q9_default": CACHED_CONDITIONS + ["Q_NarrativeFirstPerson"],
            "BoK_152_plus_Q9_default": CACHED_CONDITIONS + Q8_CONDITIONS + ["Q_NarrativeFirstPerson"],
            "BoK_152_plus_Q9_5subsets": CACHED_CONDITIONS + Q8_CONDITIONS + Q9_CONDITIONS,
            "BoK_88_plus_Q9_5subsets": CACHED_CONDITIONS + Q9_CONDITIONS,
            "BoK_full_192": CACHED_CONDITIONS + Q8_CONDITIONS + Q9_CONDITIONS,
        }
        record["by_pool"] = {}
        for pool_name, pool_conds in sub_pools.items():
            sub_local = []
            for cond in pool_conds:
                sub_local.extend(cond_to_local.get(cond, []))
            if sub_local:
                sub_ranks = rank_per_cand[sub_local]
                record["by_pool"][pool_name] = {
                    "best_rank": int(np.min(sub_ranks)),
                    "n_cands": len(sub_local),
                }
            else:
                record["by_pool"][pool_name] = {"best_rank": -1, "n_cands": 0}

        record["best_rank"] = record["by_pool"]["BoK_full_192"]["best_rank"]
        fold_records.append(record)

        if (fold_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            log(f"  fold {fold_idx+1}/{len(pair_keys)} done (elapsed {elapsed:.1f}s)")

    # === [5] Aggregate ===
    log("\n[5] Aggregating LOPO ...")

    pool_groups = [
        "Q9_default_alone",
        "Q9_5subsets_alone",
        "Q8_8subsets_alone",
        "cached_88",
        "BoK_88_plus_Q9_default",
        "BoK_152_plus_Q9_default",
        "BoK_152_plus_Q9_5subsets",
        "BoK_88_plus_Q9_5subsets",
        "BoK_full_192",
    ]

    summary = {}
    for pg in pool_groups:
        ranks = []
        n_cands_set = set()
        for r in fold_records:
            info = r["by_pool"].get(pg, {})
            br = info.get("best_rank", -1)
            ranks.append(br if br >= 0 else 200)
            n_cands_set.add(info.get("n_cands", 0))
        agg = aggregate(ranks)
        summary[pg] = {
            **agg,
            "n_cands_per_pair": list(n_cands_set)[0] if len(n_cands_set) == 1 else sorted(n_cands_set),
        }
        log(f"  {pg} (n_cands={list(n_cands_set)[0] if len(n_cands_set) == 1 else sorted(n_cands_set)}): "
            f"rank1={agg['rank1']}/{agg['n']} ({agg['rank1_pct']*100:.1f}%), "
            f"top100={agg['top100']}/{agg['n']} ({agg['top100_pct']*100:.1f}%), "
            f"mean_rank={agg['mean_rank']:.1f}")

    out = {
        "phase": "14.Q9-2",
        "method": "LOPO BoK-full (cached + Q-8 + Q_NarrativeFirstPerson)",
        "selected_conditions": SELECTED_CONDITIONS,
        "alpha": ALPHA,
        "pca_dims_ensemble": PCA_DIMS_ENSEMBLE,
        "layer": LAYER_FOCUS,
        "n_pairs": len(pair_keys),
        "n_users": N_USERS,
        "summary": summary,
        "comparison_baseline": {
            "phase14_q8_lopo_bok152_q4_8subsets": "rank1=18/30, mean_rank=12.0",
            "phase14_q7_lopo_bok120_q4_4subsets": "rank1=18/30 (60.0%), mean_rank=11.9",
            "phase14_p6_lopo_bok88": "rank1=15/30 (50.0%), mean_rank=20.1",
        },
    }
    OUT_EVAL.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log(f"  eval → {OUT_EVAL}")

    with OUT_PER_PAIR.open("w") as f:
        for r in fold_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"  per_pair → {OUT_PER_PAIR}")

    log("=" * 70)
    log("PHASE 14.Q9-2 LOPO COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()