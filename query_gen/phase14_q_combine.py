#!/usr/bin/env python3
"""Phase 14.Q-combine: Combine all Q phase candidates.

Test if combining Q_Dynamic (Q-1) + Q_DynamicExemplar (Q-1) +
Q_DynamicExemplar4 (Q-6) + 4 subsets (Q-7) + 8 subsets (Q-8) adds value
on top of BoK-152 (88 + 64 Q-8 subsets).
"""
from __future__ import annotations
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

IN_PAIRS_14B = OUT_DIR / "phase14_b_layer_alpha_sweep.jsonl"
IN_PAIRS_Q1 = OUT_DIR / "phase14_q_generated.jsonl"
IN_PAIRS_Q6 = OUT_DIR / "phase14_q6_generated.jsonl"
IN_PAIRS_Q7 = OUT_DIR / "phase14_q7_generated.jsonl"
IN_PAIRS_Q8 = OUT_DIR / "phase14_q8_generated.jsonl"

IN_USER_HIDDENS = OUT_DIR / "phase14_b_user_hiddens_5layers.npz"
IN_NEUTRAL_HIDDENS = OUT_DIR / "phase13_a_neutral_hiddens.npz"
IN_CAND_RESID_11 = OUT_DIR / "phase14_p_cand_residuals_qwen.npy"
IN_CAND_RESID_Q1 = OUT_DIR / "phase14_q_cand_residuals_qwen.npy"
IN_CAND_RESID_Q6 = OUT_DIR / "phase14_q6_cand_residuals_qwen.npy"
IN_CAND_RESID_Q7 = OUT_DIR / "phase14_q7_cand_residuals_qwen.npy"
IN_CAND_RESID_Q8 = OUT_DIR / "phase14_q8_cand_residuals_qwen.npy"

OUT_EVAL = OUT_DIR / "phase14_q_combine_eval.json"
OUT_PER_PAIR = OUT_DIR / "phase14_q_combine_per_pair.jsonl"

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
Q1_CONDITIONS = ["Q_Dynamic", "Q_DynamicExemplar"]
Q6_CONDITIONS = ["Q_Dynamic", "Q_DynamicExemplar4"]
Q7_CONDITIONS = ["Q_DynamicExemplar4_s1", "Q_DynamicExemplar4_s2", "Q_DynamicExemplar4_s3", "Q_DynamicExemplar4_s4"]
Q8_CONDITIONS = [f"Q_DynamicExemplar4_s{i}" for i in range(1, 9)]


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
    log("Phase 14.Q-combine: Combine all Q phases")
    log("=" * 70)

    # === [1] Load candidates ===
    log("[1] Loading candidates ...")
    candidates = []
    resid_arrays = []

    with IN_PAIRS_14B.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
    log(f"  cached (Phase 14.B): {len(candidates)}")
    cand_resid_11 = np.load(IN_CAND_RESID_11)
    resid_arrays.append(cand_resid_11)

    for in_path, in_resid, name in [
        (IN_PAIRS_Q1, IN_CAND_RESID_Q1, "Q-1"),
        (IN_PAIRS_Q6, IN_CAND_RESID_Q6, "Q-6"),
        (IN_PAIRS_Q7, IN_CAND_RESID_Q7, "Q-7"),
        (IN_PAIRS_Q8, IN_CAND_RESID_Q8, "Q-8"),
    ]:
        n = 0
        with in_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                candidates.append(json.loads(line))
                n += 1
        log(f"  + {name}: {n}")
        resid_arrays.append(np.load(in_resid))

    cand_residuals_all = np.concatenate(resid_arrays, axis=0)
    log(f"  total: {len(candidates)} candidates, residuals {cand_residuals_all.shape}")

    pairs = sorted({(c["user_id"], c["asin"]) for c in candidates})
    cand_by_pair_cond = defaultdict(lambda: defaultdict(list))
    for ci, c in enumerate(candidates):
        key = (c["user_id"], c["asin"])
        cand_by_pair_cond[key][c["condition"]].append(ci)

    # === [2] Load hiddens + neutral ===
    log("[2] Loading hiddens + neutral ...")
    npz = np.load(IN_USER_HIDDENS, allow_pickle=True)
    user_ids_cache = list(npz["user_ids"])
    user_hiddens = npz["hiddens"].astype(np.float32)
    uid_to_useridx = {u: i for i, u in enumerate(user_ids_cache)}

    npz_n = np.load(IN_NEUTRAL_HIDDENS, allow_pickle=True)
    neutral_vecs = npz_n["vecs"].astype(np.float32)
    global_neutral = neutral_vecs[:, LAYERS_5, :].mean(axis=0)

    layer_idx = LAYERS_5.index(LAYER_FOCUS)
    user_residuals = user_hiddens[:, :, layer_idx, :] - global_neutral[layer_idx][None, None, :]

    log("[3] Cand residuals ...")
    cand_residuals = cand_residuals_all[:, layer_idx, :]
    cand_residuals_cached_only = cand_resid_11[:, layer_idx, :]

    # === [4] Fit PCA-500 on user + cached only ===
    log(f"[4] Fitting Randomized PCA-{MAX_PCA} on (user + 2640 cached) ...")
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

    # === [5] LOPO ===
    log(f"[5] LOPO over {len(pairs)} folds ...")
    pair_keys_valid = [(uid, asin) for (uid, asin) in pairs if uid_to_useridx.get(uid) is not None]
    pair_keys = sorted(pair_keys_valid)

    fold_records = []
    t0 = time.time()

    for fold_idx, (uid, asin) in enumerate(pair_keys):
        target_idx = uid_to_useridx[uid]
        all_cand_idx = [i for i, c in enumerate(candidates) if c["user_id"] == uid and c["asin"] == asin]
        cand_residuals_per_pair = cand_residuals[np.array(all_cand_idx)]

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

        # Per-condition ranks
        cond_to_cand_local = defaultdict(list)
        for local_i, ci in enumerate(all_cand_idx):
            cond_to_cand_local[candidates[ci]["condition"]].append(local_i)

        record = {
            "fold": fold_idx,
            "user_id": uid,
            "asin": asin,
            "n_cands": len(all_cand_idx),
            "best_rank": int(np.min(rank_per_cand)),
            "by_cond": {},
        }

        for cond_name, cond_conds in [
            ("Q1_Q_Dynamic", ["Q_Dynamic"]),
            ("Q1_Q_DynamicExemplar", ["Q_DynamicExemplar"]),
            ("Q6_Q_Dynamic", ["Q_Dynamic"]),
            ("Q6_Q_DynamicExemplar4", ["Q_DynamicExemplar4"]),
            ("Q7_4subsets", Q7_CONDITIONS),
            ("Q8_8subsets", Q8_CONDITIONS),
            ("cached_88", CACHED_CONDITIONS),
            ("cached_88_plus_Q1_all_32", CACHED_CONDITIONS + Q1_CONDITIONS),
            ("cached_88_plus_Q6_all_32", CACHED_CONDITIONS + Q6_CONDITIONS),
            ("cached_88_plus_Q7_4subsets_120", CACHED_CONDITIONS + Q7_CONDITIONS),
            ("cached_88_plus_Q8_8subsets_152", CACHED_CONDITIONS + Q8_CONDITIONS),
            ("cached_88_plus_Q8_Q1_168", CACHED_CONDITIONS + Q8_CONDITIONS + Q1_CONDITIONS),
            ("cached_88_plus_all_176", CACHED_CONDITIONS + Q8_CONDITIONS + Q1_CONDITIONS + Q6_CONDITIONS),
        ]:
            sub_local = []
            for cond in cond_conds:
                sub_local.extend(cond_to_cand_local.get(cond, []))
            if sub_local:
                sub_ranks = rank_per_cand[sub_local]
                record["by_cond"][cond_name] = {
                    "best_rank": int(np.min(sub_ranks)),
                    "n_cands": len(sub_local),
                }
            else:
                record["by_cond"][cond_name] = {"best_rank": -1, "n_cands": 0}

        fold_records.append(record)

        if (fold_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            log(f"  fold {fold_idx+1}/{len(pair_keys)} done (elapsed {elapsed:.1f}s)")

    # === [6] Aggregate ===
    log("[6] Aggregating ...")

    cond_groups = [
        "Q1_Q_Dynamic",
        "Q1_Q_DynamicExemplar",
        "Q6_Q_Dynamic",
        "Q6_Q_DynamicExemplar4",
        "Q7_4subsets",
        "Q8_8subsets",
        "cached_88",
        "cached_88_plus_Q1_all_32",
        "cached_88_plus_Q6_all_32",
        "cached_88_plus_Q7_4subsets_120",
        "cached_88_plus_Q8_8subsets_152",
        "cached_88_plus_Q8_Q1_168",
        "cached_88_plus_all_176",
    ]

    summary = {}
    for cg in cond_groups:
        ranks = [r["by_cond"][cg]["best_rank"] if r["by_cond"][cg]["best_rank"] >= 0 else 200 for r in fold_records]
        agg = aggregate(ranks)
        n_cands_set = {r["by_cond"][cg]["n_cands"] for r in fold_records}
        summary[cg] = {
            **agg,
            "n_cands_per_pair": list(n_cands_set)[0] if len(n_cands_set) == 1 else sorted(n_cands_set),
        }
        log(f"  {cg} (n_cands={list(n_cands_set)[0] if len(n_cands_set) == 1 else sorted(n_cands_set)}): "
            f"rank1={agg['rank1']}/{agg['n']} ({agg['rank1_pct']*100:.1f}%), "
            f"top100={agg['top100']}/{agg['n']} ({agg['top100_pct']*100:.1f}%), "
            f"mean_rank={agg['mean_rank']:.1f}")

    out = {
        "phase": "14.Q-combine",
        "method": "Combine all Q phases on BoK-88 (full-fit LOPO-style)",
        "n_pairs": len(pair_keys),
        "n_users": N_USERS,
        "summary": summary,
    }
    OUT_EVAL.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log(f"  eval → {OUT_EVAL}")

    with OUT_PER_PAIR.open("w") as f:
        for r in fold_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"  per_pair → {OUT_PER_PAIR}")

    log("=" * 70)
    log("PHASE 14.Q-combine COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()