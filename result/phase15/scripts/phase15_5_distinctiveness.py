#!/usr/bin/env python3
"""Phase 15.5: Distinctiveness-aware scoring.

Hypothesis: rank-1 ceiling at 0.56% may be because many users are "average" (low
distinctiveness) — queries for these users are hard to disambiguate. By upweighting
distinctive users (those with unique style), we can:
1. Boost rank-1 for users whose style is genuinely unique (closer to ground truth)
2. Don't penalize already-correct rankings for "average" users

Distinctiveness metric: for each user u, compute mean Maha distance to K nearest
other users (excluding self). Higher = more unique style.

Score:
  S(q, u) = -D_Maha(q, u) + λ × Distinctiveness(u)

Larger S = more likely target user. So distinctive users get a bonus.

Eval: 5 splits × 500 test users, MRR + bootstrap CI.
"""
from __future__ import annotations
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

IN_PAIRS = OUT_DIR / "phase14_q10l_pairs.jsonl"
IN_GEN = OUT_DIR / "phase14_q10l_generated.jsonl"
IN_USER_HIDDENS = OUT_DIR / "phase14_q10l_user_hiddens_5layers.npz"
IN_NEUTRAL_HIDDENS = OUT_DIR / "phase13_a_neutral_hiddens.npz"
IN_CAND_RESID = OUT_DIR / "phase14_q10l_cand_residuals_qwen.npy"

OUT_EVAL = OUT_DIR / "phase15_5_distinct_eval.json"
OUT_PER_PAIR = OUT_DIR / "phase15_5_distinct_per_pair.jsonl"

LAYERS_5 = [8, 14, 18, 22, 26]
LAYER_FOCUS = 26
HIDDEN = 3584

PCA_DIMS_ENSEMBLE = [300, 500]
ALPHA = 0.25
MAX_PCA = max(PCA_DIMS_ENSEMBLE)
LW_SHRINK_USER = 0.1
LW_SHRINK_POOLED = 0.1
MIN_VAR = 1e-4
N_TEST_USERS = 500
SPLITS = [0, 1, 2, 3, 4]
N_BOOTSTRAP = 2000
K_NN_DISTINCT = 10
LAMBDA_DISTINCT = 0.5  # weight for distinctiveness bonus


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bootstrap_ci(values, n=N_BOOTSTRAP, seed=42):
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
        "rank1_ci95": list(bootstrap_ci([1.0 if r == 0 else 0.0 for r in ranks])[1]),
        "top10": int((arr < 10).sum()),
        "top10_pct": float((arr < 10).sum()) / max(1, len(arr)),
        "top100": int((arr < 100).sum()),
        "top100_pct": float((arr < 100).sum()) / max(1, len(arr)),
        "mrr": float((1.0 / (arr + 1)).mean()),
        "mrr_ci95": list(bootstrap_ci([1.0 / (r + 1) for r in ranks])[1]),
        "mean_rank": mean,
        "mean_rank_ci95": list(ci),
    }


def compute_distinctiveness(user_mu_pca, k=K_NN_DISTINCT):
    """For each user, mean Maha distance to K nearest other users (excluding self).
    Higher = more distinctive.
    """
    n = user_mu_pca.shape[0]
    # Cosine-style distance: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    norms = (user_mu_pca ** 2).sum(axis=1, keepdims=True)
    sq_dist = norms + norms.T - 2 * user_mu_pca @ user_mu_pca.T
    sq_dist = np.maximum(sq_dist, 0)
    sq_dist[np.arange(n), np.arange(n)] = np.inf  # exclude self
    # K nearest
    nearest_k = np.partition(sq_dist, k, axis=1)[:, :k]
    distinctiveness = nearest_k.mean(axis=1)  # (n_users,)
    return distinctiveness.astype(np.float32)


def main() -> None:
    log("=" * 70)
    log(f"Phase 15.5: Distinctiveness-aware scoring (λ={LAMBDA_DISTINCT})")
    log("=" * 70)

    log("[1] Loading data ...")
    cand_records = []
    with IN_GEN.open() as f:
        for line in f:
            line = line.strip()
            if line:
                cand_records.append(json.loads(line))
    cand_by_pair = defaultdict(list)
    for ci, c in enumerate(cand_records):
        cand_by_pair[(c["user_id"], c["asin"])].append(ci)
    pairs = sorted(cand_by_pair.keys())

    npz_u = np.load(IN_USER_HIDDENS, allow_pickle=True)
    user_ids = list(npz_u["user_ids"])
    user_hiddens = npz_u["hiddens"].astype(np.float32)
    n_users_total = user_hiddens.shape[0]

    npz_n = np.load(IN_NEUTRAL_HIDDENS, allow_pickle=True)
    neutral_vecs = npz_n["vecs"].astype(np.float32)
    global_neutral = neutral_vecs[:, LAYERS_5, :].mean(axis=0)

    layer_idx = LAYERS_5.index(LAYER_FOCUS)
    user_residuals = user_hiddens[:, :, layer_idx, :] - global_neutral[layer_idx][None, None, :]
    cand_residuals = np.load(IN_CAND_RESID).astype(np.float32)[:, layer_idx, :]
    log(f"  user_residuals: {user_residuals.shape}, cand_residuals: {cand_residuals.shape}")

    uid_to_idx = {u: i for i, u in enumerate(user_ids)}

    log(f"[2] Running {len(SPLITS)} splits ...")
    all_splits_summary = []
    all_per_pair = []

    for split_id in SPLITS:
        log(f"\n--- Split {split_id} ---")
        t0 = time.time()
        rng = np.random.default_rng(split_id)
        all_indices = np.arange(n_users_total)
        test_indices = rng.choice(all_indices, size=N_TEST_USERS, replace=False)
        test_uid_set = set(int(i) for i in test_indices)
        train_indices = np.array([i for i in range(n_users_total) if i not in test_uid_set])

        # PCA fit on train
        train_residuals_flat = user_residuals[train_indices].reshape(-1, HIDDEN)
        train_pool = np.concatenate([train_residuals_flat, cand_residuals], axis=0)

        from sklearn.decomposition import PCA
        pca = PCA(n_components=MAX_PCA, svd_solver='randomized', random_state=42)
        pca.fit(train_pool)
        components_full = pca.components_.astype(np.float32)
        pca_mean = pca.mean_.astype(np.float32)

        user_pca_full = (user_residuals.reshape(-1, HIDDEN) - pca_mean) @ components_full.T
        user_pca_full_3d = user_pca_full.reshape(n_users_total, 30, MAX_PCA)
        cand_pca_full = (cand_residuals - pca_mean) @ components_full.T

        user_mu_full = user_pca_full_3d.mean(axis=1)
        user_var_full = user_pca_full_3d.var(axis=1)
        pooled_var_full = user_var_full.mean(axis=0, keepdims=True)
        user_var_full = (1 - LW_SHRINK_USER) * user_var_full + LW_SHRINK_USER * pooled_var_full
        user_var_full = np.maximum(user_var_full, MIN_VAR)
        pooled_var_diag_full = (1 - LW_SHRINK_POOLED) * pooled_var_full[0] + LW_SHRINK_POOLED * np.ones(MAX_PCA)

        # Compute distinctiveness: for each user, K-NN distance to TRAIN users only
        # (avoid leakage from test; train-only)
        train_user_emb = user_mu_full[train_indices]
        train_norms = (train_user_emb ** 2).sum(axis=1, keepdims=True)
        full_emb = user_mu_full  # (n_users, PCA)
        full_norms = (full_emb ** 2).sum(axis=1, keepdims=True)
        sq_dist_full = full_norms + train_norms.T - 2 * full_emb @ train_user_emb.T
        sq_dist_full = np.maximum(sq_dist_full, 0)
        # Exclude self for train users
        train_self_mask = np.zeros((n_users_total, train_indices.shape[0]), dtype=bool)
        for ti, tidx in enumerate(train_indices):
            train_self_mask[tidx, ti] = True
        sq_dist_full[train_self_mask] = np.inf
        nearest_k_full = np.partition(sq_dist_full, K_NN_DISTINCT, axis=1)[:, :K_NN_DISTINCT]
        distinctiveness_full = nearest_k_full.mean(axis=1)

        # Normalize distinctiveness to [0, 1]
        d_min = distinctiveness_full.min()
        d_max = distinctiveness_full.max()
        d_norm = (distinctiveness_full - d_min) / (d_max - d_min + 1e-9)
        log(f"  distinctiveness: min={d_min:.3f}, max={d_max:.3f}, mean={distinctiveness_full.mean():.3f}")

        # LOPO on test
        test_pairs = [(uid, asin) for (uid, asin) in pairs if uid_to_idx.get(uid) in test_uid_set]
        log(f"  test pairs: {len(test_pairs)}")

        split_records = []
        for (uid, asin) in test_pairs:
            target_idx = uid_to_idx[uid]
            cand_idxs = cand_by_pair[(uid, asin)]
            if not cand_idxs:
                continue
            cand_residuals_per_pair = cand_residuals[np.array(cand_idxs)]

            # Compute Maha per test cand
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
            maha_score = ALPHA * maha_300_norm + (1 - ALPHA) * maha_500_norm  # (n_cands, n_users)

            # Combined score: -D_Maha + λ × Distinctiveness (broadcast distinctiveness across cands)
            # Normalize Maha to [0, 1] first (per-cand)
            m_min = maha_score.min(axis=1, keepdims=True)
            m_max = maha_score.max(axis=1, keepdims=True)
            maha_n = (maha_score - m_min) / (m_max - m_min + 1e-9)
            # Combined: smaller combined = better
            combined = maha_n - LAMBDA_DISTINCT * d_norm[None, :]

            target_combined = combined[:, target_idx: target_idx + 1]
            rank_per_cand = (combined < target_combined).sum(axis=1)

            best_rank = int(np.min(rank_per_cand))

            split_records.append({
                "split": split_id,
                "user_id": uid,
                "asin": asin,
                "best_rank": best_rank,
                "n_cands": len(cand_idxs),
                "distinctiveness": float(d_norm[target_idx]),
            })

        agg = aggregate([r["best_rank"] for r in split_records])
        log(f"  rank1={agg['rank1']}/{agg['n']} ({agg['rank1_pct']*100:.2f}%), "
            f"top100={agg['top100']}/{agg['n']} ({agg['top100_pct']*100:.2f}%), "
            f"MRR={agg['mrr']:.4f}, mean_rank={agg['mean_rank']:.2f}")

        all_splits_summary.append({
            "split": split_id,
            "n_test_users": len(test_uid_set),
            "n_test_pairs": len(test_pairs),
            **agg,
        })
        all_per_pair.extend(split_records)
        log(f"  elapsed: {time.time() - t0:.1f}s")

    all_ranks = [r["best_rank"] for r in all_per_pair]
    cross_split_agg = aggregate(all_ranks)
    log(f"\n=== CROSS-SPLIT AGGREGATE ({len(all_ranks)} pairs × {len(SPLITS)} splits) ===")
    log(f"  rank1={cross_split_agg['rank1']}/{cross_split_agg['n']} ({cross_split_agg['rank1_pct']*100:.2f}%) CI95={cross_split_agg['rank1_ci95']}")
    log(f"  top10={cross_split_agg['top10']}/{cross_split_agg['n']} ({cross_split_agg['top10_pct']*100:.2f}%)")
    log(f"  top100={cross_split_agg['top100']}/{cross_split_agg['n']} ({cross_split_agg['top100_pct']*100:.2f}%)")
    log(f"  MRR={cross_split_agg['mrr']:.4f} CI95={cross_split_agg['mrr_ci95']}")
    log(f"  mean_rank={cross_split_agg['mean_rank']:.2f} CI95={cross_split_agg['mean_rank_ci95']}")

    random_rank1 = 1.0 / n_users_total
    log(f"  random baseline rank1={random_rank1*100:.3f}% (1/{n_users_total})")

    # Stratified by distinctiveness quartile
    log(f"\n=== STRATIFIED BY DISTINCTIVENESS QUARTILE ===")
    distinct_vals = np.array([r["distinctiveness"] for r in all_per_pair])
    distinct_ranks = np.array([r["best_rank"] for r in all_per_pair])
    quartiles = np.percentile(distinct_vals, [25, 50, 75])
    q_bounds = [0, quartiles[0], quartiles[1], quartiles[2], 1.01]
    stratified = []
    for qi in range(4):
        lo, hi = q_bounds[qi], q_bounds[qi + 1]
        mask = (distinct_vals >= lo) & (distinct_vals < hi)
        if mask.sum() == 0:
            continue
        ranks_q = distinct_ranks[mask]
        agg_q = aggregate(ranks_q.tolist())
        stratified.append({
            "quartile": qi + 1,
            "d_lo": float(lo),
            "d_hi": float(hi),
            "n": int(mask.sum()),
            "agg": agg_q,
        })
        log(f"  Q{qi+1} (d∈[{lo:.3f},{hi:.3f}], n={mask.sum()}): rank1={agg_q['rank1']}/{agg_q['n']} ({agg_q['rank1_pct']*100:.2f}%), "
            f"top100={agg_q['top100']}/{agg_q['n']} ({agg_q['top100_pct']*100:.2f}%), MRR={agg_q['mrr']:.4f}, mean_rank={agg_q['mean_rank']:.2f}")

    out = {
        "phase": "15.5",
        "method": f"Distinctiveness-aware scoring (Maha + λ={LAMBDA_DISTINCT} × Distinctness, K_NN={K_NN_DISTINCT})",
        "n_users_total": n_users_total,
        "n_test_per_split": N_TEST_USERS,
        "splits": SPLITS,
        "lambda_distinct": LAMBDA_DISTINCT,
        "random_baseline_rank1": random_rank1,
        "cross_split_summary": cross_split_agg,
        "per_split_summary": all_splits_summary,
        "stratified_by_distinctiveness": stratified,
    }
    OUT_EVAL.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log(f"  → {OUT_EVAL}")

    with OUT_PER_PAIR.open("w") as f:
        for r in all_per_pair:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"  → {OUT_PER_PAIR}")

    log("=" * 70)
    log("PHASE 15.5 COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
