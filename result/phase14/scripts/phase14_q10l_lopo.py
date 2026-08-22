#!/usr/bin/env python3
"""Phase 14.Q10-L LOPO rerank: 2041-fold validation.

Tests Phase 14.Q10 SOTA rerank at scale.

Pipeline:
- 2041 users × 30 sents (user_hiddens)
- 2041 pairs × K=8 D_off candidates (cand_residuals)
- LOPO: each fold = 1 user, rank-1/top100/mean_rank

Output:
- phase14_q10l_lopo_eval.json
- phase14_q10l_lopo_per_pair.jsonl
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

OUT_EVAL = OUT_DIR / "phase14_q10l_lopo_eval.json"
OUT_PER_PAIR = OUT_DIR / "phase14_q10l_lopo_per_pair.jsonl"

LAYERS_5 = [8, 14, 18, 22, 26]
LAYER_FOCUS = 26
HIDDEN = 3584

PCA_DIMS_ENSEMBLE = [300, 500]
ALPHA = 0.25
MAX_PCA = max(PCA_DIMS_ENSEMBLE)
LW_SHRINK_USER = 0.1
LW_SHRINK_POOLED = 0.1
MIN_VAR = 1e-4
SEED = 42
N_BOOTSTRAP = 2000


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
    log("Phase 14.Q10-L LOPO: 2041-fold D_off rerank validation")
    log("=" * 70)

    log("[1] Loading candidates ...")
    cand_records = []
    with IN_GEN.open() as f:
        for line in f:
            line = line.strip()
            if line:
                cand_records.append(json.loads(line))
    log(f"  candidates: {len(cand_records)}")

    pairs = sorted({(c["user_id"], c["asin"]) for c in cand_records})
    log(f"  unique pairs: {len(pairs)}")

    cand_by_pair = defaultdict(list)
    for ci, c in enumerate(cand_records):
        cand_by_pair[(c["user_id"], c["asin"])].append(ci)

    log("[2] Loading user hiddens + neutral ...")
    npz_u = np.load(IN_USER_HIDDENS, allow_pickle=True)
    user_ids = list(npz_u["user_ids"])
    user_hiddens = npz_u["hiddens"].astype(np.float32)  # (n_users, 30, 5, 3584)
    n_users_total = user_hiddens.shape[0]
    log(f"  user hiddens: {user_hiddens.shape}")

    npz_n = np.load(IN_NEUTRAL_HIDDENS, allow_pickle=True)
    neutral_vecs = npz_n["vecs"].astype(np.float32)
    global_neutral = neutral_vecs[:, LAYERS_5, :].mean(axis=0)

    layer_idx = LAYERS_5.index(LAYER_FOCUS)
    user_residuals = user_hiddens[:, :, layer_idx, :] - global_neutral[layer_idx][None, None, :]
    log(f"  user_residuals shape: {user_residuals.shape}")

    cand_residuals_all = np.load(IN_CAND_RESID).astype(np.float32)  # (n_cands, 5, 3584)
    cand_residuals = cand_residuals_all[:, layer_idx, :]
    log(f"  cand_residuals shape: {cand_residuals.shape}")

    log(f"[3] Fitting Randomized PCA-{MAX_PCA} on (user + cand residuals) ...")
    user_residuals_flat = user_residuals.reshape(-1, HIDDEN)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=MAX_PCA, svd_solver='randomized', random_state=SEED)
    train_pool = np.concatenate([user_residuals_flat, cand_residuals], axis=0)
    pca.fit(train_pool)
    components_full = pca.components_.astype(np.float32)
    pca_mean = pca.mean_.astype(np.float32)
    log(f"  PCA-{MAX_PCA} fitted: explained var {pca.explained_variance_ratio_.sum():.4f}")

    user_pca_full = (user_residuals.reshape(-1, HIDDEN) - pca_mean) @ components_full.T
    user_pca_full_3d = user_pca_full.reshape(n_users_total, 30, MAX_PCA)
    cand_pca_full = (cand_residuals - pca_mean) @ components_full.T

    user_mu_full = user_pca_full_3d.mean(axis=1)
    user_var_full = user_pca_full_3d.var(axis=1)
    pooled_var_full = user_var_full.mean(axis=0, keepdims=True)
    user_var_full = (1 - LW_SHRINK_USER) * user_var_full + LW_SHRINK_USER * pooled_var_full
    user_var_full = np.maximum(user_var_full, MIN_VAR)
    pooled_var_diag_full = (1 - LW_SHRINK_POOLED) * pooled_var_full[0] + LW_SHRINK_POOLED * np.ones(MAX_PCA)

    log(f"[4] LOPO over {len(pairs)} folds ...")
    uid_to_idx = {u: i for i, u in enumerate(user_ids)}
    pair_keys_valid = [(uid, asin) for (uid, asin) in pairs if uid_to_idx.get(uid) is not None]
    pair_keys = sorted(pair_keys_valid)
    log(f"  valid pairs: {len(pair_keys)}")

    fold_records = []
    t0 = time.time()

    for fold_idx, (uid, asin) in enumerate(pair_keys):
        target_idx = uid_to_idx[uid]
        cand_idxs = cand_by_pair[(uid, asin)]

        if not cand_idxs:
            fold_records.append({"fold": fold_idx, "user_id": uid, "asin": asin, "best_rank": 200, "n_cands": 0})
            continue

        cand_residuals_per_pair = cand_residuals[np.array(cand_idxs)]

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

        best_rank = int(np.min(rank_per_cand))
        fold_records.append({
            "fold": fold_idx,
            "user_id": uid,
            "asin": asin,
            "best_rank": best_rank,
            "n_cands": len(cand_idxs),
            "cand_ranks": rank_per_cand.tolist(),
        })

        if (fold_idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (fold_idx + 1) / elapsed
            eta = (len(pair_keys) - fold_idx - 1) / rate
            log(f"  fold {fold_idx+1}/{len(pair_keys)} ({elapsed:.1f}s, {rate:.1f} fold/s, ETA {eta:.0f}s)")

    log("\n[5] Aggregating LOPO ...")
    summary = aggregate([r["best_rank"] for r in fold_records])
    log(f"  D_off alone (n_cands=8): "
        f"rank1={summary['rank1']}/{summary['n']} ({summary['rank1_pct']*100:.1f}%), "
        f"top10={summary['top10']}/{summary['n']} ({summary['top10_pct']*100:.1f}%), "
        f"top100={summary['top100']}/{summary['n']} ({summary['top100_pct']*100:.1f}%), "
        f"mean_rank={summary['mean_rank']:.2f} CI95={summary['mean_rank_ci95']}")

    out = {
        "phase": "14.Q10-L",
        "method": "LOPO 2041-fold validation (D_off K=8 baseline)",
        "n_pairs": len(pair_keys),
        "n_users": n_users_total,
        "alpha": ALPHA,
        "pca_dims_ensemble": PCA_DIMS_ENSEMBLE,
        "layer": LAYER_FOCUS,
        "summary": summary,
        "comparison_baseline": {
            "phase14_q10_bok184_30pairs": "rank1=18/30 (60.0%), mean_rank=10.9",
            "phase14_p6_bok88_30pairs": "rank1=15/30 (50.0%), mean_rank=20.1",
        },
    }
    OUT_EVAL.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log(f"  → {OUT_EVAL}")

    with OUT_PER_PAIR.open("w") as f:
        for r in fold_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"  → {OUT_PER_PAIR}")

    log("=" * 70)
    log("PHASE 14.Q10-L LOPO COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()