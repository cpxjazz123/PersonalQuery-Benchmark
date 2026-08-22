#!/usr/bin/env python3
"""Phase 15.4 LOPO: encode exemplars-generated queries + multi-split rerank.

Input: phase15_4_generated.jsonl (Q_PerUserExemplar condition, K=8 per pair)
Output: rank-1, top-100, MRR, mean_rank on 5 splits × 500 test users.

Also: side-by-side compare with D_off baseline (phase14_q10l_generated.jsonl).
"""
from __future__ import annotations
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

IN_GEN_15_4 = OUT_DIR / "phase15_4_generated.jsonl"
IN_USER_HIDDENS = OUT_DIR / "phase14_q10l_user_hiddens_5layers.npz"
IN_NEUTRAL_HIDDENS = OUT_DIR / "phase13_a_neutral_hiddens.npz"
IN_CAND_RESID_D_OFF = OUT_DIR / "phase14_q10l_cand_residuals_qwen.npy"

OUT_EVAL_15_4 = OUT_DIR / "phase15_4_lopo_eval.json"
OUT_PER_PAIR_15_4 = OUT_DIR / "phase15_4_lopo_per_pair.jsonl"

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


def encode_query_residuals(texts, client, layer_focus=LAYER_FOCUS):
    """Encode queries with Qwen2-7B-Instruct, return layer-foc residual vs neutral.

    Uses mean-pool (attention-mask weighted) consistent with Phase 14.F SOTA.
    """
    npz_n = np.load(IN_NEUTRAL_HIDDENS, allow_pickle=True)
    neutral_vecs = npz_n["vecs"].astype(np.float32)
    global_neutral = neutral_vecs[:, LAYERS_5, :].mean(axis=0)
    layer_idx = LAYERS_5.index(layer_focus)

    # get_hidden_states returns mean-pool at the requested layer
    hdict = client.get_hidden_states(texts, layers=[layer_focus], batch_size=32, max_length=256)
    layer_h = hdict[layer_focus]  # (n_texts, 3584)
    resid = layer_h - global_neutral[layer_idx]
    return resid.astype(np.float32)


def main() -> None:
    log("=" * 70)
    log("Phase 15.4 LOPO: per-user exemplar rerank")
    log("=" * 70)

    log("[1] Loading data ...")
    cand_records = []
    with IN_GEN_15_4.open() as f:
        for line in f:
            line = line.strip()
            if line:
                cand_records.append(json.loads(line))
    cand_by_pair = defaultdict(list)
    cand_text_by_idx = {}
    for ci, c in enumerate(cand_records):
        cand_by_pair[(c["user_id"], c["asin"])].append(ci)
        # Use q_final_post (with hard-copy attr fill) for consistency
        cand_text_by_idx[ci] = c.get("q_final_post") or c.get("q_styled", "")
    log(f"  cand records: {len(cand_records)}, unique pairs: {len(cand_by_pair)}")

    # Get texts in cand_records order
    cand_texts = [cand_text_by_idx[i] for i in range(len(cand_records))]

    log("[2] Encoding 15.4 candidates with Qwen ...")
    import sys
    sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
    from llm_client import QwenLocalClient
    client = QwenLocalClient(with_vllm=False)
    cand_residuals_15_4 = encode_query_residuals(cand_texts, client)
    log(f"  cand_residuals_15_4: {cand_residuals_15_4.shape}")

    npz_u = np.load(IN_USER_HIDDENS, allow_pickle=True)
    user_ids = list(npz_u["user_ids"])
    user_hiddens = npz_u["hiddens"].astype(np.float32)
    n_users_total = user_hiddens.shape[0]

    npz_n = np.load(IN_NEUTRAL_HIDDENS, allow_pickle=True)
    neutral_vecs = npz_n["vecs"].astype(np.float32)
    global_neutral = neutral_vecs[:, LAYERS_5, :].mean(axis=0)

    layer_idx = LAYERS_5.index(LAYER_FOCUS)
    user_residuals = user_hiddens[:, :, layer_idx, :] - global_neutral[layer_idx][None, None, :]
    cand_residuals_d_off = np.load(IN_CAND_RESID_D_OFF).astype(np.float32)[:, layer_idx, :]
    log(f"  user_residuals: {user_residuals.shape}, D_off cand_residuals: {cand_residuals_d_off.shape}")

    uid_to_idx = {u: i for i, u in enumerate(user_ids)}

    log(f"[3] Running {len(SPLITS)} splits ...")
    all_splits_summary = []
    all_per_pair = []

    pairs = sorted(cand_by_pair.keys())

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
        train_pool = np.concatenate([train_residuals_flat, cand_residuals_d_off, cand_residuals_15_4], axis=0)

        from sklearn.decomposition import PCA
        pca = PCA(n_components=MAX_PCA, svd_solver='randomized', random_state=42)
        pca.fit(train_pool)
        components_full = pca.components_.astype(np.float32)
        pca_mean = pca.mean_.astype(np.float32)

        user_pca_full = (user_residuals.reshape(-1, HIDDEN) - pca_mean) @ components_full.T
        user_pca_full_3d = user_pca_full.reshape(n_users_total, 30, MAX_PCA)
        cand_pca_15_4 = (cand_residuals_15_4 - pca_mean) @ components_full.T

        user_mu_full = user_pca_full_3d.mean(axis=1)
        user_var_full = user_pca_full_3d.var(axis=1)
        pooled_var_full = user_var_full.mean(axis=0, keepdims=True)
        user_var_full = (1 - LW_SHRINK_USER) * user_var_full + LW_SHRINK_USER * pooled_var_full
        user_var_full = np.maximum(user_var_full, MIN_VAR)
        pooled_var_diag_full = (1 - LW_SHRINK_POOLED) * pooled_var_full[0] + LW_SHRINK_POOLED * np.ones(MAX_PCA)

        test_pairs = [(uid, asin) for (uid, asin) in pairs if uid_to_idx.get(uid) in test_uid_set]
        log(f"  test pairs: {len(test_pairs)}")

        split_records = []
        for (uid, asin) in test_pairs:
            target_idx = uid_to_idx[uid]
            cand_idxs = cand_by_pair[(uid, asin)]
            if not cand_idxs:
                continue
            cand_residuals_per_pair = cand_residuals_15_4[np.array(cand_idxs)]

            # Compute Maha per test cand
            maha_list = []
            for pca_dim in PCA_DIMS_ENSEMBLE:
                comp = components_full[:pca_dim]
                cand_pca = cand_residuals_per_pair @ comp.T
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

            split_records.append({
                "split": split_id,
                "user_id": uid,
                "asin": asin,
                "best_rank": best_rank,
                "n_cands": len(cand_idxs),
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

    out = {
        "phase": "15.4",
        "method": "Q_PerUserExemplar (3 distinctive exemplars stripped) + Qwen layer 26 residual + PCA-300+500 Maha",
        "n_users_total": n_users_total,
        "n_test_per_split": N_TEST_USERS,
        "splits": SPLITS,
        "random_baseline_rank1": random_rank1,
        "cross_split_summary": cross_split_agg,
        "per_split_summary": all_splits_summary,
    }
    OUT_EVAL_15_4.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log(f"  → {OUT_EVAL_15_4}")

    with OUT_PER_PAIR_15_4.open("w") as f:
        for r in all_per_pair:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"  → {OUT_PER_PAIR_15_4}")

    log("=" * 70)
    log("PHASE 15.4 LOPO COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
