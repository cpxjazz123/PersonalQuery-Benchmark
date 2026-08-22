#!/usr/bin/env python3
"""Phase 15.7: Intra-product rank evaluation.

User insight: full-pool rank mixes product confound with style confound. A user who
reviewed "diapers" cannot be a valid match for a query about "USB cable" regardless of
style. Only users who reviewed the SAME product can be compared fairly.

Evaluation:
- Group test pairs by asin (find all users who reviewed same asin)
- For each cand, compute Maha ONLY against users who reviewed same asin
- Report intra-product rank-1, MRR, mean_rank (normalized by intra-product size)

Compare to full-pool rank for same methods.
"""
from __future__ import annotations
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

IN_PAIRS = OUT_DIR / "phase14_q10l_pairs.jsonl"
IN_GEN_D_OFF = OUT_DIR / "phase14_q10l_generated.jsonl"
IN_GEN_15_4 = OUT_DIR / "phase15_4_generated.jsonl"
IN_USER_HIDDENS = OUT_DIR / "phase14_q10l_user_hiddens_5layers.npz"
IN_NEUTRAL_HIDDENS = OUT_DIR / "phase13_a_neutral_hiddens.npz"
IN_CAND_RESID_D_OFF = OUT_DIR / "phase14_q10l_cand_residuals_qwen.npy"

OUT_EVAL = OUT_DIR / "phase15_7_intra_product_eval.json"
OUT_PER_PAIR = OUT_DIR / "phase15_7_intra_product_per_pair.jsonl"

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


def aggregate_intra_product(per_pair_records, asin_to_users):
    """Aggregate per-pair records using intra-product normalized rank.

    For each pair (user, asin), the intra_product_rank is already in the record.
    - intra_product_size: #users reviewing this asin (including target)
    - intra_product_norm_rank: intra_product_rank / (intra_product_size - 1) in [0, 1]
    - intra_product_rank1: intra_product_rank == 0
    """
    if not per_pair_records:
        return None

    # Per-pair metrics
    rank1 = sum(1 for r in per_pair_records if r["intra_product_rank"] == 0)
    n = len(per_pair_records)
    norm_ranks = [r["intra_product_norm_rank"] for r in per_pair_records]
    raw_ranks = [r["intra_product_rank"] for r in per_pair_records]
    sizes = [r["intra_product_size"] for r in per_pair_records]

    mean_norm, mean_norm_ci = bootstrap_ci(norm_ranks)
    mean_raw, mean_raw_ci = bootstrap_ci(raw_ranks)

    return {
        "n": n,
        "n_intra_product_only": sum(1 for r in per_pair_records if r["intra_product_size"] > 1),
        "rank1": rank1,
        "rank1_pct": rank1 / max(1, n),
        "rank1_ci95": list(bootstrap_ci([1.0 if r["intra_product_rank"] == 0 else 0.0 for r in per_pair_records])[1]),
        "mean_intra_product_size": float(np.mean(sizes)),
        "mean_intra_product_norm_rank": mean_norm,
        "mean_intra_product_norm_rank_ci95": list(mean_norm_ci),
        "mean_intra_product_rank_raw": mean_raw,
        "mean_intra_product_rank_raw_ci95": list(mean_raw_ci),
    }


def encode_query_residuals(texts, layer_focus=LAYER_FOCUS):
    """Mean-pool layer hidden states, residual vs neutral."""
    npz_n = np.load(IN_NEUTRAL_HIDDENS, allow_pickle=True)
    neutral_vecs = npz_n["vecs"].astype(np.float32)
    global_neutral = neutral_vecs[:, LAYERS_5, :].mean(axis=0)
    layer_idx = LAYERS_5.index(layer_focus)

    import sys
    sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
    from llm_client import QwenLocalClient
    client = QwenLocalClient(with_vllm=False)
    hdict = client.get_hidden_states(texts, layers=[layer_focus], batch_size=32, max_length=256)
    layer_h = hdict[layer_focus]
    resid = layer_h - global_neutral[layer_idx]
    return resid.astype(np.float32), client


def main() -> None:
    log("=" * 70)
    log("Phase 15.7: Intra-product rank evaluation")
    log("=" * 70)

    log("[1] Loading data ...")
    pairs = []
    with IN_PAIRS.open() as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    # Build asin -> [user_ids]
    asin_to_users = defaultdict(list)
    for p in pairs:
        asin_to_users[p["asin"]].append(p["user_id"])

    # Count asin sizes
    asin_sizes = {a: len(u) for a, u in asin_to_users.items()}
    log(f"  pairs: {len(pairs):,}, unique asins: {len(asin_to_users):,}")
    log(f"  asin size distribution: 1-user={sum(1 for v in asin_sizes.values() if v == 1)}, "
        f"2-3={sum(1 for v in asin_sizes.values() if 2 <= v <= 3)}, "
        f"4-9={sum(1 for v in asin_sizes.values() if 4 <= v <= 9)}, "
        f"10+={sum(1 for v in asin_sizes.values() if v >= 10)}")
    log(f"  max asin size: {max(asin_sizes.values())}")

    # Load D_off candidates
    cand_records_d_off = []
    with IN_GEN_D_OFF.open() as f:
        for line in f:
            line = line.strip()
            if line:
                cand_records_d_off.append(json.loads(line))
    cand_by_pair_d_off = defaultdict(list)
    for ci, c in enumerate(cand_records_d_off):
        cand_by_pair_d_off[(c["user_id"], c["asin"])].append(ci)

    # Load Phase 15.4 candidates (NEW SOTA)
    cand_records_15_4 = []
    with IN_GEN_15_4.open() as f:
        for line in f:
            line = line.strip()
            if line:
                cand_records_15_4.append(json.loads(line))
    cand_by_pair_15_4 = defaultdict(list)
    for ci, c in enumerate(cand_records_15_4):
        cand_by_pair_15_4[(c["user_id"], c["asin"])].append(ci)

    log("[2] Encoding 15.4 candidates ...")
    cand_texts_15_4 = [c.get("q_final_post") or c.get("q_styled", "") for c in cand_records_15_4]
    cand_residuals_15_4, client = encode_query_residuals(cand_texts_15_4)

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
    log(f"  user_residuals: {user_residuals.shape}, D_off cand_residuals: {cand_residuals_d_off.shape}, 15.4 cand_residuals: {cand_residuals_15_4.shape}")

    uid_to_idx = {u: i for i, u in enumerate(user_ids)}

    log(f"[3] Running {len(SPLITS)} splits ...")

    methods = {
        "15.1_D_off": {
            "cand_by_pair": cand_by_pair_d_off,
            "cand_residuals": cand_residuals_d_off,
        },
        "15.4_exemplar": {
            "cand_by_pair": cand_by_pair_15_4,
            "cand_residuals": cand_residuals_15_4,
        },
    }

    method_results = {m: {"per_pair": [], "per_split_summary": []} for m in methods}

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

        user_mu_full = user_pca_full_3d.mean(axis=1)
        user_var_full = user_pca_full_3d.var(axis=1)
        pooled_var_full = user_var_full.mean(axis=0, keepdims=True)
        user_var_full = (1 - LW_SHRINK_USER) * user_var_full + LW_SHRINK_USER * pooled_var_full
        user_var_full = np.maximum(user_var_full, MIN_VAR)
        pooled_var_diag_full = (1 - LW_SHRINK_POOLED) * pooled_var_full[0] + LW_SHRINK_POOLED * np.ones(MAX_PCA)

        # All cand PCA (for both methods)
        cand_pca_d_off = (cand_residuals_d_off - pca_mean) @ components_full.T
        cand_pca_15_4 = (cand_residuals_15_4 - pca_mean) @ components_full.T

        # LOPO test pairs
        test_pairs = [(p["user_id"], p["asin"]) for p in pairs if uid_to_idx.get(p["user_id"]) in test_uid_set]
        log(f"  test pairs: {len(test_pairs)}")

        # For each method, compute intra-product rank
        for method_name, method_data in methods.items():
            cand_by_pair = method_data["cand_by_pair"]
            cand_pca = cand_pca_d_off if method_name == "15.1_D_off" else cand_pca_15_4

            method_per_pair = []
            for (uid, asin) in test_pairs:
                target_idx = uid_to_idx[uid]
                cand_idxs = cand_by_pair[(uid, asin)]
                if not cand_idxs:
                    continue

                # Find intra-product users (excluding target)
                intra_users = [u for u in asin_to_users[asin] if u != uid]
                intra_indices = [uid_to_idx[u] for u in intra_users if u in uid_to_idx]
                if not intra_indices:
                    # Only target reviewed this asin → no intra-product competition
                    method_per_pair.append({
                        "split": split_id,
                        "user_id": uid,
                        "asin": asin,
                        "intra_product_size": 1,
                        "intra_product_rank": 0,
                        "intra_product_norm_rank": 0.0,
                        "intra_product_rank1_trivial": True,
                        "n_cands": len(cand_idxs),
                    })
                    continue

                cand_residuals_per_pair = method_data["cand_residuals"][np.array(cand_idxs)]

                # Compute Maha ONLY against intra-product users
                intra_indices_arr = np.array(intra_indices + [target_idx])  # include target for rank calc
                target_in_intra = len(intra_indices)  # position of target in array

                maha_list = []
                for pca_dim in PCA_DIMS_ENSEMBLE:
                    comp = components_full[:pca_dim]
                    cand_pca_sub = cand_residuals_per_pair @ comp.T
                    user_mu_sub = user_mu_full[intra_indices_arr, :pca_dim]
                    inv_var = 1.0 / (pooled_var_diag_full[:pca_dim] + 1e-6)
                    delta = cand_pca_sub[:, None, :] - user_mu_sub[None, :, :]
                    weighted = delta * inv_var[None, None, :]
                    maha = (delta * weighted).sum(axis=-1)
                    maha_list.append(maha)
                maha_300 = maha_list[0]
                maha_500 = maha_list[1]
                maha_300_norm = maha_300 / (maha_300.max() + 1e-9)
                maha_500_norm = maha_500 / (maha_500.max() + 1e-9)
                score = ALPHA * maha_300_norm + (1 - ALPHA) * maha_500_norm  # (n_cands, n_intra)

                # Rank within intra-product users (smaller score = better)
                target_score = score[:, target_in_intra: target_in_intra + 1]
                rank_per_cand = (score < target_score).sum(axis=1)
                best_rank = int(np.min(rank_per_cand))

                intra_size = len(intra_indices) + 1  # +1 for target
                norm_rank = best_rank / max(1, intra_size - 1) if intra_size > 1 else 0.0

                method_per_pair.append({
                    "split": split_id,
                    "user_id": uid,
                    "asin": asin,
                    "intra_product_size": intra_size,
                    "intra_product_rank": best_rank,
                    "intra_product_norm_rank": norm_rank,
                    "intra_product_rank1_trivial": False,
                    "n_cands": len(cand_idxs),
                })

            agg = aggregate_intra_product(method_per_pair, asin_to_users)
            log(f"  [{method_name}] rank1={agg['rank1']}/{agg['n']} ({agg['rank1_pct']*100:.2f}%), "
                f"n_intra>1={agg['n_intra_product_only']}, "
                f"mean_norm_rank={agg['mean_intra_product_norm_rank']:.4f}, "
                f"mean_intra_size={agg['mean_intra_product_size']:.2f}")

            method_results[method_name]["per_split_summary"].append({
                "split": split_id,
                **agg,
            })
            method_results[method_name]["per_pair"].extend(method_per_pair)

        log(f"  elapsed: {time.time() - t0:.1f}s")

    # Cross-split aggregates
    print(f"\n=== CROSS-SPLIT AGGREGATE ===")
    cross_split_summary = {}
    for method_name in methods:
        all_pp = method_results[method_name]["per_pair"]
        agg = aggregate_intra_product(all_pp, asin_to_users)
        cross_split_summary[method_name] = agg
        log(f"  [{method_name}] rank1={agg['rank1']}/{agg['n']} ({agg['rank1_pct']*100:.2f}%) CI95={agg['rank1_ci95']}")
        log(f"  [{method_name}] n_intra>1={agg['n_intra_product_only']}/{agg['n']} ({agg['n_intra_product_only']/agg['n']*100:.1f}%)")
        log(f"  [{method_name}] mean_norm_rank={agg['mean_intra_product_norm_rank']:.4f} CI95={agg['mean_intra_product_norm_rank_ci95']}")
        log(f"  [{method_name}] mean_intra_size={agg['mean_intra_product_size']:.2f}")

    # Stratified by intra-product size
    print(f"\n=== STRATIFIED BY INTRA-PRODUCT SIZE ===")
    stratified = {}
    for method_name in methods:
        per_pair = method_results[method_name]["per_pair"]
        sizes = np.array([r["intra_product_size"] for r in per_pair])
        ranks = np.array([r["intra_product_rank"] for r in per_pair])
        norm_ranks = np.array([r["intra_product_norm_rank"] for r in per_pair])

        size_buckets = [(1, "size=1 (trivial)"), (2, "size=2"), (3, "size=3"), (4, "size=4-9"), (10, "size=10+")]
        stratified[method_name] = []
        log(f"  [{method_name}]:")
        for lo, label in size_buckets:
            mask = (sizes >= lo) if label.startswith("size=") and lo >= 10 else (sizes == lo if lo <= 3 else (sizes >= 4))
            if lo == 4:
                mask = (sizes >= 4) & (sizes < 10)
            if mask.sum() == 0:
                continue
            r_ranks = ranks[mask]
            r_norm = norm_ranks[mask]
            n_size = int(mask.sum())
            rank1 = int((r_ranks == 0).sum())
            log(f"    {label}: n={n_size}, rank1={rank1}/{n_size} ({rank1/n_size*100:.2f}%), "
                f"mean_norm_rank={r_norm.mean():.4f}")
            stratified[method_name].append({
                "label": label,
                "size_lo": lo,
                "n": n_size,
                "rank1": rank1,
                "rank1_pct": rank1 / max(1, n_size),
                "mean_norm_rank": float(r_norm.mean()),
            })

    out = {
        "phase": "15.7",
        "method": "Intra-product rank evaluation (rank only against users sharing same asin)",
        "n_users_total": n_users_total,
        "n_test_per_split": N_TEST_USERS,
        "splits": SPLITS,
        "asin_size_distribution": {
            "n_unique_asins": len(asin_to_users),
            "size_1_count": sum(1 for v in asin_sizes.values() if v == 1),
            "size_2_3_count": sum(1 for v in asin_sizes.values() if 2 <= v <= 3),
            "size_4_9_count": sum(1 for v in asin_sizes.values() if 4 <= v <= 9),
            "size_10_plus_count": sum(1 for v in asin_sizes.values() if v >= 10),
            "max_asin_size": max(asin_sizes.values()),
        },
        "cross_split_summary": cross_split_summary,
        "stratified_by_intra_product_size": stratified,
    }
    OUT_EVAL.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log(f"  → {OUT_EVAL}")

    with OUT_PER_PAIR.open("w") as f:
        for method_name in methods:
            for r in method_results[method_name]["per_pair"]:
                r["method"] = method_name
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"  → {OUT_PER_PAIR}")

    log("=" * 70)
    log("PHASE 15.7 COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()