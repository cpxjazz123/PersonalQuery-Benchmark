#!/usr/bin/env python3
"""Phase 14.Q-Diagnostic: 12 hard pairs analysis (5 dimensions).

Diagnostic dimensions per missed pair:
1. user_residual gap: min Maha distance from user μ to any candidate
2. off-target: which user μ is the closest candidate actually closest to?
3. style stability: intra-user residual std (vector-wise and per-dim)
4. syntactic pattern coverage: does any candidate use user's dominant features?
5. attrs constraint: how long are the candidates vs user exemplars?
"""
from __future__ import annotations
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

IN_PAIRS_14B = OUT_DIR / "phase14_b_layer_alpha_sweep.jsonl"
IN_PAIRS_Q8 = OUT_DIR / "phase14_q8_generated.jsonl"
IN_USER_HIDDENS = OUT_DIR / "phase14_b_user_hiddens_5layers.npz"
IN_NEUTRAL_HIDDENS = OUT_DIR / "phase13_a_neutral_hiddens.npz"
IN_CAND_RESID_11 = OUT_DIR / "phase14_p_cand_residuals_qwen.npy"
IN_CAND_RESID_Q8 = OUT_DIR / "phase14_q8_cand_residuals_qwen.npy"
IN_PAIRS_10 = OUT_DIR / "phase10_pairs_1000.jsonl"
IN_LOPO_PER_PAIR = OUT_DIR / "phase14_q8_lopo_per_pair.jsonl"

OUT_DIAGNOSTIC = OUT_DIR / "phase14_q_diagnostic.json"
OUT_MISSED_PAIRS = OUT_DIR / "phase14_q_diagnostic_missed.jsonl"

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

FEATURE_20_NAMES = [
    "clause_rate", "acl_rate", "advcl_rate", "ccomp_rate", "relcl_rate",
    "modifier_density", "coordination_density", "mean_dep_distance", "depth_variance",
    "passive_rate", "interrogative_rate", "opener_noun", "opener_pron", "opener_verb",
    "senttype_complex", "nest_max", "nest_mean", "nest_ge2", "depth_eq3", "punct_period",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    log("=" * 70)
    log("Phase 14.Q-Diagnostic: 12 missed pairs 5-dim analysis")
    log("=" * 70)

    # === [1] Load missed pairs ===
    log("[1] Loading missed pairs ...")
    missed = []
    with IN_LOPO_PER_PAIR.open() as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            if r["best_rank"] > 0:
                missed.append(r)
    log(f"  missed: {len(missed)} pairs")
    missed_keys = [(m["user_id"], m["asin"]) for m in missed]

    # === [2] Load all candidates (cached + Q-8) ===
    log("[2] Loading candidates ...")
    candidates = []
    with IN_PAIRS_14B.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
    cand_resid_11 = np.load(IN_CAND_RESID_11)
    n_cached = len(candidates)

    with IN_PAIRS_Q8.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
    cand_resid_q8 = np.load(IN_CAND_RESID_Q8)
    n_q8 = len(candidates) - n_cached
    log(f"  cached: {n_cached}, Q-8: {n_q8}, total: {len(candidates)}")

    cand_residuals_all = np.concatenate([cand_resid_11, cand_resid_q8], axis=0)

    cand_by_pair = defaultdict(list)
    for ci, c in enumerate(candidates):
        cand_by_pair[(c["user_id"], c["asin"])].append(ci)

    # === [3] Load user hiddens + neutral ===
    log("[3] Loading user hiddens ...")
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

    # === [5] Compute maha distance for each missed pair ===
    log("[5] Computing diagnostic for each missed pair ...")

    # Use PCA-300 for Maha distance (smaller dim, more interpretable)
    pca_dim = 300
    comp = components_full[:pca_dim]
    cand_pca = (cand_residuals - pca_mean) @ comp.T
    user_pca = (user_residuals.reshape(-1, HIDDEN) - pca_mean) @ comp.T
    user_pca_3d = user_pca.reshape(N_USERS, 30, pca_dim)
    user_mu = user_pca_3d.mean(axis=1)
    inv_var = 1.0 / (pooled_var_diag_full[:pca_dim] + 1e-6)

    diagnostic_records = []
    for m in missed:
        uid = m["user_id"]
        asin = m["asin"]
        target_idx = uid_to_useridx.get(uid)
        if target_idx is None:
            log(f"  SKIP {uid[:8]} (not in cache)")
            continue

        cand_idx_for_pair = cand_by_pair.get((uid, asin), [])
        if not cand_idx_for_pair:
            continue

        cand_pca_pair = cand_pca[cand_idx_for_pair]
        # Maha to all 198 users
        delta = cand_pca_pair[:, None, :] - user_mu[None, :, :]
        weighted = delta * inv_var[None, None, :]
        maha = (delta * weighted).sum(axis=-1)  # (n_cands, n_users)

        # --- dim 1: distance from user μ to closest candidate ---
        # For target user, find min distance over cands
        target_maha = maha[:, target_idx]
        best_cand_local = int(np.argmin(target_maha))
        best_cand_global = cand_idx_for_pair[best_cand_local]
        best_cand = candidates[best_cand_global]
        min_target_maha = float(target_maha[best_cand_local])

        # --- dim 2: off-target - which user does best_cand actually match? ---
        # For best cand, which user has min maha?
        best_user_idx = int(np.argmin(maha[best_cand_local]))
        best_user_maha = float(maha[best_cand_local, best_user_idx])
        # If best_user != target, this is off-target
        is_off_target = best_user_idx != target_idx
        # Distance to target vs to best_user
        maha_to_target = float(maha[best_cand_local, target_idx])
        maha_to_best_user = best_user_maha
        off_target_ratio = maha_to_best_user / max(maha_to_target, 1e-9)

        # --- dim 3: user style stability ---
        # Intra-user residual std (in PCA-300 space)
        user_var = user_pca_3d[target_idx].var(axis=0)
        user_std_mean = float(np.sqrt(user_var).mean())
        user_std_max = float(np.sqrt(user_var).max())
        user_std_min = float(np.sqrt(user_var).min())

        # L2 norm of user σ
        user_sigma_norm = float(np.sqrt(user_var).sum())

        # --- dim 4: syntactic pattern coverage ---
        # 20d vec for this user (from phase10_pairs_1000)
        # We don't have this here, so approximate: how "extreme" is user's μ vs population?
        # Load phase10 pairs for 20d vec lookup
        # (skip for now, just record raw residual norms)

        # --- dim 5: attrs constraint ---
        # Length of best cand vs typical attr count
        attrs = best_cand.get("attrs", {})
        n_attrs = len(attrs)
        cand_text = best_cand.get("q_final_post", best_cand.get("text", best_cand.get("query", "")))
        cand_text_len = len(cand_text)
        cond = best_cand.get("condition", "")

        record = {
            "user_id": uid,
            "asin": asin,
            "best_rank_lopo": m["best_rank"],
            "n_cands": m["n_cands"],
            "dim1_user_residual_gap": {
                "min_maha_target_user": min_target_maha,
                "best_cand_local_idx": best_cand_local,
                "best_cand_global_idx": best_cand_global,
                "best_cand_condition": cond,
                "best_cand_text": cand_text[:200],
            },
            "dim2_off_target": {
                "is_off_target": is_off_target,
                "best_user_idx": int(best_user_idx),
                "best_user_id": user_ids_cache[best_user_idx] if best_user_idx < len(user_ids_cache) else None,
                "best_user_maha": best_user_maha,
                "maha_to_target": maha_to_target,
                "off_target_ratio": off_target_ratio,
            },
            "dim3_style_stability": {
                "user_sigma_norm": user_sigma_norm,
                "user_std_mean": user_std_mean,
                "user_std_max": user_std_max,
                "user_std_min": user_std_min,
            },
            "dim5_attrs_constraint": {
                "n_attrs": n_attrs,
                "cand_text_len_chars": cand_text_len,
                "cand_text_words": len(cand_text.split()),
            },
        }
        diagnostic_records.append(record)

        log(f"\n--- {uid[:8]} / {asin[:8]} (rank {m['best_rank']}) ---")
        log(f"  dim1: min_maha_target = {min_target_maha:.3f}, best cond = {cond}")
        log(f"  dim2: off_target = {is_off_target}, best_user = {user_ids_cache[best_user_idx][:8] if best_user_idx < len(user_ids_cache) else 'N/A'}, ratio = {off_target_ratio:.3f}")
        log(f"  dim3: sigma_norm = {user_sigma_norm:.2f}, std_mean = {user_std_mean:.2f}")
        log(f"  dim5: n_attrs = {n_attrs}, text_len = {cand_text_len} chars / {len(cand_text.split())} words")

    # === [6] Categorize by failure mode ===
    log("\n=== [6] Categorize failure mode ===")
    failure_modes = {
        "off_target_match": [],  # best cand closer to OTHER user
        "all_far_from_target": [],  # all cands far from target μ
        "narrow_user_style": [],  # user σ very small → rerank hyper-sensitive
        "wide_user_style": [],  # user σ very large → user has no stable style
    }
    for r in diagnostic_records:
        if r["dim2_off_target"]["is_off_target"]:
            failure_modes["off_target_match"].append(r)
        if r["dim1_user_residual_gap"]["min_maha_target_user"] > 5.0:
            failure_modes["all_far_from_target"].append(r)
        if r["dim3_style_stability"]["user_sigma_norm"] < 50:
            failure_modes["narrow_user_style"].append(r)
        if r["dim3_style_stability"]["user_sigma_norm"] > 200:
            failure_modes["wide_user_style"].append(r)

    for mode, rs in failure_modes.items():
        log(f"  {mode}: {len(rs)} pairs")

    # === [7] Save ===
    out = {
        "phase": "14.Q-Diagnostic",
        "n_missed": len(diagnostic_records),
        "summary": {
            "off_target_count": len(failure_modes["off_target_match"]),
            "all_far_count": len(failure_modes["all_far_from_target"]),
            "narrow_user_count": len(failure_modes["narrow_user_style"]),
            "wide_user_count": len(failure_modes["wide_user_style"]),
        },
        "failure_modes": {k: [r["user_id"][:10] + "/" + r["asin"][:8] for r in v] for k, v in failure_modes.items()},
        "records": diagnostic_records,
    }
    OUT_DIAGNOSTIC.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log(f"\n  diagnostic → {OUT_DIAGNOSTIC}")

    with OUT_MISSED_PAIRS.open("w") as f:
        for r in diagnostic_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"  missed detail → {OUT_MISSED_PAIRS}")

    log("=" * 70)
    log("PHASE 14.Q-DIAGNOSTIC COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()