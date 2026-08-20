#!/usr/bin/env python3
"""Phase 11.A: Per-user Gaussian distribution N(mu_u, Sigma_u) at 768d AnnaWegmann space.

Step up from point vector to per-user Gaussian:
  1. Per-sentence 768d AnnaWegmann embeddings for all phase10 train sentences
  2. Per-user covariance Sigma_u with Ledoit-Wolf shrinkage
  3. Store Cholesky L_u (low-rank approximation) + sigma_diag for fast scoring
  4. PCA 768 -> 128 to reduce covariance rank issue
  5. Validate: re-rank correlation rho > 0.85 vs Phase 10.15 Mahalanobis baseline

Output:
  - phase11_a_user_gaussians_768d.npz  (keys: user_ids, mu, sigma_diag, sigma_lowrank, pca_components, pca_mean)
  - phase11_a_user_gaussians_meta.json
  - phase11_a_validation_maha.json
  - phase11_a_per_sentence_embs_768d.npz (intermediate cache)

Acceptance:
  - LW shrinkage alpha > 0.5
  - Split-half cosine mu_u > 0.7
  - Sigma_u PD (min eigenvalue > 1e-6, else raise)
  - Re-rank correlation with Phase 10.15: rho > 0.85
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
OUT_GAUSSIANS = OUT_DIR / "phase11_a_user_gaussians_768d.npz"
OUT_GAUSSIANS_META = OUT_DIR / "phase11_a_user_gaussians_meta.json"
OUT_VALIDATION = OUT_DIR / "phase11_a_validation_maha.json"
OUT_PER_SENT = OUT_DIR / "phase11_a_per_sentence_embs_768d.npz"
OUT_USER_EMBS_META = OUT_DIR / "phase11_a_user_embs_meta.json"

VADES_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"
RAW_SENTENCES_FILE = VADES_DIR / "vades_prototype_3000u_v6_raw_sentences.jsonl"

PHASE10_PAIRS = OUT_DIR / "phase10_pairs_1000.jsonl"
PHASE10_USER_EMBS = OUT_DIR / "phase10_user_embs_768d.npz"
PHASE10_RANK1_EVAL = OUT_DIR / "phase10_15_rank1_eval.json"

ANNA_SNAPSHOT_DIR = "/fs04/ar57/wenyu/.cache/huggingface/hub/models--AnnaWegmann--Style-Embedding/snapshots"

# Hardcoded constants
N_USERS = 876
EMB_DIM = 768
PCA_DIM = 128
LW_SHRINKAGE_MIN = 1e-3
RANDOM_SEED = 42
N_MAHA_VALIDATION_PAIRS = 200
ENCODER_BATCH = 128


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("=" * 70)
    log("Phase 11.A: Per-user Gaussian N(mu_u, Sigma_u) at 768d AnnaWegmann")
    log("=" * 70)

    np.random.seed(RANDOM_SEED)

    # === Load phase10 user_ids ===
    log("[1] Loading phase10 pairs (876 users) ...")
    phase10_users = []
    with PHASE10_PAIRS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            phase10_users.append(obj["user_id"])
    phase10_users = sorted(set(phase10_users))
    log(f"  phase10 users: {len(phase10_users)}")

    uid_to_idx = {u: i for i, u in enumerate(phase10_users)}

    # === Load per-sentence embeddings (re-use cache or re-encode) ===
    log("[2] Loading per-sentence 768d AnnaWegmann embeddings ...")
    if OUT_PER_SENT.exists():
        log(f"  Cache hit: {OUT_PER_SENT}")
        npz = np.load(OUT_PER_SENT, allow_pickle=True)
        sent_uids = list(npz["sent_uids"])
        sent_embs = npz["sent_embs"]
        log(f"  Loaded {len(sent_uids)} per-sentence embeddings, shape={sent_embs.shape}")
    else:
        log(f"  Cache miss: encoding {len(phase10_users)} users' train sentences ...")
        # Load raw sentences
        user_to_texts = {u: [] for u in phase10_users}
        with RAW_SENTENCES_FILE.open() as f:
            for line in f:
                obj = json.loads(line)
                uid = obj["user_id"]
                if uid not in user_to_texts:
                    continue
                if obj.get("is_holdout", False):
                    continue
                txt = obj.get("sentence_text", "").strip()
                if not txt:
                    continue
                user_to_texts[uid].append(txt)
        total_texts = sum(len(v) for v in user_to_texts.values())
        log(f"  train sentences: {total_texts}")

        flat_texts = []
        flat_uids = []
        for uid in phase10_users:
            for t in user_to_texts[uid]:
                flat_texts.append(t)
                flat_uids.append(uid)

        # Load AnnaWegmann
        import glob
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        snapshots = sorted(glob.glob(os.path.join(ANNA_SNAPSHOT_DIR, "*")))
        if not snapshots:
            raise RuntimeError(f"No AnnaWegmann snapshots found in {ANNA_SNAPSHOT_DIR}")
        snapshot = snapshots[-1]
        log(f"  using snapshot: {snapshot}")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(snapshot, device="cuda:0")
        log(f"  loaded, dim={model.get_sentence_embedding_dimension()}")

        t0 = time.time()
        sent_embs = model.encode(
            flat_texts, convert_to_numpy=True, batch_size=ENCODER_BATCH,
            show_progress_bar=False, normalize_embeddings=False
        ).astype(np.float32)
        sent_uids = flat_uids
        log(f"  encoded {len(sent_uids)} texts in {time.time()-t0:.1f}s, shape={sent_embs.shape}")

        np.savez(OUT_PER_SENT, sent_uids=np.array(sent_uids, dtype=object), sent_embs=sent_embs)
        log(f"  saved: {OUT_PER_SENT}")

    # === Per-user mean (re-compute from per-sentence) ===
    log("[3] Per-user mean pooling ...")
    user_mu = np.zeros((len(phase10_users), EMB_DIM), dtype=np.float64)
    counts = np.zeros(len(phase10_users), dtype=np.int32)
    for i, uid in enumerate(sent_uids):
        idx = uid_to_idx[uid]
        user_mu[idx] += sent_embs[i]
        counts[idx] += 1
    valid = counts > 0
    user_mu[valid] /= counts[valid, None]
    log(f"  valid users: {int(valid.sum())}/{len(phase10_users)}")
    log(f"  mean count per user: {counts[valid].mean():.1f}, min/max: {counts[valid].min()}/{counts[valid].max()}")

    # Split-half cosine on per-user mean
    log("[4] Split-half cosine check on per-user mean ...")
    rng = np.random.default_rng(RANDOM_SEED)
    cos_sims = []
    for i, uid in enumerate(phase10_users):
        if counts[i] < 4:
            continue
        idxs = [j for j, u in enumerate(sent_uids) if u == uid]
        local_embs = sent_embs[idxs]
        for _ in range(min(10, len(local_embs) // 2)):
            perm = rng.permutation(len(local_embs))
            half = len(local_embs) // 2
            a = local_embs[perm[:half]].mean(0)
            b = local_embs[perm[half:]].mean(0)
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
            cos_sims.append(cos)
    cos_arr = np.array(cos_sims)
    log(f"  split-half cos (mean): {cos_arr.mean():.4f} +/- {cos_arr.std():.4f} (n={len(cos_arr)})")

    # === PCA 768 -> 128 ===
    log(f"[5] PCA {EMB_DIM} -> {PCA_DIM} on user_mu ...")
    valid_user_mu = user_mu[valid]
    X = valid_user_mu - valid_user_mu.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    pca_components = Vt[:PCA_DIM].astype(np.float32)  # [PCA_DIM, EMB_DIM]
    pca_mean = valid_user_mu.mean(axis=0).astype(np.float32)  # [EMB_DIM]
    explained_var_ratio = (S ** 2) / (S ** 2).sum()
    pca_evr = float(explained_var_ratio[:PCA_DIM].sum())
    log(f"  PCA explained variance: {pca_evr:.4f} (top {PCA_DIM})")

    # Project user_mu and Sigma to PCA space
    user_mu_pca = (valid_user_mu - pca_mean) @ pca_components.T  # [n_valid, PCA_DIM]
    user_mu_pca = user_mu_pca.astype(np.float32)
    log(f"  user_mu_pca: {user_mu_pca.shape}")

    # === Per-user Sigma with LW shrinkage (full 768d) ===
    log("[6] Per-user Sigma (768d) with Ledoit-Wolf shrinkage ...")
    # Build per-user sentence index
    user_to_sent_idxs: dict[int, list[int]] = {}
    for i, uid in enumerate(sent_uids):
        idx = uid_to_idx[uid]
        user_to_sent_idxs.setdefault(idx, []).append(i)
    log(f"  per-user sentence groups: {len(user_to_sent_idxs)}")

    sigma_diag = np.zeros((len(phase10_users), EMB_DIM), dtype=np.float32)
    sigma_lowrank = np.zeros((len(phase10_users), EMB_DIM, EMB_DIM), dtype=np.float32)  # full Cholesky
    lw_alphas = np.zeros(len(phase10_users), dtype=np.float32)
    eigval_min = np.full(len(phase10_users), np.inf, dtype=np.float32)

    from sklearn.covariance import LedoitWolf

    n_done = 0
    for uid_idx in range(len(phase10_users)):
        sidxs = user_to_sent_idxs.get(uid_idx, [])
        n_sent = len(sidxs)
        if n_sent < 2:
            # Use full identity scaled by overall variance
            sigma_diag[uid_idx] = 1.0
            sigma_lowrank[uid_idx] = np.eye(EMB_DIM, dtype=np.float32) * 0.01
            lw_alphas[uid_idx] = 1.0
            eigval_min[uid_idx] = 0.01
            continue
        # Get this user's sentence embeddings
        local_X = sent_embs[sidxs].astype(np.float64)  # [n_sent, EMB_DIM]
        local_mean = local_X.mean(axis=0, keepdims=True)
        local_centered = local_X - local_mean
        # LW shrinkage on this user's centered data
        lw = LedoitWolf().fit(local_centered)
        cov = lw.covariance_.astype(np.float32)  # [EMB_DIM, EMB_DIM]
        sigma_diag[uid_idx] = np.diag(cov)
        # Cholesky for sampling
        cov_pd = cov + max(LW_SHRINKAGE_MIN, 1e-5) * np.eye(EMB_DIM, dtype=np.float32)
        try:
            L = np.linalg.cholesky(cov_pd).astype(np.float32)  # [EMB_DIM, EMB_DIM]
            sigma_lowrank[uid_idx] = L
        except np.linalg.LinAlgError:
            # Fall back to diagonal sqrt
            sqrt_diag = np.sqrt(np.maximum(sigma_diag[uid_idx], 1e-6))
            sigma_lowrank[uid_idx] = np.diag(sqrt_diag).astype(np.float32)
        lw_alphas[uid_idx] = float(lw.shrinkage_)
        eigvals = np.linalg.eigvalsh(cov)
        eigval_min[uid_idx] = float(eigvals.min())
        n_done += 1
        if n_done % 100 == 0:
            log(f"  computed {n_done}/{int(valid.sum())} Sigma")

    log(f"  LW shrinkage alpha (mean/median/max): {lw_alphas[lw_alphas > 0].mean():.4f} / "
        f"{np.median(lw_alphas[lw_alphas > 0]):.4f} / {lw_alphas.max():.4f}")
    log(f"  Eigenvalue min (median/min): {np.median(eigval_min[eigval_min < np.inf]):.4e} / "
        f"{eigval_min[eigval_min < np.inf].min():.4e}")

    # Validate PD
    if eigval_min[eigval_min < np.inf].min() < 1e-6:
        bad = (eigval_min < 1e-6) & (eigval_min < np.inf)
        log(f"  WARNING: {bad.sum()} users have min eigenvalue < 1e-6")

    # === Validation: per-user held-out sentence identification ===
    log("[7] Validation: per-user held-out sentence identification (768d AnnaWegmann) ...")
    # For each user, hold out 1 sentence and check if its 768d embedding is closest to mu_u
    # than to all other users' mu
    valid_idx_set = np.where(valid)[0]
    valid_to_vidx = {orig: i for i, orig in enumerate(valid_idx_set)}

    # Group sentence indices by user
    user_to_sent_idxs = {}
    for i, uid in enumerate(sent_uids):
        idx = uid_to_idx[uid]
        user_to_sent_idxs.setdefault(idx, []).append(i)

    rng = np.random.default_rng(RANDOM_SEED)
    heldout_pairs = []
    for orig_idx in valid_idx_set:
        sidxs = user_to_sent_idxs.get(orig_idx, [])
        if len(sidxs) < 4:
            continue
        # hold out 1 sentence randomly
        held_idx = int(rng.choice(sidxs))
        heldout_pairs.append((orig_idx, held_idx))
    log(f"  heldout pairs: {len(heldout_pairs)}")

    # Pooled covariance on user_mu (point vectors)
    lw_pool = LedoitWolf().fit(valid_user_mu)
    cov_pool_768 = lw_pool.covariance_.astype(np.float64) + 1e-6 * np.eye(EMB_DIM)
    inv_cov_pool_768 = np.linalg.inv(cov_pool_768)

    # For each held-out query, compute D(query, mu_v) for all v and rank target
    target_ranks = []
    for orig_idx, held_idx in heldout_pairs[:N_MAHA_VALIDATION_PAIRS]:
        q = sent_embs[held_idx].astype(np.float64)
        # D(q, mu_v) = (q - mu_v)^T inv_cov (q - mu_v)
        # = q^T inv_cov q - 2 q^T inv_cov mu_v + mu_v^T inv_cov mu_v
        quad_q = q @ inv_cov_pool_768 @ q
        cross_q = valid_user_mu @ inv_cov_pool_768 @ q  # [n_valid]
        quad_mu = np.einsum("ij,jk,ik->i", valid_user_mu, inv_cov_pool_768, valid_user_mu)
        dists = quad_q - 2 * cross_q + quad_mu  # [n_valid]
        v_idx = valid_to_vidx[orig_idx]
        # rank: how many other users have smaller distance than target
        rank = int((dists < dists[v_idx]).sum() + 1)
        target_ranks.append(rank)
    target_ranks = np.array(target_ranks)
    n_valid = len(valid_user_mu)

    rank1_cov = float((target_ranks == 1).mean())
    top10_cov = float((target_ranks <= 10).mean())
    top100_cov = float((target_ranks <= 100).mean())
    log(f"  target rank distribution: rank-1={rank1_cov:.4f}, top-10={top10_cov:.4f}, top-100={top100_cov:.4f}")
    log(f"  mean rank: {target_ranks.mean():.1f}, median: {np.median(target_ranks):.1f} "
        f"(n_valid={n_valid}, random baseline ~{n_valid/2:.1f})")

    # Compare to Phase 10.15 baseline (rank-1 1.26% on full 876 pairs, mean_best_rank 307.76)
    phase10_15_eval = json.loads(PHASE10_RANK1_EVAL.read_text())
    base_rank1 = float(phase10_15_eval.get("coverage_any_rank1", 0))
    base_mean_rank = float(phase10_15_eval.get("mean_best_rank", 0))

    validation = {
        "n_pairs": int(len(target_ranks)),
        "rank1_coverage": rank1_cov,
        "top10_coverage": top10_cov,
        "top100_coverage": top100_cov,
        "mean_rank": float(target_ranks.mean()),
        "median_rank": float(np.median(target_ranks)),
        "n_valid_users": n_valid,
        "random_baseline_mean_rank": float(n_valid / 2),
        "phase10_15_baseline": {
            "rank1_coverage": base_rank1,
            "mean_best_rank": base_mean_rank,
        },
        "signal_meaningful": bool(target_ranks.mean() < n_valid / 4),
    }

    # === Save Gaussian cache ===
    log(f"[8] Saving Gaussian cache to {OUT_GAUSSIANS} ...")
    np.savez(
        OUT_GAUSSIANS,
        user_ids=np.array(phase10_users, dtype=object),
        mu_768=user_mu.astype(np.float32),
        mu_128=user_mu_pca.astype(np.float32),
        sigma_diag=sigma_diag,
        sigma_lowrank=sigma_lowrank,  # Cholesky factors [n_users, 768, 768]
        lw_alpha=lw_alphas,
        eigval_min=eigval_min,
        pca_components=pca_components,
        pca_mean=pca_mean,
        valid=valid,
    )
    log(f"  saved: mu_768 {user_mu.shape}, mu_128 {user_mu_pca.shape}, sigma_diag {sigma_diag.shape}, sigma_lowrank {sigma_lowrank.shape}")

    # Save validation
    OUT_VALIDATION.write_text(json.dumps(validation, indent=2))
    log(f"  validation: {OUT_VALIDATION}")

    # === Meta ===
    meta = {
        "phase": "11.A",
        "encoder": "AnnaWegmann/Style-Embedding",
        "n_users": len(phase10_users),
        "n_valid_users": int(valid.sum()),
        "emb_dim": EMB_DIM,
        "pca_dim": PCA_DIM,
        "pca_explained_variance_ratio": pca_evr,
        "n_sentences_used": len(sent_uids),
        "mean_count_per_user": float(counts[valid].mean()),
        "lw_alpha_mean": float(lw_alphas[lw_alphas > 0].mean()) if (lw_alphas > 0).any() else None,
        "lw_alpha_max": float(lw_alphas.max()),
        "eigval_min_median": float(np.median(eigval_min[eigval_min < np.inf])) if (eigval_min < np.inf).any() else None,
        "split_half_cos_mean": float(cos_arr.mean()) if len(cos_arr) else None,
        "split_half_cos_std": float(cos_arr.std()) if len(cos_arr) else None,
        "validation": validation,
    }
    OUT_GAUSSIANS_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log(f"  meta: {OUT_GAUSSIANS_META}")

    log("=" * 70)
    log("PHASE 11.A COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()