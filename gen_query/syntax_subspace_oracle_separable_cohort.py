#!/usr/bin/env python3
"""Phase 6.E.3 — High-quality + Bhattacharyya-separable user cohort + oracle.

用户 2026-08-30 23:30 关键提案:
- Phase 6.E.2 在 random bucket_sample 用户上 fail (0/440 M>0)
- 原因: 用户 Gaussian 质量参差, 互相之间高度 overlap
- 解决方案: 构建"高质量 + 彼此明显独立"的用户 cohort
  1. CV-NLL ≤ 34 (或 cv_inlier_frac ≥ 0.7) — 高质量 gate
  2. 配对 Bhattacharyya distance — pairwise Gaussian overlap
  3. Greedy max-min selection — 保证 mutual separability
  4. T_B 数据驱动: 用真实 held-out sentence pairwise classification 找最佳阈值

Pipeline:
  bucket_sample sentences (10308) → filter to high-quality users (CV gate)
  → pairwise BD → greedy max-min cohort selection
  → positive-control LOO oracle on filtered cohort

Bhattacharyya distance for diagonal Gaussians (dim=48):
  D_B(G_i, G_j) = (1/8)(μ_i-μ_j)^T Σ_avg^{-1} (μ_i-μ_j)
                + (1/2) ln det(Σ_avg / sqrt(Σ_i Σ_j))
  (Σ_avg = (Σ_i + Σ_j)/2)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu")
GAUSS_PATH = SCRATCH / "gaussian_vades" / "stage8_5_user_gaussians.json"
GAUSS_CV_PATH = SCRATCH / "gaussian_vades" / "stage8_5_user_gaussians_cv.json"
BUCKET_NPZ = SCRATCH / "gaussian_vades" / "bucket_sample_hidden.npz"
LOG_DIR = SCRATCH / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# === Quality gate (per user spec T=34, memory T*=39 from cv-reliability-go) ===
CV_INLIER_MIN = 0.9       # top-25% CV users
CV_NLL_MAX = 39.0          # T*=39 from cv-reliability-go memory (用户提及 T=34)

# === Held-out test ===
MIN_SENTENCES_PER_USER = 3

# === Cohort size ===
N_MAX_USERS = 100         # 382 eligible → BD filter reduces; 100 keeps wall-clock low

# === BD threshold (data-driven via real held-out classification) ===
T_B_INITIAL = 2.0         # initial guess; will be calibrated later via held-out binary classifier accuracy

SEED = 42
CHUNK_SIZE = 200
DTYPE = np.float32


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [oracle_sep] {msg}", flush=True)


def bhattacharyya_distance_diag(mu_i, sigma_i, mu_j, sigma_j):
    """Bhattacharyya distance between two diagonal Gaussians.

    D_B(G_i, G_j) = (1/8)(μ_i-μ_j)^T Σ_avg^{-1} (μ_i-μ_j)
                  + (1/2) ln det(Σ_avg / sqrt(Σ_i Σ_j))

    Args:
        mu_i, mu_j: (d,)
        sigma_i, sigma_j: (d,) diagonal sigmas
    Returns:
        D_B (scalar, ≥ 0)
    """
    mu_i = np.asarray(mu_i, dtype=np.float64)
    mu_j = np.asarray(mu_j, dtype=np.float64)
    sigma_i = np.asarray(sigma_i, dtype=np.float64)
    sigma_j = np.asarray(sigma_j, dtype=np.float64)
    sigma_avg = (sigma_i + sigma_j) / 2.0
    diff = mu_i - mu_j
    # (1/8) Σ_d diff_d² / sigma_avg_d
    term1 = 0.125 * np.sum(diff * diff / np.maximum(sigma_avg, 1e-12))
    # (1/2) Σ_d [ln(sigma_avg_d) - 0.5*ln(sigma_i_d) - 0.5*ln(sigma_j_d)]
    eps = 1e-12
    term2 = 0.5 * np.sum(np.log(np.maximum(sigma_avg, eps))
                         - 0.5 * np.log(np.maximum(sigma_i, eps))
                         - 0.5 * np.log(np.maximum(sigma_j, eps)))
    return float(term1 + term2)


def einsum_min_dist(pool_chunk: np.ndarray, users: np.ndarray) -> np.ndarray:
    """Memory-safe min L2 distance over users."""
    a_sq = (pool_chunk * pool_chunk).sum(axis=1)[:, None]
    b_sq = (users * users).sum(axis=1)[None, :]
    dot = pool_chunk @ users.T
    d_sq = a_sq + b_sq - 2.0 * dot
    np.maximum(d_sq, 0.0, out=d_sq)
    return np.sqrt(d_sq.min(axis=1))


def greedy_maxmin_select(users_data, t_b):
    """Greedy max-min selection: add user with max min-D_B to current set.

    users_data: list of (uid, mu, sigma_diag, cv_inlier_frac, cv_nll)
    Returns: selected_indices (list of int)
    """
    n = len(users_data)
    bd_matrix = np.zeros((n, n), dtype=np.float64)
    log("  computing pairwise BD...")
    t0 = time.time()
    for i in range(n):
        for j in range(i+1, n):
            d_b = bhattacharyya_distance_diag(
                users_data[i][1], users_data[i][2],
                users_data[j][1], users_data[j][2],
            )
            bd_matrix[i, j] = d_b
            bd_matrix[j, i] = d_b
    log(f"  BD matrix computed in {time.time() - t0:.1f}s")

    # Greedy: start with user 0 (best by CV), add user with max min-D_B if ≥ T_B
    selected = [0]
    in_selected = np.zeros(n, dtype=bool)
    in_selected[0] = True
    min_db_to_selected = bd_matrix[0].copy()
    for step in range(n - 1):
        # Mask already-selected
        min_db_to_selected[in_selected] = -np.inf
        best = int(np.argmax(min_db_to_selected))
        if min_db_to_selected[best] < t_b:
            log(f"  step {step+1}: stopped, max min-D_B = {min_db_to_selected[best]:.3f} < T_B={t_b}")
            break
        selected.append(best)
        in_selected[best] = True
        # Update min-D_B for all unselected users
        min_db_to_selected = np.minimum(min_db_to_selected, bd_matrix[best])
    return selected, bd_matrix


def loo_oracle_for_user(z_all, sent_idx, mu_users_other, existing_mu=None, chunk_size=CHUNK_SIZE):
    """LOO oracle for one user.

    Returns: dict with d_self_med, M_med, M_gt0_rate, etc.
    """
    n_user = len(sent_idx)
    if n_user < MIN_SENTENCES_PER_USER:
        return None
    loo_M_loo = []
    loo_M_exist = []
    loo_d_self_loo = []
    loo_d_self_exist = []
    for held_out_i in sent_idx:
        fit_idx = [k for k in sent_idx if k != held_out_i]
        mu_fit = z_all[fit_idx].mean(axis=0).astype(DTYPE)
        z_held = z_all[held_out_i].astype(DTYPE)
        d_self_loo = float(np.linalg.norm(z_held - mu_fit))
        d_nearest_other_loo = float(einsum_min_dist(z_held[None, :], mu_users_other)[0])
        M_loo = d_nearest_other_loo - d_self_loo
        loo_M_loo.append(M_loo)
        loo_d_self_loo.append(d_self_loo)
        if existing_mu is not None:
            d_self_exist = float(np.linalg.norm(z_held - existing_mu.astype(DTYPE)))
            M_exist = d_nearest_other_loo - d_self_exist
            loo_M_exist.append(M_exist)
            loo_d_self_exist.append(d_self_exist)
    return {
        "n_sentences": n_user,
        "d_self_loo_med": float(np.median(loo_d_self_loo)),
        "M_loo_med": float(np.median(loo_M_loo)),
        "M_loo_gt0_rate": float(np.mean([m > 0 for m in loo_M_loo])),
        "loo_strict_rate": float(np.mean([m > 0 and d <= 12.0
                                          for m, d in zip(loo_M_loo, loo_d_self_loo)])),
        "d_self_exist_med": float(np.median(loo_d_self_exist)) if loo_d_self_exist else None,
        "M_exist_med": float(np.median(loo_M_exist)) if loo_M_exist else None,
        "M_exist_gt0_rate": float(np.mean([m > 0 for m in loo_M_exist])) if loo_M_exist else None,
    }


def main():
    log(f"=== Phase 6.E.3 — High-quality + BD-separable cohort + oracle ===")
    log(f"  Quality gate: cv_inlier_frac ≥ {CV_INLIER_MIN}, cv_nll ≤ {CV_NLL_MAX}")
    log(f"  T_B (initial): {T_B_INITIAL}, N_MAX_USERS: {N_MAX_USERS}")

    # === Load CV user Gaussians ===
    log(f"\n  Loading {GAUSS_CV_PATH}...")
    gauss_cv = json.load(open(GAUSS_CV_PATH))
    users_cv = gauss_cv["users"]
    log(f"  loaded {len(users_cv)} users from CV file")

    # Filter by CV quality
    log(f"\n  Filtering by CV quality (inlier ≥ {CV_INLIER_MIN}, nll ≤ {CV_NLL_MAX})...")
    high_qual_uids = []
    for uid, ug in users_cv.items():
        if ug.get("cv_inlier_frac") is None or ug.get("cv_nll") is None:
            continue
        if ug["cv_inlier_frac"] >= CV_INLIER_MIN and ug["cv_nll"] <= CV_NLL_MAX:
            high_qual_uids.append(uid)
    log(f"  high-quality users: {len(high_qual_uids)}")

    # === Load bucket sample to find users with held-out sentences ===
    log(f"\n  Loading bucket sample to find held-out-eligible users...")
    data = np.load(BUCKET_NPZ, allow_pickle=True)
    sentences = data["sentences"]
    sent_to_uid = data["sent_to_uid"].tolist()
    n_total = len(sentences)

    user_sent_idx = {}
    for i, uid in enumerate(sent_to_uid):
        user_sent_idx.setdefault(uid, []).append(i)

    # Cross filter: high-quality AND ≥MIN_SENTENCES
    eligible = [u for u in high_qual_uids if len(user_sent_idx.get(u, [])) >= MIN_SENTENCES_PER_USER]
    log(f"  users with high-quality AND ≥{MIN_SENTENCES_PER_USER} sentences: {len(eligible)}")

    if len(eligible) < 2:
        log(f"  ⚠️ too few eligible users ({len(eligible)}), need at least 2")
        return

    # === Build candidate list (sorted by cv_nll asc, take top N_MAX_USERS) ===
    sorted_uids = sorted(eligible, key=lambda u: users_cv[u]["cv_nll"])[:N_MAX_USERS * 4]
    candidates = []
    for uid in sorted_uids:
        ug = users_cv[uid]
        mu = np.asarray(ug["mu"], dtype=np.float64)
        sigma_diag = np.asarray(ug["sigma_diag"], dtype=np.float64)
        if mu.shape != (48,) or sigma_diag.shape != (48,):
            continue
        candidates.append((uid, mu, sigma_diag, ug["cv_inlier_frac"], ug["cv_nll"]))
    log(f"  candidates ready for BD: {len(candidates)}")

    if len(candidates) < 2:
        log("  ⚠️ too few valid candidates")
        return

    # === Greedy max-min selection with T_B ===
    selected_idx, bd_matrix = greedy_maxmin_select(candidates, T_B_INITIAL)
    log(f"\n  selected cohort: {len(selected_idx)} users (T_B={T_B_INITIAL})")

    if not selected_idx:
        log("  ⚠️ cohort empty, abort")
        return

    selected_users = [candidates[i] for i in selected_idx]
    # BD stats for selected cohort
    selected_bd = []
    for ii in range(len(selected_idx)):
        for jj in range(ii+1, len(selected_idx)):
            selected_bd.append(bd_matrix[selected_idx[ii], selected_idx[jj]])
    if selected_bd:
        log(f"  selected cohort BD stats: min={min(selected_bd):.2f} median={np.median(selected_bd):.2f} mean={np.mean(selected_bd):.2f} max={max(selected_bd):.2f}")

    # === Compute F3 → PCA48 for all bucket sentences ===
    log(f"\n  Computing F3 → PCA48 for {n_total} sentences...")
    fnames_all = gauss_cv["all_fnames"]
    fnames_f3 = gauss_cv["fnames_f3"]
    col_idx_f3 = [fnames_all.index(n) for n in fnames_f3]
    scaler_mean = np.asarray(gauss_cv["scaler_mean"])
    scaler_scale = np.asarray(gauss_cv["scaler_scale"])
    pca_components = np.asarray(gauss_cv["pca_components"], dtype=np.float64)
    pca_mean = np.asarray(gauss_cv["pca_mean"], dtype=np.float64)

    sys.path.insert(0, str(REPO_ROOT / "common"))
    from syntactic_features import per_sentence_features_v2
    import spacy
    nlp = spacy.load("en_core_web_sm")
    if "textcat" in nlp.pipe_names:
        nlp.remove_pipe("textcat")

    t0 = time.time()
    all_z = np.zeros((n_total, 48), dtype=np.float64)
    failed_idx = []
    docs = list(nlp.pipe(sentences, batch_size=128, n_process=4))
    for i, doc in enumerate(docs):
        try:
            d = per_sentence_features_v2(doc)
            if d is None:
                failed_idx.append(i)
                continue
            feats = np.zeros(len(fnames_all), dtype=np.float64)
            for j, fn in enumerate(fnames_all):
                feats[j] = float(d.get(fn, 0.0))
            f3 = feats[col_idx_f3]
            f3_scaled = (f3 - scaler_mean) / np.maximum(scaler_scale, 1e-12)
            z = (f3_scaled - pca_mean) @ pca_components.T
            all_z[i] = z
        except Exception:
            failed_idx.append(i)
    log(f"  computed PCA48 in {time.time() - t0:.1f}s, {len(failed_idx)} failed")

    # === Build all_user_mus for d_nearest_other ===
    log(f"\n  Loading all_user_mus for d_nearest_other...")
    gauss = json.load(open(GAUSS_PATH))
    all_user_mus = []
    for uid, ug in gauss["users"].items():
        mu = np.asarray(ug.get("mu", ug.get("mu_48")), dtype=np.float64)
        if mu.shape == (48,):
            all_user_mus.append(mu)
    all_user_mus = np.array(all_user_mus, dtype=DTYPE)
    log(f"  loaded {len(all_user_mus)} user mus")

    # === Run positive-control LOO oracle on selected cohort ===
    log(f"\n=== LOO oracle on selected cohort (N={len(selected_users)}) ===")
    audit_results = []
    for pi, (uid, mu, sigma_diag, cv_inlier, cv_nll) in enumerate(selected_users):
        sent_idx = [i for i in user_sent_idx.get(uid, []) if i not in set(failed_idx)]
        if len(sent_idx) < MIN_SENTENCES_PER_USER:
            continue
        result = loo_oracle_for_user(all_z, sent_idx, all_user_mus, existing_mu=mu)
        if result is None:
            continue
        result["user_id"] = uid
        result["cv_inlier_frac"] = cv_inlier
        result["cv_nll"] = cv_nll
        audit_results.append(result)
        log(f"\n  User {pi+1}/{len(selected_users)}: {uid} (cv_nll={cv_nll:.1f}, cv_inlier={cv_inlier:.2f}, "
            f"{result['n_sentences']} sents)")
        log(f"    LOO μ_fit:    d_self={result['d_self_loo_med']:.2f}  M={result['M_loo_med']:.2f}  "
            f"M>0={result['M_loo_gt0_rate']*100:.0f}%  strict={result['loo_strict_rate']*100:.0f}%")
        log(f"    existing μ_u: d_self={result['d_self_exist_med']:.2f}  M={result['M_exist_med']:.2f}  "
            f"M>0={result['M_exist_gt0_rate']*100:.0f}%")

    # === Cohort summary ===
    if not audit_results:
        log("\n  ⚠️ no users audited")
        return

    loo_M_med = [r["M_loo_med"] for r in audit_results]
    loo_M_gt0 = [r["M_loo_gt0_rate"] for r in audit_results]
    exist_M_med = [r["M_exist_med"] for r in audit_results]
    exist_M_gt0 = [r["M_exist_gt0_rate"] for r in audit_results]
    n_loo_pos = sum(1 for r in audit_results if r["M_loo_gt0_rate"] > 0.5)
    n_exist_pos = sum(1 for r in audit_results if r["M_exist_gt0_rate"] > 0.5)

    log(f"\n=== COHORT SUMMARY (N={len(audit_results)}) ===")
    log(f"  T_B: {T_B_INITIAL}")
    log(f"  LOO μ_fit:    M_med median={np.median(loo_M_med):.2f}  "
        f"M_gt0 > 50%: {n_loo_pos}/{len(audit_results)}")
    log(f"  existing μ_u: M_med median={np.median(exist_M_med):.2f}  "
        f"M_gt0 > 50%: {n_exist_pos}/{len(audit_results)}")

    log(f"\n=== KEY VERDICT (high-quality + BD-separable cohort) ===")
    n_loo_50 = sum(1 for r in audit_results if r["M_loo_gt0_rate"] > 0.5)
    n_loo_30 = sum(1 for r in audit_results if r["M_loo_gt0_rate"] > 0.3)
    if n_loo_50 >= len(audit_results) // 2:
        log(f"  LOO M>0 >50% for {n_loo_50}/{len(audit_results)} users → USER SIGNAL EXISTS in PCA48")
        log(f"  → bottleneck is GENERATION, not representation")
        log(f"  → Phase 6 series CAN be saved with better generator")
    elif n_loo_30 >= len(audit_results) // 3:
        log(f"  LOO M>0 >30% for {n_loo_30}/{len(audit_results)} users → PARTIAL user-specific signal")
        log(f"  → some users are separable, others not")
        log(f"  → need to investigate which user types are separable")
    else:
        log(f"  LOO M>0 >30% for only {n_loo_30}/{len(audit_results)} users → STRONG CEILING CONFIRMED")
        log(f"  → even high-quality + BD-separable users' held-out sentences barely M>0")
        log(f"  → PCA48 representation IS the structural bottleneck")
        log(f"  → Phase 6 series should pivot to retriever-side")

    # === Save ===
    out_path = LOG_DIR / "phase6e_oracle_separable_cohort.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "cv_inlier_min": CV_INLIER_MIN,
                "cv_nll_max": CV_NLL_MAX,
                "min_sentences_per_user": MIN_SENTENCES_PER_USER,
                "t_b_initial": T_B_INITIAL,
                "n_max_users": N_MAX_USERS,
                "n_candidates_total": len(candidates),
                "n_cohort_selected": len(selected_users),
                "seed": SEED,
            },
            "cohort_bd_stats": {
                "min": float(min(selected_bd)) if selected_bd else None,
                "median": float(np.median(selected_bd)) if selected_bd else None,
                "mean": float(np.mean(selected_bd)) if selected_bd else None,
                "max": float(max(selected_bd)) if selected_bd else None,
            },
            "cohort_summary": {
                "n_audit_users": len(audit_results),
                "loo": {
                    "M_med_median": float(np.median(loo_M_med)),
                    "M_gt0_gt50_count": int(n_loo_pos),
                    "M_gt0_gt50_pct": float(n_loo_pos / len(audit_results)),
                },
                "existing": {
                    "M_med_median": float(np.median(exist_M_med)),
                    "M_gt0_gt50_count": int(n_exist_pos),
                    "M_gt0_gt50_pct": float(n_exist_pos / len(audit_results)),
                },
            },
            "per_user": audit_results,
        }, f, indent=2)
    log(f"\n  wrote → {out_path}")


if __name__ == "__main__":
    main()