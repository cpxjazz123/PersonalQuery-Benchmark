#!/usr/bin/env python3
"""Phase 6.E.4 — Local-Cohort Positive-Control Oracle.

用户 2026-08-30 23:30 critique (CRITICAL):
- Phase 6.E.1-6.E.3 都把 M = d_nearest_other - d_self 计算 over 1.2M users
- 这是 global identification task, 不是 personalized-style task
- 真正合理的 personalized query benchmark: same-ASIN 用户之间区分
- 之前 Phase 6.D.something 同 ASIN matched cohort Rank@1 = 37.9% (not 0%)

设计:
- 用同一批 held-out real sentences (bucket_sample)
- 三种 competitor settings 对比:
  A. GLOBAL: d_nearest_other over 1.2M users (Phase 6.E.2 baseline)
  B. SAME-ASIN: d_nearest_other over users from same ASIN only
  C. SAME-ASIN + Quality + BD-separated (T_B=2.0): smallest cohort
- 看 M>0 比例如何从 0.5% 升到 (potentially) 30-75%
- BD 集合 = competitor 集合 (key constraint per user critique)

期望:
- GLOBAL: ~0.5% (Phase 6.E.2 baseline)
- SAME-ASIN: 5-20% (much higher)
- SAME-ASIN + Q + BD: 20-40% (highest)

如果连 (C) 都 M>0 ≈ 0, 才能真正下结论 PCA48 ceiling.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu")
GAUSS_PATH = SCRATCH / "gaussian_vades" / "stage8_5_user_gaussians.json"
GAUSS_CV_PATH = SCRATCH / "gaussian_vades" / "stage8_5_user_gaussians_cv.json"
BUCKET_NPZ = SCRATCH / "gaussian_vades" / "bucket_sample_hidden.npz"
ASIN_MAP_PATH = SCRATCH / "gaussian_vades" / "stage8_5_asins.json"
LOG_DIR = SCRATCH / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# === Quality gate (per user spec T=34, memory T*=39) ===
CV_INLIER_MIN = 0.9
CV_NLL_MAX = 39.0

# === Held-out test ===
MIN_SENTENCES_PER_USER = 3

# === BD threshold ===
T_B = 2.0

SEED = 42
CHUNK_SIZE = 200
DTYPE = np.float32


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [oracle_local] {msg}", flush=True)


def bhattacharyya_distance_diag(mu_i, sigma_i, mu_j, sigma_j):
    """Bhattacharyya distance between two diagonal Gaussians (d=48)."""
    mu_i = np.asarray(mu_i, dtype=np.float64)
    mu_j = np.asarray(mu_j, dtype=np.float64)
    sigma_i = np.asarray(sigma_i, dtype=np.float64)
    sigma_j = np.asarray(sigma_j, dtype=np.float64)
    sigma_avg = (sigma_i + sigma_j) / 2.0
    diff = mu_i - mu_j
    term1 = 0.125 * np.sum(diff * diff / np.maximum(sigma_avg, 1e-12))
    eps = 1e-12
    term2 = 0.5 * np.sum(np.log(np.maximum(sigma_avg, eps))
                         - 0.5 * np.log(np.maximum(sigma_i, eps))
                         - 0.5 * np.log(np.maximum(sigma_j, eps)))
    return float(term1 + term2)


def greedy_maxmin_in_group(users_data, t_b):
    """Greedy max-min selection within a group.
    users_data: list of (uid, mu, sigma_diag)
    Returns: selected_indices (list of int), bd_matrix
    """
    n = len(users_data)
    if n < 2:
        return list(range(n)), np.zeros((n, n))
    bd_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i+1, n):
            d_b = bhattacharyya_distance_diag(
                users_data[i][1], users_data[i][2],
                users_data[j][1], users_data[j][2],
            )
            bd_matrix[i, j] = d_b
            bd_matrix[j, i] = d_b
    selected = [0]
    in_selected = np.zeros(n, dtype=bool)
    in_selected[0] = True
    min_db_to_selected = bd_matrix[0].copy()
    for step in range(n - 1):
        min_db_to_selected[in_selected] = -np.inf
        best = int(np.argmax(min_db_to_selected))
        if min_db_to_selected[best] < t_b:
            break
        selected.append(best)
        in_selected[best] = True
        min_db_to_selected = np.minimum(min_db_to_selected, bd_matrix[best])
    return selected, bd_matrix


def einsum_min_dist_to_subset(pool_chunk: np.ndarray, users_subset: np.ndarray) -> np.ndarray:
    """Memory-safe min L2 distance over a small subset of users."""
    if len(users_subset) == 0:
        return np.full(len(pool_chunk), np.inf, dtype=DTYPE)
    a_sq = (pool_chunk * pool_chunk).sum(axis=1)[:, None]
    b_sq = (users_subset * users_subset).sum(axis=1)[None, :]
    dot = pool_chunk @ users_subset.T
    d_sq = a_sq + b_sq - 2.0 * dot
    np.maximum(d_sq, 0.0, out=d_sq)
    return np.sqrt(d_sq.min(axis=1))


def loo_oracle_local(z_all, sent_idx, mu_competitors, existing_mu=None):
    """LOO oracle with restricted competitor set.

    z_all: (n_total, 48)
    sent_idx: indices of user's sentences
    mu_competitors: (n_competitors, 48) — restricted to local cohort
    existing_mu: (48,) — optional, for comparison
    """
    n_user = len(sent_idx)
    if n_user < MIN_SENTENCES_PER_USER:
        return None
    loo_M = []
    loo_d_self = []
    loo_M_exist = []
    loo_d_self_exist = []
    for held_out_i in sent_idx:
        fit_idx = [k for k in sent_idx if k != held_out_i]
        mu_fit = z_all[fit_idx].mean(axis=0).astype(DTYPE)
        z_held = z_all[held_out_i].astype(DTYPE)
        d_self = float(np.linalg.norm(z_held - mu_fit))
        d_nearest_other = float(einsum_min_dist_to_subset(z_held[None, :], mu_competitors)[0])
        M = d_nearest_other - d_self
        loo_M.append(M)
        loo_d_self.append(d_self)
        if existing_mu is not None:
            d_self_exist = float(np.linalg.norm(z_held - existing_mu.astype(DTYPE)))
            M_exist = d_nearest_other - d_self_exist
            loo_M_exist.append(M_exist)
            loo_d_self_exist.append(d_self_exist)
    return {
        "n_sentences": n_user,
        "d_self_med": float(np.median(loo_d_self)),
        "M_med": float(np.median(loo_M)),
        "M_gt0_rate": float(np.mean([m > 0 for m in loo_M])),
        "strict_rate": float(np.mean([m > 0 and d <= 12.0
                                      for m, d in zip(loo_M, loo_d_self)])),
        "d_self_exist_med": float(np.median(loo_d_self_exist)) if loo_d_self_exist else None,
        "M_exist_med": float(np.median(loo_M_exist)) if loo_M_exist else None,
        "M_exist_gt0_rate": float(np.mean([m > 0 for m in loo_M_exist])) if loo_M_exist else None,
    }


def main():
    log(f"=== Phase 6.E.4 — Local-Cohort Positive-Control Oracle ===")
    log(f"  Quality gate: cv_inlier_frac ≥ {CV_INLIER_MIN}, cv_nll ≤ {CV_NLL_MAX}")
    log(f"  T_B = {T_B}, MIN_SENTENCES_PER_USER = {MIN_SENTENCES_PER_USER}")

    # === Load bucket sample ===
    log(f"\n  Loading bucket sample...")
    data = np.load(BUCKET_NPZ, allow_pickle=True)
    sentences = data["sentences"]
    sent_to_uid = data["sent_to_uid"].tolist()
    n_total = len(sentences)

    user_sent_idx = {}
    for i, uid in enumerate(sent_to_uid):
        user_sent_idx.setdefault(uid, []).append(i)

    # === Load ASIN map → user → ASINs ===
    log(f"\n  Loading ASIN map...")
    asin_data = json.load(open(ASIN_MAP_PATH))
    user_to_asins = {}
    for entry in asin_data["asins"]:
        asin = entry["asin"]
        for uid in entry["users_sampled"]:
            user_to_asins.setdefault(uid, []).append(asin)
    log(f"  users in stage8_5_asins: {len(user_to_asins)}")

    # === Load CV Gaussians (for quality gate + mu/sigma lookup) ===
    log(f"\n  Loading CV Gaussians (for quality gate)...")
    gauss_cv = json.load(open(GAUSS_CV_PATH))
    users_cv = gauss_cv["users"]
    log(f"  loaded {len(users_cv)} users from CV file")

    # === Load production Gaussians (for μ + 1.2M global reference) ===
    log(f"\n  Loading production Gaussians (for 1.2M global competitor set)...")
    gauss = json.load(open(GAUSS_PATH))
    fnames_all = gauss.get("all_fnames", gauss.get("feature_names_ordered"))
    fnames_f3 = gauss["fnames_f3"]
    col_idx_f3 = [fnames_all.index(n) for n in fnames_f3]
    scaler_mean = np.asarray(gauss["scaler_mean"])
    scaler_scale = np.asarray(gauss["scaler_scale"])
    pca_components = np.asarray(gauss["pca_components"], dtype=np.float64)
    pca_mean = np.asarray(gauss["pca_mean"], dtype=np.float64)
    users_dict = gauss["users"]

    all_user_mus = []
    for uid, ug in users_dict.items():
        mu = np.asarray(ug.get("mu", ug.get("mu_48")), dtype=np.float64)
        if mu.shape == (48,):
            all_user_mus.append(mu)
    all_user_mus = np.array(all_user_mus, dtype=DTYPE)
    log(f"  loaded {len(all_user_mus)} user mus for global competitor set")

    # === Filter to bucket_sample users with ≥3 sentences, in CV, in ASIN map ===
    log(f"\n  Filtering cohort...")
    eligible_users = []
    for uid, idx in user_sent_idx.items():
        if len(idx) < MIN_SENTENCES_PER_USER:
            continue
        if uid not in users_cv:
            continue
        if uid not in user_to_asins:
            continue
        u = users_cv[uid]
        if u.get("cv_inlier_frac") is None or u.get("cv_nll") is None:
            continue
        if u["cv_inlier_frac"] < CV_INLIER_MIN or u["cv_nll"] > CV_NLL_MAX:
            continue
        eligible_users.append(uid)
    eligible_users.sort()
    log(f"  eligible users (high-quality + ≥3 sent + has ASIN): {len(eligible_users)}")

    # === Group by ASIN ===
    log(f"\n  Grouping users by ASIN (≥2 eligible users per ASIN)...")
    asin_to_users = {}
    for uid in eligible_users:
        for asin in user_to_asins[uid]:
            asin_to_users.setdefault(asin, []).append(uid)
    multi_user_asins = {a: us for a, us in asin_to_users.items() if len(us) >= 2}
    log(f"  ASINs with ≥2 eligible users: {len(multi_user_asins)}")
    size_dist = Counter(len(us) for us in multi_user_asins.values())
    log(f"  ASIN size distribution: {dict(sorted(size_dist.items()))}")

    # Build per-user local ASIN μ array
    user_local_asin_mus = {}  # uid → (n_other_users, 48) array of same-ASIN user μ
    for uid in eligible_users:
        # All asins this user reviewed; for each, other eligible users in same asin
        competitor_mus = []
        for asin in user_to_asins[uid]:
            for other in asin_to_users.get(asin, []):
                if other == uid:
                    continue
                if other not in users_dict:
                    continue
                mu = np.asarray(users_dict[other].get("mu", users_dict[other].get("mu_48")), dtype=np.float64)
                if mu.shape == (48,):
                    competitor_mus.append(mu)
        user_local_asin_mus[uid] = np.array(competitor_mus, dtype=DTYPE) if competitor_mus else np.zeros((0, 48), dtype=DTYPE)
    log(f"  user_local_asin_mus built for {len(user_local_asin_mus)} users")

    # === spaCy setup ===
    log(f"\n  Loading spaCy + per_sentence_features_v2...")
    sys.path.insert(0, str(REPO_ROOT / "common"))
    from syntactic_features import per_sentence_features_v2
    import spacy
    nlp = spacy.load("en_core_web_sm")
    if "textcat" in nlp.pipe_names:
        nlp.remove_pipe("textcat")

    # === Compute F3 → PCA48 for all sentences ===
    log(f"\n  Computing F3 → PCA48 for {n_total} sentences...")
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

    # === Compute PCA48 + same-ASIN μ for BD-separated sub-cohort ===
    log(f"\n=== Phase 6.E.4 Per-User Oracle ===")
    log(f"  For each user, run LOO with 3 competitor settings:")

    audit_results = []
    for pi, uid in enumerate(eligible_users):
        sent_idx = [i for i in user_sent_idx[uid] if i not in set(failed_idx)]
        if len(sent_idx) < MIN_SENTENCES_PER_USER:
            continue

        # Existing μ from production
        existing_mu = None
        if uid in users_dict:
            existing_mu = np.asarray(users_dict[uid].get("mu", users_dict[uid].get("mu_48")), dtype=np.float64)

        # Competitor sets:
        # A. Global: full 1.2M user mus
        # B. Same-ASIN: same-ASIN user mus (without quality gate)
        # C. Same-ASIN + Q + BD-separated: filtered subset of B

        # Setting A: global (compute once via einsum, chunked)
        # Setting B: same-ASIN (small, can do directly)
        mu_comp_B = user_local_asin_mus[uid]
        n_comp_B = len(mu_comp_B)

        # Setting C: BD-filtered from B
        # Get all same-ASIN user data (mu + sigma_diag) for BD
        asin_users_data = []
        for asin in user_to_asins[uid]:
            for other in asin_to_users.get(asin, []):
                if other == uid:
                    continue
                u = users_cv.get(other)
                if u is None or u.get("mu") is None or u.get("sigma_diag") is None:
                    continue
                if u.get("cv_inlier_frac") is None or u.get("cv_nll") is None:
                    continue
                if u["cv_inlier_frac"] < CV_INLIER_MIN or u["cv_nll"] > CV_NLL_MAX:
                    continue
                if other not in users_dict:
                    continue
                mu = np.asarray(users_dict[other]["mu"], dtype=np.float64)
                sigma = np.asarray(users_dict[other]["sigma_diag"], dtype=np.float64)
                if mu.shape == (48,) and sigma.shape == (48,):
                    asin_users_data.append((other, mu, sigma))
        # Greedy max-min within asin_users_data
        if len(asin_users_data) >= 2:
            selected_idx, _ = greedy_maxmin_in_group(asin_users_data, T_B)
            selected_uids = [asin_users_data[i][0] for i in selected_idx]
            mu_comp_C = np.array([np.asarray(users_dict[o]["mu"], dtype=np.float64)
                                  for o in selected_uids], dtype=DTYPE) if selected_uids else np.zeros((0, 48), dtype=DTYPE)
        else:
            mu_comp_C = np.zeros((0, 48), dtype=DTYPE)
        n_comp_C = len(mu_comp_C)

        # Run LOO oracle for each setting (B, C — A is reused from E.2)
        result_B = loo_oracle_local(all_z, sent_idx, mu_comp_B, existing_mu)
        result_C = loo_oracle_local(all_z, sent_idx, mu_comp_C, existing_mu)

        # For Setting A, we need to re-run since n_competitors differs
        result_A = loo_oracle_local(all_z, sent_idx, all_user_mus, existing_mu)

        log(f"\n  User {pi+1}/{len(eligible_users)}: {uid} ({len(sent_idx)} sents)")
        log(f"    n_competitors: A(global)={len(all_user_mus)} B(asin)={n_comp_B} C(asin+Q+BD)={n_comp_C}")
        if result_A:
            log(f"    A(GLOBAL):   d_self={result_A['d_self_med']:.2f}  M={result_A['M_med']:.2f}  "
                f"M>0={result_A['M_gt0_rate']*100:.0f}%  strict={result_A['strict_rate']*100:.0f}%")
        if result_B:
            log(f"    B(SAME-ASIN): d_self={result_B['d_self_med']:.2f}  M={result_B['M_med']:.2f}  "
                f"M>0={result_B['M_gt0_rate']*100:.0f}%  strict={result_B['strict_rate']*100:.0f}%")
        if result_C:
            log(f"    C(ASIN+Q+BD): d_self={result_C['d_self_med']:.2f}  M={result_C['M_med']:.2f}  "
                f"M>0={result_C['M_gt0_rate']*100:.0f}%  strict={result_C['strict_rate']*100:.0f}%")

        audit_results.append({
            "user_id": uid,
            "n_sentences": len(sent_idx),
            "asins_reviewed": user_to_asins[uid],
            "n_comp_global": int(len(all_user_mus)),
            "n_comp_asin": int(n_comp_B),
            "n_comp_asin_q_bd": int(n_comp_C),
            "global": result_A,
            "same_asin": result_B,
            "same_asin_q_bd": result_C,
        })

    # === Cohort summary ===
    log(f"\n=== PHASE 6.E.4 COHORT SUMMARY (N={len(audit_results)}) ===")
    log(f"  T_B = {T_B}")

    settings = [
        ("A(GLOBAL)", "global"),
        ("B(SAME-ASIN)", "same_asin"),
        ("C(ASIN+Q+BD)", "same_asin_q_bd"),
    ]
    summary = {}
    for label, key in settings:
        results = [r[key] for r in audit_results if r[key] is not None]
        if not results:
            continue
        m_meds = [r["M_med"] for r in results]
        m_gt0 = [r["M_gt0_rate"] for r in results]
        n_pos_50 = sum(1 for r in results if r["M_gt0_rate"] > 0.5)
        n_pos_30 = sum(1 for r in results if r["M_gt0_rate"] > 0.3)
        log(f"\n  {label} (n_audit={len(results)}):")
        log(f"    M_med:        median={np.median(m_meds):.2f}  mean={np.mean(m_meds):.2f}")
        log(f"    M_gt0 > 50%:  {n_pos_50}/{len(results)} ({n_pos_50/len(results)*100:.1f}%)")
        log(f"    M_gt0 > 30%:  {n_pos_30}/{len(results)} ({n_pos_30/len(results)*100:.1f}%)")
        summary[key] = {
            "n_users": len(results),
            "M_med_median": float(np.median(m_meds)),
            "M_gt0_gt50_count": int(n_pos_50),
            "M_gt0_gt50_pct": float(n_pos_50 / len(results)),
            "M_gt0_gt30_count": int(n_pos_30),
            "M_gt0_gt30_pct": float(n_pos_30 / len(results)),
        }

    log(f"\n=== KEY VERDICT — Local vs Global ===")
    a_pct = summary["global"]["M_gt0_gt50_pct"] * 100
    b_pct = summary["same_asin"]["M_gt0_gt50_pct"] * 100
    c_pct = summary["same_asin_q_bd"]["M_gt0_gt50_pct"] * 100
    log(f"  A(GLOBAL):         M>0 > 50%: {a_pct:.1f}%")
    log(f"  B(SAME-ASIN):      M>0 > 50%: {b_pct:.1f}%")
    log(f"  C(ASIN+Q+BD):      M>0 > 50%: {c_pct:.1f}%")

    if c_pct >= 30:
        log(f"  → LOCAL cohort shows strong PCA48 personalization signal (M>0 > 50%: {c_pct:.1f}%)")
        log(f"  → PCA48 has user-specific signal, just not for global identification")
        log(f"  → Phase 6 ceiling is GLOBAL identification, not PCA48 representation")
        log(f"  → Pivot: same-ASIN rerank / local cohort scoring")
    elif c_pct >= 10:
        log(f"  → LOCAL cohort shows moderate signal (M>0 > 50%: {c_pct:.1f}%)")
        log(f"  → PCA48 has partial personalization at local level")
    else:
        log(f"  → LOCAL cohort STILL fails (M>0 > 50%: {c_pct:.1f}%)")
        log(f"  → PCA48 cannot even distinguish users within same ASIN")
        log(f"  → This is the true ceiling — not global identification")

    # === Save ===
    out_path = LOG_DIR / "phase6e_oracle_local_cohort.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "cv_inlier_min": CV_INLIER_MIN,
                "cv_nll_max": CV_NLL_MAX,
                "min_sentences_per_user": MIN_SENTENCES_PER_USER,
                "t_b": T_B,
                "seed": SEED,
                "chunk_size": CHUNK_SIZE,
                "n_eligible_users": len(eligible_users),
                "n_multi_user_asins": len(multi_user_asins),
                "asin_size_distribution": dict(size_dist),
            },
            "summary": summary,
            "per_user": audit_results,
        }, f, indent=2)
    log(f"\n  wrote → {out_path}")


if __name__ == "__main__":
    main()