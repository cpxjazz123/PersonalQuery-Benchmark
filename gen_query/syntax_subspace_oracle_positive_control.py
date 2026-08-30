#!/usr/bin/env python3
"""Phase 6.E.2 — POSITIVE-CONTROL Oracle (target user's held-out real sentences).

用户 2026-08-30 23:30 关键 critique:
- Phase 6.E.1 REAL oracle 用 `excl self`, 实际上问的是"其他用户的真实句子能不能 serve as target user's syntax"
- 这不能证明"PCA48 是 structural ceiling"
- True ceiling test = **target user 自己的 held-out real sentences**

设计:
- Pool: bucket_sample_hidden.npz (10308 sentences from 597 users, median 10 sentences/user)
- 440 users with ≥3 sentences (held-out feasible)
- 对每个 user u, leave-one-out (LOO):
  - fit μ_fit = mean(z_k for k != i)
  - d_self = ||z_i - μ_fit||
  - d_nearest_other = min over ALL OTHER users' existing μ in stage8_5_user_gaussians.json
  - M = d_nearest_other - d_self
- 决策: 如果 user-u 自己的 LOO real sentence 也 0% M>0, 才是真的 structural ceiling.
  否则瓶颈仍在 generation.

也对比 using existing μ_u from stage8_5_user_gaussians.json (production user Gaussian).
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
BUCKET_NPZ = SCRATCH / "gaussian_vades" / "bucket_sample_hidden.npz"
LOG_DIR = SCRATCH / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# === Config ===
MIN_SENTENCES_PER_USER = 3      # LOO 需要至少 3 sentences (>=2 fit, 1 held-out)
N_MAX_USERS = 500                # 上限 cohort size, Phase 6.E.2 推荐 100-500
R_THRESHOLD = 12.0               # permissive (R_95 通常 8-15)
SEED = 42
CHUNK_SIZE = 200
DTYPE = np.float32


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [oracle_pos] {msg}", flush=True)


def einsum_min_dist(pool_chunk: np.ndarray, users: np.ndarray) -> np.ndarray:
    """Memory-safe min L2 distance over users."""
    a_sq = (pool_chunk * pool_chunk).sum(axis=1)[:, None]
    b_sq = (users * users).sum(axis=1)[None, :]
    dot = pool_chunk @ users.T
    d_sq = a_sq + b_sq - 2.0 * dot
    np.maximum(d_sq, 0.0, out=d_sq)
    return np.sqrt(d_sq.min(axis=1))


def main():
    log(f"=== Phase 6.E.2 — POSITIVE-CONTROL Oracle (LOO held-out real sentences) ===")
    log(f"  Source: bucket_sample_hidden.npz (real human-written review sentences)")
    log(f"  LOO setup: μ_fit from K-1 sentences, d_self on held-out sentence i")
    log(f"  MIN_SENTENCES_PER_USER={MIN_SENTENCES_PER_USER}, N_MAX_USERS={N_MAX_USERS}, R={R_THRESHOLD}")

    # === Load bucket sample sentences ===
    log(f"\n  Loading {BUCKET_NPZ} ...")
    data = np.load(BUCKET_NPZ, allow_pickle=True)
    sentences = data["sentences"]
    sent_to_uid = data["sent_to_uid"]
    n_total = len(sentences)
    log(f"  loaded {n_total} sentences, {len(set(sent_to_uid.tolist()))} unique users")

    # Group sentences by user
    user_sent_idx = {}
    for i, uid in enumerate(sent_to_uid.tolist()):
        user_sent_idx.setdefault(uid, []).append(i)
    eligible_users = [u for u, idx in user_sent_idx.items() if len(idx) >= MIN_SENTENCES_PER_USER]
    log(f"  eligible users (≥{MIN_SENTENCES_PER_USER} sentences): {len(eligible_users)}")

    rng = np.random.default_rng(SEED)
    rng.shuffle(eligible_users)
    test_users = eligible_users[:N_MAX_USERS]
    log(f"  selected {len(test_users)} test users for LOO oracle")

    # === Load Gaussians (for d_nearest_other + existing μ_u comparison) ===
    log(f"\n  Loading {GAUSS_PATH} ...")
    gauss = json.load(open(GAUSS_PATH))
    fnames_all = gauss["feature_names_ordered"]
    fnames_f3 = gauss["fnames_f3"]
    col_idx_f3 = [fnames_all.index(n) for n in fnames_f3]
    scaler_mean = np.asarray(gauss["scaler_mean"])
    scaler_scale = np.asarray(gauss["scaler_scale"])
    pca_components = np.asarray(gauss["pca_components"], dtype=np.float64)
    pca_mean = np.asarray(gauss["pca_mean"], dtype=np.float64)
    users_dict = gauss["users"]

    # Build all_user_mus (1.2M users) for d_nearest_other
    all_user_mus = []
    for uid, ug in users_dict.items():
        mu = np.asarray(ug.get("mu", ug.get("mu_48")), dtype=np.float64)
        if mu.shape == (48,):
            all_user_mus.append(mu)
    all_user_mus = np.array(all_user_mus, dtype=DTYPE)
    log(f"  loaded {len(all_user_mus)} user mus for d_nearest_other")
    # Build per-uid index for fast existing-μ lookup
    uid_to_idx = {uid: i for i, uid in enumerate(users_dict.keys()) if np.asarray(users_dict[uid].get("mu", users_dict[uid].get("mu_48")), dtype=np.float64).shape == (48,)}
    log(f"  uid_to_idx: {len(uid_to_idx)} entries")

    # === spaCy setup for F3 features ===
    log(f"\n  Loading spaCy + per_sentence_features_v2 ...")
    sys.path.insert(0, str(REPO_ROOT / "common"))
    from syntactic_features import per_sentence_features_v2
    import spacy
    nlp = spacy.load("en_core_web_sm")
    if "textcat" in nlp.pipe_names:
        nlp.remove_pipe("textcat")

    # === Compute F3 → PCA48 for ALL bucket sentences ===
    log(f"\n  Computing F3 → PCA48 for {n_total} sentences ...")
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
    log(f"  computed PCA48 in {time.time() - t0:.1f}s, {len(failed_idx)} failed (skipped)")

    # === Per-user LOO oracle ===
    log(f"\n=== LOO per-user oracle (N={len(test_users)}) ===")
    audit_results = []

    for pi, uid in enumerate(test_users):
        sent_idx = [i for i in user_sent_idx[uid] if i not in set(failed_idx)]
        if len(sent_idx) < MIN_SENTENCES_PER_USER:
            continue
        n_user = len(sent_idx)
        # Existing μ from production
        existing_mu = None
        if uid in uid_to_idx:
            existing_mu = np.asarray(users_dict[uid].get("mu", users_dict[uid].get("mu_48")), dtype=np.float64)

        # LOO audit: for each held-out sentence
        loo_M_loo = []           # M using LOO-fit μ
        loo_M_exist = []         # M using existing μ (only if available)
        loo_d_self_loo = []
        loo_d_self_exist = []
        rng_user = np.random.default_rng(SEED + pi)
        for held_out_i in sent_idx:
            # Fit μ from K-1 sentences (in PCA48)
            fit_idx = [k for k in sent_idx if k != held_out_i]
            mu_fit = all_z[fit_idx].mean(axis=0).astype(DTYPE)
            z_held = all_z[held_out_i].astype(DTYPE)

            # d_self_loo = ||z_held - μ_fit||
            d_self_loo = float(np.linalg.norm(z_held - mu_fit))
            # d_nearest_other: chunked einsum
            d_nearest_other_loo = float(einsum_min_dist(z_held[None, :], all_user_mus)[0])
            M_loo = d_nearest_other_loo - d_self_loo
            loo_M_loo.append(M_loo)
            loo_d_self_loo.append(d_self_loo)

            # d_self_existing
            if existing_mu is not None:
                d_self_exist = float(np.linalg.norm(z_held - existing_mu.astype(DTYPE)))
                M_exist = d_nearest_other_loo - d_self_exist
                loo_M_exist.append(M_exist)
                loo_d_self_exist.append(d_self_exist)

        # Per-user summary
        d_self_loo_med = float(np.median(loo_d_self_loo))
        M_loo_med = float(np.median(loo_M_loo))
        M_gt0_loo = float(np.mean([m > 0 for m in loo_M_loo]))
        strict_loo = float(np.mean([m > 0 and d <= R_THRESHOLD
                                     for m, d in zip(loo_M_loo, loo_d_self_loo)]))

        d_self_exist_med = float(np.median(loo_d_self_exist)) if loo_d_self_exist else None
        M_exist_med = float(np.median(loo_M_exist)) if loo_M_exist else None
        M_gt0_exist = float(np.mean([m > 0 for m in loo_M_exist])) if loo_M_exist else None
        strict_exist = float(np.mean([m > 0 and d <= R_THRESHOLD
                                       for m, d in zip(loo_M_exist, loo_d_self_exist)])) if loo_M_exist else None

        audit_results.append({
            "user_id": uid,
            "n_sentences": n_user,
            "loo": {
                "d_self_med": d_self_loo_med,
                "M_med": M_loo_med,
                "M_gt0_rate": M_gt0_loo,
                "strict_rate": strict_loo,
            },
            "existing": {
                "d_self_med": d_self_exist_med,
                "M_med": M_exist_med,
                "M_gt0_rate": M_gt0_exist,
                "strict_rate": strict_exist,
            } if existing_mu is not None else None,
        })

        log(f"\n  User {pi+1}/{len(test_users)}: {uid} ({n_user} sents)")
        log(f"    LOO μ_fit:       d_self_med={d_self_loo_med:.2f}  M_med={M_loo_med:.2f}  "
            f"M>0={M_gt0_loo*100:.0f}%  strict={strict_loo*100:.0f}%")
        if existing_mu is not None:
            log(f"    existing μ_u:    d_self_med={d_self_exist_med:.2f}  M_med={M_exist_med:.2f}  "
                f"M>0={M_gt0_exist*100:.0f}%  strict={strict_exist*100:.0f}%")

    # === Cohort summary ===
    if not audit_results:
        log("\n  ⚠️ no users with sufficient sentences")
        return

    loo_M_med = [r["loo"]["M_med"] for r in audit_results]
    loo_M_gt0 = [r["loo"]["M_gt0_rate"] for r in audit_results]
    loo_strict = [r["loo"]["strict_rate"] for r in audit_results]
    n_loo_pos = sum(1 for r in audit_results if r["loo"]["M_gt0_rate"] > 0.5)

    log(f"\n=== COHORT POSITIVE-CONTROL ORACLE SUMMARY (N={len(audit_results)}) ===")
    log(f"\n  LOO μ_fit (K-1 sentences, fresh fit):")
    log(f"    M_med:              median={np.median(loo_M_med):.2f}  mean={np.mean(loo_M_med):.2f}")
    log(f"    M_gt0_rate/user:    median={np.median(loo_M_gt0)*100:.1f}%  "
        f"users with >50%: {n_loo_pos}/{len(loo_M_gt0)}")
    log(f"    strict_rate/user:   median={np.median(loo_strict)*100:.1f}%")

    exist_results = [r for r in audit_results if r["existing"] is not None]
    if exist_results:
        exist_M_med = [r["existing"]["M_med"] for r in exist_results]
        exist_M_gt0 = [r["existing"]["M_gt0_rate"] for r in exist_results]
        exist_strict = [r["existing"]["strict_rate"] for r in exist_results]
        n_exist_pos = sum(1 for r in exist_results if r["existing"]["M_gt0_rate"] > 0.5)
        log(f"\n  Existing μ_u (from stage8_5_user_gaussians.json, n_users={len(exist_results)}):")
        log(f"    M_med:              median={np.median(exist_M_med):.2f}  mean={np.mean(exist_M_med):.2f}")
        log(f"    M_gt0_rate/user:    median={np.median(exist_M_gt0)*100:.1f}%  "
            f"users with >50%: {n_exist_pos}/{len(exist_M_gt0)}")
        log(f"    strict_rate/user:   median={np.median(exist_strict)*100:.1f}%")

    log(f"\n=== KEY VERDICT ===")
    n_loo_50 = sum(1 for r in audit_results if r["loo"]["M_gt0_rate"] > 0.5)
    n_loo_30 = sum(1 for r in audit_results if r["loo"]["M_gt0_rate"] > 0.3)
    if n_loo_50 >= len(audit_results) // 2:
        log(f"  LOO M>0 >50% for {n_loo_50}/{len(audit_results)} users → PCA48 has user-specific signal")
        log(f"  → bottleneck is GENERATION, not representation")
        log(f"  → Phase 6 series CAN be saved with better generator")
    elif n_loo_30 >= len(audit_results) // 3:
        log(f"  LOO M>0 >30% for {n_loo_30}/{len(audit_results)} users → PARTIAL user-specific signal")
        log(f"  → ceiling is partial, some users identifiable, others not")
        log(f"  → need to investigate which user types are separable")
    else:
        log(f"  LOO M>0 >30% for only {n_loo_30}/{len(audit_results)} users → STRONG CEILING")
        log(f"  → even target user's own held-out real sentences barely M>0")
        log(f"  → PCA48 representation IS the structural bottleneck")

    # === Save ===
    out_path = LOG_DIR / "phase6e_oracle_positive_control.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "min_sentences_per_user": MIN_SENTENCES_PER_USER,
                "n_max_users": N_MAX_USERS,
                "n_test_users_actual": len(test_users),
                "n_audit_users": len(audit_results),
                "r_threshold": R_THRESHOLD,
                "seed": SEED,
                "chunk_size": CHUNK_SIZE,
                "dtype": str(DTYPE),
                "source": "bucket_sample_hidden.npz",
            },
            "cohort_summary": {
                "loo": {
                    "M_med_cohort": float(np.median(loo_M_med)),
                    "M_gt0_rate_median": float(np.median(loo_M_gt0)),
                    "users_M_gt0_gt50": int(n_loo_pos),
                    "strict_rate_median": float(np.median(loo_strict)),
                },
                "existing": {
                    "n_users_with_existing": len(exist_results),
                    "M_med_cohort": float(np.median([r["existing"]["M_med"] for r in exist_results])) if exist_results else None,
                    "M_gt0_rate_median": float(np.median([r["existing"]["M_gt0_rate"] for r in exist_results])) if exist_results else None,
                    "users_M_gt0_gt50": int(sum(1 for r in exist_results if r["existing"]["M_gt0_rate"] > 0.5)),
                } if exist_results else None,
            },
            "per_user": audit_results,
        }, f, indent=2)
    log(f"\n  wrote → {out_path}")


if __name__ == "__main__":
    main()