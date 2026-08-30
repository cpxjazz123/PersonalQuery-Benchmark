"""Phase 6.F.1 — Paired Margin Consistency Audit (correct candidate_z).

User critique (23:55):
- Phase 6.F reported M_local=-5.00 < M_global=-4.30 which violates
  C_local ⊂ C_global → M_local ≥ M_global.
- Root cause: code used μ_u as proxy for candidate_z, computing
  ||μ_u − μ_comp|| instead of ||candidate_z − μ_comp||.
- Real d_nearest_other_local should be ~5-7, not 0.5-1.2.

This script:
1. Re-extracts F3 features for each Phase 6.D candidate query (180)
2. Projects to PCA48 (true candidate_z, NOT μ_u)
3. Computes d_nearest_other_local = ||candidate_z − μ_comp_local||
4. Computes d_nearest_other_global = ||candidate_z − μ_comp_global||
   (over 1.2M, used as paired reference)
5. Checks M_local − M_global ≥ 0 must hold for all 126 valid candidates
6. Reports paired delta statistics

Output: result/gen_query/phase6f1_paired_audit.json
"""
import gzip
import json
import math
import os
import statistics
import time
from collections import defaultdict

import numpy as np
import spacy

# Hardcoded paths (per Rule 3)
PILOT_JSON = "/home/wlia0047/hj82_scratch2/wenyu/logs/phase6d_pilot.json"
GAUSSIANS_JSON = "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_user_gaussians.json"
ASINS_JSON = "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_asins.json"
CV_JSON = "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_user_gaussians_cv.json"
LOG_PATH = "/home/wlia0047/hj82_scratch2/wenyu/logs/phase6f1_paired_audit.log"
RESULT_PATH = "/home/wlia0047/ar57/wenyu/PersoanlQuery/result/gen_query/phase6f1_paired_audit.json"

CV_INLIER_MIN = 0.9
CV_NLL_MAX = 39.0
R_95 = 9.0


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] [phase6f1] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def project_query_to_pca48(query_text, nlp, all_fnames, fnames_sub, col_idx,
                            scaler_mean, scaler_scale, pca_components, pca_mean):
    """Extract F3 → sub → scale → PCA48. Returns candidate_z (48,) or None."""
    import sys
    sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
    from common.syntactic_features import per_sentence_features_v2

    doc = nlp(query_text)
    sents = list(doc.sents)
    if not sents:
        return None
    feats_per_sent = []
    for sent in sents:
        f = per_sentence_features_v2(sent)
        if f is not None:
            feats_per_sent.append(f)
    if not feats_per_sent:
        return None
    # Aggregate by mean across sentences — DROP non-numeric values
    # (opener="NOUN", stype="complex" etc. are strings, can't average)
    all_keys = set()
    for f in feats_per_sent:
        all_keys.update(f.keys())
    mean_feats = {}
    for k in all_keys:
        vals = [f.get(k, 0.0) for f in feats_per_sent]
        # Keep only numeric values; skip opener/stype string fields
        if all(isinstance(v, (int, float, np.integer, np.floating)) for v in vals):
            try:
                mean_feats[k] = float(np.mean(vals))
            except (TypeError, ValueError):
                pass
    # Map to all_fnames (full feature space)
    vec_full = np.array([mean_feats.get(n, 0.0) for n in all_fnames], dtype=np.float64)
    # Subset to fnames_sub (F3_CoreStruct, 103d)
    vec_sub = vec_full[col_idx]
    # StandardScaler
    vec_sub_scaled = (vec_sub - scaler_mean) / scaler_scale
    # PCA: sklearn pca.transform(x) = (x - pca.mean_) @ pca.components_.T
    # pca_components shape (48, 103), pca_mean shape (103,)
    candidate_z = (vec_sub_scaled - pca_mean) @ pca_components.T
    return candidate_z.astype(np.float32)


def main():
    log("=== Phase 6.F.1 — Paired Margin Consistency Audit (FIX) ===")
    log(f"  CV quality gate: cv_inlier ≥ {CV_INLIER_MIN}, cv_nll ≤ {CV_NLL_MAX}")
    log(f"  R_95 = {R_95}")
    log("  KEY FIX: project candidate query text → PCA48 (NOT use μ_u as proxy)")

    # 1. Load Phase 6.D pilot
    with open(PILOT_JSON) as f:
        pilot = json.load(f)
    pilot_users = [r["user_id"] for r in pilot["results"]]
    log(f"\n  pilot users: {len(pilot_users)}")
    log(f"  total candidates: {sum(r['n_candidates'] for r in pilot['results'])}")

    # 2. Load stage8_5_user_gaussians.json → PCA + scaler
    log(f"\n  Loading PCA + scaler from {GAUSSIANS_JSON}...")
    with open(GAUSSIANS_JSON) as f:
        gdoc = json.load(f)
    all_fnames = gdoc["feature_names_ordered"]
    fnames_sub = gdoc["fnames_f3"]
    scaler_mean = np.asarray(gdoc["scaler_mean"], dtype=np.float64)
    scaler_scale = np.asarray(gdoc["scaler_scale"], dtype=np.float64)
    pca_components = np.asarray(gdoc["pca_components"], dtype=np.float64)  # (48, 103)
    pca_mean = np.asarray(gdoc["pca_mean"], dtype=np.float64)  # (103,)
    col_idx = np.array([all_fnames.index(n) for n in fnames_sub], dtype=np.int64)
    user_mus = gdoc["users"]
    log(f"  loaded: all_fnames={len(all_fnames)}, fnames_sub={len(fnames_sub)}, "
        f"PCA={pca_components.shape}, μ={len(user_mus)} users")

    # 3. Load ASIN cohort → same-ASIN local competitor set
    log(f"\n  Loading ASIN cohort from {ASINS_JSON}...")
    with open(ASINS_JSON) as f:
        asin_data = json.load(f)
    asins_list = asin_data["asins"]
    user_to_asins = defaultdict(set)
    asin_to_competitors = defaultdict(set)
    for asin_rec in asins_list:
        a = asin_rec["asin"]
        for u in asin_rec["users_sampled"]:
            user_to_asins[u].add(a)
            asin_to_competitors[a].add(u)
    log(f"  loaded {len(asins_list)} ASINs, {len(user_to_asins)} users")

    pilot_user_asins = {u: user_to_asins.get(u, set()) for u in pilot_users}

    # 4. CV quality gate
    log(f"\n  Loading CV info from {CV_JSON}...")
    with open(CV_JSON) as f:
        cv_data = json.load(f)
    cv_users = cv_data["users"]
    high_quality = set()
    n_skip = 0
    for u, info in cv_users.items():
        ci = info.get("cv_inlier_frac")
        cn = info.get("cv_nll")
        if ci is None or cn is None:
            n_skip += 1
            continue
        if ci >= CV_INLIER_MIN and cn <= CV_NLL_MAX:
            high_quality.add(u)
    log(f"  high-quality users: {len(high_quality)} (skipped {n_skip} None)")

    pilot_user_local_competitors = {}
    for u in pilot_users:
        asins = pilot_user_asins[u]
        per_asin_comps = {}
        for a in asins:
            comps = asin_to_competitors[a] - {u}
            comps = comps & high_quality
            per_asin_comps[a] = list(comps)
        pilot_user_local_competitors[u] = per_asin_comps

    # 5. Load spaCy
    log(f"\n  Loading spaCy en_core_web_sm...")
    nlp = spacy.load("en_core_web_sm")
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)

    # 6. Project each candidate query → PCA48, then paired distance
    log(f"\n=== Re-projecting 180 candidates to PCA48 (real candidate_z) ===")
    per_user_results = []
    paired_M_global = []
    paired_M_local = []
    paired_d_self = []
    paired_d_other_global_real = []
    paired_d_other_local_real = []
    paired_candidate_idx = []

    for r in pilot["results"]:
        u = r["user_id"]
        user_mu = user_mus.get(u, {}).get("mu", None)
        if user_mu is None:
            log(f"  {u}: no μ in users dict, skipping {r['n_candidates']} candidates")
            continue
        user_mu_arr = np.array(user_mu, dtype=np.float32)

        # Build local competitor μ matrix (n_local, 48)
        comps_per_asin = pilot_user_local_competitors[u]
        all_local_comp_ids = set()
        for comps in comps_per_asin.values():
            all_local_comp_ids.update(comps)
        n_local = 0
        local_mus_arr = None
        if all_local_comp_ids:
            local_mus = []
            local_ids = []
            for c in all_local_comp_ids:
                if c in user_mus:
                    local_mus.append(user_mus[c]["mu"])
                    local_ids.append(c)
            if local_mus:
                local_mus_arr = np.array(local_mus, dtype=np.float32)  # (n_local, 48)
                n_local = len(local_mus_arr)

        # Build global competitor μ matrix (full 1.2M)
        # For paired delta, just use nearest μ over all users (compute on demand)
        global_mus = []
        global_ids = []
        for c, info in user_mus.items():
            global_mus.append(info["mu"])
            global_ids.append(c)
        global_mus_arr = np.array(global_mus, dtype=np.float32)  # (1.2M, 48)
        n_global = len(global_mus_arr)
        log(f"  {u}: {n_local} local comps, {n_global} global comps, {r['n_candidates']} candidates")

        candidate_results = []
        for c in r["candidates"]:
            query_text = c.get("query", "")
            if not query_text:
                continue
            # Project query to PCA48
            try:
                candidate_z = project_query_to_pca48(
                    query_text, nlp, all_fnames, fnames_sub, col_idx,
                    scaler_mean, scaler_scale, pca_components, pca_mean,
                )
            except Exception as e:
                log(f"    [ERROR] candidate {c['candidate_idx']}: {e}")
                continue
            if candidate_z is None:
                continue

            # d_self_real = ||candidate_z − μ_u||
            d_self_real = float(np.linalg.norm(candidate_z - user_mu_arr))
            # d_nearest_other_global_real = min over 1.2M of ||candidate_z − μ_comp||
            # Memory-safe chunked einsum
            chunk = 5000
            min_d_other_global = float("inf")
            for s in range(0, n_global, chunk):
                e = min(s + chunk, n_global)
                diffs = global_mus_arr[s:e] - candidate_z[None, :]  # (chunk, 48)
                dists = np.sqrt((diffs ** 2).sum(axis=1))
                min_d_other_global = min(min_d_other_global, float(dists.min()))
            d_nearest_other_global_real = min_d_other_global
            M_global_real = d_nearest_other_global_real - d_self_real

            # d_nearest_other_local_real = min over local of ||candidate_z − μ_comp_local||
            d_nearest_other_local_real = float("inf")
            M_local_real = float("inf")
            if local_mus_arr is not None and n_local > 0:
                diffs_l = local_mus_arr - candidate_z[None, :]
                dists_l = np.sqrt((diffs_l ** 2).sum(axis=1))
                d_nearest_other_local_real = float(dists_l.min())
                M_local_real = d_nearest_other_local_real - d_self_real

            candidate_results.append({
                "candidate_idx": c["candidate_idx"],
                "skel_d": c["skel_d"],
                "d_self_phase6d": c["d_self"],  # original (using μ_u)
                "d_self_real": d_self_real,  # recomputed from real candidate_z
                "d_nearest_other_global_real": d_nearest_other_global_real,
                "d_nearest_other_local_real": d_nearest_other_local_real,
                "M_global_real": M_global_real,
                "M_local_real": M_local_real if not math.isinf(M_local_real) else None,
                "M_local_minus_M_global": (
                    M_local_real - M_global_real if not math.isinf(M_local_real) else None
                ),
                "violation_M_local_lt_M_global": (
                    M_local_real < M_global_real if not math.isinf(M_local_real) else None
                ),
                "accepted_phase6d": c["accepted"],
            })

            if not math.isinf(M_local_real):
                paired_M_global.append(M_global_real)
                paired_M_local.append(M_local_real)
                paired_d_self.append(d_self_real)
                paired_d_other_global_real.append(d_nearest_other_global_real)
                paired_d_other_local_real.append(d_nearest_other_local_real)
                paired_candidate_idx.append((u, c["candidate_idx"]))

        per_user_results.append({
            "user_id": u,
            "n_candidates": len(candidate_results),
            "n_local_competitors": n_local,
            "n_asins_reviewed": len(pilot_user_asins[u]),
            "candidates": candidate_results,
        })

    # 7. Aggregate + paired statistics
    log(f"\n=== PAIRED STATISTICS ({len(paired_M_global)} candidates with local comps) ===")

    n_paired = len(paired_M_global)
    log(f"\n  d_self_real median: {statistics.median(paired_d_self):.3f}")
    log(f"  d_nearest_other_global_real median: {statistics.median(paired_d_other_global_real):.3f}")
    log(f"  d_nearest_other_local_real  median: {statistics.median(paired_d_other_local_real):.3f}")

    log(f"\n  PAIRED M_global_real median: {statistics.median(paired_M_global):.3f}")
    log(f"  PAIRED M_local_real  median: {statistics.median(paired_M_local):.3f}")

    delta_M = [m_l - m_g for m_l, m_g in zip(paired_M_local, paired_M_global)]
    log(f"  ΔM = M_local − M_global median: {statistics.median(delta_M):.3f}")
    log(f"  ΔM mean: {statistics.mean(delta_M):.3f}")
    log(f"  ΔM min: {min(delta_M):.3f}, max: {max(delta_M):.3f}")

    violations = sum(1 for d in delta_M if d < 0)
    log(f"  VIOLATIONS (M_local < M_global): {violations}/{n_paired}")

    # Now compute M>0 with REAL candidate_z
    P_M_gt0_global_real = sum(1 for m in paired_M_global if m > 0) / n_paired * 100
    P_M_gt0_local_real = sum(1 for m in paired_M_local if m > 0) / n_paired * 100
    log(f"\n  P(M_global_real > 0): {P_M_gt0_global_real:.1f}%")
    log(f"  P(M_local_real > 0): {P_M_gt0_local_real:.1f}%")

    # Strict: M_local > 0 AND d_self ≤ R_95
    strict = [(m_l, ds) for m_l, ds in zip(paired_M_local, paired_d_self) if ds <= R_95]
    n_strict = len(strict)
    strict_p = sum(1 for m_l, _ in strict if m_l > 0) / max(n_strict, 1) * 100
    log(f"  Strict (M_local_real > 0 AND d_self ≤ {R_95}): "
        f"{sum(1 for m_l, _ in strict if m_l > 0)}/{n_strict} = {strict_p:.1f}%")

    # Per-user paired
    log(f"\n  PER-USER PAIRED:")
    for entry in per_user_results:
        u = entry["user_id"]
        cand_results = entry["candidates"]
        Ms_g = [c["M_global_real"] for c in cand_results if c["M_local_real"] is not None]
        Ms_l = [c["M_local_real"] for c in cand_results if c["M_local_real"] is not None]
        if Ms_g:
            dM = [m_l - m_g for m_l, m_g in zip(Ms_l, Ms_g)]
            log(f"    {u}: n_valid={len(Ms_g)}, M_g_med={statistics.median(Ms_g):.2f}, "
                f"M_l_med={statistics.median(Ms_l):.2f}, ΔM_med={statistics.median(dM):.2f}, "
                f"violations={sum(1 for d in dM if d < 0)}/{len(dM)}")

    summary = {
        "config": {
            "cv_inlier_min": CV_INLIER_MIN,
            "cv_nll_max": CV_NLL_MAX,
            "r_95_threshold": R_95,
        },
        "n_pilot_users": len(pilot_users),
        "n_total_candidates": sum(r["n_candidates"] for r in pilot["results"]),
        "n_paired_valid": n_paired,
        "paired_statistics": {
            "d_self_real_median": round(statistics.median(paired_d_self), 3),
            "d_nearest_other_global_real_median": round(statistics.median(paired_d_other_global_real), 3),
            "d_nearest_other_local_real_median": round(statistics.median(paired_d_other_local_real), 3),
            "M_global_real_median": round(statistics.median(paired_M_global), 3),
            "M_local_real_median": round(statistics.median(paired_M_local), 3),
            "delta_M_median": round(statistics.median(delta_M), 3),
            "delta_M_mean": round(statistics.mean(delta_M), 3),
            "delta_M_min": round(min(delta_M), 3),
            "delta_M_max": round(max(delta_M), 3),
            "n_violations_M_local_lt_M_global": violations,
            "violation_pct": round(violations / n_paired * 100, 2),
        },
        "P_M_gt0": {
            "global_real": round(P_M_gt0_global_real, 1),
            "local_real": round(P_M_gt0_local_real, 1),
        },
        "strict_M_local_gt0_AND_d_self_le_R95": {
            "n_total": n_strict,
            "n_pass": sum(1 for m_l, _ in strict if m_l > 0),
            "pct": round(strict_p, 1),
        },
        "key_finding": (
            f"Phase 6.F.1 paired audit with REAL candidate_z (NOT μ_u proxy): "
            f"M_global_real_med={statistics.median(paired_M_global):.3f}, "
            f"M_local_real_med={statistics.median(paired_M_local):.3f}, "
            f"ΔM_med={statistics.median(delta_M):.3f}, "
            f"violations={violations}/{n_paired}, "
            f"P(M_local>0)={P_M_gt0_local_real:.1f}%, "
            f"strict={strict_p:.1f}%."
        ),
        "per_user": per_user_results,
    }
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\n  wrote → {RESULT_PATH}")
    log(f"\n=== DONE ===")


if __name__ == "__main__":
    main()
