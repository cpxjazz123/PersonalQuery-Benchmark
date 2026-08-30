"""Phase 6.F.FIX — Correct candidate-to-competitor distance + Paired audit.

Phase 6.F bug: code computed `local_mus - user_mu` (user μ between-cluster tight)
instead of `candidate_z - local_mus` (query-to-competitor distance).

This fix:
1. Re-extracts F3 (103d) from saved query texts (180 candidates)
2. StandardScaler + PCA48 projects → candidate_z_48
3. Recomputes d_nearest_other_local = min over local comp μ of ||candidate_z - μ_comp||
4. Paired audit on 126 candidates (those with local competitors):
   - Verify M_local ≥ M_global (since C_local ⊂ C_global)
   - Report median ΔM and violation count
5. Confirms true generation-to-representation gap

Output: result/gen_query/phase6f_fix_paired_audit.json
"""
import json
import math
import os
import statistics
import sys
from collections import defaultdict

# Ensure project root on path so `from common.X import Y` works under nohup
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")

import numpy as np

# Hardcoded paths (per Rule 3)
PILOT_JSON = "/home/wlia0047/hj82_scratch2/wenyu/logs/phase6d_pilot.json"
ASINS_JSON = "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_asins.json"
GAUSSIANS_JSON = "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_user_gaussians.json"
CV_JSON = "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_user_gaussians_cv.json"
LOG_PATH = "/home/wlia0047/hj82_scratch2/wenyu/logs/phase6f_fix_paired_audit.log"
RESULT_PATH = "/home/wlia0047/ar57/wenyu/PersoanlQuery/result/gen_query/phase6f_fix_paired_audit.json"

CV_INLIER_MIN = 0.9
CV_NLL_MAX = 39.0


def log(msg):
    line = f"[{__import__('datetime').datetime.now().strftime('%H:%M:%S')}] [phase6f_fix] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def main():
    log("=== Phase 6.F.FIX — Correct candidate-to-competitor distance + Paired audit ===")

    # 1. Load pilot (180 candidates with query text)
    with open(PILOT_JSON) as f:
        pilot = json.load(f)
    pilot_users = [r["user_id"] for r in pilot["results"]]
    log(f"  pilot users: {len(pilot_users)}, total candidates: {sum(r['n_candidates'] for r in pilot['results'])}")

    # 2. Load PCA48 model + per-user μ
    log(f"\n  Loading PCA48 model + user μ from {GAUSSIANS_JSON}...")
    with open(GAUSSIANS_JSON) as f:
        gauss_data = json.load(f)
    scaler_mean = np.array(gauss_data["scaler_mean"], dtype=np.float32)  # (103,)
    scaler_scale = np.array(gauss_data["scaler_scale"], dtype=np.float32)  # (103,)
    pca_components = np.array(gauss_data["pca_components"], dtype=np.float32)  # (48, 103)
    pca_mean = np.array(gauss_data["pca_mean"], dtype=np.float32)  # (103,)
    feature_names = gauss_data["fnames_f3"]  # 103 F3 names (scaler/pca aligned)
    user_mus = gauss_data["users"]  # dict {user_id: {mu, sigma_diag, ...}}
    log(f"  PCA48 model loaded: scaler (103,) + pca (48, 103) + fnames_f3 (103,)")
    log(f"  user μ loaded: {len(user_mus)} users")

    # 3. Load CV quality
    log(f"\n  Loading CV quality from {CV_JSON}...")
    with open(CV_JSON) as f:
        cv_data = json.load(f)
    cv_users = cv_data["users"]
    high_quality = set()
    n_skip_none = 0
    for u, info in cv_users.items():
        ci = info.get("cv_inlier_frac"); cn = info.get("cv_nll")
        if ci is None or cn is None:
            n_skip_none += 1; continue
        if ci >= CV_INLIER_MIN and cn <= CV_NLL_MAX:
            high_quality.add(u)
    log(f"  high-quality: {len(high_quality)} (skipped {n_skip_none} None)")

    # 4. Load ASIN cohort → reverse user→asins, asin→users
    log(f"\n  Loading ASIN cohort from {ASINS_JSON}...")
    with open(ASINS_JSON) as f:
        asin_data = json.load(f)
    asins_list = asin_data["asins"]
    user_to_asins = defaultdict(set)
    asin_to_competitors = defaultdict(set)
    for asin_rec in asins_list:
        a = asin_rec["asin"]; us = asin_rec["users_sampled"]
        for u in us:
            user_to_asins[u].add(a); asin_to_competitors[a].add(u)
    log(f"  loaded {len(asins_list)} ASINs, {len(user_to_asins)} users")

    # 5. Build per-user local competitor set
    pilot_user_local_comps = {}
    for r in pilot["results"]:
        u = r["user_id"]
        asins = user_to_asins.get(u, set())
        all_comps = set()
        for a in asins:
            comps = asin_to_competitors[a] - {u}
            comps = comps & high_quality
            all_comps.update(comps)
        pilot_user_local_comps[u] = all_comps
        log(f"    {u}: {len(asins)} ASINs, {len(all_comps)} local competitors (after quality gate)")

    # 6. Re-extract F3 from saved query texts + project to PCA48
    log(f"\n  Re-extracting F3 from saved query texts...")
    import spacy
    from common.syntactic_features import per_sentence_features_v2  # noqa: E402

    nlp = spacy.load("en_core_web_sm")
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)

    feat_name_to_idx = {n: i for i, n in enumerate(feature_names)}

    def text_to_z48(text):
        """Re-extract F3 + project to PCA48. Returns (z48,) or None on failure."""
        if not text or len(text.strip()) < 3:
            return None
        doc = nlp(text)
        sents = list(doc.sents)
        if not sents:
            return None
        # Concatenate features across sentences (avg)
        feat_acc = None
        n_ok = 0
        for s in sents:
            f = per_sentence_features_v2(s)
            if f is None:
                continue
            # Build 103d vector
            v = np.zeros(len(feature_names), dtype=np.float32)
            for name, idx in feat_name_to_idx.items():
                if name in f:
                    v[idx] = f[name]
            feat_acc = v if feat_acc is None else feat_acc + v
            n_ok += 1
        if feat_acc is None or n_ok == 0:
            return None
        feat_avg = feat_acc / n_ok
        # StandardScaler
        x = (feat_avg - scaler_mean) / scaler_scale
        # PCA48
        z48 = (x - pca_mean) @ pca_components.T
        return z48

    per_user_results = []
    n_z_fail = 0
    for r in pilot["results"]:
        u = r["user_id"]
        comps = pilot_user_local_comps[u]
        n_local = len(comps)
        # Pre-build local μ matrix (n_local × 48)
        local_mus_arr = None
        if n_local > 0:
            local_mus = []
            for c in comps:
                if c in user_mus:
                    local_mus.append(user_mus[c]["mu"])
            if local_mus:
                local_mus_arr = np.array(local_mus, dtype=np.float32)

        candidates = []
        for c in r["candidates"]:
            query_text = c.get("query", "")
            z48 = text_to_z48(query_text)
            if z48 is None:
                n_z_fail += 1
                # d_self already computed in pilot
                d_self = c["d_self"]
                d_nearest_other_local = float("inf")
                M_local_fixed = float("-inf")  # missing
                M_global = c["margin"]
            else:
                d_self = c["d_self"]
                if local_mus_arr is None or len(local_mus_arr) == 0:
                    d_nearest_other_local = float("inf")
                    M_local_fixed = float("-inf")  # no local comp
                else:
                    # CORRECT: candidate_z - competitor μ
                    diffs = local_mus_arr - z48[None, :]
                    dists = np.sqrt((diffs ** 2).sum(axis=1))
                    d_nearest_other_local = float(dists.min())
                    M_local_fixed = d_nearest_other_local - d_self
                M_global = c["margin"]

            candidates.append({
                "candidate_idx": c["candidate_idx"],
                "d_self": d_self,
                "d_nearest_other_global": c["d_nearest_other"],
                "M_global": M_global,
                "d_nearest_other_local_fixed": d_nearest_other_local,
                "M_local_fixed": M_local_fixed,
                "M_local_positive_fixed": (M_local_fixed > 0) if (not math.isinf(M_local_fixed) and M_local_fixed != float("-inf")) else None,
                "n_local_competitors": n_local,
                "z_extracted_ok": z48 is not None,
            })
        per_user_results.append({
            "user_id": u,
            "mu_norm": r["mu_norm"],
            "n_candidates": r["n_candidates"],
            "n_local_competitors": n_local,
            "candidates": candidates,
        })
    log(f"  F3 extraction done; {n_z_fail} candidates failed extraction")

    # 7. Paired audit on candidates with local competitors AND valid z48
    log(f"\n=== PAIRED AUDIT (candidates with local comp & valid z48) ===")
    paired = []
    for entry in per_user_results:
        for c in entry["candidates"]:
            if c["n_local_competitors"] > 0 and c["z_extracted_ok"]:
                paired.append({
                    "user_id": entry["user_id"],
                    "candidate_idx": c["candidate_idx"],
                    "d_self": c["d_self"],
                    "d_nearest_other_global": c["d_nearest_other_global"],
                    "M_global": c["M_global"],
                    "d_nearest_other_local_fixed": c["d_nearest_other_local_fixed"],
                    "M_local_fixed": c["M_local_fixed"],
                })

    log(f"  paired candidates: {len(paired)}")

    M_g = [p["M_global"] for p in paired]
    M_l = [p["M_local_fixed"] for p in paired]
    d_og = [p["d_nearest_other_global"] for p in paired]
    d_ol = [p["d_nearest_other_local_fixed"] for p in paired]

    log(f"  M_global median: {statistics.median(M_g):.3f}")
    log(f"  M_local_fixed median: {statistics.median(M_l):.3f}")
    log(f"  d_nearest_other_global median: {statistics.median(d_og):.3f}")
    log(f"  d_nearest_other_local_fixed median: {statistics.median(d_ol):.3f}")

    # Mathematical invariant: C_local ⊂ C_global → d_other_local ≥ d_other_global
    invariant_violations = sum(1 for p in paired if p["d_nearest_other_local_fixed"] < p["d_nearest_other_global"] - 1e-6)
    log(f"  invariant violations (d_local < d_global): {invariant_violations} / {len(paired)}")

    # M_local - M_global should be ≥ 0
    delta_Ms = [p["M_local_fixed"] - p["M_global"] for p in paired]
    M_lift_pos = sum(1 for d in delta_Ms if d >= -1e-6)
    log(f"  ΔM = M_local - M_global median: {statistics.median(delta_Ms):.3f}")
    log(f"  ΔM ≥ 0 count: {M_lift_pos} / {len(paired)} = {M_lift_pos / len(paired) * 100:.1f}%")

    P_global_gt0 = sum(1 for m in M_g if m > 0)
    P_local_fixed_gt0 = sum(1 for m in M_l if m > 0)
    log(f"  P(M_global > 0): {P_global_gt0}/{len(paired)} = {P_global_gt0 / len(paired) * 100:.1f}%")
    log(f"  P(M_local_fixed > 0): {P_local_fixed_gt0}/{len(paired)} = {P_local_fixed_gt0 / len(paired) * 100:.1f}%")

    # Per-user summary
    log(f"\n  PER-USER PAIRED:")
    per_user_summary = []
    for entry in per_user_results:
        user_paired = [c for c in entry["candidates"] if c["n_local_competitors"] > 0 and c["z_extracted_ok"]]
        if not user_paired:
            continue
        pg = [c["M_global"] for c in user_paired]
        pl = [c["M_local_fixed"] for c in user_paired]
        dml = [(c["M_local_fixed"] - c["M_global"]) for c in user_paired]
        p_gt0_local = sum(1 for m in pl if m > 0)
        log(f"    {entry['user_id']} (n_local={entry['n_local_competitors']}, N={len(user_paired)}):")
        log(f"      M_global med={statistics.median(pg):.3f}, M_local med={statistics.median(pl):.3f}, ΔM med={statistics.median(dml):.3f}, P(M_local>0)={p_gt0_local / len(user_paired) * 100:.1f}%")
        per_user_summary.append({
            "user_id": entry["user_id"],
            "n_local": entry["n_local_competitors"],
            "n_paired": len(user_paired),
            "M_global_median": round(statistics.median(pg), 3),
            "M_local_median": round(statistics.median(pl), 3),
            "deltaM_median": round(statistics.median(dml), 3),
            "P_M_local_gt0": round(p_gt0_local / len(user_paired) * 100, 1),
        })

    # 8. Save result JSON
    summary = {
        "config": {"cv_inlier_min": CV_INLIER_MIN, "cv_nll_max": CV_NLL_MAX},
        "n_total_candidates": sum(r["n_candidates"] for r in pilot["results"]),
        "n_paired_candidates": len(paired),
        "n_z48_extraction_failures": n_z_fail,
        "paired_aggregate": {
            "M_global_median": round(statistics.median(M_g), 3),
            "M_global_mean": round(statistics.mean(M_g), 3),
            "M_local_fixed_median": round(statistics.median(M_l), 3),
            "M_local_fixed_mean": round(statistics.mean(M_l), 3),
            "deltaM_median": round(statistics.median(delta_Ms), 3),
            "deltaM_mean": round(statistics.mean(delta_Ms), 3),
            "P_deltaM_ge0": round(M_lift_pos / len(paired) * 100, 1),
            "invariant_violations": invariant_violations,
            "P_M_global_gt0": round(P_global_gt0 / len(paired) * 100, 1),
            "P_M_local_fixed_gt0": round(P_local_fixed_gt0 / len(paired) * 100, 1),
            "d_nearest_other_global_median": round(statistics.median(d_og), 3),
            "d_nearest_other_local_fixed_median": round(statistics.median(d_ol), 3),
        },
        "per_user": per_user_summary,
        "key_finding_placeholder": "to be filled",
    }

    if invariant_violations == 0:
        verdict = "INVARIANT_PASS"
    else:
        verdict = "INVARIANT_FAIL"

    if P_local_fixed_gt0 == 0:
        summary["key_finding"] = (
            f"Phase 6.F.FIX: After correcting the user-to-user → candidate-to-competitor distance bug, "
            f"M_local_fixed median = {statistics.median(M_l):.3f}, P(M_local > 0) = "
            f"{P_local_fixed_gt0 / len(paired) * 100:.1f}% (vs M_global median = {statistics.median(M_g):.3f}, "
            f"P(M_global > 0) = {P_global_gt0 / len(paired) * 100:.1f}%). "
            f"Invariant violations: {invariant_violations}. "
            f"VERDICT: even with correct local competitor distance, generated queries achieve 0% M>0 — "
            f"this is a TRUE generation-to-representation gap (PCA48 captures real user-local signal, "
            f"but LLM-generated queries cannot reach user-specific location)."
        )
    else:
        summary["key_finding"] = (
            f"Phase 6.F.FIX: After correction, M_local_fixed median = {statistics.median(M_l):.3f}, "
            f"P(M_local > 0) = {P_local_fixed_gt0 / len(paired) * 100:.1f}% (vs M_global = {statistics.median(M_g):.3f}, "
            f"P(M_global > 0) = {P_global_gt0 / len(paired) * 100:.1f}%). Invariant violations: {invariant_violations}."
        )

    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\n  wrote → {RESULT_PATH}")
    log(f"\n  VERDICT: {verdict}")
    log(f"\n=== DONE ===")


if __name__ == "__main__":
    main()
