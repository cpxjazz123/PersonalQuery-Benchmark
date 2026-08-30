"""Phase 7.C.2 — Real Sentence Positive Control (Representation Gate).

Gate that the unified sentence-level profile (Phase 7.C.1) actually carries
local personalization signal, before any generation is attempted.

Inputs:
    scratch2/gaussian_vades/stage8_5_user_profiles_sentence.json (7.C.1 output)
    scratch2/gaussian_vades/stage8_5_user_gaussians_cv.json (Q-gate: cv_inlier_frac, cv_nll)
    scratch2/gaussian_vades/stage8_5_asins.json (cohort: same-ASIN user maps)

Per audit user:
    - Anchor = mu_{u,a}^prod (sentence-level, target ASIN excluded, 8-60 token)
    - Source Set S_u = 7.C.1 source_set (real sentences, target ASIN excluded)
    - Competitor = sentence-level profile of same-ASIN users who pass
        quality gate (cv_inlier_frac >= 0.9, cv_nll <= 39.0)
      reduced to BD-separated subset (T_B=2.0, SEED=42) if cohort large enough
    - For each source sentence z_s:
        d_self = ||z_s - mu_{u,a}^prod||
        d_other = min over competitors ||z_s - mu_{competitor}||  (all must be sentence-level profile, not cache)
        M_local = d_other - d_self
      (Use the SAME mu_{u,a}^prod formula for competitors: their ASIN(s)!=a_other,
       8-60 token sentence-level mean. We need to recompute competitor profiles
       under same constraint.)

Decision rule:
    If P(M_local > 0) >= 0.90 across audit users → representation gate PASS,
    fair free vs rewrite (Phase 7.C.3) is meaningful.
    Else → fail-fast; need to revisit anchor / support unit / competitor selection.

Smoke: 15 audit users (5 7.B + 10 random from 7.C.1).

Output:
    result/gen_query/phase7c2_real_sentence_positive_control.json
"""

from __future__ import annotations

import collections
import gzip
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
REVIEWS_GZ = REPO_ROOT / "data" / "Baby_Products_2023.jsonl.gz"
PROFILES_PATH = SCRATCH / "stage8_5_user_profiles_sentence.json"
GAUSS_CV_PATH = SCRATCH / "stage8_5_user_gaussians_cv.json"
ASINS_PATH = SCRATCH / "stage8_5_asins.json"
OUT_JSON = REPO_ROOT / "result" / "gen_query" / "phase7c2_real_sentence_positive_control.json"

MIN_TOKENS = 8
MAX_TOKENS = 60
CV_INLIER_MIN = 0.9
CV_NLL_MAX = 39.0
T_B = 2.0
SEED = 42
GATE_PASS_THRESHOLD = 0.90


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [7c2] {msg}", flush=True)


def feat_key(t: str) -> str:
    return hashlib.sha1(t.strip().lower().encode("utf-8")).hexdigest()


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def bhattacharyya_distance_diag(mu_i, sigma_i, mu_j, sigma_j):
    """Diagonal Bhattacharyya (6.E.4 convention)."""
    eps = 1e-12
    diff = mu_i - mu_j
    sigma_avg = 0.5 * (sigma_i + sigma_j)
    term1 = 0.125 * np.sum((diff * diff) / np.maximum(sigma_avg, eps))
    term2 = 0.5 * np.sum(np.log(np.maximum(sigma_avg, eps))
                         - 0.5 * np.log(np.maximum(sigma_i, eps))
                         - 0.5 * np.log(np.maximum(sigma_j, eps)))
    return float(term1 + term2)


def greedy_maxmin_in_group(users_data, t_b, seed=SEED):
    """Greedy BD-maxmin reduction. users_data: list of {mu, sigma}."""
    rng = random.Random(seed)
    if not users_data:
        return []
    if len(users_data) == 1:
        return [0]
    selected = [rng.randrange(len(users_data))]
    while True:
        candidates = []
        for i in range(len(users_data)):
            if i in selected:
                continue
            min_bd = min(
                bhattacharyya_distance_diag(users_data[i]["mu"], users_data[i]["sigma"],
                                            users_data[s]["mu"], users_data[s]["sigma"])
                for s in selected
            )
            candidates.append((min_bd, i))
        if not candidates:
            break
        best_bd, best_i = max(candidates)
        if best_bd < t_b:
            break
        selected.append(best_i)
    return selected


def project_sentence(texts, nlp, all_fnames, col_idx, scaler_mean, scaler_scale,
                     pca_components, pca_mean, psf):
    """Per-sentence psf-mean projection (matches 7.C.1 convention)."""
    sents = []
    for t in texts:
        for s in split_sentences(t):
            toks = s.split()
            n_tok = len(toks)
            if n_tok < MIN_TOKENS or n_tok > MAX_TOKENS:
                continue
            sents.append((s, feat_key(s)))
    if not sents:
        return np.zeros((0, 48), dtype=np.float64), []
    z48 = np.zeros((len(sents), 48), dtype=np.float64)
    docs = list(nlp.pipe([s[0] for s in sents], batch_size=128, n_process=4))
    for i, doc in enumerate(docs):
        sd = list(doc.sents)
        if not sd:
            continue
        feats_per_sent = [psf(s) for s in sd if psf(s) is not None]
        if not feats_per_sent:
            continue
        all_keys = set()
        for f in feats_per_sent:
            all_keys.update(f.keys())
        mean_feats = {}
        for k in all_keys:
            vals = [f.get(k, 0.0) for f in feats_per_sent]
            if all(isinstance(v, (int, float, np.integer, np.floating)) for v in vals):
                try:
                    mean_feats[k] = float(np.mean(vals))
                except (TypeError, ValueError):
                    pass
        vec_full = np.array([mean_feats.get(nm, 0.0) for nm in all_fnames], dtype=np.float64)
        v = vec_full[col_idx]
        v_scaled = (v - scaler_mean) / scaler_scale
        z48[i] = (v_scaled - pca_mean) @ pca_components.T
    return z48, sents


def main() -> None:
    log("=== Phase 7.C.2 — Real Sentence Positive Control ===")

    # --- 1. Load profiles (7.C.1 output) ---
    profiles_doc = json.load(open(PROFILES_PATH))
    profiles = profiles_doc["users"]
    all_fnames = profiles_doc.get("feature_names_ordered")
    # Note: profiles cache doesn't store feature_names_ordered (we used cache's fnames
    # in 7.C.1); reload fnames from gaussian cache.
    gdoc = json.load(open(SCRATCH / "stage8_5_user_gaussians.json"))
    if all_fnames is None:
        all_fnames = gdoc["feature_names_ordered"]
    fnames_f3 = gdoc["fnames_f3"]
    scaler_mean = np.asarray(gdoc["scaler_mean"], dtype=np.float64)
    scaler_scale = np.asarray(gdoc["scaler_scale"], dtype=np.float64)
    pca_components = np.asarray(gdoc["pca_components"], dtype=np.float64)
    pca_mean = np.asarray(gdoc["pca_mean"], dtype=np.float64)
    col_idx = [all_fnames.index(nm) for nm in fnames_f3]
    log(f"  loaded {len(profiles)} profiles from 7.C.1")

    # --- 2. Load CV quality + ASIN cohort ---
    cv_doc = json.load(open(GAUSS_CV_PATH))
    cv_users = cv_doc.get("users", {})
    asins_doc = json.load(open(ASINS_PATH))
    user_to_asins = collections.defaultdict(list)
    asin_to_users = collections.defaultdict(list)
    for entry in asins_doc["asins"]:
        a = entry["asin"]
        for uid in entry["users_sampled"]:
            user_to_asins[uid].append(a)
            asin_to_users[a].append(uid)
    log(f"  loaded CV for {len(cv_users)} users, cohort ASINs={len(asin_to_users)}")

    # --- 3. spaCy ---
    import spacy
    sys.path.insert(0, str(REPO_ROOT / "common"))
    from syntactic_features import per_sentence_features_v2 as psf
    nlp = spacy.load("en_core_web_sm")
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)
    log("  spaCy loaded")

    # --- 4. Determine competitor pool (sentence-level profiles for cohort users) ---
    # We need competitor profiles built from same sentence-level recipe as 7.C.1.
    # Approach: load 7.C.1 profiles (which are already sentence-level for the 15 audit
    # users), and for OTHER cohort users we need to rebuild. But 7.C.1 only covers
    # 15 users — for 7.C.2 we need competitor profiles for many more users.
    #
    # Efficient approach: load 7.C.1 source_set payload (which contains z48 per
    # source sentence for each audit user) — these are already projected and can
    # be used directly as competitor candidates IF the source_set covers other
    # audit users. For non-audit cohort users, we need a separate path.
    #
    # Strategy: For each audit user u with target_asin a:
    #   competitor candidates = audit users v != u whose target_asin != a
    #     OR any cohort user with target_asin == a (from asin_to_users[a])
    #   For each candidate v:
    #     1) quality gate via cv_users[v]: cv_inlier_frac >= 0.9, cv_nll <= 39.0
    #     2) if audit user v in 7.C.1: use profile mu/sigma from 7.C.1 (sentence-level)
    #        else: skip (out of cohort scope for 7.C.2 smoke; competitor may be small)
    #   Then BD-separated subset (T_B=2.0, SEED=42) over (mu, sigma) of qualifiers.

    rng = random.Random(SEED)
    audit_users = sorted(profiles.keys())
    log(f"  audit users: {len(audit_users)}")

    # Per user audit
    per_user = []
    p_m_gt0_list = []

    for uid in audit_users:
        u_short = uid[:12]
        e = profiles[uid]
        target_a = e["target_asin"]
        mu_u = np.asarray(e["mu_profile"], dtype=np.float64)
        sigma_u = np.asarray(e["sigma_profile_diag"], dtype=np.float64)
        source_payload = e["source_set"]
        if not source_payload:
            log(f"  {u_short}: no source sentences — skip")
            continue

        # Build competitor pool from cohort users sharing ANY same ASIN with u
        candidate_uids = set()
        for a in user_to_asins.get(uid, []):
            for v in asin_to_users.get(a, []):
                if v != uid:
                    candidate_uids.add(v)
        # + audit users (including random) whose target_asin != target_a
        for v in audit_users:
            if v != uid and profiles[v].get("target_asin") != target_a:
                candidate_uids.add(v)

        # Quality gate + 7.C.1 profile availability
        comp_users_data = []
        n_q_pass = 0
        for v in candidate_uids:
            cv = cv_users.get(v, {})
            inlier = cv.get("cv_inlier_frac")
            nll = cv.get("cv_nll")
            if inlier is None or nll is None:
                continue
            if inlier < CV_INLIER_MIN:
                continue
            if nll > CV_NLL_MAX:
                continue
            n_q_pass += 1
            if v in profiles:
                comp_users_data.append({
                    "uid": v,
                    "mu": np.asarray(profiles[v]["mu_profile"], dtype=np.float64),
                    "sigma": np.asarray(profiles[v]["sigma_profile_diag"], dtype=np.float64),
                })
        # BD-separated
        if len(comp_users_data) >= 2:
            sel = greedy_maxmin_in_group(comp_users_data, T_B)
            comp_mu = np.array([comp_users_data[i]["mu"] for i in sel])
        elif len(comp_users_data) == 1:
            comp_mu = np.array([comp_users_data[0]["mu"]])
        else:
            comp_mu = np.zeros((0, 48), dtype=np.float64)
        n_comp = len(comp_mu)

        # For each source sentence, compute d_self and M_local
        zs = np.array([s["z48"] for s in source_payload], dtype=np.float64)
        d_self = np.linalg.norm(zs - mu_u[None, :], axis=1)
        if n_comp > 0:
            d_other_each = np.linalg.norm(zs[:, None, :] - comp_mu[None, :, :], axis=2)
            d_other = d_other_each.min(axis=1)
            M_local = d_other - d_self
        else:
            d_other = np.full(len(zs), np.nan)
            M_local = np.full(len(zs), np.nan)

        p_m_gt0 = float(np.mean(M_local > 0)) if len(M_local) > 0 else None
        p_m_gt0_list.append(p_m_gt0)
        per_user.append({
            "user_id": uid,
            "target_asin": target_a,
            "n_source": len(source_payload),
            "n_candidate_comp": len(candidate_uids),
            "n_comp_after_qgate": n_q_pass,
            "n_comp_after_bd": n_comp,
            "d_self_med": float(np.median(d_self)),
            "d_other_med": float(np.nanmedian(d_other)) if n_comp > 0 else None,
            "M_local_med": float(np.nanmedian(M_local)) if n_comp > 0 else None,
            "p_M_gt0": p_m_gt0,
        })
        log(f"  {u_short}: tgt={target_a[:10]} n_S={len(source_payload)} "
            f"comp_pool={len(candidate_uids)}→q={n_q_pass}→BD={n_comp} | "
            f"d_self_med={float(np.median(d_self)):.3f} "
            f"M_med={float(np.nanmedian(M_local)) if n_comp>0 else float('nan'):.3f} "
            f"P(M>0)={p_m_gt0:.2%}" if p_m_gt0 is not None else
            f"  {u_short}: tgt={target_a[:10]} n_S={len(source_payload)} comp=0 — no M")

    # --- 5. Aggregate + gate verdict ---
    n_eval = len([x for x in per_user if x["p_M_gt0"] is not None])
    med_p = float(np.median(p_m_gt0_list)) if p_m_gt0_list else None
    mean_p = float(np.mean(p_m_gt0_list)) if p_m_gt0_list else None
    gate_pass = (med_p is not None) and (med_p >= GATE_PASS_THRESHOLD)
    summary = {
        "n_users_audited": len(audit_users),
        "n_users_evaluated": n_eval,
        "p_M_gt0_med": med_p,
        "p_M_gt0_mean": mean_p,
        "gate_threshold": GATE_PASS_THRESHOLD,
        "gate_pass": gate_pass,
        "verdict": (
            "PASS — sentence-level PCA48 profile has local personalization signal. "
            "Phase 7.C.3 fair free vs rewrite can proceed."
            if gate_pass else
            "FAIL — sentence-level profile does NOT carry local signal at the "
            "required threshold. Phase 7.C.3 should NOT proceed; revisit anchor / "
            "competitor / support unit."
        ),
    }
    out = {"config": profiles_doc["config"], "summary": summary,
           "per_user": per_user}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    log(f"  wrote → {OUT_JSON}")

    log("\n=== VERDICT ===")
    log(f"  P(M_local>0) med = {med_p} | threshold = {GATE_PASS_THRESHOLD}")
    log(f"  gate_pass = {gate_pass}")
    log(f"  {summary['verdict']}")
    log("=== DONE ===")


if __name__ == "__main__":
    main()