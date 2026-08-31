"""Phase 7.I — In-Sample Syntax-Identifiable User + Discriminative Source Pool.

User goal 2026-08-31 (post 7.H clarification):
  60/20/20 split is unnecessary because the experiment goal is NOT
  "prove Gaussian generalization on unseen held-out sentences" but
  rather "construct a high-quality personalized Query benchmark cohort
  of users and discriminative source sentences for downstream rewrite".

  Methodology:
    ALL real sentences -> fit μ_u -> compute per-sentence M
    Filter users: P_u(M>0) ≥ 0.80  AND  median(M_u) > 0
    Filter sentences within selected users: M > 0 AND d_self ≤ R_95
    Output: per-user discriminative source sentences pool ready for rewrite.

  IMPORTANT CAVEAT:
    Result is IN-SAMPLE syntax-identifiable — it cannot be cited as
    "generalize to unseen held-out" without further held-out validation.

Inputs (same caches as 7.G / 7.H):
  - stage8_5_cv_users_slim.json
  - stage8_5_shared_scaler.npz
  - raw_corpus/Baby_Products_2023.jsonl
  - stage8_5_asins.json

Output:
  result/gen_query/phase7i_discriminative_source_pool.json

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    python3 gen_query/syntax_subspace_discriminative_source_pool.py
"""

from __future__ import annotations

import collections
import gzip
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
CV_SLIM = SCRATCH / "stage8_5_cv_users_slim.json"
GAUSS_CV_PATH = SCRATCH / "stage8_5_user_gaussians_cv.json"
GAUSS_PATH = SCRATCH / "stage8_5_user_gaussians.json"
SCALER_NPZ = SCRATCH / "stage8_5_shared_scaler.npz"
ASINS_PATH = SCRATCH / "stage8_5_asins.json"
RAW_REVIEWS = SCRATCH / "raw_corpus" / "Baby_Products_2023.jsonl"
REVIEWS_GZ = REPO_ROOT / "data" / "Baby_Products_2023.jsonl.gz"
OUT_JSON = REPO_ROOT / "result" / "gen_query" / "phase7i_discriminative_source_pool.json"

# Locked
SEED = "7i_v1"
MIN_TOKENS = 8
MAX_TOKENS = 60

# Q-gate
CV_INLIER_MIN = 0.85
CV_NLL_MAX = 34.0
N_REVIEWS_MIN = 15

# Cohort
N_ASIN = 80
USERS_PER_ASIN = 12
MIN_TOTAL_SENTS = 3  # at least 3 sentences per user to compute stats

# User gate
USER_P_MIN = 0.80
USER_MED_M_MIN = 0.0

# Sentence gate
SENT_M_MIN = 0.0
SENT_D_SELF_MAX_REL = "R_95"  # use per-user 95th-percentile of d_self
R_95_FALLBACK = 8.073          # theoretical sqrt(chi2(0.95, 48))


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [7I] {msg}", flush=True)


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def fit_gaussian(z48_array):
    return z48_array.mean(axis=0), z48_array.var(axis=0)


def project_one(s, nlp, all_fnames, col_idx, scaler_mean, scaler_scale,
                pca_components, pca_mean, psf):
    doc = next(nlp.pipe([s], batch_size=1, n_process=1))
    sds = list(doc.sents)
    if not sds:
        return None
    feats = [psf(sd) for sd in sds if psf(sd) is not None]
    if not feats:
        return None
    all_keys = set()
    for f in feats:
        all_keys.update(f.keys())
    mean_feats = {}
    for k in all_keys:
        vals = [f.get(k, 0.0) for f in feats]
        if all(isinstance(v, (int, float, np.integer, np.floating)) for v in vals):
            try:
                mean_feats[k] = float(np.mean(vals))
            except (TypeError, ValueError):
                pass
    vec = np.array([mean_feats.get(nm, 0.0) for nm in all_fnames], dtype=np.float64)
    v = vec[col_idx]
    v_scaled = (v - scaler_mean) / scaler_scale
    return (v_scaled - pca_mean) @ pca_components.T


def build_user_fullset(uid, user_reviews, nlp, all_fnames, col_idx,
                       scaler_mean, scaler_scale, pca_components, pca_mean, psf):
    """Collect ALL 8-60 token sentences for a user (across all ASINs)."""
    revs = user_reviews.get(uid, [])
    if not revs:
        return None
    sents = []
    for t, _ in revs:
        for s in split_sentences(t):
            toks = s.split()
            if MIN_TOKENS <= len(toks) <= MAX_TOKENS:
                sents.append(s)
    if not sents:
        return None
    z_all = []
    texts_all = []
    seen = set()
    for s in sents:
        k = s.strip().lower()
        if k in seen:
            continue
        seen.add(k)
        z = project_one(s, nlp, all_fnames, col_idx, scaler_mean,
                        scaler_scale, pca_components, pca_mean, psf)
        if z is not None:
            z_all.append(z)
            texts_all.append(s)
    if len(z_all) < MIN_TOTAL_SENTS:
        return None
    return {
        "z_all": np.asarray(z_all, dtype=np.float64),
        "texts_all": texts_all,
        "n_total": len(z_all),
    }


def main():
    log("=== Phase 7.I — In-Sample Discriminative Source Pool ===")
    log(f"  user gate: P_u(M>0) ≥ {USER_P_MIN:.0%} AND median(M_u) > {USER_MED_M_MIN}")
    log(f"  sent gate: M > {SENT_M_MIN} AND d_self ≤ R_95 (per-user)")
    log("  IMPORTANT: in-sample; not a held-out generalization proof")

    t0 = time.time()
    if CV_SLIM.exists():
        cv_users = json.load(open(CV_SLIM))
        log(f"  loaded slim cv_users: {len(cv_users)} users in {time.time()-t0:.1f}s")
    else:
        cv_doc = json.load(open(GAUSS_CV_PATH))
        cv_users = cv_doc["users"]
        log(f"  loaded full cv_users: {len(cv_users)} users in {time.time()-t0:.1f}s")
    asins_doc = json.load(open(ASINS_PATH))

    log("  building ASIN→user map (Q-gate filtered)...")
    asin_to_qual = collections.defaultdict(list)
    for entry in asins_doc["asins"]:
        a = entry["asin"]
        for u in entry.get("users_sampled", []):
            cv = cv_users.get(u, {})
            inl = cv.get("inlier_frac")
            nll = cv.get("cv_nll")
            nrev = cv.get("n_reviews", 0)
            if inl is None or nll is None:
                continue
            if inl < CV_INLIER_MIN or nll > CV_NLL_MAX or nrev < N_REVIEWS_MIN:
                continue
            asin_to_qual[a].append(u)
    qualified_asins = [(a, len(u)) for a, u in asin_to_qual.items() if len(u) >= USERS_PER_ASIN]
    qualified_asins.sort(key=lambda x: -x[1])
    log(f"  ASINs with >={USERS_PER_ASIN} qual users: {len(qualified_asins)}")
    if not qualified_asins:
        raise AssertionError(f"[7I] no ASIN with >={USERS_PER_ASIN} qual users")

    target_asins = [a for a, _ in qualified_asins[:N_ASIN]]
    target_user_set = set()
    for a in target_asins:
        target_user_set.update(asin_to_qual[a][:USERS_PER_ASIN])
    log(f"  selected {len(target_asins)} ASINs; unique cohort users: {len(target_user_set)}")

    if SCALER_NPZ.exists():
        log("  loading shared scaler npz")
        gdoc = np.load(SCALER_NPZ, allow_pickle=True)
        all_fnames = list(gdoc["feature_names_ordered"])
        fnames_f3 = list(gdoc["fnames_f3"])
        scaler_mean = np.asarray(gdoc["scaler_mean"], dtype=np.float64)
        scaler_scale = np.asarray(gdoc["scaler_scale"], dtype=np.float64)
        pca_components = np.asarray(gdoc["pca_components"], dtype=np.float64)
        pca_mean = np.asarray(gdoc["pca_mean"], dtype=np.float64)
        gdoc.close()
    else:
        gdoc = json.load(open(GAUSS_PATH))
        all_fnames = gdoc["feature_names_ordered"]
        fnames_f3 = gdoc["fnames_f3"]
        scaler_mean = np.asarray(gdoc["scaler_mean"], dtype=np.float64)
        scaler_scale = np.asarray(gdoc["scaler_scale"], dtype=np.float64)
        pca_components = np.asarray(gdoc["pca_components"], dtype=np.float64)
        pca_mean = np.asarray(gdoc["pca_mean"], dtype=np.float64)
    col_idx = [all_fnames.index(nm) for nm in fnames_f3]

    import spacy
    sys.path.insert(0, str(REPO_ROOT / "common"))
    from syntactic_features import per_sentence_features_v2 as psf
    nlp = spacy.load("en_core_web_sm")
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)

    log(f"  scanning reviews for {len(target_user_set)} users...")
    user_reviews = collections.defaultdict(list)
    target_set = target_user_set
    target_asin_set = set(target_asins)
    n_records = 0
    if RAW_REVIEWS.exists():
        log(f"  reading from {RAW_REVIEWS.name} (plain jsonl)")
        open_fn = lambda: open(RAW_REVIEWS, "r", encoding="utf-8")
    else:
        log(f"  reading from {REVIEWS_GZ.name} (gzip)")
        open_fn = lambda: gzip.open(REVIEWS_GZ, "rt", encoding="utf-8")
    with open_fn() as f:
        for line in f:
            r = json.loads(line)
            uid = r.get("reviewerID") or r.get("user_id")
            asin = r.get("asin") or r.get("parent_asin")
            if uid in target_set and asin in target_asin_set and r.get("text"):
                user_reviews[uid].append((r["text"], asin))
            n_records += 1
            if n_records % 2_000_000 == 0:
                log(f"    {n_records/1e6:.1f}M scanned, "
                    f"{len(user_reviews)}/{len(target_user_set)} users seen")
    log(f"  scanned {n_records}, found reviews for {len(user_reviews)}/{len(target_user_set)} users")

    # Per-ASIN process
    per_asin_results = []
    selected_users = []  # list of (asin, uid, profile_R_95, n_disc_sents, [disc_sentences])
    overall_n_in_cohort = 0
    overall_n_pass_user = 0
    overall_n_disc_sents = 0

    for a in target_asins:
        cohort_uids = asin_to_qual[a][:USERS_PER_ASIN]
        per_user = {}
        for uid in cohort_uids:
            sets = build_user_fullset(
                uid, user_reviews, nlp, all_fnames, col_idx,
                scaler_mean, scaler_scale, pca_components, pca_mean, psf,
            )
            if sets is None:
                continue
            per_user[uid] = sets
        if len(per_user) < 2:
            log(f"  ASIN {a}: only {len(per_user)} valid users — skip")
            continue

        # Fit profile Gaussian per user (μ, σ) from z_all
        prof_gauss = {u: fit_gaussian(per_user[u]["z_all"]) for u in per_user}

        # Per-user: compute d_self, d_other, M, R_95
        per_user_stats = {}
        for uid in per_user:
            mu_u = prof_gauss[uid][0]
            other_uids = [v for v in per_user if v != uid]
            other_mu = np.asarray([prof_gauss[v][0] for v in other_uids])
            z_all = per_user[uid]["z_all"]
            d_self = np.linalg.norm(z_all - mu_u[None, :], axis=1)
            d_other_each = np.linalg.norm(
                z_all[:, None, :] - other_mu[None, :, :], axis=2
            )
            d_other = d_other_each.min(axis=1)
            M = d_other - d_self

            # R_95 from this user's own d_self distribution (95th pct) if n>=5 else fallback
            if len(d_self) >= 5:
                R_95 = float(np.quantile(d_self, 0.95))
            else:
                R_95 = R_95_FALLBACK

            P_M_gt0 = float((M > 0).mean())
            M_med = float(np.median(M))

            # User gate
            pass_user = (P_M_gt0 >= USER_P_MIN and M_med > USER_MED_M_MIN)

            # Sentence gate (always computed; sentences are kept if user passes)
            disc_mask = (M > SENT_M_MIN) & (d_self <= R_95)
            n_disc = int(disc_mask.sum())
            disc_indices = np.where(disc_mask)[0]

            per_user_stats[uid] = {
                "n_total": int(len(z_all)),
                "n_M_pos": int((M > 0).sum()),
                "P_M_gt0": P_M_gt0,
                "M_med": M_med,
                "M_mean": float(np.mean(M)),
                "d_self_med": float(np.median(d_self)),
                "d_other_med": float(np.median(d_other)),
                "R_95": R_95,
                "n_discriminative": n_disc,
                "pass_user": pass_user,
                "_disc_indices": disc_indices.tolist(),
                "_d_self": d_self.tolist(),
                "_d_other": d_other.tolist(),
                "_M": M.tolist(),
            }

            if pass_user:
                selected_users.append({
                    "asin": a,
                    "uid": uid,
                    "n_total": int(len(z_all)),
                    "P_M_gt0": P_M_gt0,
                    "M_med": M_med,
                    "R_95": R_95,
                    "n_discriminative": n_disc,
                    "discriminative_sentences": [
                        {
                            "text": per_user[uid]["texts_all"][i],
                            "M": float(M[i]),
                            "d_self": float(d_self[i]),
                            "d_other": float(d_other[i]),
                        } for i in disc_indices
                    ],
                })

        n_pass_user = sum(1 for u in per_user_stats if per_user_stats[u]["pass_user"])
        n_disc = sum(per_user_stats[u]["n_discriminative"] for u in per_user_stats)
        per_asin_results.append({
            "asin": a,
            "n_users_in_cohort": len(per_user_stats),
            "n_pass_user": n_pass_user,
            "frac_pass_user": n_pass_user / len(per_user_stats) if per_user_stats else 0,
            "n_discriminative_total": n_disc,
            "per_user_summary": {
                u: {
                    "n_total": per_user_stats[u]["n_total"],
                    "P_M_gt0": per_user_stats[u]["P_M_gt0"],
                    "M_med": per_user_stats[u]["M_med"],
                    "R_95": per_user_stats[u]["R_95"],
                    "n_discriminative": per_user_stats[u]["n_discriminative"],
                    "pass_user": per_user_stats[u]["pass_user"],
                } for u in per_user_stats
            },
        })
        overall_n_in_cohort += len(per_user_stats)
        overall_n_pass_user += n_pass_user
        overall_n_disc_sents += n_disc
        log(f"  ASIN {a}: cohort={len(per_user_stats)}u, "
            f"pass_user={n_pass_user}/{len(per_user_stats)} ({n_pass_user/len(per_user_stats):.0%}), "
            f"disc_sents={n_disc}")

    if not per_asin_results:
        raise AssertionError("[7I] no ASIN produced results")

    summary = {
        "config": {
            "description": ("Phase 7.I: in-sample syntax-identifiable user & "
                            "discriminative source sentence pool. NOT held-out."),
            "caveat": ("In-sample only. Cannot claim held-out generalization. "
                       "Use as benchmark cohort construction."),
            "n_target_asins_target": N_ASIN,
            "users_per_asin": USERS_PER_ASIN,
            "user_gate": {"P_M_gt0_min": USER_P_MIN, "M_med_min": USER_MED_M_MIN},
            "sent_gate": {"M_min": SENT_M_MIN, "d_self_max": "R_95_per_user"},
            "R_95_method": ("per-user 95th percentile of in-sample d_self "
                            f"(>=5 samples), else {R_95_FALLBACK}"),
        },
        "n_asins_evaluated": len(per_asin_results),
        "n_users_in_cohort_total": overall_n_in_cohort,
        "n_users_pass_gate_total": overall_n_pass_user,
        "frac_users_pass_gate_total": (overall_n_pass_user / overall_n_in_cohort
                                        if overall_n_in_cohort else 0),
        "n_discriminative_sentences_total": overall_n_disc_sents,
        "n_unique_users_with_disc_sents": len(selected_users),
        "P_M_gt0_per_user_distribution": {
            "mean": float(np.mean([u["P_M_gt0"] for u in selected_users])) if selected_users else None,
            "median": float(np.median([u["P_M_gt0"] for u in selected_users])) if selected_users else None,
            "q25": float(np.quantile([u["P_M_gt0"] for u in selected_users], 0.25)) if selected_users else None,
            "q75": float(np.quantile([u["P_M_gt0"] for u in selected_users], 0.75)) if selected_users else None,
        },
        "M_med_per_user_distribution": {
            "mean": float(np.mean([u["M_med"] for u in selected_users])) if selected_users else None,
            "median": float(np.median([u["M_med"] for u in selected_users])) if selected_users else None,
        },
        "n_discriminative_per_user_distribution": {
            "mean": float(np.mean([u["n_discriminative"] for u in selected_users])) if selected_users else None,
            "median": float(np.median([u["n_discriminative"] for u in selected_users])) if selected_users else None,
            "max": int(max([u["n_discriminative"] for u in selected_users])) if selected_users else None,
        },
        "verdict": ("GO — benchmark cohort has " + str(len(selected_users)) +
                    " syntax-identifiable users with " +
                    str(overall_n_disc_sents) +
                    " total discriminative source sentences ready for rewrite."
                    if selected_users else
                    "FAIL — no user passes the in-sample syntax-identifiable gate"),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    full = {"summary": summary, "per_asin_results": per_asin_results,
            "selected_users": selected_users}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(full, f, indent=1)
    log(f"  wrote → {OUT_JSON}")

    log("\n=== OVERALL ===")
    log(f"  ASINs evaluated: {len(per_asin_results)}")
    log(f"  Users in cohort: {overall_n_in_cohort}")
    log(f"  Users pass gate: {overall_n_pass_user} ({summary['frac_users_pass_gate_total']:.1%})")
    log(f"  Unique users with ≥1 disc sentence: {len(selected_users)}")
    log(f"  Total discriminative sentences: {overall_n_disc_sents}")
    if selected_users:
        log(f"  Per-user disc sents: mean={summary['n_discriminative_per_user_distribution']['mean']:.1f}, "
            f"median={summary['n_discriminative_per_user_distribution']['median']:.0f}, "
            f"max={summary['n_discriminative_per_user_distribution']['max']}")
        log(f"  Per-user P(M>0): mean={summary['P_M_gt0_per_user_distribution']['mean']:.1%}, "
            f"median={summary['P_M_gt0_per_user_distribution']['median']:.1%}")
        log(f"  Per-user median(M): mean={summary['M_med_per_user_distribution']['mean']:.2f}, "
            f"median={summary['M_med_per_user_distribution']['median']:.2f}")
    log(f"  {summary['verdict']}")
    log("=== DONE ===")


if __name__ == "__main__":
    main()