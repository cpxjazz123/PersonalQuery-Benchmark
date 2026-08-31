"""Phase 7.G — ASIN-Stratified Held-Out M>0 Audit.

User goal 2026-08-31:
  Within each ASIN (same product), pick 5+ NLL<=34 + inlier>=0.85 + n_rev>=15
  users. For each user, fit sentence-level Gaussian on 60% profile sentences
  (drawn from user's GLOBAL reviews, all ASINs). Evaluate M>0 on the held-out
  40% sentences against the cohort of OTHER users who reviewed the SAME ASIN.

  d_self = ||z_heldout - μ_u_global|| — user's GLOBAL syntactic style match
  d_other = min_{v≠u in cohort} ||z_heldout - μ_v_global||
  M = d_other - d_self

  Goal: P(M>0) per ASIN — direct test of whether user syntax can be
  distinguished when product semantics is FIXED (because cohort = same-ASIN
  reviewers), but using user's full syntactic style.

Pipeline:
  1. Pick N_ASIN target ASINs from stage8_5_asins (prioritize those with most
     NLL<=34 users; cap at N_ASIN=20, users per ASIN capped at USERS_PER_ASIN=8).
  2. Scan reviews for all target users across all ASINs.
  3. Per user: hash-split user's GLOBAL sentences (60/40) into profile/heldout.
  4. Per ASIN cohort: fit μ_u_global from profile_set z48 for each user.
  5. For each held-out sentence of each user, compute M vs same-ASIN cohort.
  6. Aggregate: per ASIN P(M>0), median M, per-user breakdown.

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    python3 gen_query/syntax_subspace_asin_stratified_audit.py

Output:
    result/gen_query/phase7g_asin_stratified_audit.json
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
GAUSS_PATH = SCRATCH / "stage8_5_user_gaussians.json"
GAUSS_CV_PATH = SCRATCH / "stage8_5_user_gaussians_cv.json"
ASINS_PATH = SCRATCH / "stage8_5_asins.json"
# User-instructed 2026-08-31: prefer the decompressed plain jsonl over the gz
# version — saves 30s+ gzip work per run.
RAW_REVIEWS = SCRATCH / "raw_corpus" / "Baby_Products_2023.jsonl"
REVIEWS_GZ = REPO_ROOT / "data" / "Baby_Products_2023.jsonl.gz"
OUT_JSON = REPO_ROOT / "result" / "gen_query" / "phase7g_asin_stratified_audit.json"
LOG_PATH = Path("/home/wlia0047/hj82_scratch2/wenyu/logs/phase7g_asin_stratified_audit.log")

# Locked
SEED = "7g_v1"
PROFILE_FRAC = 0.60
MIN_TOKENS = 8
MAX_TOKENS = 60

# Q-gate
CV_INLIER_MIN = 0.85
CV_NLL_MAX = 34.0
N_REVIEWS_MIN = 15

# ASIN selection
N_ASIN = 5     # SMOKE; increase to 20 for full run
USERS_PER_ASIN = 8  # larger cohort for stable P(M>0) aggregate
MIN_HELD_OUT_SENTS = 1  # at least 1 held-out sentence per user (global reviews)
MIN_PROFILE_SENTS = 1   # at least 1 profile sentence per user

# Cohort filter: require per-pair distinct BD to ensure μ distinguishable
# (skip if pairwise BD too low — diagnostic only, doesn't fail the run)


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [7G] {msg}", flush=True)


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def sentence_bucket(text):
    """60% profile, 40% held-out."""
    h = int(hashlib.sha1((SEED + "|" + text.strip().lower()).encode()).hexdigest(), 16)
    return "profile" if (h % 100) < int(PROFILE_FRAC * 100) else "heldout"


def fit_gaussian(z48_array):
    return z48_array.mean(axis=0), z48_array.var(axis=0)


# ---------- per-user 60/40 split ----------
def build_user_2sets(uid, user_reviews, nlp, all_fnames, col_idx,
                     scaler_mean, scaler_scale, pca_components, pca_mean, psf):
    """Build profile_set / heldout_set (60/40 hash-split) for a user.

    Uses ALL of user's reviews (across all ASINs). The user's μ represents
    their GLOBAL syntactic style. The cohort (d_other) is determined at the
    per-ASIN level — held-out M compares against other users who also reviewed
    THAT ASIN. So d_self = global style match, d_other = ASIN-cohort match.
    """
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
    bucketed = collections.defaultdict(list)
    for s in sents:
        bucketed[sentence_bucket(s)].append(s)
    z_p, z_h = [], []
    seen_p, seen_h = set(), set()
    for s in bucketed["profile"]:
        k = s.strip().lower()
        if k in seen_p: continue
        seen_p.add(k)
        z = project_one(s, nlp, all_fnames, col_idx, scaler_mean,
                        scaler_scale, pca_components, pca_mean, psf)
        if z is not None: z_p.append(z)
    for s in bucketed["heldout"]:
        k = s.strip().lower()
        if k in seen_h: continue
        seen_h.add(k)
        z = project_one(s, nlp, all_fnames, col_idx, scaler_mean,
                        scaler_scale, pca_components, pca_mean, psf)
        if z is not None: z_h.append(z)
    if len(z_p) < MIN_PROFILE_SENTS or len(z_h) < MIN_HELD_OUT_SENTS:
        return None
    return {
        "z_profile": np.asarray(z_p, dtype=np.float64),
        "z_heldout": np.asarray(z_h, dtype=np.float64),
        "n_profile": len(z_p),
        "n_heldout": len(z_h),
    }


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


# ---------- main ----------
def main():
    log("=== Phase 7.G — ASIN-Stratified Held-Out M>0 Audit ===")
    log(f"  goal: P(M>0) on held-out sentences, cohort = same-ASIN users")

    # Load CV (slim JSON cache, user-instructed 2026-08-31), then ASINs
    t0 = time.time()
    if (SCRATCH / "stage8_5_cv_users_slim.json").exists():
        cv_users = json.load(open(SCRATCH / "stage8_5_cv_users_slim.json"))
        log(f"  loaded slim cv_users: {len(cv_users)} users in {time.time()-t0:.1f}s")
    else:
        cv_doc = json.load(open(GAUSS_CV_PATH))
        cv_users = cv_doc["users"]
        log(f"  loaded full cv_users (no slim cache): {len(cv_users)} users in {time.time()-t0:.1f}s")
    asins_doc = json.load(open(ASINS_PATH))

    # Build (asin -> [qual_uids]) map, filtered by Q-gate
    log("  building ASIN→user map (Q-gate filtered)...")
    asin_to_qual = collections.defaultdict(list)
    for entry in asins_doc["asins"]:
        a = entry["asin"]
        for u in entry.get("users_sampled", []):
            cv = cv_users.get(u, {})
            inl = cv.get("inlier_frac")  # slim cache uses 'inlier_frac'
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
        raise AssertionError(f"[7G] no ASIN with >={USERS_PER_ASIN} qual users")

    target_asins = [a for a, _ in qualified_asins[:N_ASIN]]
    target_user_set = set()
    for a in target_asins:
        target_user_set.update(asin_to_qual[a][:USERS_PER_ASIN])
    log(f"  selected {len(target_asins)} ASINs; "
        f"unique cohort users: {len(target_user_set)}")

    # Load gaussian cache for scaler/PCA (prefer slim npz if exists)
    npz_path = SCRATCH / "stage8_5_shared_scaler.npz"
    if npz_path.exists():
        log(f"  loading shared scaler npz")
        gdoc = np.load(npz_path, allow_pickle=True)
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

    # spaCy + psf
    import spacy
    sys.path.insert(0, str(REPO_ROOT / "common"))
    from syntactic_features import per_sentence_features_v2 as psf
    nlp = spacy.load("en_core_web_sm")
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)

    # Scan reviews once: collect (uid, text, asin) for all target users
    log(f"  scanning reviews for {len(target_user_set)} users...")
    user_reviews = collections.defaultdict(list)
    target_set = target_user_set
    target_asin_set = set(target_asins)
    n_records = 0
    # Prefer decompressed plain jsonl (user-instructed 2026-08-31) over .gz
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
    asin_results = []
    overall_M = []
    overall_M_pos = 0
    overall_n = 0
    for a in target_asins:
        cohort_uids = asin_to_qual[a][:USERS_PER_ASIN]
        # Build per-user 2-set from GLOBAL reviews (all ASINs).
        # d_self will use user's global μ; d_other uses cohort of OTHER users
        # who also reviewed THIS ASIN.
        per_user = {}
        build_log = []
        for uid in cohort_uids:
            sets = build_user_2sets(
                uid, user_reviews, nlp, all_fnames, col_idx,
                scaler_mean, scaler_scale, pca_components, pca_mean, psf,
            )
            if sets is None:
                build_log.append(f"{uid[:10]}=0")
                continue
            per_user[uid] = sets
            build_log.append(f"{uid[:10]}={sets['n_profile']}/{sets['n_heldout']}")
        if len(per_user) < 2:
            log(f"  ASIN {a}: only {len(per_user)} valid users — skip "
                f"({', '.join(build_log)})")
            continue
        # Fit profile Gaussian per user
        prof_gauss = {u: fit_gaussian(per_user[u]["z_profile"]) for u in per_user}
        # For each held-out sentence of each user, compute M vs cohort
        per_user_stats = {}
        for uid in per_user:
            mu_u = prof_gauss[uid][0]
            other_uids = [v for v in per_user if v != uid]
            if not other_uids:
                continue
            other_mu = np.asarray([prof_gauss[v][0] for v in other_uids])
            z_h = per_user[uid]["z_heldout"]
            d_self = np.linalg.norm(z_h - mu_u[None, :], axis=1)
            d_other_each = np.linalg.norm(
                z_h[:, None, :] - other_mu[None, :, :], axis=2
            )
            d_other = d_other_each.min(axis=1)
            M = d_other - d_self
            n_pos = int((M > 0).sum())
            per_user_stats[uid] = {
                "n_heldout": len(z_h),
                "n_M_pos": n_pos,
                "P_M_gt0": n_pos / len(M),
                "M_med": float(np.median(M)),
                "M_mean": float(np.mean(M)),
                "M_std": float(np.std(M)),
                "d_self_med": float(np.median(d_self)),
                "d_other_med": float(np.median(d_other)),
            }
        # ASIN-level aggregate
        asin_M_med = float(np.median([per_user_stats[u]["M_med"] for u in per_user_stats]))
        asin_P_med = float(np.median([per_user_stats[u]["P_M_gt0"] for u in per_user_stats]))
        asin_P_mean = float(np.mean([per_user_stats[u]["P_M_gt0"] for u in per_user_stats]))
        # Weighted (by n_heldout) for stable estimate
        n_h_total = sum(per_user_stats[u]["n_heldout"] for u in per_user_stats)
        n_pos_total = sum(per_user_stats[u]["n_M_pos"] for u in per_user_stats)
        asin_P_weighted = n_pos_total / n_h_total if n_h_total > 0 else 0
        asin_results.append({
            "asin": a,
            "n_users": len(per_user),
            "n_heldout_total": n_h_total,
            "n_M_pos_total": n_pos_total,
            "P_M_gt0_weighted": asin_P_weighted,
            "P_M_gt0_median_per_user": asin_P_med,
            "P_M_gt0_mean_per_user": asin_P_mean,
            "M_med_per_user_median": asin_M_med,
            "per_user": per_user_stats,
        })
        overall_M_pos += n_pos_total
        overall_n += n_h_total
        log(f"  ASIN {a}: {len(per_user)}u | P(M>0)_w={asin_P_weighted:.1%} "
            f"med={asin_P_med:.1%} mean={asin_P_mean:.1%} | n_heldout={n_h_total}")

    if not asin_results:
        raise AssertionError("[7G] no ASIN produced results")

    # Overall stats
    all_P_med = [r["P_M_gt0_median_per_user"] for r in asin_results]
    all_P_weighted = [r["P_M_gt0_weighted"] for r in asin_results]
    summary = {
        "config": {
            "description": ("Phase 7.G: ASIN-stratified held-out M>0 audit. "
                            "Cohort = same-ASIN users only (product semantics fixed). "
                            "Per-user 60/40 hash-split → profile Gaussian → held-out M."),
            "n_target_asins_target": N_ASIN,
            "users_per_asin": USERS_PER_ASIN,
            "profile_frac": PROFILE_FRAC,
            "split_seed": SEED,
        },
        "n_asins_evaluated": len(asin_results),
        "overall_P_M_gt0_weighted": overall_M_pos / overall_n if overall_n else 0,
        "overall_n_heldout": overall_n,
        "overall_n_M_pos": overall_M_pos,
        "P_M_gt0_median_across_asins": float(np.median(all_P_med)),
        "P_M_gt0_mean_across_asins": float(np.mean(all_P_med)),
        "P_M_gt0_weighted_median_across_asins": float(np.median(all_P_weighted)),
        "P_M_gt0_weighted_mean_across_asins": float(np.mean(all_P_weighted)),
        "verdict": (
            "GO — same-ASIN cohort separates users (median P(M>0) >= 30%)"
            if float(np.median(all_P_med)) >= 0.30
            else "PARTIAL — even same-ASIN cohort has limited separation"
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    full = {"summary": summary, "asin_results": asin_results}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(full, f, indent=1)
    log(f"  wrote → {OUT_JSON}")
    log("\n=== OVERALL ===")
    log(f"  ASINs evaluated: {len(asin_results)}")
    log(f"  P(M>0) overall (weighted): {summary['overall_P_M_gt0_weighted']:.1%}")
    log(f"  P(M>0) median across ASINs (per-user median): "
        f"{summary['P_M_gt0_median_across_asins']:.1%}")
    log(f"  P(M>0) median across ASINs (weighted): "
        f"{summary['P_M_gt0_weighted_median_across_asins']:.1%}")
    log(f"  {summary['verdict']}")
    log("=== DONE ===")


if __name__ == "__main__":
    main()