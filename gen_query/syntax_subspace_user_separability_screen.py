"""Phase 7.H — Syntax-Identifiable User Screening (60/20/20 split).

User goal 2026-08-31 (post 4-stage NO-GO feedback):
  PCA48 sentence-level CANNOT separate ALL users; instead we should FILTER
  to syntax-identifiable users. Definition:

    P_u(M>0)_validation ≥ 0.80  AND  median(M_u)_validation > 0

  Validation set is INDEPENDENT from test set — never both from same sentences.

Pipeline (per user):
  1. Hash-split all 8-60 token sentences (across all ASINs) into:
       profile (60%)     -> fit μ_u^prof
       validation (20%)  -> screen: P_u(M>0) ≥ 0.80 AND median M > 0 ?
       test (20%)        -> verify retained users
  2. Cohort = same-ASIN reviewers (product semantics fixed; d_other uses
     OTHER users who reviewed the same ASIN).
  3. d_self = ||z - μ_u^prof||, d_other = min_{v≠u in cohort} ||z - μ_v^prof||
  4. Report:
     - n_users_total   (Q-gate qual, ASIN cohort ≥ USERS_PER_ASIN)
     - n_users_in_cohort (≥ 2 valid in same ASIN cohort)
     - n_pass_validation (P_u ≥ 0.80 AND median M > 0)
     - n_pass_validation_AND_test (test P_u ≥ 0.80 AND median M > 0)
     - retention_rate = n_pass_validation_AND_test / n_pass_validation
     - per-ASIN breakdown

Inputs:
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_cv_users_slim.json
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_shared_scaler.npz
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/raw_corpus/Baby_Products_2023.jsonl
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_asins.json

Output:
  result/gen_query/phase7h_user_separability_screen.json

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    python3 gen_query/syntax_subspace_user_separability_screen.py
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
OUT_JSON = REPO_ROOT / "result" / "gen_query" / "phase7h_user_separability_screen.json"

# Locked
SEED = "7h_v1"
PROFILE_FRAC = 0.60
VAL_FRAC = 0.20
TEST_FRAC = 0.20
MIN_TOKENS = 8
MAX_TOKENS = 60

# Q-gate
CV_INLIER_MIN = 0.85
CV_NLL_MAX = 34.0
N_REVIEWS_MIN = 15

# Screening
N_ASIN = 80      # FULL run: scan 80 ASINs (~960 users) so enough pass the
USERS_PER_ASIN = 12  # val/test sentence count threshold
MIN_PROFILE_SENTS = 1  # 1 sentence enough to fit μ (PCA48 still well-defined)
MIN_VAL_SENTS = 1      # at least 1 val sentence
MIN_TEST_SENTS = 1     # at least 1 test sentence

# User-separable gate
VAL_P_MIN = 0.80        # P_u(M>0) ≥ 80% on validation
VAL_MED_M_MIN = 0.0     # median(M_u) > 0 on validation


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [7H] {msg}", flush=True)


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def sentence_bucket_3way(text):
    """Deterministic 60/20/20 bucketing by text SHA1 hash.

    Returns one of {'profile', 'val', 'test'}.
    """
    h = int(hashlib.sha1((SEED + "|" + text.strip().lower()).encode()).hexdigest(), 16)
    b = h % 100
    if b < int(PROFILE_FRAC * 100):
        return "profile"
    elif b < int((PROFILE_FRAC + VAL_FRAC) * 100):
        return "val"
    else:
        return "test"


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


def build_user_3sets(uid, user_reviews, nlp, all_fnames, col_idx,
                     scaler_mean, scaler_scale, pca_components, pca_mean, psf):
    """Build profile_set / val_set / test_set (60/20/20) for a user."""
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
        bucketed[sentence_bucket_3way(s)].append(s)
    z_p, z_v, z_t = [], [], []
    seen_p, seen_v, seen_t = set(), set(), set()
    for s in bucketed["profile"]:
        k = s.strip().lower()
        if k in seen_p:
            continue
        seen_p.add(k)
        z = project_one(s, nlp, all_fnames, col_idx, scaler_mean,
                        scaler_scale, pca_components, pca_mean, psf)
        if z is not None:
            z_p.append(z)
    for s in bucketed["val"]:
        k = s.strip().lower()
        if k in seen_v:
            continue
        seen_v.add(k)
        z = project_one(s, nlp, all_fnames, col_idx, scaler_mean,
                        scaler_scale, pca_components, pca_mean, psf)
        if z is not None:
            z_v.append(z)
    for s in bucketed["test"]:
        k = s.strip().lower()
        if k in seen_t:
            continue
        seen_t.add(k)
        z = project_one(s, nlp, all_fnames, col_idx, scaler_mean,
                        scaler_scale, pca_components, pca_mean, psf)
        if z is not None:
            z_t.append(z)
    if (len(z_p) < MIN_PROFILE_SENTS or
            len(z_v) < MIN_VAL_SENTS or
            len(z_t) < MIN_TEST_SENTS):
        return None
    return {
        "z_profile": np.asarray(z_p, dtype=np.float64),
        "z_val": np.asarray(z_v, dtype=np.float64),
        "z_test": np.asarray(z_t, dtype=np.float64),
        "n_profile": len(z_p),
        "n_val": len(z_v),
        "n_test": len(z_t),
    }


def eval_on_set(z_set, mu_self, other_mu):
    """Compute per-sentence M = d_other - d_self on a given z_set."""
    if z_set is None or len(z_set) == 0:
        return None
    d_self = np.linalg.norm(z_set - mu_self[None, :], axis=1)
    d_other_each = np.linalg.norm(
        z_set[:, None, :] - other_mu[None, :, :], axis=2
    )
    d_other = d_other_each.min(axis=1)
    M = d_other - d_self
    return {
        "n": int(len(z_set)),
        "n_M_pos": int((M > 0).sum()),
        "P_M_gt0": float((M > 0).mean()),
        "M_med": float(np.median(M)),
        "M_mean": float(np.mean(M)),
        "M_std": float(np.std(M)),
        "d_self_med": float(np.median(d_self)),
        "d_other_med": float(np.median(d_other)),
    }


def main():
    log("=== Phase 7.H — Syntax-Identifiable User Screen (60/20/20) ===")
    log(f"  gate: P_u(M>0)_val ≥ {VAL_P_MIN:.0%} AND median(M_u)_val > {VAL_MED_M_MIN}")

    # Load CV slim + ASINs
    t0 = time.time()
    if CV_SLIM.exists():
        cv_users = json.load(open(CV_SLIM))
        log(f"  loaded slim cv_users: {len(cv_users)} users in {time.time()-t0:.1f}s")
    else:
        cv_doc = json.load(open(GAUSS_CV_PATH))
        cv_users = cv_doc["users"]
        log(f"  loaded full cv_users (no slim cache): {len(cv_users)} users in {time.time()-t0:.1f}s")
    asins_doc = json.load(open(ASINS_PATH))

    # Build (asin -> [qual_uids]) map
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
        raise AssertionError(f"[7H] no ASIN with >={USERS_PER_ASIN} qual users")

    target_asins = [a for a, _ in qualified_asins[:N_ASIN]]
    target_user_set = set()
    for a in target_asins:
        target_user_set.update(asin_to_qual[a][:USERS_PER_ASIN])
    log(f"  selected {len(target_asins)} ASINs; unique cohort users: {len(target_user_set)}")

    # Load scaler/PCA
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

    # spaCy + psf
    import spacy
    sys.path.insert(0, str(REPO_ROOT / "common"))
    from syntactic_features import per_sentence_features_v2 as psf
    nlp = spacy.load("en_core_web_sm")
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)

    # Scan reviews
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
    overall_n_users_in_cohort = 0
    overall_n_pass_val = 0
    overall_n_pass_val_and_test = 0
    overall_val_P_per_user = []
    overall_test_P_per_user = []
    overall_val_med_M_per_user = []
    overall_test_med_M_per_user = []

    for a in target_asins:
        cohort_uids = asin_to_qual[a][:USERS_PER_ASIN]
        per_user_sets = {}
        build_log = []
        for uid in cohort_uids:
            sets = build_user_3sets(
                uid, user_reviews, nlp, all_fnames, col_idx,
                scaler_mean, scaler_scale, pca_components, pca_mean, psf,
            )
            if sets is None:
                build_log.append(f"{uid[:10]}=0")
                continue
            per_user_sets[uid] = sets
            build_log.append(f"{uid[:10]}={sets['n_profile']}/{sets['n_val']}/{sets['n_test']}")
        if len(per_user_sets) < 2:
            log(f"  ASIN {a}: only {len(per_user_sets)} valid users — skip ({', '.join(build_log)})")
            continue

        # Fit profile Gaussians (from profile set only)
        prof_gauss = {u: fit_gaussian(per_user_sets[u]["z_profile"]) for u in per_user_sets}

        # Eval on val AND test for each user (cohort μ uses profile, sentences are val/test)
        per_user_stats = {}
        for uid in per_user_sets:
            mu_u = prof_gauss[uid][0]
            other_uids = [v for v in per_user_sets if v != uid]
            other_mu = np.asarray([prof_gauss[v][0] for v in other_uids])
            val_stats = eval_on_set(per_user_sets[uid]["z_val"], mu_u, other_mu)
            test_stats = eval_on_set(per_user_sets[uid]["z_test"], mu_u, other_mu)
            if val_stats is None or test_stats is None:
                continue
            per_user_stats[uid] = {
                "n_profile": int(per_user_sets[uid]["n_profile"]),
                "n_val": val_stats["n"],
                "n_test": test_stats["n"],
                "val": val_stats,
                "test": test_stats,
                "pass_val": (val_stats["P_M_gt0"] >= VAL_P_MIN and val_stats["M_med"] > VAL_MED_M_MIN),
                "pass_test": (test_stats["P_M_gt0"] >= VAL_P_MIN and test_stats["M_med"] > VAL_MED_M_MIN),
            }

        if not per_user_stats:
            log(f"  ASIN {a}: per_user_stats empty — skip")
            continue

        n_users = len(per_user_stats)
        n_pass_val = sum(1 for u in per_user_stats if per_user_stats[u]["pass_val"])
        n_pass_val_and_test = sum(1 for u in per_user_stats
                                  if per_user_stats[u]["pass_val"] and per_user_stats[u]["pass_test"])

        per_asin_results.append({
            "asin": a,
            "n_users_in_cohort": n_users,
            "n_pass_val": n_pass_val,
            "n_pass_val_and_test": n_pass_val_and_test,
            "frac_pass_val": n_pass_val / n_users if n_users else 0,
            "frac_pass_val_and_test": n_pass_val_and_test / n_users if n_users else 0,
            "retention_rate": n_pass_val_and_test / n_pass_val if n_pass_val else None,
            "per_user": per_user_stats,
        })
        overall_n_users_in_cohort += n_users
        overall_n_pass_val += n_pass_val
        overall_n_pass_val_and_test += n_pass_val_and_test
        for u in per_user_stats:
            overall_val_P_per_user.append(per_user_stats[u]["val"]["P_M_gt0"])
            overall_test_P_per_user.append(per_user_stats[u]["test"]["P_M_gt0"])
            overall_val_med_M_per_user.append(per_user_stats[u]["val"]["M_med"])
            overall_test_med_M_per_user.append(per_user_stats[u]["test"]["M_med"])

        log(f"  ASIN {a}: {n_users}u | pass_val={n_pass_val}/{n_users} ({n_pass_val/n_users:.0%}) "
            f"| pass_val∩test={n_pass_val_and_test}/{n_users} ({n_pass_val_and_test/n_users:.0%})")

    if not per_asin_results:
        raise AssertionError("[7H] no ASIN produced results")

    # Overall aggregate
    summary = {
        "config": {
            "description": ("Phase 7.H: Syntax-Identifiable User Screen. "
                            "60/20/20 hash split (profile/val/test). "
                            "User gate = P_u(M>0)_val ≥ 0.80 AND median(M_u)_val > 0."),
            "n_target_asins_target": N_ASIN,
            "users_per_asin": USERS_PER_ASIN,
            "profile_frac": PROFILE_FRAC,
            "val_frac": VAL_FRAC,
            "test_frac": TEST_FRAC,
            "val_p_min": VAL_P_MIN,
            "val_med_M_min": VAL_MED_M_MIN,
            "split_seed": SEED,
        },
        "n_asins_evaluated": len(per_asin_results),
        "n_users_in_cohort_total": overall_n_users_in_cohort,
        "n_pass_val_total": overall_n_pass_val,
        "n_pass_val_and_test_total": overall_n_pass_val_and_test,
        "frac_pass_val_total": (overall_n_pass_val / overall_n_users_in_cohort
                                 if overall_n_users_in_cohort else 0),
        "frac_pass_val_and_test_total": (overall_n_pass_val_and_test / overall_n_users_in_cohort
                                          if overall_n_users_in_cohort else 0),
        "retention_rate_total": (overall_n_pass_val_and_test / overall_n_pass_val
                                  if overall_n_pass_val else None),
        "val_P_M_gt0_per_user_distribution": {
            "mean": float(np.mean(overall_val_P_per_user)),
            "median": float(np.median(overall_val_P_per_user)),
            "q25": float(np.quantile(overall_val_P_per_user, 0.25)),
            "q75": float(np.quantile(overall_val_P_per_user, 0.75)),
            "frac_ge_80pct": float(np.mean([p >= VAL_P_MIN for p in overall_val_P_per_user])),
        },
        "test_P_M_gt0_per_user_distribution": {
            "mean": float(np.mean(overall_test_P_per_user)),
            "median": float(np.median(overall_test_P_per_user)),
            "q25": float(np.quantile(overall_test_P_per_user, 0.25)),
            "q75": float(np.quantile(overall_test_P_per_user, 0.75)),
            "frac_ge_80pct": float(np.mean([p >= VAL_P_MIN for p in overall_test_P_per_user])),
        },
        "val_M_med_per_user_distribution": {
            "mean": float(np.mean(overall_val_med_M_per_user)),
            "median": float(np.median(overall_val_med_M_per_user)),
            "frac_above_0": float(np.mean([m > 0 for m in overall_val_med_M_per_user])),
        },
        "test_M_med_per_user_distribution": {
            "mean": float(np.mean(overall_test_med_M_per_user)),
            "median": float(np.median(overall_test_med_M_per_user)),
            "frac_above_0": float(np.mean([m > 0 for m in overall_test_med_M_per_user])),
        },
    }

    # Verdict
    n_sel = overall_n_pass_val
    n_ret = overall_n_pass_val_and_test
    if n_sel == 0:
        verdict = ("FAIL — no user passes the validation gate "
                   f"(P_u(M>0) ≥ {VAL_P_MIN:.0%} AND median M > 0)")
    elif n_ret == 0:
        verdict = ("FAIL — validation-passing users do NOT retain on independent test")
    elif n_ret / n_sel >= 0.80:
        verdict = ("GO — syntax-identifiable user pool is large AND retains "
                   "≥ 80% on independent test set")
    else:
        verdict = ("PARTIAL — syntax-identifiable user pool exists but "
                   "test retention < 80%")

    summary["verdict"] = verdict

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    full = {"summary": summary, "per_asin_results": per_asin_results}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(full, f, indent=1)
    log(f"  wrote → {OUT_JSON}")

    log("\n=== OVERALL ===")
    log(f"  ASINs evaluated: {len(per_asin_results)}")
    log(f"  Users in cohort (≥2 valid): {overall_n_users_in_cohort}")
    log(f"  Pass validation gate: {overall_n_pass_val} ({summary['frac_pass_val_total']:.1%})")
    log(f"  Pass validation AND test: {overall_n_pass_val_and_test} ({summary['frac_pass_val_and_test_total']:.1%})")
    if summary["retention_rate_total"] is not None:
        log(f"  Retention rate (test/val): {summary['retention_rate_total']:.1%}")
    log(f"  Val P(M>0) per-user: mean={summary['val_P_M_gt0_per_user_distribution']['mean']:.1%}, "
        f"median={summary['val_P_M_gt0_per_user_distribution']['median']:.1%}")
    log(f"  Test P(M>0) per-user: mean={summary['test_P_M_gt0_per_user_distribution']['mean']:.1%}, "
        f"median={summary['test_P_M_gt0_per_user_distribution']['median']:.1%}")
    log(f"  {verdict}")
    log("=== DONE ===")


if __name__ == "__main__":
    main()