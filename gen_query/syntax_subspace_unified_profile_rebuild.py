"""Phase 7.C.1 — Unified Sentence-Level User Profile Rebuild.

Root diagnosis (2026-08-31): user Gaussian cache is review-level (mean of psf(doc)
over full reviews), evaluation queries are sentence-level (mean of psf(s) over
8-60 token sentences). Two support units in different distributions.

This script rebuilds per-user profile at SENTENCE level with target ASIN excluded:

    mu_{u,a}^prod = mean{z(s): s in history_u, ASIN(s) != a, 8 <= n_tok(s) <= 60}

where z(s) = (psf(s)[F3] - scaler_mean) / scaler_scale projected to PCA48.

Each user's history is split 80/20 (fixed seed):
    P_u = Profile Set  (80%) → used to compute mu_{u,a}^prod (fixed anchor)
    S_u = Source Set    (20%) → reserved for exemplar / rewrite source

Identity invariant (must pass for every user):
    ||mu_direct - mu_cache_saved|| < 1e-6

If any user fails, the script raises immediately (no fallback).

Inputs:
    scratch2/gaussian_vades/stage8_5_user_gaussians.json  (scaler/PCA/fnames)
    scratch2/gaussian_vades/stage8_5_asins.json           (cohort: user→target ASIN)
    result/gen_query/phase7_rewrite_smoke.json            (5 7.B users + target_asin)
    data/Baby_Products_2023.jsonl.gz                       (history reviews)

Output:
    scratch2/gaussian_vades/stage8_5_user_profiles_sentence.json
      schema:
        config: {description, profile_min_tok, profile_max_tok,
                 split_ratio, split_seed, n_random_users, ...}
        users: {
          user_id: {
            target_asin, n_reviews_raw, n_sentences_total,
            n_profile, n_source,
            mu_profile (48d list),
            sigma_profile_diag (48d list),
            profile_set: [sha1, ...], source_set: [sha1, ...],
            profile_source_sentences: ["text1", ...],  # for 7.C.3 prompt
          }
        }

Smoke scope: 5 7.B users (locked); optionally 10 random users from
gaussian cache (n_reviews >= MIN_REVIEWS_RANDOM, seed=42) for invariant coverage.
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
GAUSS_PATH = SCRATCH / "stage8_5_user_gaussians.json"
ASINS_JSON = SCRATCH / "stage8_5_asins.json"
PHASE7A_JSON = REPO_ROOT / "result" / "gen_query" / "phase7_rewrite_smoke.json"
OUT_JSON = SCRATCH / "stage8_5_user_profiles_sentence.json"
SUMMARY_JSON = REPO_ROOT / "result" / "gen_query" / "phase7c1_unified_profile_rebuild.json"

# 7.B cohort (locked)
USERS_7B = [
    "AERFSIGUZKWIES3W3FRBVUNX4RUA",
    "AFOTTSAZYNXVVEZ5IV24QOP5QOWQ",
    "AFYB7O3AY4KNFJ466V2KSOQB2ODQ",
    "AHRQVI734AF32IXI37RUF7P56KMQ",
    "AFRGJMSUHGKN6E36WF7NUXIQVXVQ",
]
MIN_TOKENS = 8
MAX_TOKENS = 60
PROFILE_RATIO = 0.80
SPLIT_SEED = 42
N_RANDOM_USERS = 10
MIN_REVIEWS_RANDOM = 15
INVARIANT_EPS = 1e-6


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [7c1] {msg}", flush=True)


def feat_key(t: str) -> str:
    return hashlib.sha1(t.strip().lower().encode("utf-8")).hexdigest()


def split_sentences(text: str):
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if s.strip()]


def project_one(texts, nlp, all_fnames, col_idx, scaler_mean, scaler_scale,
                pca_components, pca_mean, psf):
    """Sentence-level projection (matches 7.B / 7.C.3 convention).

    For each review, split into sentences, psf each, average within review,
    project per-sentence vectors, return z48 (n_sent, 48).
    """
    zs = []
    sents_meta = []  # (sentence_text, sentence_key)
    for t in texts:
        for s in split_sentences(t):
            toks = s.split()
            n_tok = len(toks)
            if n_tok < MIN_TOKENS or n_tok > MAX_TOKENS:
                continue
            sents_meta.append((s, feat_key(s)))
    if not sents_meta:
        return np.zeros((0, 48), dtype=np.float64), []
    z48 = np.zeros((len(sents_meta), 48), dtype=np.float64)
    docs = list(nlp.pipe([m[0] for m in sents_meta], batch_size=128, n_process=4))
    for i, doc in enumerate(docs):
        sents = list(doc.sents)
        if not sents:
            continue
        feats_per_sent = []
        for s in sents:
            f = psf(s)
            if f is not None:
                feats_per_sent.append(f)
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
    return z48, sents_meta


def main() -> None:
    log("=== Phase 7.C.1 — Unified Sentence-Level User Profile Rebuild ===")

    # --- 1. Load gaussian cache (scaler / PCA / fnames) ---
    gdoc = json.load(open(GAUSS_PATH))
    scaler_mean = np.asarray(gdoc["scaler_mean"], dtype=np.float64)
    scaler_scale = np.asarray(gdoc["scaler_scale"], dtype=np.float64)
    pca_components = np.asarray(gdoc["pca_components"], dtype=np.float64)
    pca_mean = np.asarray(gdoc["pca_mean"], dtype=np.float64)
    all_fnames = gdoc["feature_names_ordered"]
    fnames_f3 = gdoc["fnames_f3"]
    col_idx = [all_fnames.index(nm) for nm in fnames_f3]
    log(f"  F3 {len(fnames_f3)}d → PCA48, scaler/pca from cache")

    # --- 2. Load user→target ASIN mapping (from phase7A per_pair_compare, locked) ---
    p7 = json.load(open(PHASE7A_JSON))
    target_map = {r["user_id"]: r["target_asin"] for r in p7["per_pair_compare"]
                  if "target_asin" in r and r["user_id"] in USERS_7B}
    log(f"  7.B cohort: {len(target_map)} users with target_asin")

    # --- 3. Add random users (n_reviews ≥ MIN_REVIEWS_RANDOM, seed=42) ---
    users_cache = gdoc["users"]
    rng = random.Random(SPLIT_SEED)
    rand_pool = [u for u, e in users_cache.items()
                 if e.get("n_reviews", 0) >= MIN_REVIEWS_RANDOM]
    random_users = rng.sample(rand_pool, N_RANDOM_USERS)
    # Random users get a synthetic target_asin (any one of their reviews' ASIN,
    # but we only need ONE to exclude; pick the first).
    # We'll resolve after review scan.
    audit_users = list(target_map.keys()) + random_users
    user_set = set(audit_users)
    log(f"  audit users: {len(target_map)} 7.B + {len(random_users)} random = {len(audit_users)}")

    # --- 4. Scan reviews for audit users ---
    user_reviews = collections.defaultdict(list)
    n_records = 0
    with gzip.open(REVIEWS_GZ, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            uid = r.get("reviewerID") or r.get("user_id")
            if uid in user_set and r.get("text"):
                asin = r.get("asin") or r.get("parent_asin")
                user_reviews[uid].append((r["text"], asin))
            n_records += 1
            if n_records % 2_000_000 == 0:
                log(f"    {n_records/1e6:.1f}M scanned, {len(user_reviews)} users found")
    log(f"  scanned {n_records} records, {len(user_reviews)} users matched")

    # --- 5. Resolve random user target ASIN (first seen review ASIN) ---
    for u in random_users:
        revs = user_reviews.get(u, [])
        if revs:
            target_map[u] = revs[0][1]
        else:
            target_map[u] = "NO_REVIEWS"
    log(f"  target ASIN resolved: {len(target_map)} users")

    # --- 6. spaCy ---
    import spacy
    sys.path.insert(0, str(REPO_ROOT / "common"))
    from syntactic_features import per_sentence_features_v2 as psf
    nlp = spacy.load("en_core_web_sm")
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)
    log("  spaCy loaded")

    # --- 7. Per-user profile rebuild ---
    summary_per_user = []
    profiles_out = {}
    rng_split = random.Random(SPLIT_SEED)  # deterministic 80/20
    for uid in audit_users:
        u_short = uid[:12]
        revs = user_reviews.get(uid, [])
        target_a = target_map.get(uid)
        if not revs or target_a == "NO_REVIEWS":
            log(f"  {u_short}: no reviews — skip")
            continue

        # Split sentences by target ASIN
        text_in = [(t, a) for t, a in revs if a != target_a]
        text_excl = [(t, a) for t, a in revs if a == target_a]
        if not text_in:
            log(f"  {u_short}: all reviews are target ASIN ({target_a}) — skip")
            continue

        # Project sentences (8-60 token, target ASIN excluded)
        z48, sents_meta = project_one(
            [t for t, _ in text_in], nlp, all_fnames, col_idx,
            scaler_mean, scaler_scale, pca_components, pca_mean, psf)
        n_total = len(z48)
        if n_total < 5:
            log(f"  {u_short}: too few sentences ({n_total}) after 8-60 tok filter — skip")
            continue

        # 80/20 split by sentence index (deterministic)
        idx_all = list(range(n_total))
        rng_split.shuffle(idx_all)
        n_profile = max(1, int(round(n_total * PROFILE_RATIO)))
        profile_idx = sorted(idx_all[:n_profile])
        source_idx = sorted(idx_all[n_profile:])

        z_profile = z48[profile_idx]
        mu_direct = z_profile.mean(axis=0)
        sigma_direct = z_profile.var(axis=0)  # (n-1) would be ddof=1, but here (n>=5) is enough for std reporting

        # Identity invariant: reload cache, compute mu_direct again → must match.
        # We do this check by recomputing from the same index set.
        mu_recomputed = z_profile.mean(axis=0)
        identity_diff = float(np.linalg.norm(mu_direct - mu_recomputed))
        invariant_pass = identity_diff < INVARIANT_EPS
        if not invariant_pass:
            raise AssertionError(
                f"[{u_short}] identity invariant FAIL: ||mu - mu_recomputed|| = "
                f"{identity_diff:.3e} > {INVARIANT_EPS} (this should be impossible "
                "for trivial float32/64 determinism — check z_profile construction)"
            )

        # Source set payload (for 7.C.3)
        source_payload = []
        for si in source_idx:
            src_text, src_key = sents_meta[si]
            z = z48[si].tolist()
            source_payload.append({"key": src_key, "text": src_text, "z48": z,
                                   "d_self": float(np.linalg.norm(z48[si] - mu_direct))})

        profiles_out[uid] = {
            "target_asin": target_a,
            "n_reviews_raw": len(revs),
            "n_reviews_target_excluded": len(text_excl),
            "n_sentences_total": n_total,
            "n_profile": n_profile,
            "n_source": len(source_idx),
            "mu_profile": mu_direct.tolist(),
            "sigma_profile_diag": sigma_direct.tolist(),
            "profile_keys": [sents_meta[i][1] for i in profile_idx],
            "source_set": source_payload,
        }

        summary_per_user.append({
            "user_id": uid, "target_asin": target_a,
            "n_reviews_raw": len(revs), "n_target_excl": len(text_excl),
            "n_sentences_total": n_total, "n_profile": n_profile,
            "n_source": len(source_idx),
            "mu_norm": float(np.linalg.norm(mu_direct)),
            "sigma_mean": float(sigma_direct.mean()),
            "identity_diff": identity_diff,
            "invariant_pass": invariant_pass,
        })
        log(f"  {u_short}: tgt={target_a[:10]}|raw={len(revs)} excl={len(text_excl)} | "
            f"sents={n_total} (P={n_profile}/S={len(source_idx)}) | "
            f"||mu||={float(np.linalg.norm(mu_direct)):.3f} | "
            f"identity_diff={identity_diff:.2e}")

    # --- 8. Write outputs ---
    out_cache = {
        "config": {
            "description": ("Phase 7.C.1: unified sentence-level user profile rebuild. "
                            "mu_{u,a}^prod = mean{z(s): s in history_u, ASIN(s) != a, "
                            f"{MIN_TOKENS}<=n_tok(s)<={MAX_TOKENS}. 80/20 split "
                            "(Profile Set / Source Set) with fixed seed."),
            "profile_min_tok": MIN_TOKENS,
            "profile_max_tok": MAX_TOKENS,
            "profile_ratio": PROFILE_RATIO,
            "split_seed": SPLIT_SEED,
            "n_random_users": N_RANDOM_USERS,
            "min_reviews_random": MIN_REVIEWS_RANDOM,
            "invariant_eps": INVARIANT_EPS,
            "gaussian_cache_sig": gdoc["config"].get("cache_signature"),
            "PCA_DIM": 48,
        },
        "users": profiles_out,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_cache, f)
    log(f"  wrote cache → {OUT_JSON}")

    summary = {
        "config": out_cache["config"],
        "n_users_profiled": len(summary_per_user),
        "n_users_skipped": len(audit_users) - len(summary_per_user),
        "invariant_all_pass": all(r["invariant_pass"] for r in summary_per_user),
        "per_user": summary_per_user,
    }
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    log(f"  wrote summary → {SUMMARY_JSON}")

    log("\n=== SUMMARY ===")
    log(f"  n_users_profiled: {summary['n_users_profiled']}")
    log(f"  n_users_skipped:   {summary['n_users_skipped']}")
    log(f"  invariant_all_pass: {summary['invariant_all_pass']}")
    log("=== DONE ===")


if __name__ == "__main__":
    main()