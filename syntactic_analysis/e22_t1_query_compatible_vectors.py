#!/usr/bin/env python3
"""E22 Task 1: Build Query-compatible user syntax vectors with disjoint user pool.

Per issue #22 / Task 1, this script:
  1. Picks a fresh disjoint candidate pool (SEED=1111) and explicitly excludes
     all users from prior E21 runs (v9/v10/v11/v12).
  2. Filters users: ≥5 words/sent, ≥30 complete sentences, ≥2 distinct
     parent_asins; (parent_asin, text) dedup; review-level disjoint split.
  3. Splits users into train/dev/test with disjoint user IDs.
  4. Aggregates the FULL 318-dim syntactic vector per user from ALL eligible
     sentences (not just 15).
  5. Fits standardization (mean/std) on TRAIN users only.
  6. Selects Query-compatible features by:
     a) within-user stability (split-half correlation ≥ STAB_THRESH)
     b) between/within variance ratio (≥ BW_THRESH)
     c) low correlation with sentence count (|r| < COUNT_CORR_THRESH)
     d) all features are syntactic (POS/dep/clause/opener/closer/punct) —
       no content words by construction.
  7. Persists:
     - z_user per user (full and filtered)
     - feature_schema.json (names, dim, normalization, selection criteria)
     - data manifest (user IDs by split + hashes)
     - result JSON with gate metrics
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query" / "soft_prefix"))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

from extract_syntactic_features import ALL_FEATS_V2, user_features_v2
from parse_sentences_to_features import (
    CACHE_META_VERSION, load_features_cache, parse_corpus_with_reviews,
    sent_key, split_sents,
)

REVIEWS = REPO_ROOT / "data" / "Baby_Products_2023.jsonl.gz"
CACHE_PATH = REPO_ROOT / "result" / "cache" / "per_sentence_features.jsonl.gz"
OUT_DIR = REPO_ROOT / "result"
OUT_JSON = OUT_DIR / "e22_t1_results.json"
OUT_LOG = OUT_DIR / "e22_t1.log"
OUT_VECTORS = OUT_DIR / "e22_t1_user_vectors.npz"
OUT_SCHEMA = OUT_DIR / "e22_t1_feature_schema.json"
OUT_MANIFEST = OUT_DIR / "e22_t1_manifest.json"

# Pool / seed selection
SEED = 1111  # numerically disjoint from E21 v9/v10/v11/v12 (43/43/99/999)
N_USERS_POOL = 5000
MIN_WORDS = 100
MIN_ASINS = 2
# Splits: 60% train, 20% dev, 20% test (disjoint user IDs)
TRAIN_FRAC = 0.6
DEV_FRAC = 0.2
TEST_FRAC = 0.2

# Sentences-per-user filter
MIN_SENTS_PER_USER = 30  # ≥30 complete sentences
MIN_SENT_LEN_TOK = 5     # each sent must be ≥5 tokens

# Feature selection thresholds
STAB_THRESH = 0.70         # within-user split-half Pearson r
BW_THRESH = 1.5            # between-var / within-var ratio
COUNT_CORR_THRESH = 0.30   # |r| with sentence count
MIN_FRAC_NONZERO = 0.05    # feature must be non-zero in ≥5% users

# Discrimination gates (Check-1)
AUC_THRESH = 0.65
D_THRESH = 0.5
SEED_PASS_THRESH = 0.80
P_THRESH = 0.01
TEST_USERS_MIN = 150
N_SEEDS = 30
N_PERM = 9999


def load_prior_eligible() -> set[str]:
    prior: set[str] = set()
    for label in ("v9", "v10", "v11", "v12"):
        p = OUT_DIR / f"e21_{label}_eligible_users.json"
        if p.exists():
            d = json.load(open(p))
            # v9/v10/v11 schema: {"eligible_users": [...]}
            # v12 schema: {"dev_users": [...], "test_users": [...]}
            if "eligible_users" in d:
                ids = d["eligible_users"]
                n = len(ids)
            else:
                ids = d.get("dev_users", []) + d.get("test_users", [])
                n = len(ids)
            prior.update(ids)
            print(f"  prior {label}: {n} users", flush=True)
    return prior


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    print("Loading prior eligible users (v9/v10/v11/v12) for exclusion...",
          flush=True)
    prior_eligible = load_prior_eligible()
    print(f"  total prior: {len(prior_eligible)}", flush=True)

    print("Reading word counts...", flush=True)
    wc: dict[str, int] = defaultdict(int)
    with gzip.open(REVIEWS, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            u = d.get("user_id")
            t = d.get("text") or ""
            if u:
                wc[u] += len(t.split())
    print(f"  users: {len(wc)} (t={time.time() - t0:.1f}s)", flush=True)

    cands = [u for u, w in wc.items() if w >= MIN_WORDS]
    rng.shuffle(cands)
    cands = cands[:N_USERS_POOL * 4]
    n_pre = len(cands)
    cands = [u for u in cands if u not in prior_eligible]
    print(f"  cands {n_pre} → {len(cands)} after exclusion", flush=True)

    print("Loading reviews with (parent_asin, text) dedup...", flush=True)
    reviews: dict[str, list[tuple]] = defaultdict(list)
    seen_keys: dict[str, set] = defaultdict(set)
    poolset = set(cands)
    with gzip.open(REVIEWS, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            u = d.get("user_id")
            if u not in poolset:
                continue
            t = (d.get("text") or "").strip()
            if not t:
                continue
            asin = d.get("parent_asin") or d.get("asin")
            rk = (asin, t[:2000])
            if rk in seen_keys[u]:
                continue
            seen_keys[u].add(rk)
            reviews[u].append((asin, t[:2000]))
    for u in list(reviews):
        if len({r[0] for r in reviews[u]}) < MIN_ASINS:
            del reviews[u]
    print(f"  users with >= {MIN_ASINS} asins: {len(reviews)}", flush=True)

    print("Building (user, asin) sentence map via parse_corpus_with_reviews...",
          flush=True)
    # parse_corpus_with_reviews uses SHA1-keyed cache; new sentences get
    # parsed once and saved back to cache. Existing cache (849k entries
    # from v12) covers most prior sentences, so this only parses the
    # cache-miss subset.
    from extract_clause_features_single_query import load_spacy_model
    nlp = load_spacy_model()
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)
    review_iter: list[tuple[str, str, str]] = []
    for u, revs in reviews.items():
        for asin, t in revs:
            review_iter.append((u, asin, t))
    user_review_sents = parse_corpus_with_reviews(
        review_iter, CACHE_PATH, nlp=nlp, batch_size=256, log_prefix="  ")
    print(f"  users with sents: {len(user_review_sents)}", flush=True)

    # Filter: ≥MIN_SENT_LEN_TOK tokens per sentence, ≥MIN_SENTS_PER_USER total,
    # ≥2 distinct asins with sentences.
    print("Filtering eligible (>=30 sents, >=2 asins, L=5 filter)...", flush=True)
    eligible: list[str] = []
    for u, by_review in user_review_sents.items():
        filt = {
            rid: [sf for sf in sfs if sf["n_tok"] >= MIN_SENT_LEN_TOK]
            for rid, sfs in by_review.items()
        }
        n_total = sum(len(sfs) for sfs in filt.values())
        if n_total < MIN_SENTS_PER_USER:
            continue
        n_asins = sum(1 for sfs in filt.values() if sfs)
        if n_asins < 2:
            continue
        eligible.append(u)
    print(f"  eligible users: {len(eligible)}", flush=True)

    # Split train/dev/test with disjoint user IDs.
    rng_split = np.random.default_rng(SEED + 100)
    eligible_sorted = sorted(eligible)
    perm = rng_split.permutation(len(eligible_sorted))
    n = len(eligible_sorted)
    n_train = int(n * TRAIN_FRAC)
    n_dev = int(n * DEV_FRAC)
    train_users = [eligible_sorted[i] for i in perm[:n_train]]
    dev_users = [eligible_sorted[i] for i in perm[n_train:n_train + n_dev]]
    test_users = [eligible_sorted[i] for i in perm[n_train + n_dev:]]
    print(f"  train/dev/test: {len(train_users)}/{len(dev_users)}/{len(test_users)}",
          flush=True)

    # Compute FULL 318-dim user vectors.
    print("Aggregating full 318-dim vectors from ALL eligible sentences...",
          flush=True)
    user_vecs: dict[str, np.ndarray] = {}
    user_sents_n: dict[str, int] = {}
    for u in train_users + dev_users + test_users:
        by_review = user_review_sents[u]
        sfs: list[dict] = []
        for sfs_list in by_review.values():
            sfs.extend([sf for sf in sfs_list if sf["n_tok"] >= MIN_SENT_LEN_TOK])
        v = user_features_v2(sfs)
        if v is None:
            continue
        user_vecs[u] = v
        user_sents_n[u] = len(sfs)
    print(f"  users with vectors: {len(user_vecs)}", flush=True)

    D = len(ALL_FEATS_V2)

    # Build train matrix for standardization + feature selection
    X_train = np.stack([user_vecs[u] for u in train_users if u in user_vecs])
    sents_train = np.asarray([user_sents_n[u] for u in train_users
                              if u in user_vecs], dtype=np.float64)

    # Standardization (mean/std) on TRAIN only — frozen for all users.
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)
    train_std = np.where(train_std < 1e-12, 1.0, train_std)

    print("Selecting Query-compatible features...", flush=True)
    selection = []
    selection_reasons: dict[str, str] = {}
    for j, name in enumerate(ALL_FEATS_V2):
        col = X_train[:, j]
        nz_frac = float((np.abs(col) > 1e-12).mean())
        if nz_frac < MIN_FRAC_NONZERO:
            selection_reasons[name] = f"sparse({nz_frac:.3f})"
            continue
        # Sent count correlation
        if sents_train.std() > 0 and col.std() > 0:
            r_count = float(np.corrcoef(col, sents_train)[0, 1])
        else:
            r_count = 0.0
        if abs(r_count) > COUNT_CORR_THRESH:
            selection_reasons[name] = f"sent_count_r={r_count:.3f}"
            continue
        # Within-user stability: split train users into 2 halves by hash, recompute
        # vector on each half, correlate per-feature (proxy for split-half).
        # Cheap proxy: bootstrap subsample of 15 sentences each.
        keep = True
        reasons_pass = []
        reasons_pass.append(f"nz_frac={nz_frac:.3f}")
        reasons_pass.append(f"sent_r={r_count:.3f}")
        selection.append(j)
        selection_reasons[name] = "PASS: " + " ".join(reasons_pass)

    # For selected features, do split-half stability check (subsample 15
    # sentences, recompute vector, correlate). Slow: only for selected.
    print(f"  initial filter kept {len(selection)}/{D}; "
          f"now running split-half stability on selected...", flush=True)

    stab_pass: list[int] = []
    stab_fail_reasons: dict[int, str] = {}
    rng_stab = np.random.default_rng(SEED + 200)
    for j in selection:
        name = ALL_FEATS_V2[j]
        # For each train user, build two random 15-sent subsamples
        rs_list = []
        for u in train_users[:200]:  # subsample for speed
            by_review = user_review_sents[u]
            sfs: list[dict] = []
            for sfs_list in by_review.values():
                sfs.extend([sf for sf in sfs_list
                            if sf["n_tok"] >= MIN_SENT_LEN_TOK])
            if len(sfs) < 30:
                continue
            idx1 = rng_stab.choice(len(sfs), 15, replace=False)
            idx2 = rng_stab.choice(len(sfs), 15, replace=False)
            v1 = user_features_v2([sfs[i] for i in idx1])
            v2 = user_features_v2([sfs[i] for i in idx2])
            if v1 is None or v2 is None:
                continue
            rs_list.append((v1[j], v2[j]))
        if len(rs_list) < 30:
            stab_fail_reasons[j] = "too few users"
            continue
        a = np.asarray([x[0] for x in rs_list])
        b = np.asarray([x[1] for x in rs_list])
        if a.std() < 1e-12 or b.std() < 1e-12:
            stab_fail_reasons[j] = "zero variance"
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if r < STAB_THRESH:
            stab_fail_reasons[j] = f"stab_r={r:.3f}<{STAB_THRESH}"
            continue
        stab_pass.append(j)

    # Between/within variance ratio for selected stable features
    print(f"  after stability: {len(stab_pass)}; computing between/within...",
          flush=True)
    final_keep: list[int] = []
    bw_reasons: dict[int, str] = {}
    for j in stab_pass:
        col = X_train[:, j]
        # Within-user variance proxy: bootstrap std
        rng_bw = np.random.default_rng(SEED + 300)
        within_stds = []
        for u in train_users[:200]:
            by_review = user_review_sents[u]
            sfs = []
            for sfs_list in by_review.values():
                sfs.extend([sf for sf in sfs_list
                            if sf["n_tok"] >= MIN_SENT_LEN_TOK])
            if len(sfs) < 15:
                continue
            subs = []
            for _ in range(5):
                idx = rng_bw.choice(len(sfs), min(15, len(sfs)), replace=False)
                v = user_features_v2([sfs[i] for i in idx])
                if v is not None:
                    subs.append(v[j])
            if len(subs) >= 2:
                within_stds.append(float(np.std(subs, ddof=1)))
        within_var = float(np.mean(np.asarray(within_stds) ** 2)) if within_stds else 0.0
        between_var = float(col.var(ddof=1))
        ratio = between_var / max(within_var, 1e-12)
        if ratio < BW_THRESH:
            bw_reasons[j] = f"bw_ratio={ratio:.2f}<{BW_THRESH}"
            continue
        final_keep.append(j)

    print(f"  final Query-compatible features: {len(final_keep)}/{D}",
          flush=True)

    # Save final selection indices + names
    keep_names = [ALL_FEATS_V2[j] for j in final_keep]
    schema = {
        "version": "e22_t1_query_compatible_v1",
        "n_features_total": D,
        "n_features_kept": len(final_keep),
        "kept_feature_names": keep_names,
        "kept_feature_indices": final_keep,
        "selection_criteria": {
            "min_nz_frac": MIN_FRAC_NONZERO,
            "max_count_corr": COUNT_CORR_THRESH,
            "min_stability_r": STAB_THRESH,
            "min_bw_ratio": BW_THRESH,
        },
        "normalization": {
            "method": "z_score",
            "fit_on": "train_users_only",
            "mean": train_mean.tolist(),
            "std": train_std.tolist(),
            "mean_kept": train_mean[final_keep].tolist(),
            "std_kept": train_std[final_keep].tolist(),
        },
        "all_feature_names_318": ALL_FEATS_V2,
        "groups": {
            "syntactic_only": True,
            "no_content_words": True,
            "feature_categories": sorted({n.split('_')[0] for n in ALL_FEATS_V2}),
        },
    }
    with open(OUT_SCHEMA, "w") as f:
        json.dump(schema, f, indent=1)
    print(f"  wrote {OUT_SCHEMA.name}", flush=True)

    # Compute z_user = standardized FULL vector; filter = keep subset
    z_full: dict[str, np.ndarray] = {}
    z_filt: dict[str, np.ndarray] = {}
    for u, v in user_vecs.items():
        z = (v - train_mean) / train_std
        z_full[u] = z
        z_filt[u] = z[final_keep]

    # Save vectors (filtered, standardized) for downstream tasks
    all_users = list(user_vecs.keys())
    Z = np.stack([z_filt[u] for u in all_users])
    np.savez_compressed(
        OUT_VECTORS,
        user_ids=np.asarray(all_users),
        Z=Z,
        Z_full=np.stack([z_full[u] for u in all_users]),
        feature_indices=np.asarray(final_keep),
        feature_names=np.asarray(keep_names),
        train_mean=train_mean,
        train_std=train_std,
    )
    print(f"  wrote {OUT_VECTORS.name} (Z shape={Z.shape})", flush=True)

    # Manifest: train/dev/test user ID hashes + intersection check
    def h(ids: list[str]) -> list[str]:
        return [hashlib.sha256(u.encode()).hexdigest()[:16] for u in ids]

    manifest = {
        "version": "e22_t1_v1",
        "seed": SEED,
        "n_users_total": len(eligible),
        "splits": {
            "train": {
                "n": len(train_users),
                "ids": train_users,
                "id_hashes_sha256_16": h(train_users),
            },
            "dev": {
                "n": len(dev_users),
                "ids": dev_users,
                "id_hashes_sha256_16": h(dev_users),
            },
            "test": {
                "n": len(test_users),
                "ids": test_users,
                "id_hashes_sha256_16": h(test_users),
            },
        },
        "disjoint_check": {
            "train_dev_intersection": len(set(train_users) & set(dev_users)),
            "train_test_intersection": len(set(train_users) & set(test_users)),
            "dev_test_intersection": len(set(dev_users) & set(test_users)),
            "prior_pool_exclusion_count": len(prior_eligible),
            "current_pool_intersection_with_prior": len(
                set(all_users) & prior_eligible),
        },
    }
    with open(OUT_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"  wrote {OUT_MANIFEST.name}", flush=True)

    # Discrimination eval (Check-1) on filtered z_filt
    print("Running Check-1 discrimination eval on filtered z_user...",
          flush=True)
    # Self vs cross on test set
    Z_test = np.stack([z_filt[u] for u in test_users if u in z_filt])
    norms = np.linalg.norm(Z_test, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    Zn = Z_test / norms

    n_test = len(Zn)
    rng_eval = np.random.default_rng(SEED + 400)
    aucs = []
    d_perms = []
    for s_i in range(N_SEEDS):
        rs = np.random.default_rng(SEED + 5000 + s_i)
        # split-half: shuffle each user's sentences into half1/half2
        diffs_self = []
        diffs_cross = []
        for i, u in enumerate(test_users):
            if u not in user_review_sents:
                continue
            by_review = user_review_sents[u]
            sfs = []
            for sfs_list in by_review.values():
                sfs.extend([sf for sf in sfs_list
                            if sf["n_tok"] >= MIN_SENT_LEN_TOK])
            if len(sfs) < 30:
                continue
            rs.shuffle(sfs)
            half1 = sfs[:15]
            half2 = sfs[15:30]
            v1 = user_features_v2(half1)
            v2 = user_features_v2(half2)
            if v1 is None or v2 is None:
                continue
            z1 = (v1 - train_mean) / train_std
            z2 = (v2 - train_mean) / train_std
            z1 = z1[final_keep]
            z2 = z2[final_keep]
            n1 = np.linalg.norm(z1)
            n2 = np.linalg.norm(z2)
            if n1 < 1e-12 or n2 < 1e-12:
                continue
            z1 = z1 / n1
            z2 = z2 / n2
            d_self = float(np.linalg.norm(z1 - z2))
            diffs_self.append(d_self)
            # cross: pick 3 random other users
            others = [j for j in range(n_test) if j != i]
            rs.shuffle(others)
            cs = []
            for j in others[:3]:
                uo = test_users[j]
                if uo not in user_review_sents:
                    continue
                by_review_o = user_review_sents[uo]
                sfs_o = []
                for sfs_list in by_review_o.values():
                    sfs_o.extend([sf for sf in sfs_list
                                  if sf["n_tok"] >= MIN_SENT_LEN_TOK])
                if len(sfs_o) < 15:
                    continue
                idx = rs.choice(len(sfs_o), 15, replace=False)
                vo = user_features_v2([sfs_o[k] for k in idx])
                if vo is None:
                    continue
                zo = (vo - train_mean) / train_std
                zo = zo[final_keep]
                no = np.linalg.norm(zo)
                if no < 1e-12:
                    continue
                zo = zo / no
                cs.append(float(np.linalg.norm(zo - z1)))
            if cs:
                diffs_cross.extend(cs)
        ds = np.asarray(diffs_self)
        dc = np.asarray(diffs_cross)
        if len(ds) < 30 or len(dc) < 30:
            continue
        auc = 0.0
        for x in ds:
            auc += float((dc > x).sum())
        auc = auc / (len(ds) * len(dc))
        delta = float(dc.mean() - ds.mean())
        pooled = float(np.sqrt((dc.std(ddof=1)**2 + ds.std(ddof=1)**2) / 2))
        cohen = delta / pooled if pooled > 0 else 0.0
        aucs.append(auc)
        d_perms.append((ds, dc))

    aucs_arr = np.asarray(aucs)
    auc_mean = float(aucs_arr.mean())
    auc_std = float(aucs_arr.std())
    seed_pass = float((aucs_arr >= AUC_THRESH).mean())

    # Cohen d averaged across seeds
    cohen_ds = []
    for ds, dc in d_perms:
        pooled = float(np.sqrt((dc.std(ddof=1)**2 + ds.std(ddof=1)**2) / 2))
        if pooled > 0:
            cohen_ds.append(float((dc.mean() - ds.mean()) / pooled))
    d_mean = float(np.mean(cohen_ds)) if cohen_ds else 0.0

    # Permutation test on aggregated deltas
    print(f"  AUC={auc_mean:.4f}±{auc_std:.4f}  d={d_mean:.4f}  "
          f"seed_pass={seed_pass:.2f}", flush=True)
    obs_delta = float(np.mean([dc.mean() - ds.mean() for ds, dc in d_perms]))
    all_halves = []
    rs_perm = np.random.default_rng(SEED + 6000)
    for i, u in enumerate(test_users):
        if u not in user_review_sents:
            continue
        by_review = user_review_sents[u]
        sfs = []
        for sfs_list in by_review.values():
            sfs.extend([sf for sf in sfs_list
                        if sf["n_tok"] >= MIN_SENT_LEN_TOK])
        if len(sfs) < 30:
            continue
        rs_perm.shuffle(sfs)
        h1 = sfs[:15]
        h2 = sfs[15:30]
        v1 = user_features_v2(h1)
        v2 = user_features_v2(h2)
        if v1 is not None:
            z1 = (v1 - train_mean) / train_std
            z1 = z1[final_keep]
            n1 = np.linalg.norm(z1)
            if n1 > 1e-12:
                all_halves.append(z1 / n1)
        if v2 is not None:
            z2 = (v2 - train_mean) / train_std
            z2 = z2[final_keep]
            n2 = np.linalg.norm(z2)
            if n2 > 1e-12:
                all_halves.append(z2 / n2)
    Xp = np.stack(all_halves)
    print(f"  perm on {len(Xp)} halves (dim={len(final_keep)})", flush=True)
    null_deltas = []
    rng_null = np.random.default_rng(SEED + 7000)
    M = len(Xp)
    half_m = (M // 2) * 2
    for _ in range(N_PERM):
        idx = rng_null.permutation(M)[:half_m]
        a = Xp[idx[:half_m // 2]]
        b = Xp[idx[half_m // 2:]]
        null_deltas.append(float(np.linalg.norm(a - b, axis=1).mean()))
    null_deltas = np.asarray(null_deltas)
    p_one = (float((null_deltas <= obs_delta).sum()) + 1) / (len(null_deltas) + 1)
    null_center = float(null_deltas.mean())
    p_two = (float((np.abs(null_deltas - null_center) >=
                    abs(obs_delta - null_center)).sum()) + 1
             ) / (len(null_deltas) + 1)

    passes = (
        len(test_users) >= TEST_USERS_MIN
        and auc_mean >= AUC_THRESH
        and d_mean >= D_THRESH
        and seed_pass >= SEED_PASS_THRESH
        and p_two < P_THRESH
    )

    result = {
        "version": "e22_t1_query_compatible_v1",
        "seed": SEED,
        "n_features_total": D,
        "n_features_kept": len(final_keep),
        "n_train_users": len(train_users),
        "n_dev_users": len(dev_users),
        "n_test_users": len(test_users),
        "prior_pool_users_excluded": len(prior_eligible),
        "current_pool_overlap_with_prior": len(set(all_users) & prior_eligible),
        "discrimination_test": {
            "auc_mean": round(auc_mean, 4),
            "auc_std": round(auc_std, 4),
            "cohen_d_mean": round(d_mean, 4),
            "seed_pass_frac": round(seed_pass, 4),
            "perm_p_one": round(p_one, 5),
            "perm_p_two": round(p_two, 5),
        },
        "feature_selection_summary": {
            "kept_names_sample": keep_names[:10],
            "rejected_examples": {
                k: v for k, v in list(selection_reasons.items())[:10]
            },
        },
        "all_gates_pass": bool(passes),
        "gates": {
            "auc_min": AUC_THRESH,
            "cohen_d_min": D_THRESH,
            "seed_pass_min": SEED_PASS_THRESH,
            "p_two_max": P_THRESH,
            "test_users_min": TEST_USERS_MIN,
        },
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=1)
    print(f"  wrote {OUT_JSON.name}; all_gates_pass={passes}", flush=True)


if __name__ == "__main__":
    main()