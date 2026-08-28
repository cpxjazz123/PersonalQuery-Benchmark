"""Syntax Subspace - Stage 3 (user Gaussians).

gaussian/ only keeps the "compute user prior distribution" logic: from the
Baby_Products review corpus, build a per-user Mahalanobis Gaussian for each
target user (PCA48 + shrinkage, LAMBDA=0.1, VAR_EPS=1e-3, MIN_REVIEWS_FOR_PER_USER=1).

No fallback chain: if a user has no Gaussian, log a warning and skip them.

Usage:
  python gaussian/syntax_subspace_user_gaussians.py

I/O:
  Input:  stage8_5_asins.json (target users + ASIN)
          data/Baby_Products_2023.jsonl.gz (review corpus)
          stage7b_query_features.jsonl.gz (spaCy 182d feature cache)
  Output: result/gaussian/user_gaussians.json
          (per-user mu / sigma_diag)

Shared utilities (log, feat_key, paths, hyperparams, _syntax_subspace_prepare)
are imported from: common/syntax_subspace_utils.py
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
import zlib
from pathlib import Path

import numpy as np

# Ensure common/ is on sys.path so we can import the shared utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from syntax_subspace_utils import (  # noqa: E402
    ASINS_IN, FEAT_CACHE, GAUSSIANS_OUT, REPO_ROOT, REVIEW_GZ,
    LAMBDA, MIN_REVIEWS_FOR_PER_USER, PCA_DIM, VAR_EPS, log, feat_key,
)

# Stage 3 also imports per_sentence_features_v2 from common/syntactic_features
sys.path.insert(0, str(REPO_ROOT / "common"))


def stage_user_gaussians():
    log("=== STAGE 3 - USER GAUSSIANS ===")

    # User directive 2026-08-28: cache Stage 3 - Stage 3 does NOT depend on pool.json,
    # only on review corpus + PCA + feature cache + target user list. If neither
    # upstream files nor config change, just load user_gaussians.json directly
    # (saves ~1.5min x multiple iterations = ~10min+).
    import hashlib as _hl_cache
    sig_payload = json.dumps({
        "PCA_DIM": PCA_DIM,
        "LAMBDA": LAMBDA,
        "VAR_EPS": VAR_EPS,
        "MIN_REVIEWS_FOR_PER_USER": MIN_REVIEWS_FOR_PER_USER,
        "ASINS_IN": str(ASINS_IN),
        "ASINS_IN_mtime": ASINS_IN.stat().st_mtime if ASINS_IN.exists() else 0,
        "FEAT_CACHE": str(FEAT_CACHE),
        "FEAT_CACHE_mtime": FEAT_CACHE.stat().st_mtime if FEAT_CACHE.exists() else 0,
        "REVIEW_GZ": str(REVIEW_GZ),
        "REVIEW_GZ_mtime": REVIEW_GZ.stat().st_mtime if REVIEW_GZ.exists() else 0,
    }, sort_keys=True)
    sig_hash = _hl_cache.sha1(sig_payload.encode("utf-8")).hexdigest()[:12]
    log(f"  cache signature: {sig_hash}")

    if GAUSSIANS_OUT.exists():
        try:
            cached = json.load(open(GAUSSIANS_OUT))
            cached_sig = cached.get("config", {}).get("cache_signature")
            if cached_sig == sig_hash:
                cached_users = cached.get("users", {})
                log(f"  CACHE HIT: {len(cached_users)} users from {GAUSSIANS_OUT}")
                log(f"  (upstream files + config unchanged, skip full pipeline)")
                return
            else:
                log(f"  cache signature mismatch (cached={cached_sig}, current={sig_hash}), recomputing...")
        except Exception as e:
            log(f"  cache load failed ({e!r}), recomputing...")
    else:
        log(f"  no cache file at {GAUSSIANS_OUT}, computing from scratch...")

    log("\n=== 1. Loading PCA48 ===")
    from syntax_subspace_utils import _syntax_subspace_prepare
    P = _syntax_subspace_prepare()
    scaler = P["scaler"]
    fnames = P["feature_names_ordered"]
    train_idx = P["train_idx"]
    log(f"  scaler mean shape: {scaler.mean_.shape}, fnames: {len(fnames)}")

    from sklearn.decomposition import PCA
    pca = PCA(n_components=PCA_DIM, random_state=42)
    pca.fit(P["X_scaled"][train_idx])
    log(f"  PCA48 EV={pca.explained_variance_ratio_.sum():.4f}")

    log("\n=== 2. Loading target users ===")
    asin_data = json.load(open(ASINS_IN))["asins"]
    target_users = set()
    user_to_asins = collections.defaultdict(set)
    for a in asin_data:
        for uid in a["users_sampled"]:
            target_users.add(uid)
            user_to_asins[uid].add(a["asin"])
    log(f"  target users: {len(target_users)}")

    log("\n=== 3. Scanning review corpus ===")
    user_review_texts = collections.defaultdict(list)
    n_records = 0
    # User directive 2026-08-27: field extraction matches attribute_extraction/build_dataset.py
    for line in gzip.open(REVIEW_GZ, "rt", encoding="utf-8"):
        r = json.loads(line)
        uid = r.get("reviewerID") or r.get("user_id")
        if uid in target_users and r.get("text"):
            user_review_texts[uid].append(r["text"])
        n_records += 1
        if n_records % 2_000_000 == 0:
            log(f"    {n_records/1e6:.1f}M records, {len(user_review_texts)} users found")
    log(f"  total records: {n_records}")
    log(f"  users with reviews: {len(user_review_texts)}")

    review_counts = [len(v) for v in user_review_texts.values()]
    if review_counts:
        log(f"  review count: min={min(review_counts)}, "
            f"mean={sum(review_counts)/len(review_counts):.1f}, "
            f"max={max(review_counts)}")
    n_high = sum(1 for c in review_counts if c >= MIN_REVIEWS_FOR_PER_USER)
    log(f"  users with >={MIN_REVIEWS_FOR_PER_USER} reviews (per_user Gaussian): {n_high}")
    n_low = sum(1 for c in review_counts if c < MIN_REVIEWS_FOR_PER_USER and c >= 1)
    log(f"  users with 1-2 reviews: {n_low}")
    n_zero = len(target_users) - len(user_review_texts)
    log(f"  users with 0 reviews: {n_zero}")

    log("\n=== 4. Loading feature cache (lazy: only seen_keys) ===")
    seen_keys = set()
    if FEAT_CACHE.exists():
        # User directive 2026-08-29: tolerate EOFError + zlib.error (chunked append not atomic).
        # Gracefully degrade to first decompression error, all earlier entries preserved.
        n_load_errors = 0
        try:
            with gzip.open(FEAT_CACHE, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        n_load_errors += 1
                        continue
                    seen_keys.add(rec["k"])
        except (EOFError, gzip.BadGzipFile, zlib.error) as e:
            log(f"  WARN cache gzip stream truncated ({type(e).__name__}: {e!r}), "
                f"loaded {len(seen_keys)} keys + {n_load_errors} corrupt lines")
    log(f"  cache keys loaded: {len(seen_keys)} (lazy mode)")

    log("\n=== 5. Extracting spaCy features for user review texts ===")
    # User directive 2026-08-27: no sentence split threshold; use entire review as 1 sample.
    # Reason: short_users (mean_wc<3) reviews are mostly 1-2 word; sentence split gives
    # nz=0-7 all-zero pollution; entire review contains token-based features.
    # After extraction, filter by non-zero count >= K; 1-2 word reviews naturally filtered.
    # User directive 2026-08-28: chunked append + reduced worker mem footprint to avoid 1.84M sentence OOM
    # User directive 2026-08-29: n_process=2, batch_size=128 (OOM safe; reduced from 4)
    all_sents = []
    sent_to_user = []
    for uid, texts in user_review_texts.items():
        for t in texts:
            t = t.replace("\n", " ").strip()
            if t:
                all_sents.append(t)
                sent_to_user.append(uid)
    log(f"  total reviews (no length threshold, no sentence split): {len(all_sents)}")

    new_sents = [s for s in all_sents if feat_key(s) not in seen_keys]
    log(f"  unique sentences: {len(set(all_sents))}, new to extract: {len(new_sents)}")

    feat_map = {}
    if new_sents:
        import spacy
        from syntactic_features import per_sentence_features_v2
        nlp = spacy.load("en_core_web_sm")
        # Disable spaCy components not used by features (30-40% speedup)
        for comp in ("ner", "lemmatizer", "attribute_ruler"):
            if comp in nlp.pipe_names:
                nlp.disable_pipe(comp)
        N_PROCESS = 2
        BATCH_SIZE = 128
        CHUNK_FLUSH = 25_000  # Halve from 50K for lower per-chunk peak (50K Doc x 4 workers ≈ 8GB peak)
        log(f"  extracting features for {len(new_sents)} new sentences "
            f"(n_process={N_PROCESS}, batch={BATCH_SIZE}, "
            f"chunked append every {CHUNK_FLUSH})...")
        new_unique = sorted(set(new_sents))
        if not FEAT_CACHE.exists():
            with gzip.open(FEAT_CACHE, "wt", encoding="utf-8") as f:
                f.write("# user-review sentence features (key=sha1(text), v=182d dict)\n")

        import gc as _gc
        n_processed = 0
        with gzip.open(FEAT_CACHE, "at", encoding="utf-8") as f_cache:
            for chunk_start in range(0, len(new_unique), CHUNK_FLUSH):
                chunk = new_unique[chunk_start:chunk_start + CHUNK_FLUSH]
                for i, doc in enumerate(
                    nlp.pipe(chunk, batch_size=BATCH_SIZE, n_process=N_PROCESS)
                ):
                    s = chunk[i]
                    k = feat_key(s)
                    try:
                        feats = per_sentence_features_v2(doc)
                        feats = feats if feats is not None else {}
                    except Exception:
                        feats = {}
                    numeric = {n: float(v) for n, v in feats.items()
                               if isinstance(v, (int, float))}
                    filtered = {n: numeric.get(n, 0.0) for n in fnames}
                    feat_map[k] = filtered
                    seen_keys.add(k)
                    f_cache.write(json.dumps({"k": k, "v": filtered}) + "\n")
                    n_processed += 1
                f_cache.flush()
                # Free spaCy Doc references + gc to avoid monotonic RSS growth
                del chunk
                _gc.collect()
                log(f"    flushed chunk {chunk_start + CHUNK_FLUSH}/{len(new_unique)} "
                    f"(mem-safe + gc.collect)")
        log(f"  saved cache: {len(feat_map)} entries (chunked append done)")

    # User directive 2026-08-29: free Stage 5 intermediates to reduce RSS.
    # feat_map (430K new entries) + all_sents (1.99M strings) + sent_to_user no longer needed.
    # Stage 6 streams cache file + Welford online accumulation (avoids user_z_list).
    del all_sents, sent_to_user, feat_map
    import gc as _gc
    _gc.collect()
    log("  Stage 5 intermediates freed (all_sents / sent_to_user / feat_map)")

    log("\n=== 6. Computing z_user per user (Welford online) ===")
    # User directive 2026-08-29: large cohort (1.94M users x 6M reviews) caused 3 OOM kills.
    # Old user_z_list held 1.94M users x avg 3 z's x 384B = 2.2GB + dict overhead +
    # sha1_to_indices 500MB + full cache load 1.96GB + runtime → 30GB limit OOM.
    # New design: Welford online accumulation, peak memory ~5GB.
    #   1) sha1_to_uid_idx: sha1 -> [uid_idx, ...] with multiplicity
    #      (1.96M keys x list ≈ 600MB)
    #   2) Welford state per user: count[N] + mean[N,48] + M2[N,48] = 1.5GB
    #   3) Stream cache file: per entry compute 1 z, then update Welford N times
    #      (N = number of users who wrote that review text)
    MIN_NONZERO_FEATS = 0

    # Build target_users ordered list + uid -> idx mapping
    target_user_list = sorted(target_users)
    uid_to_idx = {u: i for i, u in enumerate(target_user_list)}
    n_users = len(target_user_list)
    log(f"  target users ordered: {n_users}")

    # Per-user metadata (n_reviews, n_words) computed while building reverse index
    user_n_reviews: dict = {}
    user_n_words: dict = {}
    # sha1 -> list of uid_idx (with multiplicity; same uid appearing N times = N Welford updates)
    sha1_to_uid_idx: dict = collections.defaultdict(list)
    n_reviews_total = 0
    n_reviews_empty = 0
    for uid, texts in user_review_texts.items():
        user_n_reviews[uid] = len(texts)
        user_n_words[uid] = sum(len(t.split()) for t in texts)
        uid_idx = uid_to_idx.get(uid)
        if uid_idx is None:
            continue
        for t in texts:
            t = t.replace("\n", " ").strip()
            if not t:
                n_reviews_empty += 1
                continue
            sha1_to_uid_idx[feat_key(t)].append(uid_idx)
            n_reviews_total += 1
    log(f"  sha1_to_uid_idx built: {len(sha1_to_uid_idx)} unique keys, "
        f"{n_reviews_total} reviews (empty={n_reviews_empty})")
    # Free review texts (1.16GB), keep user_n_reviews/n_words (~50MB)
    del user_review_texts
    _gc.collect()
    log("  user_review_texts freed")

    # Welford state (online accumulation; no per-user z list)
    counts = np.zeros(n_users, dtype=np.int64)
    mean_acc = np.zeros((n_users, PCA_DIM), dtype=np.float64)
    M2_acc = np.zeros((n_users, PCA_DIM), dtype=np.float64)

    n_cache_seen = 0
    n_cache_matched = 0
    n_skip_sparse = 0
    if FEAT_CACHE.exists():
        try:
            with gzip.open(FEAT_CACHE, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    k = rec["k"]
                    v = rec["v"]
                    n_cache_seen += 1
                    uid_idxs = sha1_to_uid_idx.pop(k, None)
                    if uid_idxs is None:
                        continue
                    n_cache_matched += 1
                    nonzero_count = sum(1 for val in v.values() if val != 0.0)
                    if nonzero_count < MIN_NONZERO_FEATS:
                        n_skip_sparse += len(uid_idxs)
                        continue
                    vec = np.array([v.get(n, 0.0) for n in fnames], dtype=np.float64)
                    vec_scaled = scaler.transform(vec[None, :])[0]
                    z = pca.transform(vec_scaled[None, :])[0].astype(np.float64)
                    # Welford update per uid_idx (same uid multiple times = multiple updates
                    # with same z, mathematically equivalent to N-fold weighting)
                    for ui in uid_idxs:
                        counts[ui] += 1
                        delta = z - mean_acc[ui]
                        mean_acc[ui] = mean_acc[ui] + delta / counts[ui]
                        delta2 = z - mean_acc[ui]
                        M2_acc[ui] = M2_acc[ui] + delta * delta2
                    if n_cache_matched % 200_000 == 0:
                        _gc.collect()
                        n_with_rev = int((counts > 0).sum())
                        log(f"    streaming cache: {n_cache_matched}/{n_cache_seen} matched, "
                            f"sha1_to_uid_idx remaining: {len(sha1_to_uid_idx)}, "
                            f"users with reviews: {n_with_rev}/{n_users}")
        except (EOFError, gzip.BadGzipFile, zlib.error) as e:
            log(f"  WARN streaming cache truncated ({type(e).__name__}: {e!r}), "
                f"seen {n_cache_seen} entries")
    # Remaining sha1_to_uid_idx keys = cache did not cover these sentences → n_skip_no_feats
    n_skip_no_feats = sum(len(v) for v in sha1_to_uid_idx.values())
    log(f"  streaming cache done: {n_cache_seen} entries seen, "
        f"{n_cache_matched} matched, "
        f"users with reviews: {int((counts > 0).sum())}/{n_users}, "
        f"n_skip_sparse: {n_skip_sparse}, n_skip_no_feats: {n_skip_no_feats}")
    # Free reverse index (~600MB)
    sha1_to_uid_idx.clear()
    del sha1_to_uid_idx
    _gc.collect()

    log("\n=== 7. Computing per-user Gaussian (mu / sigma_diag) ===")
    # User directive 2026-08-29: compute mu/var from Welford state
    # (population var = M2/count). Skip the original global_pooled_var (dead code).
    user_gaussians = {}
    missing_users = []

    valid_mask = counts >= MIN_REVIEWS_FOR_PER_USER
    n_valid = int(valid_mask.sum())
    log(f"  users meeting MIN_REVIEWS_FOR_PER_USER={MIN_REVIEWS_FOR_PER_USER}: {n_valid}")

    # Vectorized: var_pop = M2/count, then LAMBDA shrinkage
    safe_counts = np.maximum(counts, 1).astype(np.float64)
    var_pop = M2_acc / safe_counts[:, None]
    var_per_user_mean = var_pop.mean(axis=1, keepdims=True)
    var_shrink = (1 - LAMBDA) * var_pop + LAMBDA * var_per_user_mean
    sigma_diag_arr = np.maximum(var_shrink, VAR_EPS)

    n_written = 0
    for uid_idx, uid in enumerate(target_user_list):
        if not valid_mask[uid_idx]:
            missing_users.append((uid, user_n_reviews.get(uid, 0), int(counts[uid_idx])))
            continue
        mu = mean_acc[uid_idx]
        sd = sigma_diag_arr[uid_idx]
        user_gaussians[uid] = {
            "mu": mu.tolist(),
            "sigma_diag": sd.tolist(),
            "n_reviews": user_n_reviews.get(uid, 0),
            "n_words": user_n_words.get(uid, 0),
            "n_sentences": int(counts[uid_idx]),
            "source": "per_user",
        }
        n_written += 1
    log(f"  per-user Gaussians written: {n_written}")

    if missing_users:
        # User directive 2026-08-29: 1.94M users nearly all satisfy MIN_REVIEWS_FOR_PER_USER=1.
        # Only a few users with 0 reviews. Log warning instead of raise.
        log(f"  WARN {len(missing_users)}/{n_users} users lack per-user Gaussian "
            f"(counts<{MIN_REVIEWS_FOR_PER_USER}):")
        for uid, n_reviews, n_sents in missing_users[:10]:
            log(f"      {uid[:12]}... reviews={n_reviews} sents={n_sents}")

    # Free Welford state
    del counts, mean_acc, M2_acc, var_pop, var_shrink, sigma_diag_arr
    _gc.collect()

    log("\n=== 8. Saving ===")
    GAUSSIANS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(GAUSSIANS_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": ("Stage 8.5: per-user Gaussian from Baby_Products review "
                                "corpus, PCA48 spaCy features (Welford online accumulation)"),
                "PCA_DIM": PCA_DIM,
                "LAMBDA": LAMBDA,
                "VAR_EPS": VAR_EPS,
                "MIN_REVIEWS_FOR_PER_USER": MIN_REVIEWS_FOR_PER_USER,
                "cache_signature": sig_hash,
            },
            "users": user_gaussians,
            "n_users_with_gaussian": len(user_gaussians),
            "n_users_skipped": len(target_users) - len(user_gaussians),
        }, f, ensure_ascii=False, indent=2)
    log(f"wrote -> {GAUSSIANS_OUT}")


def main():
    parser = argparse.ArgumentParser(description="Syntax Subspace - gaussian/ Stage 3 user Gaussians")
    args = parser.parse_args()
    log("=== syntax_subspace_user_gaussians.py ===")
    stage_user_gaussians()


if __name__ == "__main__":
    main()
