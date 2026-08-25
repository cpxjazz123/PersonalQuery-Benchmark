"""Stage 8.5: Build per-user Gaussians from review corpus.

For each of 1000 Stage 8 users:
1. Scan Baby_Products_2023.jsonl.gz → collect their review texts
2. Sentence-split, extract 182d spaCy features per sentence
3. Mean-pool across all sentences → 182d per user
4. Project to PCA48 → 48d z_user[uid]
5. Estimate per-user Gaussian (mean + diagonal covariance with shrinkage)

Fallbacks:
- Users with <3 reviews: mu = their single z, sigma_diag = global pooled var
- Users with 0 reviews: skip (use ASIN centroid later in selection step)

Output: stage8_5_user_gaussians.json

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
        gaussian/syntax_subspace_stage8_5_user_gaussians.py \
        > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_user_gaussians.log 2>&1 &
"""

from __future__ import annotations

import collections
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


# === Paths ===
REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
REVIEW_GZ = REPO_ROOT / "data/Baby_Products_2023.jsonl.gz"
ASINS_IN = SCRATCH / "stage8_5_asins.json"
FEAT_CACHE = SCRATCH / "stage7b_query_features.jsonl.gz"
OUT = SCRATCH / "stage8_5_user_gaussians.json"

# === Constants ===
PCA_DIM = 48
LAMBDA = 0.1   # shrinkage strength
VAR_EPS = 1e-3
MIN_REVIEWS_FOR_PER_USER = 3   # below this → use global var (per_ASIN sentence count)
# Note: ASIN sampling already requires each user has ≥35 TOTAL reviews
# (enforced in stage8_5_scan.py)


def log(msg: str) -> None:
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def feat_key(t: str) -> str:
    return hashlib.sha1(t.strip().lower().encode("utf-8")).hexdigest()


def main():
    log("=== Stage 8.5: Build per-user Gaussians ===")

    # === 1. Load PCA48 setup (frozen) ===
    log("\n=== 1. Loading PCA48 ===")
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "gaussian"))
    sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

    from gaussian_vades import _syntax_subspace_prepare
    P = _syntax_subspace_prepare()
    scaler = P["scaler"]
    fnames = P["feature_names_ordered"]
    train_idx = P["train_idx"]
    log(f"  scaler mean shape: {scaler.mean_.shape}, fnames: {len(fnames)}")

    from sklearn.decomposition import PCA
    pca = PCA(n_components=PCA_DIM, random_state=42)
    pca.fit(P["X_scaled"][train_idx])
    log(f"  PCA48 EV={pca.explained_variance_ratio_.sum():.4f}")

    # === 2. Load target users ===
    log("\n=== 2. Loading target users ===")
    asin_data = json.load(open(ASINS_IN))["asins"]
    target_users = set()
    user_to_asins = collections.defaultdict(set)
    for a in asin_data:
        for uid in a["users_sampled"]:
            target_users.add(uid)
            user_to_asins[uid].add(a["asin"])
    log(f"  target users: {len(target_users)}")

    # === 3. Scan review corpus for these users' review texts ===
    log("\n=== 3. Scanning review corpus ===")
    user_review_texts = collections.defaultdict(list)
    n_records = 0
    for line in gzip.open(REVIEW_GZ, "rt", encoding="utf-8"):
        r = json.loads(line)
        uid = r.get("user_id", "")
        if uid in target_users and r.get("text"):
            user_review_texts[uid].append(r["text"])
        n_records += 1
        if n_records % 2_000_000 == 0:
            log(f"    {n_records/1e6:.1f}M records, {len(user_review_texts)} users found")
    log(f"  total records: {n_records}")
    log(f"  users with reviews: {len(user_review_texts)}")

    review_counts = [len(v) for v in user_review_texts.values()]
    log(f"  review count: min={min(review_counts) if review_counts else 0}, "
        f"mean={sum(review_counts)/len(review_counts):.1f}, "
        f"max={max(review_counts) if review_counts else 0}")
    n_high = sum(1 for c in review_counts if c >= MIN_REVIEWS_FOR_PER_USER)
    log(f"  users with ≥{MIN_REVIEWS_FOR_PER_USER} reviews (per_user Gaussian): {n_high}")
    n_low = sum(1 for c in review_counts if c < MIN_REVIEWS_FOR_PER_USER and c >= 1)
    log(f"  users with 1-2 reviews (global var fallback): {n_low}")
    n_zero = len(target_users) - len(user_review_texts)
    log(f"  users with 0 reviews (asin centroid fallback): {n_zero}")

    # === 4. Load feature cache ===
    log("\n=== 4. Loading feature cache ===")
    feat_map = {}
    if FEAT_CACHE.exists():
        with gzip.open(FEAT_CACHE, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                rec = json.loads(line)
                feat_map[rec["k"]] = rec["v"]
    log(f"  cache loaded: {len(feat_map)} features")

    # === 5. Extract spaCy features for user review texts ===
    log("\n=== 5. Extracting spaCy features for user review texts ===")
    # Sentence-split all review texts → collect all unique sentences
    all_sents = []
    sent_to_user = []  # parallel list: which user each sentence belongs to
    for uid, texts in user_review_texts.items():
        for t in texts:
            # Simple sentence split on . ! ? followed by space
            for s in t.replace("\n", " ").split(". "):
                s = s.strip()
                if s and len(s.split()) >= 3:
                    all_sents.append(s)
                    sent_to_user.append(uid)
    log(f"  total sentences: {len(all_sents)}")

    # Find missing
    new_sents = [s for s in all_sents if feat_key(s) not in feat_map]
    log(f"  unique sentences: {len(set(all_sents))}, new to extract: {len(new_sents)}")

    if new_sents:
        import spacy
        from main import per_sentence_features_v2
        nlp = spacy.load("en_core_web_sm")
        log(f"  extracting features for {len(new_sents)} new sentences (n_process=8, batch=256)...")
        new_unique = sorted(set(new_sents))
        # Use multiprocessing for ~6-8x speedup
        docs = list(nlp.pipe(new_unique, batch_size=256, n_process=8))
        for i, doc in enumerate(docs):
            s = new_unique[i]
            k = feat_key(s)
            try:
                feats = per_sentence_features_v2(doc)
                feats = feats if feats is not None else {}
            except Exception:
                feats = {}
            numeric = {n: float(v) for n, v in feats.items() if isinstance(v, (int, float))}
            filtered = {n: numeric.get(n, 0.0) for n in fnames}
            feat_map[k] = filtered
            if (i + 1) % 5000 == 0:
                log(f"    {i + 1}/{len(new_unique)}")
        # Save cache
        with gzip.open(FEAT_CACHE, "wt", encoding="utf-8") as f:
            f.write("# user-review sentence features (key=sha1(text), v=182d dict)\n")
            for k, v in feat_map.items():
                f.write(json.dumps({"k": k, "v": v}) + "\n")
        log(f"  saved cache: {len(feat_map)} entries")

    # === 6. Compute z vectors per user (mean-pool across their sentences) ===
    log("\n=== 6. Computing z_user per user (mean-pool) ===")
    user_z_list = collections.defaultdict(list)
    n_skip = 0
    for s, uid in zip(all_sents, sent_to_user):
        feats = feat_map.get(feat_key(s))
        if not feats:
            n_skip += 1
            continue
        vec = np.array([feats.get(n, 0.0) for n in fnames], dtype=np.float64)
        vec_scaled = scaler.transform(vec[None, :])[0]
        z = pca.transform(vec_scaled[None, :])[0]
        user_z_list[uid].append(z)
    log(f"  sentences projected: {len(all_sents) - n_skip}, skipped: {n_skip}")

    # === 7. Build global pooled variance (fallback) ===
    log("\n=== 7. Building global pooled variance ===")
    all_z = []
    for uid, zs in user_z_list.items():
        all_z.extend(zs)
    all_z = np.stack(all_z, axis=0)
    global_var = all_z.var(axis=0)
    log(f"  global var shape: {global_var.shape}, mean: {global_var.mean():.4f}")

    # === 8. Build per-user Gaussians ===
    log("\n=== 8. Building per-user Gaussians ===")
    user_gaussians = {}
    n_per_user = 0
    n_fallback = 0

    for uid in target_users:
        zs = user_z_list.get(uid, [])
        n_reviews = len(user_review_texts.get(uid, []))
        n_words = sum(len(t.split()) for t in user_review_texts.get(uid, []))

        if len(zs) >= MIN_REVIEWS_FOR_PER_USER:
            # Per-user Gaussian
            Z = np.stack(zs, axis=0)
            mu = Z.mean(axis=0)
            var = Z.var(axis=0)
            var_shrink = (1 - LAMBDA) * var + LAMBDA * var.mean()
            sigma_diag = np.maximum(var_shrink, VAR_EPS)
            source = "per_user"
            n_per_user += 1
        elif len(zs) >= 1:
            # Fallback: their single z, global var
            Z = np.stack(zs, axis=0)
            mu = Z.mean(axis=0)
            sigma_diag = np.maximum(global_var, VAR_EPS)
            source = "global_var_fallback"
            n_fallback += 1
        else:
            # No reviews — skip; will use ASIN centroid in selection step
            continue

        user_gaussians[uid] = {
            "mu": mu.tolist(),
            "sigma_diag": sigma_diag.tolist(),
            "n_reviews": n_reviews,
            "n_words": n_words,
            "n_sentences": len(zs),
            "source": source,
        }

    log(f"  per-user Gaussians: {n_per_user}")
    log(f"  global-var fallback: {n_fallback}")
    log(f"  no reviews (skip): {len(target_users) - len(user_gaussians)}")

    # === 9. Save ===
    log("\n=== 9. Saving ===")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Stage 8.5: per-user Gaussian from Baby_Products review corpus, PCA48 spaCy features",
                "PCA_DIM": PCA_DIM,
                "LAMBDA": LAMBDA,
                "VAR_EPS": VAR_EPS,
                "MIN_REVIEWS_FOR_PER_USER": MIN_REVIEWS_FOR_PER_USER,
            },
            "global_var": global_var.tolist(),
            "users": user_gaussians,
            "n_users_with_gaussian": len(user_gaussians),
            "n_users_skipped": len(target_users) - len(user_gaussians),
        }, f, ensure_ascii=False, indent=2)
    log(f"wrote → {OUT}")


if __name__ == "__main__":
    main()