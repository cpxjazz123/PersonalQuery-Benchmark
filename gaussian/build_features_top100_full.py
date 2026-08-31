"""Build the FULL per_sentence_features_v2 feature cache for the top-100
reviewers cohort (8.E).

Output one row per sentence, aligned with the canonical
`feature_names_ordered` (182 names) used by every downstream 8.x /
scaler / PCA48 / F3 stage. This is the "wider" base layer of features
beneath the previous F3-only (103d) cache — every downstream column
subset (F3=103, F3+postbg, etc.) can be read from here without re-running
spaCy.

Cache:
  /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/
      phase8e_full_feats_<cohort_hash>.npz
        feats    (N, 182) float32
        uid_idx  (N,)     int32  — index into phase8e_cohort_top100 users[]
        split    (N,)     int8   — 0=train, 1=test (per is_train_sentence)
        order    (N,)     int32  — sentence position in the cache load

Sentence cache (already built by _build_top100_cache.py):
  .../phase8a_user_sents_17bb1b376e80.jsonl.gz  (86,787 sents)

Run:
  cd /home/wlia0047/ar57/wenyu/PersoanlQuery
  nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python -u \
      gaussian/build_features_top100_full.py \
      > /home/wlia0047/hj82_scratch2/wenyu/logs/phase8e_full_feats.log 2>&1 &
"""
import hashlib, json, os, sys, time
import numpy as np

REPO = "/home/wlia0047/ar57/wenyu/PersoanlQuery"
sys.path.insert(0, f"{REPO}/common")
from syntactic_features import per_sentence_features_v2 as psf
import spacy

from _user_sentence_cache import load_cache

# ────────── Config (must match learned_syntax_gaussian_8e_top100_reviews.py) ──────────
COHORT_PATH = f"{REPO}/result/gaussian/phase8e_cohort_top100.json"
SEED_PREFIX = "phase8e_v1|"
TRAIN_FRAC = 0.80

# Canonical feature_names_ordered (182) — the master 8.x scaler was fit on this
# exact order; any deviation would silently misalign the per-ASIN z-space.
gdoc = np.load("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/"
               "stage8_5_shared_scaler.npz", allow_pickle=True)
ALL_FNAMES = list(gdoc["feature_names_ordered"])
gdoc.close()
assert len(ALL_FNAMES) == 182, f"expected 182 ordered names, got {len(ALL_FNAMES)}"
D = len(ALL_FNAMES)

# Cohort
with open(COHORT_PATH) as f:
    _cohort_data = json.load(f)
COHORT = [u["uid"] for u in _cohort_data["users"]]
uid_to_idx = {uid: i for i, uid in enumerate(COHORT)}

# Cache key = cohort-hash only (sentence cache already keys by this). Naming
# consistency lets us map feats ↔ sentence cache from filename alone.
cohort_hash = hashlib.sha1("-".join(sorted(COHORT)).encode()).hexdigest()[:12]
OUT_NPZ = (f"/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/"
           f"phase8e_full_feats_{cohort_hash}.npz")

# CPU-only: spacy.require_cpu() avoids CUDA context fork conflicts with
# multiprocessing pool (the previous GPU path left the main thread polling
# cuda00001800007 forever, dead-locking the forking pool). n_process=4
# keeps RSS under 3 GB/process on shared box.
import spacy
spacy.require_cpu()
NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "attribute_ruler"])


def is_train_sentence(text):
    h = int(hashlib.sha1((SEED_PREFIX + text.strip().lower()).encode()).hexdigest(), 16)
    return (h % 100) < int(TRAIN_FRAC * 100)


def featurize_one(doc):
    """Run psf on each sentence in `doc` and average the per-sent dicts.

    Mirrors the exact pipeline in learned_syntax_gaussian_8e_top100_reviews.py
    and pca48_holdout_zero_shot_baseline.py: split-sentences → psf each → mean
    pool numeric keys → project onto ALL_FNAMES order. Per the audit, this
    matches every 8.x scaling/projection invariant.
    """
    sds = list(doc.sents)
    if not sds:
        return None
    feats = []
    for s in sds:
        f = psf(s)
        if f is not None:
            feats.append(f)
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
            except Exception:
                pass
    return np.array([mean_feats.get(nm, 0.0) for nm in ALL_FNAMES],
                    dtype=np.float32)


# ────────── Main ──────────
def main():
    if os.path.exists(OUT_NPZ):
        z = np.load(OUT_NPZ)
        print(f"[cache] hit  {OUT_NPZ}")
        print(f"  feats {z['feats'].shape}  uid_idx {z['uid_idx'].shape}  "
              f"split {z['split'].shape}  order {z['order'].shape}")
        print(f"  n_train={int((z['split']==0).sum())}  n_test={int((z['split']==1).sum())}")
        return

    print(f"[1] loading sentence cache (key hash={cohort_hash})...")
    t0 = time.time()
    user_sents = load_cache(COHORT)  # dict[uid] -> list[str]
    assert user_sents is not None, "sentence cache not built yet"
    n_sents = sum(len(v) for v in user_sents.values())
    print(f"  loaded {len(user_sents)} users / {n_sents:,} sents in {time.time()-t0:.0f}s")

    # Fixed iteration order: COHORT order (matches downstream training).
    # This makes feat[i] ↔ (COHORT[uid_idx[i]], order[i]) deterministic.
    rows_text = []
    rows_uid = []
    rows_split = []
    rows_order = []
    for uid in COHORT:
        sents = user_sents.get(uid, [])
        for j, s in enumerate(sents):
            rows_text.append(s)
            rows_uid.append(uid_to_idx[uid])
            rows_split.append(0 if is_train_sentence(s) else 1)
            rows_order.append(j)
    N = len(rows_text)
    print(f"  fixed-order rows: {N:,}")

    # Batch featurize via spaCy nlp.pipe. A "doc" here is one sentence (already
    # split by the cache build), so featurize_one(doc) will average over its
    # (typically one) sentence — matches featurize_103 semantics exactly.
    print(f"\n[2] batch featurize → {D}d via spaCy (CPU, n_process=1, batch=512)...")
    print("    (n_process=1: avoid spaCy multiprocessing pool deadlock seen at n>=2)")
    t1 = time.time()
    feats = np.zeros((N, D), dtype=np.float32)
    BATCH = 512
    PROGRESS_EVERY = 10  # batches; BATCH*PROGRESS_EVERY=5120 sents per line
    n_done = 0
    n_null = 0
    for i in range(0, N, BATCH):
        chunk = rows_text[i:i + BATCH]
        # n_process=1 keeps this in-process; spaCy's tok2vec/tagger/parser still
        # batch-parallelize internally (cython), so per-sent throughput stays
        # ~1ms, total ~90s for 86k sents. 100% reproducible, no fork overhead.
        docs = list(NLP.pipe(chunk, batch_size=BATCH, n_process=1))
        for k, d in enumerate(docs):
            v = featurize_one(d)
            if v is None:
                n_null += 1
                # leave zero row (matches downstream _featurize_one_sentence behavior)
            else:
                feats[i + k] = v
        n_done += len(chunk)
        if (i // BATCH) % PROGRESS_EVERY == 0:
            # explicit log-line write (open/close) so background buffer flushes
            with open("/home/wlia0047/hj82_scratch2/wenyu/logs/phase8e_full_feats.progress",
                      "a") as plog:
                plog.write(f"  {n_done:,}/{N:,}  ({100*n_done/N:.1f}%)  "
                           f"elapsed {time.time()-t1:.0f}s  null={n_null}\n")
            print(f"  {n_done:,}/{N:,}  ({100*n_done/N:.1f}%)  "
                  f"elapsed {time.time()-t1:.0f}s  null={n_null}", flush=True)
    print(f"  done in {time.time()-t1:.0f}s, null={n_null}/{N}")

    # Save
    print(f"\n[3] saving npz → {OUT_NPZ}")
    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez(OUT_NPZ,
             feats=feats,
             uid_idx=np.asarray(rows_uid, dtype=np.int32),
             split=np.asarray(rows_split, dtype=np.int8),
             order=np.asarray(rows_order, dtype=np.int32))
    print(f"  feats {feats.shape}  uid_idx {(np.asarray(rows_uid)).shape}  "
          f"split {(np.asarray(rows_split)).shape}  order {(np.asarray(rows_order)).shape}")
    print(f"  n_train={int((np.asarray(rows_split)==0).sum())}  "
          f"n_test={int((np.asarray(rows_split)==1).sum())}")
    print(f"\n[done] total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
