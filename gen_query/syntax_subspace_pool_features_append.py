"""Extract spaCy 182d features for EXPAND pool queries and APPEND to cache.

User directive 2026-08-29: Stage 2 features script (syntax_subspace_pool_regen.py)
uses mode='wt' which OVERWRITES the cache, destroying user-review features.
This script loads the full EXPAND pool and APPENDS new feature entries to the
existing cache (mode='at') so the cache accumulates across Stage 3 + Stage 2.

Input:
  /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/pool_K200_F3pca48_full.json
  /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage7b_query_features.jsonl.gz

Output (append-only):
  /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage7b_query_features.jsonl.gz

Run:
  python gen_query/syntax_subspace_pool_features_append.py
"""
from __future__ import annotations

import collections
import gzip
import json
import sys
import zlib
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "common"))

from syntax_subspace_utils import FEAT_CACHE, log, feat_key  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "common"))

POOL_IN_FULL = Path(
    "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/pool_K200_F3pca48_full.json"
)


def main() -> None:
    log("=== Pool feature extraction (APPEND mode) ===")

    # 1. Load full pool queries
    pool = json.load(open(POOL_IN_FULL))
    pool_queries: list[str] = []
    for asin, qs in pool["pools"].items():
        for q in qs:
            pool_queries.append(q["query"])
    log(f"  pool queries: {len(pool_queries)}, "
        f"unique: {len(set(pool_queries))}")

    # 2. Load existing cache keys (lazy: only keys)
    seen_keys: set[str] = set()
    if FEAT_CACHE.exists():
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

    # 3. Find missing queries
    unique_pool = sorted(set(pool_queries))
    missing_q = [q for q in unique_pool if feat_key(q) not in seen_keys]
    log(f"  pool queries needing features: {len(missing_q)} / {len(unique_pool)}")

    if not missing_q:
        log("  no new features needed")
        return

    # 4. Load feature names
    from syntax_subspace_utils import _syntax_subspace_prepare
    P = _syntax_subspace_prepare()
    fnames = P["feature_names_ordered"]
    log(f"  fnames: {len(fnames)}")

    # 5. Extract via spaCy pipe (chunked append, gc-safe)
    import spacy
    from syntactic_features import per_sentence_features_v2
    nlp = spacy.load("en_core_web_sm")
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)

    N_PROCESS = 4
    BATCH_SIZE = 256
    CHUNK_FLUSH = 10_000
    log(f"  extracting features for {len(missing_q)} queries "
        f"(n_process={N_PROCESS}, batch={BATCH_SIZE}, "
        f"chunked append every {CHUNK_FLUSH})...")

    import gc as _gc
    n_processed = 0
    n_skip = 0
    with gzip.open(FEAT_CACHE, "at", encoding="utf-8") as f_cache:
        for chunk_start in range(0, len(missing_q), CHUNK_FLUSH):
            chunk = missing_q[chunk_start:chunk_start + CHUNK_FLUSH]
            for i, doc in enumerate(nlp.pipe(chunk, batch_size=BATCH_SIZE,
                                              n_process=N_PROCESS)):
                q = chunk[i]
                k = feat_key(q)
                try:
                    feats = per_sentence_features_v2(doc)
                    feats = feats if feats is not None else {}
                except Exception:
                    n_skip += 1
                    continue
                numeric = {n: float(v) for n, v in feats.items()
                           if isinstance(v, (int, float))}
                filtered = {n: numeric.get(n, 0.0) for n in fnames}
                f_cache.write(json.dumps({"k": k, "v": filtered}) + "\n")
                n_processed += 1
            f_cache.flush()
            del chunk
            _gc.collect()
            log(f"    flushed chunk {chunk_start + CHUNK_FLUSH}/{len(missing_q)}")
    log(f"  done: extracted {n_processed}, skipped {n_skip}")


if __name__ == "__main__":
    main()
