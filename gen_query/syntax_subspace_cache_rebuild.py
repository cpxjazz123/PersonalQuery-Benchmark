"""Rebuild cache file: stream existing entries + extract pool features.

User directive 2026-08-29: The current cache has a broken gzip block from
Stage 5 chunked append, making appended entries unreadable. Rebuild by:
1. Streaming existing cache (1.524M valid entries) into new cache file
2. Extract spaCy features for pool queries missing from cache
3. Write consolidated cache (mode='wt')

This is a one-shot rebuild. After this, the cache is fully readable.

Input:
  /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage7b_query_features.jsonl.gz
  /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/pool_K200_F3pca48_full.json

Output (overwrite):
  /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage7b_query_features.jsonl.gz

Run:
  python gen_query/syntax_subspace_cache_rebuild.py
"""
from __future__ import annotations

import collections
import gzip
import json
import shutil
import sys
import zlib
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "common"))

from syntax_subspace_utils import FEAT_CACHE, log, feat_key  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "common"))

POOL_IN_FULL = Path(
    "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/pool_K200_F3pca48_full.json"
)
# Backup of original (broken) cache before overwrite
CACHE_BACKUP = FEAT_CACHE.with_suffix(".broken.gz")


def main() -> None:
    log("=== Cache rebuild (stream existing + extract pool features) ===")

    # Backup current (broken) cache first
    if FEAT_CACHE.exists():
        if not CACHE_BACKUP.exists():
            shutil.copy(FEAT_CACHE, CACHE_BACKUP)
            log(f"  backup broken cache -> {CACHE_BACKUP}")

    # 1. Stream existing cache to new file + collect keys
    tmp_cache = FEAT_CACHE.with_suffix(".rebuild.gz")
    seen_keys: set[str] = set()
    n_existing = 0
    if FEAT_CACHE.exists():
        try:
            with gzip.open(FEAT_CACHE, "rt", encoding="utf-8") as f_in, \
                 gzip.open(tmp_cache, "wt", encoding="utf-8") as f_out:
                f_out.write("# user-review + pool-query features (key=sha1(text), v=filtered dict)\n")
                for line in f_in:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "k" not in rec or "v" not in rec:
                        continue
                    f_out.write(json.dumps(rec) + "\n")
                    seen_keys.add(rec["k"])
                    n_existing += 1
        except (EOFError, gzip.BadGzipFile, zlib.error) as e:
            log(f"  WARN source cache truncated ({type(e).__name__}: {e!r}), "
                f"copied {n_existing} valid entries")
    log(f"  existing cache entries copied: {n_existing}")

    # 2. Find pool queries needing features
    pool = json.load(open(POOL_IN_FULL))
    unique_q = sorted(set(q for qs in pool["pools"].values() for q in [q["query"] for q in qs]))
    missing_q = [q for q in unique_q if feat_key(q) not in seen_keys]
    log(f"  pool queries needing features: {len(missing_q)} / {len(unique_q)}")

    # 3. Load fnames
    from syntax_subspace_utils import _syntax_subspace_prepare
    P = _syntax_subspace_prepare()
    fnames = P["feature_names_ordered"]
    log(f"  fnames: {len(fnames)}")

    # 4. Extract via spaCy + append
    if missing_q:
        import spacy
        from syntactic_features import per_sentence_features_v2
        nlp = spacy.load("en_core_web_sm")
        for comp in ("ner", "lemmatizer", "attribute_ruler"):
            if comp in nlp.pipe_names:
                nlp.disable_pipe(comp)

        N_PROCESS = 4
        BATCH_SIZE = 256
        CHUNK_FLUSH = 10_000
        log(f"  extracting features for {len(missing_q)} pool queries "
            f"(n_process={N_PROCESS}, batch={BATCH_SIZE}, "
            f"chunked append every {CHUNK_FLUSH})...")

        import gc as _gc
        n_processed = 0
        n_skip = 0
        with gzip.open(tmp_cache, "at", encoding="utf-8") as f_out:
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
                    f_out.write(json.dumps({"k": k, "v": filtered}) + "\n")
                    n_processed += 1
                f_out.flush()
                del chunk
                _gc.collect()
                log(f"    flushed chunk {chunk_start + CHUNK_FLUSH}/{len(missing_q)}")
        log(f"  done: extracted {n_processed}, skipped {n_skip}")

    # 5. Atomic swap: tmp -> canonical
    FEAT_CACHE.unlink()
    tmp_cache.rename(FEAT_CACHE)
    log(f"  swapped: {tmp_cache.name} -> {FEAT_CACHE.name}")

    # Verify
    n_check = 0
    try:
        with gzip.open(FEAT_CACHE, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    n_check += 1
    except (EOFError, gzip.BadGzipFile, zlib.error):
        pass
    log(f"  verify: {n_check} entries in new cache")


if __name__ == "__main__":
    main()
