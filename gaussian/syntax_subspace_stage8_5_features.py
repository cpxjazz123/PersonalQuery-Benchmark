"""Stage 8.5: Extract spaCy features for SHARED pool queries.

Reuses pattern from stage8_features.py:
- Loads existing feature cache from stage7b_query_features.jsonl.gz
- For each query in pool not yet in cache, extracts 182d features
- Appends to cache

Output: appended cache (no separate output file)

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
        gaussian/syntax_subspace_stage8_5_features.py \
        > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_features.log 2>&1 &
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
POOL_IN = SCRATCH / "stage8_5_pool.json"
FEAT_CACHE = SCRATCH / "stage7b_query_features.jsonl.gz"


def log(msg: str) -> None:
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def feat_key(t: str) -> str:
    return hashlib.sha1(t.strip().lower().encode("utf-8")).hexdigest()


def main():
    log("=== Stage 8.5: Extract features for pool queries ===")

    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "gaussian"))
    sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

    from gaussian_vades import _syntax_subspace_prepare
    P = _syntax_subspace_prepare()
    fnames = P["feature_names_ordered"]
    log(f"  fnames: {len(fnames)}")

    # Load pool
    log(f"loading {POOL_IN}")
    pool_data = json.load(open(POOL_IN))
    pools = pool_data["pools"]
    all_queries = []
    for asin, qs in pools.items():
        for q in qs:
            all_queries.append(q["query"])
    log(f"  total pool queries: {len(all_queries)}")

    # Load cache
    feat_map = {}
    if FEAT_CACHE.exists():
        with gzip.open(FEAT_CACHE, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                rec = json.loads(line)
                feat_map[rec["k"]] = rec["v"]
    log(f"  cache keys: {len(feat_map)}")

    # Find missing
    missing_q = [q for q in all_queries if feat_key(q) not in feat_map]
    log(f"  missing: {len(missing_q)}")

    if not missing_q:
        log("  no new features needed")
        return

    # Extract via spaCy
    import spacy
    from main import per_sentence_features_v2
    nlp = spacy.load("en_core_web_sm")

    log(f"  extracting features for {len(missing_q)} queries via spaCy pipe (n_process=8, batch=256)...")
    new_unique = sorted(set(missing_q))
    new_entries = []
    n_skip = 0
    docs = list(nlp.pipe(new_unique, batch_size=256, n_process=8))
    for i, doc in enumerate(docs):
        q = new_unique[i]
        k = feat_key(q)
        try:
            feats = per_sentence_features_v2(doc)
            feats = feats if feats is not None else {}
        except Exception:
            n_skip += 1
            continue
        numeric = {n: float(v) for n, v in feats.items() if isinstance(v, (int, float))}
        filtered = {n: numeric.get(n, 0.0) for n in fnames}
        feat_map[k] = filtered
        new_entries.append({"k": k, "v": filtered})
        if (i + 1) % 1000 == 0:
            log(f"    {i + 1}/{len(new_unique)}")

    log(f"  extracted: {len(new_entries)}, skipped: {n_skip}")

    # Append to cache
    with gzip.open(FEAT_CACHE, "wt", encoding="utf-8") as f:
        f.write("# spaCy 182d sentence features (key=sha1(text), v=filtered dict)\n")
        for k, v in feat_map.items():
            f.write(json.dumps({"k": k, "v": v}) + "\n")
    log(f"  saved cache: {len(feat_map)} entries")


if __name__ == "__main__":
    main()