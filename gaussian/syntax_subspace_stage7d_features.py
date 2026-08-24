"""Stage 7D: extract 182d spaCy features for queries not in 7B cache.

For the 808 unique Stage7D queries missing from stage7b_query_features.jsonl.gz,
run per_sentence_features_v2 (spaCy batched) and append to the cache so 7C robust
can compute D_syn properly. Same pattern as 7B (passes whole Doc, filters to fnames).

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
        gaussian/syntax_subspace_stage7d_features.py \
        > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage7d_features.log 2>&1 &
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
RETRIEVAL_IN = SCRATCH / "stage7d_retrieval_results.json"
CACHE = SCRATCH / "stage7b_query_features.jsonl.gz"
OUT_NEW = SCRATCH / "stage7d_new_query_features.jsonl.gz"


def log(msg: str) -> None:
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def fk(t):
    return hashlib.sha1(t.strip().lower().encode("utf-8")).hexdigest()


def main():
    log("=== Stage 7D: extract 182d features for queries missing from 7B cache ===")

    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "gaussian"))
    sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))
    from gaussian_vades import _syntax_subspace_prepare  # type: ignore
    P = _syntax_subspace_prepare()
    fnames = P["feature_names_ordered"]
    log(f"  fnames count: {len(fnames)}")

    # Load queries from Stage7D retrieval
    retrieval = json.load(open(RETRIEVAL_IN))["results"]
    unique_queries = sorted(set(e["query"].strip() for e in retrieval))
    log(f"  Stage7D unique queries: {len(unique_queries)}")

    # Load existing 7B cache
    existing = {}
    with gzip.open(CACHE, "rt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rec = json.loads(line)
                existing[rec["k"]] = rec["v"]
            except Exception:
                continue
    log(f"  7B cache keys: {len(existing)}")

    # Find missing
    missing_q = [q for q in unique_queries if fk(q) not in existing]
    log(f"  missing: {len(missing_q)}")

    if not missing_q:
        log("  nothing to do")
        return

    # Import per_sentence_features_v2 (same as 7B uses)
    from syntactic_analysis.main import per_sentence_features_v2  # type: ignore

    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        log(f"  spaCy load failed: {e!r}")
        return

    # Extract features in batch via nlp.pipe (108x speedup)
    log(f"  extracting features for {len(missing_q)} queries via spaCy pipe...")
    new_entries = []
    n_ok = 0
    n_skip = 0
    for i, doc in enumerate(nlp.pipe(missing_q, batch_size=128, n_process=1)):
        q = missing_q[i]
        k = fk(q)
        try:
            feats = per_sentence_features_v2(doc)
            feats = feats if feats is not None else {}
        except Exception as e:
            log(f"  per_sentence_features_v2 error for '{q[:50]}': {e!r}")
            n_skip += 1
            continue
        # Filter to numeric features in fnames (182)
        numeric = {n: float(v) for n, v in feats.items() if isinstance(v, (int, float))}
        filtered = {n: numeric.get(n, 0.0) for n in fnames}
        new_entries.append({"k": k, "v": filtered})
        n_ok += 1
        if (i + 1) % 200 == 0:
            log(f"    {i + 1}/{len(missing_q)}")
    log(f"  extracted: {n_ok}, skipped: {n_skip}")

    # Write new cache file (just new entries)
    OUT_NEW.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT_NEW, "wt") as f:
        for e in new_entries:
            f.write(json.dumps(e) + "\n")
    log(f"wrote → {OUT_NEW} ({len(new_entries)} new entries)")

    # Also append to original cache so 7C picks them up
    with gzip.open(CACHE, "at") as f:
        for e in new_entries:
            f.write(json.dumps(e) + "\n")
    log(f"  appended to {CACHE}")


if __name__ == "__main__":
    main()