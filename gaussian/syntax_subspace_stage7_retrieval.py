"""Stage 7: Retriever Syntactic Robustness Evaluation

For each (asin, attrs_used, user_id) tuple with K=5 strict query variants:
1. Encode query with multiple retrievers (dense + lexical)
2. Retrieve top-K from Baby Products metadata corpus (~217K products)
3. Record target asin rank, RR, Hit@10 per query
4. Aggregate per-asin: std(RR), gap, IQR, hit instability
5. Run statistical tests (Friedman, Wilcoxon, bootstrap)

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
        gaussian/syntax_subspace_stage7_retrieval.py \
        > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage7_retrieval.log 2>&1 &

Env-var mode:
    STAGE7R_MODE = rank| analyze (default rank = both rank + analyze)
"""

from __future__ import annotations

import collections
import gzip
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats as scipy_stats


# === Paths (hardcoded) ===
REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
HF_CACHE = Path("/home/wlia0047/hj82_scratch2/hf_cache")
REGEN_IN = SCRATCH / "stage7_regen.json"
META_FILE = REPO_ROOT / "data/meta_Baby_Products_2023.jsonl.gz"
CORPUS_DOCS_OUT = SCRATCH / "stage7_corpus_docs.jsonl.gz"
CORPUS_EMB_MINILM = SCRATCH / "stage7_corpus_emb_minilm.npy"
CORPUS_BM25 = SCRATCH / "stage7_corpus_bm25.pkl"
RETRIEVAL_OUT = SCRATCH / "stage7_retrieval_results.json"

# === Constants ===
MIN_USERS_FOR_VOLATILITY = 3  # need ≥3 distinct users per asin
TOP_K_RETRIEVAL = 100  # we only need rank, but cap for BM25 speed
MIN_STRICT_USERS = 3
SEED = 2024

os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE))


def log(msg: str) -> None:
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# === Build corpus ===
def build_corpus(force_rebuild: bool = False):
    """Load metadata, build asin → doc text mapping."""
    if not force_rebuild and CORPUS_DOCS_OUT.exists():
        log(f"[corpus] loading cached {CORPUS_DOCS_OUT}")
        asin_to_doc = {}
        with gzip.open(CORPUS_DOCS_OUT, "rt", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                asin_to_doc[rec["asin"]] = rec["doc"]
        log(f"  cached docs: {len(asin_to_doc)}")
        return asin_to_doc

    log(f"[corpus] building from {META_FILE}")
    asin_to_doc = {}
    with gzip.open(META_FILE, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            asin = r.get("parent_asin", "").strip()
            if not asin:
                continue
            parts = []
            t = r.get("title", "").strip()
            if t:
                parts.append(t)
            desc = r.get("description", [])
            if isinstance(desc, list):
                desc = " ".join(desc)
            elif isinstance(desc, str):
                pass
            else:
                desc = ""
            desc = desc.strip()
            if desc:
                parts.append(desc)
            # Add features for better recall
            feats = r.get("features", [])
            if isinstance(feats, list):
                feats = " ".join(feats)
            if feats:
                parts.append(feats[:500])
            doc = " | ".join(parts).strip()
            if doc:
                asin_to_doc[asin] = doc[:1000]  # cap
    log(f"  loaded {len(asin_to_doc)} docs")
    # Cache
    CORPUS_DOCS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(CORPUS_DOCS_OUT, "wt", encoding="utf-8") as f:
        for asin, doc in asin_to_doc.items():
            f.write(json.dumps({"asin": asin, "doc": doc}) + "\n")
    log(f"  cached → {CORPUS_DOCS_OUT}")
    return asin_to_doc


# === Retriever 1: MiniLM dense ===
def encode_minilm(asin_to_doc: dict):
    """Encode corpus + queries with MiniLM."""
    from sentence_transformers import SentenceTransformer
    log("[minilm] loading model all-MiniLM-L6-v2")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    asins = sorted(asin_to_doc.keys())
    docs = [asin_to_doc[a] for a in asins]
    log(f"[minilm] encoding {len(docs)} corpus docs...")
    emb = model.encode(docs, batch_size=128, show_progress_bar=False, convert_to_numpy=True)
    log(f"[minilm] corpus emb shape: {emb.shape}")
    np.save(CORPUS_EMB_MINILM, emb)
    # Save asin order too
    with open(str(CORPUS_EMB_MINILM) + ".asins.json", "w") as f:
        json.dump(asins, f)
    return model, asins, emb


def load_minilm_corpus():
    emb = np.load(CORPUS_EMB_MINILM)
    with open(str(CORPUS_EMB_MINILM) + ".asins.json") as f:
        asins = json.load(f)
    return asins, emb


def retrieve_minilm(model, asins, emb, queries, target_asins, top_k=TOP_K_RETRIEVAL):
    """For each query, compute cosine sim against corpus, return rank of target."""
    q_emb = model.encode(queries, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    # Normalize for cosine
    q_norm = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)
    c_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)
    scores = q_norm @ c_norm.T  # [n_queries, n_corpus]
    ranks = []
    for i, target in enumerate(target_asins):
        try:
            tgt_idx = asins.index(target)
        except ValueError:
            ranks.append(None)  # target not in corpus
            continue
        # Sort descending; rank = position + 1
        # Use argpartition for speed, then sort top
        sorted_idx = np.argsort(-scores[i])[:top_k]
        # Find target position in sorted order
        if tgt_idx in sorted_idx:
            # Full rank: count of scores strictly greater than target's score
            rank = int((scores[i] > scores[i, tgt_idx]).sum()) + 1
        else:
            # Target not in top_k — assign rank > top_k
            rank = top_k + 1  # we cap it; not exactly unknown but close
            # Better: count all ranks strictly greater
            rank = int((scores[i] > scores[i, tgt_idx]).sum()) + 1
        ranks.append(rank)
    return ranks


# === Retriever 2: BM25 ===
def build_bm25(asin_to_doc: dict, asins: list):
    """Build BM25 index on corpus."""
    import pickle
    if CORPUS_BM25.exists():
        log(f"[bm25] loading cached {CORPUS_BM25}")
        with open(CORPUS_BM25, "rb") as f:
            return pickle.load(f)

    log("[bm25] building index...")
    from rank_bm25 import BM25Okapi
    tokenized_docs = []
    for a in asins:
        text = asin_to_doc[a].lower()
        # Simple tokenization: split on non-word, drop punct/digits
        tokens = re.findall(r"[a-z]+", text)
        tokenized_docs.append(tokens)
    bm25 = BM25Okapi(tokenized_docs)
    log(f"[bm25] built on {len(tokenized_docs)} docs")
    with open(CORPUS_BM25, "wb") as f:
        pickle.dump(bm25, f)
    return bm25


def _bm25_worker(args):
    """Module-level worker for multiprocessing (must be picklable)."""
    chunk_tokenized, bm25_path = args
    with open(bm25_path, "rb") as f:
        b = pickle.load(f)
    out = []
    for tokens in chunk_tokenized:
        scores = b.get_scores(tokens)
        out.append(scores)
    return out


def retrieve_bm25(bm25, queries, target_asins, asins, top_k=TOP_K_RETRIEVAL, n_workers=8):
    """For each query, get BM25 scores, compute rank of target.
    Parallelized via multiprocessing because rank_bm25 is pure-Python slow.
    """
    import multiprocessing as mp
    import tempfile
    # Pre-tokenize
    tokenized = [re.findall(r"[a-z]+", q.lower()) for q in queries]
    # Build asin→idx map
    asin_to_idx = {a: i for i, a in enumerate(asins)}
    tgt_indices = [asin_to_idx.get(t) for t in target_asins]

    # Save BM25 to temp file for workers
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tf:
        pickle.dump(bm25, tf)
        bm25_path = tf.name

    # Split tokenized into chunks
    chunks = []
    chunk_size = (len(tokenized) + n_workers - 1) // n_workers
    for i in range(0, len(tokenized), chunk_size):
        chunks.append((tokenized[i:i+chunk_size], bm25_path))

    log(f"[bm25] parallel scoring across {n_workers} workers, {len(tokenized)} queries")
    with mp.Pool(n_workers) as pool:
        chunk_results = pool.map(_bm25_worker, chunks)

    # Flatten
    all_scores = []
    for cr in chunk_results:
        all_scores.extend(cr)

    # Compute rank
    ranks = []
    for scores, tgt_idx in zip(all_scores, tgt_indices):
        if tgt_idx is None:
            ranks.append(None)
            continue
        rank = int((scores > scores[tgt_idx]).sum()) + 1
        ranks.append(rank)

    # Cleanup
    try:
        os.unlink(bm25_path)
    except Exception:
        pass
    return ranks


# === Per-query attributes ===
def per_query_attrs(text: str, attrs: dict) -> int:
    if not text:
        return 0
    text_lower = text.lower()
    covered = sum(1 for v in attrs.values() if v and str(v).lower() in text_lower)
    return covered


def has_invalid_punct(text: str) -> bool:
    if not text:
        return True
    bad_patterns = [
        r"^(here|this|below|sure|okay|ok)[,:]",
        r"^attribute[s]?:",
        r"^brand:", r"^color:", r"^material:", r"^style:",
    ]
    text_lower = text.lower().strip()
    return any(re.search(p, text_lower) for p in bad_patterns)


# === Main: rank mode ===
def main_rank():
    log("=== Stage 7 RETRIEVAL mode (rank) ===")
    log(f"loading {REGEN_IN}")
    regen_data = json.load(open(REGEN_IN, "r", encoding="utf-8"))
    regen = regen_data["entries"]
    log(f"  {len(regen)} entries (strict filter applied at regen time)")

    # Filter to strict only (already done in regen, but double-check)
    strict = [e for e in regen if e["strict"]]
    log(f"  strict: {len(strict)}")

    # Per-asin strict user count
    asin_users = collections.defaultdict(set)
    for e in strict:
        asin_users[e["asin"]].add(e["user_id"])
    valid_asins = {a for a, u in asin_users.items() if len(u) >= MIN_STRICT_USERS}
    log(f"  asins with ≥{MIN_STRICT_USERS} strict users: {len(valid_asins)}")

    eval_entries = [e for e in strict if e["asin"] in valid_asins]
    log(f"  eval entries: {len(eval_entries)}")
    queries = [e["query"] for e in eval_entries]
    targets = [e["asin"] for e in eval_entries]

    # 1. Build corpus (doc texts)
    asin_to_doc = build_corpus(force_rebuild=False)

    # 2. Build/load MiniLM corpus embeddings (must exist before retrieval)
    asins_path = Path(str(CORPUS_EMB_MINILM) + ".asins.json")
    if CORPUS_EMB_MINILM.exists() and asins_path.exists():
        log(f"[minilm] loading cached embeddings from {CORPUS_EMB_MINILM}")
        asins_corpus, emb = load_minilm_corpus()
    else:
        log("[minilm] no cached embeddings, encoding corpus...")
        # encode_minilm returns (model, asins, emb) but model can be loaded later
        _, asins_corpus, emb = encode_minilm(asin_to_doc)
    # Sanity check: every target must be in corpus
    missing = [t for t in set(targets) if t not in asins_corpus]
    log(f"  target asins missing from corpus: {len(missing)}")
    if missing:
        log(f"  WARNING: {len(missing)} targets not in corpus, will have rank=None")

    # Load MiniLM model for query encoding
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    log("[minilm] retrieving...")
    ranks_minilm = retrieve_minilm(model, asins_corpus, emb, queries, targets, top_k=100)
    log(f"  minilm done: {sum(1 for r in ranks_minilm if r is not None)}/{len(ranks_minilm)} ranked")

    # 3. BM25
    bm25 = build_bm25(asin_to_doc, asins_corpus)
    log("[bm25] retrieving...")
    ranks_bm25 = retrieve_bm25(bm25, queries, targets, asins_corpus, top_k=100)
    log(f"  bm25 done: {sum(1 for r in ranks_bm25 if r is not None)}/{len(ranks_bm25)} ranked")

    # 4. Save raw retrieval results
    results = []
    for e, rm, rb in zip(eval_entries, ranks_minilm, ranks_bm25):
        results.append({
            "asin": e["asin"],
            "user_id": e["user_id"],
            "attrs_used": e["attrs_used"],
            "query": e["query"],
            "n_tok": e["n_tok"],
            "k": e["k"],
            "rank_minilm": rm,
            "rank_bm25": rb,
            "rr_minilm": (1.0 / rm) if (rm is not None and rm > 0) else 0.0,
            "rr_bm25": (1.0 / rb) if (rb is not None and rb > 0) else 0.0,
            "hit10_minilm": int(rm is not None and rm <= 10),
            "hit10_bm25": int(rb is not None and rb <= 10),
        })

    RETRIEVAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(RETRIEVAL_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Stage 7 retrieval: minilm + bm25",
                "MIN_STRICT_USERS": MIN_STRICT_USERS,
                "TOP_K_RETRIEVAL": TOP_K_RETRIEVAL,
            },
            "n_eval_entries": len(eval_entries),
            "n_valid_asins": len(valid_asins),
            "asin_user_counts": {a: len(u) for a, u in asin_users.items() if a in valid_asins},
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    log(f"wrote → {RETRIEVAL_OUT}")


if __name__ == "__main__":
    main_rank()