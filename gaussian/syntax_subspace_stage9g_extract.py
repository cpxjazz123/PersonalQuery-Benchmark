"""Stage 9G-1 — Enriched feature extraction for ablation.

For each query in 9C + 9F pool, extract THREE feature groups via spaCy:

Group A (Base182): existing 182d from stage7b_query_features cache
Group B (+ Clause/Dependency): ~20 new structural features
  - subordinate_clause_ratio, relative_clause_ratio
  - main_sub_clause_ratio, clause_depth_max, clause_depth_mean
  - coordination_span_mean, prep_count, prep_avg_attachment_distance
  - modifier_before_head_ratio, modifier_after_head_ratio
  - passive_count, copula_count
  - top dep_rel normalized counts (10 most common dep_rels)
  - dep_bigram "nsubj+verb" pattern presence
  - root_dep_type (root's dependency to grandparent)
Group C (+ Connective-specific): ~15 lexical connective features
  - which_count, that_rel_count, who_count, because_count, since_count
  - while_count, although_count, when_count, if_count, for_count
  - with_count, in_count, of_count, by_count, at_count

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
        gaussian/syntax_subspace_stage9g_extract.py \
        > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage9g_extract_run.log 2>&1 &
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
POOL_9C = SCRATCH / "stage9c_pool_pilot.json"
POOL_9F = SCRATCH / "stage9fb_pool.json"
FEAT_CACHE_182D = SCRATCH / "stage7b_query_features.jsonl.gz"
FEAT_ENRICHED = SCRATCH / "stage9g_enriched_features.jsonl.gz"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def feat_key(t: str) -> str:
    return hashlib.sha1(t.strip().lower().encode("utf-8")).hexdigest()


# === Group B: Clause/Dependency features ===
CLAUSE_DEPS = {"acl", "advcl", "ccomp", "relcl", "xcomp"}

# Top dep_rels from training corpus (commonly seen in queries)
TOP_DEP_RELS = [
    "nsubj", "nsubjpass", "obj", "iobj", "obl", "amod", "advmod",
    "prep", "pobj", "det", "compound", "conj", "cc", "aux", "auxpass",
    "punct", "case", "mark", "acl", "relcl", "advcl",
]


def extract_clause_dependency(doc) -> dict:
    """Extract Group B: Clause/Dependency features from spaCy doc."""
    feats = {}

    # Subordinate clause counts
    n_acl = sum(1 for t in doc if t.dep_ == "acl")
    n_advcl = sum(1 for t in doc if t.dep_ == "advcl")
    n_ccomp = sum(1 for t in doc if t.dep_ == "ccomp")
    n_relcl = sum(1 for t in doc if t.dep_ == "relcl")
    n_xcomp = sum(1 for t in doc if t.dep_ == "xcomp")

    feats["acl_count"] = n_acl
    feats["advcl_count"] = n_advcl
    feats["ccomp_count"] = n_ccomp
    feats["relcl_count"] = n_relcl
    feats["xcomp_count"] = n_xcomp

    total_clauses = n_acl + n_advcl + n_ccomp + n_relcl + n_xcomp
    feats["total_clause_count"] = total_clauses

    # Ratios (avoid /0)
    n_tok = len(doc)
    if total_clauses > 0:
        feats["subordinate_clause_ratio"] = (n_advcl + n_ccomp + n_xcomp) / total_clauses
        feats["relative_clause_ratio"] = n_relcl / total_clauses
    else:
        feats["subordinate_clause_ratio"] = 0.0
        feats["relative_clause_ratio"] = 0.0

    # Main/sub ratio
    feats["main_sub_clause_ratio"] = (1.0 if total_clauses == 0 else n_tok / max(total_clauses, 1)) / max(n_tok, 1)

    # Clause depth: max path length from token to root via clause deps
    def chain_depth(token):
        d = 0
        current = token
        while current.dep_ in CLAUSE_DEPS and current.head != current:
            d += 1
            current = current.head
        return d

    clause_tokens = [t for t in doc if t.dep_ in CLAUSE_DEPS]
    if clause_tokens:
        feats["clause_depth_max"] = max(chain_depth(t) for t in clause_tokens)
        feats["clause_depth_mean"] = float(np.mean([chain_depth(t) for t in clause_tokens]))
    else:
        feats["clause_depth_max"] = 0
        feats["clause_depth_mean"] = 0.0

    # Coordination span: avg tokens between coordinators
    conj_positions = [t.i for t in doc if t.dep_ in ("conj", "cc")]
    if len(conj_positions) >= 2:
        spans = [conj_positions[i + 1] - conj_positions[i] for i in range(len(conj_positions) - 1)]
        feats["coordination_span_mean"] = float(np.mean(spans))
        feats["coordination_span_max"] = int(np.max(spans))
    else:
        feats["coordination_span_mean"] = 0.0
        feats["coordination_span_max"] = 0

    # Prepositional phrases
    n_prep = sum(1 for t in doc if t.dep_ == "prep")
    feats["prep_count"] = n_prep
    if n_prep > 0:
        # avg distance from prep to its pobj
        prep_distances = []
        for t in doc:
            if t.dep_ == "prep":
                for child in t.children:
                    if child.dep_ == "pobj":
                        prep_distances.append(abs(child.i - t.i))
        feats["prep_avg_attachment_distance"] = float(np.mean(prep_distances)) if prep_distances else 0.0
    else:
        feats["prep_avg_attachment_distance"] = 0.0

    # Modifier placement (amod before vs after head noun)
    modifiers_before = 0
    modifiers_after = 0
    for t in doc:
        if t.dep_ == "amod":
            if t.i < t.head.i:
                modifiers_before += 1
            else:
                modifiers_after += 1
    total_mod = modifiers_before + modifiers_after
    if total_mod > 0:
        feats["modifier_before_head_ratio"] = modifiers_before / total_mod
        feats["modifier_after_head_ratio"] = modifiers_after / total_mod
    else:
        feats["modifier_before_head_ratio"] = 0.0
        feats["modifier_after_head_ratio"] = 0.0

    # Passive voice
    feats["passive_count"] = sum(1 for t in doc if t.dep_ in ("nsubjpass", "auxpass"))
    # Copula
    feats["copula_count"] = sum(1 for t in doc if t.dep_ == "cop")

    # Top dep_rel histogram (normalized)
    dep_counts = Counter(t.dep_ for t in doc)
    for d in TOP_DEP_RELS:
        feats[f"depnorm_{d}"] = dep_counts.get(d, 0) / max(n_tok, 1)

    # Pattern: nsubj + root + obj (canonical SVO)
    has_nsubj = any(t.dep_ == "nsubj" for t in doc)
    has_obj = any(t.dep_ in ("obj", "iobj") for t in doc)
    feats["has_svo_pattern"] = int(has_nsubj and has_obj)

    return feats


# === Group C: Connective-specific features ===
CONNECTIVE_WORDS = [
    "which", "that", "who", "whom", "whose",
    "because", "since", "while", "although", "when", "if",
    "for", "with", "in", "of", "by", "at", "on", "from", "to", "about",
]


def extract_connective(text: str) -> dict:
    """Extract Group C: connective-specific word counts."""
    tl = text.lower().strip()
    feats = {}
    for w in CONNECTIVE_WORDS:
        # Count as standalone word boundary
        pattern = r"\b" + re.escape(w) + r"\b"
        feats[f"cnt_{w}"] = len(re.findall(pattern, tl))
    return feats


def main():
    log("=== Stage 9G-1 — Enriched feature extraction ===")

    # Load existing 182d cache
    log("\n=== 1. Loading 182d cache ===")
    cache_182d = {}
    with gzip.open(FEAT_CACHE_182D, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            cache_182d[rec["k"]] = rec["v"]
    log(f"  182d cache: {len(cache_182d)} entries")

    # Load 9C + 9F pools
    log("\n=== 2. Loading pools ===")
    pools_9c = json.load(open(POOL_9C))["pools"]
    pools_9f = json.load(open(POOL_9F))["new_pool"]
    all_queries = []
    for asin, qs in pools_9c.items():
        for q in qs:
            all_queries.append({
                "asin": asin, "source": "9c",
                "family": q.get("family", "?"),
                "query": q["query"],
            })
    for asin, qs in pools_9f.items():
        for q in qs:
            all_queries.append({
                "asin": asin, "source": "9f",
                "profile": q.get("profile", "?"),
                "query": q["query"],
            })
    log(f"  total queries: {len(all_queries)}")

    # Dedupe by query text
    seen = set()
    unique_queries = []
    for q in all_queries:
        k = feat_key(q["query"])
        if k in seen:
            continue
        seen.add(k)
        unique_queries.append(q)
    log(f"  unique queries: {len(unique_queries)}")

    # Check how many have 182d features
    have_182d = sum(1 for q in unique_queries if feat_key(q["query"]) in cache_182d)
    log(f"  with 182d cache: {have_182d}/{len(unique_queries)}")

    # spaCy parse all
    log("\n=== 3. spaCy parsing + Group B + Group C extraction ===")
    sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
    sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/syntactic_analysis")
    import spacy
    from main import per_sentence_features_v2

    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    t0 = time.time()
    texts = [q["query"] for q in unique_queries]
    docs = list(nlp.pipe(texts, batch_size=128))
    log(f"  parsed {len(docs)} docs in {time.time() - t0:.1f}s")

    # Build enriched features for each
    enriched_records = []
    n_with_182d = 0
    for q, doc in zip(unique_queries, docs):
        k = feat_key(q["query"])
        try:
            clause_feats = extract_clause_dependency(doc)
        except Exception as e:
            log(f"  clause/dep failed for {q['query'][:50]}: {e}")
            clause_feats = {}
        try:
            conn_feats = extract_connective(q["query"])
        except Exception as e:
            log(f"  connective failed for {q['query'][:50]}: {e}")
            conn_feats = {}

        record = {
            "k": k,
            "asin": q["asin"],
            "source": q["source"],
            "query": q["query"],
            "182d": cache_182d.get(k),
            "clause": clause_feats,
            "connective": conn_feats,
        }
        if q["source"] == "9c":
            record["family"] = q.get("family", "?")
        else:
            record["profile"] = q.get("profile", "?")
        if record["182d"]:
            n_with_182d += 1
        enriched_records.append(record)
    log(f"  with 182d: {n_with_182d}/{len(enriched_records)}")

    # Save
    with gzip.open(FEAT_ENRICHED, "wt", encoding="utf-8") as f:
        for rec in enriched_records:
            f.write(json.dumps(rec) + "\n")
    log(f"  wrote → {FEAT_ENRICHED}")

    # Summary stats
    log("\n=== 4. Summary ===")
    log(f"  records: {len(enriched_records)}")
    log(f"  clause features: {len(enriched_records[0]['clause'])}")
    log(f"  connective features: {len(enriched_records[0]['connective'])}")
    log(f"  with 182d: {n_with_182d}")
    if enriched_records[0]["clause"]:
        log(f"  sample clause features: {list(enriched_records[0]['clause'].keys())[:10]}")
    if enriched_records[0]["connective"]:
        log(f"  sample connective features: {list(enriched_records[0]['connective'].keys())[:10]}")

    log("\n=== Stage 9G-1 complete ===")


if __name__ == "__main__":
    main()