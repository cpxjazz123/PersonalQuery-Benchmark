"""E21 v9: 235-dim syntactic features (expansion of v1's 32 dim).

Categories (target ~235 dim total, all syntactic — no char n-grams / function
words):
  A. POS unigram + bigram + trigram  (~67 dim)
  B. Dependency relation unigram + bigram  (~55 dim)
  C. Main clause structure patterns  (~9 dim)
  D. Clause co-occurrence + nesting depth  (~22 dim)
  E. Sentence opener/closer POS  (~25 dim)
  F. Dep distance / depth distribution buckets  (~15 dim)
  G. Punctuation syntactic patterns  (~15 dim)

Schema version 2 (vs v1's 32-dim). Cache file CACHE_META_VERSION must be
bumped to "v2" so the old v1 cache is rejected (fail-fast).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# A. POS n-gram vocabulary (fixed for stability; no corpus scan needed)
# ---------------------------------------------------------------------------
# Universal POS tagset (spaCy)
POS_TAGS = (
    "NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "CONJ",
    "AUX", "NUM", "INTJ", "PART", "PUNCT", "X", "SYM", "CCONJ", "SCONJ",
)
# Curated POS bigrams (top common English patterns)
POS_BIGRAMS = (
    ("DET", "NOUN"), ("PRON", "VERB"), ("AUX", "VERB"), ("VERB", "DET"),
    ("VERB", "NOUN"), ("ADJ", "NOUN"), ("NOUN", "VERB"), ("ADV", "VERB"),
    ("ADP", "DET"), ("ADP", "NOUN"), ("PRON", "AUX"), ("DET", "ADJ"),
    ("NOUN", "ADP"), ("VERB", "ADP"), ("VERB", "PRON"), ("ADV", "ADJ"),
    ("ADJ", "ADP"), ("NOUN", "CCONJ"), ("VERB", "CCONJ"), ("NOUN", "SCONJ"),
    ("VERB", "SCONJ"), ("AUX", "ADJ"), ("AUX", "NOUN"), ("PRON", "VERB"),
    ("DET", "NOUN"), ("ADV", "ADJ"), ("AUX", "PART"), ("PART", "VERB"),
    ("NUM", "NOUN"), ("PRON", "ADP"),
)
# Curated POS trigrams (English syntactic templates)
POS_TRIGRAMS = (
    ("DET", "NOUN", "VERB"), ("PRON", "AUX", "VERB"), ("PRON", "AUX", "ADJ"),
    ("AUX", "VERB", "DET"), ("AUX", "VERB", "NOUN"), ("AUX", "VERB", "ADV"),
    ("VERB", "DET", "NOUN"), ("ADP", "DET", "NOUN"), ("VERB", "ADP", "DET"),
    ("ADP", "DET", "ADJ"), ("DET", "ADJ", "NOUN"), ("VERB", "PRON", "AUX"),
    ("AUX", "ADJ", "ADP"), ("ADV", "VERB", "DET"), ("AUX", "VERB", "PRON"),
    ("VERB", "CCONJ", "VERB"), ("NOUN", "CCONJ", "NOUN"), ("VERB", "SCONJ", "VERB"),
    ("AUX", "PART", "VERB"), ("ADV", "AUX", "VERB"),
)

# ---------------------------------------------------------------------------
# B. Dependency relation vocabulary
# ---------------------------------------------------------------------------
# Core Universal Dependencies English relations
DEP_RELS = (
    "nsubj", "nsubjpass", "obj", "iobj", "csubj", "csubjpass",
    "obl", "vocative", "expl", "dislocated", "advcl", "advmod",
    "appos", "acl", "relcl", "det", "clf", "case", "amod",
    "nmod", "nummod", "compound", "flat", "fixed", "conj", "cc",
    "list", "parataxis", "root", "dep", "xcomp", "ccomp",
    "aux", "auxpass", "cop", "mark", "discourse", "punct",
)
DEP_BIGRAMS = (
    ("nsubj", "verb"), ("nsubj", "aux"), ("obj", "dobj"), ("obj", "iobj"),
    ("advmod", "verb"), ("amod", "noun"), ("det", "noun"), ("case", "noun"),
    ("nmod", "noun"), ("compound", "noun"), ("aux", "verb"), ("aux", "adj"),
    ("mark", "verb"), ("advcl", "verb"), ("ccomp", "verb"), ("conj", "verb"),
)

# ---------------------------------------------------------------------------
# C. Main clause structure patterns (based on root + nsubj + obj relations)
# ---------------------------------------------------------------------------
MAIN_CLAUSE_PATTERNS = (
    "SVO", "SVC", "SV", "SVA", "SVOA", "SVOC", "SVOO", "SVO_IOBJ",
    "EXISTS",
)

# ---------------------------------------------------------------------------
# D. Clause co-occurrence pairs
# ---------------------------------------------------------------------------
CLAUSE_DEPS = ("acl", "advcl", "ccomp", "xcomp", "relcl")
CLAUSE_PAIRS = [
    tuple(sorted((a, b))) for i, a in enumerate(CLAUSE_DEPS)
    for b in CLAUSE_DEPS[i + 1:]
]
# 10 unique sorted pairs

# ---------------------------------------------------------------------------
# G. Punctuation tags
# ---------------------------------------------------------------------------
PUNCT_TAGS = (",", ".", ";", ":", "!", "?", "-", "\"", "'", "(", ")")
PUNCT_BIGRAMS = (
    (",", "and"), (",", "but"), (",", "or"), (";", "and"), (".", "but"),
    (",", "so"), ("!", "!"), ("?", "?"), (".", "."), (",", ","),
)

# ---------------------------------------------------------------------------
# Per-sentence feature extraction
# ---------------------------------------------------------------------------
def _tokens_no_space(sent) -> list:
    return [t for t in sent if not t.is_space]


def per_sentence_features_v2(sent) -> dict | None:
    """Extract ~all per-sentence primitives needed for 235-dim features.

    Returns a dict; user-level aggregation picks the keys for ALL_FEATS_V2.
    Returns None if sentence has < 3 non-space tokens.
    """
    toks = _tokens_no_space(sent)
    n_tok = len(toks)
    if n_tok < 3:
        return None

    feats: dict[str, Any] = {
        "n_tok": n_tok,
        "n_clause": sum(1 for t in sent if t.dep_ in CLAUSE_DEPS),
        "n_coord": sum(1 for t in sent if t.dep_ in ("conj", "cc")),
        "n_mod": sum(1 for t in sent if t.dep_ in (
            "amod", "advmod", "nmod", "appos", "nummod", "poss", "det")),
        "mean_dist": 0.0,
        "max_depth": 0,
        "depth_var": 0.0,
        "opener": toks[0].pos_ if toks else "X",
        "stype": "complex" if sum(1 for t in sent if t.dep_ in CLAUSE_DEPS) > 0
                  else ("conjunctive" if sum(1 for t in sent if t.dep_ in ("conj", "cc")) > 0
                        else "simple"),
        "has_passive": any(t.dep_ in ("nsubjpass", "auxpass") for t in sent),
        "is_interrog": sent.text.rstrip().endswith("?"),
        "has_cond": any(m in sent.text.lower() for m in
                        (" if ", " when ", " unless ", " whenever ")),
        # clause type counts (used for clause_rate)
        "acl": sum(1 for t in sent if t.dep_ == "acl"),
        "advcl": sum(1 for t in sent if t.dep_ == "advcl"),
        "ccomp": sum(1 for t in sent if t.dep_ == "ccomp"),
        "xcomp": sum(1 for t in sent if t.dep_ == "xcomp"),
        "relcl": sum(1 for t in sent if t.dep_ == "relcl"),
    }

    # A. POS unigram counts
    pos_seq = [t.pos_ for t in toks]
    pos_counts = Counter(pos_seq)
    for p in POS_TAGS:
        feats[f"pos_{p}"] = pos_counts.get(p, 0)
    # A. POS bigram counts (within tokens_no_space)
    pos_bigram_counts = Counter()
    for i in range(len(pos_seq) - 1):
        pos_bigram_counts[(pos_seq[i], pos_seq[i + 1])] += 1
    for bg in POS_BIGRAMS:
        feats[f"posbg_{bg[0]}_{bg[1]}"] = pos_bigram_counts.get(bg, 0)
    # A. POS trigram counts
    pos_trigram_counts = Counter()
    for i in range(len(pos_seq) - 2):
        pos_trigram_counts[(pos_seq[i], pos_seq[i + 1], pos_seq[i + 2])] += 1
    for tg in POS_TRIGRAMS:
        feats[f"postg_{tg[0]}_{tg[1]}_{tg[2]}"] = pos_trigram_counts.get(tg, 0)

    # B. Dep rel unigram counts
    dep_counts = Counter(t.dep_ for t in sent)
    for d in DEP_RELS:
        feats[f"dep_{d}"] = dep_counts.get(d, 0)
    # B. Dep rel bigram (token i's dep + token i+1's dep)
    dep_seq = [t.dep_ for t in sent if t.dep_ != "punct"]
    dep_bigram_counts = Counter()
    for i in range(len(dep_seq) - 1):
        dep_bigram_counts[(dep_seq[i], dep_seq[i + 1])] += 1
    for bg in DEP_BIGRAMS:
        feats[f"depbg_{bg[0]}_{bg[1]}"] = dep_bigram_counts.get(bg, 0)

    # C. Main clause structure: find root + children
    root = next((t for t in sent if t.dep_ == "root"), None)
    if root is None:
        feats["main_SVO"] = feats["main_SVC"] = feats["main_SV"] = 0
        feats["main_SVA"] = feats["main_SVOA"] = feats["main_SVOC"] = 0
        feats["main_SVOO"] = feats["main_SVO_IOBJ"] = feats["main_EXISTS"] = 0
    else:
        has_nsubj = any(c.dep_ in ("nsubj", "nsubjpass") for c in root.children)
        has_obj = any(c.dep_ in ("obj", "iobj") for c in root.children)
        has_attr = any(c.dep_ in ("attr", "acomp", "oprd") for c in root.children)
        has_obl = any(c.dep_ == "obl" for c in root.children)
        n_objs = sum(1 for c in root.children if c.dep_ in ("obj", "iobj"))
        feats["main_SVO"] = int(has_nsubj and n_objs >= 1 and not has_obl)
        feats["main_SVC"] = int(has_nsubj and has_attr)
        feats["main_SV"] = int(has_nsubj and not has_obj and not has_attr)
        feats["main_SVA"] = int(has_nsubj and has_obl and not has_obj)
        feats["main_SVOA"] = int(has_nsubj and n_objs >= 1 and has_obl)
        feats["main_SVOC"] = int(has_nsubj and has_attr and n_objs >= 1)
        feats["main_SVOO"] = int(has_nsubj and n_objs >= 2)
        feats["main_SVO_IOBJ"] = int(has_nsubj and any(c.dep_ == "iobj"
                                                       for c in root.children))
        feats["main_EXISTS"] = int(any(t.dep_ in ("expl", "nsubj") and
                                       t.pos_ in ("PRON", "DET") and
                                       t.text.lower() in ("there", "it")
                                       for t in sent))

    # D. Clause co-occurrence (per sent, mark which pairs both > 0)
    sent_clause_deps = set()
    for t in sent:
        if t.dep_ in CLAUSE_DEPS:
            sent_clause_deps.add(t.dep_)
    for pair in CLAUSE_PAIRS:
        feats[f"clpair_{pair[0]}_{pair[1]}"] = int(pair[0] in sent_clause_deps
                                                    and pair[1] in sent_clause_deps)

    # D. Nesting depth: max depth of any clause-typed token
    # Recompute depth via tree
    def height(token, memo):
        if token in memo:
            return memo[token]
        children = [c for c in token.children]
        if not children:
            memo[token] = 1
            return 1
        h = 1 + max(height(c, memo) for c in children)
        memo[token] = h
        return h
    memo = {}
    depths_all = [height(t, memo) for t in sent]
    clause_depths = [height(t, memo) for t in sent if t.dep_ in CLAUSE_DEPS]
    feats["nest_max"] = max(clause_depths) if clause_depths else 0
    feats["nest_mean"] = float(np.mean(clause_depths)) if clause_depths else 0.0
    feats["nest_std"] = float(np.std(clause_depths)) if clause_depths else 0.0
    # depth buckets (clauses)
    for d in (1, 2, 3, 4, 5):
        feats[f"nest_d{d}"] = int(feats["nest_max"] == d)
    feats["nest_ge2"] = int(feats["nest_max"] >= 2)
    feats["nest_ge3"] = int(feats["nest_max"] >= 3)

    # E. Sentence opener / closer POS
    n_t = len(pos_seq)
    for pos in range(min(3, n_t)):
        feats[f"open_p{pos+1}_{pos_seq[pos]}"] = 1
    for pos in range(min(3, n_t)):
        feats[f"close_p{pos+1}_{pos_seq[-(pos+1)]}"] = 1

    # F. Dep distance distribution
    dists = [abs(t.head.i - t.i) for t in sent if t.head.i != t.i]
    feats["mean_dist"] = float(np.mean(dists)) if dists else 0.0
    feats["max_depth"] = max(depths_all) if depths_all else 0
    feats["depth_var"] = float(np.var(depths_all)) if depths_all else 0.0
    # distance buckets
    for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 5), (5, 100)]:
        bucket = sum(1 for d in dists if lo <= d < hi)
        feats[f"dist_{lo}_{hi}"] = bucket
    # depth buckets (max_depth per token)
    for d in (1, 2, 3, 4, 5):
        feats[f"depth_eq{d}"] = sum(1 for x in depths_all if x == d)

    # G. Punctuation patterns
    text = sent.text
    feats["n_punct_total"] = sum(1 for c in text if c in ",.;:!?\"'-()")
    for p in PUNCT_TAGS:
        feats[f"punct_{p}"] = text.count(p)
    # Punct bigrams: token-level
    punct_tokens = [t for t in sent if t.pos_ == "PUNCT" or
                    (t.is_punct and not t.is_space)]
    for pb in PUNCT_BIGRAMS:
        if len(pb[0]) == 1:  # single char bigram (char-level)
            feats[f"punctbg_{pb[0]}_{pb[1]}"] = sum(1 for i in range(len(text) - 1)
                                                    if text[i] == pb[0]
                                                    and text[i + 1] == pb[1])
        else:
            feats[f"punctbg_{pb[0]}_{pb[1]}"] = 0  # skip non-char bigrams

    return feats


# ---------------------------------------------------------------------------
# Build ALL_FEATS_V2 list (registration of all features to aggregate)
# ---------------------------------------------------------------------------
def _build_all_feats_v2() -> list[str]:
    feats: list[str] = []
    # Original v1 32 features (kept for compatibility)
    feats += [
        "clause_rate", "acl_rate", "advcl_rate", "ccomp_rate", "xcomp_rate",
        "relcl_rate", "modifier_density", "coordination_density",
        "mean_dep_distance", "depth_variance", "median_dep_depth",
        "passive_rate", "interrogative_rate", "conditional_rate",
    ]
    feats += [f"opener_{p}" for p in POS_TAGS]
    feats += ["senttype_simple", "senttype_conjunctive", "senttype_complex"]
    # A. POS unigram + bigram + trigram (rate per total_tok)
    for p in POS_TAGS:
        feats.append(f"pos_{p}")
    for bg in POS_BIGRAMS:
        feats.append(f"posbg_{bg[0]}_{bg[1]}")
    for tg in POS_TRIGRAMS:
        feats.append(f"postg_{tg[0]}_{tg[1]}_{tg[2]}")
    # B. Dep rel unigram + bigram (rate per total_tok)
    for d in DEP_RELS:
        feats.append(f"dep_{d}")
    for bg in DEP_BIGRAMS:
        feats.append(f"depbg_{bg[0]}_{bg[1]}")
    # C. Main clause structure
    for p in MAIN_CLAUSE_PATTERNS:
        feats.append(f"main_{p}")
    # D. Clause co-occurrence + nesting
    for pair in CLAUSE_PAIRS:
        feats.append(f"clpair_{pair[0]}_{pair[1]}")
    feats += ["nest_max", "nest_mean", "nest_std",
              "nest_d1", "nest_d2", "nest_d3", "nest_d4", "nest_d5",
              "nest_ge2", "nest_ge3"]
    # E. Opener / closer POS
    for pos in (1, 2, 3):
        for p in POS_TAGS:
            feats.append(f"open_p{pos}_{p}")
            feats.append(f"close_p{pos}_{p}")
    # F. Distance / depth distribution
    feats += ["dist_0_1", "dist_1_2", "dist_2_3", "dist_3_5", "dist_5_100"]
    feats += ["depth_eq1", "depth_eq2", "depth_eq3", "depth_eq4", "depth_eq5"]
    # G. Punctuation
    feats.append("n_punct_total")
    for p in PUNCT_TAGS:
        feats.append(f"punct_{p}")
    for pb in PUNCT_BIGRAMS:
        if len(pb[0]) == 1:
            feats.append(f"punctbg_{pb[0]}_{pb[1]}")
    return feats


ALL_FEATS_V2: list[str] = _build_all_feats_v2()


# ---------------------------------------------------------------------------
# User-level aggregation (compute rate features from per-sent dicts)
# ---------------------------------------------------------------------------
_RATE_KEYS = {
    "clause_rate": ("n_clause", "n_tok", "sum"),
    "acl_rate": ("acl", "n_tok", "sum"),
    "advcl_rate": ("advcl", "n_tok", "sum"),
    "ccomp_rate": ("ccomp", "n_tok", "sum"),
    "xcomp_rate": ("xcomp", "n_tok", "sum"),
    "relcl_rate": ("relcl", "n_tok", "sum"),
    "modifier_density": ("n_mod", "n_tok", "sum"),
    "coordination_density": ("n_coord", "n_tok", "sum"),
    "mean_dep_distance": ("mean_dist", None, "mean"),
    "depth_variance": ("depth_var", None, "mean"),
}
_MEAN_KEYS = {"mean_dep_distance", "depth_variance"}
_RATE_TOK_KEYS = {k for k, v in _RATE_KEYS.items() if v[2] == "sum"}
_FRAC_SENT_KEYS = {"passive_rate", "interrogative_rate", "conditional_rate"}
_FRAC_SENT_FLAG = {"passive_rate": "has_passive",
                   "interrogative_rate": "is_interrog",
                   "conditional_rate": "has_cond"}
# Rate per total_tok (POS/dep/clause co-occ)
_RATE_PER_TOK = (
    [f"pos_{p}" for p in POS_TAGS]
    + [f"posbg_{bg[0]}_{bg[1]}" for bg in POS_BIGRAMS]
    + [f"postg_{tg[0]}_{tg[1]}_{tg[2]}" for tg in POS_TRIGRAMS]
    + [f"dep_{d}" for d in DEP_RELS]
    + [f"depbg_{bg[0]}_{bg[1]}" for bg in DEP_BIGRAMS]
)
# Rate per sentence count (boolean flags / per-sent counts)
_RATE_PER_SENT = (
    [f"main_{p}" for p in MAIN_CLAUSE_PATTERNS]
    + [f"clpair_{a}_{b}" for a, b in CLAUSE_PAIRS]
    + ["nest_d1", "nest_d2", "nest_d3", "nest_d4", "nest_d5",
       "nest_ge2", "nest_ge3",
       "n_punct_total"]
    + [f"punct_{p}" for p in PUNCT_TAGS]
    + [f"punctbg_{p[0]}_{p[1]}" for p in PUNCT_BIGRAMS if len(p[0]) == 1]
)
# Mean over sentences
_MEAN_OVER_SENT = ("nest_max", "nest_mean", "nest_std")
# Opener / closer: rate per total_tok
_OPEN_CLOSE_KEYS = []
for pos in (1, 2, 3):
    for p in POS_TAGS:
        _OPEN_CLOSE_KEYS.append(f"open_p{pos}_{p}")
        _OPEN_CLOSE_KEYS.append(f"close_p{pos}_{p}")
# Distance / depth buckets: rate per total_tok
_DIST_DEPTH_KEYS = (
    [f"dist_{lo}_{hi}" for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 5), (5, 100)]]
    + [f"depth_eq{d}" for d in (1, 2, 3, 4, 5)]
)
# Opener POS (kept from v1)
_OPENER_KEYS = [f"opener_{p}" for p in POS_TAGS]


# Pre-build column index buckets (once at module load) for fast batched aggregation
_NAME_TO_IDX: dict[str, int] = {n: i for i, n in enumerate(ALL_FEATS_V2)}

# Bucket A: rate-per-tok (sum / total_tok) — most common (POS/dep/etc)
_RATE_TOK_IDX: list[int] = sorted(
    _NAME_TO_IDX[n] for n in (
        list(_RATE_KEYS.keys())
        + list(_RATE_PER_TOK)
        + list(_OPEN_CLOSE_KEYS)
        + list(_DIST_DEPTH_KEYS)
    )
)
_RATE_TOK_KEY_LOOKUP: dict[int, str] = {}  # col_idx -> per-sent dict key to sum
for n in _RATE_KEYS:
    _RATE_TOK_KEY_LOOKUP[_NAME_TO_IDX[n]] = _RATE_KEYS[n][0]
for n in _RATE_PER_TOK:
    _RATE_TOK_KEY_LOOKUP[_NAME_TO_IDX[n]] = n  # key == name
for n in _OPEN_CLOSE_KEYS:
    _RATE_TOK_KEY_LOOKUP[_NAME_TO_IDX[n]] = n
for n in _DIST_DEPTH_KEYS:
    _RATE_TOK_KEY_LOOKUP[_NAME_TO_IDX[n]] = n

# Bucket B: rate-per-sent (sum / n_sent)
_RATE_SENT_IDX: list[int] = sorted(
    _NAME_TO_IDX[n] for n in (
        list(_RATE_PER_SENT)
        + [n for n in ALL_FEATS_V2 if n.startswith("senttype_")]
    )
)
_RATE_SENT_KEY_LOOKUP: dict[int, str] = {}
for n in _RATE_PER_SENT:
    _RATE_SENT_KEY_LOOKUP[_NAME_TO_IDX[n]] = n
for n in ALL_FEATS_V2:
    if n.startswith("senttype_"):
        _RATE_SENT_KEY_LOOKUP[_NAME_TO_IDX[n]] = "stype_" + n[len("senttype_"):]

# Bucket C: mean over sentences
_MEAN_SENT_IDX: list[int] = sorted(_NAME_TO_IDX[n] for n in _MEAN_OVER_SENT)
_MEAN_SENT_KEY_LOOKUP: dict[int, str] = {
    _NAME_TO_IDX[n]: n for n in _MEAN_OVER_SENT
}

# Special scalar features
_MEDIAN_DEPTH_IDX: int = _NAME_TO_IDX["median_dep_depth"]
_MAX_DEPTH_IDX: int = -1  # placeholder; per-sent dict stores "max_depth" as int


def user_features_v2(sent_feats: list[dict]) -> np.ndarray | None:
    """Aggregate per-sentence features to a fixed-length user vector (v2).
    Output dim = len(ALL_FEATS_V2).

    Optimized: numpy batched aggregation. ~3-5x faster than per-name if-elif
    dispatch over 318 names.
    """
    if not sent_feats:
        return None
    n_sent = len(sent_feats)
    total_tok = sum(s["n_tok"] for s in sent_feats)
    if total_tok == 0 or n_sent == 0:
        return None

    D = len(ALL_FEATS_V2)
    vec = np.zeros(D, dtype=np.float32)

    # Build [n_sent, D] matrix aligned to ALL_FEATS_V2
    M = np.zeros((n_sent, D), dtype=np.float32)
    for i, sf in enumerate(sent_feats):
        for name, val in sf.items():
            j = _NAME_TO_IDX.get(name)
            if j is not None:
                M[i, j] = val

    # Bucket A: rate-per-tok aggregations
    if _RATE_TOK_IDX:
        M_a = M[:, _RATE_TOK_IDX]  # [n_sent, K_a]
        sums_a = M_a.sum(axis=0)  # [K_a]
        for k, col in enumerate(_RATE_TOK_IDX):
            vec[col] = sums_a[k] / total_tok

    # Bucket B: rate-per-sent aggregations (includes senttype_* via key lookup)
    if _RATE_SENT_IDX:
        M_b = M[:, _RATE_SENT_IDX]
        sums_b = M_b.sum(axis=0)
        for k, col in enumerate(_RATE_SENT_IDX):
            vec[col] = sums_b[k] / n_sent

    # Bucket C: mean over sentences
    for col in _MEAN_SENT_IDX:
        vec[col] = M[:, col].mean()

    # Median dep depth (uses per-sent "max_depth" key)
    if "max_depth" in _NAME_TO_IDX:
        # already in M via key alias mapping; use that column
        pass
    depth_col = _NAME_TO_IDX.get("max_depth")
    if depth_col is not None:
        vec[_MEDIAN_DEPTH_IDX] = float(np.median(M[:, depth_col]))
    else:
        # fallback: gather from sent_feats
        vec[_MEDIAN_DEPTH_IDX] = float(
            np.median([s["max_depth"] for s in sent_feats])
        )

    # Fraction features (boolean flag per sent)
    for name in _FRAC_SENT_KEYS:
        flag = _FRAC_SENT_FLAG[name]
        cnt = sum(1 for s in sent_feats if s.get(flag, False))
        vec[_NAME_TO_IDX[name]] = cnt / n_sent

    # Opener POS fractions
    for name in _OPENER_KEYS:
        tag = name[len("opener_"):]
        cnt = sum(1 for s in sent_feats if s.get("opener") == tag)
        vec[_NAME_TO_IDX[name]] = cnt / n_sent

    return vec


__all__ = [
    "POS_TAGS", "POS_BIGRAMS", "POS_TRIGRAMS",
    "DEP_RELS", "DEP_BIGRAMS", "MAIN_CLAUSE_PATTERNS",
    "CLAUSE_PAIRS", "PUNCT_TAGS", "PUNCT_BIGRAMS",
    "per_sentence_features_v2",
    "user_features_v2",
    "ALL_FEATS_V2",
]


if __name__ == "__main__":
    print(f"ALL_FEATS_V2 dim: {len(ALL_FEATS_V2)}")
    # Smoke test
    import spacy
    nlp = spacy.load("en_core_web_sm")
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)
    text = "I bought this for my baby and she loves it. It works great."
    doc = nlp(text)
    sents = list(doc.sents)
    feats_per_sent = [per_sentence_features_v2(s) for s in sents]
    feats_per_sent = [f for f in feats_per_sent if f is not None]
    user_vec = user_features_v2(feats_per_sent)
    print(f"vec dim: {user_vec.shape}, sum: {user_vec.sum():.2f}")