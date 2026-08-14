"""E14 — Per-user statistic style vector (comments only, no review text).

The user condition vector is built from SYNTACTIC STATISTICS of the user's
review sentences: 20-dim clause-feature means + 10-dim opener histogram = 30
dimensions. Unlike the VADES user_mu (which only covers the 204 legacy
candidate users), this covers every user with reviews (all 900 multi-product
dataset users). Statistics only — no review sentence text is stored or used.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "10_complexity_analysis" / "common"))

from extract_clause_features_single_query import (  # noqa: E402
    load_spacy_model,
    extract_clause_features_from_doc,
)
from opener_stats import (  # noqa: E402
    OPENER_CLASSES,
    opener_class_of,
)

FEATURES20 = [
    "max_dependency_depth", "mean_dependency_depth", "dependency_tree_height",
    "depth_variance", "acl_count", "relcl_count", "ccomp_count", "xcomp_count",
    "advcl_count", "clause_nesting_depth", "mean_dependency_distance",
    "max_dependency_distance", "long_dependency_ratio", "amod_count",
    "advmod_count", "nmod_count", "compound_count", "modifier_density",
    "coordination_count", "max_branching_factor",
]
STAT_DIM = len(FEATURES20) + len(OPENER_CLASSES)  # 30


def load_user_reviews(category: str, user_id: str) -> List[str]:
    p = REPO_ROOT / "result" / "personal_query" / "01_preference_extraction" / category / "stage1_filtered_users_reviews.json"
    data = json.load(open(p))
    texts: List[str] = []
    for u in data["users"]:
        if u["user_id"] != user_id:
            continue
        for r in u.get("results", []):
            for txt in (r.get("target_reviews") or []):
                if txt and isinstance(txt, str) and txt.strip():
                    texts.append(txt[:1000])
        break
    return texts


CACHE_PATH = REPO_ROOT / "result" / "personal_query" / "e14_multiproduct" / "user_stat_vectors.json"


def build_user_stat_vectors(category: str, user_ids: List[str], max_sentences: int = 12,
                            use_cache: bool = True) -> Dict[str, np.ndarray]:
    """Per-user 30-dim statistic vector: [20 feature means, 10 opener hist].

    Cached to result/personal_query/e14_multiproduct/user_stat_vectors.json;
    subsequent calls load the cache instead of re-parsing (spaCy extraction
    is ~3 min for 900 users). Cache is keyed by nothing but user_id; delete
    the file to recompute.
    """
    if use_cache and CACHE_PATH.exists():
        import json as _json
        cached = _json.load(open(CACHE_PATH))
        if cached.get("dim") == STAT_DIM:
            got = {u: np.asarray(v, dtype=np.float32) for u, v in cached["vectors"].items()}
            missing = [u for u in user_ids if u not in got]
            if not missing:
                return got
    nlp = load_spacy_model()
    # single read of the review file, indexed by user
    p = REPO_ROOT / "result" / "personal_query" / "01_preference_extraction" / category / "stage1_filtered_users_reviews.json"
    data = json.load(open(p))
    target = set(user_ids)
    texts_by_user: Dict[str, List[str]] = defaultdict(list)
    for u in data["users"]:
        if u["user_id"] not in target:
            continue
        for r in u.get("results", []):
            for txt in (r.get("target_reviews") or []):
                if txt and isinstance(txt, str) and txt.strip():
                    texts_by_user[u["user_id"]].append(txt[:1000])
    del data
    # single nlp.pipe pass over ALL paragraphs (one pipe call, no per-user
    # overhead); paragraphs split into sentences; features per sentence via
    # Span.as_doc().
    flat: List[tuple] = []
    for uid in user_ids:
        for t in texts_by_user.get(uid, [])[: max_sentences * 3]:
            flat.append((uid, t))
    out: Dict[str, np.ndarray] = {}
    counters: Dict[str, int] = defaultdict(int)
    buffers: Dict[str, list] = defaultdict(list)
    opener_buf: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(OPENER_CLASSES), dtype=np.float32))
    n_open_buf: Dict[str, int] = defaultdict(int)
    for (uid, t), doc in zip(flat, nlp.pipe([t for _, t in flat], batch_size=256)):
        if counters[uid] >= max_sentences:
            continue
        for sent in doc.sents:
            if counters[uid] >= max_sentences:
                break
            toks = [tok for tok in sent if not tok.is_punct and not tok.is_space]
            if len(toks) < 3:
                continue
            try:
                extracted = extract_clause_features_from_doc(sent, sent.text)
                buffers[uid].append(np.asarray([float(extracted.get(k, 0.0)) for k in FEATURES20], dtype=np.float32))
            except Exception:
                continue
            opener_buf[uid][OPENER_CLASSES.index(opener_class_of(toks[0].text))] += 1
            n_open_buf[uid] += 1
            counters[uid] += 1
    for uid in user_ids:
        if not buffers[uid]:
            continue
        fmean = np.mean(buffers[uid], axis=0)
        n = float(np.linalg.norm(fmean))
        if n > 1e-12:
            fmean = fmean / n
        ohist = opener_buf[uid]
        if n_open_buf[uid] > 0:
            ohist = ohist / n_open_buf[uid]
        out[uid] = np.concatenate([fmean, ohist]).astype(np.float32)
    if use_cache:
        import json as _json
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _json.dump({"n_users": len(out), "dim": STAT_DIM,
                    "vectors": {u: v.tolist() for u, v in out.items()}},
                   open(CACHE_PATH, "w"))
    return out
