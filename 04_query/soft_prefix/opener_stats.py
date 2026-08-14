"""E14-P — Opener statistics as part of the user style condition.

The VADES 20-dim syntactic statistics contain NO first-token/opener signal,
so the model collapses to a single opener ("I'm looking for" 81% vs 13.8% in
the data). Fix: augment the user condition vector with a deterministic
OPENER-STATISTICS histogram of the user's review sentences (first-word
lexical class of each review sentence). Still statistics-only — no review
sentence text is used.

opener class histogram (10 dims, normalized):
  pronoun / adverb / determiner / noun / verb / conjunction / preposition /
  proper_noun / numeral / other
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")

OPENER_CLASSES = [
    "PRON", "ADV", "DET", "NOUN", "VERB", "CCONJ", "ADP", "PROPN", "NUM", "OTHER",
]
def opener_class_of(word: str) -> str:
    """Coarse lexical class of a word (deterministic; used for the histogram)."""
    w = word.lower()
    if w in {"i", "we", "you", "they", "he", "she", "it", "my", "me", "our"}:
        return "PRON"
    if w in {"the", "a", "an", "this", "that", "these", "those", "my", "your"}:
        return "DET"
    if w in {"and", "but", "or", "so", "because", "although", "though"}:
        return "CCONJ"
    if w in {"in", "on", "at", "for", "with", "of", "to", "after", "before", "during"}:
        return "ADP"
    if w.endswith("ly"):
        return "ADV"
    return "OTHER"


def review_first_words(category: str, user_id: str) -> List[str]:
    """First content word of each review sentence of the user (statistics
    only; full sentences are discarded immediately)."""
    p = (
        REPO_ROOT / "result" / "personal_query" / "01_preference_extraction"
        / category / "stage1_filtered_users_reviews.json"
    )
    data = json.load(open(p))
    first_words: List[str] = []
    for u in data["users"]:
        if u["user_id"] != user_id:
            continue
        for r in u.get("results", []):
            for txt in (r.get("target_reviews") or []):
                if not txt or not isinstance(txt, str):
                    continue
                toks = txt.split()
                if toks:
                    first_words.append(toks[0])
        break
    return first_words


def opener_stats_vector(category: str, user_id: str) -> np.ndarray:
    """Normalized 10-dim opener-class histogram over the user's review
    sentences. Pure statistics; reproducible."""
    first_words = review_first_words(category, user_id)
    hist = np.zeros(len(OPENER_CLASSES), dtype=np.float32)
    if not first_words:
        hist[OPENER_CLASSES.index("OTHER")] = 1.0
        return hist
    for w in first_words:
        cls = opener_class_of(w)
        hist[OPENER_CLASSES.index(cls)] += 1
    hist /= max(hist.sum(), 1.0)
    return hist


def combined_user_vector(category: str, user_mu: np.ndarray, user_id: str) -> np.ndarray:
    """user_mu (VADES 20-dim, normalized) + opener histogram (10-dim)."""
    mu = np.asarray(user_mu, dtype=np.float32)
    n = float(np.linalg.norm(mu))
    if n > 1e-12:
        mu = mu / n
    opener = opener_stats_vector(category, user_id)
    return np.concatenate([mu, opener]).astype(np.float32)


_OPENER_CACHE: Dict[str, np.ndarray] = {}


def build_opener_stats_for_users(category: str, user_ids) -> Dict[str, np.ndarray]:
    """Batch-build opener histograms for many users (reads the review file
    once). Results cached in-process."""
    global _OPENER_CACHE
    need = [u for u in user_ids if u not in _OPENER_CACHE]
    if not need:
        return {u: _OPENER_CACHE[u] for u in user_ids}
    p = (
        REPO_ROOT / "result" / "personal_query" / "01_preference_extraction"
        / category / "stage1_filtered_users_reviews.json"
    )
    data = json.load(open(p))
    target = set(need)
    for u in data["users"]:
        if u["user_id"] not in target:
            continue
        hist = np.zeros(len(OPENER_CLASSES), dtype=np.float32)
        n = 0
        for r in u.get("results", []):
            for txt in (r.get("target_reviews") or []):
                if not txt or not isinstance(txt, str):
                    continue
                toks = txt.split()
                if toks:
                    hist[OPENER_CLASSES.index(opener_class_of(toks[0]))] += 1
                    n += 1
        if n > 0:
            hist /= n
        else:
            hist[OPENER_CLASSES.index("OTHER")] = 1.0
        _OPENER_CACHE[u["user_id"]] = hist
    return {u: _OPENER_CACHE[u] for u in user_ids}


def get_opener_stats(category: str, user_id: str, cache: Dict[str, np.ndarray]) -> np.ndarray:
    return cache.get(user_id, np.zeros(len(OPENER_CLASSES), dtype=np.float32))
