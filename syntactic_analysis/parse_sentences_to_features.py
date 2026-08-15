"""Sentence-level syntactic feature parsing with on-disk cache.

Public API:
  parse_corpus(reviews_iter, cache_path, nlp=None) -> dict[user_id, list[features]]
  load_features_cache(cache_path) -> dict[sent_key, features]
  save_features_cache(cache_path, cache) -> None
  parse_miss_sentences(miss_sents, nlp) -> dict[sent_key, features]
  split_sents(text) -> list[str]
  sent_key(text) -> str

Cache layout: gzip-compressed JSONL.
  line 1:    "#META " + JSON({version, spacy_model, n_entries})
  line 2..N: JSON({k: sent_key, v: features_dict})

Fail-fast: load_features_cache raises RuntimeError if the file is missing the
meta header, version mismatch, or corrupt. Caller decides whether to delete.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import spacy
from spacy.language import Language

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "query" / "soft_prefix"))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

from extract_clause_features_single_query import load_spacy_model  # noqa: E402
from e20_lopo_v2 import per_sentence_features  # noqa: E402

CACHE_META_VERSION = "v1"

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sents(text: str) -> list[str]:
    """Cheap regex-based sentence split. Good enough to drive cache keys."""
    parts = _SENT_SPLIT_RE.split(text.strip())
    return [p for p in parts if p.strip()]


def sent_key(text: str) -> str:
    """SHA1 over normalized (stripped + lowered) sentence text."""
    return hashlib.sha1(text.strip().lower().encode("utf-8")).hexdigest()


def load_features_cache(cache_path: Path) -> dict[str, dict]:
    """Load {sent_key: features_dict}. Returns {} if file missing.
    Raises RuntimeError on version mismatch or corruption (fail-fast)."""
    if not cache_path.exists():
        return {}
    cache: dict[str, dict] = {}
    try:
        with gzip.open(cache_path, "rt", encoding="utf-8") as f:
            header = f.readline()
            if not header.startswith("#META "):
                raise ValueError(f"cache missing meta header: {cache_path}")
            meta = json.loads(header[len("#META "):])
            if meta.get("version") != CACHE_META_VERSION:
                raise ValueError(
                    f"cache version mismatch: file={meta.get('version')} "
                    f"current={CACHE_META_VERSION}; delete {cache_path} to rebuild"
                )
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cache[rec["k"]] = rec["v"]
    except (OSError, ValueError, json.JSONDecodeError) as e:
        raise RuntimeError(
            f"cache {cache_path} corrupted/incompatible: {e!r}. "
            f"Delete the file to force rebuild."
        ) from e
    return cache


def save_features_cache(cache_path: Path, cache: dict[str, dict]) -> None:
    """Atomic write (tmp + rename). Always rewrites the full file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        meta = {"version": CACHE_META_VERSION,
                "spacy_model": "en_core_web_sm",
                "n_entries": len(cache)}
        f.write("#META " + json.dumps(meta) + "\n")
        for k, v in cache.items():
            f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")
    tmp.replace(cache_path)


def parse_miss_sentences(miss_sents: list[str],
                         nlp: Language,
                         batch_size: int = 256,
                         log_every: int = 5000,
                         t0: float | None = None) -> dict[str, dict]:
    """Run spaCy over miss sentences; return {sent_key: features}.
    Each input sentence maps to exactly one entry (or is dropped if feature
    extraction yields None)."""
    out: dict[str, dict] = {}
    if not miss_sents:
        return out
    t0 = t0 if t0 is not None else time.time()
    for i in range(0, len(miss_sents), batch_size):
        batch = miss_sents[i:i + batch_size]
        for doc in nlp.pipe(batch, batch_size=batch_size):
            for sent in doc.sents:
                sf = per_sentence_features(sent)
                if sf is not None:
                    out[sent_key(sent.text)] = sf
        n_done = min(i + batch_size, len(miss_sents))
        if n_done % log_every < batch_size:
            print(f"    parsed {n_done}/{len(miss_sents)} "
                  f"(t={time.time() - t0:.1f}s)", flush=True)
    return out


def parse_corpus(reviews_iter: Iterable[tuple[str, str]],
                 cache_path: Path,
                 nlp: Language | None = None,
                 batch_size: int = 256,
                 log_prefix: str = "") -> dict[str, list[dict]]:
    """Parse (user_id, review_text) corpus, using cache.

    Returns {user_id: [features_dict, ...]} aligned to per-user sentence list.
    Side effect: writes cache_path on completion.
    """
    nlp = nlp or load_spacy_model()
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)

    sent_cache = load_features_cache(cache_path)
    print(f"{log_prefix}cache loaded: {len(sent_cache)} sentences from "
          f"{cache_path.name}", flush=True)

    # Collect (user, sent_text) pairs
    sent_pairs: list[tuple[str, str]] = []
    for u, t in reviews_iter:
        for s in split_sents(t):
            sent_pairs.append((u, s))
    n_unique = len({sent_key(s) for _, s in sent_pairs})
    print(f"{log_prefix}total sentences: {len(sent_pairs)} "
          f"(unique: {n_unique})", flush=True)

    # Lookup cache; collect unique miss
    miss_sents: list[str] = []
    miss_keys: set[str] = set()
    for _, s in sent_pairs:
        k = sent_key(s)
        if k not in sent_cache and k not in miss_keys:
            miss_keys.add(k)
            miss_sents.append(s)
    n_hit = len(sent_pairs) - len(miss_sents)
    print(f"{log_prefix}cache hits: {n_hit}/{len(sent_pairs)}, "
          f"to parse: {len(miss_sents)} unique", flush=True)

    t0 = time.time()
    if miss_sents:
        new_feats = parse_miss_sentences(miss_sents, nlp,
                                         batch_size=batch_size, t0=t0)
        sent_cache.update(new_feats)

    save_features_cache(cache_path, sent_cache)
    print(f"{log_prefix}cache saved: {len(sent_cache)} entries", flush=True)

    # Assemble per-user sentence lists from cache
    user_sents: dict[str, list[dict]] = {}
    for u, s in sent_pairs:
        sf = sent_cache.get(sent_key(s))
        if sf is not None:
            user_sents.setdefault(u, []).append(sf)
    return user_sents


__all__ = [
    "CACHE_META_VERSION",
    "split_sents",
    "sent_key",
    "load_features_cache",
    "save_features_cache",
    "parse_miss_sentences",
    "parse_corpus",
]