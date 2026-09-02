"""
Parallel batch stylometric feature extraction using spaCy's nlp.pipe.
Replaces per-sentence spacy loading with batched multi-process pipeline.

Usage:
    from common.spacy_features_parallel import extract_styles_batch
    features = extract_styles_batch(texts)  # list of np.array or None

Cache:
    Results are cached to {SCRATCH}/spacy_features_cache.npz
    Cache keyed by hash of first 200 chars of each text.
    To clear cache: rm {SCRATCH}/spacy_features_cache.npz
"""
import numpy as np
import spacy
import hashlib
import os
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.syntactic_features import per_sentence_features_v2

SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu")
CACHE_FILE = SCRATCH / "spacy_features_cache.npz"
FEATURE_DIM = 291

# Load spaCy model once (lazy)
_nlp = None

# In-memory cache: key (hash) -> np.array (FEATURE_DIM,) or None
_cache: Dict[str, Optional[np.ndarray]] = {}
_cache_loaded = False


def _text_hash(text: str) -> str:
    """Deterministic hash key for a text (first 200 chars)."""
    return hashlib.md5(text[:200].encode()).hexdigest()


def _load_cache() -> Dict[str, Optional[np.ndarray]]:
    """Load cache from disk npz."""
    if not CACHE_FILE.exists():
        return {}
    try:
        data = np.load(CACHE_FILE, allow_pickle=True)
        keys = data['keys']
        vals = data['vals']
        result = {}
        for k, v in zip(keys, vals):
            if v.dtype == object:
                result[str(k)] = None
            elif np.all(np.isnan(v)):
                result[str(k)] = None
            else:
                result[str(k)] = v
        return result
    except Exception:
        return {}


def _save_cache(cache: Dict[str, Optional[np.ndarray]]):
    """Persist cache to disk as npz."""
    keys = []
    vals = []
    null_marker = np.array([np.nan] * FEATURE_DIM, dtype=np.float32)
    for k, v in cache.items():
        keys.append(k)
        if v is None:
            vals.append(null_marker)
        else:
            vals.append(v)
    np.savez_compressed(CACHE_FILE, keys=np.array(keys), vals=vals)


def _init_cache():
    """Load cache from disk into memory (once per session)."""
    global _cache, _cache_loaded
    if _cache_loaded:
        return
    _cache = _load_cache()
    _cache_loaded = True


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def extract_style_from_doc(doc) -> Optional[np.ndarray]:
    """
    Extract numeric stylometric features from a spaCy Doc.
    Returns 291d float32 array or None if extraction fails.
    """
    feats = per_sentence_features_v2(doc)
    if feats is None:
        return None
    numeric = []
    for v in feats.values():
        if isinstance(v, (int, float, np.integer, np.floating)):
            numeric.append(float(v))
        else:
            numeric.append(0.0)
    return np.array(numeric, dtype=np.float32)


def extract_styles_batch(texts: List[str], batch_size: int = 200,
                         disable: List[str] = None,
                         save_cache: bool = True) -> List[Optional[np.ndarray]]:
    """
    Batch extract stylometric features from a list of texts.
    Uses spaCy's nlp.pipe for efficient batch processing.
    Results are cached by text hash (first 200 chars) for reuse.

    Args:
        texts: List of input strings (each max 500 chars)
        batch_size: spaCy batch size (default 200)
        disable: List of spaCy pipeline components to disable
        save_cache: Whether to save results to disk cache (default True)

    Returns:
        List of np.ndarray (291d) or None per input text
    """
    _init_cache()

    if disable is None:
        disable = ["tagger", "attribute_ruler", "lemmatizer"]

    # Separate cached vs uncached texts
    hashes = [_text_hash(t) for t in texts]
    results: List[Optional[np.ndarray]] = [None] * len(texts)
    to_process_texts = []
    to_process_indices = []

    for i, (h, t) in enumerate(zip(hashes, texts)):
        if h in _cache:
            results[i] = _cache[h]
        else:
            to_process_texts.append(t)
            to_process_indices.append(i)

    if to_process_texts:
        nlp = _get_nlp()
        new_results = []
        for doc in nlp.pipe([t[:500] for t in to_process_texts],
                             batch_size=batch_size, disable=disable):
            new_results.append(extract_style_from_doc(doc))

        for idx, res in zip(to_process_indices, new_results):
            h = hashes[idx]
            _cache[h] = res
            results[idx] = res

        if save_cache:
            _save_cache(_cache)

    return results


def extract_styles_parallel(texts: List[str], n_process: int = 8,
                            batch_size: int = 200) -> List[Optional[np.ndarray]]:
    """
    Parallel version using spacy's n_process for true multi-process speedup.
    Note: n_process forks spaCy, only use if n_process > 1.

    WARNING: n_process with spaCy uses fork() which can cause issues with
    cached _nlp global. Use extract_styles_batch (serial) for safety,
    or use extract_styles_batch with pre-cached texts.

    Args:
        texts: List of input strings
        n_process: Number of processes (1 = serial, 2+ = parallel)
        batch_size: Batch size per process

    Returns:
        List of np.ndarray (291d) or None per input text
    """
    if n_process <= 1:
        return extract_styles_batch(texts, batch_size=batch_size)

    # Use spacy's built-in parallel processing
    from spacy.language import Language
    nlp = spacy.load("en_core_web_sm")
    results = []
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        results.append(extract_style_from_doc(doc))
    return results


def clear_cache():
    """Manually clear the on-disk cache."""
    global _cache, _cache_loaded
    if CACHE_FILE.exists():
        os.remove(CACHE_FILE)
    _cache = {}
    _cache_loaded = False


def cache_stats() -> Tuple[int, int]:
    """Return (n_cached, n_total_unique) for cache."""
    _init_cache()
    return len(_cache), len(set(_cache.keys()))
