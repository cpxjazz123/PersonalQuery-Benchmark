"""Cache path/loader overrides used by the 14-stage eval driver.

The 08-stage `08_fast_fullscale_eval_<Cat>.py` hardcodes cache path
construction in 5 places:

  - L396 `get_query_cache_path`
  - L416 `load_query_cache`       (uses get_query_cache_path)
  - L434 `load_bm25_query_cache`  (self-paths the subdirectory)
  - L454 `load_result_query_cache` (self-paths the subdirectory)
  - L1455 `load_splade_query_cache` (self-paths the subdirectory)

This module returns a dict of 8 functions (those 5 + `query_cache_exists`,
`list_retrievers_with_query_cache`, `load_colbertv2_query_cache`) that
the driver `setattr`s onto the importlib-loaded 08 eval module. The
08-stage module's own global references resolve to the patched versions
because Python looks them up in the module's `__dict__` at call time.

Path convention: `<query_cache_base_dir>/<cache_subdir>/<retriever>__<cache_subdir>_cache.pkl`

`cache_subdir` is parameterised; default `template_query` (clean), pass
`template_noisy_query` for the noisy pipeline.
"""
from __future__ import annotations

import os
import pickle
from typing import Callable, Dict, List, Optional


def _file_template_for(cache_subdir: str) -> str:
    return f"{{retriever}}__{cache_subdir}_cache.pkl"


def _build_path(query_cache_base_dir: str, cache_subdir: str, retriever_name: str) -> str:
    return os.path.join(
        query_cache_base_dir,
        cache_subdir,
        _file_template_for(cache_subdir).format(retriever=retriever_name),
    )


def make_overrides(
    query_cache_base_dir: str,
    retrievers: List[str],
    cache_subdir: str = "template_query",
) -> Dict[str, Callable]:
    """Return a dict of 8 functions whose values replace the 08 module names."""
    if not retrievers:
        raise ValueError("retrievers list must be non-empty")
    if not cache_subdir:
        raise ValueError("cache_subdir must be a non-empty string")

    def get_query_cache_path(retriever_name: str, query_type: str = "template", query_category: str = "template") -> str:
        return _build_path(query_cache_base_dir, cache_subdir, retriever_name)

    def query_cache_exists(retriever_name: str, query_type: str = "template", query_category: str = "template") -> bool:
        return os.path.exists(get_query_cache_path(retriever_name, query_type, query_category))

    def list_retrievers_with_query_cache(query_type: str = "template", query_category: str = "template") -> List[str]:
        return [r for r in retrievers if query_cache_exists(r, query_type, query_category)]

    def _load_pkl(path: str) -> Dict:
        with open(path, "rb") as f:
            cache = pickle.load(f)
        if not isinstance(cache, dict):
            raise TypeError(f"query cache must be dict, got {type(cache).__name__}: {path}")
        return cache

    def load_query_cache(retriever_name: str, query_type: str = "template", query_category: str = "template") -> Dict:
        cache_path = get_query_cache_path(retriever_name, query_type, query_category)
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"{retriever_name} query cache not found: {cache_path}")
        return _load_pkl(cache_path)

    def load_bm25_query_cache(query_type: str = "template", query_category: str = "template") -> Dict:
        cache_path = get_query_cache_path("bm25", query_type, query_category)
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"BM25 query result cache not found: {cache_path}")
        return _load_pkl(cache_path)

    def load_result_query_cache(retriever_name: str, query_type: str = "template", query_category: str = "template") -> Dict:
        cache_path = get_query_cache_path(retriever_name, query_type, query_category)
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"{retriever_name} query result cache not found: {cache_path}")
        return _load_pkl(cache_path)

    def load_splade_query_cache(query_type: str = "template", query_category: str = "template") -> Dict:
        cache_path = get_query_cache_path("splade", query_type, query_category)
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"SPLADE query cache not found: {cache_path}")
        return _load_pkl(cache_path)

    def load_colbertv2_query_cache(query_type: str = "template", query_category: str = "template") -> Dict:
        cache = load_query_cache("colbertv2", query_type, query_category)
        if not isinstance(cache, dict):
            raise TypeError(f"ColBERTv2 query cache must be dict, got {type(cache).__name__}")
        return cache

    return {
        "get_query_cache_path": get_query_cache_path,
        "query_cache_exists": query_cache_exists,
        "list_retrievers_with_query_cache": list_retrievers_with_query_cache,
        "load_query_cache": load_query_cache,
        "load_bm25_query_cache": load_bm25_query_cache,
        "load_result_query_cache": load_result_query_cache,
        "load_splade_query_cache": load_splade_query_cache,
        "load_colbertv2_query_cache": load_colbertv2_query_cache,
    }


def apply_overrides_to_module(
    eval_module,
    query_cache_base_dir: str,
    retrievers: List[str],
    cache_subdir: str = "template_query",
) -> List[str]:
    """Patch 8 names on `eval_module` in place. Returns the list of names patched."""
    overrides = make_overrides(query_cache_base_dir, retrievers, cache_subdir=cache_subdir)
    for name, fn in overrides.items():
        setattr(eval_module, name, fn)
    return list(overrides.keys())
