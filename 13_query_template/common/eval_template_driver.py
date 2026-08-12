"""14-stage evaluation driver.

Loads the 08-stage `08_fast_fullscale_eval_<Cat>.py` module via importlib
into a uniquely-named module slot, patches 8 cache path/loader names to
point at the 14-stage `template_query/` subdirectory, and calls the four
`evaluate_*` functions directly. Writes a single summary JSON whose schema
mirrors 08's `retrieval_expression_style_summary.json` but with 14-stage
identifiers.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE_14_DIR = os.path.dirname(THIS_DIR)
REPO_ROOT = os.path.dirname(STAGE_14_DIR)
STAGE_08_DIR = os.path.join(REPO_ROOT, "08_retrieval")
STAGE_08_UTILS = os.path.join(STAGE_08_DIR, "utils")
STAGE_08_EVAL_FILE = "08_fast_fullscale_eval_{category}.py"

sys.path.insert(0, STAGE_08_DIR)

from .cache_path_overrides import apply_overrides_to_module  # noqa: E402
from .config import get_category_config  # noqa: E402
from .user_data_loader import load_template_user_queries  # noqa: E402


TEMPLATE_QUERY_TYPE = "template"
TEMPLATE_QUERY_CATEGORY = "template"


from common_utils import log  # 统一 log 函数


def load_08_eval_module(category: str) -> Any:
    """importlib-load the 08 eval module into a uniquely-named slot.

    The 08 module imports `from config import ...` and `from revised_query_utils
    import ...`, both of which resolve via sys.path. We pre-add STAGE_08_DIR and
    STAGE_08_UTILS so those relative imports succeed.

    The 08 module's top-level code calls `prepare_colbert_torch_extensions_dir()`
    which requires a real CUDA GPU. To allow smoke-testing the patch layer
    without a GPU, we monkey-patch `torch.cuda.is_available` and
    `torch.cuda.get_device_capability` to return a stub arch tag. The actual
    ColBERTv2 evaluation still requires a real GPU at run time.
    """
    eval_path = os.path.join(STAGE_08_DIR, STAGE_08_EVAL_FILE.format(category=category))
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"08 eval module not found: {eval_path}")

    _install_cuda_stub()

    module_name = f"_eval_08_template_{category}"
    spec = importlib.util.spec_from_file_location(module_name, eval_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to build importlib spec for {eval_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_CUDA_STUB_INSTALLED = False


def _install_cuda_stub() -> None:
    """Make `torch.cuda.is_available` return True (and provide a fake
    `get_device_capability`) so the 08 module's module-level ColBERT
    initialisation can complete without a real GPU.

    Skipped on machines where `torch.cuda.is_available()` is already True.
    """
    global _CUDA_STUB_INSTALLED
    if _CUDA_STUB_INSTALLED:
        return
    try:
        import torch
        if torch.cuda.is_available():
            _CUDA_STUB_INSTALLED = True
            return
    except ImportError:
        return

    import torch
    original_is_available = torch.cuda.is_available
    original_get_capability = getattr(torch.cuda, "get_device_capability", None)

    def stub_is_available() -> bool:
        return True

    def stub_get_capability(device: Any = None) -> Tuple[int, int]:
        return (8, 0)

    torch.cuda.is_available = stub_is_available
    if original_get_capability is not None:
        torch.cuda.get_device_capability = stub_get_capability
    _CUDA_STUB_INSTALLED = True


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k) if isinstance(k, tuple) else k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    if isinstance(obj, tuple):
        return [_sanitize_for_json(item) for item in obj]
    try:
        import numpy as np
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return obj


def _run_evaluators(
    eval_module: Any,
    user_queries: Dict[str, List[Dict[str, Any]]],
    user_to_group: Dict[str, str],
    retrievers: List[str],
    k_values: List[int],
    query_type: str,
    query_category: str,
    cache_subdir: str,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    qt = query_type
    qc = query_category

    dense_set = set(getattr(eval_module, "DENSE_RETRIEVERS", []))
    sparse_set = set(getattr(eval_module, "SPARSE_RETRIEVERS", []))
    colbert_set = set(getattr(eval_module, "COLBERTV2_RETRIEVERS", []))

    available = eval_module.list_retrievers_with_query_cache(qt, qc)
    log(f"  available retrievers with cache in {cache_subdir}/: {available}")

    for retriever_name in retrievers:
        if retriever_name not in available:
            log(f"  skip {retriever_name}: no cache available in {cache_subdir}/")
            continue

        if retriever_name in dense_set:
            fn = eval_module.evaluate_dense_retriever
        elif retriever_name == "bm25":
            fn = eval_module.evaluate_bm25_retriever
        elif retriever_name in colbert_set:
            fn = eval_module.evaluate_cached_result_retriever
        elif retriever_name in sparse_set:
            fn = eval_module.evaluate_splade_retriever
        else:
            log(f"  skip {retriever_name}: no evaluator found")
            continue

        try:
            if retriever_name == "bm25":
                result = fn(user_queries, user_to_group, k_values, word_idf=None, query_type=qt, query_category=qc)
            elif retriever_name in colbert_set:
                result = fn(retriever_name, user_queries, user_to_group, k_values, word_idf=None, query_type=qt, query_category=qc)
            elif retriever_name in sparse_set:
                result = fn(user_queries, user_to_group, k_values, word_idf=None, query_type=qt, query_category=qc)
            else:
                result = fn(retriever_name, user_queries, user_to_group, k_values, word_idf=None, query_type=qt, query_category=qc)
        except Exception as exc:
            log(f"  [ERROR] evaluator for {retriever_name} raised: {exc}")
            raise

        result["query_type"] = qt
        result["query_category"] = qc
        results.append(result)
        log(f"  ✓ {retriever_name} evaluated")
    return results


def run_template_eval(
    category: str,
    k_values: Optional[List[int]] = None,
    retrievers: Optional[List[str]] = None,
    mode: str = "clean",
) -> Dict[str, Any]:
    if mode not in ("clean", "noisy"):
        raise ValueError(f"unknown mode: {mode!r}, expected 'clean' or 'noisy'")
    if k_values is None:
        k_values = [1, 3, 5, 10]

    cat_cfg = get_category_config(category)
    if mode == "noisy":
        query_template_file = cat_cfg["output_query_file_noisy"]
        output_summary_file = cat_cfg["output_summary_file_noisy"]
        cache_subdir = "template_noisy_query"
        qt = TEMPLATE_QUERY_TYPE
        qc = "noisy"
    else:
        query_template_file = cat_cfg["output_query_file"]
        output_summary_file = cat_cfg["output_summary_file"]
        cache_subdir = "template_query"
        qt = TEMPLATE_QUERY_TYPE
        qc = TEMPLATE_QUERY_CATEGORY
    query_cache_base_dir = cat_cfg["query_cache_base_dir"]

    log("=" * 60)
    log(f"14-stage template evaluation - {category} [{mode}]")
    log(f"  mode: {mode}")
    log(f"  query_template_file: {query_template_file}")
    log(f"  output_summary_file: {output_summary_file}")
    log(f"  query_cache_base_dir: {query_cache_base_dir}")
    log(f"  cache_subdir: {cache_subdir}")
    log("=" * 60)

    user_queries, user_to_group, meta = load_template_user_queries(query_template_file)
    log(f"  loaded {meta['user_count']} users / {meta['query_count']} queries from template")
    log(f"  template_id={meta['template_id']} template_version={meta['template_version']}")

    eval_module = load_08_eval_module(category)
    all_retrievers = list(getattr(eval_module, "RETRIEVERS", []))
    if retrievers is None:
        retrievers = all_retrievers
    else:
        unknown = [r for r in retrievers if r not in all_retrievers]
        if unknown:
            raise ValueError(f"unknown retrievers: {unknown}")

    apply_overrides_to_module(eval_module, query_cache_base_dir, retrievers, cache_subdir=cache_subdir)
    log(f"  patched 8 cache functions to {cache_subdir}/ subdir")

    query_type_results = _run_evaluators(
        eval_module,
        user_queries,
        user_to_group,
        retrievers,
        k_values,
        query_type=qt,
        query_category=qc,
        cache_subdir=cache_subdir,
    )

    if not query_type_results:
        cache_dir = os.path.join(query_cache_base_dir, cache_subdir)
        listing = os.listdir(cache_dir) if os.path.isdir(cache_dir) else "MISSING"
        raise ValueError(
            f"No retriever produced results for category={category} mode={mode}. "
            f"Available caches in {cache_dir}: {listing}"
        )

    try:
        title_suffix = "TEMPLATE_NOISY_QUERY" if mode == "noisy" else "TEMPLATE_QUERY"
        eval_module.print_summary_table_wide(query_type_results, f"{category} {title_suffix}")
        eval_module.print_hit10_complexity_table(query_type_results)
    except Exception as exc:
        log(f"  [WARN] summary print failed: {exc}")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "category_name": category,
        "mode": mode,
        "query_source_file": query_template_file,
        "template_id": meta["template_id"],
        "template_version": meta["template_version"],
        "query_types": [qt],
        "query_categories": [qc],
        "k_values": k_values,
        "user_count": meta["user_count"],
        "query_count": meta["query_count"],
        "results": _sanitize_for_json(query_type_results),
    }

    out_path = Path(output_summary_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, out_path)
    log(f"  summary written: {out_path}")

    return summary
