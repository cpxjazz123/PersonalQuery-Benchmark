"""Stage 5 unified multi-retriever on strict alignment (NO rerank).

按用户指令 2026-08-29: 整合所有 retrievers 到单一主 pipeline,删除
cross-encoder rerank。Stage 4 选出的 strict_personalized queries 在 7 个
retrievers 上全量 retrieval,然后算 per-retriever headline + volatility。

7 retrievers:
  1. BM25 (lexical_sparse, bm25s lucene k1=1.5 b=0.75)
  2. SPLADE (learned_sparse, naver/splade-cocondenser-ensembledistil)
  3. MiniLM-L6-v2 (dense_biencoder, 384d)
  4. MPNet-base-v2 (dense_biencoder, 768d)
  5. BGE-base-en-v1.5 (dense_biencoder, 768d)
  6. GTE-base (dense_biencoder, 768d)
  7. ColBERTv2 (late_interaction, 768→128 linear projection)

(Cross-encoder rerank 已删除 — full corpus CE 不实际,Stage 5 默认
BM25 top-100 + cross-encoder 的 BM25-only rerank pipeline 见已弃用版本
`stage8_5_rerank_*`。)

输出:
  scratch2/.../stage8_5_retrieval_per_query.json  (per-query intermediate, 7 retrievers)
  result/syntactic_evaluation/retrieval_summary.json  (per-retriever headline + volatility)
  result/syntactic_evaluation/volatility.json  (canonical volatility, sim09 slice)
"""
from __future__ import annotations

import collections
import gzip
import hashlib
import heapq
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

# 2026-09-23: pq_env 多版本 dist-info 残留（torch ×5, transformers ×4, accelerate ×2, torchvision ×4），
# 实际加载的是 transformers 4.57.6 + accelerate 1.15.0，但 site-packages 同时存在的
# torchvision 0.19.0 与 torch 2.14.0 不兼容，import torchvision 时 `_meta_registrations`
# 注册 fake `torchvision::nms` 抛 RuntimeError，级联导致 transformers.image_utils 加载失败，
# 进而 sentence_transformers / TrainerCallback 全部炸。
# 解决：在 import sentence_transformers 前 stub torchvision.transforms（提供 InterpolationMode
# enum）和 torchvision.transforms.v2.functional（空模块），让 transformers 检测到 torchvision
# "已安装"但不会触发原生 _meta_registrations。同时把 4.57.6 已移除的 AutoProcessor 从
# processing_auto 子模块重新 export 到顶层（sentence_transformers 4.x 仍依赖 `from transformers
# import AutoProcessor`）。
import enum as _enum
import importlib.machinery as _im
import types as _types
_tv = _types.ModuleType("torchvision")
_tv.__path__ = ["/home/wlia0047/ar57_scratch/wenyu/pq_env/lib/python3.10/site-packages/torchvision"]
_tv.__spec__ = _im.ModuleSpec("torchvision", None, is_package=True)
sys.modules["torchvision"] = _tv


class _InterpolationMode(_enum.Enum):
    NEAREST = "nearest"
    NEAREST_EXACT = "nearest-exact"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"


_tv_t = _types.ModuleType("torchvision.transforms")
_tv_t.__spec__ = _im.ModuleSpec("torchvision.transforms", None)
_tv_t.InterpolationMode = _InterpolationMode
sys.modules["torchvision.transforms"] = _tv_t
_tv.transforms = _tv_t
_tv_t_v2 = _types.ModuleType("torchvision.transforms.v2")
_tv_t_v2.__spec__ = _im.ModuleSpec("torchvision.transforms.v2", None)
_tv_t_v2.functional = _types.SimpleNamespace()
sys.modules["torchvision.transforms.v2"] = _tv_t_v2
_tv_t.v2 = _tv_t_v2

# Ensure AutoProcessor is importable from transformers top-level (removed in 4.57.x)
import transformers as _transformers
from transformers.models.auto.processing_auto import AutoProcessor as _AutoProcessor
_transformers.AutoProcessor = _AutoProcessor

# 2026-09-23: 本脚本只读 `result/11_syntactic_evaluation/asin_to_doc.json`。
# asin_to_doc.json 的构建/清洗由 `common/build_asin_to_doc.py`（即将迁移）独立负责，
# 本脚本不再调用 `build_meta_corpus`。`build_meta_corpus` 需要的 4 层清洗逻辑
# （Tier 1/2/3/5）+ cache invalidation 指纹由 build_asin_to_doc.py 维护。
# 注意：_corpus_signature 包含 asin_to_doc 内容 hash（cache invalidation bug 修复），
# 下游 corpus_sig 会随清洗而变化。
from build_asin_to_doc import _corpus_signature, _sig_path_for

# 用户指令 2026-08-30: 支持 SEL_OUT_SUFFIX 让 strict34 cohort 跑独立 cache, 不覆盖 canonical
_SEL_SUFFIX = os.environ.get("SEL_OUT_SUFFIX", "")

# ===========================================================================
# PATHS (inlined from common/syntax_subspace_utils.py 2026-09-06: common/ deleted)
# ===========================================================================
REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
# 用户指令 2026-09-23: data 目录从 REPO_ROOT/data 迁移到 hj82 同名 data 目录.
DATA_DIR = Path("/home/wlia0047/hj82/wenyu/PersoanlQuery/data")
ASIN_TO_DOC_CACHE = REPO_ROOT / "result/11_syntactic_evaluation/asin_to_doc.json"
META_FILE = DATA_DIR / "meta_Baby_Products_2023.jsonl"
SEL_IN = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/08_select_query/selected_queries.json")  # 2026-09-19: canonical = current Stage 8 output
RESULT_DIR = REPO_ROOT / "result/11_syntactic_evaluation"
PER_QUERY_OUT = RESULT_DIR / f"per_query{_SEL_SUFFIX}.json"
SUMMARY_OUT = RESULT_DIR / f"retrieval_summary{_SEL_SUFFIX}.json"
VOLATILITY_OUT = RESULT_DIR / f"volatility{_SEL_SUFFIX}.json"
# 用户指令 2026-09-23: 3 个 category 各自一份 (Baby / Musical / Video_Games),
# main() 改为串行跑 3 个 domain, 产物写到 result/11_syntactic_evaluation/<subdir>/.
CATEGORY_INPUTS = [
    # (category_key, subdir)
    ("Baby",                "baby"),
    ("Musical_Instruments", "musical"),
    ("Video_Games",         "video_games"),
]
EMBED_CACHE_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/multiretrieval_embeds")

# 2026-09-19: Top-100 candidates cache for each retriever (used by Stage 11/12 LLM rerank).
# Each retriever writes <topk_save_dir>/<retr_name>_top100.npz with key 'topk_asins' (n_queries, 100).
TOPK_SAVE_DIR = RESULT_DIR / "top100_cache"
TOPK_SAVE_K = 100

# 硬编码运行配置（Rule 3）；首次运行必须先用最小 smoke 验证端到端链路。
SMOKE = False
N_SMOKE_QUERIES = 0


def log(msg: str) -> None:
    """Timestamped log print (inlined from common/syntax_subspace_utils.py)."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# 2026-09-23: Stage 12 / Stage 14 通过 module-level 调用 `retr_mod.build_meta_corpus()`。
# 提供 read-only 包装：直接读 ASIN_TO_DOC_CACHE（已被 build_asin_to_doc.py 写入），不再重建。
# 删除后会影响 12_typo_evaluation/typo_retrieval_eval.py 与 14_typo_rerank/typo_rerank_eval.py。
def build_meta_corpus(force: bool = False):  # noqa: ARG001
    """Read-only loader for ASIN_TO_DOC_CACHE (replaces former builder).

    2026-09-23: asin_to_doc.json 的构建/清洗已迁移到 `build_asin_to_doc.py`（独立模块）。
    本函数仅在 Stage 12 / 14 通过 module-level 调用时保留——直接读 cache，不重建。
    """
    if not ASIN_TO_DOC_CACHE.exists():
        raise FileNotFoundError(
            f"ASIN_TO_DOC_CACHE missing: {ASIN_TO_DOC_CACHE}. "
            f"Run `python build_asin_to_doc.py` first."
        )
    return json.load(open(ASIN_TO_DOC_CACHE, encoding="utf-8"))

# ===========================================================================
# RETRIEVER REGISTRY
# ===========================================================================
RETRIEVERS = [
    {"name": "bm25", "kind": "sparse_lexical", "hf_id": None, "dim": 0},
    {"name": "splade", "kind": "sparse_learned", "hf_id": "naver/splade-cocondenser-ensembledistil", "dim": 0},
    {"name": "minilm", "kind": "dense", "hf_id": "sentence-transformers/all-MiniLM-L6-v2", "dim": 384},
    {"name": "mpnet", "kind": "dense", "hf_id": "sentence-transformers/all-mpnet-base-v2", "dim": 768},
    {"name": "bge_base_v15", "kind": "dense", "hf_id": "BAAI/bge-base-en-v1.5", "dim": 768},
    {"name": "gte_base", "kind": "dense", "hf_id": "thenlper/gte-base", "dim": 768},
    {"name": "colbertv2", "kind": "late_interaction", "hf_id": "colbert-ir/colbertv2.0", "dim": 128},
]
RETR_NAMES = [r["name"] for r in RETRIEVERS]

# ===========================================================================
# SELECTION SIGNATURE (per-query cache key)
# ===========================================================================
def _compute_selection_signature(selection: dict) -> str:
    """SHA1 over (n_entries + sorted (asin, user_id, query_text)).

    Catches any change to Stage 4 selection: file path, n_entries, or any
    (asin, user_id, query) triple. Stable across re-runs as long as
    Stage 4 selection is unchanged.
    """
    entries = _flatten_selection(selection)
    triples = sorted(
        (e["asin"], e["user_id"], (e.get("selected") or {}).get("query", ""))
        for e in entries
    )
    h = hashlib.sha1()
    h.update(f"n={len(triples)}|".encode())
    for asin, uid, q in triples:
        h.update(f"{asin}|{uid}|{q}\x00".encode())
    return h.hexdigest()[:16]


def _flatten_selection(selection: dict) -> list[dict]:
    """兼容新 schema (selections=[{asin, users:[{uid, query}]}]) 与旧 schema (entries=[...])。

    2026-09-06: 新产物的 selections block 展开为 flat entries, 每条形如
        {"asin": a, "user_id": u, "selected": {"query": q, ...}}
    """
    if "entries" in selection:
        return selection["entries"]
    flat = []
    for blk in selection.get("selections", []):
        a = blk["asin"]
        for u in blk.get("users", []):
            flat.append({
                "asin": a,
                "user_id": u["uid"],
                "selected": {"query": u["query"]},
            })
    return flat


# ===========================================================================
# HELPERS
# ===========================================================================
# _corpus_signature / _sig_path_for 同目录 build_asin_to_doc（2026-09-22，曾外移至 common/ 已撤回）


def _queries_signature(queries: list[str]) -> str:
    """Fast fingerprint over the query list (used to key per-retriever query cache).

    Cheap: SHA1 over joined queries (~MBs/s). Selection signature in main()
    is more robust but is only computed once and threaded down for consistency.
    """
    h = hashlib.sha1()
    h.update(f"n={len(queries)}|".encode())
    for q in queries:
        h.update(f"{q}\x00".encode())
    return h.hexdigest()[:16]


def rr_hit_from_rank(rank: int | None) -> dict:
    if rank is None or rank < 0:
        return {"rank": None, "RR": 0.0, "hit1": 0, "hit5": 0, "hit10": 0}
    return {"rank": rank, "RR": 1.0 / rank,
            "hit1": 1 if rank == 1 else 0,
            "hit5": 1 if rank <= 5 else 0,
            "hit10": 1 if rank <= 10 else 0}


def rank_target_in_sorted(target_idx: int, sorted_docs: np.ndarray, max_k: int) -> int:
    if target_idx < 0:
        return -1
    positions = np.where(sorted_docs == target_idx)[0]
    if len(positions) == 0:
        return max_k + 1
    return int(positions[0]) + 1


# ===========================================================================
# BM25 (lexical sparse)
# ===========================================================================
def bm25_retrieve(queries: list[str], corpus_texts: list[str],
                  target_indices: np.ndarray, bm25_k: int = 20000,
                  save_topk_path: Path | None = None,
                  topk_k: int = TOPK_SAVE_K) -> list[dict]:
    import bm25s
    log("\n=== BM25 (lexical_sparse) ===")

    # ---- Item-document cache: tokenized corpus + built index, keyed by corpus sig ----
    cache_dir = EMBED_CACHE_DIR / "bm25"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = cache_dir / "corpus_tokens.pkl"
    tokens_sig_path = cache_dir / "corpus_tokens.pkl.sig"
    index_dir = cache_dir / "bm25_index"
    index_sig_path = cache_dir / "bm25_index.sig"
    current_sig = _corpus_signature(meta_file=META_FILE)
    n_corpus = len(corpus_texts)

    def _load_sig(p: Path) -> dict | None:
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    corpus_tokens = None
    cached = _load_sig(tokens_sig_path)
    if cached is not None and tokens_path.exists():
        if (cached.get("sig") == current_sig
                and cached.get("n_corpus") == n_corpus):
            t0 = time.time()
            with open(tokens_path, "rb") as f:
                corpus_tokens = pickle.load(f)
            log(f"  ✓ corpus tokens cache hit ({len(corpus_tokens.ids)} docs, "
                f"{time.time() - t0:.1f}s, sig={current_sig})")
        else:
            log(f"  ⚠ tokens cache stale (sig/n_corpus mismatch), rebuilding...")
    if corpus_tokens is None:
        t0 = time.time()
        corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", show_progress=False)
        log(f"  corpus tokenized in {time.time() - t0:.1f}s")
        with open(tokens_path, "wb") as f:
            pickle.dump(corpus_tokens, f, protocol=pickle.HIGHEST_PROTOCOL)
        tokens_sig_path.write_text(json.dumps({"sig": current_sig, "n_corpus": n_corpus}))
        log(f"  cached → {tokens_path} (sig={current_sig}, n_corpus={n_corpus})")

    retriever = None
    cached = _load_sig(index_sig_path)
    if cached is not None and index_dir.exists():
        if cached.get("sig") == current_sig and cached.get("n_corpus") == n_corpus:
            t0 = time.time()
            try:
                retriever = bm25s.BM25.load(str(index_dir), load_corpus=False,
                                             mmap=False, show_progress=False)
                log(f"  ✓ bm25 index cache hit ({time.time() - t0:.1f}s, sig={current_sig})")
            except Exception as ex:
                log(f"  ⚠ bm25 index load failed ({ex}), rebuilding...")
                retriever = None
        else:
            log(f"  ⚠ index cache stale (sig/n_corpus mismatch), rebuilding...")
    if retriever is None:
        t0 = time.time()
        retriever = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
        retriever.index(corpus_tokens, show_progress=False)
        log(f"  index built in {time.time() - t0:.1f}s")
        if index_dir.exists():
            import shutil
            shutil.rmtree(index_dir)
        retriever.save(str(index_dir), corpus=corpus_texts, show_progress=False)
        index_sig_path.write_text(json.dumps({"sig": current_sig, "n_corpus": n_corpus}))
        log(f"  cached → {index_dir} (sig={current_sig}, n_corpus={n_corpus})")

    t0 = time.time()
    query_tokens = bm25s.tokenize(queries, stopwords="en", show_progress=False)
    BM25_BATCH = 1000
    results = [None] * len(queries)
    topk_buffer: list[np.ndarray] = [] if save_topk_path is not None else None

    def _slice_tok(tok, s, e):
        return type(tok)(tok.ids[s:e], tok.vocab)

    for s in range(0, len(queries), BM25_BATCH):
        e = min(s + BM25_BATCH, len(queries))
        sub_tokens = _slice_tok(query_tokens, s, e)
        sub_res = retriever.retrieve(sub_tokens, k=bm25_k, show_progress=False)
        for i in range(e - s):
            gi = s + i
            tgt_idx = int(target_indices[gi])
            sorted_docs = sub_res.documents[i]
            rank = rank_target_in_sorted(tgt_idx, sorted_docs, bm25_k)
            results[gi] = rr_hit_from_rank(rank)
            if topk_buffer is not None:
                topk_buffer.append(np.asarray(sorted_docs[:topk_k], dtype=np.int32))
    log(f"  retrieved in {time.time() - t0:.1f}s")
    if save_topk_path is not None:
        save_topk_path.parent.mkdir(parents=True, exist_ok=True)
        topk_arr = np.stack(topk_buffer, axis=0) if topk_buffer else np.zeros((0, topk_k), dtype=np.int32)
        np.savez_compressed(save_topk_path, topk_asins=topk_arr)
        log(f"  saved top-{topk_k} → {save_topk_path} (shape={topk_arr.shape})")
    return results


# ===========================================================================
# SPLADE (learned sparse)
# ===========================================================================
def splade_encode(model, tok, texts: list[str], max_length: int = 128,
                  batch_size: int = 128) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    n_texts = len(texts)
    n_done = 0
    for s in range(0, n_texts, batch_size):
        e = min(s + batch_size, n_texts)
        inputs = tok(texts[s:e], return_tensors="pt", truncation=True,
                     max_length=max_length, padding=True).to(model.device)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**inputs).logits  # (B, L, V)
        rep = torch.log(1 + torch.relu(logits.float()))
        rep = rep * inputs.attention_mask.unsqueeze(-1)
        rep = rep.max(dim=1)[0]  # (B, V)
        chunks.append(rep.cpu().half())
        del logits, rep, inputs
        torch.cuda.empty_cache()
        n_done += e - s
        if n_done % (batch_size * 50) == 0 or n_done == n_texts:
            log(f"    encoded {n_done}/{n_texts} ({100 * n_done / n_texts:.1f}%)")
    return torch.cat(chunks, dim=0)


def splade_encode_stream(model, tok, texts: list[str], cache_dir: Path,
                         max_length: int = 128, batch_size: int = 128,
                         tag: str = "corpus") -> None:
    """Stream-encode texts to disk chunk-by-chunk. Each chunk is (batch, V) fp16.

    Writes <cache_dir>/<tag>_chunk_NNN.pt files. Avoids the 14GB full-tensor
    concat that triggered OOM-kill on 30GB cgroup.
    """
    n_texts = len(texts)
    n_chunks = (n_texts + batch_size - 1) // batch_size
    n_done = 0
    t0 = time.time()
    for chunk_idx in range(n_chunks):
        chunk_path = cache_dir / f"{tag}_chunk_{chunk_idx:04d}.pt"
        if chunk_path.exists():
            n_done += min(batch_size, n_texts - chunk_idx * batch_size)
            continue
        s = chunk_idx * batch_size
        e = min(s + batch_size, n_texts)
        inputs = tok(texts[s:e], return_tensors="pt", truncation=True,
                     max_length=max_length, padding=True).to(model.device)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**inputs).logits  # (B, L, V)
        rep = torch.log(1 + torch.relu(logits.float()))
        rep = rep * inputs.attention_mask.unsqueeze(-1)
        rep = rep.max(dim=1)[0]  # (B, V)
        torch.save(rep.cpu().half(), chunk_path)
        del logits, rep, inputs
        torch.cuda.empty_cache()
        n_done += (e - s)
        if n_done % (batch_size * 50) == 0 or n_done == n_texts:
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (n_texts - n_done) / rate if rate > 0 else 0
            log(f"    {tag} encoded {n_done}/{n_texts} ({100 * n_done / n_texts:.1f}%) "
                f"rate={rate:.0f}/s eta={eta:.0f}s")


def splade_retrieve(queries: list[str], corpus_texts: list[str],
                    target_indices: np.ndarray, *,
                    corpus_sig: str, query_sig: str,
                    save_topk_path: Path | None = None,
                    topk_k: int = TOPK_SAVE_K) -> list[dict]:
    # 2026-09-22: monkey-patch transformers' is_torch_greater_or_equal —
    # pq_env 装了 5 个 torch dist-info（2.4/2.5.1/2.9.0/2.13.0/2.14.0），
    # importlib.metadata.version('torch') 返回 '2.5.1' 但实际加载是 2.9.0+cu128。
    # transformers 4.57.6 的 check_torch_load_is_safe() 错误判断为 torch<2.6，
    # 在 SPLADE 加载 .bin 权重时 raise ValueError。这里强制返回 True 以跳过检查。
    import transformers.utils.import_utils as _tui
    import torch as _torch_for_patch
    from packaging import version as _v_for_patch
    def _patched_is_torch_ge(library_version, accept_dev=False):
        return _v_for_patch.parse(_torch_for_patch.__version__.split('+')[0]) >= _v_for_patch.parse(library_version)
    _tui.is_torch_greater_or_equal = _patched_is_torch_ge
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    log("\n=== SPLADE (learned_sparse, chunked-streaming) ===")
    cache_dir = EMBED_CACHE_DIR / "splade"
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = "naver/splade-cocondenser-ensembledistil"
    HF_CACHE = "/home/wlia0047/hj82_scratch2/wenyu/hf_cache"
    tok = AutoTokenizer.from_pretrained(name, cache_dir=HF_CACHE)
    model = AutoModelForMaskedLM.from_pretrained(name, cache_dir=HF_CACHE).to("cuda").eval()

    n_corpus = len(corpus_texts)
    n_query = len(queries)
    n_corpus_chunks = (n_corpus + 127) // 128
    corpus_sig_file = cache_dir / "corpus.sig"

    # Verify corpus chunk-stream cache: JSON {sig, n_corpus, n_chunks}; rebuild on mismatch
    corpus_cached_ok = False
    if corpus_sig_file.exists():
        try:
            meta = json.loads(corpus_sig_file.read_text())
            if (meta.get("sig") == corpus_sig
                    and meta.get("n_corpus") == n_corpus
                    and meta.get("n_chunks") == n_corpus_chunks
                    and all((cache_dir / f"corpus_chunk_{i:04d}.pt").exists()
                            for i in range(n_corpus_chunks))):
                corpus_cached_ok = True
        except (json.JSONDecodeError, KeyError):
            corpus_cached_ok = False
    if not corpus_cached_ok:
        if corpus_sig_file.exists():
            log(f"  ⚠ corpus cache stale, rebuilding chunked-stream...")
        # Sweep stale chunks first (splade_encode_stream skips existing files)
        for old in cache_dir.glob("corpus_chunk_*.pt"):
            old.unlink()
        log(f"  corpus: {n_corpus} docs in {n_corpus_chunks} chunks of 128")
        splade_encode_stream(model, tok, corpus_texts, cache_dir, batch_size=128, tag="corpus")
        corpus_sig_file.write_text(json.dumps({"sig": corpus_sig, "n_corpus": n_corpus,
                                               "n_chunks": n_corpus_chunks}))
        log(f"  cached → corpus chunks (sig={corpus_sig})")
    else:
        log(f"  ✓ corpus chunk-stream cache hit ({n_corpus_chunks} chunks, "
            f"n_corpus={n_corpus}, sig={corpus_sig})")

    # Encode queries — cache if query_sig matches
    query_cache = cache_dir / "query_embeds.pt"
    query_sig_file = _sig_path_for(query_cache)
    splade_queries = None
    if query_cache.exists() and query_sig_file.exists():
        cached_sig = query_sig_file.read_text().strip()
        if cached_sig == query_sig:
            cached_q = torch.load(query_cache, weights_only=True)
            if cached_q.shape[0] == n_query:
                splade_queries = cached_q
                log(f"  ✓ query embeds cache hit ({splade_queries.shape}, sig={query_sig})")
            else:
                log(f"  ⚠ query embeds shape mismatch (cached={cached_q.shape[0]}, current={n_query})")
        else:
            log(f"  ⚠ query sig mismatch (cached={cached_sig}, current={query_sig})")
    if splade_queries is None:
        log(f"  encoding {n_query} queries...")
        splade_queries = splade_encode(model, tok, queries, batch_size=128)
        log(f"  queries shape={splade_queries.shape}")
        torch.save(splade_queries, query_cache)
        query_sig_file.write_text(query_sig)
        log(f"  cached → {query_cache} (sig={query_sig})")

    # Cleanup old monolithic cache (legacy from previous OOM-killed run)
    legacy_corpus_cache = cache_dir / "corpus_splade.pt"
    legacy_query_cache = cache_dir / "query_splade.pt"
    if legacy_corpus_cache.exists():
        log(f"  removing legacy monolithic corpus cache")
        legacy_corpus_cache.unlink()
    if legacy_query_cache.exists():
        legacy_query_cache.unlink()

    # Matmul: stream-load corpus chunks, compute scores chunk by chunk
    log(f"  streaming matmul Q × N (chunked)...")
    t0 = time.time()
    q_gpu = splade_queries.cuda().float()  # (Q, V) fp32 — small
    Q_BATCH = 500
    N = n_corpus
    results: list[dict] = [None] * n_query  # filled in place

    # Phase 1: collect top-similarity-per-query (we need the per-query target_rank)
    # We process chunk by chunk, keeping running top counts per query.
    # For target_idx we need to compare against all corpus docs, so accumulate
    # all scores in a (Q, N) matrix on GPU. Q=6357, N=217K, fp32=5.5GB → fits.
    log(f"  allocating scores matrix ({n_query}×{N})...")
    scores_gpu = torch.empty((n_query, N), dtype=torch.float32, device="cuda")

    for chunk_idx in range(n_corpus_chunks):
        s_idx = chunk_idx * 128
        e_idx = min(s_idx + 128, N)
        chunk_path = cache_dir / f"corpus_chunk_{chunk_idx:04d}.pt"
        chunk = torch.load(chunk_path, weights_only=True).cuda().float()  # (chunk, V)
        chunk_scores = q_gpu @ chunk.T  # (Q, chunk)
        scores_gpu[:, s_idx:e_idx] = chunk_scores
        del chunk, chunk_scores
        if (chunk_idx + 1) % 100 == 0 or chunk_idx == n_corpus_chunks - 1:
            elapsed = time.time() - t0
            rate = (chunk_idx + 1) / elapsed if elapsed > 0 else 0
            eta = (n_corpus_chunks - chunk_idx - 1) / rate if rate > 0 else 0
            log(f"    matmul chunk {chunk_idx + 1}/{n_corpus_chunks} "
                f"rate={rate:.1f}/s eta={eta:.0f}s")

    # Compute rank from scores
    log(f"  computing target ranks...")
    t_rank = time.time()
    tgt_all = torch.as_tensor(target_indices, device="cuda", dtype=torch.long)
    topk_buffer: list[np.ndarray] = [] if save_topk_path is not None else None
    for gi in range(n_query):
        tgt_idx = int(tgt_all[gi].item())
        if tgt_idx < 0:
            results[gi] = rr_hit_from_rank(-1)
            if topk_buffer is not None:
                topk_buffer.append(np.zeros(topk_k, dtype=np.int32))
            continue
        sc = scores_gpu[gi]
        rank = int((sc > sc[tgt_idx]).sum().item()) + 1
        results[gi] = rr_hit_from_rank(rank)
        if topk_buffer is not None:
            top_idx = torch.topk(sc, k=min(topk_k, sc.shape[0])).indices.cpu().numpy().astype(np.int32)
            if top_idx.shape[0] < topk_k:
                pad = np.full(topk_k - top_idx.shape[0], -1, dtype=np.int32)
                top_idx = np.concatenate([top_idx, pad])
            topk_buffer.append(top_idx)
    log(f"  rank computation done in {time.time() - t_rank:.1f}s")
    log(f"  full matmul done in {time.time() - t0:.1f}s")

    del scores_gpu, q_gpu, splade_queries
    torch.cuda.empty_cache()
    del model, tok
    torch.cuda.empty_cache()
    if save_topk_path is not None and topk_buffer is not None:
        save_topk_path.parent.mkdir(parents=True, exist_ok=True)
        topk_arr = np.stack(topk_buffer, axis=0) if topk_buffer else np.zeros((0, topk_k), dtype=np.int32)
        np.savez_compressed(save_topk_path, topk_asins=topk_arr)
        log(f"  saved top-{topk_k} → {save_topk_path} (shape={topk_arr.shape})")
    return results


# ===========================================================================
# Dense bi-encoders (MiniLM / MPNet / BGE / GTE)
# ===========================================================================
def dense_retrieve(retr_name: str, hf_id: str, queries: list[str],
                   target_indices: np.ndarray, *,
                   corpus_sig: str, query_sig: str,
                   save_topk_path: Path | None = None,
                   topk_k: int = TOPK_SAVE_K) -> tuple[list[dict], np.ndarray]:
    from sentence_transformers import SentenceTransformer
    log(f"\n=== {retr_name} (dense, {hf_id}) ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(hf_id, device=device)

    cache_dir = EMBED_CACHE_DIR / retr_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    corpus_cache = cache_dir / "corpus_embeds.npy"
    corpus_sig_file = _sig_path_for(corpus_cache)
    query_cache = cache_dir / "query_embeds.npy"
    query_sig_file = _sig_path_for(query_cache)

    # Load pre-built ASIN corpus (built by build_asin_to_doc.py); this script is read-only.
    if not ASIN_TO_DOC_CACHE.exists():
        raise FileNotFoundError(
            f"ASIN_TO_DOC_CACHE missing: {ASIN_TO_DOC_CACHE}. "
            f"Run `python build_asin_to_doc.py` first (Rule 7: no fallback)."
        )
    asin_to_doc = json.load(open(ASIN_TO_DOC_CACHE, encoding="utf-8"))
    asins = sorted(asin_to_doc.keys())
    corpus_texts = [asin_to_doc[a] for a in asins]

    if corpus_cache.exists() and corpus_sig_file.exists():
        cached_sig = corpus_sig_file.read_text().strip()
        if cached_sig == corpus_sig:
            corpus_embeds = np.load(corpus_cache)
            log(f"  ✓ corpus embeds cache ({corpus_embeds.shape}, sig={corpus_sig})")
        else:
            log(f"  ⚠ corpus sig mismatch (cached={cached_sig}, current={corpus_sig}), re-encoding...")
            corpus_embeds = None
    else:
        corpus_embeds = None
    if corpus_embeds is None:
        log(f"  encoding {len(corpus_texts)} corpus (batch=128)...")
        t0 = time.time()
        corpus_embeds = model.encode(corpus_texts, batch_size=128, show_progress_bar=False,
                                     convert_to_numpy=True, normalize_embeddings=True)
        log(f"  encoded in {time.time() - t0:.1f}s, shape={corpus_embeds.shape}")
        np.save(corpus_cache, corpus_embeds)
        corpus_sig_file.write_text(corpus_sig)
        log(f"  cached → {corpus_cache} (sig={corpus_sig})")

    if query_cache.exists() and query_sig_file.exists():
        cached_sig = query_sig_file.read_text().strip()
        if cached_sig == query_sig:
            q_embeds_cached = np.load(query_cache)
            if q_embeds_cached.shape[0] == len(queries):
                q_embeds = q_embeds_cached
                log(f"  ✓ query embeds cache ({q_embeds.shape}, sig={query_sig})")
            else:
                log(f"  ⚠ query embeds cache shape mismatch (cached={q_embeds_cached.shape[0]}, "
                    f"current={len(queries)}), re-encoding...")
                q_embeds = None
        else:
            log(f"  ⚠ query sig mismatch (cached={cached_sig}, current={query_sig}), re-encoding...")
            q_embeds = None
    else:
        q_embeds = None
    if q_embeds is None:
        log(f"  encoding {len(queries)} queries (batch=512)...")
        t0 = time.time()
        q_embeds = model.encode(queries, batch_size=512, show_progress_bar=False,
                                convert_to_numpy=True, normalize_embeddings=True)
        log(f"  encoded in {time.time() - t0:.1f}s, shape={q_embeds.shape}")
        np.save(query_cache, q_embeds)
        query_sig_file.write_text(query_sig)
        log(f"  cached → {query_cache} (sig={query_sig})")

    log(f"  matmul + ranks on GPU...")
    t0 = time.time()
    corpus_gpu = torch.from_numpy(corpus_embeds).cuda()
    q_gpu = torch.from_numpy(q_embeds).cuda()
    results = []
    BATCH = 4000 if retr_name == "minilm" else 2000
    tgt_all = torch.as_tensor(target_indices, device="cuda", dtype=torch.long)
    topk_buffer: list[np.ndarray] = [] if save_topk_path is not None else None
    for s in range(0, q_gpu.shape[0], BATCH):
        e = min(s + BATCH, q_gpu.shape[0])
        sims = q_gpu[s:e] @ corpus_gpu.T
        tgt_chunk = tgt_all[s:e]
        for j in range(sims.shape[0]):
            tgt_idx = int(tgt_chunk[j].item())
            sc = sims[j]
            if tgt_idx < 0:
                results.append(rr_hit_from_rank(-1))
                if topk_buffer is not None:
                    top_idx = torch.topk(sc, k=min(topk_k, sc.shape[0])).indices.cpu().numpy().astype(np.int32)
                    if top_idx.shape[0] < topk_k:
                        pad = np.full(topk_k - top_idx.shape[0], -1, dtype=np.int32)
                        top_idx = np.concatenate([top_idx, pad])
                    topk_buffer.append(top_idx)
                continue
            rank = int((sc > sc[tgt_idx]).sum().item()) + 1
            results.append(rr_hit_from_rank(rank))
            if topk_buffer is not None:
                top_idx = torch.topk(sc, k=min(topk_k, sc.shape[0])).indices.cpu().numpy().astype(np.int32)
                if top_idx.shape[0] < topk_k:
                    pad = np.full(topk_k - top_idx.shape[0], -1, dtype=np.int32)
                    top_idx = np.concatenate([top_idx, pad])
                topk_buffer.append(top_idx)
        del sims
    del corpus_gpu, q_gpu, tgt_all
    torch.cuda.empty_cache()
    log(f"  matmul done in {time.time() - t0:.1f}s")
    if save_topk_path is not None and topk_buffer is not None:
        save_topk_path.parent.mkdir(parents=True, exist_ok=True)
        topk_arr = np.stack(topk_buffer, axis=0) if topk_buffer else np.zeros((0, topk_k), dtype=np.int32)
        np.savez_compressed(save_topk_path, topk_asins=topk_arr)
        log(f"  saved top-{topk_k} → {save_topk_path} (shape={topk_arr.shape})")
    return results, q_embeds


# ===========================================================================
# ColBERTv2 (late interaction, 768→128 linear projection)
# ===========================================================================
def _load_colbert_projection(snapshot_dir: Path) -> torch.Tensor:
    import safetensors.torch as st
    snap_str = str(snapshot_dir) + "/"
    sd = st.load_file(snap_str + "model.safetensors")
    return sd["linear.weight"].float()  # (128, 768)


def _colbert_encode_all(model, tok, proj_weight: torch.Tensor, texts: list[str],
                        max_length: int = 64, batch_size: int = 128
                        ) -> tuple[np.ndarray, np.ndarray]:
    max_L = max_length
    proj = proj_weight.cuda()  # (128, 768)
    D_out = proj.shape[0]
    reps_arr = np.zeros((len(texts), max_L, D_out), dtype=np.float16)
    valid_lens = np.zeros(len(texts), dtype=np.int32)
    n_done = 0
    t0 = time.time()
    for s in range(0, len(texts), batch_size):
        e = min(s + batch_size, len(texts))
        inputs = tok(texts[s:e], return_tensors="pt", truncation=True, max_length=max_length,
                     padding="max_length").to(model.device)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden = model(**inputs).last_hidden_state  # (B, L, 768)
        projected = hidden.float() @ proj.T  # (B, L, 128) fp32
        norms = projected.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        projected = projected / norms
        proj_cpu = projected.cpu().half().numpy()
        for j in range(e - s):
            L = int(inputs.attention_mask[j].sum().item())
            if L == 0:
                L = 1
            reps_arr[s + j, :L] = proj_cpu[j, :L]
            valid_lens[s + j] = L
        del hidden, projected, inputs
        torch.cuda.empty_cache()
        n_done += e - s
        if n_done % (batch_size * 50) == 0 or n_done == len(texts):
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (len(texts) - n_done) / rate if rate > 0 else 0
            log(f"    encoded {n_done}/{len(texts)} ({100 * n_done / len(texts):.1f}%) "
                f"rate={rate:.0f}/s eta={eta:.0f}s")
    return reps_arr, valid_lens


def colbertv2_retrieve(queries: list[str], corpus_texts: list[str],
                       target_indices: np.ndarray, *,
                       corpus_sig: str, query_sig: str,
                       save_topk_path: Path | None = None,
                       topk_k: int = TOPK_SAVE_K) -> tuple[list[dict], np.ndarray]:
    from transformers import AutoModel, AutoTokenizer
    log("\n=== ColBERTv2 (late_interaction, 768→128) ===")
    cache_dir = EMBED_CACHE_DIR / "colbertv2"
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = "colbert-ir/colbertv2.0"
    HF_CACHE = "/home/wlia0047/hj82_scratch2/wenyu/hf_cache"
    snap_root = Path(HF_CACHE) / "models--colbert-ir--colbertv2.0" / "snapshots"
    snap_dir = snap_root / sorted([p.name for p in snap_root.iterdir()])[0]

    tok = AutoTokenizer.from_pretrained(name, cache_dir=HF_CACHE)
    model = AutoModel.from_pretrained(name, cache_dir=HF_CACHE).to("cuda").eval()
    proj_weight = _load_colbert_projection(snap_dir)

    corpus_reps_file = cache_dir / "corpus_reps.npy"
    corpus_lens_file = cache_dir / "corpus_lens.npy"
    corpus_sig_file = cache_dir / "corpus.sig"
    query_reps_file = cache_dir / "query_reps.npy"
    query_lens_file = cache_dir / "query_lens.npy"
    query_sig_file = _sig_path_for(query_reps_file)

    n_corpus = len(corpus_texts)
    if corpus_reps_file.exists() and corpus_lens_file.exists() and corpus_sig_file.exists():
        try:
            meta = json.loads(corpus_sig_file.read_text())
            cached_ok = (meta.get("sig") == corpus_sig
                         and meta.get("n_corpus") == n_corpus)
        except (json.JSONDecodeError, KeyError, OSError):
            cached_ok = False
        if cached_ok:
            corpus_reps = np.load(corpus_reps_file)
            corpus_lens = np.load(corpus_lens_file)
            log(f"  ✓ corpus reps cache {corpus_reps.shape} (sig={corpus_sig})")
        else:
            log(f"  ⚠ corpus sig mismatch, re-encoding...")
            corpus_reps = None
    else:
        corpus_reps = None
    if corpus_reps is None:
        log(f"  encoding {len(corpus_texts)} corpus with ColBERTv2...")
        corpus_reps, corpus_lens = _colbert_encode_all(model, tok, proj_weight, corpus_texts,
                                                       max_length=64, batch_size=128)
        np.save(corpus_reps_file, corpus_reps)
        np.save(corpus_lens_file, corpus_lens)
        corpus_sig_file.write_text(json.dumps({"sig": corpus_sig, "n_corpus": n_corpus}))
        log(f"  cached → corpus reps (sig={corpus_sig}, n_corpus={n_corpus})")

    if query_reps_file.exists() and query_lens_file.exists() and query_sig_file.exists():
        cached_sig = query_sig_file.read_text().strip()
        if cached_sig == query_sig:
            query_reps = np.load(query_reps_file)
            query_lens = np.load(query_lens_file)
            log(f"  ✓ query reps cache {query_reps.shape} (sig={query_sig})")
        else:
            log(f"  ⚠ query sig mismatch (cached={cached_sig}, current={query_sig}), re-encoding...")
            query_reps = None
    else:
        query_reps = None
    if query_reps is None:
        log(f"  encoding {len(queries)} queries with ColBERTv2...")
        query_reps, query_lens = _colbert_encode_all(model, tok, proj_weight, queries,
                                                    max_length=32, batch_size=256)
        np.save(query_reps_file, query_reps)
        np.save(query_lens_file, query_lens)
        query_sig_file.write_text(query_sig)
        log(f"  cached → query reps (sig={query_sig})")

    log(f"  GPU maxsim Q × N...")
    t0 = time.time()
    corpus_gpu = torch.from_numpy(corpus_reps).cuda()
    results = []
    Q_cache = query_reps.shape[0]
    Q = min(Q_cache, len(target_indices))
    if Q < Q_cache:
        log(f"  ⚠ cache has {Q_cache} query reps but only {len(target_indices)} target indices; truncating to Q={Q}")
    N = corpus_reps.shape[0]
    CHUNK_N = 4096
    topk_buffer: list[np.ndarray] = [] if save_topk_path is not None else None
    for qi in range(Q):
        tgt_idx = int(target_indices[qi])
        if tgt_idx < 0:
            results.append(rr_hit_from_rank(-1))
            if topk_buffer is not None:
                topk_buffer.append(np.zeros(topk_k, dtype=np.int32))
            continue
        Lq = int(query_lens[qi])
        qt = torch.from_numpy(query_reps[qi, :Lq]).cuda().float()
        all_scores = np.empty(N, dtype=np.float32)
        for s in range(0, N, CHUNK_N):
            e = min(s + CHUNK_N, N)
            d_chunk = corpus_gpu[s:e].float()
            Ld = d_chunk.shape[1]
            lens_chunk = corpus_lens[s:e]
            mask = torch.arange(Ld, device="cuda")[None, :] < torch.as_tensor(lens_chunk, device="cuda")[:, None]
            d_chunk = d_chunk * mask.unsqueeze(-1)
            sim = torch.einsum("qd,cld->cql", qt, d_chunk)
            max_per_q = sim.max(dim=-1)[0]
            all_scores[s:e] = max_per_q.sum(dim=-1).cpu().numpy()
            del d_chunk, sim, max_per_q, mask
        tgt_score = all_scores[tgt_idx]
        all_scores_for_rank = all_scores.copy()
        all_scores_for_rank[tgt_idx] = -np.inf
        better = int((all_scores_for_rank > tgt_score).sum())
        rank = better + 1
        results.append(rr_hit_from_rank(rank))
        if topk_buffer is not None:
            top_idx = np.argpartition(-all_scores, topk_k)[:topk_k]
            # sort within top_k by score desc
            top_idx = top_idx[np.argsort(-all_scores[top_idx])]
            topk_buffer.append(top_idx.astype(np.int32))
        if (qi + 1) % 500 == 0 or qi == Q - 1:
            elapsed = time.time() - t0
            rate = (qi + 1) / elapsed if elapsed > 0 else 0
            eta = (Q - qi - 1) / rate if rate > 0 else 0
            log(f"    {qi + 1}/{Q} ({100 * (qi + 1) / Q:.1f}%) rate={rate:.1f}/s eta={eta:.0f}s")
    del corpus_gpu
    torch.cuda.empty_cache()
    log(f"  maxsim done in {time.time() - t0:.1f}s")
    del model, tok
    torch.cuda.empty_cache()
    # ColBERTv2 doesn't expose per-query 128d pooling easily; for volatility sim09
    # use mean of valid token reps as query embedding surrogate
    q_embeds_pooled = np.zeros((Q, query_reps.shape[2]), dtype=np.float32)
    for qi in range(Q):
        Lq = int(query_lens[qi])
        if Lq == 0:
            Lq = 1
        q_embeds_pooled[qi] = query_reps[qi, :Lq].mean(axis=0).astype(np.float32)
    norms = np.linalg.norm(q_embeds_pooled, axis=1, keepdims=True).clip(min=1e-9)
    q_embeds_pooled = q_embeds_pooled / norms
    if save_topk_path is not None and topk_buffer is not None:
        save_topk_path.parent.mkdir(parents=True, exist_ok=True)
        topk_arr = np.stack(topk_buffer, axis=0) if topk_buffer else np.zeros((0, topk_k), dtype=np.int32)
        np.savez_compressed(save_topk_path, topk_asins=topk_arr)
        log(f"  saved top-{topk_k} → {save_topk_path} (shape={topk_arr.shape})")
    return results, q_embeds_pooled


# ===========================================================================
# VOLATILITY (per retriever, sim09 slice)
# ===========================================================================
def compute_volatility_by_retriever(per_query: list[dict], retr_name: str,
                                    q_embeds: np.ndarray,
                                    sim_threshold: float = 0.9) -> dict:
    """Compute per-ASIN volatility (flip rate + RR std) for one retriever.

    q_embeds is the canonical sim09 reference embedding (always minilm's 384d
    dense), shared across all retrievers per user directive 2026-08-29.
    The retriever's own rank/RR are used for flip detection; only the
    pairwise query-query similarity threshold (sim09) comes from minilm.
    """
    by_asin: dict[str, list] = collections.defaultdict(list)
    for r in per_query:
        by_asin[r["asin"]].append(r)
    rank_key = f"{retr_name}_rank"
    rr_key = f"{retr_name}_RR"
    if q_embeds is None:
        raise ValueError(f"q_embeds must not be None for {retr_name} (canonical sim09 reference required)")
    n_total_asins = len(by_asin)
    n_asins_with_ge2 = sum(1 for qs in by_asin.values() if len(qs) >= 2)
    n_asins_with_ge3 = sum(1 for qs in by_asin.values() if len(qs) >= 3)
    n = sum(len(qs) for qs in by_asin.values())
    log(f"  ({retr_name}) {n_total_asins} ASINs ({n_asins_with_ge2} with >=2 queries, {n_asins_with_ge3} with >=3 queries) ...")
    selected_qs = [q for qs in by_asin.values() for q in qs]
    qid_to_embed = {id(q): q_embeds[i] for i, q in enumerate(selected_qs)}
    f1_list, f5_list, f10_list, f20_list = [], [], [], []
    rr_std_list = []
    n_asins_used = 0
    for asin, qs in by_asin.items():
        if len(qs) < 2:
            continue
        ranks = [q.get(rank_key) for q in qs]
        if any(r is None for r in ranks):
            continue
        hit1 = [1 if r == 1 else 0 for r in ranks]
        hit5 = [1 if r <= 5 else 0 for r in ranks]
        hit10 = [1 if r <= 10 else 0 for r in ranks]
        hit20 = [1 if r <= 20 else 0 for r in ranks]
        rrs = [float(q.get(rr_key) or 0.0) for q in qs]
        rr_std = float(np.std(rrs, ddof=0)) if len(rrs) >= 2 else None
        embeds = np.stack([qid_to_embed[id(q)] for q in qs], axis=0)
        sim_mat = embeds @ embeds.T

        def flip_rate(hit_labels):
            n_q = len(hit_labels)
            used, disagree = 0, 0
            for i in range(n_q):
                for j in range(i + 1, n_q):
                    if sim_mat[i, j] >= sim_threshold:
                        used += 1
                        if hit_labels[i] != hit_labels[j]:
                            disagree += 1
            if used == 0:
                return None
            return disagree / used

        for hit_labels, lst in [(hit1, f1_list), (hit5, f5_list),
                                (hit10, f10_list), (hit20, f20_list)]:
            v = flip_rate(hit_labels)
            if v is not None:
                lst.append(v)
        if rr_std is not None:
            rr_std_list.append(rr_std)
        n_asins_used += 1
    return {
        "n_total_asins": n_total_asins,
        "n_asins_with_ge2_queries": n_asins_with_ge2,
        "n_asins_with_ge3_queries": n_asins_with_ge3,
        "n_asins": n_asins_used,
        "Hit@1_FlipRate_mean": float(np.mean(f1_list)) if f1_list else None,
        "Hit@5_FlipRate_mean": float(np.mean(f5_list)) if f5_list else None,
        "Hit@10_FlipRate_mean": float(np.mean(f10_list)) if f10_list else None,
        "Hit@20_FlipRate_mean": float(np.mean(f20_list)) if f20_list else None,
        "RR_Std_mean": float(np.mean(rr_std_list)) if rr_std_list else None,
        "RR_Std_median": float(np.median(rr_std_list)) if rr_std_list else None,
        "RR_Std_std": float(np.std(rr_std_list, ddof=0)) if rr_std_list else None,
    }


# ===========================================================================
# MAIN
# ===========================================================================
def main_task_body():
    t_start = time.time()
    log("=== Stage 5 unified multi-retriever (NO rerank) ===")
    log(f"  retrievers: {RETR_NAMES}")

    # ---- 0. Cache check (selection signature) ----
    # If PER_QUERY_OUT already has results for the current selection,
    # skip all 7 retrievers and go straight to aggregate.
    log("\n=== 0. Selection signature + cache check ===")
    selection = json.load(open(SEL_IN))
    entries = _flatten_selection(selection)
    if SMOKE:
        if len(entries) < N_SMOKE_QUERIES:
            raise ValueError(
                f"SMOKE requires at least {N_SMOKE_QUERIES} selection entries, got {len(entries)}"
            )
        entries = entries[:N_SMOKE_QUERIES]
        selection = {"entries": entries}
        log(f"  SMOKE selection: first {len(entries)} queries")
    selection_sig = _compute_selection_signature(selection)
    log(f"  selection_sig = {selection_sig}")

    cached: dict | None = None
    if PER_QUERY_OUT.exists():
        try:
            cached = json.load(open(PER_QUERY_OUT))
        except (json.JSONDecodeError, OSError):
            cached = None
    if (
        cached is not None
        and cached.get("config", {}).get("signature") == selection_sig
        and set(RETR_NAMES).issubset(set(cached.get("config", {}).get("retrievers", [])))
    ):
        log(f"  ✓ cache hit ({PER_QUERY_OUT.stat().st_size / 1e6:.1f} MB, "
            f"sig={selection_sig}), skipping all 7 retrievers")
        log(f"  ✓ loaded {cached.get('n_queries', 0)} cached query records")
        # ---- 4-10 from cache: build aggregates ----
        query_records = cached["queries"]
        asins_count = cached["config"]["corpus_size"]

        # Load minilm q_embeds from disk so canonical sim09 reference is available
        # without re-running all 7 retrievers
        minilm_q_cache = EMBED_CACHE_DIR / "minilm" / "query_embeds.npy"
        if minilm_q_cache.exists():
            cached_minilm = np.load(minilm_q_cache)
            if cached_minilm.shape[0] == len(query_records):
                retr_q_embeds_cached = {n: cached_minilm for n in RETR_NAMES}
                canonical_name = "minilm"
                log(f"  ✓ loaded minilm q_embeds from disk ({cached_minilm.shape}) "
                    f"for canonical sim09 reference")
            else:
                raise RuntimeError(
                    f"minilm q_embeds cache stale (cached={cached_minilm.shape[0]}, "
                    f"current={len(query_records)}); delete {minilm_q_cache} and re-run"
                )
        else:
            raise RuntimeError(
                f"minilm q_embeds cache not found at {minilm_q_cache}; "
                f"cannot build canonical sim09 reference"
            )
        # 2026-09-19: top-100 cache 已被 fresh-run 路径保存 (主循环中所有 retriever
        # 都调用 save_topk_path=topk_path). 这里是 cache-hit 路径, top-100 应已存在.
        missing_topk = [n for n in RETR_NAMES
                        if not (TOPK_SAVE_DIR / f"{n}_top100.npz").exists()]
        if missing_topk:
            log(f"  ⚠ top-100 cache missing for {missing_topk}, "
                f"delete {PER_QUERY_OUT} to re-run retrievers and populate top-100 cache")
        _build_aggregates_and_save(query_records, asins_count, t_start,
                                   retr_q_embeds_cached, canonical_name)
        return
    log("  no cache hit (signature mismatch or missing), running 7 retrievers fresh")

    # ---- 1. Load selection (already done above) ----
    log("\n=== 1. Building query_records from selection ===")
    query_records: list[dict] = []
    for e in entries:
        q = e["selected"]
        if q is None:
            continue
        query_records.append({
            "asin": e["asin"],
            "query": q["query"],
        })
    log(f"  strict queries: {len(query_records)}")

    # ---- 2. Load ASIN corpus (read-only; built by build_asin_to_doc.py) ----
    log("\n=== 2. Loading ASIN corpus ===")
    if not ASIN_TO_DOC_CACHE.exists():
        raise FileNotFoundError(
            f"ASIN_TO_DOC_CACHE missing: {ASIN_TO_DOC_CACHE}. "
            f"Run `python build_asin_to_doc.py` first (Rule 7: no fallback)."
        )
    t0 = time.time()
    asin_to_doc = json.load(open(ASIN_TO_DOC_CACHE, encoding="utf-8"))
    asins = sorted(asin_to_doc.keys())
    asin_to_idx = {a: i for i, a in enumerate(asins)}
    log(f"  corpus size: {len(asins)} ASINs ({time.time() - t0:.1f}s)")
    corpus_texts = [asin_to_doc[a] for a in asins]
    queries = [r["query"] for r in query_records]
    target_indices = np.array([asin_to_idx.get(r["asin"], -1) for r in query_records])
    n_missing = int((target_indices < 0).sum())
    if n_missing:
        log(f"  WARNING: {n_missing} queries have missing target ASINs in corpus")

    # ---- 3. Retrieve with each retriever ----
    log("\n=== 3. Per-retriever retrieval (7 retrievers, no rerank) ===")
    corpus_sig = _corpus_signature(asin_to_doc=asin_to_doc, meta_file=META_FILE)
    query_sig = selection_sig  # selection_sig already computed in step 0
    log(f"  corpus_sig={corpus_sig}  query_sig={query_sig}")
    retr_results: dict[str, list[dict]] = {}
    retr_q_embeds: dict[str, np.ndarray | None] = {}

    for retr in RETRIEVERS:
        kind = retr["kind"]
        # 2026-09-19: Top-100 candidates cache for LLM rerank on all 7 retrievers.
        topk_path = TOPK_SAVE_DIR / f"{retr['name']}_top100.npz"
        if kind == "sparse_lexical":
            results = bm25_retrieve(queries, corpus_texts, target_indices,
                                   save_topk_path=topk_path)
            retr_q_embeds[retr["name"]] = None
        elif kind == "sparse_learned":
            results = splade_retrieve(queries, corpus_texts, target_indices,
                                      corpus_sig=corpus_sig, query_sig=query_sig,
                                      save_topk_path=topk_path)
            retr_q_embeds[retr["name"]] = None  # SPLADE sparse, no pooled embed for sim09
        elif kind == "dense":
            results, q_embeds = dense_retrieve(retr["name"], retr["hf_id"], queries,
                                              target_indices, corpus_sig=corpus_sig,
                                              query_sig=query_sig,
                                              save_topk_path=topk_path)
            retr_q_embeds[retr["name"]] = q_embeds
        elif kind == "late_interaction":
            results, q_embeds = colbertv2_retrieve(queries, corpus_texts, target_indices,
                                                  corpus_sig=corpus_sig, query_sig=query_sig,
                                                  save_topk_path=topk_path)
            retr_q_embeds[retr["name"]] = q_embeds
        else:
            raise ValueError(f"Unknown retriever kind: {kind}")
        retr_results[retr["name"]] = results

    # ---- 3b. Canonical sim09 reference embedding (minilm) ----
    # Per user directive 2026-08-29: ALL retrievers use the same query-query
    # similarity threshold for sim09 clustering, anchored to minilm's 384d
    # dense embedding. BM25/SPLADE have no dense query embed — they borrow
    # minilm's. This makes cross-retriever volatility comparable.
    canonical_q_embeds = retr_q_embeds.get("minilm")
    if canonical_q_embeds is None:
        # Fallback: pick any available dense retriever's embeds
        for n in ("mpnet", "bge_base_v15", "gte_base", "colbertv2"):
            if retr_q_embeds.get(n) is not None:
                canonical_q_embeds = retr_q_embeds[n]
                log(f"  canonical sim09 reference: minilm missing → using {n}")
                break
    if canonical_q_embeds is None:
        raise RuntimeError("No dense retriever produced query embeddings; cannot build sim09 reference")
    canonical_embeds_name = "minilm" if retr_q_embeds.get("minilm") is not None else \
        next((n for n in ("mpnet", "bge_base_v15", "gte_base", "colbertv2")
              if retr_q_embeds.get(n) is not None), "unknown")
    log(f"  canonical sim09 reference: {canonical_embeds_name} (shape={canonical_q_embeds.shape})")
    # Override: every retriever's sim09 clustering uses canonical_embeds
    for n in RETR_NAMES:
        retr_q_embeds[n] = canonical_q_embeds

    # ---- 4. Merge into query_records ----
    log("\n=== 4. Merging per-retriever results ===")
    log(f"  query_records: {len(query_records)}")
    for n in RETR_NAMES:
        log(f"  retr_results[{n}]: {len(retr_results[n])}")
    for gi, r in enumerate(query_records):
        for n in RETR_NAMES:
            if gi >= len(retr_results[n]):
                raise IndexError(
                    f"gi={gi} out of range for retr_results[{n}] "
                    f"(len={len(retr_results[n])}). "
                    f"query_records len={len(query_records)}. "
                    f"All lengths: " + ", ".join(f"{x}={len(retr_results[x])}" for x in RETR_NAMES)
                )
            res = retr_results[n][gi]
            r[f"{n}_rank"] = res["rank"]
            r[f"{n}_RR"] = res["RR"]
            r[f"{n}_hit1"] = res["hit1"]
            r[f"{n}_hit5"] = res["hit5"]
            r[f"{n}_hit10"] = res["hit10"]

    # ---- 5. Save per-query intermediate (with signature for cache) ----
    PER_QUERY_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(PER_QUERY_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Stage 5 unified multi-retriever on strict alignment (NO rerank)",
                "retrievers": RETR_NAMES,
                "selection_file": str(SEL_IN),
                "corpus_size": len(asins),
                "signature": selection_sig,
            },
            "n_queries": len(query_records),
            "queries": query_records,
        }, f, ensure_ascii=False)
    log(f"  wrote → {PER_QUERY_OUT} (signature={selection_sig})")

    # ---- 6-10. Build aggregates + save ----
    _build_aggregates_and_save(query_records, len(asins), t_start, retr_q_embeds, canonical_embeds_name)


def _build_aggregates_and_save(query_records: list[dict], asins_count: int,
                               t_start: float,
                               retr_q_embeds: dict[str, np.ndarray | None] | None = None,
                               canonical_embeds_name: str = "minilm") -> None:
    """Build headline + volatility + save summary JSONs.

    retr_q_embeds is None when called from cache-hit path (volatility is skipped
    since q_embeds aren't cached). Fresh-run path passes the in-memory dict.
    canonical_embeds_name labels the sim09 reference embedding in the config.
    """
    if retr_q_embeds is None:
        # Cache hit: skip volatility (no q_embeds), only compute headline
        retr_q_embeds = {n: None for n in RETR_NAMES}

    # ---- 6. Volatility (per retriever, sim09 slice) ----
    log("\n=== 6. Per-retriever volatility (sim09) ===")
    volatility: dict = {}
    for n in RETR_NAMES:
        v = compute_volatility_by_retriever(query_records, n, retr_q_embeds[n])
        volatility[n] = v
        log(f"  {n:<12} n_asins={v.get('n_asins')}  "
            f"Hit@1_flip={v.get('Hit@1_FlipRate_mean')}  "
            f"Hit@10_flip={v.get('Hit@10_FlipRate_mean')}  "
            f"RR_Std={v.get('RR_Std_mean')}")

    # ---- 7. Save retrieval_summary.json (per Rule 13/16) ----
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Stage 5 unified multi-retriever volatility (sim09 selected_only slice, NO rerank). sim09 clustering anchored to minilm 384d for ALL retrievers (BM25/SPLADE borrow minilm embed).",
                "retrievers": RETR_NAMES,
                "n_retrievers": len(RETR_NAMES),
                "sim09_reference": canonical_embeds_name,
                "selection_file": str(SEL_IN),
                "corpus_size": asins_count,
            },
            "volatility_sim09": volatility,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {SUMMARY_OUT}")

    # ---- 9. Save volatility.json (canonical, backward-compat) ----
    # Keep BM25 + MiniLM as headline volatility for back-compat with existing dashboards
    canonical_volatility: dict = {}
    for n in ("bm25", "minilm"):
        v = volatility[n]
        canonical_volatility[n] = {
            "n_asins": v.get("n_asins", 0),
            "Hit@1_FlipRate_mean": v.get("Hit@1_FlipRate_mean"),
            "Hit@1_FlipRate_median": None,
            "Hit@5_FlipRate_mean": v.get("Hit@5_FlipRate_mean"),
            "Hit@5_FlipRate_median": None,
            "Hit@10_FlipRate_mean": v.get("Hit@10_FlipRate_mean"),
            "Hit@10_FlipRate_median": None,
            "Hit@20_FlipRate_mean": v.get("Hit@20_FlipRate_mean"),
            "Hit@20_FlipRate_median": None,
            "RR_Std_mean": v.get("RR_Std_mean"),
            "RR_Std_median": v.get("RR_Std_median"),
            "RR_Std_std": v.get("RR_Std_std"),
        }
    with open(VOLATILITY_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Canonical Stage 5 volatility (BM25 + MiniLM, selected_only_sim09 slice) — "
                               "supersedes prior volatility.json. Full multi-retriever volatility "
                               "see retrieval_summary.json.",
                "slices": ["selected_only_sim09"],
                "retrievers": ["bm25", "minilm"],
            },
            "stability_flip": {
                "selected_only_sim09": canonical_volatility,
            },
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {VOLATILITY_OUT}")

    # ---- 10. Final volatility table ----
    log("\n=== Final Volatility (multi-retriever on strict alignment, NO rerank) ===")
    header = (f"{'retriever':<14} {'n_asins':>8} {'Hit@1_flip':>11} {'Hit@10_flip':>12} {'RR_Std':>8}")
    log(header)
    log("-" * len(header))
    for n in RETR_NAMES:
        v = volatility[n]
        n_asins = v.get("n_asins", 0)
        h1 = f"{v['Hit@1_FlipRate_mean'] * 100:>10.2f}%" if v.get("Hit@1_FlipRate_mean") is not None else f"{'n/a':>11}"
        h10 = f"{v['Hit@10_FlipRate_mean'] * 100:>11.2f}%" if v.get("Hit@10_FlipRate_mean") is not None else f"{'n/a':>12}"
        rrs = f"{v['RR_Std_mean']:.4f}" if v.get("RR_Std_mean") is not None else f"{'n/a':>8}"
        log(f"{n:<14} {n_asins:>8d} {h1} {h10} {rrs}")

    log(f"\n=== Stage 5 unified complete ({time.time() - t_start:.1f}s) ===")


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    """用户指令 2026-09-23: 串行运行 3 个 category.

    每个 category 重新绑定该脚本使用的路径常量为 category-specific 路径,
    然后调原 main_task_body() (保持原有逻辑不动). 产物写到
    result/<stage>/<baby|musical|video_games>/ 子目录.
    """
    global SENT_CACHE, UID_TO_SENTS, ASIN_USERS_PATH, ATTRIBUTES_PATH, META_FILE, OUT_DIR, OUT_PATH, ASIN_TO_DOC_CACHE, SEL_IN, RESULT_DIR, PER_QUERY_OUT, SUMMARY_OUT, VOLATILITY_OUT, TOPK_SAVE_DIR, EMBED_CACHE_DIR  # noqa
    # backup current (Baby) defaults
    saved = {
        k: v for k, v in globals().items()
        if k in {"SENT_CACHE", "UID_TO_SENTS", "ASIN_USERS_PATH", "ATTRIBUTES_PATH",
                 "META_FILE", "OUT_DIR", "OUT_PATH", "ASIN_TO_DOC_CACHE", "SEL_IN",
                 "RESULT_DIR", "PER_QUERY_OUT", "SUMMARY_OUT", "VOLATILITY_OUT",
                 "TOPK_SAVE_DIR", "EMBED_CACHE_DIR"}
        and isinstance(v, Path)
    }
    base_out = REPO_ROOT / "result" / Path(__file__).parent.name
    for category, subdir in CATEGORY_INPUTS:
        log(f"\n========== [{category}] (subdir={subdir}) ==========")
        # Reset all known category-dependent paths to point at the per-category subdir.
        if "SENT_CACHE" in saved:
            SENT_CACHE = REPO_ROOT / "result/02_user_review_sentence_extract" / f"uid_to_sentences_{subdir}.pkl"
        if "UID_TO_SENTS" in saved:
            UID_TO_SENTS = REPO_ROOT / "result/02_user_review_sentence_extract" / f"uid_to_sentences_{subdir}.pkl"
        if "ASIN_USERS_PATH" in saved:
            ASIN_USERS_PATH = REPO_ROOT / "result/02_user_review_sentence_extract" / f"asin_to_users_{subdir}.pkl"
        if "ATTRIBUTES_PATH" in saved:
            ATTRIBUTES_PATH = REPO_ROOT / "result/01_attribute_extraction" / f"product_attributes_{subdir}.pkl"
        if "META_FILE" in saved:
            META_FILE = Path("/home/wlia0047/hj82/wenyu/PersoanlQuery/data") / {
                "baby": "meta_Baby_Products_2023.jsonl",
                "musical": "meta_Musical_Instruments.jsonl",
                "video_games": "meta_Video_Games.jsonl",
            }[subdir]
        if "OUT_DIR" in saved:
            OUT_DIR = base_out / subdir
        if "OUT_PATH" in saved:
            OUT_PATH = base_out / subdir / saved["OUT_PATH"].name
        if "ASIN_TO_DOC_CACHE" in saved:
            ASIN_TO_DOC_CACHE = base_out / subdir / saved["ASIN_TO_DOC_CACHE"].name
        if "SEL_IN" in saved:
            SEL_IN = REPO_ROOT / "result/08_select_query" / subdir / saved["SEL_IN"].name
        if "RESULT_DIR" in saved:
            RESULT_DIR = base_out / subdir
        if "PER_QUERY_OUT" in saved:
            PER_QUERY_OUT = base_out / subdir / saved["PER_QUERY_OUT"].name
        if "SUMMARY_OUT" in saved:
            SUMMARY_OUT = base_out / subdir / saved["SUMMARY_OUT"].name
        if "VOLATILITY_OUT" in saved:
            VOLATILITY_OUT = base_out / subdir / saved["VOLATILITY_OUT"].name
        if "TOPK_SAVE_DIR" in saved:
            TOPK_SAVE_DIR = base_out / subdir / saved["TOPK_SAVE_DIR"].name
        if "EMBED_CACHE_DIR" in saved:
            EMBED_CACHE_DIR = (Path("/home/wlia0047/hj82_scratch2/wenyu") /
                               "gaussian_vades/multiretrieval_embeds" / subdir)
        OUT_DIR.mkdir(parents=True, exist_ok=True) if "OUT_DIR" in saved else None
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True) if "OUT_PATH" in saved else None
        ASIN_TO_DOC_CACHE.parent.mkdir(parents=True, exist_ok=True) if "ASIN_TO_DOC_CACHE" in saved else None
        SEL_IN.parent.mkdir(parents=True, exist_ok=True) if "SEL_IN" in saved else None
        RESULT_DIR.mkdir(parents=True, exist_ok=True) if "RESULT_DIR" in saved else None
        PER_QUERY_OUT.parent.mkdir(parents=True, exist_ok=True) if "PER_QUERY_OUT" in saved else None
        SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True) if "SUMMARY_OUT" in saved else None
        VOLATILITY_OUT.parent.mkdir(parents=True, exist_ok=True) if "VOLATILITY_OUT" in saved else None
        TOPK_SAVE_DIR.mkdir(parents=True, exist_ok=True) if "TOPK_SAVE_DIR" in saved else None
        EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True) if "EMBED_CACHE_DIR" in saved else None
        try:
            main_task_body()
        except Exception as e:
            log(f"[{category}] FAILED: {e!r}")
            raise
    # Restore Baby defaults (for import compatibility with downstream).
    for k, v in saved.items():
        globals()[k] = v


if __name__ == "__main__":
    main()