"""Stage 5 unified multi-retriever on strict alignment (NO rerank).

按当前检索配置，在 Stage 4 选出的 strict_personalized queries 上运行 6 个 retrievers，计算 per-retriever headline + volatility。

6 retrievers:
  1. BM25 (lexical_sparse, bm25s lucene k1=1.5 b=0.75)
  2. SPLADE (learned_sparse, naver/splade-cocondenser-ensembledistil)
  3. GTE-base (唯一 Dense Bi-Encoder, 768d)
  4. BGE-M3 dense (FlagEmbedding, 1024d)
  5. BGE-M3 hybrid (dense + sparse + ColBERT)
  6. ColBERTv2 (late_interaction, 768→128 linear projection)

(Cross-encoder rerank 已删除 — full corpus CE 不实际,Stage 5 默认
BM25 top-100 + cross-encoder 的 BM25-only rerank pipeline 见已弃用版本
`stage8_5_rerank_*`。)

输出:
  scratch2/.../stage8_5_retrieval_per_query.json  (per-query intermediate, 6 retrievers)
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

# Stage 11 只读 scratch 中预构建的 asin_to_doc corpus cache。
# 构建/清洗由 `11_syntactic_evaluation/build_asin_to_doc.py` 独立负责。
# 本脚本不再调用 `build_meta_corpus`。cache invalidation 指纹由 builder 维护。
# 注意：_corpus_signature 包含 asin_to_doc 内容 hash，下游 corpus_sig 随清洗变化。
from build_asin_to_doc import _corpus_signature, _sig_path_for

# 用户指令 2026-08-30: 支持 SEL_OUT_SUFFIX 让 strict34 cohort 跑独立 cache, 不覆盖 canonical
_SEL_SUFFIX = os.environ.get("SEL_OUT_SUFFIX", "")

# ===========================================================================
# PATHS (inlined from common/syntax_subspace_utils.py 2026-09-06: common/ deleted)
# ===========================================================================
REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
# 用户指令 2026-09-23: data 目录从 REPO_ROOT/data 迁移到 hj82 同名 data 目录.
DATA_DIR = Path("/home/wlia0047/hj82/wenyu/PersoanlQuery/data")
ASIN_TO_DOC_CACHE = Path("/home/wlia0047/hj82_scratch2/wenyu/stage11_corpus_cache/baby/asin_to_doc.json")
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
TOPK_SAVE_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/stage11_retrieval_cache/baby/top100_cache")
TOPK_SAVE_K = 100

# 硬编码运行配置（Rule 3）；首次运行必须先用最小 smoke 验证端到端链路。
SMOKE = os.environ.get("STAGE11_SMOKE") == "1"
N_SMOKE_QUERIES = 5


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
    {"name": "gte_base", "kind": "dense", "hf_id": "thenlper/gte-base", "dim": 768},
    {"name": "bge_m3", "kind": "dense_bge_m3", "hf_id": "BAAI/bge-m3", "dim": 1024},
    {"name": "bge_m3_hybrid", "kind": "dense_bge_m3_hybrid", "hf_id": "BAAI/bge-m3", "dim": 1024,
     "weights": (1.0, 0.3, 1.0)},  # (w_dense, w_sparse, w_colbert)
    {"name": "colbertv2", "kind": "late_interaction", "hf_id": "colbert-ir/colbertv2.0", "dim": 128},
]
RETR_NAMES = [r["name"] for r in RETRIEVERS]
BGE_M3_HF_CACHE = Path("/home/wlia0047/hj82/wenyu/hf_cache")
# 2026-09-25: raise batch — bs=64 used ~2.5GB/46GB VRAM; larger batch cuts steps.
BGE_M3_CORPUS_BATCH = 512
BGE_M3_QUERY_BATCH = 512
# 2026-09-26: BGE-M3 hybrid (dense + sparse + colbert) parameters.
# - batch lowered vs dense-only to leave headroom for sparse + colbert matmul
# - colbert matmul is O(B * corpus * 1024 * max_token_len_q * max_token_len_d)
#   → keep B=64 to fit 46 GB VRAM with corpus up to 220 k docs.
BGE_M3_HYBRID_CORPUS_BATCH = 64
BGE_M3_HYBRID_QUERY_BATCH = 64
BGE_M3_HYBRID_COLBERT_MAXLEN = 16  # cap token len to keep colbert matmul cheap
BGE_M3_HYBRID_COLBERT_SCORE_BATCH = 4096  # corpus chunks per query in colbert matmul

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
        return {"rank": None, "RR": 0.0, "hit1": 0, "hit5": 0, "hit10": 0, "hit20": 0}
    return {"rank": rank, "RR": 1.0 / rank,
            "hit1": 1 if rank == 1 else 0,
            "hit5": 1 if rank <= 5 else 0,
            "hit10": 1 if rank <= 10 else 0,
            "hit20": 1 if rank <= 20 else 0}


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
        corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", show_progress=True)
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
        retriever.index(corpus_tokens, show_progress=True)
        log(f"  index built in {time.time() - t0:.1f}s")
        if index_dir.exists():
            import shutil
            shutil.rmtree(index_dir)
        retriever.save(str(index_dir), corpus=corpus_texts, show_progress=False)
        index_sig_path.write_text(json.dumps({"sig": current_sig, "n_corpus": n_corpus}))
        log(f"  cached → {index_dir} (sig={current_sig}, n_corpus={n_corpus})")

    t0 = time.time()
    query_tokens = bm25s.tokenize(queries, stopwords="en", show_progress=True)
    BM25_BATCH = 1000
    n_batches = (len(queries) + BM25_BATCH - 1) // BM25_BATCH
    progress_every = max(1, (n_batches + 19) // 20)
    log(f"  BM25 retrieval start: {len(queries)} queries in {n_batches} batches")
    results = [None] * len(queries)
    topk_buffer: list[np.ndarray] = [] if save_topk_path is not None else None

    def _slice_tok(tok, s, e):
        return type(tok)(tok.ids[s:e], tok.vocab)

    for batch_num, s in enumerate(range(0, len(queries), BM25_BATCH), start=1):
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
        if batch_num % progress_every == 0 or batch_num == n_batches:
            log(f"  BM25 retrieval progress: {e}/{len(queries)} queries "
                f"({batch_num}/{n_batches} batches), "
                f"elapsed={time.time() - t0:.1f}s")
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
    HF_CACHE = "/home/wlia0047/hj82/wenyu/hf_cache"
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
# Dense bi-encoder (GTE-base only)
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
        corpus_embeds = model.encode(corpus_texts, batch_size=128, show_progress_bar=True,
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
        q_embeds = model.encode(queries, batch_size=512, show_progress_bar=True,
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
    BATCH = 2000
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
# BGE-M3 dense retriever (FlagEmbedding, 1024d)
# ===========================================================================
def bge_m3_retrieve(queries: list[str], corpus_texts: list[str],
                    target_indices: np.ndarray, *,
                    corpus_sig: str, query_sig: str,
                    save_topk_path: Path | None = None,
                    topk_k: int = TOPK_SAVE_K) -> tuple[list[dict], np.ndarray]:
    from FlagEmbedding import BGEM3FlagModel

    log("\n=== bge_m3 (dense, BAAI/bge-m3) ===")
    cache_dir = EMBED_CACHE_DIR / "bge_m3"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # 2026-09-26: dense cache files renamed to corpus_dense.npy so the same
    # artifacts can be shared with bge_m3_hybrid (which also needs dense).
    # Legacy corpus_embeds.npy / query_embeds.npy are still honored on read.
    corpus_cache = cache_dir / "corpus_dense.npy"
    query_cache = cache_dir / "query_dense.npy"
    legacy_corpus = cache_dir / "corpus_embeds.npy"
    legacy_query = cache_dir / "query_embeds.npy"
    if not corpus_cache.exists() and legacy_corpus.exists():
        corpus_cache = legacy_corpus
    if not query_cache.exists() and legacy_query.exists():
        query_cache = legacy_query
    corpus_sig_file = _sig_path_for(corpus_cache)
    query_sig_file = _sig_path_for(query_cache)

    model = BGEM3FlagModel(
        "BAAI/bge-m3", use_fp16=True, devices=["cuda:0"],
        cache_dir=str(BGE_M3_HF_CACHE),
    )

    def encode(texts: list[str], batch_size: int) -> np.ndarray:
        out = model.encode(
            texts, batch_size=batch_size, max_length=512,
            return_dense=True, return_sparse=False, return_colbert_vecs=False,
        )
        vecs = np.asarray(out["dense_vecs"], dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-12)

    corpus_embeds = None
    if corpus_cache.exists() and corpus_sig_file.exists() and corpus_sig_file.read_text().strip() == corpus_sig:
        corpus_embeds = np.load(corpus_cache)
        log(f"  ✓ corpus embeds cache ({corpus_embeds.shape}, sig={corpus_sig})")
    if corpus_embeds is None:
        log(f"  encoding {len(corpus_texts)} corpus with BGE-M3 (batch={BGE_M3_CORPUS_BATCH})...")
        corpus_embeds = encode(corpus_texts, BGE_M3_CORPUS_BATCH)
        np.save(corpus_cache, corpus_embeds)
        corpus_sig_file.write_text(corpus_sig)
        log(f"  cached → {corpus_cache} (shape={corpus_embeds.shape}, sig={corpus_sig})")

    q_embeds = None
    if query_cache.exists() and query_sig_file.exists() and query_sig_file.read_text().strip() == query_sig:
        candidate = np.load(query_cache)
        if candidate.shape[0] == len(queries):
            q_embeds = candidate
            log(f"  ✓ query embeds cache ({q_embeds.shape}, sig={query_sig})")
    if q_embeds is None:
        log(f"  encoding {len(queries)} queries with BGE-M3 (batch={BGE_M3_QUERY_BATCH})...")
        q_embeds = encode(queries, BGE_M3_QUERY_BATCH)
        np.save(query_cache, q_embeds)
        query_sig_file.write_text(query_sig)
        log(f"  cached → {query_cache} (shape={q_embeds.shape}, sig={query_sig})")

    del model
    torch.cuda.empty_cache()
    corpus_gpu = torch.from_numpy(corpus_embeds).cuda()
    query_gpu = torch.from_numpy(q_embeds).cuda()
    targets_gpu = torch.as_tensor(target_indices, device="cuda", dtype=torch.long)
    results: list[dict] = []
    topk_rows: list[np.ndarray] = []
    t0 = time.time()
    for start in range(0, len(queries), 512):
        end = min(start + 512, len(queries))
        scores_batch = query_gpu[start:end] @ corpus_gpu.T
        for j in range(end - start):
            scores = scores_batch[j]
            target = int(targets_gpu[start + j].item())
            if target < 0:
                results.append(rr_hit_from_rank(-1))
            else:
                rank = int((scores > scores[target]).sum().item()) + 1
                results.append(rr_hit_from_rank(rank))
            if save_topk_path is not None:
                row = torch.topk(scores, k=min(topk_k, scores.shape[0])).indices.cpu().numpy().astype(np.int32)
                if len(row) < topk_k:
                    row = np.pad(row, (0, topk_k-len(row)), constant_values=-1)
                topk_rows.append(row)
        del scores_batch
    del corpus_gpu, query_gpu, targets_gpu
    torch.cuda.empty_cache()
    log(f"  matmul + ranks done in {time.time() - t0:.1f}s")
    if save_topk_path is not None:
        save_topk_path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.stack(topk_rows) if topk_rows else np.zeros((0, topk_k), dtype=np.int32)
        np.savez_compressed(save_topk_path, topk_asins=arr)
        log(f"  saved top-{topk_k} → {save_topk_path} (shape={arr.shape})")
    return results, q_embeds


# ===========================================================================
# BGE-M3 hybrid (dense + sparse + colbert) retriever
# 2026-09-26: implements full multi-vector + lexical fusion (BGE-M3 native).
# Cache layout under cache_dir/bge_m3/:
#   corpus_dense.npy          (N, 1024) float32  ← shared with bge_m3 dense
#   query_dense.npy           (Q, 1024) float32
#   corpus_sparse.npz         {vocab:(V,), data:(nnz,) float16, indices:(nnz,) int32, indptr:(N+1,) int32}
#   query_sparse.npz          (same layout)
#   corpus_colbert.npz        {data:(sum_L, 1024) float16, lengths:(N,) int32}
#   query_colbert.npz         (same layout)
# Scoring: dense cosine + w_s·sparse ip + w_c·colbert max-sim sum (ColBERT formula)
# ===========================================================================
def _bge_m3_encode_hybrid(model, texts: list[str], batch_size: int) -> dict:
    """Run BGE-M3 returning dense + sparse + colbert outputs (single pass)."""
    out = model.encode(
        texts, batch_size=batch_size, max_length=512,
        return_dense=True, return_sparse=True, return_colbert_vecs=True,
    )
    dense = np.asarray(out["dense_vecs"], dtype=np.float32)
    dense = dense / np.linalg.norm(dense, axis=1, keepdims=True).clip(min=1e-12)
    sparse = out["lexical_weights"]  # list[defaultdict[int,float]]
    colbert = out["colbert_vecs"]    # list[(L, 1024) float32]
    return {"dense": dense, "sparse": sparse, "colbert": colbert}


def _sparse_to_csr(sparse_list: list, vocab: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack variable-length sparse dicts into CSR (data/indices/indptr).

    2026-09-26: indices now store the REAL BGE-M3 token ids (not column indices
    into vocab). The vocab argument is kept for backward compatibility but is
    ignored — we always produce a CSR whose column space is [0, max(token_id)+1).
    Downstream sparse_mm uses the original BGE-M3 vocab (250002) so this is
    consistent across corpus and queries.
    """
    indptr = np.zeros(len(sparse_list) + 1, dtype=np.int64)
    cols_list: list[int] = []
    vals_list: list[float] = []
    for di, d in enumerate(sparse_list):
        for tok, w in d.items():
            cols_list.append(int(tok))  # real token id
            vals_list.append(float(w))
        indptr[di + 1] = len(cols_list)
    if cols_list:
        cols = np.array(cols_list, dtype=np.int32)
        vals = np.array(vals_list, dtype=np.float16)
    else:
        cols = np.zeros(0, dtype=np.int32)
        vals = np.zeros(0, dtype=np.float16)
    return vals, cols, indptr.astype(np.int32)


def _colbert_to_packed(colbert_list: list, max_len: int = BGE_M3_HYBRID_COLBERT_MAXLEN
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Pack variable-length colbert vecs → (sum_L, 1024) float16 + lengths (N,) int32."""
    flat = []
    lens = np.zeros(len(colbert_list), dtype=np.int32)
    for i, arr in enumerate(colbert_list):
        L = min(arr.shape[0], max_len)
        lens[i] = L
        if L > 0:
            flat.append(arr[:L].astype(np.float16))
    if flat:
        packed = np.concatenate(flat, axis=0)
    else:
        packed = np.zeros((0, 1024), dtype=np.float16)
    return packed, lens


def _sparse_scores_chunked(
    q_sparse_vocab: np.ndarray, q_sparse_indptr: np.ndarray,
    q_sparse_indices: np.ndarray, q_sparse_data: np.ndarray,
    c_sparse_vocab: np.ndarray, c_sparse_indptr: np.ndarray,
    c_sparse_indices: np.ndarray, c_sparse_data: np.ndarray,
    chunk: int = 4096,
) -> np.ndarray:
    """Compute sparse ip (Q x N) as CPU CSR @ CSR with chunked corpus output.

    BGE-M3 sparse: each doc has ~25 weighted tokens, each query ~10. So a
    per-query loop over corpus docs only needs to check the token intersection
    of (q.vocab ∩ d.vocab). Use a hash-set per query for O(N * avg_tokens)
    which is ~5M ops for baby (2714 * 217710 * 25 = 1.5e10 — still too slow).

    Better: build corpus token → list-of-doc-weights dict once, then for each
    query token do a token-bucket dot product. Total ops: O(sum_q L_q * sum_d
    |bucket_d_tok|) which is ~Q * 10 * (N * 25 / V_active) — tractable.
    """
    # 1. Build per-corpus sparse dicts as plain python for fast inner loop
    # (Corpus never changes during one run.)
    corpus_dicts: list[dict[int, float]] = []
    for di in range(len(c_sparse_indptr) - 1):
        s = int(c_sparse_indptr[di])
        e = int(c_sparse_indptr[di + 1])
        idx = c_sparse_indices[s:e].tolist()
        w = c_sparse_data[s:e].astype(np.float32).tolist()
        corpus_dicts.append({idx[k]: w[k] for k in range(len(idx))})

    # 2. Build inverse index: corpus token id → list of (doc_id, weight)
    inv: dict[int, list[tuple[int, float]]] = {}
    for di, d in enumerate(corpus_dicts):
        for tok, w in d.items():
            inv.setdefault(int(tok), []).append((di, w))

    # 3. Query loop: for each query token, accumulate w_q * w_d over its bucket
    Q = len(q_sparse_indptr) - 1
    N = len(corpus_dicts)
    out = np.zeros((Q, N), dtype=np.float32)
    # Process in chunks of docs to bound memory
    chunk_doc_ids: dict[int, list[float]] = {}
    chunk_dense: np.ndarray  # current chunk scores
    for qi in range(Q):
        s = int(q_sparse_indptr[qi])
        e = int(q_sparse_indptr[qi])
        e = int(q_sparse_indptr[qi + 1])
        q_idx = q_sparse_indices[s:e].astype(np.int64)
        q_w = q_sparse_data[s:e].astype(np.float32)
        if q_w.shape[0] == 0:
            continue
        scores_q = out[qi]
        # For each query token, fold its corpus bucket into scores_q
        # The bucket is a sorted list of (doc_id, w_d) — accumulate
        for ti, wq in zip(q_idx.tolist(), q_w.tolist()):
            bucket = inv.get(int(ti))
            if not bucket:
                continue
            # vectorized: convert bucket to numpy arrays
            arr = np.asarray(bucket, dtype=np.float32)  # (k, 2)
            di_arr = arr[:, 0].astype(np.int64)
            wd_arr = arr[:, 1]
            np.add.at(scores_q, di_arr, wq * wd_arr)
    return out


def _save_sparse_npz(path: Path, data: np.ndarray, indices: np.ndarray, indptr: np.ndarray,
                     vocab: np.ndarray):
    np.savez_compressed(path, vocab=vocab, data=data, indices=indices, indptr=indptr)


def _save_colbert_npz(path: Path, packed: np.ndarray, lengths: np.ndarray):
    np.savez_compressed(path, data=packed, lengths=lengths)


def _load_colbert_npz(path: Path, dim: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    packed = z["data"]
    if packed.dtype != np.float16:
        packed = packed.astype(np.float16, copy=False)
    if packed.ndim == 1:
        packed = packed.reshape(-1, dim)
    return packed, z["lengths"].astype(np.int32)


def _colbert_maxsim_chunked(
    q_packed: torch.Tensor, q_lens: torch.Tensor,
    d_packed: torch.Tensor, d_lens: torch.Tensor,
    chunk_size: int = BGE_M3_HYBRID_COLBERT_SCORE_BATCH,
    normalize: bool = True,
) -> torch.Tensor:
    """Score flat-packed query/document token vectors with ColBERT MaxSim."""
    B = q_lens.numel()
    N = d_lens.numel()
    device = q_packed.device
    if B == 0 or N == 0:
        return torch.zeros(B, N, dtype=torch.float32, device=device)

    q_lens = q_lens.to(device=device, dtype=torch.long)
    d_lens = d_lens.to(device=device, dtype=torch.long)
    Lq = int(q_lens.max().item())
    Ld = int(d_lens.max().item())
    if Lq == 0 or Ld == 0:
        return torch.zeros(B, N, dtype=torch.float32, device=device)

    q_offsets = torch.cat((q_lens.new_zeros(1), q_lens.cumsum(0)))
    d_offsets = torch.cat((d_lens.new_zeros(1), d_lens.cumsum(0)))
    q_pos = torch.arange(Lq, device=device)
    q_valid = q_pos.unsqueeze(0) < q_lens.unsqueeze(1)
    q_idx = (q_offsets[:-1, None] + q_pos).clamp_(max=q_packed.shape[0] - 1)
    q_pad = q_packed[q_idx].float()
    if normalize:
        q_pad = q_pad / q_pad.norm(dim=-1, keepdim=True).clamp(min=1e-9)

    out = torch.zeros(B, N, dtype=torch.float32, device=device)
    d_pos = torch.arange(Ld, device=device)
    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        lens = d_lens[s:e]
        d_valid = d_pos.unsqueeze(0) < lens.unsqueeze(1)
        d_idx = (d_offsets[s:e, None] + d_pos).clamp_(max=d_packed.shape[0] - 1)
        d_pad = d_packed[d_idx].float()
        if normalize:
            d_pad = d_pad / d_pad.norm(dim=-1, keepdim=True).clamp(min=1e-9)

        # (B, Lq, C, Ld): query-token/document-token dot products.
        scores = torch.einsum("btf,clf->btcl", q_pad, d_pad)
        scores.masked_fill_(~d_valid[None, None, :, :], -1e4)
        maxsim = scores.max(dim=-1).values
        maxsim.masked_fill_(~q_valid[:, :, None], 0.0)
        chunk_scores = maxsim.sum(dim=1)
        chunk_scores.masked_fill_(lens[None, :] == 0, 0.0)
        out[:, s:e] = chunk_scores
    return out


def _colbert_maxsim_topk_only(
    q_packed: torch.Tensor, q_lens: torch.Tensor,
    d_packed: torch.Tensor, d_lens: torch.Tensor,
    target_indices: torch.Tensor,
    chunk_size: int = BGE_M3_HYBRID_COLBERT_SCORE_BATCH,
    topk_k: int = 100,
) -> torch.Tensor:
    """Compute ColBERT max-sim score only for the doc range that contains each
    target, then take top-k around it. Cheaper than full N.
    For each query, we compute scores vs the union of {target} ∪ topk neighbors
    from dense matmul — but here we just compute scores vs ALL corpus (chunked)
    which is what we already do. Use the chunked version."""
    raise NotImplementedError("use _colbert_maxsim_chunked instead")


def bge_m3_hybrid_retrieve(queries: list[str], corpus_texts: list[str],
                           target_indices: np.ndarray, *,
                           corpus_sig: str, query_sig: str,
                           save_topk_path: Path | None = None,
                           topk_k: int = TOPK_SAVE_K,
                           weights: tuple = (1.0, 0.3, 1.0)) -> tuple[list[dict], np.ndarray]:
    from FlagEmbedding import BGEM3FlagModel

    w_dense, w_sparse, w_colbert = weights
    log(f"\n=== bge_m3_hybrid (dense+sparse+colbert, weights={weights}) ===")
    cache_dir = EMBED_CACHE_DIR / "bge_m3"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ---------- shared dense cache ----------
    corpus_dense = cache_dir / "corpus_dense.npy"
    query_dense = cache_dir / "query_dense.npy"
    legacy_corpus = cache_dir / "corpus_embeds.npy"
    legacy_query = cache_dir / "query_embeds.npy"
    if not corpus_dense.exists() and legacy_corpus.exists():
        corpus_dense = legacy_corpus
    if not query_dense.exists() and legacy_query.exists():
        query_dense = legacy_query
    corpus_dense_sig = _sig_path_for(corpus_dense)
    query_dense_sig = _sig_path_for(query_dense)

    sparse_corpus_npz = cache_dir / "corpus_sparse.npz"
    sparse_query_npz = cache_dir / "query_sparse.npz"
    sparse_corpus_sig = _sig_path_for(sparse_corpus_npz)
    sparse_query_sig = _sig_path_for(sparse_query_npz)
    colbert_corpus_npz = cache_dir / "corpus_colbert.npz"
    colbert_query_npz = cache_dir / "query_colbert.npz"
    colbert_corpus_sig = _sig_path_for(colbert_corpus_npz)
    colbert_query_sig = _sig_path_for(colbert_query_npz)

    model = BGEM3FlagModel(
        "BAAI/bge-m3", use_fp16=True, devices=["cuda:0"],
        cache_dir=str(BGE_M3_HF_CACHE),
    )

    # ---------- 1. encode corpus ----------
    # 2026-09-26: stream the corpus in fixed-size batches and write each
    # artifact to disk + free Python references immediately. The original
    # full-corpus BGEM3FlagModel.encode call kept 217710 × (L, 1024) fp32
    # colbert tensors in RAM until the function returned, blowing past the
    # 245 GB cgroup limit at ~83% corpus (OOM-killed at RSS=241 GB).
    have_dense = (corpus_dense.exists() and corpus_dense_sig.exists()
                  and corpus_dense_sig.read_text().strip() == corpus_sig)
    have_sparse = (sparse_corpus_npz.exists() and sparse_corpus_sig.exists()
                   and sparse_corpus_sig.read_text().strip() == corpus_sig)
    have_colbert = (colbert_corpus_npz.exists() and colbert_corpus_sig.exists()
                    and colbert_corpus_sig.read_text().strip() == corpus_sig)
    if have_dense and have_sparse and have_colbert:
        log(f"  ✓ corpus cache (dense+sparse+colbert) all hit, sig={corpus_sig}")
        corpus_dense_arr = np.load(corpus_dense)
        sparse_corpus_z = np.load(sparse_corpus_npz)
        sparse_corpus_vocab = sparse_corpus_z["vocab"].astype(np.int64)
        sparse_corpus_data = sparse_corpus_z["data"].astype(np.float32)
        sparse_corpus_indices = sparse_corpus_z["indices"].astype(np.int32)
        sparse_corpus_indptr = sparse_corpus_z["indptr"].astype(np.int64)
        colbert_corpus_packed, colbert_corpus_lens = _load_colbert_npz(colbert_corpus_npz)
    else:
        import gc
        BS = BGE_M3_HYBRID_CORPUS_BATCH
        log(f"  encoding {len(corpus_texts)} corpus with BGE-M3 hybrid "
            f"(stream batch={BS}, max_len={BGE_M3_HYBRID_COLBERT_MAXLEN})...")
        # Sparse corpus strategy: maintain (a) vocab: int id (the real BGE-M3
        # token id, 0..250001), (b) per-batch local CSR — at end we write a
        # single npz with global indices into the vocab. Vocab stays as the
        # original BGE-M3 token ids (no remap) so that downstream sparse mm
        # can keep using the original token id space.
        corpus_dense_arr = (np.load(corpus_dense)
                            if have_dense
                            else np.zeros((len(corpus_texts), 1024), dtype=np.float32))
        if have_sparse:
            sparse_corpus_z = np.load(sparse_corpus_npz)
            sparse_corpus_vocab = sparse_corpus_z["vocab"].astype(np.int64)
            sparse_corpus_data = sparse_corpus_z["data"].astype(np.float16)
            sparse_corpus_indices = sparse_corpus_z["indices"].astype(np.int32)
            sparse_corpus_indptr = sparse_corpus_z["indptr"].astype(np.int64)
            # vocab field now holds sorted unique BGE-M3 token ids actually used;
            # indices are real token ids (not column indices into vocab). The CSR
            # column space is [0, max_token_id+1) which downstream allocates.
        else:
            sparse_corpus_vocab = np.zeros(0, dtype=np.int64)
            sparse_corpus_data = np.zeros(0, dtype=np.float16)
            sparse_corpus_indices = np.zeros(0, dtype=np.int32)
            sparse_corpus_indptr = np.zeros(0, dtype=np.int64)
            # per-doc nnz counts to build indptr at the end
            sparse_doc_nnz = np.zeros(len(corpus_texts), dtype=np.int64)
        if have_colbert:
            colbert_corpus_packed, colbert_corpus_lens = _load_colbert_npz(colbert_corpus_npz)
        else:
            colbert_tmp_packed = []
            colbert_tmp_lens = np.zeros(len(corpus_texts), dtype=np.int32)
        # main encode loop
        t_corpus = time.time()
        for b_start in range(0, len(corpus_texts), BS):
            b_end = min(b_start + BS, len(corpus_texts))
            chunk_texts = corpus_texts[b_start:b_end]
            out = _bge_m3_encode_hybrid(model, chunk_texts, BS)
            chunk_dense = out["dense"]
            chunk_sparse = out["sparse"]
            chunk_colbert = out["colbert"]
            if not have_dense:
                corpus_dense_arr[b_start:b_end] = chunk_dense
            if not have_sparse:
                # collect (token_id, weight, doc_local_idx) → append to flat arrays
                rows_local = []
                cols_tok = []
                vals_w = []
                for i, d in enumerate(chunk_sparse):
                    for tok, w in d.items():
                        ti = int(tok)
                        rows_local.append(b_start + i)
                        cols_tok.append(ti)  # store real BGE-M3 token id directly
                        vals_w.append(float(w))
                if rows_local:
                    new_rows = np.array(rows_local, dtype=np.int64)
                    new_cols = np.array(cols_tok, dtype=np.int32)
                    new_vals = np.array(vals_w, dtype=np.float16)
                    # use np.add.at to compute nnz per doc
                    np.add.at(sparse_doc_nnz, new_rows, 1)
                    sparse_corpus_indices = np.concatenate([sparse_corpus_indices, new_cols])
                    sparse_corpus_data = np.concatenate([sparse_corpus_data, new_vals])
                del rows_local, cols_tok, vals_w, new_rows, new_cols, new_vals
            if not have_colbert:
                packed_chunk, lens_chunk = _colbert_to_packed(chunk_colbert, BGE_M3_HYBRID_COLBERT_MAXLEN)
                colbert_tmp_packed.append(packed_chunk)
                colbert_tmp_lens[b_start:b_end] = lens_chunk
            del out, chunk_dense, chunk_sparse, chunk_colbert
            if b_start % (BS * 8) == 0:
                elapsed = time.time() - t_corpus
                eta = elapsed / max(1, b_end) * (len(corpus_texts) - b_end)
                log(f"    encoded {b_end}/{len(corpus_texts)} "
                    f"({100 * b_end / len(corpus_texts):.1f}%) "
                    f"rate={b_end / elapsed:.0f}/s eta={eta:.0f}s")
            gc.collect()
        # finalize sparse: build indptr via cumsum, vocab = sorted unique token ids
        if not have_sparse:
            sparse_corpus_vocab = np.unique(sparse_corpus_indices).astype(np.int64)
            sparse_corpus_indptr = np.zeros(len(corpus_texts) + 1, dtype=np.int64)
            np.cumsum(sparse_doc_nnz, out=sparse_corpus_indptr[1:])
            _save_sparse_npz(sparse_corpus_npz,
                             sparse_corpus_data,
                             sparse_corpus_indices,
                             sparse_corpus_indptr.astype(np.int32),
                             sparse_corpus_vocab)
            sparse_corpus_sig.write_text(corpus_sig)
            log(f"  cached → {sparse_corpus_npz} (vocab={sparse_corpus_vocab.shape}, nnz={sparse_corpus_data.shape})")
            del sparse_doc_nnz
        if not have_dense:
            np.save(corpus_dense, corpus_dense_arr)
            corpus_dense_sig.write_text(corpus_sig)
            log(f"  cached → {corpus_dense} (shape={corpus_dense_arr.shape})")
        if not have_colbert:
            colbert_corpus_packed = np.concatenate(colbert_tmp_packed, axis=0)
            colbert_corpus_lens = colbert_tmp_lens
            _save_colbert_npz(colbert_corpus_npz, colbert_corpus_packed, colbert_corpus_lens)
            colbert_corpus_sig.write_text(corpus_sig)
            log(f"  cached → {colbert_corpus_npz} (n={len(colbert_corpus_lens)}, total_tokens={colbert_corpus_packed.shape})")
            del colbert_tmp_packed
        gc.collect()

    # ---------- 2. encode queries ----------
    have_qd = (query_dense.exists() and query_dense_sig.exists()
               and query_dense_sig.read_text().strip() == query_sig)
    have_qs = (sparse_query_npz.exists() and sparse_query_sig.exists()
               and sparse_query_sig.read_text().strip() == query_sig)
    have_qc = (colbert_query_npz.exists() and colbert_query_sig.exists()
               and colbert_query_sig.read_text().strip() == query_sig)
    if have_qd and have_qs and have_qc:
        log(f"  ✓ query cache (dense+sparse+colbert) all hit, sig={query_sig}")
        q_dense = np.load(query_dense)
        sq = np.load(sparse_query_npz)
        q_sparse_vocab = sq["vocab"].astype(np.int64)
        q_sparse_data = sq["data"].astype(np.float32)
        q_sparse_indices = sq["indices"].astype(np.int32)
        q_sparse_indptr = sq["indptr"].astype(np.int64)
        q_colbert_packed, q_colbert_lens = _load_colbert_npz(colbert_query_npz)
    else:
        log(f"  encoding {len(queries)} queries with BGE-M3 hybrid "
            f"(batch={BGE_M3_HYBRID_QUERY_BATCH})...")
        qout = _bge_m3_encode_hybrid(model, queries, BGE_M3_HYBRID_QUERY_BATCH)
        q_dense = qout["dense"]
        q_sparse_raw = qout["sparse"]
        q_colbert_raw = qout["colbert"]
        if not have_qd:
            np.save(query_dense, q_dense)
            query_dense_sig.write_text(query_sig)
            log(f"  cached → {query_dense} (shape={q_dense.shape})")
        if not have_qs:
            qvocab = np.array(sorted({int(t) for d in q_sparse_raw for t in d.keys()}),
                              dtype=np.int64)
            qdata, qindices, qindptr = _sparse_to_csr(q_sparse_raw, qvocab)
            _save_sparse_npz(sparse_query_npz, qdata, qindices, qindptr.astype(np.int32), qvocab)
            sparse_query_sig.write_text(query_sig)
            log(f"  cached → {sparse_query_npz} (vocab={qvocab.shape}, nnz={qdata.shape})")
            q_sparse_vocab, q_sparse_data, q_sparse_indices, q_sparse_indptr = (
                qvocab, qdata.astype(np.float32), qindices, qindptr)
        if not have_qc:
            qpacked, qlens = _colbert_to_packed(q_colbert_raw, BGE_M3_HYBRID_COLBERT_MAXLEN)
            _save_colbert_npz(colbert_query_npz, qpacked, qlens)
            colbert_query_sig.write_text(query_sig)
            log(f"  cached → {colbert_query_npz} (n={len(qlens)}, total_tokens={qpacked.shape})")
            q_colbert_packed, q_colbert_lens = qpacked, qlens
        del q_sparse_raw, q_colbert_raw

    del model
    torch.cuda.empty_cache()

    # ---------- 3. hybrid scoring ----------
    # Vocabularies are kept in their CSR form (union vocab is reconstructed from
    # corpus ∪ query vocab in sparse_scores.py). We don't try to build a dense
    # (N, V) sparse matrix — for baby V is 250002 → 109 GB, doesn't fit.
    # Instead, we use torch.sparse.mm on (N, V) sparse_csr @ (V, B) dense per
    # chunk which is feasible because torch sparse matmul keeps CSR sparsity
    # and emits dense B-vector outputs.
    # Per BGE-M3 sparse behavior: each doc has ~25 nonzero tokens out of 250002;
    # CSR @ dense yields 217710 * 25 * B = 5.4e6 * B FLOPs per chunk, fast.
    log(f"  corpus sparse nnz={int(sparse_corpus_data.shape[0])}, "
        f"query sparse nnz={int(q_sparse_data.shape[0])}")
    # build GPU sparse tensors (re-using the cached CSR arrays)
    corpus_sparse_t = torch.sparse_csr_tensor(
        torch.from_numpy(sparse_corpus_indptr.astype(np.int64)),
        torch.from_numpy(sparse_corpus_indices.astype(np.int64)),
        torch.from_numpy(sparse_corpus_data),
        size=(len(corpus_texts), 250002), dtype=torch.float32,
    ).cuda()
    q_dense_t = torch.from_numpy(q_dense).cuda()
    cd_t = torch.from_numpy(corpus_dense_arr).cuda()
    targets_gpu = torch.as_tensor(target_indices, device="cuda", dtype=torch.long)
    q_packed_t = torch.from_numpy(q_colbert_packed).cuda()
    q_lens_t = torch.from_numpy(q_colbert_lens).cuda()
    d_packed_t = torch.from_numpy(colbert_corpus_packed).cuda()
    d_lens_t = torch.from_numpy(colbert_corpus_lens).cuda()

    # Re-index query sparse columns to a contiguous dense V' representation.
    # We pack each query's tokens into a dense (Q, V_qmax) fp16 mat where
    # V_qmax = max query token id + 1, then do sparse @ dense.
    # But for batched mm we need (V, B) dense — and V=250002, B=64 → 64 MB.
    # This fits easily. So pack queries per batch.
    # query_sparse_t is CSR (Q, 250002). To batch mm we need each query as a
    # dense vector in (250002, B). Use sparse_csr_tensor.to_dense() in slices.

    results: list[dict] = []
    topk_rows: list[np.ndarray] = []
    t0 = time.time()
    SCORE_BATCH = 64  # queries per chunk to bound memory
    q_offsets_full = torch.cat([torch.zeros(1, dtype=torch.long, device="cuda"),
                                q_lens_t.cumsum(0).long()])
    for start in range(0, len(queries), SCORE_BATCH):
        end = min(start + SCORE_BATCH, len(queries))
        B = end - start
        # dense cosine
        sd = q_dense_t[start:end] @ cd_t.T  # (B, N) fp32
        # sparse ip: corpus_sparse.T (sparse, 250002×N) @ query_dense_v (250002×B, dense)
        # Build query dense v by to_dense on the query CSR slice — too costly
        # (250002 * 64 fp32 = 64 MB per batch, OK). But to_dense is dense
        # extraction — actually fine.
        q_dense_v = torch.zeros(250002, B, dtype=torch.float32, device="cuda")
        # Place query sparse columns at their original positions in the 250002
        # vector space, then transpose → (250002, B). Actually we need:
        #   scores_sparse = q_dense_v.T (B, 250002) @ corpus_sparse.T (sparse) — NOT supported.
        # Equivalent: scores_sparse = corpus_sparse (N, 250002) @ q_dense_v (250002, B)
        # Sparse @ dense works in PyTorch.
        # We need q_dense_v: for each query b, set q_dense_v[token_id, b] = weight.
        for j in range(B):
            qi = start + j
            s = int(sparse_corpus_indptr[0])  # not used
            # query CSR slice
            qs = q_sparse_indptr[qi]
            qe = q_sparse_indptr[qi + 1]
            tok_ids = q_sparse_indices[qs:qe]
            tok_vals = q_sparse_data[qs:qe]
            if tok_vals.shape[0] > 0:
                q_dense_v[tok_ids, j] = torch.from_numpy(tok_vals.astype(np.float32)).cuda()
        # sparse @ dense: corpus_sparse (N, 250002) @ q_dense_v (250002, B) → (N, B)
        ss = torch.sparse.mm(corpus_sparse_t, q_dense_v).T  # (B, N) fp32
        # colbert max-sim
        sl = q_offsets_full[start].item()
        sr = q_offsets_full[end].item()
        q_chunk_packed = q_packed_t[sl:sr]
        q_chunk_lens = q_lens_t[start:end]
        sc = _colbert_maxsim_chunked(
            q_chunk_packed, q_chunk_lens, d_packed_t, d_lens_t,
            chunk_size=BGE_M3_HYBRID_COLBERT_SCORE_BATCH,
        )  # (B, N)
        # z-normalize each component for fair weighting
        def _zscore(x: torch.Tensor) -> torch.Tensor:
            mu = x.mean(dim=1, keepdim=True)
            sd = x.std(dim=1, keepdim=True).clamp(min=1e-6)
            return (x - mu) / sd
        sd_z = _zscore(sd)
        ss_z = _zscore(ss)
        sc_z = _zscore(sc)
        combined = w_dense * sd_z + w_sparse * ss_z + w_colbert * sc_z
        for j in range(B):
            scores = combined[j]
            target = int(targets_gpu[start + j].item())
            if target < 0:
                results.append(rr_hit_from_rank(-1))
            else:
                rank = int((scores > scores[target]).sum().item()) + 1
                results.append(rr_hit_from_rank(rank))
            if save_topk_path is not None:
                row = torch.topk(scores, k=min(topk_k, scores.shape[0])).indices.cpu().numpy().astype(np.int32)
                if len(row) < topk_k:
                    row = np.pad(row, (0, topk_k - len(row)), constant_values=-1)
                topk_rows.append(row)
        del sd, ss, sc, combined, sd_z, ss_z, sc_z, q_dense_v
        torch.cuda.empty_cache()
    log(f"  hybrid matmul + ranks done in {time.time() - t0:.1f}s")
    if save_topk_path is not None:
        save_topk_path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.stack(topk_rows) if topk_rows else np.zeros((0, topk_k), dtype=np.int32)
        np.savez_compressed(save_topk_path, topk_asins=arr)
        log(f"  saved top-{topk_k} → {save_topk_path} (shape={arr.shape})")
    # Return q_dense (1024) for canonical sim09 reference parity with bge_m3.
    return results, q_dense


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
    HF_CACHE = "/home/wlia0047/hj82/wenyu/hf_cache"
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

    q_embeds is the canonical sim09 reference embedding (GTE-base, 768d),
    shared across all retrievers per current model policy. BM25/SPLADE have no
    pooled embedding and borrow GTE-base for query-query similarity.
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
def _load_gte_q_embeds_for_volatility(query_records: list[dict], selection_sig: str) -> np.ndarray:
    gte_q_cache = EMBED_CACHE_DIR / "gte_base" / "query_embeds.npy"
    gte_sig_file = _sig_path_for(gte_q_cache)
    if gte_q_cache.exists() and gte_sig_file.exists():
        cached_sig = gte_sig_file.read_text().strip()
        if cached_sig == selection_sig:
            candidate = np.load(gte_q_cache)
            if candidate.ndim == 2 and candidate.shape == (len(query_records), 768):
                log(f"  ✓ loaded GTE-base q_embeds from disk ({candidate.shape}) "
                    f"for canonical sim09 reference")
                return candidate
            log(f"  ⚠ stale GTE-base q_embeds shape {candidate.shape}; re-encoding")
        else:
            log(f"  ⚠ stale GTE-base q_embeds signature ({cached_sig} != "
                f"{selection_sig}); re-encoding")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(
        "thenlper/gte-base",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    cached_gte = model.encode(
        [q["query"] for q in query_records], batch_size=512,
        show_progress_bar=False, convert_to_numpy=True,
        normalize_embeddings=True,
    )
    np.save(gte_q_cache, cached_gte)
    gte_sig_file.write_text(selection_sig)
    del model
    torch.cuda.empty_cache()
    log(f"  ✓ rebuilt GTE-base q_embeds ({cached_gte.shape})")
    return cached_gte

def _migrate_hit20_fields(query_records: list[dict]) -> None:
    for record in query_records:
        for retr_name in RETR_NAMES:
            rank = record.get(f"{retr_name}_rank")
            record[f"{retr_name}_hit20"] = int(rank is not None and rank <= 20)


def _remove_retired_retriever_fields(
        query_records: list[dict], previous_names: list[str] | set[str]) -> None:
    retired_names = set(previous_names) - set(RETR_NAMES)
    if not retired_names:
        return
    prefixes = tuple(f"{name}_" for name in retired_names)
    for record in query_records:
        for key in tuple(record):
            if key.startswith(prefixes):
                del record[key]
    log(f"  removed retired retriever fields: {sorted(retired_names)}")


def _finalize_from_cached_queries(
        query_records: list[dict], cached: dict, selection_sig: str,
        t_start: float) -> None:
    _remove_retired_retriever_fields(
        query_records, cached.get("config", {}).get("retrievers", []))
    _migrate_hit20_fields(query_records)
    cached["queries"] = query_records
    cached["config"]["retrievers"] = list(RETR_NAMES)
    with open(PER_QUERY_OUT, "w", encoding="utf-8") as f:
        json.dump(cached, f, ensure_ascii=False)
    asins_count = cached["config"]["corpus_size"]
    cached_gte = _load_gte_q_embeds_for_volatility(query_records, selection_sig)
    retr_q_embeds_cached = {n: cached_gte for n in RETR_NAMES}
    missing_topk = [n for n in RETR_NAMES
                    if not (TOPK_SAVE_DIR / f"{n}_top100.npz").exists()]
    if missing_topk:
        log(f"  ⚠ top-100 cache missing for {missing_topk}, "
            f"delete {PER_QUERY_OUT} to re-run retrievers and populate top-100 cache")
    _build_aggregates_and_save(query_records, asins_count, t_start,
                               retr_q_embeds_cached, "gte_base")


def _run_one_retriever(retr: dict, queries: list[str], corpus_texts: list[str],
                       target_indices: np.ndarray, corpus_sig: str,
                       query_sig: str) -> tuple[list[dict], np.ndarray | None]:
    kind = retr["kind"]
    topk_path = TOPK_SAVE_DIR / f"{retr['name']}_top100.npz"
    if kind == "sparse_lexical":
        return bm25_retrieve(queries, corpus_texts, target_indices,
                             save_topk_path=topk_path), None
    if kind == "sparse_learned":
        return splade_retrieve(queries, corpus_texts, target_indices,
                               corpus_sig=corpus_sig, query_sig=query_sig,
                               save_topk_path=topk_path), None
    if kind == "dense":
        return dense_retrieve(retr["name"], retr["hf_id"], queries,
                              target_indices, corpus_sig=corpus_sig,
                              query_sig=query_sig, save_topk_path=topk_path)
    if kind == "dense_bge_m3":
        return bge_m3_retrieve(
            queries, corpus_texts, target_indices,
            corpus_sig=corpus_sig, query_sig=query_sig,
            save_topk_path=topk_path,
        )
    if kind == "dense_bge_m3_hybrid":
        weights = (1.0, 0.3, 1.0)
        # Allow per-call override via env for sweeps (e.g., BGE_M3_W_S=0.5)
        try:
            wd = float(os.environ.get("BGE_M3_W_D", weights[0]))
            ws = float(os.environ.get("BGE_M3_W_S", weights[1]))
            wc = float(os.environ.get("BGE_M3_W_C", weights[2]))
            weights = (wd, ws, wc)
        except Exception:
            pass
        return bge_m3_hybrid_retrieve(
            queries, corpus_texts, target_indices,
            corpus_sig=corpus_sig, query_sig=query_sig,
            save_topk_path=topk_path, weights=weights,
        )
    if kind == "late_interaction":
        return colbertv2_retrieve(queries, corpus_texts, target_indices,
                                  corpus_sig=corpus_sig, query_sig=query_sig,
                                  save_topk_path=topk_path)
    raise ValueError(f"Unknown retriever kind: {kind}")


def main_task_body():
    t_start = time.time()
    log("=== Stage 5 unified multi-retriever (NO rerank) ===")
    log(f"  retrievers: {RETR_NAMES}")

    retr_filter_raw = os.environ.get("STAGE11_RETRIEVERS", "").strip()
    if retr_filter_raw:
        keep_names = {x.strip() for x in retr_filter_raw.split(",") if x.strip()}
        active_retrievers = [r for r in RETRIEVERS if r["name"] in keep_names]
        if not active_retrievers:
            raise ValueError(f"STAGE11_RETRIEVERS={retr_filter_raw!r} matched no retrievers")
        log(f"  STAGE11_RETRIEVERS={sorted(keep_names)} (incremental onto existing per_query)")
    else:
        active_retrievers = RETRIEVERS
        keep_names = set()

    # ---- 0. Cache check (selection signature) ----
    # If PER_QUERY_OUT already has results for the current selection,
    # skip the configured retrievers and go straight to aggregate.
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
        retr_filter_raw
        and cached is not None
        and cached.get("config", {}).get("signature") == selection_sig
    ):
        query_records = cached["queries"]
        existing_retrs = set(cached.get("config", {}).get("retrievers", []))
        todo = [r for r in active_retrievers if r["name"] not in existing_retrs]
        if not todo:
            log(f"  ✓ incremental cache hit: {sorted(keep_names)} already present, "
                f"rebuilding aggregates only")
            _finalize_from_cached_queries(query_records, cached, selection_sig, t_start)
            return
        log(f"  incremental add: running {[r['name'] for r in todo]} on "
            f"{len(query_records)} cached queries")
        if not ASIN_TO_DOC_CACHE.exists():
            raise FileNotFoundError(
                f"ASIN_TO_DOC_CACHE missing: {ASIN_TO_DOC_CACHE}. "
                f"Run `python build_asin_to_doc.py` first.")
        asin_to_doc = json.load(open(ASIN_TO_DOC_CACHE, encoding="utf-8"))
        asins = sorted(asin_to_doc.keys())
        asin_to_idx = {a: i for i, a in enumerate(asins)}
        corpus_texts = [asin_to_doc[a] for a in asins]
        queries = [r["query"] for r in query_records]
        target_indices = np.array([asin_to_idx.get(r["asin"], -1) for r in query_records])
        corpus_sig = _corpus_signature(asin_to_doc=asin_to_doc, meta_file=META_FILE)
        retr_results: dict[str, list[dict]] = {}
        for retr in todo:
            results, _ = _run_one_retriever(
                retr, queries, corpus_texts, target_indices, corpus_sig, selection_sig)
            retr_results[retr["name"]] = results
        for gi, r in enumerate(query_records):
            for n, res_list in retr_results.items():
                res = res_list[gi]
                r[f"{n}_rank"] = res["rank"]
                r[f"{n}_RR"] = res["RR"]
                r[f"{n}_hit1"] = res["hit1"]
                r[f"{n}_hit5"] = res["hit5"]
                r[f"{n}_hit10"] = res["hit10"]
                r[f"{n}_hit20"] = res["hit20"]
        _remove_retired_retriever_fields(query_records, existing_retrs)
        cached["config"]["retrievers"] = sorted(
            (existing_retrs & set(RETR_NAMES)) | set(retr_results)
        )
        cached["queries"] = query_records
        with open(PER_QUERY_OUT, "w", encoding="utf-8") as f:
            json.dump(cached, f, ensure_ascii=False)
        log(f"  wrote → {PER_QUERY_OUT} (added {list(retr_results)})")
        cached_gte = _load_gte_q_embeds_for_volatility(query_records, selection_sig)
        retr_q_embeds_full = {n: cached_gte for n in RETR_NAMES}
        _build_aggregates_and_save(
            query_records, len(asins), t_start, retr_q_embeds_full, "gte_base")
        return

    if (
        not retr_filter_raw
        and cached is not None
        and cached.get("config", {}).get("signature") == selection_sig
        and set(RETR_NAMES).issubset(set(cached.get("config", {}).get("retrievers", [])))
    ):
        log(f"  ✓ cache hit ({PER_QUERY_OUT.stat().st_size / 1e6:.1f} MB, "
            f"sig={selection_sig}), skipping all {len(RETR_NAMES)} retrievers")
        log(f"  ✓ loaded {cached.get('n_queries', 0)} cached query records")
        query_records = cached["queries"]
        _finalize_from_cached_queries(query_records, cached, selection_sig, t_start)
        return
    log(f"  no cache hit (signature mismatch or missing), running {len(RETR_NAMES)} retrievers fresh")

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
    log(f"\n=== 3. Per-retriever retrieval ({len(RETR_NAMES)} retrievers, no rerank) ===")
    corpus_sig = _corpus_signature(asin_to_doc=asin_to_doc, meta_file=META_FILE)
    query_sig = selection_sig  # selection_sig already computed in step 0
    log(f"  corpus_sig={corpus_sig}  query_sig={query_sig}")
    retr_results: dict[str, list[dict]] = {}
    retr_q_embeds: dict[str, np.ndarray | None] = {}

    for retr in RETRIEVERS:
        kind = retr["kind"]
        # Top-100 candidate cache for every configured retriever.
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
        elif kind == "dense_bge_m3":
            results, q_embeds = bge_m3_retrieve(
                queries, corpus_texts, target_indices,
                corpus_sig=corpus_sig, query_sig=query_sig,
                save_topk_path=topk_path,
            )
            retr_q_embeds[retr["name"]] = q_embeds
        elif kind == "dense_bge_m3_hybrid":
            weights = (1.0, 0.3, 1.0)
            try:
                wd = float(os.environ.get("BGE_M3_W_D", weights[0]))
                ws = float(os.environ.get("BGE_M3_W_S", weights[1]))
                wc = float(os.environ.get("BGE_M3_W_C", weights[2]))
                weights = (wd, ws, wc)
            except Exception:
                pass
            results, q_embeds = bge_m3_hybrid_retrieve(
                queries, corpus_texts, target_indices,
                corpus_sig=corpus_sig, query_sig=query_sig,
                save_topk_path=topk_path, weights=weights,
            )
            retr_q_embeds[retr["name"]] = q_embeds
        elif kind == "late_interaction":
            results, q_embeds = colbertv2_retrieve(queries, corpus_texts, target_indices,
                                                  corpus_sig=corpus_sig, query_sig=query_sig,
                                                  save_topk_path=topk_path)
            retr_q_embeds[retr["name"]] = q_embeds
        else:
            raise ValueError(f"Unknown retriever kind: {kind}")
        retr_results[retr["name"]] = results

    # ---- 3b. Canonical sim09 reference embedding (GTE-base) ----
    # Every retriever uses the same query-query similarity reference, anchored
    # to the sole Dense Bi-Encoder (GTE-base, 768d).
    canonical_q_embeds = retr_q_embeds.get("gte_base")
    if canonical_q_embeds is None:
        # A partial retriever run can fall back to another embedding-capable method.
        for n in ("bge_m3", "bge_m3_hybrid", "colbertv2"):
            if retr_q_embeds.get(n) is not None:
                canonical_q_embeds = retr_q_embeds[n]
                log(f"  GTE-base missing → using {n} for sim09 reference")
                break
    if canonical_q_embeds is None:
        raise RuntimeError("No retriever produced query embeddings; cannot build sim09 reference")
    canonical_embeds_name = "gte_base" if retr_q_embeds.get("gte_base") is not None else \
        next((n for n in ("bge_m3", "bge_m3_hybrid", "colbertv2")
              if retr_q_embeds.get(n) is not None), "unknown")
    log(f"  canonical sim09 reference: {canonical_embeds_name} "
        f"(shape={canonical_q_embeds.shape})")
    # Override: every retriever's sim09 clustering uses canonical_embeds.
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
            r[f"{n}_hit20"] = res["hit20"]

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
                               canonical_embeds_name: str = "gte_base") -> None:
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
                "description": "Stage 5 unified multi-retriever volatility (sim09 selected_only slice, NO rerank). sim09 clustering anchored to GTE-base 768d for ALL retrievers (BM25/SPLADE borrow GTE-base embed).",
                "retrievers": RETR_NAMES,
                "n_retrievers": len(RETR_NAMES),
                "sim09_reference": canonical_embeds_name,
                "selection_file": str(SEL_IN),
                "corpus_size": asins_count,
            },
            "retrieval_metrics": {
                n: {
                    "n_queries": len(query_records),
                    **{
                        f"hit@{k}": float(np.mean([
                            q.get(f"{n}_hit{k}", 0) for q in query_records
                        ]))
                        for k in (1, 5, 10, 20)
                    },
                    "MRR": float(np.mean([
                        float(q.get(f"{n}_RR") or 0.0)
                        for q in query_records
                    ])),
                }
                for n in RETR_NAMES
            },
            "volatility_sim09": volatility,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {SUMMARY_OUT}")

    # ---- 9. Save volatility.json (canonical headline) ----
    # Keep BM25 + GTE-base as headline volatility.
    canonical_volatility: dict = {}
    for n in ("bm25", "gte_base"):
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
                "description": "Canonical Stage 5 volatility (BM25 + GTE-base, selected_only_sim09 slice). Full multi-retriever volatility see retrieval_summary.json.",
                "slices": ["selected_only_sim09"],
                "retrievers": ["bm25", "gte_base"],
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
            ASIN_TO_DOC_CACHE = (Path("/home/wlia0047/hj82_scratch2/wenyu/stage11_corpus_cache")
                                 / subdir / saved["ASIN_TO_DOC_CACHE"].name)
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
        if SMOKE:
            smoke_dir = (Path("/home/wlia0047/hj82_scratch2/wenyu") /
                         "stage11_smoke" / subdir)
            OUT_DIR = smoke_dir
            RESULT_DIR = smoke_dir
            PER_QUERY_OUT = smoke_dir / saved["PER_QUERY_OUT"].name
            SUMMARY_OUT = smoke_dir / saved["SUMMARY_OUT"].name
            VOLATILITY_OUT = smoke_dir / saved["VOLATILITY_OUT"].name
        if "TOPK_SAVE_DIR" in saved:
            TOPK_SAVE_DIR = (Path("/home/wlia0047/hj82_scratch2/wenyu/stage11_retrieval_cache")
                             / subdir / saved["TOPK_SAVE_DIR"].name)
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