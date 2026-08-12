#!/usr/bin/env python3
"""Stage 15 LLM 重排序共享逻辑。

输入：
- Stage 8 产出的 retriever cache（doc embeddings / doc_ids / metadata）
- Stage 8 产出的 query embedding cache（{user_id: {query_text: numpy_embedding}}）
- Stage 8 产出的 BM25 缓存

处理：
1. 对每条 (user_id, query) 用 first-stage retriever 取 top-K 候选 asin + score
2. 用 metadata cache 拼出每个候选的可读描述
3. 调用 M2.5 LLM，让它从 K 个候选中选 top-10
4. 缓存 LLM 输出到本地 JSONL，避免重复调用

输出：
- {category}__{retriever}__{query_type}_top{top_k}_rerank.jsonl：每行一个 (user_id, query, relevant_asin, candidates, llm_top10)
- 同名 .cache.pkl：LLM 输出缓存，键为 (user_id, query, retriever, query_type) -> llm_ranked_asins
"""

import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
STAGE15_ROOT = CURRENT_DIR.parent
PERSOANLQUERY_ROOT = STAGE15_ROOT.parent
STAGE8_ROOT = PERSOANLQUERY_ROOT / "08_retrieval"
for p in (str(STAGE15_ROOT), str(PERSOANLQUERY_ROOT), str(STAGE8_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


SYNTAX_DEPTH_QUERY_CATEGORY = "syntax_depth"
CORRECT_QUERY_TYPE = "correct"
NOISY_QUERY_TYPE = "noisy"
TEMPLATE_CORRECT_QUERY_TYPE = "template"
TEMPLATE_NOISY_QUERY_TYPE = "template_noisy"
PRF_NOISY_QUERY_TYPE = "prf_noisy"
PREPROCESSED_NOISY_QUERY_TYPE = "preprocessed_noisy"
QUERY_TYPES = [CORRECT_QUERY_TYPE, NOISY_QUERY_TYPE]
TEMPLATE_QUERY_TYPES = [TEMPLATE_CORRECT_QUERY_TYPE, TEMPLATE_NOISY_QUERY_TYPE]
PRF_QUERY_TYPES = [PRF_NOISY_QUERY_TYPE]
PREPROCESSED_QUERY_TYPES = [PREPROCESSED_NOISY_QUERY_TYPE]

PRF_CACHE_BASE_DIR = "/fs04/ar57/wenyu/result/personal_query/14_prf_retrieval"
PREPROCESSED_CACHE_BASE_DIR = "/fs04/ar57/wenyu/result/personal_query/13_preprocessed_retrieval"
PRF_QUERY_JSON_DIR = "/fs04/ar57/wenyu/result/personal_query/14_prf"
PREPROCESSED_QUERY_JSON_DIR = "/fs04/ar57/wenyu/result/personal_query/13_preprocessed"

DENSE_RETRIEVERS = {"bge", "e5", "minilm", "star", "ance"}
SPARSE_RETRIEVERS = {"bm25", "splade"}
COLBERTV2_RETRIEVERS = {"colbertv2"}

# iter #186 (P1 backlog DeepSeek-v4 覆盖): explicit registry for first-stage
# retrievers + LLM rerankers so config validation can fail loudly when a
# paper-claimed retriever (BM25, SPLADE, BGE, E5, MiniLM, STAR, ANCE, ColBERTv2,
# DeepSeek-v4 Reranker) is not registered. DeepSeek-v4 Reranker is the LLM
# reranker — it acts as a second-stage reranker, not a first-stage retriever.
ALL_PAPER_FIRST_STAGE_RETRIEVERS = DENSE_RETRIEVERS | SPARSE_RETRIEVERS | COLBERTV2_RETRIEVERS
DEFAULT_FIRST_STAGE_RETRIEVERS = ["bge", "e5"]

# LLM-based rerankers (paper Table 1 9th retriever)
AVAILABLE_RERANKERS = {"M2.5", "DeepSeek-v4"}
DEFAULT_RERANKER = "M2.5"

RERANK_CONFIG_PATH = CURRENT_DIR / "rerank_config.json"
RERANK_PROMPTS_PATH = CURRENT_DIR / "rerank_prompts.json"

RANK_LINE_PATTERN = re.compile(
    r"^\s*rank\s*=\s*(\d+)\s+asin\s*=\s*([A-Z0-9]{8,})\s*$",
    re.IGNORECASE,
)


from common_utils import log  # 统一 log 函数


def load_rerank_config() -> Dict:
    with open(RERANK_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_rerank_prompts() -> Dict:
    with open(RERANK_PROMPTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_stage15_paths(category_name: str) -> Dict[str, str]:
    cfg = load_rerank_config()
    base_result = cfg["paths"]["base_result_dir"]
    base_scratch = cfg["paths"]["base_scratch_dir"]
    out_dir = os.path.join(base_result, category_name)
    cache_dir = os.path.join(base_scratch, category_name, "llm_cache")
    topk_dir = os.path.join(base_scratch, category_name, "topk_candidates")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(topk_dir, exist_ok=True)
    return {
        "output_dir": out_dir,
        "llm_cache_dir": cache_dir,
        "topk_dir": topk_dir,
    }


def _resolve_doc_hash_from_cache_dir(retriever_cache_dir: str, retriever_name: str) -> str:
    """从 retriever cache 目录中找到对应 retriever 的 hash 前缀。"""
    for fname in sorted(os.listdir(retriever_cache_dir)):
        if not fname.startswith(f"{retriever_name}_"):
            continue
        if retriever_name in DENSE_RETRIEVERS and fname.endswith("_doc_ids.pkl"):
            return fname[len(f"{retriever_name}_"):-len("_doc_ids.pkl")]
        if retriever_name in SPARSE_RETRIEVERS and fname.startswith(f"{retriever_name}_") and fname.endswith(".pkl"):
            return fname[len(f"{retriever_name}_"):-len(".pkl")]
        if retriever_name == "colbertv2" and fname.startswith("colbertv2_") and os.path.isdir(os.path.join(retriever_cache_dir, fname)):
            return fname[len("colbertv2_"):]
    raise FileNotFoundError(
        f"Cannot find any cache file for retriever '{retriever_name}' under {retriever_cache_dir}"
    )


def _dense_paths(retriever_cache_dir: str, retriever_name: str) -> Dict[str, str]:
    h = _resolve_doc_hash_from_cache_dir(retriever_cache_dir, retriever_name)
    base = os.path.join(retriever_cache_dir, f"{retriever_name}_{h}")
    return {
        "embeddings": f"{base}_embeddings.npy",
        "doc_ids": f"{base}_doc_ids.pkl",
        "metadata": f"{base}_metadata.pkl",
        "config": f"{base}_config.pkl",
    }


def load_retriever_bundle(category_config: Dict, retriever_name: str) -> Dict:
    """加载 retriever 的 doc embeddings、doc_ids、metadata。

    metadata 始终从 bge 的 metadata cache 加载（按 asin 索引的 dict），
    适用于 dense / sparse 任何 first-stage。"""
    cache_dir = category_config["retriever_cache_dir"]
    bge_meta_path = _dense_paths(cache_dir, "bge")["metadata"]
    if not os.path.exists(bge_meta_path):
        raise FileNotFoundError(
            f"Required bge metadata cache not found (used as universal metadata source): {bge_meta_path}"
        )
    with open(bge_meta_path, "rb") as f:
        shared_metadata = pickle.load(f)
    if not isinstance(shared_metadata, dict):
        raise TypeError(f"bge metadata cache must be dict, got {type(shared_metadata).__name__}: {bge_meta_path}")

    if retriever_name in DENSE_RETRIEVERS:
        paths = _dense_paths(cache_dir, retriever_name)
        embeddings = np.load(paths["embeddings"], mmap_mode="r")
        with open(paths["doc_ids"], "rb") as f:
            doc_ids = pickle.load(f)
        return {
            "type": "dense",
            "embeddings": embeddings,
            "doc_ids": list(doc_ids),
            "metadata": shared_metadata,
            "norms": None,
        }
    if retriever_name == "bm25":
        bm25_path = None
        for fname in sorted(os.listdir(cache_dir)):
            if fname.startswith("bm25_") and fname.endswith(".pkl"):
                bm25_path = os.path.join(cache_dir, fname)
                break
        if bm25_path is None:
            raise FileNotFoundError(f"BM25 cache not found in {cache_dir}")
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)
        # BM25 对象有 doc_asins 字段（utils.retrievers.BM25 实现）
        doc_ids = list(getattr(bm25, "doc_asins", []) or [])
        if not doc_ids:
            raise ValueError(
                f"BM25 cache at {bm25_path} has no doc_asins; cannot align to metadata"
            )
        return {
            "type": "sparse_bm25",
            "bm25": bm25,
            "doc_ids": doc_ids,
            "metadata": shared_metadata,
        }
    raise ValueError(
        f"Stage 15 currently supports first-stage retrievers in "
        f"{sorted(DENSE_RETRIEVERS | {'bm25'})}, got {retriever_name}"
    )


def load_query_embedding_cache(category_config: Dict, retriever_name: str, query_type: str) -> Dict[str, Dict[str, np.ndarray]]:
    """加载 Stage 8 query embedding cache：{user_id: {query_text: ndarray}}"""
    if query_type in TEMPLATE_QUERY_TYPES:
        cache_path = os.path.join(
            category_config["query_cache_dir"],
            f"{query_type}_query",
            f"{retriever_name}__{query_type}_query_cache.pkl",
        )
    elif query_type in PRF_QUERY_TYPES:
        cache_path = os.path.join(
            PRF_CACHE_BASE_DIR,
            category_config["name"],
            "syntax_depth_prf_query",
            f"{retriever_name}__syntax_depth_prf_cache.pkl",
        )
    elif query_type in PREPROCESSED_QUERY_TYPES:
        cache_path = os.path.join(
            PREPROCESSED_CACHE_BASE_DIR,
            category_config["name"],
            "syntax_depth_preprocessed_query",
            f"{retriever_name}__syntax_depth_preprocessed_cache.pkl",
        )
    else:
        cache_path = os.path.join(
            category_config["query_cache_dir"],
            f"syntax_depth_{query_type}_query",
            f"{retriever_name}__syntax_depth_{query_type}_cache.pkl",
        )
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Query embedding cache not found: {cache_path}")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    if not isinstance(cache, dict):
        raise TypeError(f"Query cache must be dict, got {type(cache).__name__}: {cache_path}")
    return cache


def dense_topk_search(
    bundle: Dict,
    query_embeddings: List[np.ndarray],
    top_k: int,
    device: str = "cuda",
) -> List[List[Tuple[str, float]]]:
    """对一批 query embedding 在 GPU 上做余弦相似度 top-K 检索。"""
    if not query_embeddings:
        return []
    embeddings = bundle["embeddings"]
    doc_ids = bundle["doc_ids"]
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 15 dense first-stage requires CUDA GPU for top-K retrieval")

    # 归一化 doc embeddings
    if bundle.get("norms") is None:
        embs = np.asarray(embeddings)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalized = embs / norms
        bundle["norms"] = True
        doc_tensor = torch.from_numpy(normalized).float().to(device)
    else:
        embs = np.asarray(embeddings)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        doc_tensor = torch.from_numpy(embs / norms).float().to(device)

    results: List[List[Tuple[str, float]]] = []
    batch_size = 64
    with torch.no_grad():
        for start in range(0, len(query_embeddings), batch_size):
            batch = np.stack([np.asarray(e, dtype=np.float32) for e in query_embeddings[start:start + batch_size]], axis=0)
            q_norms = np.linalg.norm(batch, axis=1, keepdims=True)
            q_norms = np.where(q_norms == 0, 1.0, q_norms)
            q_tensor = torch.from_numpy(batch / q_norms).float().to(device)
            scores = torch.mm(q_tensor, doc_tensor.T)
            top_k_eff = min(top_k, scores.shape[1])
            top_scores, top_indices = torch.topk(scores, top_k_eff, dim=1)
            for i in range(scores.shape[0]):
                results.append([
                    (doc_ids[int(idx)], float(score))
                    for score, idx in zip(top_scores[i].cpu().tolist(), top_indices[i].cpu().tolist())
                ])
            del scores, q_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def bm25_topk_search(bundle: Dict, queries: List[str], top_k: int) -> List[List[Tuple[str, float]]]:
    bm25 = bundle["bm25"]
    results = []
    for q in queries:
        # BM25 search returns list of (asin, score)
        retrieved = bm25.search(q, top_k=top_k)
        if not isinstance(retrieved, list):
            raise TypeError(f"BM25 search must return list, got {type(retrieved).__name__}")
        results.append([(str(asin), float(score)) for asin, score in retrieved])
    return results


def retrieve_topk(
    bundle: Dict,
    retriever_name: str,
    query_embeddings_for_dense: Optional[List[np.ndarray]],
    query_texts: List[str],
    top_k: int,
) -> List[List[Tuple[str, float]]]:
    if bundle["type"] == "dense":
        if query_embeddings_for_dense is None:
            raise ValueError("Dense first-stage requires query embeddings")
        if len(query_embeddings_for_dense) != len(query_texts):
            raise ValueError(
                f"dense embedding count {len(query_embeddings_for_dense)} != query text count {len(query_texts)}"
            )
        return dense_topk_search(bundle, query_embeddings_for_dense, top_k)
    if bundle["type"] == "sparse_bm25":
        return bm25_topk_search(bundle, query_texts, top_k)
    raise ValueError(f"Unsupported bundle type: {bundle['type']}")


def truncate(s: Optional[str], max_chars: int) -> str:
    if s is None:
        return ""
    text = str(s).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def format_candidate_block(
    candidates: List[Tuple[str, float]],
    bundle: Dict,
    cfg: Dict,
) -> str:
    """把候选 [(asin, score), ...] 拼成 LLM 可读的多行 block。"""
    meta = bundle.get("metadata") or {}
    title_max = cfg["rerank"]["doc_title_max_chars"]
    brand_max = cfg["rerank"]["doc_brand_max_chars"]
    feat_max = cfg["rerank"]["doc_feature_max_chars"]
    desc_max = cfg["rerank"]["doc_description_max_chars"]
    lines: List[str] = []
    for idx, (asin, score) in enumerate(candidates, start=1):
        doc_meta = meta.get(asin) if isinstance(meta, dict) else None
        if isinstance(doc_meta, dict):
            title = truncate(doc_meta.get("title", ""), title_max)
            brand = truncate(doc_meta.get("brand", ""), brand_max)
            feature_raw = doc_meta.get("feature", [])
            if isinstance(feature_raw, list):
                feature = " | ".join(truncate(x, 80) for x in feature_raw[:3])
            else:
                feature = truncate(feature_raw, feat_max)
            desc = truncate(doc_meta.get("description", ""), desc_max)
        else:
            title = brand = feature = desc = ""
        lines.append(
            f"[Candidate {idx}] asin={asin} | title={title} | brand={brand} | features={feature} | description={desc}"
        )
    return "\n".join(lines)


def build_rerank_prompts(
    query: str,
    candidates: List[Tuple[str, float]],
    bundle: Dict,
    cfg: Dict,
    prompts: Dict,
) -> Tuple[str, str]:
    top_k = len(candidates)
    top_n = cfg["rerank"]["top_n_final"]
    system_template = prompts["system"]
    user_template = prompts["user"]
    system_text = (
        system_template
        .replace("{k}", str(top_k))
        .replace("{top_n}", str(top_n))
    )
    candidates_block = format_candidate_block(candidates, bundle, cfg)
    user_text = (
        user_template
        .replace("{query}", query)
        .replace("{candidates}", candidates_block)
        .replace("{top_n}", str(top_n))
    )
    return system_text, user_text


def parse_llm_ranking(text: str, valid_asins: List[str], top_n: int) -> List[str]:
    """从 LLM 文本回复中抽取排序后的 asin 列表。"""
    valid_set = set(valid_asins)
    parsed: Dict[int, str] = {}
    for line in text.splitlines():
        m = RANK_LINE_PATTERN.match(line.strip())
        if not m:
            continue
        rank = int(m.group(1))
        asin = m.group(2)
        if asin in valid_set and 1 <= rank <= top_n and rank not in parsed:
            parsed[rank] = asin
    if not parsed:
        return []
    ordered = [parsed[r] for r in sorted(parsed.keys())]
    return ordered


def fill_remaining_with_first_stage(
    llm_ordered: List[str],
    candidates: List[Tuple[str, float]],
    top_n: int,
) -> List[str]:
    """用 first-stage 候选顺序补齐 LLM 没有覆盖的位置。"""
    result = list(llm_ordered)
    seen = set(result)
    for asin, _score in candidates:
        if asin in seen:
            continue
        result.append(asin)
        seen.add(asin)
        if len(result) >= top_n:
            break
    return result[:top_n]


def compute_mixed_rerank_score(
    llm_ordered: List[str],
    candidates: List[Tuple[str, float]],
    top_n: int,
    sim_weight: float = 0.1,
) -> Tuple[List[str], List[float]]:
    """listwise 模式下的混合打分：final = llm_score_proxy + sim_weight * sim_score。

    公式对齐 stark (stark/stark_qa/models/llm_reranker.py:114-115)：
        sim_score = (cand_len - idx) / cand_len
        final_score = llm_score + sim_weight * sim_score

    listwise 模式下 LLM 只输出 rank 顺序没有 0~1 浮点分，因此用 rank 代理：
        llm_score_proxy = (top_n - rank) / (top_n - 1)  # rank=1→1.0, rank=top_n→0.0
    未排进 top_n 的 asin llm_score_proxy=0.0，sim_score 主导。

    Args:
        llm_ordered: parse_llm_ranking + fill_remaining 后的 asin 顺序（长度 ≥ top_n）
        candidates: first-stage 候选 [(asin, score), ...]，按 first-stage 排名
        top_n: 最终输出 top-N
        sim_weight: 相似度权重（默认 0.1 对齐 stark）

    Returns:
        (final_ranked_asins, final_scores) — 按 final_score 降序，长度 = top_n
    """
    cand_len = len(candidates)
    if cand_len == 0:
        return [], []

    llm_score_per_asin: Dict[str, float] = {}
    if top_n > 1:
        for rank, asin in enumerate(llm_ordered[:top_n], start=1):
            llm_score_per_asin[asin] = (top_n - rank) / (top_n - 1)
    else:
        if llm_ordered:
            llm_score_per_asin[llm_ordered[0]] = 1.0

    sim_score_per_asin: Dict[str, float] = {
        asin: (cand_len - idx) / cand_len
        for idx, (asin, _) in enumerate(candidates)
    }

    mixed: Dict[str, float] = {
        asin: llm_score_per_asin.get(asin, 0.0) + sim_weight * sim_score_per_asin.get(asin, 0.0)
        for asin, _ in candidates
    }

    sorted_asins = sorted(mixed.keys(), key=lambda a: -mixed[a])
    final_ranked = sorted_asins[:top_n]
    final_scores = [mixed[a] for a in final_ranked]
    return final_ranked, final_scores


def get_llm_client(cfg: Dict, category_name: str = None):
    """通过 llm_client.py 工厂构造 LLM 客户端。

    优先按 PersoanlQuery/llm_client.py 绝对路径加载（不依赖 PersoanlQuery 是包），
    失败时再尝试 cfg 指定的模块路径。

    如果 cfg["llm"]["per_category_client"][category_name] 存在，则用其指定的 client
    覆盖默认的 cfg["llm"]["client_class"]，实现 per-category client 切换。
    """
    llm_cfg = cfg["llm"]
    per_cat = llm_cfg.get("per_category_client") or {}
    if category_name and category_name in per_cat:
        client_class_name = per_cat[category_name]
    else:
        client_class_name = llm_cfg["client_class"]
    import importlib.util
    llm_client_path = PERSOANLQUERY_ROOT / "llm_client.py"
    if not llm_client_path.exists():
        raise FileNotFoundError(f"llm_client.py not found at {llm_client_path}")
    spec = importlib.util.spec_from_file_location("persoanlquery_llm_client", str(llm_client_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {llm_client_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    client_cls = getattr(mod, client_class_name)
    return client_cls(model=llm_cfg["model"])


_prompt_cache: Dict[Tuple[str, str], str] = {}


def call_llm_with_cache(
    client,
    system_text: str,
    user_text: str,
    _disk_cache_unused,  # 保留签名以兼容旧调用，但不再写入
    cfg: Dict,
) -> Tuple[str, Dict]:
    """调用 LLM，结果写入 module-level in-memory prompt cache（不持久化）。

    返回 (text, usage)：
    - text: LLM 文本回复
    - usage: {cache_creation_input_tokens, cache_read_input_tokens, input_tokens, output_tokens}
    本地 cache 命中时 usage 全部为 0。

    业务级 LLM response cache 改用 jsonl-as-cache 方案（在 rerank_runner 中通过 load_llm_cache_from_jsonl 加载），
    此处仅保留 in-memory prompt cache 用于同一次 run 内 system+user 文本完全相同的重复调用。
    """
    key = (system_text, user_text)
    if key in _prompt_cache:
        return _prompt_cache[key], {
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
    llm_cfg = cfg["llm"]
    text, usage = client.call_with_cache(
        system_base=system_text,
        user_content=user_text,
        max_tokens=llm_cfg["max_tokens"],
        temperature=llm_cfg["temperature"],
        max_retries=llm_cfg["max_retries"],
        retry_on_empty_response=llm_cfg.get("retry_on_empty_response", False),
    )
    _prompt_cache[key] = text
    return text, usage


def load_llm_response_cache(cache_file: str) -> Dict[str, Dict]:
    if not os.path.exists(cache_file):
        return {}
    with open(cache_file, "rb") as f:
        cache = pickle.load(f)
    if not isinstance(cache, dict):
        raise TypeError(f"LLM response cache must be dict, got {type(cache).__name__}: {cache_file}")
    return cache


def save_llm_response_cache(cache_file: str, cache: Dict) -> None:
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_llm_cache_from_jsonl(jsonl_path: str) -> Dict[Tuple[str, str, str, str], str]:
    """从 rerank output jsonl 加载 LLM response cache。

    取代 pickle cache 方案：jsonl 本身就是 cache。
    启动时 parse jsonl 一次，构建 {(user_id, query, retriever, query_type): response_text}。
    - jsonl 损坏只会丢最后未 flush 的 1 行（vs pickle 整个损坏）
    - 单线程写（main thread），无并发竞争
    - failed record 不写 jsonl → 下次 run 自动重试
    """
    cache: Dict[Tuple[str, str, str, str], str] = {}
    if not os.path.exists(jsonl_path):
        return cache
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (r["user_id"], r["query"], r["retriever"], r["query_type"])
            cache[key] = r.get("llm_raw_response", "") or ""
    return cache


def load_candidate_cache(cache_file: str) -> Dict[Tuple[str, str], List[Tuple[str, float]]]:
    """加载 top-K 候选 cache：{(user_id, query): [(asin, score), ...]}。

    第一次 GPU 检索后写盘，后续重跑（retry failed LLM、改 prompt、跑 noisy query 等）直接读 cache，跳过 GPU。
    """
    if not os.path.exists(cache_file):
        return {}
    with open(cache_file, "rb") as f:
        cache = pickle.load(f)
    if not isinstance(cache, dict):
        raise TypeError(f"Candidate cache must be dict, got {type(cache).__name__}: {cache_file}")
    return cache


def save_candidate_cache(cache_file: str, cache: Dict) -> None:
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_syntax_depth_query_records(category_config: Dict) -> List[Dict]:
    """从 Stage 6 输出加载每条 (user_id, asin, query) 记录。"""
    candidates = [
        "/home/wlia0047/ar57/wenyu/result/personal_query/06_query/{cat}/query_by_syntax_depth_vades_lite_sentence_user_distribution_train10_holdout10.json",
    ]
    cat = category_config["name"]
    for tmpl in candidates:
        path = tmpl.format(cat=cat)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise TypeError(f"expression_style query file must be list: {path}")
            return data
    raise FileNotFoundError(
        f"Cannot find expression_style query file for {cat} under any known path"
    )


def build_evaluation_records(
    query_records: List[Dict],
    query_type: str,
) -> List[Dict]:
    """把 Stage 6 expression_style records 转为 (user_id, asin, query) 评估用列表。"""
    out: List[Dict] = []
    for row in query_records:
        if not isinstance(row, dict):
            raise TypeError(f"expression_style row must be dict, got {type(row).__name__}")
        user_id = row.get("user_id")
        asin = row.get("asin")
        syntax_query = row.get("syntax_depth_query")
        if not (isinstance(user_id, str) and user_id and isinstance(asin, str) and asin and isinstance(syntax_query, dict)):
            continue
        if query_type == CORRECT_QUERY_TYPE:
            q = syntax_query.get("query")
        elif query_type == NOISY_QUERY_TYPE:
            noisy_q = syntax_query.get("noisy_query") or syntax_query.get("query")
            q = noisy_q
        else:
            raise ValueError(f"Unsupported query_type: {query_type}")
        if not isinstance(q, str) or not q.strip():
            continue
        out.append({
            "user_id": user_id,
            "asin": asin,
            "query": q.strip(),
        })
    return out


def load_noisy_query_text_map(category_name: str) -> Dict[Tuple[str, str, str], str]:
    """从 Stage 7 noisy_query.json 加载 (user, asin, clean_query) -> noisy_query 的映射。"""
    candidates = [
        f"/home/wlia0047/ar57/wenyu/result/personal_query/07_inject_noisy/{category_name}/noisy_query.json",
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise TypeError(f"noisy_query.json must be list, got {type(data).__name__}: {path}")
        mapping: Dict[Tuple[str, str, str], str] = {}
        for row in data:
            if not isinstance(row, dict):
                continue
            uid = row.get("uid")
            asin = row.get("asin")
            clean = row.get("clean_query")
            noisy = row.get("noisy_query")
            if not (isinstance(uid, str) and isinstance(asin, str) and isinstance(clean, str) and isinstance(noisy, str)):
                continue
            mapping[(uid, asin, clean)] = noisy
        return mapping
    raise FileNotFoundError(
        f"Cannot find noisy_query.json for {category_name} under any known path"
    )


def load_template_query_records(category_name: str, query_type: str) -> List[Dict]:
    """从 Stage 13_query_template 输出加载每条 (user_id, asin, query) 记录。

    - query_type=TEMPLATE_CORRECT_QUERY_TYPE: 读 query_template.json (clean template)，
      且只保留那些 (user_id, asin) 在 query_template_noisy.json 中也存在的记录，
      保证 clean 与 noisy 两条 rerank 路径在 (user_id, asin) 维度上完全可比。
    - query_type=TEMPLATE_NOISY_QUERY_TYPE: 读 query_template_noisy.json (noisy template)
    返回的 dict 形状与 syntax_depth eval_records 一致：{user_id, asin, query}。
    """
    clean_path = f"/home/wlia0047/ar57/wenyu/result/personal_query/13_query_template/{category_name}/query_template.json"
    noisy_path = f"/home/wlia0047/ar57/wenyu/result/personal_query/13_query_template/{category_name}/query_template_noisy.json"

    def _load_pairs(p: str) -> set:
        if not os.path.exists(p):
            return set()
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise TypeError(f"template query file must be list, got {type(data).__name__}: {p}")
        return {(r["user_id"], r["asin"]) for r in data if isinstance(r, dict) and r.get("user_id") and r.get("asin")}

    if query_type == TEMPLATE_CORRECT_QUERY_TYPE:
        path = clean_path
        noisy_pairs = _load_pairs(noisy_path)
    elif query_type == TEMPLATE_NOISY_QUERY_TYPE:
        path = noisy_path
        noisy_pairs = None
    else:
        raise ValueError(f"Unsupported template query_type: {query_type}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"template query file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"template query file must be list, got {type(data).__name__}: {path}")
    out: List[Dict] = []
    skipped_no_noisy = 0
    for row in data:
        if not isinstance(row, dict):
            continue
        user_id = row.get("user_id")
        asin = row.get("asin")
        q = row.get("query")
        if not (isinstance(user_id, str) and user_id and isinstance(asin, str) and asin and isinstance(q, str) and q.strip()):
            continue
        if noisy_pairs is not None and (user_id, asin) not in noisy_pairs:
            skipped_no_noisy += 1
            continue
        out.append({
            "user_id": user_id,
            "asin": asin,
            "query": q.strip(),
        })
    if query_type == TEMPLATE_CORRECT_QUERY_TYPE and skipped_no_noisy:
        # 用 log 而非直接 print，避免在 import 阶段没装上 log handler 时报错
        print(f"  [load_template_query_records] 跳过 {skipped_no_noisy} 条无对应 noisy 的 clean template")
    return out


def load_prf_query_records(category_name: str) -> List[Dict]:
    """从 14_prf/{cat}/prf_query.json 加载每条 (user_id, asin, query) 记录。

    query 字段使用 noisy_query 文本（PRF 增强作用于其 embedding，rerank LLM 看到的是原 noisy 文本）。
    返回的 dict 形状与 syntax_depth eval_records 一致：{user_id, asin, query}。
    """
    prf_path = os.path.join(PRF_QUERY_JSON_DIR, category_name, "prf_query.json")
    if not os.path.exists(prf_path):
        raise FileNotFoundError(f"prf_query.json not found: {prf_path}")
    with open(prf_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"prf_query.json must be list, got {type(data).__name__}: {prf_path}")
    out: List[Dict] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        user_id = row.get("uid")
        asin = row.get("asin")
        noisy_q = row.get("noisy_query")
        if not (
            isinstance(user_id, str) and user_id
            and isinstance(asin, str) and asin
            and isinstance(noisy_q, str) and noisy_q.strip()
        ):
            continue
        out.append({
            "user_id": user_id,
            "asin": asin,
            "query": noisy_q.strip(),
        })
    return out


def load_preprocessed_query_records(category_name: str) -> List[Dict]:
    """从 13_preprocessed/{cat}/preprocessed_query.json 加载每条 (user_id, asin, query) 记录。

    query 字段使用 preprocessed_query 文本（与对应 retriever cache 里的 key 一致）。
    返回的 dict 形状与 syntax_depth eval_records 一致：{user_id, asin, query}。
    """
    pp_path = os.path.join(PREPROCESSED_QUERY_JSON_DIR, category_name, "preprocessed_query.json")
    if not os.path.exists(pp_path):
        raise FileNotFoundError(f"preprocessed_query.json not found: {pp_path}")
    with open(pp_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"preprocessed_query.json must be list, got {type(data).__name__}: {pp_path}")
    out: List[Dict] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        user_id = row.get("uid")
        asin = row.get("asin")
        pre_q = row.get("preprocessed_query")
        if not (
            isinstance(user_id, str) and user_id
            and isinstance(asin, str) and asin
            and isinstance(pre_q, str) and pre_q.strip()
        ):
            continue
        out.append({
            "user_id": user_id,
            "asin": asin,
            "query": pre_q.strip(),
        })
    return out


    return out


def compute_metrics(relevant_asin: str, retrieved_asins: List[str], k_values: List[int]) -> Dict:
    metrics: Dict[str, float] = {}
    for k in k_values:
        top_k = retrieved_asins[:k]
        metrics[f"P@{k}"] = 1.0 if relevant_asin in top_k else 0.0
        if relevant_asin in top_k:
            rank = top_k.index(relevant_asin) + 1
            metrics[f"N@{k}"] = 1.0 / np.log2(rank + 1)
            metrics[f"MR@{k}"] = 1.0 / rank
        else:
            metrics[f"N@{k}"] = 0.0
            metrics[f"MR@{k}"] = 0.0
        metrics[f"H@{k}"] = 1.0 if relevant_asin in top_k else 0.0
    return metrics


def compute_average_metrics(metrics_list: List[Dict], k_values: List[int]) -> Dict:
    if not metrics_list:
        raise ValueError("Cannot compute average metrics for empty list")
    out: Dict[str, float] = {}
    for k in k_values:
        out[f"P@{k}"] = float(np.mean([m[f"P@{k}"] for m in metrics_list]))
        out[f"N@{k}"] = float(np.mean([m[f"N@{k}"] for m in metrics_list]))
        out[f"MR@{k}"] = float(np.mean([m[f"MR@{k}"] for m in metrics_list]))
        out[f"H@{k}"] = float(np.mean([m[f"H@{k}"] for m in metrics_list]))
    return out


def write_jsonl(path: str, records: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: str) -> List[Dict]:
    out: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def validate_rerank_config(cfg: Dict) -> Dict:
    """iter #186: explicit validation of rerank config against paper-claimed retrievers.

    Raises ValueError with actionable diagnostic for any first-stage retriever
    in cfg["rerank"]["first_stage_retrievers"] that's not registered.

    Returns a diagnostic dict:
        {
          "first_stage_retrievers": [...],
          "reranker": str,
          "missing_paper_coverage": [...],
          "all_first_stage_registered": bool,
          "warnings": [...],
        }
    """
    diagnostics: Dict = {"warnings": []}
    fs_list = cfg.get("rerank", {}).get("first_stage_retrievers", [])
    diagnostics["first_stage_retrievers"] = list(fs_list)

    # Validate each first-stage retriever is registered
    all_known = ALL_PAPER_FIRST_STAGE_RETRIEVERS
    unknown = [r for r in fs_list if r not in all_known]
    if unknown:
        raise ValueError(
            f"first_stage_retrievers contains unregistered values: {unknown}. "
            f"Known: {sorted(all_known)}. "
            f"DeepSeek-v4 is an LLM reranker, not first-stage — set it as "
            f"cfg['llm']['model'] instead."
        )
    diagnostics["all_first_stage_registered"] = True

    # Check reranker
    reranker = cfg.get("llm", {}).get("model", DEFAULT_RERANKER)
    diagnostics["reranker"] = reranker
    if reranker not in AVAILABLE_RERANKERS:
        diagnostics["warnings"].append(
            f"reranker '{reranker}' not in AVAILABLE_RERANKERS={sorted(AVAILABLE_RERANKERS)}. "
            f"Will attempt to construct LLM client anyway."
        )

    # Paper coverage: flag any paper-claimed 8 first-stage retrievers missing
    expected_paper_first_stage = {
        "bm25", "splade",          # sparse
        "bge", "e5", "minilm", "star", "ance",  # dense
        "colbertv2",                # multi-vector
    }
    missing = sorted(expected_paper_first_stage - set(fs_list))
    diagnostics["missing_paper_coverage"] = missing
    if len(fs_list) < 2:
        diagnostics["warnings"].append(
            f"first_stage_retrievers={fs_list} is sparse; consider expanding "
            f"to a multi-retriever comparison for Table 1 rerank-cell coverage."
        )
    return diagnostics