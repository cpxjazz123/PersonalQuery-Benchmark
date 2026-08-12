#!/usr/bin/env python3
"""Stage 15 LLM 重排序 runner（3 个类别共享）。"""

import os
os.environ.setdefault("HF_HOME", "/home/wlia0047/ar57_scratch/wenyu/hf_models")
os.environ.setdefault("HF_HUB_CACHE", "/home/wlia0047/ar57_scratch/wenyu/hf_models")

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
STAGE15_ROOT = CURRENT_DIR.parent
PERSOANLQUERY_ROOT = STAGE15_ROOT.parent
STAGE8_ROOT = PERSOANLQUERY_ROOT / "08_retrieval"
for p in (str(STAGE15_ROOT), str(PERSOANLQUERY_ROOT), str(STAGE8_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import get_category_config  # noqa: E402
from llm_rerank_common import (  # noqa: E402
    CORRECT_QUERY_TYPE,
    NOISY_QUERY_TYPE,
    PREPROCESSED_NOISY_QUERY_TYPE,
    PREPROCESSED_QUERY_TYPES,
    PRF_NOISY_QUERY_TYPE,
    PRF_QUERY_TYPES,
    QUERY_TYPES,
    TEMPLATE_CORRECT_QUERY_TYPE,
    TEMPLATE_NOISY_QUERY_TYPE,
    TEMPLATE_QUERY_TYPES,
    build_evaluation_records,
    build_rerank_prompts,
    call_llm_with_cache,
    compute_mixed_rerank_score,
    fill_remaining_with_first_stage,
    get_llm_client,
    get_stage15_paths,
    load_candidate_cache,
    load_llm_cache_from_jsonl,
    load_noisy_query_text_map,
    load_preprocessed_query_records,
    load_prf_query_records,
    load_rerank_config,
    load_rerank_prompts,
    load_retriever_bundle,
    load_query_embedding_cache,
    load_syntax_depth_query_records,
    load_template_query_records,
    log,
    parse_llm_ranking,
    retrieve_topk,
    save_candidate_cache,
)


def collect_user_query_records(category_name: str, category_config: Dict, query_type: str) -> list:
    if query_type in TEMPLATE_QUERY_TYPES:
        template_records = load_template_query_records(category_name, query_type)
        log(f"  [template] 加载 {query_type} {len(template_records)} 条 (from 13_query_template)")
        if not template_records:
            raise ValueError(
                f"No template records for {category_name} ({query_type})"
            )
        return template_records

    if query_type in PRF_QUERY_TYPES:
        prf_records = load_prf_query_records(category_name)
        log(f"  [prf] 加载 {query_type} {len(prf_records)} 条 (from 14_prf)")
        if not prf_records:
            raise ValueError(
                f"No PRF records for {category_name} ({query_type})"
            )
        return prf_records

    if query_type in PREPROCESSED_QUERY_TYPES:
        pp_records = load_preprocessed_query_records(category_name)
        log(f"  [preprocessed] 加载 {query_type} {len(pp_records)} 条 (from 13_preprocessed)")
        if not pp_records:
            raise ValueError(
                f"No preprocessed records for {category_name} ({query_type})"
            )
        return pp_records

    records = load_syntax_depth_query_records(category_config)
    eval_records = build_evaluation_records(records, query_type)

    if query_type == NOISY_QUERY_TYPE:
        noisy_map = load_noisy_query_text_map(category_name)
        new_records = []
        skipped = 0
        for r in eval_records:
            key = (r["user_id"], r["asin"], r["query"])
            noisy = noisy_map.get(key)
            if noisy is None:
                skipped += 1
                continue
            new_records.append({
                "user_id": r["user_id"],
                "asin": r["asin"],
                "query": noisy,
                "clean_query": r["query"],
            })
        log(f"  [noisy] 覆盖 noisy_query {len(new_records)}/{len(eval_records)} 条 (跳过 {skipped} 条无 mapping)")
        if not new_records:
            raise ValueError(
                f"No noisy records with mapping for {category_name}; noisy_query.json 覆盖率为 0"
            )
        return new_records

    return eval_records


def process_one_record_sync(
    idx: int,
    record: Dict,
    candidates: List,
    bundle: Dict,
    cfg: Dict,
    prompts: Dict,
    retriever_name: str,
    query_type: str,
    top_n: int,
    llm_cache: Dict,
    client,
) -> Tuple[Dict, str, str, Dict, bool]:
    """单条 record 的处理（线程安全：cache_key 含 user_id+query 保证唯一，不会跨 worker 冲突）。"""
    if not candidates:
        raise ValueError(
            f"Empty candidates for user={record['user_id']} query={record['query'][:80]}"
        )
    system_text, user_text = build_rerank_prompts(
        query=record["query"],
        candidates=candidates,
        bundle=bundle,
        cfg=cfg,
        prompts=prompts,
    )
    cache_key = (record["user_id"], record["query"], retriever_name, query_type)
    status = "ok"
    is_cache_hit = False
    if cache_key in llm_cache:
        text = llm_cache[cache_key]
        usage = {
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        is_cache_hit = True
        if not text:
            status = "failed"
    else:
        text, usage = call_llm_with_cache(client, system_text, user_text, llm_cache, cfg)
        if not text:
            status = "failed"
        else:
            llm_cache[cache_key] = text
    valid_asins = [asin for asin, _score in candidates]
    first_stage_top10 = [asin for asin, _score in candidates[:top_n]]
    if status == "failed":
        llm_ordered = list(first_stage_top10)
    else:
        llm_ordered = parse_llm_ranking(text, valid_asins, top_n)
        if len(llm_ordered) < top_n:
            llm_ordered = fill_remaining_with_first_stage(llm_ordered, candidates, top_n)
    sim_weight = cfg["rerank"].get("sim_weight", 0.1)
    final_ranked, final_scores = compute_mixed_rerank_score(
        llm_ordered=llm_ordered,
        candidates=candidates,
        top_n=top_n,
        sim_weight=sim_weight,
    )
    out_record = {
        "user_id": record["user_id"],
        "asin": record["asin"],
        "query": record["query"],
        "clean_query": record.get("clean_query"),
        "retriever": retriever_name,
        "query_type": query_type,
        "status": status,
        "top_k_candidates": [
            {"asin": asin, "score": float(score)}
            for asin, score in candidates
        ],
        "llm_ranked_asins": llm_ordered,
        "llm_final_ranked_asins": final_ranked,
        "llm_rerank_scores": final_scores,
        "sim_weight": sim_weight,
        "llm_usage": usage,
        "llm_raw_response": text,
        "first_stage_top10": first_stage_top10,
    }
    return out_record, status, text, usage, is_cache_hit


def run_for_retriever_and_query_type(
    cfg: Dict,
    prompts: Dict,
    paths: Dict,
    client,
    category_name: str,
    category_config: Dict,
    retriever_name: str,
    query_type: str,
    top_k: int,
    top_n: int,
    smoke_limit: Optional[int] = None,
) -> None:
    log(f"\n=== retriever={retriever_name} | query_type={query_type} | top_k={top_k} ===")

    bundle = load_retriever_bundle(category_config, retriever_name)

    eval_records = collect_user_query_records(category_name, category_config, query_type)
    log(f"  [records] {len(eval_records)} 条 (user_id, asin, query)")

    candidate_cache_file = os.path.join(
        paths["llm_cache_dir"],
        f"{retriever_name}__{query_type}_top{top_k}_candidates.pkl",
    )
    candidate_cache = load_candidate_cache(candidate_cache_file)
    log(f"  [cand_cache] 已加载 {len(candidate_cache)} 条历史候选")

    query_cache = None

    def _ensure_query_cache() -> Dict:
        nonlocal query_cache
        if query_cache is None and bundle["type"] == "dense":
            query_cache = load_query_embedding_cache(category_config, retriever_name, query_type)
        return query_cache

    filtered_records = []
    cached_records = []
    uncached_records = []
    skipped = 0
    for r in eval_records:
        key = (r["user_id"], r["query"])
        if key in candidate_cache:
            r["_candidates"] = candidate_cache[key]
            filtered_records.append(r)
            cached_records.append(r)
            continue
        if bundle["type"] == "dense":
            qc = _ensure_query_cache()
            user_qmap = qc.get(r["user_id"]) if qc else None
            if not user_qmap or r["query"] not in user_qmap:
                skipped += 1
                continue
            r["_query_embedding"] = user_qmap[r["query"]]
            filtered_records.append(r)
            uncached_records.append(r)
        else:
            filtered_records.append(r)
            uncached_records.append(r)
    log(f"  [cache] 命中 {len(filtered_records)}/{len(eval_records)}，跳过 {skipped}")
    log(f"  [cand_cache] 命中 {len(cached_records)}/{len(filtered_records)}，未命中 {len(uncached_records)}")

    smoke_limit = smoke_limit if smoke_limit is not None else cfg["rerank"].get("smoke_test_limit")
    if smoke_limit and isinstance(smoke_limit, int) and smoke_limit > 0:
        if len(filtered_records) > smoke_limit:
            log(f"  [smoke] 限制为前 {smoke_limit} 条 (override={smoke_limit})")
            filtered_records = filtered_records[:smoke_limit]
            cached_records = [r for r in cached_records if r in filtered_records]
            uncached_records = [r for r in uncached_records if r in filtered_records]

    if not filtered_records:
        raise ValueError(f"{retriever_name} {query_type} 无可评估记录")

    if not uncached_records:
        log(f"  [retrieve] 全部命中 cache，跳过 GPU 检索")
        candidates_per_record = [r["_candidates"] for r in filtered_records]
    else:
        log(f"  [retrieve] top-{top_k} 检索中... ({len(uncached_records)} 条未命中)")
        retrieve_start = time.time()
        if bundle["type"] == "dense":
            query_embeddings = [r["_query_embedding"] for r in uncached_records]
            query_texts = [r["query"] for r in uncached_records]
            new_candidates = retrieve_topk(
                bundle, retriever_name, query_embeddings, query_texts, top_k
            )
        else:
            query_texts = [r["query"] for r in uncached_records]
            new_candidates = retrieve_topk(
                bundle, retriever_name, None, query_texts, top_k
            )
        log(f"  [retrieve] 完成，耗时 {time.time() - retrieve_start:.1f}s")

        for r, cands in zip(uncached_records, new_candidates):
            candidate_cache[(r["user_id"], r["query"])] = cands
        save_candidate_cache(candidate_cache_file, candidate_cache)
        log(f"  [cand_cache] 已保存 {len(candidate_cache)} 条到 {os.path.basename(candidate_cache_file)}")

        cand_map = {(r["user_id"], r["query"]): r["_candidates"] for r in cached_records}
        for r, cands in zip(uncached_records, new_candidates):
            cand_map[(r["user_id"], r["query"])] = cands
        candidates_per_record = [cand_map[(r["user_id"], r["query"])] for r in filtered_records]

    out_file = os.path.join(
        paths["output_dir"],
        f"{retriever_name}__{query_type}_top{top_k}_rerank.jsonl",
    )
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    llm_cache = load_llm_cache_from_jsonl(out_file)
    log(f"  [llm_cache] 从 jsonl 加载 {len(llm_cache)} 条历史响应")
    out_fh = open(out_file, "a", encoding="utf-8")

    total = len(filtered_records)
    max_workers = cfg["rerank"].get("max_concurrent_llm_calls", 1)
    log(f"  [rerank] 开始 LLM 重排序，total={total} max_concurrent_llm_calls={max_workers}")
    log(f"  [output] 逐条写入 {out_file}")
    rerank_start = time.time()
    cache_hits = 0
    new_calls = 0
    failed_calls = 0
    written_records = 0
    total_input_tokens = 0
    total_cache_creation_tokens = 0
    total_cache_read_tokens = 0
    total_output_tokens = 0
    completed = 0
    skipped_failed = 0

    def _log_progress() -> None:
        if completed % 10 == 0 or completed == total:
            elapsed = time.time() - rerank_start
            total_billable_input = total_input_tokens + total_cache_creation_tokens + total_cache_read_tokens
            cache_hit_rate = (
                total_cache_read_tokens / total_billable_input * 100
                if total_billable_input > 0 else 0.0
            )
            log(
                f"    进度: {completed}/{total} "
                f"({100 * completed / total:.1f}%) | "
                f"cache_hit(local)={cache_hits} new_call={new_calls} failed={failed_calls} | "
                f"input={total_input_tokens} cache_creation={total_cache_creation_tokens} "
                f"cache_read={total_cache_read_tokens} output={total_output_tokens} | "
                f"system_cache_hit_rate={cache_hit_rate:.1f}% | "
                f"elapsed={elapsed:.1f}s"
            )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_record = {
            executor.submit(
                process_one_record_sync,
                idx, record, candidates, bundle, cfg, prompts,
                retriever_name, query_type, top_n, llm_cache, client,
            ): (idx, record)
            for idx, (record, candidates) in enumerate(zip(filtered_records, candidates_per_record))
        }

        for future in as_completed(future_to_record):
            idx, record = future_to_record[future]
            try:
                out_record, status, text, usage, is_cache_hit = future.result()
            except Exception as e:
                log(
                    f"      [ERROR] record {idx + 1} user={record['user_id']} "
                    f"asin={record['asin']} raised: {e}"
                )
                raise

            if is_cache_hit:
                cache_hits += 1
                if status == "failed":
                    failed_calls += 1
            else:
                new_calls += 1
                if status == "failed":
                    failed_calls += 1
                    log(
                        f"      [FAILED] user={record['user_id']} asin={record['asin']} "
                        f"query='{record['query'][:60]}' -> LLM returned empty response, "
                        f"fallback to first_stage_top10 (will retry on next run)"
                    )
            total_input_tokens += int(usage.get("input_tokens", 0) or 0)
            total_cache_creation_tokens += int(usage.get("cache_creation_input_tokens", 0) or 0)
            total_cache_read_tokens += int(usage.get("cache_read_input_tokens", 0) or 0)
            total_output_tokens += int(usage.get("output_tokens", 0) or 0)

            completed += 1

            if is_cache_hit:
                _log_progress()
                continue

            if status == "failed":
                skipped_failed += 1
                log(
                    f"      [SKIPPED] user={record['user_id']} asin={record['asin']} "
                    f"query='{record['query'][:60]}' -> status=failed, 不写入 jsonl"
                )
                continue

            out_fh.write(json.dumps(out_record, ensure_ascii=False, default=str) + "\n")
            out_fh.flush()
            written_records += 1

            log(
                f"      [record {idx + 1}/{total}] user={record['user_id']} asin={record['asin']} "
                f"query='{record['query'][:60]}' -> done (status={status})"
            )

            _log_progress()

    log(
        f"  [rerank] 完成，total={total} cache_hits(local)={cache_hits} new_calls={new_calls} failed={failed_calls} skipped={skipped_failed} | "
        f"input={total_input_tokens} cache_creation={total_cache_creation_tokens} "
        f"cache_read={total_cache_read_tokens} output={total_output_tokens} | "
        f"耗时={time.time() - rerank_start:.1f}s"
    )

    out_fh.close()
    log(f"  [output] 共写入 {written_records} 条到 {out_file} (skipped {skipped_failed} 条 failed record)")


def run_for_category(category_name: str, query_types: list = None, smoke_limit: Optional[int] = None) -> None:
    if query_types is None:
        query_types = QUERY_TYPES
    log("=" * 80)
    log(f"Stage 15 LLM 重排序 | {category_name} | query_types={query_types} | smoke_limit={smoke_limit}")
    log("=" * 80)

    cfg = load_rerank_config()
    prompts = load_rerank_prompts()
    category_config = get_category_config(category_name)
    paths = get_stage15_paths(category_name)

    log(f"  first_stage_retrievers = {cfg['rerank']['first_stage_retrievers']}")
    log(f"  top_k_candidates = {cfg['rerank']['top_k_candidates']}")
    log(f"  top_n_final = {cfg['rerank']['top_n_final']}")
    log(f"  llm_model = {cfg['llm']['model']}")

    client = get_llm_client(cfg, category_name=category_name)
    log(f"  llm_client = {type(client).__name__}(model={cfg['llm']['model']})")

    top_k = cfg["rerank"]["top_k_candidates"]
    top_n = cfg["rerank"]["top_n_final"]

    for retriever_name in cfg["rerank"]["first_stage_retrievers"]:
        for query_type in query_types:
            run_for_retriever_and_query_type(
                cfg=cfg,
                prompts=prompts,
                paths=paths,
                client=client,
                category_name=category_name,
                category_config=category_config,
                retriever_name=retriever_name,
                query_type=query_type,
                top_k=top_k,
                top_n=top_n,
                smoke_limit=smoke_limit,
            )

    log(f"\nStage 15 LLM 重排序完成 | {category_name}")
