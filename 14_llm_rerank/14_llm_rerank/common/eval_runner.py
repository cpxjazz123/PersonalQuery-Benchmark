#!/usr/bin/env python3
"""Stage 15 LLM 重排序评估 runner（3 个类别共享）。"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
STAGE15_ROOT = CURRENT_DIR.parent
PERSOANLQUERY_ROOT = STAGE15_ROOT.parent
STAGE8_ROOT = PERSOANLQUERY_ROOT / "08_retrieval"
for p in (str(STAGE15_ROOT), str(PERSOANLQUERY_ROOT), str(STAGE8_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import get_category_config  # noqa: E402
from llm_rerank_common import (  # noqa: E402
    compute_average_metrics,
    compute_metrics,
    get_stage15_paths,
    load_rerank_config,
    log,
    read_jsonl,
)


def evaluate_one_file(
    records: List[Dict],
    k_values: List[int],
) -> Dict:
    """对一条 JSONL 中所有记录计算指标。

    返回：
    - first_stage_metrics: 取 first_stage_top10 截断到 k 后的指标
    - llm_rerank_metrics: 取 llm_ranked_asins 截断到 k 后的指标
    """
    if not records:
        raise ValueError("No records to evaluate")
    first_stage_records = []
    llm_rerank_records = []
    failed_n = 0
    for r in records:
        relevant_asin = r["asin"]
        first_stage_top = r.get("first_stage_top10") or []
        llm_ranked = r.get("llm_final_ranked_asins") or r.get("llm_ranked_asins") or []
        if not first_stage_top:
            raise ValueError(f"first_stage_top10 missing for user={r['user_id']} query={r['query'][:80]}")
        if not llm_ranked:
            raise ValueError(f"llm_ranked_asins missing for user={r['user_id']} query={r['query'][:80]}")
        if r.get("status") == "failed":
            failed_n += 1
        first_stage_records.append(compute_metrics(relevant_asin, first_stage_top, k_values))
        llm_rerank_records.append(compute_metrics(relevant_asin, llm_ranked, k_values))

    return {
        "first_stage": {
            "metrics": compute_average_metrics(first_stage_records, k_values),
            "n_records": len(first_stage_records),
        },
        "llm_rerank": {
            "metrics": compute_average_metrics(llm_rerank_records, k_values),
            "n_records": len(llm_rerank_records),
        },
        "failed_n": failed_n,
    }


def print_comparison_table(all_results: List[Dict], k_values: List[int]) -> None:
    """打印 first-stage vs LLM rerank 对比表。"""
    log("\n" + "=" * 120)
    log(f"{'category':<24} {'retriever':<10} {'query_type':<10} "
        f"{'P@10_first':>11} {'P@10_llm':>9} {'ΔP@10':>8} "
        f"{'N@10_first':>11} {'N@10_llm':>9} {'ΔN@10':>8} "
        f"{'MRR@10_first':>13} {'MRR@10_llm':>11} {'ΔMRR@10':>10} "
        f"{'#records':>8} {'#failed':>8}")
    log("-" * 120)
    for r in all_results:
        fs = r["first_stage_metrics"]
        llm = r["llm_rerank_metrics"]
        p_first = fs["metrics"].get("P@10", 0.0)
        p_llm = llm["metrics"].get("P@10", 0.0)
        n_first = fs["metrics"].get("N@10", 0.0)
        n_llm = llm["metrics"].get("N@10", 0.0)
        mrr_first = fs["metrics"].get("MR@10", 0.0)
        mrr_llm = llm["metrics"].get("MR@10", 0.0)
        failed_n = r.get("failed_n", 0)
        log(
            f"{r['category']:<24} {r['retriever']:<10} {r['query_type']:<10} "
            f"{p_first:>11.4f} {p_llm:>9.4f} {(p_llm - p_first):>+8.4f} "
            f"{n_first:>11.4f} {n_llm:>9.4f} {(n_llm - n_first):>+8.4f} "
            f"{mrr_first:>13.4f} {mrr_llm:>11.4f} {(mrr_llm - mrr_first):>+10.4f} "
            f"{fs['n_records']:>8d} {failed_n:>8d}"
        )
    log("-" * 120)


def evaluate_for_category(category_name: str, query_types: list = None) -> Dict:
    if query_types is None:
        query_types = ["correct", "noisy"]
    log("=" * 80)
    log(f"Stage 15 LLM 重排序评估 | {category_name} | query_types={query_types}")
    log("=" * 80)

    cfg = load_rerank_config()
    category_config = get_category_config(category_name)
    paths = get_stage15_paths(category_name)

    top_k = cfg["rerank"]["top_k_candidates"]
    k_values = cfg["rerank"]["k_values_for_eval"]

    log(f"  first_stage_retrievers = {cfg['rerank']['first_stage_retrievers']}")
    log(f"  top_k_candidates = {top_k}")
    log(f"  k_values = {k_values}")

    all_results: List[Dict] = []
    for retriever_name in cfg["rerank"]["first_stage_retrievers"]:
        for query_type in query_types:
            in_file = os.path.join(
                paths["output_dir"],
                f"{retriever_name}__{query_type}_top{top_k}_rerank.jsonl",
            )
            if not os.path.exists(in_file):
                log(f"  [skip] {in_file} 不存在，跳过")
                continue
            log(f"\n  [eval] {retriever_name} / {query_type}")
            records = read_jsonl(in_file)
            log(f"    loaded {len(records)} 条记录")
            eval_result = evaluate_one_file(records, k_values)
            all_results.append({
                "category": category_name,
                "retriever": retriever_name,
                "query_type": query_type,
                "first_stage_metrics": eval_result["first_stage"],
                "llm_rerank_metrics": eval_result["llm_rerank"],
                "failed_n": eval_result.get("failed_n", 0),
            })

    if not all_results:
        raise FileNotFoundError(
            f"No rerank JSONL found under {paths['output_dir']}"
        )

    log(f"\n[summary] {category_name} first-stage vs LLM rerank 对比")
    print_comparison_table(all_results, k_values)

    summary_file = os.path.join(
        paths["output_dir"],
        f"rerank_eval_summary_top{top_k}.json",
    )
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "category": category_name,
                "top_k_candidates": top_k,
                "k_values": k_values,
                "results": all_results,
            },
            f,
            indent=2,
            default=str,
            ensure_ascii=False,
        )
    log(f"\n[output] 汇总写入 {summary_file}")
    return {"category": category_name, "results": all_results}
