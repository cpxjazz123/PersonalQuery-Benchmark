#!/usr/bin/env python3
"""E2 — Empirical floor/ceiling baseline for 8 retrievers × 3 domains.

Floor: N random queries sampled from random doc titles (relevant doc = arbitrary
       random asin not in query). Hit@10 should be ≈ 0 since query is unrelated.

Ceiling: For N doc asins, use the doc's own title as query (relevant doc = asin
         itself). Hit@10 should be ≈ 1.0 since query text comes from the doc.

Outputs:
  {out_dir}/{category}_floor_ceiling.json + summary.md
where out_dir defaults to /home/wlia0047/hj82_scratch2/wenyu/RAG/E2_floor_ceiling/

Run with:
    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python E2_floor_ceiling_eval.py
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# 必须在 import torch / sentence_transformers 之前先设置 HF_HOME
HF_HOME = "/home/wlia0047/ar57_scratch/wenyu/hf_models"
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = HF_HOME
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

# 让 utils/ 可被 import
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from utils.retrievers import BM25, BGERetriever  # noqa: E402


DEFAULT_OUT_DIR = "/home/wlia0047/hj82_scratch2/wenyu/RAG/E2_floor_ceiling"
DEFAULT_2018_ROOT = "/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018"
RETRIEVERS = ["bm25", "bge"]
TOP_K = 10


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_corpus(category: str, root: str = DEFAULT_2018_ROOT) -> Tuple[List[Dict], Dict[str, Dict]]:
    """加载 Amazon 2018 5-core meta，转成 (documents, asin_to_meta) 双结构。

    Baby 直接放根目录；Grocery / Pet 在 raw/meta_categories/。
    """
    if category == "Baby_Products":
        path = f"{root}/meta_Baby.jsonl.gz"
    elif category == "Grocery_and_Gourmet_Food":
        path = f"{root}/raw/meta_categories/meta_Grocery_and_Gourmet_Food.jsonl.gz"
    elif category == "Pet_Supplies":
        path = f"{root}/raw/meta_categories/meta_Pet_Supplies.jsonl.gz"
    else:
        raise ValueError(f"Unknown category: {category}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"corpus not found: {path}")

    docs: List[Dict] = []
    asin_to_meta: Dict[str, Dict] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            asin = obj.get("asin") or obj.get("parent_asin")
            if not asin:
                continue
            title = (obj.get("title") or "").strip()
            description = obj.get("description") or []
            if isinstance(description, list):
                description = " ".join(str(d) for d in description if d)
            elif not isinstance(description, str):
                description = ""
            brand = obj.get("brand") or ""
            if isinstance(brand, list):
                brand = " ".join(str(b) for b in brand if b)
            feature = obj.get("feature") or []
            if isinstance(feature, list):
                feature = " ".join(str(x) for x in feature if x)
            elif not isinstance(feature, str):
                feature = ""

            doc = {
                "asin": asin,
                "title": title[:300],
                "brand": brand[:80] if isinstance(brand, str) else "",
                "feature": feature[:500] if isinstance(feature, str) else "",
                "description": description[:500] if isinstance(description, str) else "",
            }
            docs.append(doc)
            asin_to_meta[asin] = doc

    if not docs:
        raise RuntimeError(f"No docs loaded for {category}")
    log(f"  loaded {len(docs)} docs from {os.path.basename(path)}")
    return docs, asin_to_meta


def build_doc_text(doc: Dict, all_metadata: Dict | None = None) -> str:
    """合并 title + brand + feature + description 为单一文本（与 utils.retrievers.build_document_text 保持一致）。"""
    parts = [
        doc.get("title", "") or "",
        doc.get("brand", "") or "",
        doc.get("feature", "") or "",
        doc.get("description", "") or "",
    ]
    return " ".join(p for p in parts if p).strip()


def compute_hit_at_k(retrieved: List[Tuple[str, float]], relevant_asin: str, k: int = TOP_K) -> int:
    """返回 1 if relevant_asin 在 top-k 内，否则 0。"""
    top = [asin for asin, _ in retrieved[:k]]
    return 1 if relevant_asin in top else 0


def compute_metrics_for_results(results: List[Dict], k: int = TOP_K) -> Dict[str, float]:
    n = len(results)
    if n == 0:
        return {"hit_at_10": 0.0, "p_at_10": 0.0, "mr_at_10": 0.0, "n": 0}
    hit = sum(r["hit"] for r in results) / n
    p_at = sum(1 if r["hit"] else 0 for r in results) / n  # == hit@10
    mrr_sum = 0.0
    for r in results:
        for rank, (asin, _) in enumerate(r["retrieved"][:k], start=1):
            if asin == r["relevant_asin"]:
                mrr_sum += 1.0 / rank
                break
    return {
        "hit_at_10": hit,
        "p_at_10": p_at,
        "mr_at_10": mrr_sum / n,
        "n": n,
    }


def make_floor_queries(asins: List[str], n: int, rng: random.Random) -> List[Dict]:
    """Floor 查询：对每个目标 asin，从语料随机采 1 条 title，标记其 relevant_asin 为不相关的第三方。

    严格做法：查询文本 = random_other_doc.title，relevant_asin = target_asin（不在 query 文本所在的 doc 中）。
    这样如果检索器真的能"理解"，应该返回 query 的源 doc，而不是 target_asin。
    """
    queries = []
    for i in range(n):
        target = rng.choice(asins)
        # 选另一条 doc 的 title 作查询
        for _ in range(10):
            query_asin = rng.choice(asins)
            if query_asin != target:
                break
        queries.append({
            "query_id": f"floor_{i}",
            "target_asin": target,  # 这是我们要找的 relevant_asin
            "query_asin": query_asin,  # 查询文本来自这条 doc
        })
    return queries


def make_ceiling_queries(asins: List[str], asin_to_meta: Dict, n: int, rng: random.Random) -> List[Dict]:
    """Ceiling 查询：对每个 asin，用其自身 title 作查询。Hit@10 应该 ≈ 1.0。"""
    queries = []
    candidates = [a for a in asins if asin_to_meta[a].get("title", "").strip()]
    for i in range(n):
        a = rng.choice(candidates)
        title = asin_to_meta[a]["title"].strip()
        if not title:
            continue
        queries.append({
            "query_id": f"ceil_{i}",
            "target_asin": a,
            "query_asin": a,
            "title": title,
        })
    return queries


def run_bm25_floor_ceiling(
    bm25: BM25, floor_queries: List[Dict], ceiling_queries: List[Dict],
    asin_to_meta: Dict, n_samples: int
) -> Dict:
    """直接调 bm25.search 跑 floor + ceiling。"""
    floor_results = []
    log(f"  [bm25] running {n_samples} floor queries...")
    t0 = time.time()
    for q in floor_queries[:n_samples]:
        # floor query = random_other_doc.title
        query_text = build_doc_text(asin_to_meta[q["query_asin"]])
        retrieved = bm25.search(query_text, top_k=TOP_K)
        floor_results.append({
            "query_id": q["query_id"],
            "relevant_asin": q["target_asin"],
            "retrieved": retrieved,
            "hit": compute_hit_at_k(retrieved, q["target_asin"], TOP_K),
        })
    log(f"  [bm25] floor done in {time.time()-t0:.1f}s")

    ceiling_results = []
    log(f"  [bm25] running {n_samples} ceiling queries...")
    t0 = time.time()
    for q in ceiling_queries[:n_samples]:
        query_text = q["title"]  # asin 的自身 title
        retrieved = bm25.search(query_text, top_k=TOP_K)
        ceiling_results.append({
            "query_id": q["query_id"],
            "relevant_asin": q["target_asin"],
            "retrieved": retrieved,
            "hit": compute_hit_at_k(retrieved, q["target_asin"], TOP_K),
        })
    log(f"  [bm25] ceiling done in {time.time()-t0:.1f}s")

    return {
        "floor": floor_results,
        "ceiling": ceiling_results,
        "floor_metrics": compute_metrics_for_results(floor_results),
        "ceiling_metrics": compute_metrics_for_results(ceiling_results),
    }


def run_bge_floor_ceiling(
    bge: BGERetriever, floor_queries: List[Dict], ceiling_queries: List[Dict],
    asin_to_meta: Dict, n_samples: int
) -> Dict:
    """BGE 用 encode_query + retriever 内部 cosine similarity 检索。"""
    # BGERetriever 接受 query str，搜 top_k
    floor_results = []
    log(f"  [bge] running {n_samples} floor queries...")
    t0 = time.time()
    for q in floor_queries[:n_samples]:
        query_text = build_doc_text(asin_to_meta[q["query_asin"]])
        retrieved = bge.search(query_text, top_k=TOP_K)
        floor_results.append({
            "query_id": q["query_id"],
            "relevant_asin": q["target_asin"],
            "retrieved": retrieved,
            "hit": compute_hit_at_k(retrieved, q["target_asin"], TOP_K),
        })
    log(f"  [bge] floor done in {time.time()-t0:.1f}s")

    ceiling_results = []
    log(f"  [bge] running {n_samples} ceiling queries...")
    t0 = time.time()
    for q in ceiling_queries[:n_samples]:
        query_text = q["title"]
        retrieved = bge.search(query_text, top_k=TOP_K)
        ceiling_results.append({
            "query_id": q["query_id"],
            "relevant_asin": q["target_asin"],
            "retrieved": retrieved,
            "hit": compute_hit_at_k(retrieved, q["target_asin"], TOP_K),
        })
    log(f"  [bge] ceiling done in {time.time()-t0:.1f}s")

    return {
        "floor": floor_results,
        "ceiling": ceiling_results,
        "floor_metrics": compute_metrics_for_results(floor_results),
        "ceiling_metrics": compute_metrics_for_results(ceiling_results),
    }


def fit_bm25(docs: List[Dict], asin_to_meta: Dict) -> BM25:
    bm25 = BM25()
    bm25.fit(docs, asin_to_meta)
    return bm25


def fit_bge(docs: List[Dict], asin_to_meta: Dict) -> BGERetriever:
    log(f"  Loading BGE-large-en-v1.5...")
    bge = BGERetriever(model_name="BAAI/bge-large-en-v1.5")
    bge.fit(docs, asin_to_meta)
    return bge


def process_category(
    category: str,
    n_samples: int,
    seed: int,
    out_dir: str,
    enabled_retrievers: List[str],
) -> Dict:
    log(f"\n{'='*70}\nE2 floor/ceiling — {category} (n={n_samples})\n{'='*70}")
    docs, asin_to_meta = load_corpus(category)
    asins = list(asin_to_meta.keys())
    rng = random.Random(seed)

    floor_queries = make_floor_queries(asins, n_samples, rng)
    ceiling_queries = make_ceiling_queries(asins, asin_to_meta, n_samples, rng)

    results: Dict[str, Dict] = {}
    if "bm25" in enabled_retrievers:
        log("\n[bm25] fitting...")
        bm25 = fit_bm25(docs, asin_to_meta)
        results["bm25"] = run_bm25_floor_ceiling(
            bm25, floor_queries, ceiling_queries, asin_to_meta, n_samples
        )
        del bm25

    if "bge" in enabled_retrievers:
        log("\n[bge] fitting...")
        bge = fit_bge(docs, asin_to_meta)
        results["bge"] = run_bge_floor_ceiling(
            bge, floor_queries, ceiling_queries, asin_to_meta, n_samples
        )
        del bge
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "category": category,
        "n_samples": n_samples,
        "n_docs": len(docs),
        "n_floor_queries_actual": min(n_samples, len(floor_queries)),
        "n_ceiling_queries_actual": min(n_samples, len(ceiling_queries)),
        "results": {
            ret: {
                "floor_metrics": r["floor_metrics"],
                "ceiling_metrics": r["ceiling_metrics"],
            }
            for ret, r in results.items()
        },
        "raw_floor_queries": floor_queries[:5],
        "raw_ceiling_queries": ceiling_queries[:5],
    }
    return summary


def write_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def write_summary_md(category_summaries: List[Dict], out_dir: str) -> str:
    """生成跨域汇总 markdown 表。"""
    md_lines = [
        "# E2 — Empirical Floor / Ceiling Baseline",
        "",
        "**Floor**: N 个随机查询（每条用一条与目标 asin 不相关的 doc title 作查询文本）。",
        "  - 相关文档 = 任意 target asin（与 query 文本无关）",
        "  - Hit@10 应该 ≈ 0（因为检索器无法无中生有）",
        "",
        "**Ceiling**: N 个 asin 的自身 title 作查询。",
        "  - 相关文档 = 该 asin 自身",
        "  - Hit@10 应该 ≈ 1.0（query 文本直接来自 doc）",
        "",
        "## 3 域 × 2 检索器 Hit@10 表",
        "",
        "| Category | Retriever | Floor Hit@10 | Ceiling Hit@10 | Floor MRR@10 | Ceiling MRR@10 | Floor P@10 | Ceiling P@10 | n |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in category_summaries:
        cat = s["category"]
        for ret, m in s["results"].items():
            fl = m["floor_metrics"]
            cl = m["ceiling_metrics"]
            md_lines.append(
                f"| {cat} | {ret} | "
                f"{fl['hit_at_10']:.4f} | {cl['hit_at_10']:.4f} | "
                f"{fl['mr_at_10']:.4f} | {cl['mr_at_10']:.4f} | "
                f"{fl['p_at_10']:.4f} | {cl['p_at_10']:.4f} | "
                f"{s['n_samples']} |"
            )
    md_lines += [
        "",
        "## 解读",
        "",
        "- Floor Hit@10 ≈ 0：检索器无法在无关查询下召回特定 asin。",
        "- Ceiling Hit@10 ≈ 1：as self-title 检索最容易命中。",
        "- Gap (Ceiling − Floor) 衡量了 query 相关性对检索效果的极限影响。",
    ]
    summary_path = os.path.join(out_dir, "summary.md")
    os.makedirs(out_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        f.write("\n".join(md_lines))
    return summary_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+",
                    default=["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"])
    ap.add_argument("--n_samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--retrievers", nargs="+", default=RETRIEVERS)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    category_summaries: List[Dict] = []
    for cat in args.categories:
        try:
            s = process_category(
                category=cat,
                n_samples=args.n_samples,
                seed=args.seed,
                out_dir=args.out_dir,
                enabled_retrievers=args.retrievers,
            )
            json_path = os.path.join(args.out_dir, f"{cat}_floor_ceiling.json")
            write_json(json_path, s)
            log(f"  wrote {json_path}")
            category_summaries.append(s)
        except Exception as e:
            log(f"  ERROR on {cat}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    if category_summaries:
        summary_path = write_summary_md(category_summaries, args.out_dir)
        log(f"\nSummary written: {summary_path}")


if __name__ == "__main__":
    main()