#!/usr/bin/env python3
"""E1 — Full quantity noisy retrieval.

Loads expanded 07_inject_noisy noisy pairs (656/302/440 per domain).
Runs BGE (fastest dense retriever, cached at 06_retrieval/document_cache/<cat>)
on both clean and noisy queries for the (uid, asin) overlap with 04_query.
Computes Hit@10 on the gold asin, paired Wilcoxon, Spearman across retrievers.

Outputs /home/wlia0047/hj82_scratch2/wenyu/RAG/E1_multimetric/full_quantity_<cat>.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
DEFAULT_RESULT = REPO_ROOT / "result" / "personal_query"
DEFAULT_OUT = Path("/home/wlia0047/hj82_scratch2/wenyu/RAG/E1_multimetric")
SCRATCH_RESULT = Path("/home/wlia0047/ar57_scratch/wenyu/result/personal_query")
CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_noisy_pairs(category: str) -> List[Dict]:
    p = DEFAULT_RESULT / "07_inject_noisy" / category / "noisy_query.json"
    if not p.exists():
        return []
    with open(p) as f:
        return json.load(f)


def load_clean_queries(category: str) -> Dict[Tuple[str, str], List[Dict]]:
    """从 04_query 取所有 (uid, asin) → syntax_depth_queries 列表."""
    p = DEFAULT_RESULT / "04_query" / category / "query_by_syntax_depth_no_depth_check_10.json"
    with open(p) as f:
        data = json.load(f)
    out = {}
    for r in data:
        key = (r.get("user_id"), r.get("asin"))
        out[key] = r.get("syntax_depth_queries", [])
    return out


def get_doc_cache_path(category: str) -> Path:
    """08_retrieval 文档索引缓存路径."""
    return DEFAULT_OUT / f"bge_index_{category}.pkl"


def load_bge_index(category: str, device: torch.device):
    """Load BGERetriever + cached document embeddings."""
    sys.path.insert(0, str(REPO_ROOT / "06_retrieval" / "utils"))
    os.environ.setdefault("HF_HOME", "/home/wlia0047/.cache/huggingface/hub")
    from retrievers import BGERetriever
    import retrievers as r
    # use cuda if available else cpu
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    r._require_cuda_device = lambda: device

    cache = get_doc_cache_path(category)
    if cache.exists():
        with open(cache, "rb") as f:
            cache_data = pickle.load(f)
        log(f"  [BGE/{category}] loaded cache with {len(cache_data['asins'])} docs")
    else:
        log(f"  [BGE/{category}] NO cache at {cache}")
        return None, None, None

    obj = BGERetriever()
    obj.device = device
    obj.doc_ids = cache_data["asins"]
    obj.doc_embeddings = torch.from_numpy(cache_data["embeddings"]).float().to(device)
    return obj, cache_data, device


def encode_queries_bge(obj, queries: List[str], batch_size: int = 32) -> np.ndarray:
    return obj.encode_queries(queries, batch_size=batch_size)


def retrieve_topk(obj, query_emb: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
    """按 query embedding 检索 top-k asin."""
    device = obj.device
    if isinstance(query_emb, np.ndarray):
        q = torch.from_numpy(query_emb).float().to(device)
        if q.ndim == 1:
            q = q.unsqueeze(0)
    else:
        q = query_emb.to(device)
    from sentence_transformers import util
    scores = util.cos_sim(q, obj.doc_embeddings)[0]
    topk_values, topk_indices = torch.topk(scores, k=min(top_k, len(obj.doc_ids)))
    return [(obj.doc_ids[i], float(topk_values[j])) for j, i in enumerate(topk_indices.tolist())]


def hit_at_k(retrieved: List[Tuple[str, float]], gold: str, k: int = 10) -> float:
    return 1.0 if gold in [d for d, _ in retrieved[:k]] else 0.0


def mrr_at_k(retrieved: List[Tuple[str, float]], gold: str, k: int = 10) -> float:
    for i, (d, _) in enumerate(retrieved[:k]):
        if d == gold:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: List[Tuple[str, float]], gold: str, k: int = 10) -> float:
    for i, (d, _) in enumerate(retrieved[:k]):
        if d == gold:
            return 1.0 / np.log2(i + 2)
    return 0.0


def run_for_category(category: str, retriever_name: str = "bge") -> Dict:
    log(f"=== {category} ({retriever_name}) ===")
    noise_pairs = load_noisy_pairs(category)
    log(f"  loaded {len(noise_pairs)} noisy pairs")
    if not noise_pairs:
        return {}
    clean_lookup = load_clean_queries(category)
    log(f"  clean_lookup has {len(clean_lookup)} (uid, asin) keys")
    obj, cache_data, device = load_bge_index(category, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if obj is None:
        return {}

    # group by (uid, asin) to encode clean + noisy together
    by_pair: Dict[Tuple[str, str], List[Dict]] = {}
    for p in noise_pairs:
        key = (p.get("uid"), p.get("asin"))
        by_pair.setdefault(key, []).append(p)
    log(f"  {len(by_pair)} unique (uid, asin)")

    per_query_metrics = {"correct": [], "noisy": []}
    n_processed = 0
    for key, pairs in by_pair.items():
        uid, asin = key
        # find clean query from 04_query
        clean_qs = clean_lookup.get(key, [])
        # use the accepted candidate from syntax_depth_query
        clean_query = None
        target_idx = None
        for sq in clean_qs:
            if sq.get("query") and sq.get("accepted_candidate_index") is not None:
                clean_query = sq["query"]
                target_idx = sq["accepted_candidate_index"]
                break
        if not clean_query:
            continue
        # encode clean query
        with torch.no_grad():
            clean_emb = encode_queries_bge(obj, [clean_query], batch_size=1)[0]
            retrieved_clean = retrieve_topk(obj, clean_emb, top_k=10)
            h_clean = hit_at_k(retrieved_clean, asin, 10)
            mrr_clean = mrr_at_k(retrieved_clean, asin, 10)
            ndcg_clean = ndcg_at_k(retrieved_clean, asin, 10)
            per_query_metrics["correct"].append({"uid": uid, "asin": asin, "H@10": h_clean, "MR@10": mrr_clean, "N@10": ndcg_clean})
        # encode each noisy query
        for p in pairs:
            noisy_q = p.get("noisy_query")
            if not noisy_q:
                continue
            with torch.no_grad():
                noisy_emb = encode_queries_bge(obj, [noisy_q], batch_size=1)[0]
                retrieved_noisy = retrieve_topk(obj, noisy_emb, top_k=10)
                h_noisy = hit_at_k(retrieved_noisy, asin, 10)
                mrr_noisy = mrr_at_k(retrieved_noisy, asin, 10)
                ndcg_noisy = ndcg_at_k(retrieved_noisy, asin, 10)
                per_query_metrics["noisy"].append({"uid": uid, "asin": asin, "H@10": h_noisy, "MR@10": mrr_noisy, "N@10": ndcg_noisy})
        n_processed += 1
        if n_processed % 100 == 0:
            log(f"    processed {n_processed} pairs")
    log(f"  done: {n_processed} pairs, correct={len(per_query_metrics['correct'])}, noisy={len(per_query_metrics['noisy'])}")
    del obj
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return per_query_metrics


def wilcoxon_paired(correct: List[float], noisy: List[float]) -> Dict:
    from scipy.stats import wilcoxon
    c = np.asarray(correct, dtype=np.float64)
    n = np.asarray(noisy, dtype=np.float64)
    n_pairs = min(len(c), len(n))
    if n_pairs < 2:
        return {"stat": float("nan"), "p": float("nan"), "n_pairs": n_pairs}
    try:
        stat, p = wilcoxon(c, n)
        return {"stat": float(stat), "p": float(p), "n_pairs": n_pairs}
    except ValueError as e:
        return {"stat": float("nan"), "p": float("nan"), "n_pairs": n_pairs, "err": str(e)}


def main() -> None:
    DEFAULT_OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for cat in CATEGORIES:
        per_q = run_for_category(cat, "bge")
        if not per_q:
            summary[cat] = {"error": "no data"}
            continue
        h_c = [r["H@10"] for r in per_q["correct"]]
        h_n = [r["H@10"] for r in per_q["noisy"]]
        m_c = [r["MR@10"] for r in per_q["correct"]]
        m_n = [r["MR@10"] for r in per_q["noisy"]]
        n_c = [r["N@10"] for r in per_q["correct"]]
        n_n = [r["N@10"] for r in per_q["noisy"]]
        # align by (uid, asin)
        c_map = {(r["uid"], r["asin"]): r for r in per_q["correct"]}
        n_map = {(r["uid"], r["asin"]): r for r in per_q["noisy"]}
        common = set(c_map) & set(n_map)
        h_c_aligned = [c_map[k]["H@10"] for k in common]
        h_n_aligned = [n_map[k]["H@10"] for k in common]
        summary[cat] = {
            "retriever": "bge",
            "n_correct": len(h_c),
            "n_noisy": len(h_n),
            "n_paired": len(common),
            "H@10_correct_mean": float(np.mean(h_c)) if h_c else 0.0,
            "H@10_noisy_mean": float(np.mean(h_n)) if h_n else 0.0,
            "MR@10_correct_mean": float(np.mean(m_c)) if m_c else 0.0,
            "MR@10_noisy_mean": float(np.mean(m_n)) if m_n else 0.0,
            "N@10_correct_mean": float(np.mean(n_c)) if n_c else 0.0,
            "N@10_noisy_mean": float(np.mean(n_n)) if n_n else 0.0,
            "wilcoxon_H@10": wilcoxon_paired(h_c_aligned, h_n_aligned),
            "wilcoxon_MR@10": wilcoxon_paired([c_map[k]["MR@10"] for k in common], [n_map[k]["MR@10"] for k in common]),
            "wilcoxon_N@10": wilcoxon_paired([c_map[k]["N@10"] for k in common], [n_map[k]["N@10"] for k in common]),
        }
        log(f"  {cat}: H@10 correct={summary[cat]['H@10_correct_mean']:.4f} noisy={summary[cat]['H@10_noisy_mean']:.4f} n_paired={summary[cat]['n_paired']}")
    out_path = DEFAULT_OUT / "full_quantity_bge.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"wrote {out_path}")

    # write summary.md
    md = ["# E1 — Full-quantity Multi-metric + Wilcoxon (BGE-large-en-v1.5)\n",
          "Inputs: 04_query syntax_depth_queries (clean) + 07_inject_noisy (noisy)\n",
          "Index: 10K Amazon meta distractors + gold asins, BGE-encoded\n",
          "Method: paired Wilcoxon (correct vs noisy per query) per metric per domain\n",
          "\n## Per-domain metrics\n",
          "| Category | n_paired | H@10_c | H@10_n | ΔH@10 | MR@10_c | MR@10_n | N@10_c | N@10_n |\n",
          "|---|---|---|---|---|---|---|---|---|"]
    for cat, d in summary.items():
        if "error" in d:
            md.append(f"| {cat} | ERROR | | | | | | | |")
            continue
        md.append(
            f"| {cat} | {d['n_paired']} | {d['H@10_correct_mean']:.4f} | "
            f"{d['H@10_noisy_mean']:.4f} | {d['H@10_noisy_mean'] - d['H@10_correct_mean']:+.4f} | "
            f"{d['MR@10_correct_mean']:.4f} | {d['MR@10_noisy_mean']:.4f} | "
            f"{d['N@10_correct_mean']:.4f} | {d['N@10_noisy_mean']:.4f} |"
        )
    md += ["\n## Paired Wilcoxon (correct vs noisy)\n",
           "| Category | Metric | stat | p | n |\n",
           "|---|---|---|---|---|"]
    for cat, d in summary.items():
        if "error" in d:
            continue
        for m in ["H@10", "MR@10", "N@10"]:
            w = d[f"wilcoxon_{m}"]
            p_str = "NaN" if str(w['p']) == 'nan' else f"{w['p']:.4e}"
            md.append(f"| {cat} | {m} | {w['stat']:.4f} | {p_str} | {w['n_pairs']} |")
    md_path = DEFAULT_OUT / "full_quantity_summary.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    log(f"wrote {md_path}")
    log("=== done ===")


if __name__ == "__main__":
    main()