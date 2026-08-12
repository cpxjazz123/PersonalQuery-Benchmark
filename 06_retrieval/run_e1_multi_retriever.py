#!/usr/bin/env python3
"""E1 — Multi-retriever (BGE + E5 + MiniLM) full-quantity noisy retrieval.

For each (uid, asin) pair with noisy queries:
- Encode clean + noisy queries with each retriever
- Retrieve top-10 from each retriever's index
- Compute H@10, MR@10, N@10
- Run paired Wilcoxon + Spearman across retrievers
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
CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]
RETRIEVERS = ["bge", "e5", "minilm"]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_noisy_pairs(category: str) -> List[Dict]:
    p = DEFAULT_RESULT / "07_inject_noisy" / category / "noisy_query.json"
    with open(p) as f:
        return json.load(f)


def load_clean_queries(category: str) -> Dict[Tuple[str, str], Dict]:
    p = DEFAULT_RESULT / "04_query" / category / "query_by_syntax_depth_no_depth_check_10.json"
    with open(p) as f:
        data = json.load(f)
    out = {}
    for r in data:
        key = (r.get("user_id"), r.get("asin"))
        # find accepted candidate
        for sq in r.get("syntax_depth_queries", []):
            if sq.get("query") and sq.get("accepted_candidate_index") is not None:
                out[key] = sq
                break
    return out


def load_index(category: str) -> Dict:
    p = DEFAULT_OUT / f"multi_index_{category}.pkl"
    with open(p, "rb") as f:
        return pickle.load(f)


def load_retriever(name: str, device: torch.device):
    sys.path.insert(0, str(REPO_ROOT / "06_retrieval" / "utils"))
    os.environ.setdefault("HF_HOME", "/home/wlia0047/.cache/huggingface/hub")
    import retrievers as r
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    r._require_cuda_device = lambda: device
    cls_map = {"bge": r.BGERetriever, "e5": r.E5Retriever, "minilm": r.DenseRetriever}
    obj = cls_map[name]()
    obj.device = device
    return obj, device


def hit_at_k(retrieved: List[str], gold: str, k: int = 10) -> float:
    return 1.0 if gold in retrieved[:k] else 0.0


def mrr_at_k(retrieved: List[str], gold: str, k: int = 10) -> float:
    for i, d in enumerate(retrieved[:k]):
        if d == gold:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: List[str], gold: str, k: int = 10) -> float:
    for i, d in enumerate(retrieved[:k]):
        if d == gold:
            return 1.0 / np.log2(i + 2)
    return 0.0


def retrieve_topk(emb: np.ndarray, doc_embs: np.ndarray, doc_ids: List[str], top_k: int = 10) -> List[str]:
    """cosine sim top-k, return doc_ids list."""
    q = torch.from_numpy(emb).float()
    if q.ndim == 1:
        q = q.unsqueeze(0)
    d = torch.from_numpy(doc_embs).float()
    from sentence_transformers import util
    scores = util.cos_sim(q, d)[0]
    _, idx = torch.topk(scores, k=min(top_k, len(doc_ids)))
    return [doc_ids[i] for i in idx.tolist()]


def run_for_category(category: str) -> Dict:
    log(f"=== {category} ===")
    idx = load_index(category)
    asins = idx["asins"]
    embeddings = idx["embeddings"]  # dict retriever -> np.array
    log(f"  index: {len(asins)} asins, retrievers {list(embeddings.keys())}")

    clean_lookup = load_clean_queries(category)
    noise_pairs = load_noisy_pairs(category)
    log(f"  clean_lookup={len(clean_lookup)}, noisy_pairs={len(noise_pairs)}")

    by_pair: Dict[Tuple[str, str], List[Dict]] = {}
    for p in noise_pairs:
        key = (p.get("uid"), p.get("asin"))
        by_pair.setdefault(key, []).append(p)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # results: per-retriever, per-(uid, asin), correct/noisy
    results: Dict[str, Dict[str, Dict[str, float]]] = {r: {"correct": [], "noisy": []} for r in RETRIEVERS}
    keys_processed: List[Tuple[str, str]] = []

    # Load each retriever once
    objs: Dict[str, object] = {}
    for name in RETRIEVERS:
        obj, _ = load_retriever(name, device)
        objs[name] = obj
    log(f"  loaded all {len(objs)} retrievers")

    # Pre-encode all clean queries in batch
    clean_keys = []
    clean_texts = []
    for key, pairs in by_pair.items():
        clean = clean_lookup.get(key)
        if not clean:
            continue
        clean_keys.append(key)
        clean_texts.append(clean["query"])
    log(f"  encoding {len(clean_texts)} clean queries per retriever...")
    clean_embs_by_name: Dict[str, np.ndarray] = {}
    for name in RETRIEVERS:
        obj = objs[name]
        with torch.no_grad():
            clean_embs_by_name[name] = obj.encode_queries(clean_texts, batch_size=64)
    log(f"  clean embeddings done")

    # Pre-encode all noisy queries
    noisy_keys_idx: Dict[int, Tuple[str, str]] = {}  # idx -> (uid, asin)
    noisy_texts = []
    n_per_pair = []
    for idx, key in enumerate(clean_keys):
        pairs = by_pair.get(key, [])
        n_per_pair.append(len(pairs))
        for p in pairs:
            noisy_keys_idx[len(noisy_texts)] = (key[0], key[1])
            noisy_texts.append(p.get("noisy_query") or "")
    log(f"  encoding {len(noisy_texts)} noisy queries per retriever...")
    noisy_embs_by_name: Dict[str, np.ndarray] = {}
    for name in RETRIEVERS:
        obj = objs[name]
        with torch.no_grad():
            noisy_embs_by_name[name] = obj.encode_queries(noisy_texts, batch_size=64)
    log(f"  noisy embeddings done")

    # retrieve & score
    for name in RETRIEVERS:
        clean_embs = clean_embs_by_name[name]
        for idx, key in enumerate(clean_keys):
            uid, asin = key
            retrieved = retrieve_topk(clean_embs[idx], embeddings[name], asins, top_k=10)
            h = hit_at_k(retrieved, asin, 10)
            m = mrr_at_k(retrieved, asin, 10)
            n = ndcg_at_k(retrieved, asin, 10)
            results[name]["correct"].append({"uid": uid, "asin": asin, "H@10": h, "MR@10": m, "N@10": n})
        # noisy
        noisy_embs = noisy_embs_by_name[name]
        start = 0
        for idx, key in enumerate(clean_keys):
            n_pairs = n_per_pair[idx]
            uid, asin = key
            for j in range(n_pairs):
                emb = noisy_embs[start + j]
                if (start + j) >= len(noisy_embs):
                    continue
                retrieved_n = retrieve_topk(emb, embeddings[name], asins, top_k=10)
                h = hit_at_k(retrieved_n, asin, 10)
                m = mrr_at_k(retrieved_n, asin, 10)
                n = ndcg_at_k(retrieved_n, asin, 10)
                results[name]["noisy"].append({"uid": uid, "asin": asin, "H@10": h, "MR@10": m, "N@10": n})
            start += n_pairs
        keys_processed.append(key)
    del objs
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    log(f"  done: {len(clean_keys)} pairs processed")
    return results


def wilcoxon_paired(c: List[float], n: List[float]) -> Dict:
    from scipy.stats import wilcoxon
    arr_c = np.asarray(c, dtype=np.float64)
    arr_n = np.asarray(n, dtype=np.float64)
    n_pairs = min(len(arr_c), len(arr_n))
    if n_pairs < 2:
        return {"stat": float("nan"), "p": float("nan"), "n_pairs": n_pairs}
    try:
        stat, p = wilcoxon(arr_c, arr_n)
        return {"stat": float(stat), "p": float(p), "n_pairs": n_pairs}
    except ValueError as e:
        return {"stat": float("nan"), "p": float("nan"), "n_pairs": n_pairs, "err": str(e)}


def spearman_across_retrievers(per_retriever: Dict[str, Dict[str, float]], metric: str = "H@10") -> Dict:
    from scipy.stats import spearmanr
    retrievers = list(per_retriever.keys())
    if len(retrievers) < 3:
        return {"rho": float("nan"), "p": float("nan"), "n": len(retrievers)}
    delta = []
    h = []
    for r in retrievers:
        d = per_retriever[r]
        delta.append(d.get(f"{metric}_noisy_mean", 0.0) - d.get(f"{metric}_correct_mean", 0.0))
        h.append(d.get(f"{metric}_correct_mean", 0.0))
    if np.std(delta) < 1e-6 or np.std(h) < 1e-6:
        return {"rho": float("nan"), "p": float("nan"), "n": len(retrievers)}
    rho, p = spearmanr(delta, h)
    return {"rho": float(rho), "p": float(p), "n": len(retrievers)}


def main() -> None:
    DEFAULT_OUT.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Dict] = {}
    for cat in CATEGORIES:
        per_q = run_for_category(cat)
        cat_summary: Dict[str, Dict] = {}
        for name in RETRIEVERS:
            c = per_q[name]["correct"]
            n = per_q[name]["noisy"]
            # align by (uid, asin)
            c_map = {(r["uid"], r["asin"]): r for r in c}
            n_map = {(r["uid"], r["asin"]): r for r in n}
            common = set(c_map) & set(n_map)
            h_c_aligned = [c_map[k]["H@10"] for k in common]
            h_n_aligned = [n_map[k]["H@10"] for k in common]
            m_c_aligned = [c_map[k]["MR@10"] for k in common]
            m_n_aligned = [n_map[k]["MR@10"] for k in common]
            n_c_aligned = [c_map[k]["N@10"] for k in common]
            n_n_aligned = [n_map[k]["N@10"] for k in common]
            cat_summary[name] = {
                "n_correct": len(c),
                "n_noisy": len(n),
                "n_paired": len(common),
                "H@10_correct_mean": float(np.mean(h_c_aligned)) if h_c_aligned else 0.0,
                "H@10_noisy_mean": float(np.mean(h_n_aligned)) if h_n_aligned else 0.0,
                "MR@10_correct_mean": float(np.mean(m_c_aligned)) if m_c_aligned else 0.0,
                "MR@10_noisy_mean": float(np.mean(m_n_aligned)) if m_n_aligned else 0.0,
                "N@10_correct_mean": float(np.mean(n_c_aligned)) if n_c_aligned else 0.0,
                "N@10_noisy_mean": float(np.mean(n_n_aligned)) if n_n_aligned else 0.0,
                "wilcoxon_H@10": wilcoxon_paired(h_c_aligned, h_n_aligned),
                "wilcoxon_MR@10": wilcoxon_paired(m_c_aligned, m_n_aligned),
                "wilcoxon_N@10": wilcoxon_paired(n_c_aligned, n_n_aligned),
            }
            log(f"  {cat}/{name}: H@10 c={cat_summary[name]['H@10_correct_mean']:.4f} n={cat_summary[name]['H@10_noisy_mean']:.4f} paired={cat_summary[name]['n_paired']}")
        # Spearman across retrievers
        cat_summary["spearman_H@10"] = spearman_across_retrievers(cat_summary, "H@10")
        cat_summary["spearman_MR@10"] = spearman_across_retrievers(cat_summary, "MR@10")
        cat_summary["spearman_N@10"] = spearman_across_retrievers(cat_summary, "N@10")
        summary[cat] = cat_summary

    out_path = DEFAULT_OUT / "multi_retriever_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"wrote {out_path}")

    # write markdown
    md = ["# E1 — Multi-retriever full-quantity (BGE + E5 + MiniLM)\n",
          "Inputs: 04_query syntax_depth_queries (clean) + 07_inject_noisy (noisy)\n",
          "Index: 10K Amazon meta distractors + gold asins, encoded by each retriever\n",
          "Method: paired Wilcoxon per (retriever, metric); Spearman across retrievers\n",
          "\n## Per-retriever per-domain H@10 (correct vs noisy)\n",
          "| Category | Retriever | n_paired | H@10_c | H@10_n | ΔH@10 | Wilcoxon p (H@10) |\n",
          "|---|---|---|---|---|---|---|"]
    for cat, d in summary.items():
        for name in RETRIEVERS:
            r = d[name]
            p = r["wilcoxon_H@10"]["p"]
            p_str = "NaN" if str(p) == 'nan' else f"{p:.4e}"
            md.append(
                f"| {cat} | {name} | {r['n_paired']} | {r['H@10_correct_mean']:.4f} | "
                f"{r['H@10_noisy_mean']:.4f} | {r['H@10_noisy_mean'] - r['H@10_correct_mean']:+.4f} | {p_str} |"
            )
    md += ["\n## Spearman rho across retrievers (Δmetric vs metric_correct)\n",
           "| Category | Metric | rho | p | n_retrievers |\n",
           "|---|---|---|---|---|"]
    for cat, d in summary.items():
        for m in ["H@10", "MR@10", "N@10"]:
            sp = d[f"spearman_{m}"]
            p_str = "NaN" if str(sp["p"]) == 'nan' else f"{sp['p']:.4e}"
            md.append(f"| {cat} | {m} | {sp['rho']:.4f} | {p_str} | {sp['n']} |")
    md_path = DEFAULT_OUT / "multi_retriever_summary.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    log(f"wrote {md_path}")
    log("=== done ===")


if __name__ == "__main__":
    main()