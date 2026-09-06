#!/usr/bin/env python3
"""Stage 12 — Typo Injection Retrieval Evaluation (paired degradation).

For each successfully injected (uid, asin, original_query, typo_query) pair from
Stage 10, run the SAME 7 retrievers used in Stage 11 (BM25 / SPLADE / MiniLM /
MPNet / BGE / GTE / ColBERTv2) on BOTH queries, then compare hit@1, hit@5,
hit@10 degradation. NO flip-rate computation (per user directive 2026-09-06).

Hit@k degradation = original_hit@k - typo_hit@k
  - 0   : typo did not affect recall
  - +1  : typo fully lost the hit (e.g. orig=1, typo=0)
  - -1  : typo IMPROVED retrieval (rare, indicates lucky match)

Output:
  - result/12_typo_evaluation/per_query.json     (paired original+typo records, 7 retr × 45 pairs)
  - result/12_typo_evaluation/retrieval_degradation.json  (per-retriever hit@k means)

Smoke (SMOKE=True):  5 pairs → 10 queries, ~2min (embed cache cold)
Full  (SMOKE=False): 45 pairs → 90 queries, ~3min with warm cache
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
EVAL_DIR = REPO_ROOT / "12_typo_evaluation"
sys.path.insert(0, str(REPO_ROOT / "11_syntactic_evaluation"))
sys.path.insert(0, str(EVAL_DIR))

OUT_PER_QUERY = REPO_ROOT / "result/12_typo_evaluation/per_query.json"
OUT_DEGRADATION = REPO_ROOT / "result/12_typo_evaluation/retrieval_degradation.json"

# Hardcoded (Rule 3)
SMOKE = False
N_SMOKE_PAIRS = 5
TYPO_RESULTS = REPO_ROOT / "result/10_typo_injection/typo_injection_results.json"

# Hit@k values reported (NO flip rate per user directive)
KS = (1, 5, 10)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_retrieval_module():
    """Import 11_syntactic_evaluation/syntax_subspace_retrieval_unified.py as module.

    Pre-seeds globals (REPO_ROOT, ASIN_TO_DOC_CACHE) BEFORE exec_module because
    the source has path constants that reference REPO_ROOT before its definition
    (line 48 uses REPO_ROOT, line 50 defines it).
    """
    spec = importlib.util.spec_from_file_location(
        "retrieval_unified",
        str(REPO_ROOT / "11_syntactic_evaluation/syntax_subspace_retrieval_unified.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    mod.REPO_ROOT = REPO_ROOT
    mod.ASIN_TO_DOC_CACHE = REPO_ROOT / "result/11_syntactic_evaluation/asin_to_doc.json"
    mod.META_FILE = REPO_ROOT / "data/meta_Baby_Products_2023.jsonl"
    mod.SEL_IN = REPO_ROOT / "result/08_select_query/selected_queries.json"
    mod.PER_QUERY_OUT = Path("/home/wlia0047/hj82_scratch2/wenyu/typo_eval/per_query_stage11.json")
    mod.SUMMARY_OUT = Path("/home/wlia0047/hj82_scratch2/wenyu/typo_eval/retrieval_summary_stage11.json")
    mod.VOLATILITY_OUT = Path("/home/wlia0047/hj82_scratch2/wenyu/typo_eval/volatility_stage11.json")
    mod.EMBED_CACHE_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/multiretrieval_embeds")
    spec.loader.exec_module(mod)
    return mod


def _queries_sig(query_records: list[dict]) -> str:
    """Stable signature over the (uid, asin, original_query, typo_query) list."""
    h = hashlib.sha1()
    h.update(f"n={len(query_records)}|".encode())
    for r in sorted(query_records, key=lambda x: (x["uid"], x["asin"])):
        h.update(f"{r['uid']}|{r['asin']}|{r['original_query']}|{r['typo_query']}\x00".encode())
    return h.hexdigest()[:16]


def load_pairs():
    with open(TYPO_RESULTS) as f:
        data = json.load(f)
    pairs = data["results"]
    if SMOKE:
        pairs = pairs[:N_SMOKE_PAIRS]
    return pairs


def main():
    t0 = time.time()
    log("=== Stage 12 Typo Retrieval Evaluation (paired degradation) ===")
    log(f"  SMOKE={SMOKE}  KS={KS}")

    # ---- Load 11 module ----
    retr_mod = _load_retrieval_module()
    log(f"  loaded retrievers: {retr_mod.RETR_NAMES}")

    # ---- Load 45 pairs (uid, asin, original, typo) ----
    pairs = load_pairs()
    log(f"  loaded {len(pairs)} (uid, asin, original, typo) pairs")

    # ---- Build corpus (asin → product doc) ----
    asin_to_doc = retr_mod.build_meta_corpus()
    asins = sorted(asin_to_doc.keys())
    asin_to_idx = {a: i for i, a in enumerate(asins)}
    corpus_texts = [asin_to_doc[a] for a in asins]
    corpus_sig = retr_mod._corpus_signature(meta_file=retr_mod.META_FILE)
    log(f"  corpus: {len(asins)} ASINs  sig={corpus_sig}")

    # ---- Build combined query list: original_i, typo_i ----
    records = []
    combined_queries = []
    combined_types = []   # "orig" or "typo"
    combined_targets = []
    for p in pairs:
        asin = p["asin"]
        if asin not in asin_to_idx:
            log(f"  skip {asin}: not in corpus")
            continue
        ti = asin_to_idx[asin]
        combined_queries.append(p["original_query"])
        combined_queries.append(p["typo_query"])
        combined_types.append("orig")
        combined_types.append("typo")
        combined_targets.append(ti)
        combined_targets.append(ti)
        records.append({
            "uid": p["uid"],
            "asin": asin,
            "original_query": p["original_query"],
            "typo_query": p["typo_query"],
        })
    target_indices = np.array(combined_targets)
    log(f"  combined queries: {len(combined_queries)} ({len(records)} pairs × 2)")
    query_sig = _queries_sig(records)
    log(f"  query_sig={query_sig}")

    # ---- Run each retriever ----
    retr_results: dict[str, list[dict]] = {}
    for retr in retr_mod.RETRIEVERS:
        kind = retr["kind"]
        name = retr["name"]
        if kind == "sparse_lexical":
            results = retr_mod.bm25_retrieve(combined_queries, corpus_texts, target_indices)
        elif kind == "sparse_learned":
            results = retr_mod.splade_retrieve(
                combined_queries, corpus_texts, target_indices,
                corpus_sig=corpus_sig, query_sig=query_sig,
            )
        elif kind == "dense":
            results, _q_embeds = retr_mod.dense_retrieve(
                name, retr["hf_id"], combined_queries, target_indices,
                corpus_sig=corpus_sig, query_sig=query_sig,
            )
        elif kind == "late_interaction":
            results, _q_embeds = retr_mod.colbertv2_retrieve(
                combined_queries, corpus_texts, target_indices,
                corpus_sig=corpus_sig, query_sig=query_sig,
            )
        else:
            raise ValueError(f"Unknown retriever kind: {kind}")
        retr_results[name] = results

    # ---- Pair back: orig_records[i], typo_records[i+1] for each record ----
    for ri, rec in enumerate(records):
        for retr in retr_mod.RETR_NAMES:
            res_orig = retr_results[retr][2 * ri]
            res_typo = retr_results[retr][2 * ri + 1]
            for k in KS:
                rec[f"{retr}_hit{k}_orig"] = bool(res_orig[f"hit{k}"])
                rec[f"{retr}_hit{k}_typo"] = bool(res_typo[f"hit{k}"])
                rec[f"{retr}_rank_orig"] = res_orig["rank"]
                rec[f"{retr}_rank_typo"] = res_typo["rank"]
                rec[f"{retr}_RR_orig"] = res_orig["RR"]
                rec[f"{retr}_RR_typo"] = res_typo["RR"]

    # ---- Aggregate: per retriever × per k, mean orig/typo hit@k + degradation ----
    summary = {
        "config": {
            "stage": "12_typo_evaluation",
            "n_pairs": len(records),
            "smoke": SMOKE,
            "ks": list(KS),
            "corpus_size": len(asins),
            "query_sig": query_sig,
            "corpus_sig": corpus_sig,
            "note": "paired degradation: orig - typo hit@k; positive = typo hurts recall",
        },
        "per_retriever": {},
    }
    log("\n=== Per-retriever hit@k degradation (orig - typo) ===")
    header = f"{'retriever':<14}" + "".join(f" {'hit@'+str(k)+'_orig':>13}" for k in KS) + \
             "".join(f" {'hit@'+str(k)+'_typo':>14}" for k in KS) + \
             "".join(f" {'deg@'+str(k):>8}" for k in KS)
    log(header)
    for retr in retr_mod.RETR_NAMES:
        per_k = {}
        for k in KS:
            orig_arr = np.array([rec[f"{retr}_hit{k}_orig"] for rec in records], dtype=np.float32)
            typo_arr = np.array([rec[f"{retr}_hit{k}_typo"] for rec in records], dtype=np.float32)
            deg = orig_arr - typo_arr
            per_k[f"hit@{k}"] = {
                "orig_mean": float(orig_arr.mean()),
                "typo_mean": float(typo_arr.mean()),
                "degradation_mean": float(deg.mean()),
                "degradation_median": float(np.median(deg)),
                "n_pairs": len(records),
                "n_typo_helps": int((deg < 0).sum()),    # negative deg = typo IMPROVED
                "n_typo_hurts": int((deg > 0).sum()),    # positive deg = typo HURT
                "n_typo_neutral": int((deg == 0).sum()),
            }
        summary["per_retriever"][retr] = per_k
        row = f"{retr:<14}"
        for k in KS:
            row += f" {per_k[f'hit@{k}']['orig_mean']*100:>12.2f}%"
        for k in KS:
            row += f" {per_k[f'hit@{k}']['typo_mean']*100:>13.2f}%"
        for k in KS:
            row += f" {per_k[f'hit@{k}']['degradation_mean']*100:>7.2f}%"
        log(row)

    # ---- Save ----
    OUT_PER_QUERY.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PER_QUERY, "w", encoding="utf-8") as f:
        json.dump({
            "config": summary["config"],
            "records": records,
        }, f, ensure_ascii=False)
    log(f"wrote → {OUT_PER_QUERY} ({len(records)} paired records)")

    with open(OUT_DEGRADATION, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log(f"wrote → {OUT_DEGRADATION}")

    log(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()