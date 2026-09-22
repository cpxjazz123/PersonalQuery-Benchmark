"""Stage 14 — Typo paired rerank evaluation (per-retriever P(Yes) rerank).

2026-09-21: Split from the former ``13_llm_rerank`` umbrella. Stage 11
syntactic rerank now lives in ``13_syntactic_rerank/syntactic_rerank_eval.py``;
this script owns the Stage 12 typo paired degradation evaluation.

For each of the 7 retrievers in
{bm25, splade, minilm, mpnet, bge_base_v15, gte_base, colbertv2}:

    typo query → typo retriever Top-100 → Qwen P(Yes) rerank → top-10

We compare the resulting rerank Top-10 against the Stage 11 rerank Top-10
for the *original* (clean) query to obtain the paired Hit@K degradation
and MRR degradation.

The same LLM (Qwen via ``llm_client``), same prompt, same Top-100 depth,
same scoring rule are used across retrievers. The LLM score has no
contribution from the original retrieval score; the original rank is
used only as an exact tie-breaker.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "13_syntactic_rerank"))

from syntactic_rerank_eval import (  # noqa: E402  (sys.path tweak above)
    RETRIEVERS,
    KS,
    N_SMOKE,
    ONLY_BM25_SMOKE,
    SMOKE,
    log,
    load_corpus,
    load_topk_cache,
    idx_to_asin,
    compute_hit_metrics,
    compute_rerank_diagnostics,
    llm_rerank_per_retriever,
    load_stage11_baseline_metrics,
    print_and_save_stage11_vs_stage13,
)

OUT_DIR = REPO_ROOT / "result/14_typo_rerank"
OUT_DIR.mkdir(parents=True, exist_ok=True)
STAGE12_OUT = OUT_DIR / "llm_rerank_typo_results.json"
STAGE11_OUT = REPO_ROOT / "result/13_syntactic_rerank/llm_rerank_results.json"

TYPO_TOPK_DIR = REPO_ROOT / "result/12_typo_evaluation/top100_cache_typo"
TYPO_PAIRS = REPO_ROOT / "result/10_typo_injection/typo_injection_results.json"
ASIN_TO_DOC = REPO_ROOT / "result/11_syntactic_evaluation/asin_to_doc.json"
STAGE12_PER_QUERY = REPO_ROOT / "result/12_typo_evaluation/per_query.json"
STAGE13_RESULTS_DIR = REPO_ROOT / "result/13_syntactic_rerank"

# 2026-09-22: Stage 14 baseline = Stage 13 (same reranker on clean query).
# When True, skip LLM rerank and reuse cached per_query_typo_rerank_by_retriever
# from existing llm_rerank_typo_results_<variant>.json; only recompute the
# paired_degradation table against Stage 13 baseline.
REGEN_FROM_CACHE = False


def load_typo_pairs(smoke: bool = False) -> list[dict]:
    with open(TYPO_PAIRS) as f:
        data = json.load(f)
    pairs = data["results"]
    if smoke:
        pairs = pairs[:N_SMOKE]
    return pairs


def load_stage12_orig_baseline() -> dict:
    """Stage 12 typo evaluation holds per-retriever orig/typo hit@k for each pair.

    Retained for backward reference only — Stage 14 now uses Stage 13 (same
    reranker on clean query) as the baseline via :func:`load_stage13_baseline`.
    Returns a lookup keyed by ``(asin, original_query)`` -> per-retriever dict.
    """
    if not STAGE12_PER_QUERY.exists():
        raise FileNotFoundError(
            f"Stage 12 per_query.json missing: {STAGE12_PER_QUERY}")
    with open(STAGE12_PER_QUERY) as f:
        d = json.load(f)
    records = d["records"]
    lookup = {}
    for rec in records:
        key = (rec["asin"], rec["original_query"])
        lookup[key] = rec
    return lookup


def load_stage13_baseline(variant: str) -> dict:
    """Stage 13 LLM rerank results for the same reranker variant on clean queries.

    Paired within-system baseline: same reranker, same pipeline, same prompt,
    same candidate depth (BM25 top-100 → rerank top-10), only the input query
    differs (clean vs typo). Returns a lookup keyed by
    ``(target_asin, query)`` -> ``{retr: {"top10": [...], "retrieval_top10": [...]}}``.
    """
    if variant is None:
        raise ValueError("variant must be specified for Stage 13 baseline")
    path = STAGE13_RESULTS_DIR / f"llm_rerank_results_{variant}.json"
    if not path.exists():
        raise FileNotFoundError(f"Stage 13 results missing for variant={variant}: {path}")
    with open(path) as f:
        d = json.load(f)
    lookup: dict = {}
    per_query_by_retr = d.get("per_query_top10_by_retriever", {})
    for retr, records in per_query_by_retr.items():
        for r in records:
            key = (r["target_asin"], r["query"])
            lookup.setdefault(key, {})[retr] = {
                "top10": r["top10"],
                "retrieval_top10": r.get("retrieval_top10", []),
            }
    return lookup


def _typo_metrics_from_cached(typo_reranked: list[dict],
                              stage13_lookup: dict,
                              retr: str) -> dict:
    """Recompute hit@k / MRR on the typo side from cached top-10 records,
    restricted to queries that joined Stage 13 baseline (same cohort as
    paired_degradation).
    """
    hit_counts = {k: 0 for k in KS}
    rr_sum = 0.0
    rr_n = 0
    n = 0
    for r in typo_reranked:
        target = r["target_asin"]
        top10 = r["top10"]
        rank = top10.index(target) + 1 if target in top10 else None
        for k in KS:
            if rank is not None and rank <= k:
                hit_counts[k] += 1
        if rank is not None:
            rr_sum += 1.0 / rank
            rr_n += 1
        n += 1
    return {
        "hit@1": hit_counts[1] / n if n else 0.0,
        "hit@5": hit_counts[5] / n if n else 0.0,
        "hit@10": hit_counts[10] / n if n else 0.0,
        "MRR": rr_sum / rr_n if rr_n else 0.0,
        "n_queries": n,
        "n_eligible_hit10": hit_counts[10],
    }


def run_typo_paired_degradation(smoke: bool = False, variant: str | None = None,
                                regen_from_cache: bool = False) -> dict:
    """Stage 14 typo paired degradation with Stage 13 baseline (same reranker on clean).

    For each retriever in RETRIEVERS:
      orig_hit@k  = Stage 13 LLM rerank on clean query (same reranker variant)
      typo_hit@k  = Stage 14 LLM rerank on typo query (same reranker variant)
      deg_hit@k   = orig_hit@k - typo_hit@k   (positive = typo hurts)
      MRR_deg     = orig_MRR - typo_MRR       (positive = typo hurts)

    When ``regen_from_cache`` is True, the LLM rerank step is skipped and the
    typo-side per-query top-10 is loaded from
    ``result/14_typo_rerank/llm_rerank_typo_results_<variant>.json``. This is
    used when only the baseline source has changed.
    """
    log(f"=== Stage 14 / Stage 13 LLM rerank (typo paired) | variant={variant} ===")

    pairs = load_typo_pairs(smoke=smoke)
    log(f"  typo pairs: {len(pairs)}")

    stage13_lookup = load_stage13_baseline(variant)
    matched = sum(1 for p in pairs
                  if (p["asin"], p["original_query"]) in stage13_lookup)
    if matched == 0:
        raise KeyError(
            f"No Stage 13 records matched the typo pairs (variant={variant}). "
            f"Stage 13 lookup size={len(stage13_lookup)}")
    if matched < len(pairs):
        log(f"  ⚠ {matched}/{len(pairs)} typo pairs matched to Stage 13 baseline "
            f"({len(stage13_lookup)} records); unmatched will be excluded")
    log(f"  ✓ {matched}/{len(pairs)} typo pairs matched to Stage 13 baseline "
        f"({len(stage13_lookup)} Stage 13 records)")

    asin_to_doc, asins = load_corpus()

    per_retr_typo_idx: dict[str, np.ndarray] = {}
    typo_retr_list = ["bm25"] if SMOKE and ONLY_BM25_SMOKE else RETRIEVERS
    log(f"  typo retriever set: {typo_retr_list}")
    for retr in typo_retr_list:
        topk_idx = load_topk_cache(TYPO_TOPK_DIR, retr)
        if topk_idx is None:
            raise FileNotFoundError(f"Stage 12 typo top-100 cache missing for {retr}")
        per_retr_typo_idx[retr] = topk_idx

    if smoke:
        rerank_n = min(N_SMOKE, len(pairs),
                       min(arr.shape[0] for arr in per_retr_typo_idx.values()))
        per_retr_typo_idx = {retr: arr[:rerank_n] for retr, arr in per_retr_typo_idx.items()}
    else:
        for retr, arr in per_retr_typo_idx.items():
            if arr.shape[0] != len(pairs):
                raise ValueError(f"[{retr}] typo cache has {arr.shape[0]} rows, full requires {len(pairs)}")
        rerank_n = len(pairs)
    topk_by_retr = {retr: idx_to_asin(arr, asins) for retr, arr in per_retr_typo_idx.items()}

    from llm_client import get_client
    # 2026-09-22: vLLM removed from the whitelist (Rule 9). All rerankers
    # — Qwen3, BGE Gemma2, RankLLaMA — run on the transformers backend.
    client = get_client(backend="transformers") if not regen_from_cache else None

    per_retriever_typo_metrics = {}
    per_retriever_typo_diagnostics = {}
    per_query_typo_rerank_by_retr = {}
    per_retriever_paired_deg = {}
    typo_queries_text = [pairs[i]["typo_query"] for i in range(rerank_n)]
    typo_pairs = [(pairs[i]["asin"], i) for i in range(rerank_n)]

    # Optional regen path: load existing per_query_typo_rerank_by_retr from
    # the cached Stage 14 variant JSON (the LLM rerank output never changes
    # in this branch; only the baseline source differs).
    cached_typ_top10_by_retr: dict[str, list] = {}
    if regen_from_cache:
        cache_path = OUT_DIR / f"llm_rerank_typo_results_{variant}.json"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"REGEN_FROM_CACHE=True but cached Stage 14 result missing: {cache_path}")
        with open(cache_path) as f:
            cached = json.load(f)
        cached_typ_top10_by_retr = cached.get("per_query_typo_rerank_by_retriever", {})
        if not cached_typ_top10_by_retr:
            raise ValueError(f"Cached Stage 14 result has empty per_query_typo_rerank_by_retr: {cache_path}")
        log(f"  ✓ loaded cached typo-side top-10 for {len(cached_typ_top10_by_retr)} retrievers "
            f"({sum(len(v) for v in cached_typ_top10_by_retr.values())} records)")

    for retr in typo_retr_list:
        if regen_from_cache:
            cached_records = cached_typ_top10_by_retr.get(retr, [])
            if len(cached_records) != rerank_n:
                raise ValueError(
                    f"[{retr}] cached records {len(cached_records)} != rerank_n {rerank_n}")
            # Reconstruct typo_reranked shape expected by the paired-deg loop.
            typo_reranked = [
                {
                    "target_asin": r["target_asin"],
                    "top10": r["top10"],
                }
                for r in cached_records
            ]
            log(f"\n  [{retr}] reusing cached typo rerank top-10 for {len(typo_reranked)} queries")
            # Recompute typo-side hit@k / MRR from cached records (paired with
            # Stage 13 join, so n_paired may differ from len(cached_records)).
            typo_metrics = _typo_metrics_from_cached(
                typo_reranked, stage13_lookup, retr)
            typo_diagnostics = None
        else:
            log(f"\n  [{retr}] typo reranking {rerank_n}/{len(pairs)} queries with top-100 candidates...")
            typo_reranked = llm_rerank_per_retriever(
                typo_queries_text, topk_by_retr[retr], asin_to_doc, client)
            for qi, record in enumerate(typo_reranked):
                record["target_asin"] = typo_pairs[qi][0]
            typo_metrics = compute_hit_metrics(typo_reranked)
            typo_diagnostics = compute_rerank_diagnostics(typo_reranked)
        per_retriever_typo_metrics[retr] = typo_metrics
        per_retriever_typo_diagnostics[retr] = typo_diagnostics
        if not regen_from_cache:
            per_query_typo_rerank_by_retr[retr] = [
                {
                    "target_asin": r["target_asin"],
                    "query": pairs[i]["typo_query"],
                    "original_query": pairs[i]["original_query"],
                    "retrieval_top10": r["retrieval_top10"],
                    "top10": r["top10"],
                    "candidates": r["candidates"],
                    "diagnostics": r["diagnostics"],
                }
                for i, r in enumerate(typo_reranked)
            ]
            log(f"  [{retr}] typo score unique={len(typo_diagnostics['llm_score_unique_values'])} "
                f"mean={typo_diagnostics['llm_score_mean']} "
                f"std={typo_diagnostics['llm_score_std']} "
                f"top10_changed={typo_diagnostics['queries_top10_order_changed']}/"
                f"{typo_diagnostics['n_queries']} "
                f"Kendall={typo_diagnostics['average_kendall_tau_a']:.4f} "
                f"Spearman={typo_diagnostics['average_spearman_rho']:.4f}")
        else:
            per_query_typo_rerank_by_retr[retr] = cached_typ_top10_by_retr[retr]

        if retr not in RETRIEVERS:
            raise KeyError(f"Unknown retriever: {retr}")
        # Stage 13 baseline: orig hit@k / MRR computed from same reranker's
        # top-10 on the clean query. Join key: (target_asin, original_query).
        orig_hits = {k: [] for k in KS}
        typo_hits = {k: [] for k in KS}
        deg_per_k = {k: [] for k in KS}
        orig_rrs = []
        typo_rrs = []
        n_paired = 0
        for pi in range(rerank_n):
            p = pairs[pi]
            key = (p["asin"], p["original_query"])
            s13 = stage13_lookup.get(key)
            if s13 is None:
                # Stage 13 did not run this (asin, clean-query); skip
                # (must not fallback per Rule 7).
                continue
            s13_retr = s13.get(retr)
            if s13_retr is None:
                continue
            orig_top10 = s13_retr["top10"]
            target = p["asin"]
            orig_rank = orig_top10.index(target) + 1 \
                if target in orig_top10 else None
            for k in KS:
                orig_hit = 1 if (orig_rank is not None and orig_rank <= k) else 0
                typo_rank = typo_reranked[pi]["top10"].index(target) + 1 \
                    if target in typo_reranked[pi]["top10"] else None
                typo_hit = 1 if (typo_rank is not None and typo_rank <= k) else 0
                orig_hits[k].append(orig_hit)
                typo_hits[k].append(typo_hit)
                deg_per_k[k].append(orig_hit - typo_hit)
            if orig_rank is not None:
                orig_rrs.append(1.0 / orig_rank)
            typo_rank_final = typo_reranked[pi]["top10"].index(target) + 1 \
                if target in typo_reranked[pi]["top10"] else None
            if typo_rank_final is not None:
                typo_rrs.append(1.0 / typo_rank_final)
            n_paired += 1
        if n_paired == 0:
            raise ValueError(
                f"[{retr}] 0 paired queries after Stage 13 join; aborting")
        paired = {"n_paired": n_paired}
        for k in KS:
            paired[f"orig_hit@{k}"] = float(np.mean(orig_hits[k])) if orig_hits[k] else 0.0
            paired[f"typo_hit@{k}"] = float(np.mean(typo_hits[k])) if typo_hits[k] else 0.0
            paired[f"deg_hit@{k}"] = float(np.mean(deg_per_k[k])) if deg_per_k[k] else 0.0
        paired["orig_MRR"] = float(np.mean(orig_rrs)) if orig_rrs else 0.0
        paired["typo_MRR"] = float(np.mean(typo_rrs)) if typo_rrs else 0.0
        paired["MRR_deg"] = paired["orig_MRR"] - paired["typo_MRR"]
        per_retriever_paired_deg[retr] = paired
        log(f"  [{retr}] orig_hit@10={paired['orig_hit@10']*100:.2f}% "
            f"typo_hit@10={paired['typo_hit@10']*100:.2f}% "
            f"deg@10={paired['deg_hit@10']*100:+.2f}% "
            f"MRR_deg={paired['MRR_deg']*100:+.2f}%")

    out = {
        "config": {
            "rerank_topk": KS[-1],
            "n_typo_pairs": len(pairs),
            "smoke": smoke,
            "retrievers": list(typo_retr_list),
            "score_range": [0.0, 1.0],
            "score_semantics": "P(Yes) over (Yes, No) via vLLM sampled-token logprobs",
            "max_new_tokens": 1,
            "tie_break": "retrieval_rank",
            "candidate_depth": 100,
            "rerank_scope": "per-retriever Top-100",
        },
        "typo_metrics_by_retriever": per_retriever_typo_metrics,
        "typo_rerank_diagnostics_by_retriever": per_retriever_typo_diagnostics,
        "per_query_typo_rerank_by_retriever": per_query_typo_rerank_by_retr,
        "paired_degradation_by_retriever": per_retriever_paired_deg,
    }

    # Aggregated per-retriever summary_table: orig / typo / deg absolute values
    # (no flip rate — Stage14 reports degradation only).
    summary_rows = []
    for retr in typo_retr_list:
        m = per_retriever_typo_metrics.get(retr) or {}
        d = per_retriever_paired_deg.get(retr, {})
        summary_rows.append({
            "retriever": retr,
            "n_paired": d.get("n_paired"),
            "typo_hit@1": m.get("hit@1"),
            "typo_hit@5": m.get("hit@5"),
            "typo_hit@10": m.get("hit@10"),
            "typo_MRR": m.get("MRR"),
            "typo_Recall@100": m.get("Recall@100"),
            "orig_hit@1": d.get("orig_hit@1"),
            "orig_hit@5": d.get("orig_hit@5"),
            "orig_hit@10": d.get("orig_hit@10"),
            "orig_MRR": d.get("orig_MRR"),
            "deg_hit@1": d.get("deg_hit@1"),
            "deg_hit@5": d.get("deg_hit@5"),
            "deg_hit@10": d.get("deg_hit@10"),
            "MRR_deg": d.get("MRR_deg"),
        })
    out["summary_table"] = {
        "columns": ["retriever", "n_paired",
                    "orig_hit@1", "orig_hit@5", "orig_hit@10", "orig_MRR",
                    "typo_hit@1", "typo_hit@5", "typo_hit@10", "typo_MRR",
                    "typo_Recall@100",
                    "deg_hit@1", "deg_hit@5", "deg_hit@10", "MRR_deg"],
        "metric_definition": (
            "Stage14 paired degradation over typo-paired subset (same reranker, "
            "same pipeline, clean vs typo): "
            "deg_hit@k = Stage13_orig_hit@k - Stage14_typo_hit@k (positive = typo hurts); "
            "MRR_deg = Stage13_orig_MRR - Stage14_typo_MRR. No flip rate at Stage14."
        ),
        "rerank_weights": "final_score = 1.0 * LLM_score + 0.1 * rank_prior",
        "model": f"variant={variant}",
        "rows": summary_rows,
    }
    with open(STAGE12_OUT, "w") as f:
        json.dump(out, f, indent=2)
    log(f"\n  wrote → {STAGE12_OUT}")
    return out


def build_and_save_stage14_paired_table(stage14_out: dict,
                                         out_path: Path, variant: str | None = None) -> dict:
    """Build the Stage 13 / Stage 14 side-by-side table + save to JSON.

    Stage 13 baseline: same reranker rerank on clean query, same pipeline
                      (BM25 top-100 → rerank → top-10), typo-paired subset join.
    Stage 14 (typo rerank): typo-pair subset, LLM rerank hit@k

    Stage14 reports paired DEGRADATION only (deg_hit@k = Stage13_clean - Stage14_typo,
    MRR_deg = Stage13_clean_MRR - Stage14_typo_MRR) over the typo-paired subset;
    no flip rate is computed at Stage14.
    """
    stage11_metrics = load_stage11_baseline_metrics()
    paired = stage14_out.get("paired_degradation_by_retriever", {})
    log("\n=== Stage 13 baseline (typo-pair subset, same reranker clean) | "
        "Stage 14 typo rerank (typo-pair subset) ===")
    log("  retr         hit@10                                          MRR")
    log("  " + "-" * 76)
    rows = []
    for retr in sorted(set(stage11_metrics) | set(paired)):
        s11 = stage11_metrics.get(retr, {})
        sp = paired.get(retr, {})
        row = {
            "retriever": retr,
            "stage11_hit@10_full_n1947": s11.get("hit@10"),
            "stage11_MRR_full_n1947": s11.get("MRR"),
            "stage13_orig_hit@10_paired": sp.get("orig_hit@10"),
            "stage13_orig_MRR_paired": sp.get("orig_MRR"),
            "stage14_typo_hit@10_paired": sp.get("typo_hit@10"),
            "stage14_typo_MRR_paired": sp.get("typo_MRR"),
            "stage14_deg_hit@10_paired": sp.get("deg_hit@10"),
            "stage14_MRR_deg_paired": sp.get("MRR_deg"),
            "n_paired": sp.get("n_paired"),
        }
        rows.append(row)

        def cell(v):
            return f"{v*100:6.2f}%" if v is not None else "   N/A"
        sign = lambda v: (f"{v*100:+5.2f}%" if v is not None else "   N/A")
        n_str = f"n={row['n_paired']}" if row['n_paired'] is not None else ""
        log(f"  {retr:12s}  "
            f"S13_clean={cell(row['stage13_orig_hit@10_paired'])}  "
            f"S14_typo={cell(row['stage14_typo_hit@10_paired'])} "
            f"(deg {sign(row['stage14_deg_hit@10_paired'])})  "
            f"  MRR {cell(row['stage13_orig_MRR_paired'])} → "
            f"{cell(row['stage14_typo_MRR_paired'])} "
            f"(deg {sign(row['stage14_MRR_deg_paired'])})  {n_str}")
    payload = {
        "config": {
            "stage11_source": "result/11_syntactic_evaluation/per_query.json",
            "stage11_denominator": "full cohort (n=1947)",
            "stage13_source": (
                f"result/13_syntactic_rerank/llm_rerank_results_{variant}.json"
                if variant else "result/13_syntactic_rerank/llm_rerank_results.json"),
            "stage13_baseline": (
                "Same reranker rerank on clean query (BM25 top-100 → rerank → top-10)"),
            "stage14_source": str(STAGE12_OUT),
            "stage14_denominator": "typo-paired subset",
            "stage14_metric_definition": (
                "Stage14 reports paired DEGRADATION only (same reranker, same "
                "pipeline, clean vs typo): "
                "deg_hit@k = Stage13_clean_hit@k - Stage14_typo_hit@k; "
                "MRR_deg = Stage13_clean_MRR - Stage14_typo_MRR. "
                "No flip rate is computed at Stage14."
            ),
            "rerank_weights": "final_score = 1.0 * LLM_score + 0.1 * rank_prior",
        },
        "rows": rows,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    log(f"\n  wrote → {out_path}")
    return payload


if __name__ == "__main__":
    # === reranker selection ===
    # Match the Stage 13 reranker variant. Supported:
    #   "qwen3"      -> Qwen3-Reranker
    #   "bge_gemma2" -> BAAI/bge-reranker-v2-gemma
    #   "rankllama"  -> castorini/rankllama-v1-7b-lora-passage
    #                   (SequenceClassification on Llama-2-7b-hf)
    RERANKER_VARIANTS = [
        ("qwen3",      "/home/wlia0047/hj82_scratch2/wenyu/RAG/Qwen3-Reranker-8B", None),
        ("bge_gemma2", "/home/wlia0047/hj82_scratch2/wenyu/RAG/BGE-reranker-Gemma2-9B", None),
        ("rankllama",  "/fs04/scratch2/hj82/wenyu/RAG/Llama-2-7b-hf",
                       "/fs04/scratch2/hj82/wenyu/RAG/rankllama-v1-7b-lora-passage"),
    ]

    log("=== Stage 14 typo paired rerank (loops over 3 rerankers) ===")
    log(f"  SMOKE={SMOKE}  REGEN_FROM_CACHE={REGEN_FROM_CACHE}  "
        f"variants={[v[0] for v in RERANKER_VARIANTS]}")
    t_global = time.time()
    for variant, model_path, peft_adapter in RERANKER_VARIANTS:
        log(f"\n--- [{variant}] ---")
        import llm_client  # noqa: E402
        llm_client.DEFAULT_QWEN_MODEL = model_path
        llm_client.DEFAULT_PEFT_ADAPTER = peft_adapter
        llm_client.reset_client()
        import syntactic_rerank_eval as base_mod  # noqa: E402
        base_mod.RERANKER_VARIANT = variant

        t0 = time.time()
        out14 = run_typo_paired_degradation(smoke=SMOKE, variant=variant,
                                           regen_from_cache=REGEN_FROM_CACHE)
        variant_out = OUT_DIR / f"llm_rerank_typo_results_{variant}.json"
        with open(variant_out, "w") as f:
            json.dump(out14, f, indent=2)
        with open(STAGE12_OUT, "w") as f:
            json.dump(out14, f, indent=2)
        log(f"  wrote → {variant_out} + {STAGE12_OUT}")
        print_and_save_stage11_vs_stage13(
            {"metrics_by_retriever": out14.get("typo_metrics_by_retriever", {})},
            OUT_DIR / f"stage11_vs_stage14_{variant}.json",
            label=f"Stage 14 ({variant})")
        build_and_save_stage14_paired_table(
            out14,
            OUT_DIR / f"stage11_stage13_stage14_paired_{variant}.json",
            variant=variant)
        log(f"  [{variant}] done in {time.time()-t0:.1f}s")
    log(f"\n=== total: {time.time()-t_global:.1f}s ===")