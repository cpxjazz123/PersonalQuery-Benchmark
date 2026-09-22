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


def load_typo_pairs(smoke: bool = False) -> list[dict]:
    with open(TYPO_PAIRS) as f:
        data = json.load(f)
    pairs = data["results"]
    if smoke:
        pairs = pairs[:N_SMOKE]
    return pairs


def load_stage12_orig_baseline() -> dict:
    """Stage 12 typo evaluation holds per-retriever orig/typo hit@k for each pair.

    Used as the orig-side baseline for Stage 14 paired degradation: we measure
    ``Stage12_orig_hit@k - Stage14_LLM_rerank_typo_hit@k`` directly, no Stage 13
    rerank intermediate needed. Returns a lookup keyed by
    ``(asin, original_query)`` -> per-retriever dict with hit@{1,5,10}_orig,
    rank_orig, RR_orig fields (per retriever).
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


def run_typo_paired_degradation(smoke: bool = False) -> dict:
    """Stage 14 typo paired degradation with Stage 12 baseline (no Stage 13 needed).

    For each retriever in RETRIEVERS:
      orig_hit@k  = Stage 12 per_query.json orig hit@k (clean retriever on clean query)
      typo_hit@k  = Stage 14 LLM rerank on typo query
      deg_hit@k   = orig_hit@k - typo_hit@k
      MRR_deg     = orig_MRR - typo_MRR (using 1/rank)
    """
    log("=== Stage 14 / Stage 12 LLM rerank (typo paired) ===")

    pairs = load_typo_pairs(smoke=smoke)
    log(f"  typo pairs: {len(pairs)}")

    stage12_lookup = load_stage12_orig_baseline()
    missing = [p for p in pairs
               if (p["asin"], p["original_query"]) not in stage12_lookup]
    if missing:
        raise KeyError(f"Stage 12 record missing for typo pair: {missing[:3]}")
    log(f"  ✓ all {len(pairs)} typo pairs matched to Stage 12 baseline "
        f"({len(stage12_lookup)} records)")

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
    client = get_client(backend="transformers")

    per_retriever_typo_metrics = {}
    per_retriever_typo_diagnostics = {}
    per_query_typo_rerank_by_retr = {}
    per_retriever_paired_deg = {}
    typo_queries_text = [pairs[i]["typo_query"] for i in range(rerank_n)]
    typo_pairs = [(pairs[i]["asin"], i) for i in range(rerank_n)]

    for retr in typo_retr_list:
        log(f"\n  [{retr}] typo reranking {rerank_n}/{len(pairs)} queries with top-100 candidates...")
        typo_reranked = llm_rerank_per_retriever(
            typo_queries_text, topk_by_retr[retr], asin_to_doc, client)
        for qi, record in enumerate(typo_reranked):
            record["target_asin"] = typo_pairs[qi][0]
        typo_metrics = compute_hit_metrics(typo_reranked)
        typo_diagnostics = compute_rerank_diagnostics(typo_reranked)
        per_retriever_typo_metrics[retr] = typo_metrics
        per_retriever_typo_diagnostics[retr] = typo_diagnostics
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

        if retr not in RETRIEVERS:
            raise KeyError(f"Unknown retriever: {retr}")
        # Stage 12 baseline: orig hit@k / rank_orig / RR_orig are 0/1 bool /
        # None / float per query, looked up via per_query.json (no LLM rerank).
        orig_hits = {k: [] for k in KS}
        typo_hits = {k: [] for k in KS}
        deg_per_k = {k: [] for k in KS}
        orig_rrs = []
        typo_rrs = []
        n_paired = 0
        for pi in range(rerank_n):
            p = pairs[pi]
            key = (p["asin"], p["original_query"])
            if key not in stage12_lookup:
                raise KeyError(f"Stage 12 record missing: {retr} {key!r}")
            s12 = stage12_lookup[key]
            target = p["asin"]
            # Stage 12 orig rank → hit@k and MRR (1/rank) per query.
            orig_rank = s12.get(f"{retr}_rank_orig")
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
        m = per_retriever_typo_metrics.get(retr, {})
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
            "Stage14 paired degradation over typo-paired subset: "
            "deg_hit@k = orig_hit@k - typo_hit@k (positive = typo hurts); "
            "MRR_deg = orig_MRR - typo_MRR. No flip rate at Stage14."
        ),
        "rerank_weights": "final_score = 1.0 * LLM_score + 0.1 * rank_prior",
        "model": "Qwen3-Reranker (see llm_client.DEFAULT_QWEN_MODEL)",
        "rows": summary_rows,
    }
    with open(STAGE12_OUT, "w") as f:
        json.dump(out, f, indent=2)
    log(f"\n  wrote → {STAGE12_OUT}")
    return out


def build_and_save_stage14_paired_table(stage14_out: dict,
                                         out_path: Path) -> dict:
    """Build the Stage 12 / Stage 14 side-by-side table + save to JSON.

    Stage 12 baseline: typo-pair subset, BM25 (or other retriever) orig hit@k
                      read directly from result/12_typo_evaluation/per_query.json
    Stage 14 (typo rerank): typo-pair subset, LLM rerank hit@k

    Stage14 reports paired DEGRADATION only (deg_hit@k = Stage12_orig - Stage14_typo,
    MRR_deg = Stage12_orig_MRR - Stage14_typo_MRR) over the typo-paired subset;
    no flip rate is computed at Stage14.
    """
    stage11_metrics = load_stage11_baseline_metrics()
    paired = stage14_out.get("paired_degradation_by_retriever", {})
    log("\n=== Stage 12 baseline (typo-pair subset, BM25 etc.) | "
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
            "stage12_orig_hit@10_paired": sp.get("orig_hit@10"),
            "stage12_orig_MRR_paired": sp.get("orig_MRR"),
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
            f"S12_orig={cell(row['stage12_orig_hit@10_paired'])}  "
            f"S14_typo={cell(row['stage14_typo_hit@10_paired'])} "
            f"(deg {sign(row['stage14_deg_hit@10_paired'])})  "
            f"  MRR {cell(row['stage12_orig_MRR_paired'])} → "
            f"{cell(row['stage14_typo_MRR_paired'])} "
            f"(deg {sign(row['stage14_MRR_deg_paired'])})  {n_str}")
    payload = {
        "config": {
            "stage11_source": "result/11_syntactic_evaluation/per_query.json",
            "stage11_denominator": "full cohort (n=1947)",
            "stage12_source": str(STAGE12_PER_QUERY),
            "stage12_baseline": "Per-retriever orig hit@k / rank / RR on clean queries",
            "stage13_denominator": "NOT USED — Stage 14 baseline is Stage 12 (no Stage 13 dependency)",
            "stage14_source": str(STAGE12_OUT),
            "stage14_denominator": "typo-paired subset (n=1635)",
            "stage14_metric_definition": (
                "Stage14 reports paired DEGRADATION only: "
                "deg_hit@k = Stage12_orig_hit@k - Stage14_LLM_rerank_typo_hit@k; "
                "MRR_deg = Stage12_orig_MRR - Stage14_LLM_rerank_typo_MRR. "
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
    log(f"  SMOKE={SMOKE}  variants={[v[0] for v in RERANKER_VARIANTS]}")
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
        out14 = run_typo_paired_degradation(smoke=SMOKE)
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
            OUT_DIR / f"stage11_stage13_stage14_paired_{variant}.json")
        log(f"  [{variant}] done in {time.time()-t0:.1f}s")
    log(f"\n=== total: {time.time()-t_global:.1f}s ===")