"""Stage 13 — LLM rerank evaluation (consolidated from Stage 11/12).

2026-09-19: Unified LLM rerank pipeline.
  - run_stage11_llm_rerank: 7 retrievers × 1947 selected queries → Qwen2.5
    pointwise rerank on top-100 (from Stage 11 top100_cache) → top-10 hit@k
    + per-retriever volatility (flip rate + RR std).
  - run_stage12_typo_paired_degradation: 7 retrievers × 1912 typo pairs →
    Qwen rerank on typo top-100 (from Stage 12 top100_cache_typo) → paired
    orig-vs-typo hit@k degradation per retriever (loads Stage 11 orig rerank
    results for paired comparison).

Inputs:
  Stage 8 selection → Stage 11 top100_cache/{name}_top100.npz (orig)
  Stage 10 typo injection → Stage 12 top100_cache_typo/{name}_top100.npz (typo)
  Stage 11 main rerank → result/11_syntactic_evaluation/llm_rerank_results.json

Outputs:
  result/13_llm_rerank/llm_rerank_results.json      (Stage 11 metrics)
  result/13_llm_rerank/llm_rerank_typo_results.json (Stage 12 paired deg)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "result/13_llm_rerank"
OUT_DIR.mkdir(parents=True, exist_ok=True)
STAGE11_OUT = OUT_DIR / "llm_rerank_results.json"
STAGE12_OUT = OUT_DIR / "llm_rerank_typo_results.json"

STAGE11_TOPK_DIR = REPO_ROOT / "result/11_syntactic_evaluation/top100_cache"
TYPO_TOPK_DIR = REPO_ROOT / "result/12_typo_evaluation/top100_cache_typo"
STAGE8_SEL = REPO_ROOT / "result/08_select_query/selected_queries.json"
TYPO_PAIRS = REPO_ROOT / "result/10_typo_injection/typo_injection_results.json"
ASIN_TO_DOC = REPO_ROOT / "result/11_syntactic_evaluation/asin_to_doc.json"
# Read Stage 11 LLM rerank results from canonical Stage 11 location (which has full data).
STAGE11_LLM_RERANK_SRC = REPO_ROOT / "result/11_syntactic_evaluation/llm_rerank_results.json"

RETRIEVERS = ["bm25", "splade", "minilm", "mpnet", "bge_base_v15", "gte_base", "colbertv2"]

LLM_RERANK_TOPK = 10
LLM_RERANK_BATCH = 512  # 2026-09-19: Stage 11 same config achieves 0.87s/batch GPU limit
KS = (1, 5, 10)
N_SMOKE = 5


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_corpus() -> tuple[dict, list[str]]:
    with open(ASIN_TO_DOC) as f:
        asin_to_doc = json.load(f)
    asins = sorted(asin_to_doc.keys())
    return asin_to_doc, asins


def load_topk_cache(topk_dir: Path, retr: str) -> np.ndarray | None:
    p = topk_dir / f"{retr}_top100.npz"
    if not p.exists():
        return None
    return np.load(p)["topk_asins"]


def idx_to_asin(topk_idx: np.ndarray, asins: list[str]) -> list[list[str]]:
    return [[asins[i] for i in row if 0 <= i < len(asins)] for row in topk_idx]


def llm_rerank_batch(queries_asin_pairs, topk_asins_list, asin_to_doc,
                    client, top_k_out=10):
    """对 (query, top-K candidates) 拼 prompts → Qwen rerank → 取 top_k_out."""
    from llm_client import _build_rerank_prompt, _parse_rerank_score

    flat = []
    for qi, (target_asin, qtext) in enumerate(queries_asin_pairs):
        for cand_asin in topk_asins_list[qi]:
            cand_doc = asin_to_doc.get(cand_asin, "").replace("\n", " ").strip()[:300]
            flat.append((qi, cand_asin, _build_rerank_prompt(qtext, cand_doc)))

    log(f"  Qwen rerank: {len(flat)} prompts "
        f"({len(queries_asin_pairs)} queries × "
        f"top-{len(topk_asins_list[0]) if topk_asins_list and topk_asins_list[0] else 0})")

    scores_by_qi = {qi: {} for qi in range(len(queries_asin_pairs))}
    t0 = time.time()
    for batch_start in range(0, len(flat), LLM_RERANK_BATCH):
        batch = flat[batch_start:batch_start + LLM_RERANK_BATCH]
        prompts = [b[2] for b in batch]
        outs = client.generate(prompts, n=1, temperature=0.0, top_p=1.0, max_tokens=1)
        for (qi, asin, _prompt), out in zip(batch, outs):
            txt = out[0] if out else ""
            score = _parse_rerank_score(txt)
            scores_by_qi[qi][asin] = score
        if (batch_start // LLM_RERANK_BATCH) % 20 == 0:
            log(f"    batch {batch_start // LLM_RERANK_BATCH}/"
                f"{(len(flat) + LLM_RERANK_BATCH - 1) // LLM_RERANK_BATCH}, "
                f"elapsed {time.time()-t0:.1f}s")
    log(f"  Qwen rerank done in {time.time()-t0:.1f}s")

    results = []
    for qi, (target_asin, _q) in enumerate(queries_asin_pairs):
        scored = scores_by_qi[qi]
        ranked = sorted(scored.items(), key=lambda x: -x[1])
        top_out = [a for a, _ in ranked[:top_k_out]]
        results.append({
            "top10": top_out,
            "target_asin": target_asin,
        })
    return results


def compute_hit_metrics(reranked):
    n = len(reranked)
    hit = {k: 0 for k in KS}
    rr_sum = 0.0
    for r in reranked:
        target = r["target_asin"]
        try:
            rank = r["top10"].index(target) + 1
        except ValueError:
            rank = None
        if rank is not None:
            rr_sum += 1.0 / rank
            for k in KS:
                if rank <= k:
                    hit[k] += 1
    return {
        "n_queries": n,
        "hit@1": hit[1] / n,
        "hit@5": hit[5] / n,
        "hit@10": hit[10] / n,
        "MRR": rr_sum / n,
    }


def compute_flip_rate(reranked, min_queries_per_asin=2):
    """Per-ASIN volatility: 同 ASIN 多 query 间的 rank stability."""
    from collections import defaultdict
    asin_to_ranks = defaultdict(list)
    for r in reranked:
        target = r["target_asin"]
        try:
            rank = r["top10"].index(target) + 1
        except ValueError:
            rank = None
        asin_to_ranks[target].append(rank)
    flip_rates = {k: [] for k in (1, 5, 10, 20)}
    rr_stds = []
    for asin, ranks in asin_to_ranks.items():
        if len(ranks) < min_queries_per_asin:
            continue
        for k in (1, 5, 10, 20):
            hits_k = [(r is not None and r <= k) for r in ranks]
            if len(hits_k) >= 2:
                n_flip = sum(1 for i in range(len(hits_k))
                             for j in range(i + 1, len(hits_k))
                             if hits_k[i] != hits_k[j])
                n_pairs = len(hits_k) * (len(hits_k) - 1) / 2
                flip_rates[k].append(n_flip / n_pairs)
        rrs = [1.0 / r for r in ranks if r is not None]
        if len(rrs) >= 2:
            mean_rr = sum(rrs) / len(rrs)
            var = sum((rr - mean_rr) ** 2 for rr in rrs) / len(rrs)
            rr_stds.append(var ** 0.5)
    import statistics as _stats
    out = {"n_asins": len(flip_rates[1])}
    for k in (1, 5, 10, 20):
        if flip_rates[k]:
            out[f"Hit@{k}_FlipRate_mean"] = sum(flip_rates[k]) / len(flip_rates[k])
        else:
            out[f"Hit@{k}_FlipRate_mean"] = 0.0
    if rr_stds:
        out["RR_Std_mean"] = sum(rr_stds) / len(rr_stds)
        out["RR_Std_median"] = _stats.median(rr_stds)
        out["RR_Std_std"] = _stats.stdev(rr_stds) if len(rr_stds) > 1 else 0.0
    else:
        out["RR_Std_mean"] = 0.0
        out["RR_Std_median"] = 0.0
        out["RR_Std_std"] = 0.0
    return out


def load_stage8_selection() -> list[tuple[str, str]]:
    with open(STAGE8_SEL) as f:
        sel = json.load(f)
    entries = []
    for e in sel.get("selections", []):
        for u in e.get("users", []):
            q = u.get("query")
            if q:
                entries.append((e["asin"], q))
    return entries


def load_typo_pairs(smoke=False) -> list[dict]:
    with open(TYPO_PAIRS) as f:
        data = json.load(f)
    pairs = data["results"]
    if smoke:
        pairs = pairs[:N_SMOKE]
    return pairs


# ============================================================================
# Stage 11: Multi-retriever LLM rerank on orig queries (1947 pairs)
# ============================================================================
def run_stage11_llm_rerank(smoke=False):
    log("=== Stage 13 / Stage 11 LLM rerank (7 retrievers × top-100 → Qwen → top-10) ===")
    entries = load_stage8_selection()
    if smoke:
        entries = entries[:N_SMOKE]
    log(f"  loaded {len(entries)} query pairs")

    asin_to_doc, asins = load_corpus()
    queries_asin_pairs = [(a, q) for a, q in entries]

    from llm_client import get_client
    client = get_client()

    per_retriever_metrics = {}
    per_retriever_flip = {}
    per_query_top10_by_retr = {}

    for retr in RETRIEVERS:
        topk_idx = load_topk_cache(STAGE11_TOPK_DIR, retr)
        if topk_idx is None:
            log(f"  ⚠ [{retr}] top-100 cache missing, skipping")
            continue
        if topk_idx.shape[0] != len(entries):
            log(f"  ⚠ [{retr}] cache shape mismatch "
                f"(cached={topk_idx.shape[0]}, current={len(entries)}), skipping")
            continue
        log(f"\n  [{retr}] reranking {len(entries)} queries with top-{topk_idx.shape[1]} candidates...")
        topk_asins = idx_to_asin(topk_idx, asins)
        reranked = llm_rerank_batch(queries_asin_pairs, topk_asins, asin_to_doc, client,
                                    top_k_out=LLM_RERANK_TOPK)
        m = compute_hit_metrics(reranked)
        f = compute_flip_rate(reranked)
        per_retriever_metrics[retr] = m
        per_retriever_flip[retr] = f
        per_query_top10_by_retr[retr] = [
            {"target_asin": r["target_asin"], "top10": r["top10"]} for r in reranked
        ]
        log(f"  [{retr}] hit@1={m['hit@1']*100:.2f}% hit@10={m['hit@10']*100:.2f}% "
            f"MRR={m['MRR']*100:.2f}% RR_Std={f['RR_Std_mean']:.4f}")

    out = {
        "config": {
            "rerank_topk": LLM_RERANK_TOPK,
            "n_pairs": len(entries),
            "smoke": smoke,
            "retrievers": list(per_retriever_metrics.keys()),
        },
        "metrics_by_retriever": per_retriever_metrics,
        "flip_rate_by_retriever": per_retriever_flip,
        "per_query_top10_by_retriever": per_query_top10_by_retr,
    }
    with open(STAGE11_OUT, "w") as f:
        json.dump(out, f, indent=2)
    log(f"\n  wrote → {STAGE11_OUT}")
    return out


# ============================================================================
# Stage 12: Paired orig-vs-typo LLM rerank degradation (1912 pairs)
# ============================================================================
def run_stage12_typo_paired_degradation(smoke=False):
    log("=== Stage 13 / Stage 12 LLM rerank (typo top-100 × 7 retrievers + Stage 11 paired deg) ===")

    # Load Stage 8 selection for typo→sel index mapping (1947 entries in Stage 11 retrieval order)
    sel_entries = load_stage8_selection()
    sel_lookup = {(a, q): i for i, (a, q) in enumerate(sel_entries)}

    # Load Stage 11 LLM rerank results (orig side) from canonical Stage 11 location.
    # Stage 11 main output at result/11_syntactic_evaluation/llm_rerank_results.json has
    # the full 1947-query rerank (this stage13/llm_rerank_results.json may be empty after smoke).
    with open(STAGE11_LLM_RERANK_SRC) as f:
        stage11 = json.load(f)
    stage11_top10_by_retr = stage11["per_query_top10_by_retriever"]
    log(f"  Stage 11 LLM rerank loaded for {len(stage11_top10_by_retr)} retrievers "
        f"({len(next(iter(stage11_top10_by_retr.values())))} queries each)")

    # Load typo pairs
    pairs = load_typo_pairs(smoke=smoke)
    log(f"  typo pairs: {len(pairs)}")

    # Build (asin, original_query) → stage11_index map
    typo_to_sel = []
    n_unmatched = 0
    for p in pairs:
        key = (p["asin"], p["original_query"])
        idx = sel_lookup.get(key, -1)
        if idx < 0:
            n_unmatched += 1
        typo_to_sel.append(idx)
    if n_unmatched:
        log(f"  ⚠ {n_unmatched}/{len(pairs)} typo pairs not in Stage 11 selection")
    else:
        log(f"  ✓ all {len(pairs)} typo pairs matched to Stage 11 selection")

    typo_entries = [(p["asin"], p["typo_query"]) for p in pairs]
    asin_to_doc, asins = load_corpus()

    from llm_client import get_client
    client = get_client()

    per_retriever_typo_metrics = {}
    per_retriever_paired_deg = {}

    for retr in RETRIEVERS:
        topk_idx = load_topk_cache(TYPO_TOPK_DIR, retr)
        if topk_idx is None:
            log(f"  ⚠ [{retr}] typo top-100 cache missing, skipping")
            continue
        if smoke:
            rerank_n = min(N_SMOKE, len(pairs), topk_idx.shape[0])
        else:
            if topk_idx.shape[0] != len(pairs):
                log(f"  ⚠ [{retr}] typo cache shape mismatch, skipping")
                continue
            rerank_n = len(pairs)
        log(f"\n  [{retr}] typo reranking {rerank_n}/{len(pairs)} queries with "
            f"top-{topk_idx.shape[1]} candidates...")
        topk_asins = idx_to_asin(topk_idx[:rerank_n], asins)
        typo_reranked = llm_rerank_batch(typo_entries[:rerank_n], topk_asins, asin_to_doc,
                                        client, top_k_out=LLM_RERANK_TOPK)
        typo_metrics = compute_hit_metrics(typo_reranked)
        per_retriever_typo_metrics[retr] = typo_metrics

        if retr not in stage11_top10_by_retr:
            continue
        stage11_top10_list = stage11_top10_by_retr[retr]

        orig_hits = {k: [] for k in KS}
        typo_hits = {k: [] for k in KS}
        deg_per_k = {k: [] for k in KS}
        orig_rrs = []
        typo_rrs = []
        n_paired = 0
        for pi in range(rerank_n):
            p = pairs[pi]
            sel_idx = typo_to_sel[pi]
            if sel_idx < 0 or sel_idx >= len(stage11_top10_list):
                continue
            orig_top10 = stage11_top10_list[sel_idx]["top10"]
            target = p["asin"]
            try:
                orig_rank = orig_top10.index(target) + 1
            except ValueError:
                orig_rank = None
            try:
                typo_rank = typo_reranked[pi]["top10"].index(target) + 1
            except ValueError:
                typo_rank = None
            for k in KS:
                orig_hit = 1 if (orig_rank is not None and orig_rank <= k) else 0
                typo_hit = 1 if (typo_rank is not None and typo_rank <= k) else 0
                orig_hits[k].append(orig_hit)
                typo_hits[k].append(typo_hit)
                deg_per_k[k].append(orig_hit - typo_hit)
            if orig_rank is not None:
                orig_rrs.append(1.0 / orig_rank)
            if typo_rank is not None:
                typo_rrs.append(1.0 / typo_rank)
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
            "rerank_topk": LLM_RERANK_TOPK,
            "n_typo_pairs": len(pairs),
            "smoke": smoke,
            "retrievers": list(per_retriever_typo_metrics.keys()),
        },
        "typo_metrics_by_retriever": per_retriever_typo_metrics,
        "paired_degradation_by_retriever": per_retriever_paired_deg,
    }
    with open(STAGE12_OUT, "w") as f:
        json.dump(out, f, indent=2)
    log(f"\n  wrote → {STAGE12_OUT}")
    return out


if __name__ == "__main__":
    SMOKE = "--smoke" in sys.argv
    # Default: run BOTH Stage 11 (orig LLM rerank) and Stage 12 (typo paired degradation).
    # Skip flags: --no-stage11 / --no-stage12 to run only one.
    RUN_STAGE11 = "--no-stage11" not in sys.argv
    RUN_STAGE12 = "--no-stage12" not in sys.argv
    log(f"=== Stage 13 LLM rerank pipeline ===")
    log(f"  SMOKE={SMOKE}  RUN_STAGE11={RUN_STAGE11}  RUN_STAGE12={RUN_STAGE12}")
    t0 = time.time()
    if RUN_STAGE11:
        s11 = run_stage11_llm_rerank(smoke=SMOKE)
        log(f"=== Stage 11 done in {time.time()-t0:.1f}s ===")
    if RUN_STAGE12:
        t1 = time.time()
        s12 = run_stage12_typo_paired_degradation(smoke=SMOKE)
        log(f"=== Stage 12 done in {time.time()-t1:.1f}s ===")
    log(f"=== total: {time.time()-t0:.1f}s ===")
