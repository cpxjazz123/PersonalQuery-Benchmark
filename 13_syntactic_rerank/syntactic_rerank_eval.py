"""Stage 13 — LLM rerank evaluation (per-retriever P(Yes) rerank).

2026-09-21: Refactored to focus solely on the Stage 11 syntactic rerank.
The Stage 12 typo paired rerank has moved to
``14_typo_rerank/typo_rerank_eval.py``; importing it from this module
raises. Stage 11 / 12 retrieval pipelines must remain retrieval-only
(no LLM calls). For each of the 7 retrievers in
{bm25, splade, minilm, mpnet, bge_base_v15, gte_base, colbertv2}:

    retriever Top-100 -> Qwen P(Yes) rerank on those 100 -> top-10.

The same LLM (Qwen via ``llm_client``), same prompt, same Top-100 depth,
same scoring rule are used across retrievers. The LLM score has no
contribution from the original retrieval score; the original rank is
used only as an exact tie-breaker.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "result/13_syntactic_rerank"
OUT_DIR.mkdir(parents=True, exist_ok=True)
STAGE11_OUT = OUT_DIR / "llm_rerank_results.json"
TYPO_OUT_DIR = REPO_ROOT / "result/14_typo_rerank"

STAGE11_TOPK_DIR = REPO_ROOT / "result/11_syntactic_evaluation/top100_cache"
STAGE8_SEL = REPO_ROOT / "result/08_select_query/selected_queries.json"
TYPO_PAIRS = REPO_ROOT / "result/10_typo_injection/typo_injection_results.json"
ASIN_TO_DOC = REPO_ROOT / "result/11_syntactic_evaluation/asin_to_doc.json"

RETRIEVERS = ["bm25"]

LLM_RERANK_TOPK = 10
LLM_RERANK_CANDIDATES = 15
LLM_WEIGHT = 1.0
RETRIEVAL_WEIGHT = 0.1
# 兼容旧 λ sweep: 若代码里仍引用 RANK_PRIOR_LAMBDA,默认 2.0(本公式不再使用)
RANK_PRIOR_LAMBDA = 2.0
LLM_RERANK_BATCH = 512
LLM_RERANK_WINDOW = 20
LLM_RERANK_STRIDE = 10
LLM_RERANK_LISTWISE_MAX_TOKENS = 64
KS = (1, 5, 10)
N_SMOKE = 1635
ONLY_BM25_SMOKE = True
SMOKE = False


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_corpus() -> tuple[dict, list[str]]:
    if not ASIN_TO_DOC.exists():
        raise FileNotFoundError(
            f"Stage 11 corpus missing: {ASIN_TO_DOC}. "
            f"Run Stage 11 retrieval first."
        )
    with open(ASIN_TO_DOC) as f:
        asin_to_doc = json.load(f)
    return asin_to_doc, sorted(asin_to_doc.keys())


def load_topk_cache(topk_dir: Path, retr: str) -> np.ndarray | None:
    p = topk_dir / f"{retr}_top100.npz"
    if not p.exists():
        return None
    return np.load(p)["topk_asins"]


def idx_to_asin(topk_idx: np.ndarray, asins: list[str]) -> list[list[str]]:
    return [[asins[i] for i in row if 0 <= i < len(asins)] for row in topk_idx]


def _rank_correlation(retrieval_ranks, llm_ranks):
    n = len(retrieval_ranks)
    if n < 2:
        return 0.0, 0.0
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            left = retrieval_ranks[i] - retrieval_ranks[j]
            right = llm_ranks[i] - llm_ranks[j]
            product = left * right
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    pairs = n * (n - 1) / 2
    kendall = (concordant - discordant) / pairs
    mean_x = sum(retrieval_ranks) / n
    mean_y = sum(llm_ranks) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(retrieval_ranks, llm_ranks))
    var_x = sum((x - mean_x) ** 2 for x in retrieval_ranks)
    var_y = sum((y - mean_y) ** 2 for y in llm_ranks)
    spearman = cov / (var_x * var_y) ** 0.5 if var_x and var_y else 0.0
    return kendall, spearman


def _build_yes_no_prompt(query: str, doc_text: str) -> str:
    return (
        "You are a product-search relevance judge.\n\n"
        "Query:\n" + query + "\n\n"
        "Product:\n" + doc_text + "\n\n"
        "Is this product relevant to the user's query?\n"
        "Answer only: Yes or No\n"
        "Answer:"
    )


def _resolve_yes_no_ids(client):
    client._init_backend()
    tokenizer = client._backend.get_tokenizer()

    def _ids(token_strs):
        ids = []
        for s in token_strs:
            ids.extend(tokenizer.encode(s, add_special_tokens=False))
        return ids

    return _ids([" Yes", "Yes", " yes", "yes"]), _ids([" No", "No", " no", "no"])


_QWEN3_RERANKER_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
_QWEN3_RERANKER_PROMPT_PREFIX = (
    "<|im_start|>system\n"
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    f"<Instruct>: {_QWEN3_RERANKER_INSTRUCTION}\n"
    "<Query>: "
)
_QWEN3_RERANKER_DOCUMENT_TEMPLATE = (
    "\n<Document>: {document}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
_YES_TOKEN_STR = " yes"
_NO_TOKEN_STR = " no"

# BGE-reranker-v2-gemma (BAAI) uses Gemma2 chat format.
_BGE_GEMMA2_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
_BGE_GEMMA2_PROMPT_PREFIX = (
    "<bos><start_of_turn>user\n"
    f"<Instruct>: {_BGE_GEMMA2_INSTRUCTION}\n"
    "<Query>: "
)
_BGE_GEMMA2_DOCUMENT_TEMPLATE = (
    "\n<Document>: {document}\n"
    "<start_of_turn>model\n"
)

# RankLLaMA (castorini/rankllama-v1-7b-lora-passage) is a
# SequenceClassification reranker (num_labels=1) on top of Llama-2-7b-hf.
# It does NOT use a chat template or yes/no logits — instead it takes
# SentencePair tokenization ``"query: {q}" / "document: {title} {passage}"``
# and returns the raw classification logit per pair. The official scoring
# method is in ``QwenLocalClient.score_seqcls``.
_RANKLLAMA_BASE = (
    "/fs04/scratch2/hj82/wenyu/RAG/Llama-2-7b-hf"
)
_RANKLLAMA_PEFT = (
    "/fs04/scratch2/hj82/wenyu/RAG/rankllama-v1-7b-lora-passage"
)
_RANKLLAMA_TITLE_SEP = ""  # official format joins title+passage by space

# Active reranker variant. Hardcoded in __main__ to one of:
#   "qwen3"      - Qwen3-Reranker (default; Qwen chat template)
#   "bge_gemma2" - BAAI/bge-reranker-v2-gemma (Gemma2 chat template)
#   "rankllama"  - castorini/rankllama-v1-7b-lora-passage
#                  (SequenceClassification on Llama-2-7b-hf)
RERANKER_VARIANT = "qwen3"


def _active_prompt_prefix() -> str:
    if RERANKER_VARIANT == "qwen3":
        return _QWEN3_RERANKER_PROMPT_PREFIX
    if RERANKER_VARIANT == "bge_gemma2":
        return _BGE_GEMMA2_PROMPT_PREFIX
    if RERANKER_VARIANT == "rankllama":
        # The rankllama path does NOT use the chat-template prompt;
        # score_seqcls builds its own ``"query: {q}" / "document: ..."``
        # SentencePair tokenization internally.
        return ""
    raise ValueError(f"Unknown RERANKER_VARIANT: {RERANKER_VARIANT}")


def _active_document_template() -> str:
    if RERANKER_VARIANT == "qwen3":
        return _QWEN3_RERANKER_DOCUMENT_TEMPLATE
    if RERANKER_VARIANT == "bge_gemma2":
        return _BGE_GEMMA2_DOCUMENT_TEMPLATE
    if RERANKER_VARIANT == "rankllama":
        return "{document}"
    raise ValueError(f"Unknown RERANKER_VARIANT: {RERANKER_VARIANT}")


def _model_label_for_variant() -> str:
    if RERANKER_VARIANT == "qwen3":
        return "Qwen3-Reranker (see llm_client.DEFAULT_QWEN_MODEL)"
    if RERANKER_VARIANT == "bge_gemma2":
        return "BAAI/bge-reranker-v2-gemma (transformers score_pairs)"
    if RERANKER_VARIANT == "rankllama":
        return (
            "castorini/rankllama-v1-7b-lora-passage "
            "(Llama-2-7b-hf + LoRA, SequenceClassification logits)"
        )
    raise ValueError(f"Unknown RERANKER_VARIANT: {RERANKER_VARIANT}")


def _resolve_yes_no_ids(client) -> tuple[list[int], list[int]]:
    """Resolve yes/no token ids for Qwen3-Reranker's LogitScore head.

    Sentence-transformers 默认用 `` yes`` / `` no``(前导空格),Qwen3
    tokenizer 同时也有无空格 ``yes`` / ``Yes`` / ``No`` / ``no`` 等 token。
    返回所有这些 id,score_logit_diff 在生成首个 token 的 top-k logprobs
    中取最大值,容忍空格/大小写变体。
    """
    client._init_backend()
    # ``QwenLocalClient._backend`` is either a vLLM ``LLM`` (has
    # ``.get_tokenizer()``) or a ``(model, tokenizer)`` tuple under the
    # transformers backend. Branch on the container type so both paths work.
    if isinstance(client._backend, tuple):
        tok = client._backend[1]
    else:
        tok = client._backend.get_tokenizer()

    def _all_ids(token_strs):
        out = []
        for s in token_strs:
            ids = tok.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                out.append(ids[0])
            else:
                out.extend(ids)
        # De-dup while preserving order.
        seen = set()
        uniq = []
        for t in out:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        return uniq

    yes_ids = _all_ids([" Yes", "Yes", " yes", "yes"])
    no_ids = _all_ids([" No", "No", " no", "no"])
    if not yes_ids or not no_ids:
        raise RuntimeError(
            f"Failed to resolve yes/no token ids for Qwen3-Reranker; "
            f"yes={yes_ids} no={no_ids}"
        )
    return yes_ids, no_ids


def _build_reranker_prompt(query: str, doc_text: str) -> str:
    return (
        _active_prompt_prefix()
        + query
        + _active_document_template().format(document=doc_text)
    )


def llm_rerank_per_retriever(queries_text, topk_asins_list, asin_to_doc, client):
    """Rerank each retriever's Top-K candidates via the active reranker.

    Dispatches to two scoring paths based on ``RERANKER_VARIANT``:
      - ``"qwen3"`` (default): vLLM next-token LogitScore via
        ``client.score_logit_diff`` on the Qwen chat-template prompt.
      - ``"bge_gemma2"``: transformers cross-encoder score via
        ``client.score_pairs`` on the BAAI/FlagEmbedding official
        ``"<bos>{query}</s>\\n{paragraph}"`` prompt (last non-padding
        position yes/no logit diff → sigmoid).

    The rank-prior fusion ``final_score = 1.0*LLM + 0.1*rank_prior`` is
    applied identically for both variants so absolute numbers stay
    comparable across the Stage13 rerank variants.
    """
    if len(topk_asins_list) != len(queries_text):
        raise ValueError(
            f"Query/candidate count mismatch: {len(queries_text)} != {len(topk_asins_list)}"
        )
    n_source = len(topk_asins_list[0]) if topk_asins_list else 0
    if n_source < 1:
        raise ValueError("Each query must have at least one retrieval candidate")
    if any(len(row) != n_source for row in topk_asins_list):
        raise ValueError("Retrieval candidate rows have inconsistent lengths")
    n_candidates = min(LLM_RERANK_CANDIDATES, n_source)
    topk_asins_list = [row[:n_candidates] for row in topk_asins_list]

    yes_ids, no_ids = _resolve_yes_no_ids(client)

    flat = []
    keep = []
    pairs = []
    for qi, qtext in enumerate(queries_text):
        for retrieval_rank, cand_asin in enumerate(topk_asins_list[qi], start=1):
            cand_doc = asin_to_doc[cand_asin].strip()[:240]
            flat.append(_build_reranker_prompt(qtext, cand_doc))
            keep.append((qi, cand_asin, retrieval_rank))
            pairs.append((qtext, cand_doc))

    log(f"  Rerank variant={RERANKER_VARIANT}: {len(flat)} prompts "
        f"({len(queries_text)} queries × top-{n_candidates})")
    if flat:
        first_qi, first_asin, first_rank = keep[0]
        first_prompt = flat[0]
        log("  === FIRST CANDIDATE PROMPT DEBUG ===")
        log(f"  query_index={first_qi} candidate_asin={first_asin} original_rank={first_rank}")
        log(f"  prompt_chars={len(first_prompt)}")
        log("  prompt_begin")
        for prompt_line in first_prompt.splitlines():
            log("    " + prompt_line)
        log("  prompt_end")
    scores_by_qi = {qi: [] for qi in range(len(queries_text))}
    t0 = time.time()
    n_total = len(flat)
    all_scores: list[float] = []
    if RERANKER_VARIANT == "bge_gemma2":
        # Cross-encoder score_pairs via transformers backend. The bge_gemma2
        # path constructs its own prompt internally (``<bos>{q}</s>\n{d}``);
        # we therefore pass the raw (query, doc) pairs rather than ``flat``.
        score_batch = max(1, LLM_RERANK_BATCH // 32)
        all_scores = client.score_pairs(
            pairs, prompt_style="bge_gemma2",
            batch_size=score_batch,
            yes_tokens=yes_ids, no_tokens=no_ids,
        )
        if len(all_scores) != n_total:
            raise RuntimeError(
                f"BGE Gemma2 score_pairs output mismatch: "
                f"{len(all_scores)} != {n_total}"
            )
        log(f"  BGE Gemma2 score_pairs done in {time.time()-t0:.1f}s")
    elif RERANKER_VARIANT == "rankllama":
        # SequenceClassification reranker head — raw logits, no sigmoid
        # (ranking is order-sensitive only). score_seqcls builds the
        # SentencePair tokenization internally.
        score_batch = max(1, LLM_RERANK_BATCH // 32)
        all_scores = client.score_seqcls(
            pairs, batch_size=score_batch,
        )
        if len(all_scores) != n_total:
            raise RuntimeError(
                f"RankLLaMA score_seqcls output mismatch: "
                f"{len(all_scores)} != {n_total}"
            )
        log(f"  RankLLaMA score_seqcls done in {time.time()-t0:.1f}s")
    else:
        # Default (Qwen3) path: transformers score_pairs with the Qwen3
        # chat-template prompt. vLLM was removed from the whitelist on
        # 2026-09-22; this path serves Qwen3-Reranker entirely via
        # transformers.
        score_batch = max(1, LLM_RERANK_BATCH // 32)
        all_scores = client.score_pairs(
            pairs, prompt_style="qwen3",
            batch_size=score_batch,
            yes_tokens=yes_ids, no_tokens=no_ids,
        )
        if len(all_scores) != n_total:
            raise RuntimeError(
                f"Qwen3 score_pairs output mismatch: "
                f"{len(all_scores)} != {n_total}"
            )
        log(f"  Qwen3-Reranker (transformers) rerank done in {time.time()-t0:.1f}s")

    for (qi, asin, retrieval_rank), llm_score in zip(keep, all_scores):
        rank_prior = (n_candidates - retrieval_rank) / (n_candidates - 1) if n_candidates > 1 else 1.0
        final_score = LLM_WEIGHT * llm_score + RETRIEVAL_WEIGHT * rank_prior
        scores_by_qi[qi].append({
            "asin": asin,
            "original_rank": retrieval_rank,
            "retrieval_rank": retrieval_rank,
            "llm_score": llm_score,
            "rank_prior": rank_prior,
            "final_score": final_score,
            "raw_output": f"logit_diff={llm_score:.4f}",
        })

    results = []
    for qi, _q in enumerate(queries_text):
        candidates = scores_by_qi[qi]
        ranked = sorted(candidates, key=lambda x: (-x["final_score"], x["original_rank"]))
        retrieval_top10 = [x["asin"] for x in candidates[:LLM_RERANK_TOPK]]
        retrieval_ranks = [x["original_rank"] for x in candidates]
        final_ranks = [0] * len(candidates)
        for final_rank, cand in enumerate(ranked, start=1):
            final_ranks[cand["original_rank"] - 1] = final_rank
        kendall, spearman = _rank_correlation(retrieval_ranks, final_ranks)
        final_top10 = [x["asin"] for x in ranked[:LLM_RERANK_TOPK]]
        results.append({
            "top10": final_top10,
            "retrieval_top10": retrieval_top10,
            "target_asin": None,
            "candidates": ranked,
            "diagnostics": {
                "n_unique_scores": len({x["llm_score"] for x in candidates}),
                "n_unique_final_scores": len({x["final_score"] for x in candidates}),
                "kendall_tau_a": kendall,
                "spearman_rho": spearman,
                "top10_order_changed": final_top10 != retrieval_top10,
            },
        })
    return results

def compute_hit_metrics(reranked):
    n = len(reranked)
    if n == 0:
        raise ValueError("Cannot compute metrics for zero queries")
    hit = {k: 0 for k in KS}
    rr_sum = 0.0
    recall_at_100 = 0
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
        for cand in r["candidates"]:
            if cand["asin"] == target:
                recall_at_100 += 1
                break
    return {
        "n_queries": n,
        "hit@1": hit[1] / n,
        "hit@5": hit[5] / n,
        "Recall@100": recall_at_100 / n,
        "hit@10": hit[10] / n,
        "MRR": rr_sum / n,
    }


def compute_rerank_diagnostics(reranked, max_examples=20):
    if not reranked:
        raise ValueError("Cannot compute diagnostics for zero queries")
    all_candidates = [c for r in reranked for c in r["candidates"]]
    scores = [c["llm_score"] for c in all_candidates if "llm_score" in c]
    unique_scores = sorted(set(scores))
    mean = sum(scores) / len(scores) if scores else None
    variance = (sum((score - mean) ** 2 for score in scores) / len(scores)
                if scores else None)
    changed = [r for r in reranked if r["diagnostics"]["top10_order_changed"]]
    histogram = {f"{score:.4f}": count for score, count in sorted(Counter(scores).items())} if scores else {}
    examples = []
    for r in changed[:max_examples]:
        examples.append({
            "target_asin": r["target_asin"],
            "before_top10": r["retrieval_top10"],
            "after_top10": r["top10"],
        })
    return {
        "n_queries": len(reranked),
        "n_candidates": len(all_candidates),
        "llm_score_unique_values": unique_scores,
        "llm_score_mean": mean,
        "llm_score_std": variance ** 0.5 if variance is not None else None,
        "llm_score_histogram": histogram,
        "queries_top10_order_changed": len(changed),
        "queries_top10_order_changed_rate": len(changed) / len(reranked),
        "average_kendall_tau_a": sum(r["diagnostics"]["kendall_tau_a"] for r in reranked) / len(reranked),
        "average_spearman_rho": sum(r["diagnostics"]["spearman_rho"] for r in reranked) / len(reranked),
        "examples_before_after": examples,
    }


def compute_flip_rate(reranked, min_queries_per_asin=2):
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
    """Backwards-compat shim — moved to 14_typo_rerank/typo_rerank_eval.py."""
    raise RuntimeError(
        "load_typo_pairs has moved to 14_typo_rerank/typo_rerank_eval.py"
    )


def run_stage11_llm_rerank(smoke=False):
    log("=== Stage 13 / Stage 11 LLM rerank (per-retriever top-100 + Qwen P(Yes)) ===")
    entries = load_stage8_selection()
    selection_indices = list(range(len(entries)))
    if smoke:
        bm25_cache = load_topk_cache(STAGE11_TOPK_DIR, "bm25")
        if bm25_cache is None:
            raise FileNotFoundError("Stage 11 BM25 top-100 cache missing for eligible smoke")
        _asin_to_doc_smoke, _asins_smoke = load_corpus()
        eligible_keys = []
        for idx, (asin, query) in enumerate(entries):
            row_asins = {_asins_smoke[int(x)] for x in bm25_cache[idx][:LLM_RERANK_TOPK]}
            if asin in row_asins:
                eligible_keys.append((idx, (asin, query)))
                break
        if not eligible_keys:
            raise ValueError("No BM25 Hit@100 query available for eligible smoke")
        selection_indices = [eligible_keys[0][0]]
        entries = [eligible_keys[0][1]]
    log(f"  loaded {len(entries)} query pairs")

    asin_to_doc, asins = load_corpus()
    queries_asin_pairs = [(a, q) for a, q in entries]

    per_retr_topk_idx: dict[str, np.ndarray] = {}
    _retr_list = ["bm25"] if SMOKE and ONLY_BM25_SMOKE else RETRIEVERS
    _typo_retr_list = _retr_list if SMOKE else RETRIEVERS
    log(f"  smoke retriever set: stage11={_retr_list} stage12={_typo_retr_list}")
    for retr in _retr_list:
        topk_idx = load_topk_cache(STAGE11_TOPK_DIR, retr)
        if topk_idx is None:
            raise FileNotFoundError(f"Stage 11 top-100 cache missing for {retr}")
        per_retr_topk_idx[retr] = topk_idx
    if smoke:
        for retr, arr in per_retr_topk_idx.items():
            if arr.shape[0] < len(entries):
                raise ValueError(f"[{retr}] cache has {arr.shape[0]} rows, smoke requires {len(entries)}")
        per_retr_topk_idx = {retr: arr[selection_indices] for retr, arr in per_retr_topk_idx.items()}
    else:
        for retr, arr in per_retr_topk_idx.items():
            if arr.shape[0] != len(entries):
                raise ValueError(f"[{retr}] cache has {arr.shape[0]} rows, full requires {len(entries)}")

    topk_by_retr = {retr: idx_to_asin(arr, asins) for retr, arr in per_retr_topk_idx.items()}

    from llm_client import get_client
    # 2026-09-22: vLLM removed from the whitelist (Rule 9). All rerankers
    # — Qwen3, BGE Gemma2, RankLLaMA — run on the transformers backend.
    client = get_client(backend="transformers")

    per_retriever_metrics = {}
    per_retriever_flip = {}
    per_query_top10_by_retr = {}
    queries_text = [q for _a, q in queries_asin_pairs]

    for retr in _retr_list:
        topk_for_retr = topk_by_retr[retr]
        eligible = [
            i for i, (target, _query) in enumerate(queries_asin_pairs)
            if target in topk_for_retr[i][:LLM_RERANK_TOPK]
        ]
        excluded_no_hit10 = len(queries_asin_pairs) - len(eligible)
        if not eligible:
            raise ValueError(f"[{retr}] no target appears in original Top-10")
        eligible_queries = [queries_asin_pairs[i][1] for i in eligible]
        eligible_pairs = [queries_asin_pairs[i] for i in eligible]
        eligible_topk = [topk_for_retr[i] for i in eligible]
        log(f"\n  [{retr}] reranking eligible Hit@10 queries: "
            f"{len(eligible)}/{len(entries)} (excluded={excluded_no_hit10})")
        reranked = llm_rerank_per_retriever(eligible_queries, eligible_topk, asin_to_doc, client)
        for qi, record in enumerate(reranked):
            record["target_asin"] = eligible_pairs[qi][0]
            record["query"] = eligible_pairs[qi][1]

        # LLM is called only on eligible queries; final Hit/MRR/Flip are over all input queries.
        eligible_by_index = {idx: reranked[pos] for pos, idx in enumerate(eligible)}
        full_records = []
        for idx, (target, query) in enumerate(queries_asin_pairs):
            if idx in eligible_by_index:
                full_records.append(eligible_by_index[idx])
                continue
            original = topk_for_retr[idx][:LLM_RERANK_CANDIDATES]
            full_records.append({
                "target_asin": target,
                "query": query,
                "retrieval_top10": original[:LLM_RERANK_TOPK],
                "top10": original[:LLM_RERANK_TOPK],
                "candidates": [
                    {"asin": asin, "original_rank": rank, "retrieval_rank": rank}
                    for rank, asin in enumerate(original, start=1)
                ],
                "diagnostics": {
                    "n_unique_scores": 0,
                    "kendall_tau_a": 1.0,
                    "spearman_rho": 1.0,
                    "top10_order_changed": False,
                    "excluded_no_hit10": True,
                },
            })
        m = compute_hit_metrics(full_records)
        m["n_input_queries"] = len(queries_asin_pairs)
        m["n_eligible_hit10"] = len(eligible)
        m["n_excluded_no_hit10"] = excluded_no_hit10
        eligible_m = compute_hit_metrics(reranked)
        eligible_m["n_input_queries"] = len(queries_asin_pairs)
        eligible_m["n_eligible_hit10"] = len(eligible)
        eligible_m["n_excluded_no_hit10"] = excluded_no_hit10
        f = compute_flip_rate(full_records)
        eligible_f = compute_flip_rate(reranked)
        flip_dict = f
        d = compute_rerank_diagnostics(reranked)
        per_retriever_metrics[retr] = m
        per_retriever_flip[retr] = f
        per_query_top10_by_retr[retr] = full_records
        log(f"  [{retr}] ALL queries n={len(full_records)} eligible={len(eligible)} "
            f"hit@1={m['hit@1']*100:.2f}% hit@10={m['hit@10']*100:.2f}% "
            f"MRR={m['MRR']*100:.2f}% Recall@100={m['Recall@100']*100:.2f}% "
            f"RR_Std={f['RR_Std_mean']:.4f}")
        log(f"    LLM score unique={len(d['llm_score_unique_values'])} "
            f"mean={d['llm_score_mean']} std={d['llm_score_std']} "
            f"top10_changed={d['queries_top10_order_changed']}/{d['n_queries']} "
            f"Kendall={d['average_kendall_tau_a']:.4f} "
            f"Spearman={d['average_spearman_rho']:.4f}")

        retr_out_path = OUT_DIR / f"per_retr_{retr}.json"
        with open(retr_out_path, "w") as fout:
            json.dump({
                "retriever": retr,
                "metrics": m,
                "flip_rate": flip_dict,
                "diagnostics": d,
                "per_query_top10_by_retriever": per_query_top10_by_retr[retr],
            }, fout, indent=2, default=str)
        log(f"  → per-retriever saved to {retr_out_path}")

        cumulative_path = STAGE11_OUT.with_name(STAGE11_OUT.stem + "_cumulative.json")
        cumulative = {
            "config": {
                "rerank_topk": LLM_RERANK_TOPK,
                "n_pairs": len(entries),
                "smoke": smoke,
                "retrievers": list(per_retriever_metrics.keys()),
                "score_range": [0.0, 1.0],
                "score_semantics": "P(Yes) over (Yes, No) via vLLM sampled-token logprobs",
                "max_new_tokens": 1,
                "tie_break": "retrieval_rank",
                "candidate_depth": 100,
                "rerank_scope": "per-retriever Top-100",
            },
            "metrics_by_retriever": per_retriever_metrics,
            "flip_rate_by_retriever": per_retriever_flip,
            "per_query_top10_by_retriever": per_query_top10_by_retr,
        }
        with open(cumulative_path, "w") as fout:
            json.dump(cumulative, fout, indent=2, default=str)
        log(f"  → cumulative saved to {cumulative_path}")

    out = {
        "config": {
            "rerank_topk": LLM_RERANK_TOPK,
            "n_pairs": len(entries),
            "smoke": smoke,
            "retrievers": list(RETRIEVERS),
            "score_range": [0.0, 1.0],
            "score_semantics": "P(Yes) over (Yes, No) via vLLM sampled-token logprobs",
            "max_new_tokens": 1,
            "tie_break": "retrieval_rank",
            "candidate_depth": 100,
            "rerank_scope": "per-retriever Top-100",
        },
        "metrics_by_retriever": per_retriever_metrics,
        "flip_rate_by_retriever": per_retriever_flip,
        "rerank_diagnostics_by_retriever": {
            retr: compute_rerank_diagnostics([
                {
                    "target_asin": item["target_asin"],
                    "top10": item["top10"],
                    "retrieval_top10": item["retrieval_top10"],
                    "candidates": item["candidates"],
                    "diagnostics": item["diagnostics"],
                }
                for item in per_query_top10_by_retr[retr]
            ])
            for retr in per_query_top10_by_retr
        },
        "per_query_top10_by_retriever": per_query_top10_by_retr,
    }

    # Aggregated per-retriever summary_table (single source of truth):
    # combines absolute Stage13 metrics with per-ASIN Flip + RR_Std, so the
    # downstream consumer can read all Stage13 numbers from one block.
    summary_rows = []
    for retr in RETRIEVERS:
        m = per_retriever_metrics.get(retr, {})
        f = per_retriever_flip.get(retr, {})
        summary_rows.append({
            "retriever": retr,
            "stage13_n_queries": m.get("n_queries"),
            "stage13_hit@1": m.get("hit@1"),
            "stage13_hit@5": m.get("hit@5"),
            "stage13_hit@10": m.get("hit@10"),
            "stage13_MRR": m.get("MRR"),
            "stage13_Recall@100": m.get("Recall@100"),
            "stage13_flip@1": f.get("Hit@1_FlipRate_mean"),
            "stage13_flip@5": f.get("Hit@5_FlipRate_mean"),
            "stage13_flip@10": f.get("Hit@10_FlipRate_mean"),
            "stage13_flip@20": f.get("Hit@20_FlipRate_mean"),
            "stage13_flip_n_asins": f.get("n_asins"),
            "stage13_RR_Std_mean": f.get("RR_Std_mean"),
            "stage13_RR_Std_median": f.get("RR_Std_median"),
            "stage13_RR_Std_std": f.get("RR_Std_std"),
        })
    out["summary_table"] = {
        "columns": ["retriever", "stage13_n_queries",
                    "stage13_hit@1", "stage13_hit@5", "stage13_hit@10", "stage13_MRR",
                    "stage13_Recall@100",
                    "stage13_flip@1", "stage13_flip@5", "stage13_flip@10",
                    "stage13_flip@20", "stage13_flip_n_asins",
                    "stage13_RR_Std_mean", "stage13_RR_Std_median", "stage13_RR_Std_std"],
        "flip_definition": (
            "Per-ASIN query-pair Hit@K disagreement rate among Stage13 rerank "
            "Top-K vs Stage11 baseline Top-K (mean over ASINs with >=2 queries, "
            "no sim threshold filter)."
        ),
        "rerank_weights": "final_score = 1.0 * LLM_score + 0.1 * rank_prior",
        "model": _model_label_for_variant(),
        "rows": summary_rows,
    }
    with open(STAGE11_OUT, "w") as fout:
        json.dump(out, fout, indent=2, default=str)
    log(f"\n  wrote → {STAGE11_OUT}")
    return out


def run_stage12_typo_paired_degradation(smoke=False, orig_rerank=None):
    """Placeholder kept for backward import compatibility.

    2026-09-21: typo rerank relocated to ``14_typo_rerank/typo_rerank_eval.py``.
    Importing the function here raises so any caller must migrate.
    """
    raise RuntimeError(
        "run_stage12_typo_paired_degradation has moved to "
        "14_typo_rerank/typo_rerank_eval.py"
    )


STAGE11_PER_QUERY = REPO_ROOT / "result/11_syntactic_evaluation/per_query.json"


def load_stage11_baseline_metrics() -> dict[str, dict]:
    """Aggregate Stage 11 retrieval baseline hit@1/@5/@10/MRR per retriever."""
    if not STAGE11_PER_QUERY.exists():
        raise FileNotFoundError(f"Stage 11 per_query.json missing: {STAGE11_PER_QUERY}")
    with open(STAGE11_PER_QUERY) as f:
        d = json.load(f)
    queries = d["queries"]
    if not queries:
        raise ValueError("Stage 11 per_query.json has empty queries")
    retrs = ["bm25", "splade", "minilm", "mpnet", "bge_base_v15", "gte_base", "colbertv2"]
    out: dict[str, dict] = {retr: {} for retr in retrs}
    for retr in retrs:
        for metric_norm, key_suffix in (("hit@1", "hit1"), ("hit@5", "hit5"),
                                        ("hit@10", "hit10"), ("MRR", "RR")):
            key = f"{retr}_{key_suffix}"
            vals = [float(q[key]) for q in queries
                    if q.get(key) is not None]
            if not vals:
                continue
            out[retr][metric_norm] = float(np.mean(vals))
    return out


def build_stage11_vs_stage13_table(stage11_metrics: dict,
                                   stage13_metrics: dict) -> list[dict]:
    """Build per-retriever comparison rows Stage11 vs Stage13."""
    rows = []
    for retr in sorted(set(stage11_metrics) | set(stage13_metrics)):
        s11 = stage11_metrics.get(retr, {})
        s13 = stage13_metrics.get(retr, {})
        row = {"retriever": retr}
        for k in ("hit@1", "hit@5", "hit@10", "MRR"):
            v11 = s11.get(k)
            v13 = s13.get(k)
            row[f"stage11_{k}"] = v11
            row[f"stage13_{k}"] = v13
            row[f"delta_{k}"] = (v13 - v11) if (v11 is not None and v13 is not None) else None
        rows.append(row)
    return rows


def print_and_save_stage11_vs_stage13(stage13_out: dict,
                                      out_path: Path,
                                      label: str = "Stage 13") -> None:
    """Print + save Stage11 baseline vs Stage13 rerank side-by-side table.

    Each row carries:
      - absolute Stage11 / Stage13 hit@{1,5,10}, MRR, Recall@100
      - Stage13 Flip@{1,5,10,20}, RR_Std_mean (per-ASIN query-pair disagreement
        rate among Stage13 rerank Top-K vs Stage11 baseline Top-K)
    All numeric values are stored as raw floats (not pre-formatted strings).
    """
    stage11_metrics = load_stage11_baseline_metrics()
    stage13_metrics = stage13_out.get("metrics_by_retriever", {})
    stage13_flip = stage13_out.get("flip_rate_by_retriever", {})
    base_rows = build_stage11_vs_stage13_table(stage11_metrics, stage13_metrics)
    retr_to_row = {r["retriever"]: r for r in base_rows}
    for retr in set(retr_to_row) | set(stage13_flip):
        fr = stage13_flip.get(retr, {})
        row = retr_to_row.setdefault(retr, {"retriever": retr})
        for k in (1, 5, 10, 20):
            row[f"stage13_flip@{k}"] = fr.get(f"Hit@{k}_FlipRate_mean")
        row["stage13_flip_n_asins"] = fr.get("n_asins")
        row["stage13_RR_Std_mean"] = fr.get("RR_Std_mean")
        row["stage13_RR_Std_median"] = fr.get("RR_Std_median")
        row["stage13_RR_Std_std"] = fr.get("RR_Std_std")
    rows = [retr_to_row[retr] for retr in sorted(retr_to_row)]
    log(f"\n=== {label}: Stage 11 baseline vs Stage 13 rerank ===")
    header = ("  retr         hit@1(11→13 Δ)        hit@5(11→13 Δ)        "
              "hit@10(11→13 Δ)       MRR(11→13 Δ)")
    log(header)
    log("  " + "-" * (len(header) - 2))
    for r in rows:
        cells = []
        for k in ("hit@1", "hit@5", "hit@10", "MRR"):
            v11 = r.get(f"stage11_{k}"); v13 = r.get(f"stage13_{k}"); d = r.get(f"delta_{k}")
            if v11 is None or v13 is None:
                cells.append(f"    N/A      ")
            else:
                cells.append(f"{v11*100:6.2f}%→{v13*100:6.2f}%({d*100:+5.2f})")
        f1 = r.get("stage13_flip@1"); f5 = r.get("stage13_flip@5")
        f10 = r.get("stage13_flip@10"); f20 = r.get("stage13_flip@20")
        rr_std = r.get("stage13_RR_Std_mean")
        n_a = r.get("stage13_flip_n_asins")
        def pc(v): return f"{v*100:>6.2f}%" if v is not None else "   N/A "
        rr_std_s = f"{rr_std:>6.3f}" if rr_std is not None else "   N/A"
        n_a_s = f"n_asins={n_a}" if n_a is not None else ""
        log(f"  {r['retriever']:12s}  " + "  ".join(cells)
            + f"  flip: {pc(f1)} {pc(f5)} {pc(f10)} {pc(f20)}  RR_Std={rr_std_s}  {n_a_s}")
    payload = {
        "config": {"label": label,
                   "stage11_source": str(STAGE11_PER_QUERY),
                   "stage13_source": str(STAGE11_OUT),
                   "flip_definition": (
                       "Per-ASIN query-pair Hit@K disagreement rate among Stage13 "
                       "rerank Top-K vs Stage11 baseline Top-K (mean over ASINs "
                       "with >=2 queries, no sim threshold filter)."
                   ),
                   "rerank_weights": "final_score = 1.0 * LLM_score + 0.1 * rank_prior",
                   "model": _model_label_for_variant()},
        "rows": rows,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    log(f"\n  wrote → {out_path}")


if __name__ == "__main__":
    RUN_STAGE11 = True
    RUN_STAGE12 = False

    # === reranker selection ===
    # Hardcode the active reranker here. Switch ``RERANKER_VARIANT`` and the
    # llm_client model path together. Supported variants:
    #   "qwen3"      -> Qwen3-Reranker (default; uses Qwen chat template)
    #   "bge_gemma2" -> BAAI/bge-reranker-v2-gemma (Gemma2 chat template)
    #   "rankllama"  -> castorini/rankllama-v1-7b-lora-passage
    #                   (SequenceClassification on Llama-2-7b-hf)
    RERANKER_VARIANT = "qwen3"
    if RERANKER_VARIANT == "qwen3":
        import llm_client  # noqa: E402
        llm_client.DEFAULT_QWEN_MODEL = "/home/wlia0047/hj82_scratch2/wenyu/RAG/Qwen3-Reranker-8B"
        llm_client.DEFAULT_PEFT_ADAPTER = None
        llm_client.reset_client()
    elif RERANKER_VARIANT == "bge_gemma2":
        import llm_client  # noqa: E402
        llm_client.DEFAULT_QWEN_MODEL = "/home/wlia0047/hj82_scratch2/wenyu/RAG/BGE-reranker-Gemma2-9B"
        llm_client.DEFAULT_PEFT_ADAPTER = None
        llm_client.reset_client()
    elif RERANKER_VARIANT == "rankllama":
        # RankLLaMA = Llama-2-7b base + PEFT LoRA merged into a
        # SequenceClassification head (num_labels=1). Forces transformers.
        import llm_client  # noqa: E402
        llm_client.DEFAULT_QWEN_MODEL = _RANKLLAMA_BASE
        llm_client.DEFAULT_PEFT_ADAPTER = _RANKLLAMA_PEFT
        llm_client.reset_client()
    else:
        raise ValueError(f"Unknown RERANKER_VARIANT: {RERANKER_VARIANT}")

    log(f"=== Stage 13 / Stage 11 syntactic rerank (per-retriever P(Yes)) ===")
    log(f"  SMOKE={SMOKE}  RUN_STAGE11={RUN_STAGE11}  RUN_STAGE12={RUN_STAGE12}  "
        f"RERANKER_VARIANT={RERANKER_VARIANT}")
    t0 = time.time()
    if RUN_STAGE11:
        out13 = run_stage11_llm_rerank(smoke=SMOKE)
        log(f"=== Stage 11 syntactic rerank done in {time.time()-t0:.1f}s ===")
        print_and_save_stage11_vs_stage13(out13, OUT_DIR / "stage11_vs_stage13.json",
                                         label="Stage 13")
    if RUN_STAGE12:
        raise RuntimeError(
            "Typo rerank moved to 14_typo_rerank/typo_rerank_eval.py; "
            "do not invoke from this script."
        )
    log(f"=== total: {time.time()-t0:.1f}s ===")