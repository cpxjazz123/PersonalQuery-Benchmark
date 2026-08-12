#!/usr/bin/env python3
"""[Reviewer-pilot] iter #84: LLM full-set quality eval (paper §3.3 Table 2 right panel).

Paper Table 2 reports three aggregate %s on the LLM full-set:
  - Semantic plausibility rate: 96.7%
  - Target Structure Conformity: 97.3%
  - Semantic preservation after error injection: 94.6%

This script emits those three pass-rates by running real-LLM-as-judge
(Qwen2.5-7B-Instruct via vLLM :8000) on:
  - clean queries from Stage 06 per-user list
    (result/personal_query/06_query/<cat>/query_by_syntax_depth_vades_lite_sentence_user_distribution_train10_holdout10.json)
  - noisy query pairs (clean, noisy) from Stage 07
    (result/personal_query/07_inject_noisy/<cat>/noisy_query.json)

Pilot design: N_PILOT_PER_DOMAIN clean queries per domain × 3 domains
(configurable via PERSOANLQUERY_FULLSET_N env, default 50) for plausibility
and structure-conformity; ALL available noisy pairs for semantic-preservation.

Three binary pass/fail system prompts anchored on paper §3.3 wording.

Output:
  result/personal_query/02_writing_analysis/llm_human_eval/llm_full_set_quality_eval.json
  stdout cross-metric × cross-domain table

Run: python3 02_writing_analysis/llm_full_set_quality_eval.py
"""

from __future__ import annotations

import json
import os
import random
import re
import statistics
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu")
EVAL_DIR = REPO_ROOT / "result" / "personal_query" / "02_writing_analysis" / "llm_human_eval"

VLLM_URL = "http://localhost:8000/v1/chat/completions"
VLLM_MODEL = "Qwen2.5-7B-Instruct"
N_PILOT_PER_DOMAIN = int(os.environ.get("PERSOANLQUERY_FULLSET_N", "50"))
SEED = 42
TIMEOUT_S = 60

CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]

CLEAN_QUERY_PATH = (
    "result/personal_query/06_query/{cat}/"
    "query_by_syntax_depth_vades_lite_sentence_user_distribution_train10_holdout10.json"
)
NOISY_QUERY_PATH = "result/personal_query/07_inject_noisy/{cat}/noisy_query.json"

PROMPT_PLAUSIBILITY = (
    "You are an expert e-commerce search-query evaluator. Given a user-written "
    "search query, decide whether it is SEMANTICALLY PLAUSIBLE as a real shopping "
    "search query (i.e., someone could plausibly type it into Amazon search to "
    "find a product). A query is plausible if it expresses a coherent product "
    "intent (what the user wants to buy) even if the grammar or wording is "
    "informal. A query is implausible if it is gibberish, has no identifiable "
    "product intent, or is internally contradictory in a way that makes "
    "retrieval meaningless.\n\n"
    "Respond with EXACTLY one line:\n"
    "VERDICT: PASS  (if the query is semantically plausible)\n"
    "VERDICT: FAIL  (if not plausible)\n"
    "Justify in one short sentence on the next line starting with REASON:"
)

PROMPT_STRUCTURE_CONFORMITY = (
    "You are an expert e-commerce search-query evaluator. Given a user-written "
    "search query, decide whether it has the TARGET STRUCTURE expected of a "
    "well-formed shopping query. The expected structure includes: (i) clear "
    "reference to a PRODUCT CATEGORY (what kind of item), (ii) optionally one "
    "or more CONSTRAINTS (brand, material, size, price, color, age range), and "
    "(iii) a SEARCH INTENT verb or formulation (e.g., 'looking for', 'show me', "
    "'need', 'find', or an imperative noun phrase). Queries that only mention a "
    "category without any intent framing are FAIL. Queries that mention "
    "constraints but no clear category are FAIL. Queries that are just keyword "
    "stacks with no syntactic structure are FAIL.\n\n"
    "Respond with EXACTLY one line:\n"
    "VERDICT: PASS  (if the query matches the target structure)\n"
    "VERDICT: FAIL  (otherwise)\n"
    "Justify in one short sentence on the next line starting with REASON:"
)

PROMPT_PRESERVATION = (
    "You are an expert e-commerce search-query evaluator. You are given a "
    "CLEAN query and a NOISY version of the same query (the noisy version has "
    "a writing error such as a typo, homophone swap, or word omission). "
    "Decide whether the NOISY query still PRESERVES the original retrieval "
    "intent of the CLEAN query — i.e., a search engine given the noisy query "
    "would still return essentially the same set of relevant products.\n\n"
    "CLEAN: {clean}\n\n"
    "NOISY: {noisy}\n\n"
    "Respond with EXACTLY one line:\n"
    "VERDICT: PASS  (if intent preserved)\n"
    "VERDICT: FAIL  (if intent lost)\n"
    "Justify in one short sentence on the next line starting with REASON:"
)


def _post_chat(messages: list[dict], max_tokens: int = 128, temperature: float = 0.0) -> str:
    """POST a chat-completion request to the local vLLM server; return assistant text."""
    payload = {
        "model": VLLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    req = urllib.request.Request(
        VLLM_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def _parse_verdict(text: str) -> tuple[str, str]:
    """Extract VERDICT and REASON from LLM response. Raises on malformed output.

    Tolerant parsing: VERDICT and REASON may appear on the same line, or the
    LLM may continue with explanatory text after the VERDICT tag. We extract
    the first PASS/FAIL token from the VERDICT line.
    """
    verdict = None
    reason = ""
    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith("VERDICT:"):
            tag_part = s.split(":", 1)[1].strip().upper()
            # Take the first token (PASS or FAIL) even if more text follows.
            token = tag_part.split()[0] if tag_part else ""
            if token not in ("PASS", "FAIL"):
                raise ValueError(f"VERDICT must be PASS or FAIL; got: {tag_part!r}")
            verdict = token
            # If REASON appears on the same line after VERDICT, capture it.
            tail = tag_part[len(token):].strip()
            if tail.upper().startswith("REASON:"):
                reason = tail.split(":", 1)[1].strip()
        elif s.upper().startswith("REASON:"):
            reason = s.split(":", 1)[1].strip()
    if verdict is None:
        raise ValueError(f"no VERDICT line in LLM response: {text!r}")
    return verdict, reason


def judge(system_prompt: str, user_text: str) -> tuple[str, str]:
    """Single LLM judge call: returns (verdict, reason)."""
    raw = _post_chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
    )
    return _parse_verdict(raw)


def _load_clean_queries(cat: str) -> list[dict]:
    """Load Stage 06 per-user query list, flatten to list of {user_id, asin, query, attrs_used}."""
    path = REPO_ROOT / CLEAN_QUERY_PATH.format(cat=cat)
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    flat = []
    for row in rows:
        sdqs = row.get("syntax_depth_queries", [])
        for q in sdqs:
            text = q.get("query")
            if not isinstance(text, str) or not text.strip():
                continue
            flat.append(
                {
                    "user_id": row.get("user_id"),
                    "asin": row.get("asin"),
                    "query": text.strip(),
                    "attrs_used": q.get("attrs_used", {}),
                    "word_count": q.get("word_count"),
                }
            )
    return flat


def _load_noisy_pairs(cat: str) -> list[dict]:
    """Load Stage 07 noisy pairs as list of {uid, asin, clean, noisy}."""
    path = REPO_ROOT / NOISY_QUERY_PATH.format(cat=cat)
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    out = []
    for r in rows:
        if r.get("status") != "success":
            continue
        c = r.get("clean_query")
        n = r.get("noisy_query")
        if not (isinstance(c, str) and isinstance(n, str) and c.strip() and n.strip()):
            continue
        out.append(
            {
                "uid": r.get("uid"),
                "asin": r.get("asin"),
                "clean": c.strip(),
                "noisy": n.strip(),
                "applied_error": r.get("applied_error"),
            }
        )
    return out


def _sample_clean(cat: str, n: int, rng: random.Random) -> list[dict]:
    """Sample N clean queries from the full per-domain pool."""
    pool = _load_clean_queries(cat)
    if not pool:
        raise ValueError(f"{cat}: no clean queries loaded from Stage 06 output")
    if len(pool) <= n:
        return pool
    return rng.sample(pool, n)


def _metric_pass_rate(judgments: list[tuple[str, str, str]]) -> dict:
    """Aggregate pass-rate from a list of (query, verdict, reason)."""
    n = len(judgments)
    n_pass = sum(1 for _, v, _ in judgments if v == "PASS")
    return {
        "n": n,
        "n_pass": n_pass,
        "pass_rate": (n_pass / n) if n else None,
    }


def evaluate_domain(cat: str, rng: random.Random, n_pilot: int) -> dict:
    """Run all three LLM-judge metrics for one domain."""
    # 1) plausibility on N_pilot clean queries
    plaus_pool = _sample_clean(cat, n_pilot, rng)
    plaus_judgments = []
    for q in plaus_pool:
        v, r = judge(PROMPT_PLAUSIBILITY, q["query"])
        plaus_judgments.append((q["query"], v, r))
    # 2) structure conformity on the same N_pilot pool
    struct_judgments = []
    for q in plaus_pool:
        v, r = judge(PROMPT_STRUCTURE_CONFORMITY, q["query"])
        struct_judgments.append((q["query"], v, r))
    # 3) semantic preservation on ALL noisy pairs available
    noisy_pairs = _load_noisy_pairs(cat)
    pres_judgments = []
    for p in noisy_pairs:
        v, r = judge(
            PROMPT_PRESERVATION.format(clean=p["clean"], noisy=p["noisy"]),
            f"CLEAN: {p['clean']}\nNOISY: {p['noisy']}",
        )
        pres_judgments.append((f"clean={p['clean']!r}/noisy={p['noisy']!r}", v, r))
    return {
        "category": cat,
        "n_clean_pool_total": len(_load_clean_queries(cat)),
        "n_clean_sampled": len(plaus_judgments),
        "n_noisy_pairs_total": len(_load_noisy_pairs(cat)),
        "n_noisy_pairs_used": len(pres_judgments),
        "semantic_plausibility": {
            **_metric_pass_rate(plaus_judgments),
            "sample_first_3": [
                {"query": q, "verdict": v, "reason": r} for q, v, r in plaus_judgments[:3]
            ],
        },
        "target_structure_conformity": {
            **_metric_pass_rate(struct_judgments),
            "sample_first_3": [
                {"query": q, "verdict": v, "reason": r} for q, v, r in struct_judgments[:3]
            ],
        },
        "semantic_preservation_after_error_injection": {
            **_metric_pass_rate(pres_judgments),
            "sample_first_3": [
                {"pair": q, "verdict": v, "reason": r} for q, v, r in pres_judgments[:3]
            ],
        },
    }


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    per_domain = []
    for cat in CATEGORIES:
        print(f"\n=== {cat} ===", flush=True)
        result = evaluate_domain(cat, rng, N_PILOT_PER_DOMAIN)
        per_domain.append(result)
        sp = result["semantic_plausibility"]
        ts = result["target_structure_conformity"]
        pv = result["semantic_preservation_after_error_injection"]
        print(f"  Semantic plausibility (n={sp['n']}): {sp['pass_rate']}", flush=True)
        print(f"  Target Structure Conformity (n={ts['n']}): {ts['pass_rate']}", flush=True)
        print(f"  Semantic preservation (n={pv['n']}): {pv['pass_rate']}", flush=True)

    # Aggregate across domains (3-domain mean) — matches paper's reported numbers.
    def _mean_rate(key: str) -> Optional[float]:
        rates = [d[key]["pass_rate"] for d in per_domain if d[key]["pass_rate"] is not None]
        return statistics.mean(rates) if rates else None

    paper_baseline = {
        "semantic_plausibility_rate_paper": 0.967,
        "target_structure_conformity_rate_paper": 0.973,
        "semantic_preservation_after_error_injection_rate_paper": 0.946,
    }
    summary = {
        "model": VLLM_MODEL,
        "endpoint": VLLM_URL,
        "pilot_n_clean_per_domain": N_PILOT_PER_DOMAIN,
        "pilot_n_noisy_total": sum(d["n_noisy_pairs_used"] for d in per_domain),
        "3_domain_mean_pass_rate": {
            "semantic_plausibility": _mean_rate("semantic_plausibility"),
            "target_structure_conformity": _mean_rate("target_structure_conformity"),
            "semantic_preservation_after_error_injection": _mean_rate(
                "semantic_preservation_after_error_injection"
            ),
        },
        "paper_baseline": paper_baseline,
        "per_domain": per_domain,
    }

    out_path = EVAL_DIR / "llm_full_set_quality_eval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nWrote: {out_path}")
    print("\n=== 3-domain mean pass rates (LLM-as-judge pilot) ===")
    for k, v in summary["3_domain_mean_pass_rate"].items():
        print(f"  {k:55s} {v:.4f}" if v is not None else f"  {k}: n/a")
    print("\n=== paper baseline (Table 2 right panel) ===")
    for k, v in paper_baseline.items():
        print(f"  {k:55s} {v:.4f}")
    print("\nNB: Pilot uses N_PILOT_PER_DOMAIN clean queries (configurable via")
    print("    PERSOANLQUERY_FULLSET_N env) and ALL available noisy pairs.")
    print("    Full-set scaling requires Stage 07 noisy_query.json expansion.")


if __name__ == "__main__":
    main()