#!/usr/bin/env python3
"""E14 — Syntactically diversified query targets (direction 1).

The synthetic candidates are syntactically simple; the model collapses to a
narrow template space. This script rewrites each candidate into richer
syntactic variants (adverbial modifiers, coordination, relative clauses,
multiple modifiers) with a frozen 7B model (few-shot), keeping every product
attribute string verbatim (the copy path still guarantees fidelity at
generation). Output: hardcoded jsonl used as copy-aware training targets.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
BASE_MODEL = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"
SRC = REPO_ROOT / "result" / "personal_query" / "04_query" / "Baby_Products" / "query_by_syntax_depth_no_depth_check_10.json"
OUT = REPO_ROOT / "result" / "personal_query" / "e14_syntax_diverse_targets.jsonl"
VARIANTS_PER_QUERY = 2
MAX_QUERIES = 400  # smoke budget; full = 2050


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


FEW_SHOT = [
    ("I'm looking for KK BETO Ties that are Small for Baby at 8.99.",
     "Honestly, I'm looking for KK BETO Ties that are Small, which I want to use for my Baby, and they are priced at 8.99.",
     "I've been searching for Small KK BETO Ties for a while now, and since my Baby needs them at 8.99, I'd like to get a pair."),
    ("I want to buy Eanceil Hair Clippers for Kids in Light Green at 26.99.",
     "Actually, I want to buy Eanceil Hair Clippers that are designed for Kids, in a nice Light Green color, costing 26.99.",
     "I am hoping to find Eanceil Hair Clippers, because my Kids really need one, in Light Green, and my budget is 26.99."),
]

REWRITE_PROMPT = (
    "Rewrite the shopping query into {n} DIFFERENT sentences. Each rewrite "
    "must: (1) keep every product attribute value verbatim (brands, prices, "
    "colors, sizes, use-cases); (2) use a richer sentence structure — add an "
    "adverbial opener, a coordinating clause, a relative clause, or more "
    "modifiers. Examples:\\n{few_shot}\\n"
    "Query: {query}\\n"
    "Rewrite 1:"
)


def main() -> None:
    src = json.load(open(SRC))
    rows = []
    n_queries = 0
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16,
                                                 device_map="cuda:0", trust_remote_code=True)
    model.eval()
    fs = "\n".join(
        f"Query: {q}\nRewrite 1: {r1}\nRewrite 2: {r2}"
        for q, r1, r2 in FEW_SHOT
    )
    log(f"few-shot: {len(FEW_SHOT)} pairs")
    for rec in src:
        attrs = (rec.get("syntax_depth_query") or {}).get("attrs_used")
        candidates = rec.get("syntax_depth_queries", [])
        if not attrs:
            for cand in candidates:
                if cand.get("attrs_used"):
                    attrs = cand["attrs_used"]
                    break
        if not attrs:
            continue
        for cand in candidates[:2]:  # first 2 candidates per product
            q = cand.get("query", "")
            if not q:
                continue
            prompt = REWRITE_PROMPT.format(n=VARIANTS_PER_QUERY, few_shot=fs, query=q)
            ids = tok.encode(prompt, add_special_tokens=False, return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                out = model.generate(input_ids=ids, max_new_tokens=128, do_sample=False,
                                     pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            text = tok.decode(out[0][ids.size(1):], skip_special_tokens=True)
            parts = [p.strip() for p in text.replace("Rewrite 2:", "\n<<<S>>>").split("\n<<<S>>>")]
            variants = [p for p in parts if p][:VARIANTS_PER_QUERY]
            for v in variants:
                rows.append({"user_id": rec["user_id"], "asin": rec["asin"],
                             "attrs_used": attrs, "y_plus_query": v,
                             "source_query": q})
            n_queries += 1
            if n_queries >= MAX_QUERIES:
                break
        if n_queries >= MAX_QUERIES:
            break
        if len(rows) % 100 == 0:
            log(f"  {len(rows)} variants...")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"wrote {OUT}: {len(rows)} targets")
    log("=== done ===")


if __name__ == "__main__":
    main()
