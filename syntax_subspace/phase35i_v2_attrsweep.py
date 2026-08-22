#!/usr/bin/env python3
"""Phase 35.I-v2: Stricter attribute filter + no word limit.

Changes vs Phase 35.I:
  1. filter_useful_attrs_v2: skip phrase-style values (>4 words, contains . , ; : x-list)
  2. Skip duplicate keys (case-insensitive)
  3. Remove "15-30 words" hard constraint in system prompt
  4. Let model decide length naturally
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/syntax_subspace")
PHASE35E_DIR = REPO_ROOT / "result/phase35e"
OUT_DIR = REPO_ROOT / "result/phase35i_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_ATTRS_LIST = [4, 5, 6, 7, 8, 9, 10]
K_SAMPLES = int(os.environ.get("P35I_K", "4"))
GEN_BATCH = int(os.environ.get("P35I_BATCH", "16"))
GEN_MAX_NEW = int(os.environ.get("P35I_MAX_NEW", "256"))
GEN_TEMP = float(os.environ.get("P35I_TEMP", "0.7"))
GEN_TOP_P = float(os.environ.get("P35I_TOP_P", "0.95"))
GEN_REP_PENALTY = float(os.environ.get("P35I_REP_PENALTY", "1.1"))
N_EXEMPLARS = 3
MAX_INPUT_LENGTH = 480
MASK_CJK = os.environ.get("P35I_MASK_CJK", "1") == "1"

# === V2 PROMPT: no length constraint ===
GEN_SYSTEM = (
    "You are an Amazon shopper writing a search query. Use EVERY attribute "
    "value below verbatim (mention duplicates only once). DO NOT add any "
    "product type (no bottle/clothes/toy/candle/tumbler), use case, personal "
    "context, or inferred property. DO NOT include attribute field names "
    "(no 'Brand:', 'material_type:'). Each value keeps its attribute meaning. "
    "Write a natural query with varied syntax. Length is up to you; include "
    "every attribute value. Output ONLY the query, no preamble.\n\n"
    "Attributes:\n{ATTRIBUTES}\n\n"
    "Style reference (your past searches, write a query with similar syntax):\n"
    "{EXEMPLARS}"
)


def build_attr_block(attrs: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in attrs.items() if v)


def build_exemplar_block(exemps: list) -> str:
    return "\n".join(f"- {e}" for e in exemps[:N_EXEMPLARS])


# V2 — STRICTER filter
USEFUL_ATTR_KEYS_SKIP = {
    "Customer Reviews", "Best Sellers Rank", "Amazon Best Sellers Rank",
    "Date First Available", "Average Rating", "Rating", "ASIN",
    "Is Discontinued By Manufacturer", "Batteries required", "Batteries included",
    "Number Of Items", "Item model number", "Model Name", "Manufacturer",
    "Country/Region of origin", "Specific Uses For Product",
    "International Shipping", "Domestic Shipping",
    "Care instructions", "Warranty Description", "Warranty Type",
    "Safety warning", "Directions", "Important information",
    "Batteries Required?", "Item Package Quantity", "Item Weight",
    "Pattern", "Shape", "Included Components", "Special Feature",
    "Style",  # ambiguous
}

VALUE_BAD_CHARS = {".", ",", ";", ":", "(", ")"}
MAX_VALUE_WORDS = 4  # stricter than before (was lenient)


def is_bad_value(v: str) -> bool:
    if not v: return True
    s = str(v).strip()
    if s.lower() in ("", "no", "yes", "unknown", "none", "n/a", "0", "1"): return True
    if len(s) <= 1: return True
    if any(c in s for c in VALUE_BAD_CHARS): return True
    if len(s.split()) > MAX_VALUE_WORDS: return True
    return False


def filter_useful_attrs_v2(raw_attrs: dict) -> dict:
    """Stricter: skip phrase-style values, dedup case-insensitive keys."""
    seen_keys_lower = set()
    out = {}
    for k, v in raw_attrs.items():
        if k in USEFUL_ATTR_KEYS_SKIP: continue
        if not v: continue
        kl = k.lower()
        if kl in seen_keys_lower: continue  # dedup
        if is_bad_value(v): continue
        seen_keys_lower.add(kl)
        out[k] = str(v).strip()
    return out


def main() -> int:
    t0 = time.time()
    records = json.load(open(SCRATCH / "phase35b_records.json"))
    user_sents = json.load(open(SCRATCH / "user_sents_phase35b.json"))
    prod_attrs = json.load(open(REPO_ROOT / "result/product_attributes.json"))
    print(f"[load] {len(records)} records, {len(user_sents)} users")

    records_with_attrs = []
    for r in records:
        uid = r["user_id"]
        asin = r["asin"]
        exemps = user_sents.get(uid, [])[:N_EXEMPLARS]
        if not exemps: continue
        full_attrs = filter_useful_attrs_v2(prod_attrs.get(asin, {}))
        sorted_keys = sorted(full_attrs.keys())
        sorted_attrs = {k: full_attrs[k] for k in sorted_keys}
        records_with_attrs.append({
            "user_id": uid,
            "asin": asin,
            "exemps": exemps,
            "sorted_keys": sorted_keys,
            "sorted_attrs": sorted_attrs,
            "n_full": len(full_attrs),
        })
    n_dist = Counter([r["n_full"] for r in records_with_attrs])
    print(f"[records] {len(records_with_attrs)} records with usable attrs")
    print(f"[n_full attrs dist] {dict(sorted(n_dist.items()))}")

    todo_per_n: dict[int, list[dict]] = {N: [] for N in N_ATTRS_LIST}
    for r in records_with_attrs:
        sa = r["sorted_attrs"]
        keys = r["sorted_keys"]
        for N in N_ATTRS_LIST:
            if len(keys) < N:
                continue
            top_keys = keys[:N]
            attrs_n = {k: sa[k] for k in top_keys}
            for k in range(K_SAMPLES):
                todo_per_n[N].append({
                    "user_id": r["user_id"],
                    "asin": r["asin"],
                    "attrs": attrs_n,
                    "exemps": r["exemps"],
                    "N": N,
                    "k": k,
                })

    print(f"[todo] prompts per N: {[(N, len(v)) for N, v in todo_per_n.items()]}")

    print("[main] loading Qwen...")
    sys.path.insert(0, str(REPO_ROOT))
    from llm_client import create_qwen_local_client
    client = create_qwen_local_client(with_vllm=False)
    print(f"[main] Qwen loaded, {time.time() - t0:.1f}s")

    cands_per_n: dict[int, list[dict]] = {N: [] for N in N_ATTRS_LIST}
    for N in N_ATTRS_LIST:
        todo = todo_per_n[N]
        if not todo: continue
        prompts = []
        for it in todo:
            attrs = it["attrs"]
            exemps = it["exemps"]
            prompt = (
                f"Attributes:\n{build_attr_block(attrs)}\n\n"
                f"Style reference (your past searches, similar syntax):\n"
                f"{build_exemplar_block(exemps)}"
            )
            prompts.append(prompt + f"\n<!-- variant: {it['k']} -->")

        import torch
        hidden_dim = client._hidden_backend.model.config.hidden_size
        zero_bias = torch.zeros(hidden_dim, dtype=torch.float32)
        injections = [zero_bias.clone() for _ in prompts]

        out_qs = []
        for i in range(0, len(prompts), GEN_BATCH):
            chunk_p = prompts[i:i + GEN_BATCH]
            chunk_r = todo[i:i + GEN_BATCH]
            chunk_inj = injections[i:i + GEN_BATCH]
            queries = client.generate_with_hidden_injection(
                system_text=GEN_SYSTEM,
                user_texts=chunk_p,
                injection_per_row=chunk_inj,
                injection_layers=[26],
                injection_alpha=0.0,
                max_new_tokens=GEN_MAX_NEW,
                temperature=GEN_TEMP,
                top_p=GEN_TOP_P,
                repetition_penalty=GEN_REP_PENALTY,
                batch_size=GEN_BATCH,
                max_input_length=MAX_INPUT_LENGTH,
                mask_cjk=MASK_CJK,
            )
            out_qs.extend(queries)
            done = min(i + GEN_BATCH, len(prompts))
            rate = done / max(time.time() - t0, 1e-6)
            print(f"[N={N}] {done}/{len(prompts)} ({rate:.2f}/s)", flush=True)

        for it, q in zip(todo, out_qs):
            cands_per_n[N].append({
                "user_id": it["user_id"],
                "asin": it["asin"],
                "attrs_used": it["attrs"],
                "N": N,
                "k": it["k"],
                "query": q,
            })

    for N in N_ATTRS_LIST:
        json.dump(cands_per_n[N], open(SCRATCH / f"phase35i_v2_cands_N{N}.json", "w"),
                  indent=2, ensure_ascii=False)
    print(f"\n[save] candidates per N → {SCRATCH}/phase35i_v2_cands_N{{N}}.json")
    return 0


if __name__ == "__main__":
    from collections import Counter
    sys.exit(main())
