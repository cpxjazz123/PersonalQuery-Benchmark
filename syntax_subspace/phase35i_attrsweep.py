#!/usr/bin/env python3
"""Phase 35.I: N_attrs sweep (4..10) — attribute count vs rerank quality.

For each record (74 records, 17 asins), for each N in [4..10]:
  - Use top-N attrs from product_attributes.json (deterministic order, useful filter)
  - Generate K candidates with Qwen
  - Score via Phase 35.G SOTA (softmax_g32_τ0.5)
  - Report per N: full_cov, intra-product Rank-1 size>=10, mean_own_rank

Goal: find optimal attribute count (does more attrs help/hurt ranking?).
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
OUT_DIR = REPO_ROOT / "result/phase35i"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_ATTRS_LIST = [4, 5, 6, 7, 8, 9, 10]
K_SAMPLES = int(os.environ.get("P35I_K", "4"))
GEN_BATCH = int(os.environ.get("P35I_BATCH", "16"))
GEN_MAX_NEW = int(os.environ.get("P35I_MAX_NEW", "128"))
GEN_TEMP = float(os.environ.get("P35I_TEMP", "0.7"))
GEN_TOP_P = float(os.environ.get("P35I_TOP_P", "0.95"))
GEN_REP_PENALTY = float(os.environ.get("P35I_REP_PENALTY", "1.1"))
N_EXEMPLARS = 3
MAX_INPUT_LENGTH = 480
MASK_CJK = os.environ.get("P35I_MASK_CJK", "1") == "1"

GEN_SYSTEM = (
    "You are an Amazon shopper writing a search query. Use EVERY attribute "
    "value below verbatim (mention duplicates only once). DO NOT add any "
    "product type (no bottle/clothes/toy/candle/tumbler), use case, personal "
    "context, or inferred property. DO NOT include attribute field names "
    "(no 'Brand:', 'material_type:'). Each value keeps its attribute meaning. "
    "Write ONE short natural sentence (15-30 words) with varied syntax. "
    "Output ONLY the query, no preamble.\n\n"
    "Attributes:\n{ATTRIBUTES}\n\n"
    "Style reference (your past searches, write a query with similar syntax):\n"
    "{EXEMPLARS}"
)


def build_attr_block(attrs: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in attrs.items() if v)


def build_exemplar_block(exemps: list) -> str:
    return "\n".join(f"- {e}" for e in exemps[:N_EXEMPLARS])


USEFUL_ATTR_KEYS_SKIP = {
    "Customer Reviews", "Best Sellers Rank", "Amazon Best Sellers Rank",
    "Date First Available", "Average Rating", "Rating", "ASIN",
    "Is Discontinued By Manufacturer", "Batteries required", "Batteries included",
    "Number Of Items", "Item model number", "Model Name", "Manufacturer",
    "Country/Region of origin", "Specific Uses For Product",
}


def filter_useful_attrs(raw_attrs: dict) -> dict:
    """Skip useless / generic keys and ensure values are non-empty."""
    out = {}
    for k, v in raw_attrs.items():
        if k in USEFUL_ATTR_KEYS_SKIP: continue
        if not v: continue
        s = str(v).strip()
        if s.lower() in ("", "no", "yes", "unknown", "none", "n/a", "0", "1"): continue
        if len(s) <= 1: continue
        out[k] = s
    return out


def main() -> int:
    t0 = time.time()
    records = json.load(open(SCRATCH / "phase35b_records.json"))
    user_sents = json.load(open(SCRATCH / "user_sents_phase35b.json"))
    prod_attrs = json.load(open(REPO_ROOT / "result/product_attributes.json"))
    print(f"[load] {len(records)} records, {len(user_sents)} users")

    # Build per-record top-N attrs
    records_with_attrs = []
    for r in records:
        uid = r["user_id"]
        asin = r["asin"]
        exemps = user_sents.get(uid, [])[:N_EXEMPLARS]
        if not exemps: continue
        full_attrs = filter_useful_attrs(prod_attrs.get(asin, {}))
        # Deterministic ordering: sort by key (alphabetic)
        sorted_keys = sorted(full_attrs.keys())
        sorted_attrs = {k: full_attrs[k] for k in sorted_keys}
        records_with_attrs.append({
            "user_id": uid,
            "asin": asin,
            "exemps": exemps,
            "sorted_keys": sorted_keys,
            "sorted_attrs": sorted_attrs,
        })
    print(f"[records] {len(records_with_attrs)} records with usable attrs")

    # For each N, build the (record, attrs_used) tuple
    todo_per_n: dict[int, list[dict]] = {N: [] for N in N_ATTRS_LIST}
    for r in records_with_attrs:
        sa = r["sorted_attrs"]
        keys = r["sorted_keys"]
        # For N=4, prefer to match Phase 35.B's `attrs_used` if it has 4 attrs and is consistent with full attrs
        # For other N, just take top-N from sorted
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

    # === Generation ===
    print("[main] loading Qwen...")
    sys.path.insert(0, str(REPO_ROOT))
    from llm_client import create_qwen_local_client
    client = create_qwen_local_client(with_vllm=False)
    print(f"[main] Qwen loaded, {time.time() - t0:.1f}s")

    # Per-N candidates
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
            eta = (sum(len(todo_per_n[n]) for n in N_ATTRS_LIST) -
                    sum(done + sum(len(todo_per_n[n]) for n in N_ATTRS_LIST if n < N)
                        for n in N_ATTRS_LIST if n <= N)
                    + done) / max(rate, 1e-6) if N < N_ATTRS_LIST[-1] else 0
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

    # Save per-N candidates
    for N in N_ATTRS_LIST:
        json.dump(cands_per_n[N], open(SCRATCH / f"phase35i_cands_N{N}.json", "w"),
                  indent=2, ensure_ascii=False)
    print(f"\n[save] candidates per N → {SCRATCH}/phase35i_cands_N{{N}}.json")

    # === Build per-N asin pool (each record's own cands) ===
    # For intra-product evaluation, we use the candidate set as is — each (asin, N) has
    # records from various users for that asin, and we want to score them.
    print(f"\n[eval] {time.time() - t0:.0f}s elapsed, starting eval")
    return 0


if __name__ == "__main__":
    sys.exit(main())
