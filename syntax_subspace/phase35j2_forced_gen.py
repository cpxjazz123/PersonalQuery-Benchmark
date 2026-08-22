#!/usr/bin/env python3
"""Phase 35.J-2: Forced generation N=7 + L=36-50.

User requested:
  N=7, L=36-50 是最值得单独复现的 operating point
  强制生成 36-50 词 query (而不是 post-hoc 筛选)

Implementation:
  - For each record, use V2 strict filter, take top-7 attrs
  - System prompt: "Write a natural query of 36-50 words"
  - K=8 candidates per record
  - Then score with softmax_g32_τ0.5

Compare to:
  - V2 N=7 K=4 raw: 50.7% Rank-1
  - V2 N=7 L=36-50 length-matched (n=15): 100% Rank-1
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
OUT_DIR = REPO_ROOT / "result/phase35j2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_FIXED = 7
K_SAMPLES = int(os.environ.get("P35J2_K", "8"))
GEN_BATCH = int(os.environ.get("P35J2_BATCH", "16"))
GEN_MAX_NEW = int(os.environ.get("P35J2_MAX_NEW", "256"))
GEN_TEMP = float(os.environ.get("P35J2_TEMP", "0.7"))
GEN_TOP_P = float(os.environ.get("P35J2_TOP_P", "0.95"))
GEN_REP_PENALTY = float(os.environ.get("P35J2_REP_PENALTY", "1.1"))
N_EXEMPLARS = 3
MAX_INPUT_LENGTH = 480
MASK_CJK = os.environ.get("P35J2_MASK_CJK", "1") == "1"

# === FORCED LENGTH PROMPT ===
GEN_SYSTEM = (
    "You are an Amazon shopper writing a search query. Use EVERY attribute "
    "value below verbatim (mention duplicates only once). DO NOT add any "
    "product type (no bottle/clothes/toy/candle/tumbler), use case, personal "
    "context, or inferred property. DO NOT include attribute field names "
    "(no 'Brand:', 'material_type:'). Each value keeps its attribute meaning. "
    "Write a natural query of approximately 36 to 50 words with varied syntax. "
    "Output ONLY the query, no preamble, no extra context.\n\n"
    "Attributes:\n{ATTRIBUTES}\n\n"
    "Style reference (your past searches, write a query with similar syntax):\n"
    "{EXEMPLARS}"
)


def build_attr_block(attrs: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in attrs.items() if v)


def build_exemplar_block(exemps: list) -> str:
    return "\n".join(f"- {e}" for e in exemps[:N_EXEMPLARS])


# Strict attr filter (same as V2)
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
    "Style",
}
VALUE_BAD_CHARS = {".", ",", ";", ":", "(", ")"}
MAX_VALUE_WORDS = 4


def is_bad_value(v: str) -> bool:
    if not v: return True
    s = str(v).strip()
    if s.lower() in ("", "no", "yes", "unknown", "none", "n/a", "0", "1"): return True
    if len(s) <= 1: return True
    if any(c in s for c in VALUE_BAD_CHARS): return True
    if len(s.split()) > MAX_VALUE_WORDS: return True
    return False


def filter_useful_attrs(raw_attrs: dict) -> dict:
    seen_keys_lower = set()
    out = {}
    for k, v in raw_attrs.items():
        if k in USEFUL_ATTR_KEYS_SKIP: continue
        if not v: continue
        kl = k.lower()
        if kl in seen_keys_lower: continue
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

    # Build per-record top-N=7 attrs (V2 strict filter)
    todo_records = []
    for r in records:
        uid = r["user_id"]
        asin = r["asin"]
        exemps = user_sents.get(uid, [])[:N_EXEMPLARS]
        if not exemps: continue
        full_attrs = filter_useful_attrs(prod_attrs.get(asin, {}))
        sorted_keys = sorted(full_attrs.keys())
        if len(sorted_keys) < N_FIXED: continue
        top_keys = sorted_keys[:N_FIXED]
        attrs_n = {k: full_attrs[k] for k in top_keys}
        for k in range(K_SAMPLES):
            todo_records.append({
                "user_id": uid, "asin": asin,
                "attrs": attrs_n, "exemps": exemps,
                "N": N_FIXED, "k": k,
            })
    print(f"[todo] {len(todo_records)} prompts ({len(set((r['user_id'], r['asin']) for r in todo_records))} records × K={K_SAMPLES})")

    print("[main] loading Qwen...")
    sys.path.insert(0, str(REPO_ROOT))
    from llm_client import create_qwen_local_client
    client = create_qwen_local_client(with_vllm=False)
    print(f"[main] Qwen loaded, {time.time() - t0:.1f}s")

    prompts = []
    for it in todo_records:
        prompt = (
            f"Attributes:\n{build_attr_block(it['attrs'])}\n\n"
            f"Style reference (your past searches, similar syntax):\n"
            f"{build_exemplar_block(it['exemps'])}"
        )
        prompts.append(prompt + f"\n<!-- variant: {it['k']} -->")

    import torch
    hidden_dim = client._hidden_backend.model.config.hidden_size
    zero_bias = torch.zeros(hidden_dim, dtype=torch.float32)
    injections = [zero_bias.clone() for _ in prompts]

    out_qs = []
    for i in range(0, len(prompts), GEN_BATCH):
        chunk_p = prompts[i:i + GEN_BATCH]
        chunk_r = todo_records[i:i + GEN_BATCH]
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
        print(f"[main] {done}/{len(prompts)} ({rate:.2f}/s)", flush=True)

    out_records = []
    for it, q in zip(todo_records, out_qs):
        out_records.append({
            "user_id": it["user_id"],
            "asin": it["asin"],
            "attrs_used": it["attrs"],
            "N": N_FIXED,
            "k": it["k"],
            "query": q,
        })

    json.dump(out_records, open(SCRATCH / "phase35j2_forced_cands.json", "w"),
              indent=2, ensure_ascii=False)
    print(f"[save] → {SCRATCH}/phase35j2_forced_cands.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
