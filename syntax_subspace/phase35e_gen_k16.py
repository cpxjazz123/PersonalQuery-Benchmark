#!/usr/bin/env python3
"""Phase 35.C: K=8 candidates generation on Phase 15.7 size≥10 subset records.

Reuses Phase 34 generator logic but:
- Reads phase35b_records.json (74 records with attrs)
- Reads user_sents_phase35b.json (74 user exemplars)
- Generates K=8 per record → 592 total

Output: /home/wlia0047/hj82_scratch2/wenyu/syntax_subspace/phase35e_candidates_k16.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/syntax_subspace")
OUT_DIR = REPO_ROOT / "result/phase35c"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = SCRATCH / "phase35e_candidates_k16.json"

K_SAMPLES = int(os.environ.get("P35E_K", "16"))
GEN_BATCH = int(os.environ.get("P35E_BATCH", "16"))
GEN_MAX_NEW = int(os.environ.get("P35E_MAX_NEW", "128"))
GEN_TEMP = float(os.environ.get("P35E_TEMP", "0.7"))
GEN_TOP_P = float(os.environ.get("P35E_TOP_P", "0.95"))
GEN_REP_PENALTY = float(os.environ.get("P35E_REP_PENALTY", "1.1"))
N_EXEMPLARS = 3
MAX_INPUT_LENGTH = 480
MASK_CJK = os.environ.get("P35E_MASK_CJK", "1") == "1"

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


def build_exemplar_block(exemplars: list[str]) -> str:
    return "\n".join(f"- {e.strip()}" for e in exemplars if e.strip())


def main() -> int:
    t0 = time.time()
    records = json.load(open(SCRATCH / "phase35b_records.json"))
    user_sents = json.load(open(SCRATCH / "user_sents_phase35b.json"))
    print(f"[load] {len(records)} records, {len(user_sents)} users, K={K_SAMPLES}")

    print("[main] loading Qwen...")
    sys.path.insert(0, str(REPO_ROOT))
    from llm_client import create_qwen_local_client
    client = create_qwen_local_client(with_vllm=False)
    print(f"[main] Qwen loaded, {time.time() - t0:.1f}s")

    todo_records, todo_prompts = [], []
    for r in records:
        uid = r["user_id"]
        attrs = r.get("attrs_used", {})
        if not attrs:
            continue
        exemps = user_sents.get(uid, [])[:N_EXEMPLARS]
        if not exemps:
            continue
        for k in range(K_SAMPLES):
            todo_records.append((r, k))
            todo_prompts.append(
                f"Attributes:\n{build_attr_block(attrs)}\n\n"
                f"Style reference (your past searches, similar syntax):\n"
                f"{build_exemplar_block(exemps)}"
            )

    print(f"[main] todo={len(todo_prompts)} ({len(records)} records × K={K_SAMPLES})")

    import torch
    hidden_dim = client._hidden_backend.model.config.hidden_size
    zero_bias = torch.zeros(hidden_dim, dtype=torch.float32)
    todo_injections = [zero_bias.clone() for _ in todo_prompts]

    todo_user_texts = []
    for p, (_, k_idx) in zip(todo_prompts, todo_records):
        todo_user_texts.append(p + f"\n<!-- variant: {k_idx} -->")

    out_records = []
    for i in range(0, len(todo_prompts), GEN_BATCH):
        chunk_p = todo_user_texts[i:i + GEN_BATCH]
        chunk_r = todo_records[i:i + GEN_BATCH]
        chunk_inj = todo_injections[i:i + GEN_BATCH]
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
        for (r, k), q in zip(chunk_r, queries):
            out_records.append({
                "user_id": r["user_id"],
                "asin": r["asin"],
                "attrs_used": r["attrs_used"],
                "k_idx": k,
                "query": q,
            })
        done = min(i + GEN_BATCH, len(todo_prompts))
        rate = done / max(time.time() - t0, 1e-6)
        eta = (len(todo_prompts) - done) / max(rate, 1e-6)
        print(f"[main] {done}/{len(todo_prompts)} ({rate:.2f}/s, ETA {eta:.0f}s)", flush=True)

    per_record = {}
    for row in out_records:
        key = (row["user_id"], row["asin"])
        if key not in per_record:
            per_record[key] = {
                "user_id": row["user_id"],
                "asin": row["asin"],
                "attrs_used": row["attrs_used"],
                "intra_product_size": 1,
                "candidates": [],
            }
        per_record[key]["candidates"].append(row["query"])

    # Restore intra_product_size from phase35b records
    for r in records:
        key = (r["user_id"], r["asin"])
        if key in per_record:
            per_record[key]["intra_product_size"] = r.get("intra_product_size", 1)

    out_list = list(per_record.values())
    json.dump(out_list, open(OUT_FILE, "w"), indent=2, ensure_ascii=False)
    print(f"\n[save] {len(out_list)} records × K={K_SAMPLES} → {OUT_FILE}")
    print(f"[main] total {time.time() - t0:.1f}s")
    if out_list:
        s = out_list[0]
        print(f"\n--- sample (asin={s['asin']}) ---")
        for i, c in enumerate(s["candidates"][:3]):
            print(f"  cand{i+1}: {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
