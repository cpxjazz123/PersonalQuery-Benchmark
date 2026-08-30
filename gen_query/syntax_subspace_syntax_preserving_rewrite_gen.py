"""Phase 7.A.2 — Smoke generation 3 conditions (A/B/C).

User proposal (2026-08-31):
- A. Free generation (existing Phase 6.D baseline, 5 attrs only)
- B. Exemplar prompting (show s*, ask LLM to write new query)
- C. Syntax-preserving rewrite (show s*, ask LLM for minimal-edit content swap)

Per Rule 18: smoke first with N=3 users × K=2.
"""
import collections
import json
import os
import sys
import time
from typing import List

# Add project root to sys.path so we can import llm_client (Rule 11)
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")

# === Hardcoded config (Rule 3) ===
PHASE7A1_JSON = "/home/wlia0047/ar57/wenyu/PersoanlQuery/result/gen_query/phase7a_representative_sentences.json"
PHASE6D_PILOT = "/home/wlia0047/hj82_scratch2/wenyu/logs/phase6d_pilot.json"

LOG_PATH = "/home/wlia0047/hj82_scratch2/wenyu/logs/phase7a2_smoke_gen.log"
RESULT_PATH = "/home/wlia0047/ar57/wenyu/PersoanlQuery/result/gen_query/phase7a_smoke_rewrites.json"

# Smoke cohort (Rule 18): N=3 users × K=2 generations × 3 conditions = 18 prompts
SMOKE_USERS = 3
SMOKE_K = 2

# Generation params (match Phase 6.D baseline for Condition A)
TEMP = 0.7
MAX_TOKENS = 120

VLLM_URL = "http://localhost:8800/v1/completions"
MODEL_NAME = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] [phase7a2] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# === Prompt templates ===

# A. Free generation (Phase 6.D baseline, simpler version)
GEN_SYSTEM_TMPL_FREE = """You are a customer writing a search query for a product you want to buy.

Product attributes:
{attrs_block}

Write a single natural-sounding search query that mentions ALL of the above attributes.

Requirements:
- First-person voice (use "I", "my", "we")
- No emoji
- No self-talk (no "can someone help", "thanks!", etc.)
- Length: 8-60 tokens
- Mention EVERY attribute listed above
- Start directly with the product, no "here:", "brand:" prefixes"""

# B. Exemplar prompting (Phase 4 style — show s*, ask LLM to write NEW query in same style)
GEN_SYSTEM_TMPL_EXEMPLAR = """You are a customer writing a search query for a product you want to buy.

Product attributes:
{attrs_block}

Here is a sentence this same customer wrote about a different product:
---
{exemplar}
---

Write a NEW query about the product above. The query should sound like this customer — same personal style, vocabulary, sentence rhythm — but be ABOUT THE NEW PRODUCT, not the old one.

Requirements:
- First-person voice (use "I", "my", "we")
- Mention ALL of the product attributes above
- Length: 8-60 tokens
- No emoji
- No self-talk
- Start directly with the product, no "here:", "brand:" prefixes"""

# C. Syntax-preserving rewrite (minimal-edit content swap)
GEN_SYSTEM_TMPL_REWRITE = """You will REWRITE a single sentence the customer wrote, keeping its structure but swapping the product content.

ORIGINAL SENTENCE (about a different product):
---
{exemplar}
---

PRODUCT ATTRIBUTES TO INSERT (about a new product):
{attrs_block}

CRITICAL RULES — read carefully:
1. KEEP every function word exactly (although, because, while, that, if, when, etc.)
2. KEEP the same clause structure and order (don't reorder)
3. KEEP the same dependency structure (subject/verb/object)
4. KEEP the same sentence length (±5 tokens)
5. ONLY swap content words: product names, brand names, specific objects, colors, materials
6. DO NOT add new clauses
7. DO NOT change verb tenses
8. DO NOT change the subject pronoun (if it was "they", stay "they"; if "it", stay "it")
9. Output ONLY the rewritten sentence — no preamble, no explanation, no quotes

Output the rewritten sentence now."""


def format_attrs_block(attrs):
    """Format attrs as numbered list."""
    return "\n".join(f"- {a}" for a in attrs)


def batch_generate_vllm(prompts: List[str], temp: float = TEMP, max_tokens: int = MAX_TOKENS) -> List[str]:
    """Batched vLLM generation (HTTP /v1/completions).

    Per Rule 4: 必须批量解码,禁止逐条串行生成.
    Per Rule 7: 不允许 fallback,失败直接 raise.
    """
    import requests
    outputs = []
    full_prompts = []
    for p in prompts:
        full_prompts.append(
            f"system\n{p}\n"
            f"user\n\n"
            f"assistant\n"
        )
    bs = 64  # smoke 用小 batch
    n_total = len(prompts)
    n_chunks = (n_total + bs - 1) // bs
    t_start = time.time()
    for ci, i in enumerate(range(0, n_total, bs)):
        chunk = full_prompts[i: i + bs]
        last_err = None
        for attempt in range(3):
            try:
                resp = requests.post(
                    VLLM_URL,
                    json={
                        "model": MODEL_NAME,
                        "prompt": chunk,
                        "temperature": temp,
                        "max_tokens": max_tokens,
                        "top_p": 0.95,
                        "stop": ["\n\n\n", "system\n", "user\n"],
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                results = resp.json()["choices"]
                outputs.extend(r["text"] for r in results)
                last_err = None
                break
            except Exception as e:
                last_err = e
                log(f"    [WARN] batch {ci+1}/{n_chunks} attempt {attempt+1} failed: {e!r}")
                time.sleep(2)
        if last_err is not None:
            raise RuntimeError(f"vLLM batch {ci+1}/{n_chunks} failed after 3 retries: {last_err!r}")
        log(f"  batch {ci+1}/{n_chunks} done ({len(outputs)}/{n_total} outputs, "
            f"{time.time()-t_start:.1f}s)")
    return outputs


def main():
    log("=== Phase 7.A.2 — Smoke generation 3 conditions (A/B/C) ===")
    log(f"  smoke config: N={SMOKE_USERS} users × K={SMOKE_K} cands × 3 conditions "
        f"= {SMOKE_USERS*SMOKE_K*3} prompts")

    # 1. Load representative sentences
    with open(PHASE7A1_JSON) as f:
        repr_data = json.load(f)
    per_user = {r["user_id"]: r for r in repr_data["per_user"] if "representative_sentence" in r}
    log(f"\n  loaded {len(per_user)} users with representative sentences from {PHASE7A1_JSON}")

    # 2. Load pilot attrs (target ASIN attrs)
    with open(PHASE6D_PILOT) as f:
        pilot = json.load(f)
    user_attrs = {r["user_id"]: r["attrs"] for r in pilot["results"]}

    # 3. Build prompts (3 conditions × N × K)
    # Take first SMOKE_USERS for smoke
    pilot_users = list(per_user.keys())[:SMOKE_USERS]
    log(f"\n  smoke users: {pilot_users}")

    prompts_a = []  # Free generation
    prompts_b = []  # Exemplar
    prompts_c = []  # Syntax-preserving rewrite
    meta = []       # (user_id, condition, k_idx)

    for u in pilot_users:
        attrs = user_attrs.get(u, [])
        attrs_block = format_attrs_block(attrs)
        s_star = per_user[u]["representative_sentence"]
        for k in range(SMOKE_K):
            prompts_a.append(GEN_SYSTEM_TMPL_FREE.format(attrs_block=attrs_block))
            prompts_b.append(GEN_SYSTEM_TMPL_EXEMPLAR.format(
                attrs_block=attrs_block, exemplar=s_star))
            prompts_c.append(GEN_SYSTEM_TMPL_REWRITE.format(
                attrs_block=attrs_block, exemplar=s_star))
            meta.append({"user_id": u, "condition": "A_free", "k": k,
                         "attrs": attrs, "exemplar": s_star})
            meta.append({"user_id": u, "condition": "B_exemplar", "k": k,
                         "attrs": attrs, "exemplar": s_star})
            meta.append({"user_id": u, "condition": "C_rewrite", "k": k,
                         "attrs": attrs, "exemplar": s_star})

    log(f"  total prompts: A={len(prompts_a)}, B={len(prompts_b)}, C={len(prompts_c)}")

    # 4. Generate (batch)
    log(f"\n=== Batched generation (TEMP={TEMP}, MAX_TOKENS={MAX_TOKENS}) ===")
    all_prompts = prompts_a + prompts_b + prompts_c
    all_outputs = batch_generate_vllm(all_prompts)

    # 5. Aggregate by condition
    n_a = len(prompts_a)
    n_b = len(prompts_b)
    outputs_a = all_outputs[:n_a]
    outputs_b = all_outputs[n_a:n_a+n_b]
    outputs_c = all_outputs[n_a+n_b:]

    # 6. Save results
    results = {
        "config": {
            "temp": TEMP,
            "max_tokens": MAX_TOKENS,
            "smoke_n_users": SMOKE_USERS,
            "smoke_k_per_user": SMOKE_K,
        },
        "n_smoke_users": SMOKE_USERS,
        "per_user_results": [],
        "all_outputs": {
            "A_free": outputs_a,
            "B_exemplar": outputs_b,
            "C_rewrite": outputs_c,
        },
        "meta": meta,
    }
    for u in pilot_users:
        results["per_user_results"].append({
            "user_id": u,
            "attrs": user_attrs.get(u, []),
            "representative_sentence": per_user[u]["representative_sentence"],
            "d_i_min": per_user[u]["d_i_min"],
        })

    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\n  wrote → {RESULT_PATH}")

    # 7. Quick preview
    log(f"\n=== Sample outputs (1 per condition, first user) ===")
    if outputs_a:
        log(f"  A_free[0]: {outputs_a[0][:200]}")
    if outputs_b:
        log(f"  B_exemplar[0]: {outputs_b[0][:200]}")
    if outputs_c:
        log(f"  C_rewrite[0]: {outputs_c[0][:200]}")

    log(f"\n=== DONE ===")


if __name__ == "__main__":
    main()
