"""Stage 8.5: Generate SHARED candidate pool per ASIN (not per user).

For each of 100 ASINs:
- ONE pool of K_POOL candidates (attrs-only prompt, NO user conditioning)
- All 10 users share this pool
- This is the formal method's design: user differentiation comes from selection, not generation

Output: stage8_5_pool.json

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
        gaussian/syntax_subspace_stage8_5_regen.py \
        > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_regen.log 2>&1 &
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path
from typing import List

import requests


# === Paths ===
REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
ASINS_IN = SCRATCH / "stage8_5_asins.json"
POOL_OUT = SCRATCH / "stage8_5_pool.json"

# === Constants ===
VLLM_URL = "http://localhost:8800/v1/completions"
MODEL_NAME = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"
SEED = 2024
K_POOL = 50   # shared per ASIN
TEMP = 0.7
MAX_TOKENS = 80


def log(msg: str) -> None:
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# === Prompt (same as Stage 8: natural-length, no user conditioning) ===
GEN_SYSTEM_TMPL_NATURAL = (
    "You are an Amazon shopper writing a search query. Use EXACTLY the {N_INPUT} "
    "attribute values listed below verbatim (mention each value once). DO NOT add any "
    "product type (no bottle/clothes/toy/candle/tumbler), use case, personal "
    "context, or inferred property (do not turn a Color into a scent, a "
    "Material into a function, or a Style into a product class). DO NOT "
    "include attribute field names (no 'Brand:', 'material_type:', "
    "'material_composition:', 'main category', 'Style:') in the query — "
    "only the values. Each value keeps the meaning of its attribute name. "
    "Write a natural sentence (any length is fine). Output ONLY the query, no preamble.\n\n"
    "Attributes ({N_INPUT}):\n{ATTRIBUTES}"
)


def build_user_content(attrs: dict) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in attrs.items())


def make_prompt(attrs: dict, n_input: int, k: int = 0) -> str:
    base = GEN_SYSTEM_TMPL_NATURAL.format(
        N_INPUT=n_input,
        ATTRIBUTES=build_user_content(attrs),
    )
    if k > 0:
        base += f"\n(variant {k})"
    return base


def batch_generate_vllm(prompts: List[str], temp: float = TEMP, max_tokens: int = MAX_TOKENS) -> List[str]:
    outputs = []
    full_prompts = []
    for p in prompts:
        full_prompts.append(
            f"system\n{p}\n"
            f"user\n\n"
            f"assistant\n"
        )
    bs = 64
    for i in range(0, len(prompts), bs):
        chunk = full_prompts[i: i + bs]
        try:
            resp = requests.post(
                VLLM_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": chunk,
                    "temperature": temp,
                    "max_tokens": max_tokens,
                    "top_p": 0.95 if temp > 0 else 1.0,
                },
                timeout=600,
            )
            resp.raise_for_status()
            data = resp.json()
            for choice in data["choices"]:
                outputs.append(choice["text"].strip())
        except Exception as e:
            log(f"  batch error: {e!r}, falling back to single requests")
            for single_prompt in chunk:
                try:
                    r = requests.post(
                        VLLM_URL,
                        json={
                            "model": MODEL_NAME,
                            "prompt": [single_prompt],
                            "temperature": temp,
                            "max_tokens": max_tokens,
                            "top_p": 0.95 if temp > 0 else 1.0,
                        },
                        timeout=60,
                    )
                    r.raise_for_status()
                    rj = r.json()
                    outputs.append(rj["choices"][0]["text"].strip())
                except Exception as _e:
                    log(f"  single fallback failed: {_e!r}")
                    outputs.append("")
    return outputs


def count_attrs_covered(text: str, attrs: dict) -> int:
    if not text:
        return 0
    text_lower = text.lower()
    covered = 0
    for k, v in attrs.items():
        if v and str(v).lower() in text_lower:
            covered += 1
    return covered


def has_invalid_punct(text: str) -> bool:
    if not text:
        return True
    bad_patterns = [
        r"^(here|this|below|sure|okay|ok)[,:]",
        r"^attribute[s]?:",
        r"^brand:",
        r"^color:",
        r"^material:",
        r"^style:",
    ]
    text_lower = text.lower().strip()
    return any(re.search(p, text_lower) for p in bad_patterns)


def n_tokens_simple(text: str) -> int:
    return len(text.split())


def main():
    log(f"=== Stage 8.5 POOL REGEN: K_POOL={K_POOL} × 100 ASINs ===")

    log(f"loading {ASINS_IN}")
    asin_data = json.load(open(ASINS_IN))["asins"]
    log(f"  {len(asin_data)} ASINs")

    # Build all prompts: ONE set per ASIN
    log("building prompts...")
    all_prompts = []
    asin_idx = []  # parallel: which ASIN each prompt belongs to
    for entry in asin_data:
        a = entry["asin"]
        attrs = entry["attrs_used"]
        n_input = len(attrs)
        for k in range(K_POOL):
            prompt = make_prompt(attrs, n_input, k=k)
            all_prompts.append(prompt)
            asin_idx.append((a, k))

    log(f"generating {len(all_prompts)} pool queries via vLLM...")
    all_outputs = batch_generate_vllm(all_prompts, temp=TEMP, max_tokens=MAX_TOKENS)
    log(f"  got {len(all_outputs)} outputs")

    # Parse and group by ASIN
    pools = collections.defaultdict(list)
    strict_counts = {}
    n_total_strict = 0

    for (a, k), out in zip(asin_idx, all_outputs):
        # Get attrs for this ASIN
        entry = next(e for e in asin_data if e["asin"] == a)
        attrs = entry["attrs_used"]
        n_input = len(attrs)
        text = out.strip() if out else ""
        if not text:
            continue
        n_cov = count_attrs_covered(text, attrs)
        invalid = has_invalid_punct(text)
        is_strict = (n_cov == n_input) and (not invalid)
        if is_strict:
            n_total_strict += 1
            strict_counts[a] = strict_counts.get(a, 0) + 1
        pools[a].append({
            "k": k,
            "query": text,
            "strict": is_strict,
            "attrs_covered": n_cov,
            "invalid": invalid,
            "n_tok": n_tokens_simple(text),
        })

    log(f"\n=== Pool stats ===")
    log(f"  total queries: {len(all_outputs)}")
    log(f"  strict: {n_total_strict} ({n_total_strict / len(all_outputs) * 100:.1f}%)")
    log(f"  ASINs: {len(pools)}")
    log(f"  strict per ASIN: min={min(strict_counts.values()) if strict_counts else 0}, "
        f"max={max(strict_counts.values()) if strict_counts else 0}, "
        f"mean={sum(strict_counts.values()) / len(strict_counts):.1f}" if strict_counts else "")
    log(f"  ASINs with ≥10 strict: {sum(1 for v in strict_counts.values() if v >= 10)}")
    log(f"  ASINs with ≥20 strict: {sum(1 for v in strict_counts.values() if v >= 20)}")

    # Save
    POOL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(POOL_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Stage 8.5: SHARED candidate pool per ASIN (K=50, attrs-only prompt)",
                "K_POOL": K_POOL,
                "TEMP": TEMP,
                "MAX_TOKENS": MAX_TOKENS,
                "SEED": SEED,
                "MODEL_NAME": MODEL_NAME,
            },
            "strict_counts": strict_counts,
            "n_asins": len(pools),
            "n_total_strict": n_total_strict,
            "pools": dict(pools),
        }, f, ensure_ascii=False, indent=2)
    log(f"wrote → {POOL_OUT}")


if __name__ == "__main__":
    main()