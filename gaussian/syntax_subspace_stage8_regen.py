"""Stage 8: Generate K=3 query variants per (asin, user, attrs) tuple.

Reads:
  - scratch2/.../stage8_asins.json: 100 ASINs × 10 users × 4 canonical attrs
Writes:
  - scratch2/.../stage8_regen.json

Uses Stage 6B-γ natural-length prompt (no length instruction).
Variant suffix `(variant {k})` to avoid vLLM dedup.

Expected: 100 × 10 × 3 = 3000 generated → ~1500 strict (50% pass rate).

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
        gaussian/syntax_subspace_stage8_regen.py \
        > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_regen.log 2>&1 &
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path
from typing import Dict, List

import requests


# === Paths ===
REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
ASINS_IN = SCRATCH / "stage8_asins.json"
REGEN_OUT = SCRATCH / "stage8_regen.json"

# === Constants ===
VLLM_URL = "http://localhost:8800/v1/completions"
MODEL_NAME = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"
SEED = 2024
K_SAMPLES = 3  # Stage 8 spec
TEMP = 0.7
MAX_TOKENS = 80


def log(msg: str) -> None:
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# === Prompt (Stage 6B-γ natural-length) ===
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
    """Send batched requests to vLLM /v1/completions."""
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
    """Count how many attr values appear in text (verbatim, case-insensitive)."""
    if not text:
        return 0
    text_lower = text.lower()
    covered = 0
    for k, v in attrs.items():
        if v and str(v).lower() in text_lower:
            covered += 1
    return covered


def has_invalid_punct(text: str) -> bool:
    """Check if query has broken structure (preamble, colons, etc.)."""
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
    log("=== Stage 8 REGEN: K=3 variants × 100 ASINs × 10 users ===")
    log(f"K_SAMPLES={K_SAMPLES}")

    log(f"loading {ASINS_IN}")
    asin_data = json.load(open(ASINS_IN))["asins"]
    log(f"  {len(asin_data)} ASINs")

    # Build tasks: (asin, user_id, attrs, n_input)
    tasks = []
    for entry in asin_data:
        a = entry["asin"]
        attrs = entry["attrs_used"]
        n_input = len(attrs)
        for uid in entry["users_sampled"]:
            tasks.append((a, uid, attrs, n_input))

    log(f"  total tasks: {len(tasks)} (× K={K_SAMPLES} = {len(tasks) * K_SAMPLES} queries)")

    # Build all prompts
    log("building prompts...")
    all_prompts = []
    task_indices = []  # (task_idx, k)
    for ti, (asin, uid, attrs, n_input) in enumerate(tasks):
        for k in range(K_SAMPLES):
            prompt = make_prompt(attrs, n_input, k=k)
            all_prompts.append(prompt)
            task_indices.append((ti, k))

    log(f"generating {len(all_prompts)} queries via vLLM...")
    all_outputs = batch_generate_vllm(all_prompts, temp=TEMP, max_tokens=MAX_TOKENS)
    log(f"  got {len(all_outputs)} outputs")

    # Parse and filter strict
    regen = []
    n_invalid = 0
    n_strict = 0
    for (ti, k), out in zip(task_indices, all_outputs):
        asin, uid, attrs, n_input = tasks[ti]
        text = out.strip() if out else ""
        if not text:
            n_invalid += 1
            continue
        n_cov = count_attrs_covered(text, attrs)
        invalid = has_invalid_punct(text)
        is_strict = (n_cov == n_input) and (not invalid)
        if is_strict:
            n_strict += 1
        else:
            n_invalid += 1
        regen.append({
            "asin": asin,
            "user_id": uid,
            "attrs_used": attrs,
            "N_input": n_input,
            "k": k,
            "query": text,
            "attrs_covered": n_cov,
            "invalid": invalid,
            "n_tok": n_tokens_simple(text),
            "strict": is_strict,
        })

    log(f"  total regen entries: {len(regen)}")
    log(f"  strict (attr_pass=1): {n_strict} ({n_strict / len(regen) * 100:.1f}%)")
    log(f"  non-strict: {n_invalid}")

    # Stats: per-asin strict coverage
    asin_strict = collections.defaultdict(set)
    for e in regen:
        if e["strict"]:
            asin_strict[e["asin"]].add(e["user_id"])
    asin_stats = {a: len(u) for a, u in asin_strict.items()}
    log(f"  asins with ≥1 strict user: {len(asin_strict)}")
    for t in [3, 5, 8, 10]:
        n_g = sum(1 for v in asin_stats.values() if v >= t)
        log(f"    ≥{t} users with strict query: {n_g}")

    # Save
    REGEN_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REGEN_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Stage 8 regen: K=3 variants × 100 ASINs × 10 users, attrs from product_attributes.json",
                "K_SAMPLES": K_SAMPLES,
                "TEMP": TEMP,
                "MAX_TOKENS": MAX_TOKENS,
                "SEED": SEED,
            },
            "asin_strict_user_counts": dict(sorted(asin_stats.items(), key=lambda x: -x[1])),
            "entries": regen,
        }, f, ensure_ascii=False, indent=2)
    log(f"wrote → {REGEN_OUT}")


if __name__ == "__main__":
    main()