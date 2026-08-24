"""Stage 8 retry: target the 8 stubborn ASINs with reduced attrs (N=3) + high temp + K=12.

Drop the awkward attr that the generator omits (slashes, parens, list-like).
Generate with temp=1.0 + K=12 to maximize strict hit rate.

Appends to existing stage8_regen.json entries (don't overwrite).

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
        gaussian/syntax_subspace_stage8_regen_retry.py \
        > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_retry.log 2>&1 &
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
REGEN_IN = SCRATCH / "stage8_regen.json"
REGEN_OUT = SCRATCH / "stage8_regen.json"

# === Constants ===
VLLM_URL = "http://localhost:8800/v1/completions"
MODEL_NAME = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"
SEED = 2024
K_SAMPLES = 12  # Aggressive retry for stubborn ASINs
TEMP = 1.0      # Higher diversity
MAX_TOKENS = 80


def log(msg: str) -> None:
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d")
    print(f"[{ts}] {msg}", flush=True)


# === Prompt ===
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


def pick_strict_attrs(attrs: dict, drop_keys: List[str] = None) -> dict:
    """Drop awkward attrs (slashes/parens/numbers) and return N=3."""
    if drop_keys:
        attrs = {k: v for k, v in attrs.items() if k not in drop_keys}
    # Pick top 3 attrs (already preferred order from scan)
    items = list(attrs.items())[:3]
    return dict(items)


def main():
    log("=== Stage 8 REGEN RETRY: 8 stubborn ASINs, N=3 attrs, K=12 ===")

    log(f"loading {ASINS_IN}")
    asin_data = json.load(open(ASINS_IN))["asins"]
    asin_attrs = {a["asin"]: a["attrs_used"] for a in asin_data}

    log(f"loading {REGEN_IN}")
    regen_data = json.load(open(REGEN_IN))
    entries = regen_data["entries"]

    # Find asins with <3 strict users
    asin_strict_users = collections.defaultdict(set)
    for e in entries:
        if e["strict"]:
            asin_strict_users[e["asin"]].add(e["user_id"])

    target_asins = sorted([
        a for a in asin_attrs.keys()
        if len(asin_strict_users.get(a, set())) < 3
    ])
    log(f"target ASINs (with <3 strict users): {len(target_asins)}")
    for a in target_asins:
        n_strict = len(asin_strict_users.get(a, set()))
        attrs = asin_attrs[a]
        log(f"  {a}: n_strict={n_strict}, attrs={attrs}")

    # Build retry tasks with N=3 attrs
    tasks = []
    for entry in asin_data:
        a = entry["asin"]
        if a not in target_asins:
            continue
        # Reduced attrs (N=3)
        attrs_reduced = pick_strict_attrs(entry["attrs_used"])
        n_input = len(attrs_reduced)
        for uid in entry["users_sampled"]:
            tasks.append((a, uid, attrs_reduced, n_input))

    log(f"  total retry tasks: {len(tasks)} (× K={K_SAMPLES} = {len(tasks) * K_SAMPLES} queries)")

    # Build all prompts
    log("building prompts...")
    all_prompts = []
    task_indices = []
    for ti, (asin, uid, attrs, n_input) in enumerate(tasks):
        for k in range(K_SAMPLES):
            prompt = make_prompt(attrs, n_input, k=k)
            all_prompts.append(prompt)
            task_indices.append((ti, k))

    log(f"generating {len(all_prompts)} queries via vLLM...")
    all_outputs = batch_generate_vllm(all_prompts, temp=TEMP, max_tokens=MAX_TOKENS)
    log(f"  got {len(all_outputs)} outputs")

    # Parse and filter strict
    retry_regen = []
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
        retry_regen.append({
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
            "retry": True,
        })

    log(f"  total retry entries: {len(retry_regen)}")
    log(f"  strict: {n_strict} ({n_strict / len(retry_regen) * 100:.1f}%)")

    # Append retry entries to main regen
    entries.extend(retry_regen)

    # Recompute stats
    asin_strict_users = collections.defaultdict(set)
    for e in entries:
        if e["strict"]:
            asin_strict_users[e["asin"]].add(e["user_id"])

    n_total = len(entries)
    n_strict = sum(1 for e in entries if e["strict"])
    log(f"\n=== After retry ===")
    log(f"  total entries: {n_total}")
    log(f"  strict: {n_strict} ({n_strict / n_total * 100:.1f}%)")
    log(f"  asins with ≥1 strict user: {len(asin_strict_users)}")
    for t in [3, 5, 8, 10]:
        n_g = sum(1 for u in asin_strict_users.values() if len(u) >= t)
        log(f"    ≥{t} users with strict query: {n_g}")

    # Save
    regen_data["entries"] = entries
    regen_data["asin_strict_user_counts"] = {
        a: len(u) for a, u in sorted(asin_strict_users.items(), key=lambda x: -len(x[1]))
    }
    with open(REGEN_OUT, "w", encoding="utf-8") as f:
        json.dump(regen_data, f, ensure_ascii=False, indent=2)
    log(f"wrote → {REGEN_OUT}")


if __name__ == "__main__":
    main()