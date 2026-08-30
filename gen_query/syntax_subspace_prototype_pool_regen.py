#!/usr/bin/env python3
"""Phase 2 — Syntax-Prototype Guided Pool Regen.

用户指令 2026-08-30: K-sweep 已证 F3% 在 K=50/100/200 flat, 瓶颈是 generation
distribution 没覆盖真实用户句法空间. Phase 1 已从 T=34 user 历史句子提取
K_syntax prototypes (centroids + abstract structural pattern) in F3+PCA48
whitened space (同 Stage 4 Mahalanobis 空间).

Phase 2 (本脚本):
  1. 读 syntax_prototypes_K{8,16,32}.json, 选 K_syntax=16 (中庸)
  2. 对每个 ASIN, 把 abstract structural pattern 转成 LLM prompt guidance
     (注意: 不传真实 review 文本, 仅传 abstract 结构描述)
  3. 每个 prototype 生成 K_per_proto=4 query (total K_pool_eff = 16 × 4 = 64)
  4. 应用 5-layer strict filter (与 baseline pool 一致, 保证可比)
  5. 写出 prototype_pool_K{8,16,32}.json (格式与 pool_K*_F3pca48.json 一致)

**输入**:
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/syntax_prototypes_K{8,16,32}.json
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_asins_strict34_intersect2174_ksweep_K{50,100,200}.json
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_selection_ksweep_K200.json (可选: 用于 ASIN list)

**输出**:
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/prototype_pool_K{8,16,32}.json
  - /home/wlia0047/ar57/wenyu/PersoanlQuery/result/gen_query/prototype_pool_summary.json

参数全部硬编码 (Rule 3), 不接受 CLI 参数.
"""
from __future__ import annotations

import collections
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")

# 输入
PROTOTYPES_PATHS = {
    8: SCRATCH / "syntax_prototypes_K8.json",
    16: SCRATCH / "syntax_prototypes_K16.json",
    32: SCRATCH / "syntax_prototypes_K32.json",
}

ASINS_IN = SCRATCH / "stage8_5_asins_strict34_intersect2174_ksweep_K200.json"

PATTRS_PATH = REPO_ROOT / "result/product_attributes.json"
N_INPUT = 5
K_PER_PROTO = 4  # 4 query per prototype per ASIN; 16*4=64 / 8*4=32 / 32*4=128
TEMP = 0.7       # 比 baseline 0.5 高一点, 鼓励多样性
MAX_TOKENS = 120
MAX_QUERY_TOKENS = 60

VLLM_URL = "http://localhost:8800/v1/chat/completions"
MODEL_NAME = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [prototype_gen] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 5-layer strict filter (与 baseline pool_regen 一致, 保证可比)
# ---------------------------------------------------------------------------
ATTR_KEYS = ["brand", "material", "color", "style", "pattern", "theme",
             "type", "subject", "occasion", "size", "shape", "feature"]

_NEG_TOKENS = {"none", "n/a", "na", "null", "unknown", "-", "—", "?", "n/a"}


def normalize_attr_value(v: str) -> str:
    return re.sub(r"\s+", " ", str(v).strip()).lower()


def get_attr_phrases(attrs: dict) -> dict[str, str]:
    """Return {attr_key: phrase} for top attrs."""
    out = {}
    for k in ATTR_KEYS:
        v = attrs.get(k)
        if not v:
            continue
        s = str(v).strip()
        if not s or s.lower() in _NEG_TOKENS:
            continue
        out[k] = s
    return out


def count_attrs_covered(text: str, attrs: dict) -> int:
    """Count how many attrs are explicitly mentioned (any form)."""
    text_l = text.lower()
    n = 0
    for k, v in attrs.items():
        v_l = v.lower()
        # split on '|' or '/' or ';' for multi-value attrs
        parts = re.split(r"[|/;]", v_l)
        if any(p.strip() and p.strip() in text_l for p in parts):
            n += 1
    return n


def has_invalid_punct(text: str) -> bool:
    return bool(re.match(r"^\s*(here|brand|attribute|item|product)[\s:]", text, re.IGNORECASE))


_FIRST_PERSON_RE = re.compile(r"\b(i|my|i've|i'm|i'd)\b", re.IGNORECASE)
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FFFF"
    "☀-➿"
    "⌀-⏿"
    "]+",
    flags=re.UNICODE,
)
_SELF_TALK_PATTERNS = [
    r"\bcan someone\b", r"\bthanks!?\b", r"\blet's try\b", r"\bthose exact attributes\b",
    r"\bappreciate (it|any|the)\b", r"\bhelp me (out|please)\b", r"\bi'?d appreciate\b",
]


def has_first_person(text: str) -> bool:
    return bool(_FIRST_PERSON_RE.search(text))


def has_emoji(text: str) -> bool:
    return bool(_EMOJI_RE.search(text))


def has_self_talk(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in _SELF_TALK_PATTERNS)


def n_tokens_simple(text: str) -> int:
    return len(text.split())


def is_query_too_long(text: str, max_tokens: int = MAX_QUERY_TOKENS) -> bool:
    return n_tokens_simple(text) > max_tokens


# ---------------------------------------------------------------------------
# Prototype → LLM prompt guidance
# ---------------------------------------------------------------------------
def pattern_to_guidance(pattern: dict, cluster_id: int, n_members: int) -> str:
    """Convert abstract structural pattern into LLM-friendly English guidance.
    IMPORTANT: NO verbatim real review text — only abstract counts and instructions.
    """
    nt = pattern.get("n_tok") or {}
    n_clause = pattern.get("n_clause") or {}
    n_advcl = pattern.get("n_advcl") or {}
    n_relcl = pattern.get("n_relcl") or {}
    n_ccomp = pattern.get("n_ccomp") or {}
    n_coord = pattern.get("n_coord") or {}

    median_tok = int(nt.get("median") or 12)
    median_clause = int(n_clause.get("median") or 1)
    has_advcl = (n_advcl.get("median") or 0) >= 1
    has_relcl = (n_relcl.get("median") or 0) >= 1
    has_ccomp = (n_ccomp.get("median") or 0) >= 1
    has_coord = (n_coord.get("median") or 0) >= 1

    lines = [f"Syntactic target (prototype cluster {cluster_id}, n={n_members} user sentences):"]
    lines.append(f"- Sentence length: ~{median_tok} tokens ({nt.get('min', 0)}-{nt.get('max', median_tok)} range).")
    lines.append(f"- Number of clauses: ~{median_clause}.")
    structure = []
    if has_advcl:
        structure.append("adverbial clauses (because/while/when)")
    if has_relcl:
        structure.append("relative clauses (which/that/who)")
    if has_ccomp:
        structure.append("complement clauses (what/that/how)")
    if has_coord:
        structure.append("coordinated phrases (and/but/or)")
    if structure:
        lines.append(f"- Include: {', '.join(structure)}.")
    else:
        lines.append("- Use simple/clausal structure (no nested subordinate clauses).")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt building (similar to make_prompt in syntax_subspace_pool_regen)
# ---------------------------------------------------------------------------
GEN_SYSTEM_TMPL_NATURAL = (
    "You are a real customer searching for a product. "
    "Write a single natural search query as if you are a shopper looking for "
    "this exact product. The query MUST mention ALL of these attributes:\n"
    "{ATTRIBUTES}\n\n"
    "{PROTOTYPE_GUIDANCE}\n\n"
    "Rules:\n"
    "- First-person voice: use 'I', 'my', 'I'm looking for', 'I want', etc.\n"
    "- One natural sentence (NOT a bulleted list).\n"
    "- Do NOT start with 'Here:', 'Brand:', 'Attribute:' or any prefix.\n"
    "- Do NOT use emoji.\n"
    "- Do NOT talk to the AI (no 'can someone help', no 'thanks!').\n"
    "- Maximum 60 tokens.\n"
    "Write only the query, no commentary."
)


def build_user_content(attrs: dict) -> str:
    """Render attribute list as a paragraph."""
    items = []
    for k in ATTR_KEYS:
        v = attrs.get(k)
        if v is None:
            continue
        v = str(v).strip()
        if not v or v.lower() in _NEG_TOKENS:
            continue
        items.append(f"- {k}: {v}")
    return "\n".join(items)


def make_prototype_prompt(attrs: dict, prototype: dict, k_idx: int) -> str:
    n_members = prototype["n_members"]
    cluster_id = prototype["cluster_id"]
    pattern = prototype["abstract_pattern"]
    guidance = pattern_to_guidance(pattern, cluster_id, n_members)
    return GEN_SYSTEM_TMPL_NATURAL.format(
        ATTRIBUTES=build_user_content(attrs),
        PROTOTYPE_GUIDANCE=guidance,
    )


# ---------------------------------------------------------------------------
# vLLM batched generation (reuse pattern from syntax_subspace_pool_regen)
# ---------------------------------------------------------------------------
def batch_generate_vllm(prompts: list[str], temp: float = TEMP,
                        max_tokens: int = MAX_TOKENS) -> list[str]:
    """Batched vLLM generation.

    IMPORTANT: max_workers must stay LOW (≤8) — vLLM 0.27.1 chat API server
    returns HTTP 500 "'str' object has no attribute 'get'" under high
    concurrent load (verified with 32 workers; works fine at 8).
    """
    import requests
    headers = {"Content-Type": "application/json"}
    results: list[str] = []
    BATCH = 16           # keep batch size == worker count
    MAX_WORKERS = 16
    MAX_RETRIES = 3
    RETRY_BACKOFF = 2.0
    n_done = 0
    t0 = time.time()
    for s in range(0, len(prompts), BATCH):
        batch = prompts[s:s + BATCH]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            def _call(p, attempt=0):
                payload = {
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": p}],
                    "temperature": temp,
                    "max_tokens": max_tokens,
                }
                try:
                    r = requests.post(VLLM_URL, json=payload, headers=headers, timeout=60)
                    if r.status_code == 200:
                        d = r.json()
                        return d["choices"][0]["message"]["content"].strip()
                    elif attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_BACKOFF * (attempt + 1))
                        return _call(p, attempt + 1)
                    else:
                        log(f"    WARN vllm status={r.status_code} after {MAX_RETRIES} retries: {r.text[:150]}")
                        return ""
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_BACKOFF * (attempt + 1))
                        return _call(p, attempt + 1)
                    log(f"    WARN vllm exc after {MAX_RETRIES} retries: {e!r}")
                    return ""

            futs = [ex.submit(_call, p) for p in batch]
            for f in futs:
                results.append(f.result())
        n_done += len(batch)
        rate = n_done / (time.time() - t0 + 1e-9)
        eta = (len(prompts) - n_done) / rate if rate > 0 else 0
        log(f"    vllm {n_done}/{len(prompts)} rate={rate:.1f}/s eta={eta:.0f}s")
    return results


def get_top_n_attrs(attrs: dict, n: int) -> dict:
    """Pick top-N real non-numeric attrs (use baseline get_top_n_attrs from
    _n_ablation_pool_v2 for consistency). Returns dict {attr_key: attr_value}.
    """
    sys.path.insert(0, str(REPO_ROOT / "gen_query"))
    from _n_ablation_pool_v2 import get_top_n_attrs as _baseline_get_top_n_attrs
    return _baseline_get_top_n_attrs(attrs, n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_one_K(K_syntax: int) -> dict:
    log(f"\n=== K_syntax={K_syntax} ===")
    # ---- 1. Load prototypes ----
    with open(PROTOTYPES_PATHS[K_syntax], "r", encoding="utf-8") as f:
        proto_data = json.load(f)
    prototypes = proto_data["prototypes"]
    log(f"  loaded {len(prototypes)} prototypes")

    # ---- 2. Load ASINs ----
    with open(ASINS_IN, "r", encoding="utf-8") as f:
        asin_data = json.load(f)["asins"]
    log(f"  loaded {len(asin_data)} ASINs from intersect cohort")

    # ---- 3. Load product attributes ----
    pattrs = json.load(open(PATTRS_PATH))
    log(f"  product_attributes: {len(pattrs)} ASINs")

    # ---- 4. Build per-ASIN attrs + prototype prompts ----
    n_no_attrs = n_lt_n = 0
    asin_attrs: dict[str, dict] = {}
    for entry in asin_data:
        a = entry["asin"]
        pa = pattrs.get(a)
        if pa is None:
            n_no_attrs += 1
            continue
        top_n = get_top_n_attrs(pa, N_INPUT)
        if len(top_n) < N_INPUT:
            n_lt_n += 1
            continue
        asin_attrs[a] = top_n
    log(f"  ASINs with ≥{N_INPUT} non-numeric attrs: {len(asin_attrs)}")

    # ---- 5. Build all prompts (asin × prototype × K_PER_PROTO) ----
    all_prompts = []
    asin_proto_k = []
    for a, attrs_dict in asin_attrs.items():
        for proto in prototypes:
            for k in range(K_PER_PROTO):
                prompt = make_prototype_prompt(attrs_dict, proto, k)
                all_prompts.append(prompt)
                asin_proto_k.append((a, proto["cluster_id"], k))
    log(f"  total prompts: {len(all_prompts)} = {len(asin_attrs)} ASINs × {len(prototypes)} prototypes × {K_PER_PROTO}")

    # ---- 6. vLLM generation ----
    log(f"  generating via vLLM (TEMP={TEMP}, MAX_TOKENS={MAX_TOKENS})...")
    outputs = batch_generate_vllm(all_prompts)

    # ---- 7. Build pools + strict filter ----
    pools: dict[str, list[dict]] = collections.defaultdict(list)
    n_total_strict = 0
    filter_counts = {"cov_full": 0, "invalid": 0, "first_p": 0, "emoji": 0, "self_talk": 0, "too_long": 0}

    for (a, proto_id, k), out in zip(asin_proto_k, outputs):
        attrs_dict = asin_attrs[a]
        text = out.strip() if out else ""
        if not text:
            continue
        n_cov = count_attrs_covered(text, attrs_dict)
        invalid = has_invalid_punct(text)
        first_p = has_first_person(text)
        emoji = has_emoji(text)
        self_talk = has_self_talk(text)
        too_long = is_query_too_long(text)
        strict = (n_cov == N_INPUT) and (not invalid) and first_p and (not emoji) and (not self_talk) and (not too_long)

        if n_cov == N_INPUT:
            filter_counts["cov_full"] += 1
        if invalid:
            filter_counts["invalid"] += 1
        if first_p:
            filter_counts["first_p"] += 1
        if emoji:
            filter_counts["emoji"] += 1
        if self_talk:
            filter_counts["self_talk"] += 1
        if too_long:
            filter_counts["too_long"] += 1

        if strict:
            n_total_strict += 1

        pools[a].append({
            "k": k,
            "query": text,
            "strict": strict,
            "attrs_covered": n_cov,
            "invalid": invalid,
            "first_person": first_p,
            "emoji": emoji,
            "self_talk": self_talk,
            "too_long": too_long,
            "n_tok": n_tokens_simple(text),
            "prototype_id": proto_id,
        })

    log(f"  filter breakdown: {filter_counts}")
    log(f"  total strict: {n_total_strict}")

    out_path = SCRATCH / f"prototype_pool_K{K_syntax}.json"
    out_doc = {
        "config": {
            "description": (
                f"Syntax-Prototype guided pool: K_syntax={K_syntax} prototypes, "
                f"K_per_proto={K_PER_PROTO}, TEMP={TEMP}, MAX_TOKENS={MAX_TOKENS}, "
                f"MAX_QUERY_TOKENS={MAX_QUERY_TOKENS}, N_INPUT={N_INPUT}. "
                f"Each prototype abstracts structural pattern (clause counts / opener / "
                f"length) from T=34 user sentences in F3+PCA48 whitened space. "
                f"NO verbatim review text injected — guidance is abstract only."
            ),
            "K_syntax": K_syntax,
            "K_per_proto": K_PER_PROTO,
            "TEMP": TEMP,
            "MAX_TOKENS": MAX_TOKENS,
            "MAX_QUERY_TOKENS": MAX_QUERY_TOKENS,
            "N_INPUT": N_INPUT,
            "K_POOL_eff": K_syntax * K_PER_PROTO,
        },
        "n_asins": len(asin_attrs),
        "n_total_strict": n_total_strict,
        "filter_counts": filter_counts,
        "pools": pools,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, ensure_ascii=False)
    log(f"  wrote → {out_path}")
    return {
        "K_syntax": K_syntax,
        "n_asins": len(asin_attrs),
        "n_total_strict": n_total_strict,
        "filter_counts": filter_counts,
        "out_path": str(out_path),
    }


def main():
    log("=== Phase 2 — Syntax-Prototype Guided Pool Regen ===")
    log(f"  K_syntax values to run: [16] (K=8 and K=32 also supported but slower)")
    log(f"  K_PER_PROTO: {K_PER_PROTO}, TEMP: {TEMP}")

    # First run K=16 (the middle / primary test)
    result_16 = run_one_K(16)

    summary = {
        "description": (
            "Phase 2 syntax-prototype guided pool regen. K_syntax=16 (center of sweep) "
            "tested. K=8 / K=32 also available but skipped by default to save vLLM time."
        ),
        "results": {"16": result_16},
        "next_step": (
            "Phase 3: Re-run K-sweep on prototype_pool_K16.json via Stage 4 strict "
            "alignment + audit (build_strict34_ksweep_audit.py). Compare F3% / "
            "min_d_self / max_M against baseline K=50/100/200 (86.9% / 11.15 / -1.99)."
        ),
    }
    summary_out = REPO_ROOT / "result/gen_query/prototype_pool_summary.json"
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"\n  wrote → {summary_out}")


if __name__ == "__main__":
    main()
