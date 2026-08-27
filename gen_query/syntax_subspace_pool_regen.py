"""Syntax Subspace — Stage 1 (pool regen) + Stage 2 (features).

归 gen_query/: 用 vLLM LLM 为 100 个 ASIN 各生成 K=50 共享候选查询池,
并对生成的查询抽取 spaCy 182d 句法特征(JSONL.gz 缓存)。

用法:
  python gen_query/syntax_subspace_pool_regen.py --stage pool_regen
  python gen_query/syntax_subspace_pool_regen.py --stage features

I/O 路径:
  输入: stage8_5_asins.json
  输出: stage8_5_pool.json          (Stage 1)
        stage7b_query_features.jsonl.gz (Stage 2)

共享工具 (log, feat_key, paths, hyperparams) 来自:
  common/syntax_subspace_utils.py
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import sys
import time
from pathlib import Path
from typing import List

import requests

# Ensure common/ is on sys.path so we can import the shared utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from syntax_subspace_utils import (  # noqa: E402
    ASINS_IN, FEAT_CACHE, POOL_IN, POOL_OUT, REPO_ROOT, SCRATCH, VLLM_URL, MODEL_NAME,
    K_POOL, TEMP, MAX_TOKENS, PCA_DIM, log, feat_key,
)

# Stage 2 also imports per_sentence_features_v2 from common/syntactic_features
sys.path.insert(0, str(REPO_ROOT / "common"))


# ===========================================================================
# Stage 1 helpers (LLM pool generation)
# ===========================================================================

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
    bs = 512   # 用户指令 2026-08-27: 8× 客户端 batch, 减少 HTTP overhead
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


# ===========================================================================
# STAGE 1 — POOL REGEN
# ===========================================================================

def stage_pool_regen():
    log(f"=== STAGE 1 — POOL REGEN: K_POOL={K_POOL} × 100 ASINs ===")

    log(f"loading {ASINS_IN}")
    asin_data = json.load(open(ASINS_IN))["asins"]
    log(f"  {len(asin_data)} ASINs")

    log("building prompts...")
    all_prompts = []
    asin_idx = []
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

    pools = collections.defaultdict(list)
    strict_counts = {}
    n_total_strict = 0

    for (a, k), out in zip(asin_idx, all_outputs):
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
    log(f"  strict: {n_total_strict} ({n_total_strict / max(1, len(all_outputs)) * 100:.1f}%)")
    log(f"  ASINs: {len(pools)}")
    if strict_counts:
        log(f"  strict per ASIN: min={min(strict_counts.values())}, "
            f"max={max(strict_counts.values())}, "
            f"mean={sum(strict_counts.values()) / len(strict_counts):.1f}")
    log(f"  ASINs with ≥10 strict: {sum(1 for v in strict_counts.values() if v >= 10)}")
    log(f"  ASINs with ≥20 strict: {sum(1 for v in strict_counts.values() if v >= 20)}")

    POOL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(POOL_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Stage 8.5: SHARED candidate pool per ASIN (K=50, attrs-only prompt)",
                "K_POOL": K_POOL,
                "TEMP": TEMP,
                "MAX_TOKENS": MAX_TOKENS,
                "SEED": 2024,
                "MODEL_NAME": MODEL_NAME,
            },
            "strict_counts": strict_counts,
            "n_asins": len(pools),
            "n_total_strict": n_total_strict,
            "pools": dict(pools),
        }, f, ensure_ascii=False, indent=2)
    log(f"wrote → {POOL_OUT}")


# ===========================================================================
# STAGE 2 — FEATURES
# ===========================================================================

def stage_features():
    log("=== STAGE 2 — FEATURES ===")

    from syntax_subspace_utils import _syntax_subspace_prepare
    P = _syntax_subspace_prepare()
    fnames = P["feature_names_ordered"]
    log(f"  fnames: {len(fnames)}")

    log(f"loading {POOL_IN}")
    pool_data = json.load(open(POOL_IN))
    pools = pool_data["pools"]
    all_queries = []
    for asin, qs in pools.items():
        for q in qs:
            all_queries.append(q["query"])
    log(f"  total pool queries: {len(all_queries)}")

    feat_map = {}
    if FEAT_CACHE.exists():
        with gzip.open(FEAT_CACHE, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                rec = json.loads(line)
                feat_map[rec["k"]] = rec["v"]
    log(f"  cache keys: {len(feat_map)}")

    missing_q = [q for q in all_queries if feat_key(q) not in feat_map]
    log(f"  missing: {len(missing_q)}")

    if not missing_q:
        log("  no new features needed")
        return

    import spacy
    from syntactic_features import per_sentence_features_v2
    nlp = spacy.load("en_core_web_sm")
    # 用户指令 2026-08-27: 关闭 features 用不到的 spaCy 组件, 提速 30-40%
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)

    log(f"  extracting features for {len(missing_q)} queries via spaCy pipe (n_process=8, batch=512)...")
    new_unique = sorted(set(missing_q))
    new_entries = []
    n_skip = 0
    docs = list(nlp.pipe(new_unique, batch_size=512, n_process=8))
    for i, doc in enumerate(docs):
        q = new_unique[i]
        k = feat_key(q)
        try:
            feats = per_sentence_features_v2(doc)
            feats = feats if feats is not None else {}
        except Exception:
            n_skip += 1
            continue
        numeric = {n: float(v) for n, v in feats.items() if isinstance(v, (int, float))}
        filtered = {n: numeric.get(n, 0.0) for n in fnames}
        feat_map[k] = filtered
        new_entries.append({"k": k, "v": filtered})
        if (i + 1) % 1000 == 0:
            log(f"    {i + 1}/{len(new_unique)}")

    log(f"  extracted: {len(new_entries)}, skipped: {n_skip}")

    with gzip.open(FEAT_CACHE, "wt", encoding="utf-8") as f:
        f.write("# spaCy 182d sentence features (key=sha1(text), v=filtered dict)\n")
        for k, v in feat_map.items():
            f.write(json.dumps({"k": k, "v": v}) + "\n")
    log(f"  saved cache: {len(feat_map)} entries")


# ===========================================================================
# MAIN
# ===========================================================================

STAGE_FUNCTIONS = {
    "pool_regen": stage_pool_regen,
    "features": stage_features,
}


def main():
    parser = argparse.ArgumentParser(description="Syntax Subspace — gen_query (pool regen + features)")
    parser.add_argument(
        "--stage",
        required=True,
        choices=list(STAGE_FUNCTIONS.keys()) + ["all"],
        help="Which stage to run",
    )
    args = parser.parse_args()

    log(f"=== syntax_subspace_pool_regen.py — stage={args.stage} ===")

    if args.stage == "all":
        for stage_name in STAGE_FUNCTIONS:
            log(f"\n>>> Running stage: {stage_name}")
            STAGE_FUNCTIONS[stage_name]()
    else:
        STAGE_FUNCTIONS[args.stage]()


if __name__ == "__main__":
    main()