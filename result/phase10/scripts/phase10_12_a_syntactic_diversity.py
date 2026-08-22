#!/usr/bin/env python3
"""Phase 10.12.A: 句法多样性候选生成器.

核心思想: Qwen 自由生成 query 趋同, 候选 pool 句式单一 → 60% win rate 是天花板.

方案: 显式句法控制变量 (6 维), Qwen 自由生成但 prompt 引导使用指定句式.
候选 = (N_conditions × N_seeds_per_condition × N_candidates_per_seed).

句法控制变量 (6 维, 二值或 3 值):
1. QUESTION: 0=陈述句, 1=疑问句
2. CLAUSE: 0=单句, 1=复合句 (with/but/however)
3. MODIFIER: 0=无修饰, 1=带修饰 (really/very/specifically)
4. PASSIVE: 0=主动, 1=被动
5. RELATIVE: 0=无关系从句, 1=with/which/that 从句
6. COMPLEXITY: 0=简单, 1=中等, 2=复杂多从句

每种控制组合产生 K=2 candidates (2 个随机种子).

Pipeline:
- 876 pairs × 12 conditions × 2 seeds × 1 candidate = 21,024 candidates
  (vs Phase 10.10.1 的 876 × 1 × 10 = 8,760)
- 期望: 候选池句法多样性显著提升, pool 内 pairwise L2 clause feature distance 增大
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# All paths/data via pq_env with REPO_ROOT
REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
OUT_CANDIDATES = OUT_DIR / "phase10_12_a_candidates_syntactic.jsonl"
OUT_META = OUT_DIR / "phase10_12_a_meta.json"

PAIRS_FILE = OUT_DIR / "phase10_pairs_1000.jsonl"

# === Hard-coded config (CLAUDE.md Rule 3) ===
N_CONDITIONS = 12  # 12 种典型句法控制组合
N_SEEDS = 2  # 每种条件 2 个独立随机种子
N_CANDIDATES_PER_SEED = 1  # 每个种子 1 个候选
# Total candidates: 876 pairs × 12 × 2 × 1 = 21,024
# Total generations: 21,024
SEED_BASE = 100  # 与 Phase 10.10.1 区分


# === 12 种句法控制组合 (典型多样性) ===
# (QUESTION, CLAUSE, MODIFIER, PASSIVE, RELATIVE, COMPLEXITY)
# 0/1/2 表示对应维度的取值
# === Module-level utilities (also used by Phase 10.14.B+C) ===
def needs_hard_copy(query: str, attrs: dict) -> bool:
    """Check if any required attr is missing from query."""
    q_lower = query.lower()
    for k, v in attrs.items():
        if v and str(v).lower() not in q_lower:
            return True
    return False


def hard_copy(query: str, attrs: dict) -> str:
    """Append missing attrs."""
    q_lower = query.lower()
    missing = []
    for k, v in attrs.items():
        if v and str(v).lower() not in q_lower:
            missing.append(f"{v}")
    if missing:
        if not query.endswith(("?", ".", "!")):
            query = query.rstrip() + "."
        return query + " Looking for " + ", ".join(missing) + "."
    return query


SYNTAX_CONDITIONS = [
    # === 基线 (Phase 10.10.1 风格, 简单陈述) ===
    (0, 0, 0, 0, 0, 0),  # C0: simple declarative
    # === 简单疑问 ===
    (1, 0, 0, 0, 0, 0),  # C1: simple question
    # === 复合句 ===
    (0, 1, 0, 0, 0, 1),  # C2: declarative + clause + medium
    (0, 1, 1, 0, 0, 1),  # C3: declarative + clause + modifier + medium
    # === 带关系从句 ===
    (0, 0, 0, 0, 1, 1),  # C4: declarative + relative clause
    (1, 0, 0, 0, 1, 1),  # C5: question + relative clause
    # === 被动 ===
    (0, 0, 0, 1, 0, 0),  # C6: passive simple
    (0, 0, 0, 1, 0, 1),  # C7: passive medium
    # === 复杂修饰 ===
    (0, 1, 1, 0, 1, 2),  # C8: complex multi-clause
    (1, 1, 1, 0, 1, 2),  # C9: complex question
    # === 极繁 ===
    (1, 1, 1, 1, 1, 2),  # C10: maximal (passive + clauses + ...)
    # === 单 clause 复合 ===
    (0, 1, 0, 0, 1, 1),  # C11: declarative + clause + relative (medium)
]


def syntax_condition_to_text(cond: Tuple[int, int, int, int, int, int]) -> str:
    """Convert syntax control tuple to natural language hint for prompt."""
    question, clause, modifier, passive, relative, complexity = cond
    hints = []

    if question:
        hints.append("Use a question form (ending with ?)")
    else:
        hints.append("Use a declarative statement (no question mark)")

    if clause:
        hints.append("Combine multiple clauses with and/but/while/however")
    else:
        hints.append("Keep it as a single clause")

    if modifier:
        hints.append("Add intensifiers or specific descriptors (really, very, specifically)")
    else:
        hints.append("Avoid extra modifiers")

    if passive:
        hints.append("Use passive voice if natural (e.g. 'made of cotton' instead of 'cotton makes it')")
    else:
        hints.append("Use active voice")

    if relative:
        hints.append("Include a relative clause (with/which/that)")
    else:
        hints.append("No relative clauses")

    if complexity == 2:
        hints.append("Use multiple nested structures (≥2 clauses)")
    elif complexity == 1:
        hints.append("Medium complexity (1-2 clauses)")
    else:
        hints.append("Simple and concise")

    return "STYLE REQUIREMENTS:\n- " + "\n- ".join(hints)


def build_prompt(attrs: Dict[str, str], attrs_5: Dict[str, str], cond: Tuple[int, int, int, int, int, int]) -> str:
    """Build prompt with syntactic control hints."""
    # Pick 3 main attrs (Brand, Color, Material usually)
    main_attrs = []
    for k in ["Brand", "Color", "Material", "Item Weight", "Product Dimensions"]:
        if k in attrs_5 and attrs_5[k]:
            main_attrs.append(f"{k}: {attrs_5[k]}")

    attr_str = "\n".join(main_attrs[:3])  # 3 attrs
    style_str = syntax_condition_to_text(cond)

    prompt = f"""Write ONE short shopping query (1 sentence, ≤25 words) that mentions every attribute below.

Product attributes:
{attr_str}

{style_str}

Output ONLY the query itself on a single line. No prefix, no explanation, no bullet points. Start directly with the query.
"""
    return prompt


def main():
    log = lambda m: print(f"[phase10-12.A] {m}", flush=True)
    log("=" * 70)
    log("Phase 10.12.A: 句法多样性候选生成")
    log("=" * 70)

    # === Load pairs ===
    log(f"[1] Loading pairs from {PAIRS_FILE} ...")
    pairs = []
    with PAIRS_FILE.open() as f:
        for line in f:
            pairs.append(json.loads(line))
    log(f"  pairs: {len(pairs)}")

    # Filter pairs that have all 3 main attrs
    valid_pairs = []
    for p in pairs:
        attrs = p.get("attrs_5", {}) or p.get("attrs", {})
        has_main = all(k in attrs and attrs[k] for k in ["Brand", "Color", "Material"])
        if has_main:
            valid_pairs.append(p)
    log(f"  valid pairs (≥3 attrs): {len(valid_pairs)}")

    # === Load LLM client ===
    log("[2] Loading LLM client (Qwen local) ...")
    from llm_client import create_qwen_local_client
    client = create_qwen_local_client()
    log(f"  client: {type(client).__name__}, model={client.model_name}")

    # === Build all prompts ===
    log("[3] Building prompts ...")
    all_prompts = []  # list of (pair_idx, cond_idx, seed_idx, prompt)
    for pair_idx, p in enumerate(valid_pairs):
        attrs = p.get("attrs_5", {}) or p.get("attrs", {})
        for cond_idx, cond in enumerate(SYNTAX_CONDITIONS):
            for seed_idx in range(N_SEEDS):
                prompt = build_prompt(attrs, attrs, cond)
                all_prompts.append((pair_idx, cond_idx, seed_idx, prompt))
    total = len(all_prompts)
    log(f"  total prompts: {total} (= {len(valid_pairs)} × {N_CONDITIONS} × {N_SEEDS})")

    # === Generate via vllm direct batch (fastest path) ===
    from vllm import SamplingParams

    log(f"[4] Generating {total} candidates via vllm batch ...")
    t0 = time.time()
    candidates = []
    n_failed = 0

    backend = client._backend
    if backend is None:
        raise RuntimeError("vllm backend not initialized (need with_vllm=True)")

    prompts_only = [p[3] for p in all_prompts]
    sampling = SamplingParams(
        max_tokens=120,
        temperature=0.9,
        top_p=0.95,
        stop=["\n\n", "Product attributes:", "<|im_end|>"],
    )

    # vllm continuous batching handles all prompts in one call
    log(f"  calling model.generate({total} prompts) ...")
    outputs = backend.model.generate(prompts_only, sampling)
    log(f"  vllm returned {len(outputs)} outputs in {time.time()-t0:.1f}s")

    for i, out in enumerate(outputs):
        pair_idx, cond_idx, seed_idx, _ = all_prompts[i]
        pair = valid_pairs[pair_idx]
        if not out.outputs:
            result = ""
            n_failed += 1
        else:
            result = out.outputs[0].text.strip()
            if not result:
                n_failed += 1
        candidates.append({
            "user_id": pair["user_id"],
            "asin": pair["asin"],
            "cond_idx": cond_idx,
            "syntax_cond": list(SYNTAX_CONDITIONS[cond_idx]),
            "seed_idx": seed_idx,
            "candidate_query": result,
            "attrs": pair.get("attrs", {}),
            "attrs_5": pair.get("attrs_5") or pair.get("attrs", {}),
            "prompt_used": all_prompts[i][3],
        })

    elapsed = time.time() - t0
    log(f"  generated {len(candidates)} in {elapsed:.1f}s, rate={len(candidates)/elapsed:.1f}/s")
    log(f"  failed: {n_failed}")

    # === Hard-copy attribute post-processing (Phase 10.10.1 logic) ===
    log("[5] Hard-copy attribute post-processing ...")
    def needs_hard_copy(query: str, attrs: dict) -> bool:
        """Check if any required attr is missing from query."""
        q_lower = query.lower()
        for k, v in attrs.items():
            if v and str(v).lower() not in q_lower:
                return True
        return False

    def hard_copy(query: str, attrs: dict) -> str:
        """Append missing attrs."""
        q_lower = query.lower()
        missing = []
        for k, v in attrs.items():
            if v and str(v).lower() not in q_lower:
                missing.append(f"{v}")
        if missing:
            if not query.endswith(("?", ".", "!")):
                query = query.rstrip() + "."
            return query + " Looking for " + ", ".join(missing) + "."
        return query

    n_with_append = 0
    for c in candidates:
        attrs_3 = {k: (c["attrs_5"] or {}).get(k, "") for k in ["Brand", "Color", "Material"]}
        c["attrs_3"] = attrs_3
        if needs_hard_copy(c["candidate_query"], attrs_3):
            c["candidate_query"] = hard_copy(c["candidate_query"], attrs_3)
            c["needed_append"] = True
            n_with_append += 1
        else:
            c["needed_append"] = False

    log(f"  hard-copy appended: {n_with_append}/{len(candidates)}")

    # === Save ===
    log(f"[6] Saving {len(candidates)} candidates to {OUT_CANDIDATES} ...")
    with OUT_CANDIDATES.open("w") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    log(f"  → {OUT_CANDIDATES}")

    meta = {
        "n_pairs": len(valid_pairs),
        "n_conditions": len(SYNTAX_CONDITIONS),
        "n_seeds_per_cond": N_SEEDS,
        "n_candidates": len(candidates),
        "syntax_conditions": [list(c) for c in SYNTAX_CONDITIONS],
        "elapsed_sec": elapsed,
        "n_failed": n_failed,
        "n_with_append": n_with_append,
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    log(f"  → {OUT_META}")

    log("=" * 70)
    log("PHASE 10.12.A COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()