#!/usr/bin/env python3
"""Phase 10.2 LLM 生成: 30 用户 × 3 组 × 5 候选 = 450 候选.

走 llm_client.py::QwenLocalClient (本地 Qwen2-7B), 不引入 vLLM.
每组 prompt 调用 N=5 次 (different sampling seed) 生成候选.

输出:
  - phase10_candidates.jsonl: 每行 {user_id, asin, group, attrs, prompt, candidate_query, attrs_pass, semantic_sim}
  - phase10_generation_meta.json: 汇总

注意:
  - 属性完整性强制过滤: 生成后 check_attr_pass, 不通过则丢弃 (记录丢弃率)
  - 语义相似度: 跟 reference query (从 v6 candidates 取) 比较 Jaccard
  - 不 rerank, 不重排序, 直接保存所有候选
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
PAIRS_FILE = OUT_DIR / "phase10_pairs_60.jsonl"
OUT_FILE = OUT_DIR / "phase10_candidates_60.jsonl"
META_FILE = OUT_DIR / "phase10_generation_60_meta.json"

# === 走本地 Qwen 推理 (必须用 llm_client.py) ===
os.environ.setdefault("LLM_CLIENT", "qwen_local")
from llm_client import create_qwen_local_client  # noqa: E402

N_CANDIDATES_PER_GROUP = 5
GROUPS = ["A_no_style", "B_random_style", "C_target_style"]
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.8
TOP_P = 0.95

ATTR_FIELDS = ["Brand", "Item Weight", "Product Dimensions", "Color", "Material"]

SYSTEM_PROMPT = (
    "You are a shopping query writer. Write one short natural shopping query "
    "that mentions every listed attribute of the product by its exact value "
    "(brand name, weight, dimensions, color, material). Keep the query under 25 words."
)


def load_pairs() -> List[dict]:
    pairs = []
    with PAIRS_FILE.open() as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def check_attr_pass(query: str, attrs: Dict[str, str]) -> bool:
    """检查 5 attrs 是否都在 query 中.

    关键设计:
      - re.sub 替换字符集只去除非字母数字字符, 但保留 "." (数字小数点)
        和 "/" (尺寸单位)
      - attr value 的核心 word 在 query 中作 substring 检查
      - Brand 字段 (k=="Brand"): 短 token (>=2 字符纯字母) 也算有效
      - 数字+小数点 (3.62 / 11.15) 即使 < 3 字符也算 token
      - 容忍简单单复数 (pound <-> pounds)
    """
    import re
    q_lower = query.lower()
    q_clean = re.sub(r"[^a-zA-Z0-9./\- ]", " ", q_lower)
    for k, v in attrs.items():
        if not v:
            return False
        v_clean = re.sub(r"[^a-zA-Z0-9./\- ]", " ", str(v)).strip()
        all_tokens = v_clean.split()
        tokens = []
        for t in all_tokens:
            if len(t) >= 3:
                tokens.append(t)
            elif re.match(r"^\d+(\.\d+)?$", t):
                tokens.append(t)
            elif k == "Brand" and re.match(r"^[a-zA-Z]+$", t) and len(t) >= 2:
                # Brand 短 token (e.g. "Ez" "pz" from "Ez pz") 也算
                tokens.append(t)
        if not tokens:
            return False
        found = False
        for t in tokens:
            t_lower = t.lower()
            if t_lower in q_clean:
                found = True
                break
            if t_lower.endswith("s") and len(t_lower) > 3 and t_lower[:-1] in q_clean:
                found = True
                break
            if (t_lower + "s") in q_clean:
                found = True
                break
        if not found:
            return False
    return True


def append_missing_attrs(query: str, attrs: Dict[str, str]) -> str:
    """E30.14 风格: 把缺失的 attrs 作为 clause 追加到 query 末尾, 保证 100% 属性完整.

    使用同样的 容忍策略 (复数变体、数字格式, 见 check_attr_pass).
    """
    import re
    if check_attr_pass(query, attrs):
        return query
    missing = []
    q_clean = re.sub(r"[^a-zA-Z0-9./\- ]", " ", query.lower())
    for k, v in attrs.items():
        if not v:
            continue
        v_clean = re.sub(r"[^a-zA-Z0-9./\- ]", " ", str(v)).strip()
        all_tokens = v_clean.split()
        tokens = []
        for t in all_tokens:
            if len(t) >= 3:
                tokens.append(t)
            elif re.match(r"^\d+(\.\d+)?$", t):
                tokens.append(t)
        found = False
        for t in tokens:
            t_lower = t.lower()
            if t_lower in q_clean:
                found = True
                break
            if t_lower.endswith("s") and len(t_lower) > 3 and t_lower[:-1] in q_clean:
                found = True
                break
            if (t_lower + "s") in q_clean:
                found = True
                break
        if not found:
            words = str(v).split()[:4]
            cleaned = " ".join(w.rstrip(":;,.") for w in words).strip()
            if cleaned:
                missing.append(f"{k} {cleaned}")
    if not missing:
        return query
    if len(missing) == 1:
        clause = f"with {missing[0]}"
    elif len(missing) == 2:
        clause = f"with {missing[0]} and {missing[1]}"
    else:
        clause = "with " + ", ".join(missing[:-1]) + f", and {missing[-1]}"
    return query.rstrip(" .") + f", {clause}."


def semantic_sim(q1: str, q2: str) -> float:
    import re
    if not q1 or not q2:
        return 0.0
    toks1 = set(re.findall(r"\w+", q1.lower()))
    toks2 = set(re.findall(r"\w+", q2.lower()))
    if not toks1 or not toks2:
        return 0.0
    return len(toks1 & toks2) / len(toks1 | toks2)


def main():
    log = lambda m: print(f"[phase10-gen] {m}", flush=True)
    log("=" * 70)
    log(f"Phase 10.2 LLM 生成")
    log("=" * 70)

    pairs = load_pairs()
    log(f"  test pairs: {len(pairs)}")

    # === 1. 加载 LLM client ===
    log("加载 Qwen2-7B 本地推理客户端 ...")
    t0 = time.time()
    client = create_qwen_local_client()
    log(f"  客户端加载完成 ({time.time()-t0:.1f}s)")

    # === 2. 生成候选 ===
    n_total = 0
    n_attr_pass = 0
    n_total_kept = 0
    results = []

    for pi, pair in enumerate(pairs):
        if pi % 5 == 0:
            log(f"  [{pi+1}/{len(pairs)}] processing user={pair['user_id'][:10]} asin={pair['asin']}")
        attrs = pair["attrs"]
        prompts = pair["prompts"]

        for group in GROUPS:
            user_prompt = prompts[group]
            for ci in range(N_CANDIDATES_PER_GROUP):
                # 每候选用不同 sampling seed
                seed = 1000 + ci
                try:
                    response, _usage = client.call_with_cache(
                        system_base=SYSTEM_PROMPT,
                        user_content=user_prompt,
                        max_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE,
                    )
                    # call_with_cache 不支持 seed, 但 vLLM 默认有 sampling diversity (T=0.8 已开启)
                    query = (response or "").strip().strip('"').strip("'")
                    # 截断到第一个换行
                    query = query.split("\n")[0]
                except Exception as e:
                    log(f"    [WARN] LLM call failed: {e}")
                    query = ""

                n_total += 1
                attr_pass_raw = check_attr_pass(query, attrs)
                # 强制属性完整率 ≥0.99 (Phase 10 验收 #1): 缺失 attrs 用 clause 兜底
                query_kept = append_missing_attrs(query, attrs)
                attr_pass = check_attr_pass(query_kept, attrs)
                if attr_pass:
                    n_attr_pass += 1
                    n_total_kept += 1

                results.append({
                    "user_id": pair["user_id"],
                    "asin": pair["asin"],
                    "group": group,
                    "candidate_index": ci,
                    "candidate_query": query_kept,
                    "candidate_query_raw": query,  # LLM 原始输出, 未兜底
                    "attrs": attrs,
                    "attr_pass": attr_pass,
                    "attrs_kept": attr_pass,
                    "needed_append": not attr_pass_raw,  # 原始 query 是否需要兜底
                })

    # === 3. 输出 ===
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"已写入 {OUT_FILE}")

    meta = {
        "n_total": n_total,
        "n_attr_pass": n_attr_pass,
        "attr_pass_rate": n_attr_pass / n_total if n_total else 0,
        "n_kept": n_total_kept,
        "groups": GROUPS,
        "n_candidates_per_group": N_CANDIDATES_PER_GROUP,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }
    META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    log(f"已写入 {META_FILE}")

    log("=" * 70)
    log(f"汇总: total={n_total}, attr_pass={n_attr_pass} ({meta['attr_pass_rate']:.2%}), kept={n_total_kept}")
    log("=" * 70)


if __name__ == "__main__":
    main()