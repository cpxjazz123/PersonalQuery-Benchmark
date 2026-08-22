#!/usr/bin/env python3
"""Generate y_plus_query training targets for copy-aware copy_aware_train.py.

Inputs:  result/personal_query/attribute_extraction/Baby_Products/query_records.json
         (含 user_id / asin / attrs_used, 来自 build_query_records.py)

Outputs: result/personal_query/attribute_extraction/Baby_Products/query_records_with_query.json
         (注入 y_plus_query: 用 Qwen-7B batched vLLM 生成自然 shopping query,
         严格按 build_attr_prompt_lines 列出每个属性)

Pipeline: 走 llm_client.py (Rule 8), vLLM batched (Rule 4/5h), prompt 与
copy_aware_train.py::build_messages 一致 (training 时同一 system prompt 让
inference 行为对齐)。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH_LOG = Path("/home/wlia0047/hj82_scratch2/wenyu/copy_aware")
SCRATCH_LOG.mkdir(parents=True, exist_ok=True)

# 必须先 sys.path.insert 再 import (Rule 12 允许运行目录是项目内, import 用绝对)
sys.path.insert(0, str(REPO_ROOT))

# === 硬编码路径 (Rule 3) ===
RECORDS_IN = REPO_ROOT / "result/personal_query/attribute_extraction/Baby_Products/query_records.json"
RECORDS_OUT = REPO_ROOT / "result/personal_query/attribute_extraction/Baby_Products/query_records_with_query.json"
GENERATION_LOG = SCRATCH_LOG / "generate_targets.log"

# === 硬编码推理配置 ===
GEN_BATCH = int(os.environ.get("CA_GEN_BATCH", "32"))
GEN_MAX_NEW = 96
GEN_TEMP = 0.5
GEN_TOP_P = 0.95
GEN_FREQ_PENALTY = 0.3  # 减少 "I'm looking for..." 这种模板套话
GEN_PRES_PENALTY = 0.1
MAX_RECORDS = int(os.environ.get("CA_GEN_MAX_RECORDS", "500"))

# Amazon-style 自然 search query 模板训练目标: 短 (< 80 字符)、搜索意图驱动、
# 不啰嗦、不用 "I'm looking for / Can you recommend" 等模板。
GEN_SYSTEM = (
    "You write realistic Amazon-style search queries. Output ONLY the query "
    "text — no preamble, no quotes, no sentence like 'I'm looking for'. "
    "Match the style of an Amazon search bar input: short, intent-driven, "
    "keywords joined by commas or natural phrasing. Mention only the most "
    "important 2-4 attributes; not every field. Never start with 'I'."
)


def build_user_content(attrs: dict) -> str:
    """与 copy_aware.py::build_attr_prompt_lines 一致的 prompt 拼接。"""
    lines = ["Product attributes:"]
    for k, v in attrs.items():
        s = str(v).strip() if v else ""
        if s:
            lines.append(s)
    lines.append("Write a short Amazon-style search query (no preamble).")
    return "\n".join(lines)


def batch_generate_queries(prompts: list[str]) -> list[str]:
    """走 llm_client.py + 直接调 vLLM backend.model.generate (Rule 4/5h batched)."""
    from vllm import SamplingParams
    from llm_client import create_qwen_local_client

    client = create_qwen_local_client()
    backend = client._backend
    assert backend is not None, "vLLM backend 未初始化"

    sampling = SamplingParams(
        max_tokens=GEN_MAX_NEW,
        temperature=GEN_TEMP,
        top_p=GEN_TOP_P,
        frequency_penalty=GEN_FREQ_PENALTY,
        presence_penalty=GEN_PRES_PENALTY,
    )
    full_prompts = [
        backend.tokenizer.apply_chat_template(
            [{"role": "system", "content": GEN_SYSTEM},
             {"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True,
        )
        for p in prompts
    ]
    outputs = backend.model.generate(full_prompts, sampling)
    texts: list[str] = []
    for o in outputs:
        if o.outputs:
            t = o.outputs[0].text.strip()
            # 截断到第一个换行 (防御性, 模型偶尔会续写额外内容)
            t = t.split("\n")[0].strip()
            texts.append(t)
        else:
            texts.append("")
    return texts


def main() -> int:
    t0 = time.time()
    print(f"[gen] loading records: {RECORDS_IN}", flush=True)
    records = json.load(open(RECORDS_IN, "r", encoding="utf-8"))
    if MAX_RECORDS and len(records) > MAX_RECORDS:
        records = records[:MAX_RECORDS]
    print(f"[gen] {len(records)} records to generate query for")

    # 增量加载: 已生成的跳过 (按 user_id+asin 索引)
    existing: dict[tuple[str, str], str] = {}
    if RECORDS_OUT.exists():
        prev = json.load(open(RECORDS_OUT, "r", encoding="utf-8"))
        for r in prev:
            if r.get("y_plus_query") and "user_id" in r and "asin" in r:
                existing[(r["user_id"], r["asin"])] = r["y_plus_query"]
        print(f"[gen] {len(existing)} already-generated targets (增量跳过)")

    # 构造 prompts
    todo = []
    todo_idx = []  # 对应 records 中的下标
    prompts = []
    for i, r in enumerate(records):
        uid, asin = r["user_id"], r["asin"]
        if (uid, asin) in existing:
            continue
        attrs = r.get("attrs_used", {})
        if not attrs:
            continue
        todo.append(r)
        todo_idx.append(i)
        prompts.append(build_user_content(attrs))
    print(f"[gen] todo={len(todo)}, already={len(existing)}")

    # 写出初始结果 (已有 query 的直接 fill)
    out_records = []
    for i, r in enumerate(records):
        if (r["user_id"], r["asin"]) in existing:
            r2 = dict(r)
            r2["y_plus_query"] = existing[(r["user_id"], r["asin"])]
            out_records.append(r2)
        else:
            out_records.append(dict(r))
    # 找到 todo 顺序对应的 out_records index
    todo_out_idx = [out_records.index(r) for r in todo]  # 简化版

    # 分批 vLLM 生成
    new_rows: list[dict] = []
    for i in range(0, len(prompts), GEN_BATCH):
        chunk = prompts[i:i + GEN_BATCH]
        chunk_records = todo[i:i + GEN_BATCH]
        try:
            queries = batch_generate_queries(chunk)
        except Exception as exc:
            import traceback
            print(f"[gen] ✗ batch failed at chunk {i}: {exc!r}")
            traceback.print_exc()
            raise
        for r, q in zip(chunk_records, queries):
            r["y_plus_query"] = q
            new_rows.append(r)
        done = min(i + GEN_BATCH, len(prompts))
        elapsed = time.time() - t0
        rate = done / max(elapsed, 1e-6)
        eta = (len(prompts) - done) / max(rate, 1e-6)
        print(f"[gen]  {done}/{len(prompts)} ({rate:.1f}/s, ETA {eta:.0f}s)", flush=True)

    # 把新生成的 fill 回 out_records
    for r in new_rows:
        for orec in out_records:
            if orec["user_id"] == r["user_id"] and orec["asin"] == r["asin"]:
                orec["y_plus_query"] = r["y_plus_query"]
                break

    # 写出
    RECORDS_OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_records, open(RECORDS_OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    n_with_query = sum(1 for r in out_records if r.get("y_plus_query"))
    print(f"[gen] ✓ wrote {RECORDS_OUT}: {n_with_query}/{len(out_records)} with y_plus_query")
    print(f"[gen] total {time.time()-t0:.1f}s")
    # 打印前 3 条样本
    for r in out_records[:3]:
        print(f"\n--- asin={r['asin']} ---")
        print(f"  attrs: {r['attrs_used']}")
        print(f"  y_plus_query: {r.get('y_plus_query', '')[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())