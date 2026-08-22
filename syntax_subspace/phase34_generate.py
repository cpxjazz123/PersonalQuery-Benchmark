#!/usr/bin/env python3
"""Phase 34 Step 1: Generator — strict prompt + 3 user exemplars → K=8 candidates.

设计 (iter_034):
  - 固定 generator, 不变. 4 conditions 只切换 rerank.
  - 复用 STRICT v8 prompt (no hallucination + no field-name leak) + 3 user exemplars
  - K=8 per record, 批量解码 (Rule 4)

输入:
  result/query_records.json (500 records, 5 attrs no-numeric)

输出:
  result/phase34/candidates.json (per record: 8 candidates + meta)

依赖:
  llm_client.py (Rule 8) → QwenLocalClient
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/syntax_subspace")
OUT_DIR = REPO_ROOT / "result/phase34"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT))

# === 硬编码 ===
RECORDS_IN = Path(os.environ.get("P34_RECORDS_IN", str(REPO_ROOT / "result/query_records.json")))
USER_SENTS_JSONL = Path(os.environ.get("P34_USER_SENTS", str(SCRATCH / "user_sents_per_user.json")))
OUT_FILE = Path(os.environ.get("P34_OUT_FILE", str(OUT_DIR / "candidates.json")))

# === 配置 ===
K_SAMPLES = int(os.environ.get("P34_K", "4"))
GEN_BATCH = int(os.environ.get("P34_BATCH", "8"))
GEN_MAX_NEW = int(os.environ.get("P34_MAX_NEW", "128"))
GEN_TEMP = float(os.environ.get("P34_TEMP", "0.7"))
GEN_TOP_P = float(os.environ.get("P34_TOP_P", "0.95"))
GEN_REP_PENALTY = float(os.environ.get("P34_REP_PENALTY", "1.1"))
MAX_RECORDS = int(os.environ.get("P34_MAX_RECORDS", "10"))
N_EXEMPLARS = 3
MAX_INPUT_LENGTH = 480
MASK_CJK = os.environ.get("P34_MASK_CJK", "1") == "1"

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
    """Attributes dict → "Brand: X\nMaterial: Y\n..." block."""
    lines = []
    for k, v in attrs.items():
        s = str(v).strip() if v else ""
        if s:
            lines.append(f"{k}: {s}")
    return "\n".join(lines)


def build_exemplar_block(exemplars: list[str]) -> str:
    """List of sentences → "- sentence\n..." block."""
    return "\n".join(f"- {e.strip()}" for e in exemplars if e.strip())


def make_prompt(attrs: dict, exemplars: list[str], variant_idx: int = 0) -> str:
    """Build full prompt via chat template (System + User content)."""
    user_content = (
        f"Attributes:\n{build_attr_block(attrs)}\n\n"
        f"Style reference (your past searches, similar syntax):\n"
        f"{build_exemplar_block(exemplars)}"
    )
    return user_content


def load_user_exemplars() -> dict[str, list[str]]:
    """Load user_id → [sents] from USER_SENTS_JSONL."""
    if not USER_SENTS_JSONL.exists():
        raise FileNotFoundError(f"USER_SENTS_JSONL not found: {USER_SENTS_JSONL}")
    return json.load(open(USER_SENTS_JSONL))


def main() -> int:
    t0 = time.time()
    user_sents = load_user_exemplars()
    print(f"[load] {len(user_sents)} users, exemplar pool ready")

    records = json.load(open(RECORDS_IN))
    if MAX_RECORDS:
        records = records[:MAX_RECORDS]
    print(f"[load] {len(records)} records to generate, K={K_SAMPLES}")

    # 1) 加载 Qwen
    print("[main] loading Qwen...", flush=True)
    from llm_client import create_qwen_local_client
    client = create_qwen_local_client(with_vllm=False)
    print(f"[main] Qwen loaded, {time.time()-t0:.1f}s")

    # 2) build prompts
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
            todo_prompts.append(make_prompt(attrs, exemps))

    print(f"[main] todo={len(todo_prompts)} ({len(records)} records × K={K_SAMPLES})")

    # 3) batched generate via generate_with_hidden_injection (alpha=0)
    import torch
    out_records: list[dict] = []
    hidden_dim = client._hidden_backend.model.config.hidden_size

    # todo_injections: 全零 (alpha=0 等同无注入)
    zero_bias = torch.zeros(hidden_dim, dtype=torch.float32)
    todo_injections = [zero_bias.clone() for _ in todo_prompts]

    # 加 variant suffix 防同 prompt dedup (在 user_texts 末尾注入 hidden marker)
    todo_user_texts = []
    for p, (_, k_idx) in zip(todo_prompts, todo_records):
        todo_user_texts.append(p + f"\n<!-- variant: {k_idx} -->")

    for i in range(0, len(todo_prompts), GEN_BATCH):
        chunk_p = todo_user_texts[i:i + GEN_BATCH]
        chunk_r = todo_records[i:i + GEN_BATCH]
        chunk_inj = todo_injections[i:i + GEN_BATCH]
        try:
            queries = client.generate_with_hidden_injection(
                system_text=GEN_SYSTEM,
                user_texts=chunk_p,
                injection_per_row=chunk_inj,
                injection_layers=[26],   # 任意一层 (alpha=0 时无影响)
                injection_alpha=0.0,     # 关键: alpha=0 = 无 bias
                max_new_tokens=GEN_MAX_NEW,
                temperature=GEN_TEMP,
                top_p=GEN_TOP_P,
                repetition_penalty=GEN_REP_PENALTY,
                batch_size=GEN_BATCH,
                max_input_length=MAX_INPUT_LENGTH,
                mask_cjk=MASK_CJK,
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            raise

        for (r, k), q in zip(chunk_r, queries):
            out_records.append({
                "user_id": r["user_id"],
                "asin": r["asin"],
                "attrs_used": r.get("attrs_used", {}),
                "k_idx": k,
                "query": q,
            })
        done = min(i + GEN_BATCH, len(todo_prompts))
        rate = done / max(time.time() - t0, 1e-6)
        eta = (len(todo_prompts) - done) / max(rate, 1e-6)
        print(f"[main] {done}/{len(todo_prompts)} ({rate:.2f}/s, ETA {eta:.0f}s)", flush=True)

    # 4) reshape: per-record, K candidates
    per_record: dict[tuple[str, str], dict] = {}
    for row in out_records:
        key = (row["user_id"], row["asin"])
        if key not in per_record:
            per_record[key] = {
                "user_id": row["user_id"],
                "asin": row["asin"],
                "attrs_used": row["attrs_used"],
                "candidates": [],
            }
        per_record[key]["candidates"].append(row["query"])

    out_list = list(per_record.values())
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_list, open(OUT_FILE, "w"), indent=2, ensure_ascii=False)
    print(f"\n[save] {len(out_list)} records × K={K_SAMPLES} → {OUT_FILE}")
    print(f"[main] total {time.time()-t0:.1f}s")
    # print 1 sample
    if out_list:
        s = out_list[0]
        print(f"\n--- sample (asin={s['asin']}) ---")
        for i, c in enumerate(s["candidates"][:3]):
            print(f"  cand{i+1}: {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
