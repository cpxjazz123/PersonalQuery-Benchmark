#!/usr/bin/env python3
"""严格 no-hallucination + K-sample + post-validation 过滤 query 生成。

回应用户 2026-08-22 反馈:
  - 当前 LLM 太"负责", 自动补 product type (water bottle / baby clothes) /
    personal context (for my friend's baby) / 误解 Color (scented) →
    污染 benchmark information need, 无法归因于句法风格
  - 单一 prompt 难以 100% 避免 hallucination, 需要 K-sample + 后过滤

设计 (Semantic Fidelity + No Hallucination + Style Diversity):
  1. STRICT prompt: 显式禁止 model 添加 product type / use case / personal
     context / 任何隐含语义, 让模型只组织"已提供的属性"
  2. K=4 sample per record: 同一 prompt 重复 4 次生成, 利用 do_sample 拿到
     不同句法骨架 (句式不同 / attribute ordering 不同)
  3. POST-VALIDATION filter: 检测每条候选的 "幻觉":
     - product_type 词 (bottle / clothes / toy / candle / tumbler / carrier
       / basket / socks / shoes 等通用商品类名词)
     - personal_context 词 (friend / my / for me / for her / for him /
       sister's / wife's / mom's)
     - implied_use 词 (scented / for sleeping / for bathing / decorative)
     - 必须包含 N 个 attrs_used 中的 attr (semantic fidelity check)
  4. 输出: 每 record 选 1 个 valid candidate (优先覆盖 attrs 最多的)

依赖: result/query_records.json (5 attrs no-numeric) + user_style_vectors_real.jsonl

路径全部硬编码 (Rule 3), 不接受 CLI 参数; 通过 ENV_* 覆盖。
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/user_style_steering")
SCRATCH.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT))

# === 硬编码路径 (Rule 3) ===
RECORDS_IN = REPO_ROOT / "result/query_records.json"
OUT_SUFFIX = os.environ.get("STRICT_OUT_SUFFIX", "strict_v1")
RECORDS_OUT = REPO_ROOT / "result" / f"query_records_with_query_inject_{OUT_SUFFIX}.json"
USER_STYLE_VECTORS_JSONL = SCRATCH / "user_style_vectors_real.jsonl"

# === 实验参数 (env-var 覆盖) ===
K_SAMPLES = int(os.environ.get("STRICT_K", "4"))  # 每个 record 生成 K 个候选
INJECT_LAYERS = [int(x) for x in os.environ.get("STRICT_LAYERS", "16").split(",")]
INJECT_ALPHA = float(os.environ.get("STRICT_ALPHA", "0.0"))
GEN_BATCH = int(os.environ.get("STRICT_BATCH", "4"))
GEN_MAX_NEW = int(os.environ.get("STRICT_MAX_NEW", "128"))
GEN_TEMP = float(os.environ.get("STRICT_TEMP", "0.7"))  # 中等温度
GEN_TOP_P = float(os.environ.get("STRICT_TOP_P", "0.95"))
GEN_REP_PENALTY = float(os.environ.get("STRICT_REP_PENALTY", "1.1"))
MAX_RECORDS = int(os.environ.get("STRICT_MAX_RECORDS", "10"))
MAX_INPUT_LENGTH = 384
MASK_CJK = os.environ.get("STRICT_MASK_CJK", "1") == "1"


# === 短 strict prompt (用户 2026-08-22 详细约束压缩到核心) ===
# 原因: 长 prompt + few-shot 让模型 over-instruction-following (echo / meta reply).
# v1 短 prompt 工作良好, 现在把用户三大约束 (语义保真 / 无幻觉 / 句法多样) 压到 3 句.
# v8 加: 禁止字段名 (Brand:, material_type:, etc.) 出现在 query 里.
GEN_SYSTEM = (
    "You are an Amazon shopper writing a search query. Use EVERY attribute "
    "value below verbatim (mention duplicates only once). DO NOT add any "
    "product type (no bottle/clothes/toy/candle/tumbler), use case, personal "
    "context, or inferred property (do not turn a Color into a scent, a "
    "Material into a function, or a Style into a product class). DO NOT "
    "include attribute field names (no 'Brand:', 'material_type:', "
    "'material_composition:', 'main category', 'Style:') in the query — "
    "only the values. Each value keeps the meaning of its attribute name. "
    "Write one natural sentence (20-35 words) with varied syntax — clauses, "
    "coordination, prepositional phrases, relative clauses. Output ONLY the "
    "query, no preamble.\n\n"
    "Attributes:\n{ATTRIBUTES}"
)


def build_user_content(attrs: dict) -> str:
    """把 attrs 拼成 few-shot 一致的格式, 让 {ATTRIBUTES} 占位符替换。

    例:
      Brand: Ajulkrio
      Main Category: Baby
      Material Type: Acrylonitrile Butadiene Styrene
      ...
    """
    lines = []
    for k, v in attrs.items():
        s = str(v).strip() if v else ""
        if s:
            lines.append(f"{k}: {s}")
    return "\n".join(lines)


def make_prompt(attrs: dict, variant_idx: int = 0) -> str:
    """把 GEN_SYSTEM 的 {ATTRIBUTES} 替换为实际 attrs (并加 variant suffix 避免 dedup)。

    注意: prompt 不带 "Query:" 后缀, 否则模型可能 echo prompt 内容而不是生成新 query。
    variant suffix 用 hidden 标记, 不出现在 user-visible prompt 文本里。
    """
    attr_block = build_user_content(attrs)
    prompt = GEN_SYSTEM.replace("{ATTRIBUTES}", attr_block)
    return prompt


# === Post-validation: 检测幻觉 ===
# 这些通用商品类名词 + 用途词 + 人物关系词, 即便不在 attrs 里也常被 LLM 补上
HALLUCINATION_WORDS = [
    # 通用产品类型 (品牌/类目推断)
    "bottle", "bottles", "clothing", "clothes", "toy", "toys", "candle",
    "candles", "tumbler", "tumblers", "carrier", "carriers", "basket",
    "baskets", "socks", "shoes", "shirt", "shirts", "pants", "dress",
    "dresses", "blanket", "blankets", "pillow", "pillows", "diaper",
    "diapers", "wipes", "formula", "pacifier", "stroller", "highchair",
    "swaddle", "swaddles", "romper", "rompers", "onesie", "onesies",
    "mittens", "booties", "teether", "teethers", "rattle", "rattles",
    "mobile", "mobiles", "nightlight", "soap", "lotion", "shampoo",
    "brush", "comb", "drinkware", "beverage", "container", "holder",
    "vessel", "equipment", "gear", "appliance", "utensil", "dish",
    "sneaker", "sneakers", "sandal", "sandals", "boot", "boots",
    "headband", "headbands", "hairband", "hairbands", "cap", "hat",
    "jacket", "coats", "vest", "shorts", "skirt", "legging", "leggings",
    # 用途推断词
    "scented", "scent", "fragrance", "aroma", "flavor", "flavour",
    "decorative", "decoration", "ornament", "ornamental",
    # 人物关系 / 拥有者
    "friend", "friends", "sister", "sister's", "brother", "brother's",
    "wife", "wife's", "husband", "husband's", "mom", "mom's", "dad",
    "dad's", "mother", "father", "daughter", "son", "grandma",
    "grandpa", "aunt", "uncle", "cousin", "neighbor",
    # 推测产品描述
    "lightweight", "heavyweight", "premium", "luxury", "cheap",
    "professional-grade", "commercial", "industrial",
    "handmade", "handcrafted", "artisan",
    # accessory / tool / decor (推断词)
    "accessory", "accessories", "tool", "tools", "decor", "gadget",
    "gizmo", "implement", "device", "apparatus",
]


def count_attrs_covered(text: str, attrs: dict) -> int:
    """attrs 中出现在 query 里的字段数 (case-insensitive, exact substring)。"""
    if not attrs:
        return 0
    text_lower = text.lower()
    n = 0
    for v in attrs.values():
        s = str(v).strip() if v else ""
        if s and s.lower() in text_lower:
            n += 1
    return n


def has_hallucination(text: str, attrs: dict) -> list[str]:
    """返回 query 里出现的所有 hallucination 词 (不在 attrs 的内容词)。

    排除 attrs 中本来就含的词 (例如 "Style: bottle" → 'bottle' 不算幻觉)。
    """
    text_lower = text.lower()
    # 收集 attrs 里所有合法的 substring
    attr_substrings: set[str] = set()
    for v in attrs.values():
        s = str(v).strip() if v else ""
        if s:
            attr_substrings.add(s.lower())
            # 也拆词加入, 避免 attr 词被误判
            for tok in re.split(r"[\s,/]+", s.lower()):
                if len(tok) >= 4:
                    attr_substrings.add(tok)
    hits = []
    for w in HALLUCINATION_WORDS:
        # 用 word boundary 匹配 (避免 'a' 匹配到 'all')
        if re.search(rf"\b{re.escape(w)}\b", text_lower):
            if w not in attr_substrings:
                hits.append(w)
    return hits


def has_invalid_punct(text: str) -> bool:
    """检测引号 / 括号 leak (model 自言自语)。"""
    return bool(re.search(r'["\(\)“”]', text))


def score_candidate(text: str, attrs: dict) -> tuple[int, int, int]:
    """返回 (n_covered, -n_hallu, -has_invalid_punct) 用于 ranking。

    优先级: 覆盖 attr 数 > 幻觉数越少 > 无标点 leak。
    """
    n_cov = count_attrs_covered(text, attrs)
    n_hallu = len(has_hallucination(text, attrs))
    invalid = 1 if has_invalid_punct(text) else 0
    return (n_cov, -n_hallu, -invalid)


def main() -> int:
    import torch
    t0 = time.time()

    # === 1) 加载 Qwen ===
    print(f"[main] loading Qwen (transformers path)...", flush=True)
    from llm_client import create_qwen_local_client
    client = create_qwen_local_client(with_vllm=False)
    hidden_dim = client._hidden_backend.model.config.hidden_size
    print(f"[main] hidden_dim={hidden_dim}, K={K_SAMPLES}, layers={INJECT_LAYERS}, alpha={INJECT_ALPHA}")

    # === 2) 加载 user style vector (B 路线, 直接 mean) ===
    if not USER_STYLE_VECTORS_JSONL.exists():
        raise FileNotFoundError(
            f"{USER_STYLE_VECTORS_JSONL} 不存在, 先跑 gen_query/soft_prefix/build_user_style_vector_real.py"
        )
    profiles: dict[str, dict[int, list[float]]] = {}
    with open(USER_STYLE_VECTORS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            uid = row["user_id"]
            profiles[uid] = {}
            for L in INJECT_LAYERS:
                key = f"residual_mean_layer_{L}"
                if key in row:
                    profiles[uid][L] = row[key]
    print(f"[main] loaded {len(profiles)} user profiles (B-direct)")

    # === 3) 加载 records ===
    records = json.load(open(RECORDS_IN, "r", encoding="utf-8"))
    if MAX_RECORDS and len(records) > MAX_RECORDS:
        records = records[:MAX_RECORDS]
    print(f"[main] {len(records)} records to generate")

    # === 4) 增量跳过 ===
    existing: dict[tuple[str, str], str] = {}
    if RECORDS_OUT.exists():
        prev = json.load(open(RECORDS_OUT, "r", encoding="utf-8"))
        for r in prev:
            if r.get("y_plus_query_inject") and "user_id" in r and "asin" in r:
                existing[(r["user_id"], r["asin"])] = r["y_plus_query_inject"]
        print(f"[main] {len(existing)} already-generated (增量跳过)")

    # === 5) 准备 todo ===
    todo_records: list[dict] = []
    todo_prompts: list[str] = []
    todo_injections: list = []
    todo_sample_idx: list[int] = []  # 每条 (record, sample_idx)
    n_no_profile = 0
    for r in records:
        uid, asin = r["user_id"], r["asin"]
        if (uid, asin) in existing:
            continue
        attrs = r.get("attrs_used", {})
        if not attrs:
            continue
        # bias 拿 layer[0]
        prof = profiles.get(uid)
        if prof is None or INJECT_LAYERS[0] not in prof:
            n_no_profile += 1
            bias = torch.zeros(hidden_dim, dtype=torch.float32)
        else:
            bias = torch.as_tensor(prof[INJECT_LAYERS[0]], dtype=torch.float32)
        # K-sample: 同 record 重复 K 次, prompt 加 variant suffix 让模型不再 dedup
        for k in range(K_SAMPLES):
            todo_records.append(r)
            todo_prompts.append(make_prompt(attrs, variant_idx=k + 1))
            todo_injections.append(bias)
            todo_sample_idx.append(k)
    print(
        f"[main] todo={len(todo_records)} (= {len(records) - len(existing)} records × K={K_SAMPLES}), "
        f"no_profile={n_no_profile}",
        flush=True,
    )

    if not todo_records:
        print(f"[main] ✓ 无 todo, 退出")
        return 0

    # === 6) 写出初始结果 (含已有) ===
    out_records: list[dict] = []
    for r in records:
        if (r["user_id"], r["asin"]) in existing:
            r2 = dict(r)
            r2["y_plus_query_inject"] = existing[(r["user_id"], r["asin"])]
            out_records.append(r2)
        else:
            out_records.append(dict(r))

    # === 7) 分批生成 ===
    # 把同一 record 的 K 个 candidate 收在一起, 排序选最优
    candidates_by_key: dict[tuple[str, str], list[str]] = {}
    new_records_by_key: dict[tuple[str, str], dict] = {}
    for i in range(0, len(todo_prompts), GEN_BATCH):
        chunk_prompts = todo_prompts[i:i + GEN_BATCH]
        chunk_records = todo_records[i:i + GEN_BATCH]
        chunk_inj = todo_injections[i:i + GEN_BATCH]
        try:
            queries = client.generate_with_hidden_injection(
                system_text=GEN_SYSTEM,
                user_texts=chunk_prompts,
                injection_per_row=chunk_inj,
                injection_layers=INJECT_LAYERS,
                injection_alpha=INJECT_ALPHA,
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
            print(f"[main] ✗ batch failed at chunk {i}: {exc!r}")
            traceback.print_exc()
            raise
        for r, q in zip(chunk_records, queries):
            key = (r["user_id"], r["asin"])
            candidates_by_key.setdefault(key, [])
            candidates_by_key[key].append(q)
            new_records_by_key[key] = r
        done = min(i + GEN_BATCH, len(todo_prompts))
        elapsed = time.time() - t0
        rate = done / max(elapsed, 1e-6)
        eta = (len(todo_prompts) - done) / max(rate, 1e-6)
        print(
            f"[main] {done}/{len(todo_prompts)} ({rate:.2f}/s, ETA {eta:.0f}s)",
            flush=True,
        )

    # === 8) 每个 record 从 K 个候选中选 best (覆盖最多 + 幻觉最少) ===
    n_with_inject = 0
    n_all_hallu = 0
    for key, cands in candidates_by_key.items():
        r = new_records_by_key[key]
        attrs = r.get("attrs_used", {})
        scored = [(score_candidate(c, attrs), c) for c in cands]
        # 按 (n_cov, -n_hallu, -invalid_punct) 降序
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_q = scored[0]
        n_cov, neg_hallu, neg_inv = best_score
        n_hallu = -neg_hallu
        n_cov_real, _, _ = (n_cov, n_hallu, -neg_inv)
        r["y_plus_query_inject"] = best_q
        r["n_candidates"] = len(cands)
        r["n_attrs_covered"] = n_cov_real
        r["n_hallucinations"] = n_hallu
        r["hallucination_words"] = has_hallucination(best_q, attrs)
        r["all_candidates"] = cands
        r["all_candidates_scores"] = [list(s) for s, _ in scored]
        n_with_inject += 1
        if n_hallu > 0:
            n_all_hallu += 1

    # === 9) fill back ===
    for key, r in new_records_by_key.items():
        for orec in out_records:
            if orec["user_id"] == r["user_id"] and orec["asin"] == r["asin"]:
                orec.update({
                    "y_plus_query_inject": r.get("y_plus_query_inject"),
                    "n_candidates": r.get("n_candidates"),
                    "n_attrs_covered": r.get("n_attrs_covered"),
                    "n_hallucinations": r.get("n_hallucinations"),
                    "hallucination_words": r.get("hallucination_words"),
                    "all_candidates": r.get("all_candidates"),
                    "all_candidates_scores": r.get("all_candidates_scores"),
                })
                break

    # === 10) 写出 ===
    RECORDS_OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_records, open(RECORDS_OUT, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"[main] ✓ wrote {RECORDS_OUT}: {n_with_inject} records")
    print(f"[main]   records with hallucinations (best of K): {n_all_hallu}/{n_with_inject}")
    print(f"[main] total {time.time()-t0:.1f}s")

    # === 11) per-record detail ===
    for key in sorted(candidates_by_key.keys(), key=lambda k: str(k)):
        r = new_records_by_key[key]
        attrs = r.get("attrs_used", {})
        best_q = r["y_plus_query_inject"]
        cands = r["all_candidates"]
        print(f"\n--- asin={r['asin']} (user={r['user_id'][:14]}) ---")
        print(f"  attrs ({len(attrs)}): {dict(list(attrs.items()))}")
        print(f"  BEST  (cov={r['n_attrs_covered']}/5 hallu={r['n_hallucinations']}): {best_q}")
        if r["hallucination_words"]:
            print(f"         hallucination: {r['hallucination_words']}")
        for i, c in enumerate(cands):
            cov = count_attrs_covered(c, attrs)
            hallu = has_hallucination(c, attrs)
            mark = " ★" if c == best_q else "  "
            print(f"  {mark} cand{i+1} (cov={cov}/5 hallu={len(hallu)}): {c}")
            if hallu:
                print(f"            hallucination: {hallu}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
