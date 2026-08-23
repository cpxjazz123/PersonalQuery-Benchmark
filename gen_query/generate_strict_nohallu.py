#!/usr/bin/env python3
"""Query 生成主脚本：strict no-hallucination + K-sample + post-validation。

Pipeline:
  1. 加载 result/query_records_10k.json (5 attrs per user,无数值)
  2. 加载 user_style_vectors_real.jsonl (per-user layer 16/20/24/26 mean residual)
     如不存在，用 zero vector 自动构建 (ALPHA=0 模式)
  3. 每条 record 生成 K=4 个候选 (hidden injection + style steering)
  4. Post-validation: 过滤 hallucination 词，选覆盖 attrs 最多 + 幻觉最少的
  5. 输出 result/query_records_with_query_inject_strict_10k.json

路径全部硬编码，不接受 CLI 参数; 通过 ENV_* 覆盖。

依赖:
  result/query_records_10k.json
  user_style_vectors_real.jsonl (auto-build if missing and ALPHA=0)
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

# === 硬编码路径 ===
RECORDS_IN = REPO_ROOT / "result/query_records_10k.json"
OUT_SUFFIX = os.environ.get("STRICT_OUT_SUFFIX", "strict_10k")
RECORDS_OUT = REPO_ROOT / "result" / f"query_records_with_query_inject_{OUT_SUFFIX}.json"
USER_STYLE_VECTORS_JSONL = SCRATCH / "user_style_vectors_real.jsonl"

# === 实验参数 (env-var 覆盖) ===
K_SAMPLES = int(os.environ.get("STRICT_K", "4"))
INJECT_LAYERS = [int(x) for x in os.environ.get("STRICT_LAYERS", "16").split(",")]
INJECT_ALPHA = float(os.environ.get("STRICT_ALPHA", "0.0"))
GEN_BATCH = int(os.environ.get("STRICT_BATCH", "8"))
GEN_MAX_NEW = int(os.environ.get("STRICT_MAX_NEW", "128"))
GEN_TEMP = float(os.environ.get("STRICT_TEMP", "0.7"))
GEN_TOP_P = float(os.environ.get("STRICT_TOP_P", "0.95"))
GEN_REP_PENALTY = float(os.environ.get("STRICT_REP_PENALTY", "1.1"))
MAX_RECORDS = int(os.environ.get("STRICT_MAX_RECORDS", "0"))
MAX_INPUT_LENGTH = 384
MASK_CJK = os.environ.get("STRICT_MASK_CJK", "1") == "1"


# === Prompt ===
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
    lines = []
    for k, v in attrs.items():
        s = str(v).strip() if v else ""
        if s:
            lines.append(f"{k}: {s}")
    return "\n".join(lines)


def make_prompt(attrs: dict, variant_idx: int = 0) -> str:
    attr_block = build_user_content(attrs)
    return GEN_SYSTEM.replace("{ATTRIBUTES}", attr_block)


# === Hallucination detection ===
HALLUCINATION_WORDS = [
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
    "scented", "scent", "fragrance", "aroma", "flavor", "flavour",
    "decorative", "decoration", "ornament", "ornamental",
    "friend", "friends", "sister", "sister's", "brother", "brother's",
    "wife", "wife's", "husband", "husband's", "mom", "mom's", "dad",
    "dad's", "mother", "father", "daughter", "son", "grandma",
    "grandpa", "aunt", "uncle", "cousin", "neighbor",
    "lightweight", "heavyweight", "premium", "luxury", "cheap",
    "professional-grade", "commercial", "industrial",
    "handmade", "handcrafted", "artisan",
    "accessory", "accessories", "tool", "tools", "decor", "gadget",
    "gizmo", "implement", "device", "apparatus",
]


def count_attrs_covered(text: str, attrs: dict) -> int:
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
    text_lower = text.lower()
    attr_substrings: set[str] = set()
    for v in attrs.values():
        s = str(v).strip() if v else ""
        if s:
            attr_substrings.add(s.lower())
            for tok in re.split(r"[\s,/]+", s.lower()):
                if len(tok) >= 4:
                    attr_substrings.add(tok)
    hits = []
    for w in HALLUCINATION_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", text_lower):
            if w not in attr_substrings:
                hits.append(w)
    return hits


def has_invalid_punct(text: str) -> bool:
    return bool(re.search(r'["\(\][""]', text))


def score_candidate(text: str, attrs: dict) -> tuple[int, int, int]:
    n_cov = count_attrs_covered(text, attrs)
    n_hallu = len(has_hallucination(text, attrs))
    invalid = 1 if has_invalid_punct(text) else 0
    return (n_cov, -n_hallu, -invalid)


# === Style vector loading ===
LAYERS = [16, 20, 24, 26]


def load_user_profiles() -> dict:
    if not USER_STYLE_VECTORS_JSONL.exists():
        if INJECT_ALPHA == 0.0:
            print(f"[profile] {USER_STYLE_VECTORS_JSONL} not found, ALPHA=0 → zero vectors", flush=True)
            return {}
        raise FileNotFoundError(f"{USER_STYLE_VECTORS_JSONL} not found")
    profiles: dict = {}
    with open(USER_STYLE_VECTORS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            uid = row["user_id"]
            profiles[uid] = {}
            for L in INJECT_LAYERS:
                key = f"residual_mean_layer_{L}"
                if key in row:
                    profiles[uid][L] = row[key]
    print(f"[profile] loaded {len(profiles)} user profiles", flush=True)
    return profiles


def build_zero_profiles(records: list) -> dict:
    """ALPHA=0 时用 zero vector，文件不存在则自动构建。"""
    if USER_STYLE_VECTORS_JSONL.exists():
        return load_user_profiles()
    print(f"[profile] building zero-vector style file: {USER_STYLE_VECTORS_JSONL}", flush=True)
    HIDDEN_DIM = 3584
    rows = []
    seen = set()
    for r in records:
        uid = r.get("user_id")
        if uid and uid not in seen:
            seen.add(uid)
            row = {"user_id": uid}
            for L in LAYERS:
                row[f"residual_mean_layer_{L}"] = [0.0] * HIDDEN_DIM
            rows.append(row)
    USER_STYLE_VECTORS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(USER_STYLE_VECTORS_JSONL, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[profile] wrote {len(rows)} zero-vector profiles", flush=True)
    return load_user_profiles()


# === Main ===
def main() -> int:
    import torch
    t0 = time.time()

    # 1) Load records
    records = json.load(open(RECORDS_IN, "r", encoding="utf-8"))
    if MAX_RECORDS and len(records) > MAX_RECORDS:
        records = records[:MAX_RECORDS]
    print(f"[main] {len(records)} records from {RECORDS_IN}", flush=True)

    # 2) Load user profiles
    profiles = build_zero_profiles(records)

    # 3) Load Qwen
    print(f"[main] loading Qwen...", flush=True)
    from llm_client import create_qwen_local_client
    client = create_qwen_local_client(with_vllm=False)
    hidden_dim = client._hidden_backend.model.config.hidden_size
    print(f"[main] hidden_dim={hidden_dim}, K={K_SAMPLES}, layers={INJECT_LAYERS}, alpha={INJECT_ALPHA}", flush=True)

    # 4) Skip existing
    existing: dict = {}
    if RECORDS_OUT.exists():
        prev = json.load(open(RECORDS_OUT, "r", encoding="utf-8"))
        for r in prev:
            if r.get("y_plus_query_inject") and "user_id" in r:
                existing[(r["user_id"], r["asin"])] = r["y_plus_query_inject"]
        print(f"[main] {len(existing)} already done (skip)", flush=True)

    # 5) Build todo
    todo_records, todo_prompts, todo_injections, todo_sample_idx = [], [], [], []
    n_no_profile = 0
    for r in records:
        uid, asin = r["user_id"], r["asin"]
        if (uid, asin) in existing:
            continue
        attrs = r.get("attrs_used", {})
        if not attrs:
            continue
        prof = profiles.get(uid)
        if prof is None or INJECT_LAYERS[0] not in prof:
            n_no_profile += 1
            bias = torch.zeros(hidden_dim, dtype=torch.float32)
        else:
            bias = torch.as_tensor(prof[INJECT_LAYERS[0]], dtype=torch.float32)
        for k in range(K_SAMPLES):
            todo_records.append(r)
            todo_prompts.append(make_prompt(attrs, variant_idx=k + 1))
            todo_injections.append(bias)
            todo_sample_idx.append(k)
    print(f"[main] todo={len(todo_records)}, no_profile={n_no_profile}", flush=True)
    if not todo_records:
        print("[main] nothing to do", flush=True)
        return 0

    # 6) Init output
    out_records = []
    for r in records:
        if (r["user_id"], r["asin"]) in existing:
            r2 = dict(r)
            r2["y_plus_query_inject"] = existing[(r["user_id"], r["asin"])]
            out_records.append(r2)
        else:
            out_records.append(dict(r))

    # 7) Generate in batches
    candidates_by_key, new_records_by_key = {}, {}
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
            print(f"[main] batch failed at {i}: {exc!r}", flush=True)
            traceback.print_exc()
            raise
        for r, q in zip(chunk_records, queries):
            key = (r["user_id"], r["asin"])
            candidates_by_key.setdefault(key, []).append(q)
            new_records_by_key[key] = r
        done = min(i + GEN_BATCH, len(todo_prompts))
        rate = done / max(time.time() - t0, 1e-6)
        eta = (len(todo_prompts) - done) / max(rate, 1e-6)
        print(f"[main] {done}/{len(todo_prompts)} ({rate:.1f}/s, ETA {eta:.0f}s)", flush=True)

    # 8) Pick best per record
    n_all_hallu = 0
    for key, cands in candidates_by_key.items():
        r = new_records_by_key[key]
        attrs = r.get("attrs_used", {})
        scored = [(score_candidate(c, attrs), c) for c in cands]
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_q = scored[0]
        n_cov, neg_hallu, neg_inv = best_score
        n_hallu = -neg_hallu
        r["y_plus_query_inject"] = best_q
        r["n_candidates"] = len(cands)
        r["n_attrs_covered"] = n_cov
        r["n_hallucinations"] = n_hallu
        r["hallucination_words"] = has_hallucination(best_q, attrs)
        r["all_candidates"] = cands
        r["all_candidates_scores"] = [list(s) for s, _ in scored]
        if n_hallu > 0:
            n_all_hallu += 1

    # 9) Merge back
    for key, r in new_records_by_key.items():
        for out in out_records:
            if out["user_id"] == r["user_id"] and out["asin"] == r["asin"]:
                out.update({
                    "y_plus_query_inject": r.get("y_plus_query_inject"),
                    "n_candidates": r.get("n_candidates"),
                    "n_attrs_covered": r.get("n_attrs_covered"),
                    "n_hallucinations": r.get("n_hallucinations"),
                    "hallucination_words": r.get("hallucination_words"),
                    "all_candidates": r.get("all_candidates"),
                    "all_candidates_scores": r.get("all_candidates_scores"),
                })
                break

    # 10) Write
    RECORDS_OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_records, open(RECORDS_OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    n_done = len(candidates_by_key)
    print(f"[main] wrote {n_done} records to {RECORDS_OUT}", flush=True)
    print(f"[main] hallucinations in best: {n_all_hallu}/{n_done}", flush=True)
    print(f"[main] total {time.time()-t0:.1f}s", flush=True)

    # 11) Detail
    for key in sorted(candidates_by_key.keys()):
        r = new_records_by_key[key]
        attrs = r.get("attrs_used", {})
        best_q = r["y_plus_query_inject"]
        cands = r["all_candidates"]
        print(f"\n--- uid={r['user_id'][:14]} asin={r['asin']} ---", flush=True)
        print(f"  attrs: {dict(list(attrs.items())[:4])}", flush=True)
        print(f"  BEST (cov={r['n_attrs_covered']} hallu={r['n_hallucinations']}): {best_q}", flush=True)
        if r["hallucination_words"]:
            print(f"         hallu: {r['hallucination_words']}", flush=True)
        for i, c in enumerate(cands):
            cov = count_attrs_covered(c, attrs)
            h = has_hallucination(c, attrs)
            mark = " *" if c == best_q else "  "
            print(f"  {mark}cand{i+1}(cov={cov} hallu={len(h)}): {c}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
