#!/usr/bin/env python3
"""SFT pool generation via vLLM.

用户 2026-09-05 指令: 接入 vLLM 加载 SFT adapter (result/06_training_model/sft_lora)
生成 50 ASIN × K_POOL 个 candidates, 输出到 result/07_gen_query/pool.json.

设计:
  - vLLM LLM engine 加载 Qwen2.5-0.5B-Instruct + enable_lora=True
  - LoRARequest 指向 sft_lora (LoRA r=8)
  - 50 ASIN × K_POOL batched generation (单次 generate 调用)
  - 复用 06_training_model/sft_pipeline.py 的 prompt 格式 (含 <|im_end|> 终止符)
  - 按 5-layer strict filter (含 ≤3 extras 阈值, 用户 2026-09-05 反馈)
  - generate_candidates() / get_or_create_llm() 供 08_select_query regen 复用

用法 (Rule 3: no args):
  cd /home/wlia0047/ar57/wenyu/PersoanlQuery
  $PY 07_gen_query/sft_pool_generate.py

输出:
  result/07_gen_query/pool_queries.json
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "result/07_gen_query"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Stage 04 canonical artifact 提供 fitted users 与 ASIN cohort coverage。
STAGE04_PATH = REPO_ROOT / "result/04_gaussian/user_gaussian_stats.json"
SFT_POOL_SOURCE = "stage04_fitted_cohort"  # 标注抽样源

# CONTRASTIVE_5558 ABLATION: 5558 contrastive cohort stats
CONTRASTIVE_5558 = False
if CONTRASTIVE_5558:
    STAGE04_PATH = REPO_ROOT / "result/04_gaussian/user_gaussian_stats_5558.json"
    SFT_POOL_SOURCE = "stage04_fitted_cohort_5558_contrastive"

QWEN_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
SFT_ADAPTER_DIR = REPO_ROOT / "result/06_training_model/sft_lora"
ATTRIBUTES_PATH = REPO_ROOT / "result/01_attribute_extraction/product_attributes.json"

# --- Hardcoded hyperparams (Rule 3) ---
SFT_POOL_SEED = 42
SFT_POOL_SMOKE = False        # True=5 ASIN smoke, False=100 ASIN full (Rule 20)
SFT_POOL_N_ASIN = 5 if SFT_POOL_SMOKE else None  # SMOKE=5, full=None=不限制(全量 asin_to_valid)
SFT_POOL_K = 25               # 2026-09-06: 提速 C 方案, n=25 跑 2 轮, 总候选 50/ASIN (KV cache 复用)
SFT_POOL_ROUNDS = 2            # 2026-09-06: 同 prompt 跑 2 轮, 合并为 50/ASIN
SFT_POOL_TEMP = 0.7
SFT_POOL_TOP_P = 0.95
SFT_POOL_MAX_NEW_TOKENS = 40
SFT_POOL_EXCLUDE_TRAIN = True # 排除 SFT 训练 ASIN (用户 query_samples_100.json 用过)
# 2026-09-13 去掉 extras ≤ 3 阈值:用户指令,只保留全覆盖 + 无重复两条 filter。
# extras 信号仍记录在 candidate 字典,但不再 fail pass。

# vLLM 引擎单例 (07 regen 复用, lazy 初始化)
_LLM = None


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _build_sft_prompt(attrs: Dict[str, str]) -> str:
    """复用 06_training_model/sft_pipeline._build_sft_prompt (含 <|im_end|> 终止符)."""
    attrs_text = "\n".join(f"- {k}: {v}" for k, v in attrs.items())
    return (
        f"You are shopping for a product with these 5 known attributes:\n"
        f"{attrs_text}\n\n"
        f"Write one short natural shopping search query (5-20 words) that "
        f"uses these attributes as keywords. Do NOT ask questions about the "
        f"attributes (they are already known). Do NOT invent or add new "
        f"product information not listed above.\n\n"
        f"Query:<|im_end|>\n"
    )


def load_attributes() -> Dict[str, Dict[str, str]]:
    with open(ATTRIBUTES_PATH) as f:
        return json.load(f)


def filter_pass_queries(asin: str, queries: List[str],
                        attrs_all: Dict[str, Dict[str, str]]) -> List[str]:
    """对单个 ASIN 的 query 列表按 content_pass 过滤,只保留 pass=True。

    用户 2026-09-13:写入 pool_queries.json 前,去掉不合格 query。
    复刻 generate_candidates.check_content 的 tokenize + 全覆盖 + 无重复(去掉 extras 阈值)。
    cached pool 也走这条路径,因为旧产物没经过滤。
    """
    import re
    if not queries:
        return []
    if asin not in attrs_all:
        # attrs 缺失 → 全部不合格,跟 generate_candidates 行为一致(raise)
        raise ValueError(f"ASIN {asin} missing in product_attributes")
    attrs = dict(list(attrs_all[asin].items())[:5])

    def tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+(?:['-]\w+)*\b", text)

    kept: List[str] = []
    n_total = len(queries)
    for text in queries:
        tokens = tokenize(text.lower())
        normalized = " ".join(tokens)
        missing = []
        for value in attrs.values():
            value_tokens = tokenize(str(value).lower())
            if not value_tokens:
                continue
            value_text = " ".join(value_tokens)
            if value_text not in normalized:
                missing.append(value_text)
                break
        repeated = len(tokens) != len(set(tokens))
        if not missing and not repeated:
            kept.append(text)
    return kept


def select_eval_asins(attrs_all: Dict[str, Dict[str, str]],
                      n_asin: int) -> List[str]:
    """从 Stage 04 fitted cohort 中选择有属性的 ASIN。"""
    if not STAGE04_PATH.exists():
        raise FileNotFoundError(
            f"{STAGE04_PATH} 不存在, 请先跑 04_gaussian/fit_per_user_gaussian.py"
        )
    with open(STAGE04_PATH) as f:
        stage04 = json.load(f)
    asin_to_valid = stage04.get("cohort_gates")
    if not isinstance(asin_to_valid, dict) or not asin_to_valid:
        raise ValueError("Stage 04 artifact requires non-empty cohort_gates")
    for asin, cohort in asin_to_valid.items():
        if not isinstance(cohort, dict) or len(cohort) < 2:
            raise ValueError(f"Stage 04 cohort {asin} has fewer than two users")

    excluded = set()
    if SFT_POOL_EXCLUDE_TRAIN:
        sft_train_path = REPO_ROOT / "result/06_training_model/sft_train_asins.json"
        if sft_train_path.exists():
            with open(sft_train_path) as f:
                excluded = set(json.load(f).get("train_asins", []))

    # valid ≥ 2 users ∩ 有 attrs ≥ 5 字段 ∩ 非 train
    candidates = [(a, len(v)) for a, v in asin_to_valid.items()
                  if a not in excluded
                  and a in attrs_all
                  and len(list(attrs_all[a].items())[:5]) >= 5]
    log(f"  coverage ASINs (≥2 valid users): {len(asin_to_valid)}")
    log(f"  after exclude-train ∩ attrs≥5: {len(candidates)}")

    # 降序 (高 valid uid density 优先), tie-break seed shuffle 保证可复现
    candidates.sort(key=lambda av: (-av[1], av[0]))
    rng = random.Random(SFT_POOL_SEED)
    rng.shuffle(candidates)  # 仅打散同密度的 ASIN
    candidates.sort(key=lambda av: -av[1])
    if n_asin is None:
        selected = [a for a, _ in candidates]
    else:
        selected = [a for a, _ in candidates[:n_asin]]
        if len(selected) < n_asin:
            log(f"  WARNING: only {len(selected)} ASINs available, requested {n_asin}")
    return selected


def get_or_create_llm():
    """vLLM 引擎 lazy 单例 (GPU 与 qwen_hidden_server 共享, util=0.42)."""
    global _LLM
    if _LLM is not None:
        return _LLM
    from vllm import LLM

    log(f"  loading vLLM engine (Qwen2.5-0.5B + enable_lora=True) ...")
    t0 = time.time()
    _LLM = LLM(
        model=QWEN_PATH,
        enable_lora=True,
        max_lora_rank=8,
        max_model_len=384,        # 150 prompt + 40 gen + 32 query = 222, 384 留 slack
        gpu_memory_utilization=0.85,  # 2026-09-06: 用户提速要求, 0.55→0.85 + max_num_seqs 扩并发
        max_num_seqs=1024,             # 2026-09-06: vLLM 默认 256, 扩并发 batch
        enforce_eager=False,           # 2026-09-06: 启用 CUDA graph, 预期 +20-30% 吞吐
        dtype="bfloat16",
        trust_remote_code=True,
    )
    log(f"  vLLM engine loaded in {time.time()-t0:.1f}s")
    return _LLM


def generate_candidates(attrs_list: List[Dict[str, str]],
                        k: int) -> List[List[Dict]]:
    """对每个 ASIN 的 attrs 生成 k 个 candidates (batched), 返回 content-check 后的列表.

    供本脚本 main 和 08_select_query regen 复用.
    """
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    import re

    def tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+(?:['-]\w+)*\b", text)

    def check_content(text: str, attrs: Dict[str, str]):
        tokens = tokenize(text.lower())
        normalized = " ".join(tokens)
        covered = {}
        missing = []
        for key, value in attrs.items():
            value_tokens = tokenize(str(value).lower())
            if not value_tokens:
                raise ValueError(f"attribute {key} has no tokenized value")
            value_text = " ".join(value_tokens)
            covered[key] = value_text in normalized
            if not covered[key]:
                missing.append(key)
        repeated = len(tokens) != len(set(tokens))
        known_tokens = {
            token for value in attrs.values()
            for token in tokenize(str(value).lower())
        }
        extras = [token for token in tokens if token not in known_tokens]
        return covered, missing, repeated, extras

    llm = get_or_create_llm()
    prompts = [_build_sft_prompt(attrs) for attrs in attrs_list]

    sampling_params = SamplingParams(
        n=k,
        temperature=SFT_POOL_TEMP,
        top_p=SFT_POOL_TOP_P,
        max_tokens=SFT_POOL_MAX_NEW_TOKENS,
        stop=["<|im_end|>", "\n\n"],  # 防止 model 续写 system echo
    )
    lora_request = LoRARequest("sft_adapter", 1, str(SFT_ADAPTER_DIR))

    t0 = time.time()
    log(f"  starting vLLM generate: {len(prompts)} prompts × n={k} "
        f"= {len(prompts)*k} candidates")
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request,
                           use_tqdm=True)
    log(f"  vLLM generate done in {time.time()-t0:.1f}s")

    all_cands: List[List[Dict]] = []
    for attrs, out in zip(attrs_list, outputs):
        cands = []
        for choice in out.outputs:
            text = choice.text.strip().split("\n")[0].strip()
            cov, miss, rep, extras = check_content(text, attrs)
            content_pass = (len(miss) == 0 and not rep)
            cands.append({
                "text": text,
                "cov": cov,
                "missing": miss,
                "repeated": rep,
                "extras": extras,
                "pass": content_pass,
                "n_words": len(tokenize(text)),
                "n_tokens": len(choice.token_ids),
            })
        all_cands.append(cands)
    return all_cands


def main_pipeline():
    log("=== sft_pool_generate ===")
    log(f"  Qwen={QWEN_PATH}  adapter={SFT_ADAPTER_DIR}")
    log(f"  N_ASIN={SFT_POOL_N_ASIN}  K={SFT_POOL_K}  rounds={SFT_POOL_ROUNDS}  temp={SFT_POOL_TEMP} "
        f"top_p={SFT_POOL_TOP_P}  max_new_tokens={SFT_POOL_MAX_NEW_TOKENS}")

    attrs_all = load_attributes()
    eval_asins = select_eval_asins(attrs_all, SFT_POOL_N_ASIN)
    log(f"  eval_asins (n={len(eval_asins)}): {eval_asins[:5]} ...")

    # --- Incremental cache: 已有 pool_queries.json 时按 ASIN 复用 ---
    out_path = OUT_DIR / "pool_queries.json"
    expected_per_asin = SFT_POOL_K * SFT_POOL_ROUNDS
    cached_pool: Dict[str, List[str]] = {}
    cached_source = ""
    cached_n_total = 0
    if out_path.exists():
        with open(out_path) as _cf:
            _cached_doc = json.load(_cf)
        cached_pool = dict(_cached_doc.get("pool", {}))
        cached_source = _cached_doc.get("config", {}).get("source", "")
        cached_n_total = sum(len(v) for v in cached_pool.values())
        log(f"  [cache] loaded {len(cached_pool)} ASINs from existing pool "
            f"(n_total={cached_n_total}, source={cached_source})")
        if cached_source and cached_source != SFT_POOL_SOURCE:
            raise ValueError(
                f"existing pool source='{cached_source}' differs from "
                f"current SFT_POOL_SOURCE='{SFT_POOL_SOURCE}', refusing to merge")

    # 哪些 ASIN 已完整 (count == expected_per_asin) — 直接复用, 不重新生成
    reuse_asins = [a for a in eval_asins
                   if a in cached_pool and len(cached_pool[a]) >= expected_per_asin]
    fresh_asins = [a for a in eval_asins if a not in reuse_asins]
    log(f"  [cache] reuse {len(reuse_asins)} ASINs, regen {len(fresh_asins)} ASINs")

    fresh_pool: Dict[str, List[str]] = {}
    if fresh_asins:
        attrs_list = [dict(list(attrs_all[a].items())[:5]) for a in fresh_asins]
        cands_per_asin_rounds: List[List[List[Dict]]] = []
        for round_idx in range(SFT_POOL_ROUNDS):
            log(f"  --- round {round_idx+1}/{SFT_POOL_ROUNDS} ---")
            cands_per_asin_rounds.append(generate_candidates(attrs_list, SFT_POOL_K))
        cands_per_asin = [c1 + c2 for c1, c2 in zip(*cands_per_asin_rounds)]
        fresh_pool = {entry_asin: [c["text"] for c in cands]
                      for entry_asin, cands in zip(fresh_asins, cands_per_asin)}

    pool: Dict[str, List[str]] = {}
    n_dropped_total = 0
    n_zero_query_asins = 0
    n_below_min_query_asins = 0
    MIN_KEEP_QUERIES = 2  # 用户 2026-09-13:<2 query 的 ASIN 不进 pool
    for asin in eval_asins:
        if asin in reuse_asins:
            raw = list(cached_pool[asin])
        elif asin in fresh_pool:
            raw = fresh_pool[asin]
        else:
            continue
        kept = filter_pass_queries(asin, raw, attrs_all)
        n_dropped_total += len(raw) - len(kept)
        if len(kept) < MIN_KEEP_QUERIES:
            if not kept:
                n_zero_query_asins += 1
            else:
                n_below_min_query_asins += 1
            continue
        pool[asin] = kept
    n_total = sum(len(v) for v in pool.values())

    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "model": QWEN_PATH,
                "lora_adapter": str(SFT_ADAPTER_DIR),
                "n_asin": len(pool),
                "k_per_asin": SFT_POOL_K,
                "rounds": SFT_POOL_ROUNDS,
                "temp": SFT_POOL_TEMP,
                "top_p": SFT_POOL_TOP_P,
                "max_new_tokens": SFT_POOL_MAX_NEW_TOKENS,
                "source": SFT_POOL_SOURCE,
                "stage04_source": str(STAGE04_PATH),
                "incremental_reuse": len(reuse_asins),
                "incremental_fresh": len(fresh_asins),
                "pass_filter_applied": True,
                "pass_filter_criteria": "full_coverage_5_attrs AND no_repeat_token",
                "min_keep_queries_per_asin": MIN_KEEP_QUERIES,
                "n_dropped_by_filter": n_dropped_total,
                "n_asins_dropped_all_queries": n_zero_query_asins,
                "n_asins_dropped_below_min_queries": n_below_min_query_asins,
            },
            "pool": pool,
        }, f, indent=2, ensure_ascii=False)
    log(f"  saved -> {out_path} (n_total={n_total}, dropped={n_dropped_total}, "
        f"all-removed={n_zero_query_asins}, below-min={n_below_min_query_asins})")
    log(f"  SUMMARY: {n_total} queries, {len(pool)} ASINs "
        f"(reuse={len(reuse_asins)}, fresh={len(fresh_asins)}, dropped={n_dropped_total}, "
        f"all-removed={n_zero_query_asins}, below-min={n_below_min_query_asins})")
    log("=== sft_pool_generate DONE ===")


if __name__ == "__main__":
    main_pipeline()
