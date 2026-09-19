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
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "result/07_gen_query"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Stage 04 cohort3mlp lowrank+diag artifact 提供 fitted users 与 ASIN cohort coverage。
STAGE04_PATH = REPO_ROOT / "result/04_gaussian/user_gaussian_stats_cohort3mlp16_30_lowrankdiag_rank1.json"
SFT_POOL_SOURCE = "stage04_cohort3mlp_lowrankdiag"  # 标注抽样源

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
SFT_POOL_K = 50               # 2026-09-19: 每 round 每个 ASIN 生成 50 个候选, 2 轮 = 100/ASIN
SFT_POOL_ROUNDS = 2            # 2026-09-06: 同 prompt 跑 2 轮, 合并为 50/ASIN
MIN_KEPT_QUERIES = 30          # 2026-09-18: filter 后必须 ≥30 个 query; 不足则继续重新生成
MAX_REGEN_ROUNDS = 10          # 每个 ASIN 最多重新生成多少轮 (limit 防止 vLLM 跑死)
POOL_DUMP_EVERY_N_ROUNDS = 1   # 每 N 个 round 把 raw_per_asin dump 到 disk, 防止崩溃丢失
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


def clean_attr_value(value: str, is_brand: bool = False) -> str:
    """清洗单个 attribute 字符串,在明显"两词粘连无空格"时拆开。

    保留策略 (2026-09-19):
      - 品牌字段 (Brand / Manufacturer / Item model number): **永远保留原样**,
        query 可能整串出现 (BabyBjörn / WubbaNub / StrollAir)。
      - 其他字段: 激进 split, 把 CamelCase / 多段大写词 / 段下划线全拆开。
        "Dishwasher SafeMicrowave Safe" → "Dishwasher Safe Microwave Safe"
        "machine_wash" → "machine wash"
        "Reclining SeatCanopy" → "Reclining Seat Canopy"
        "Walk_through" → "Walk through"
        "MusicalLightsInteractive ToysAdjustable Height" → 多个词
      - 单 token 全大写 (BPA / DC) / 短串: 保留。
    """
    if not isinstance(value, str):
        return str(value)
    v = value.strip()
    if not v:
        return v
    if is_brand:
        return v
    # 2026-09-19: 激进拆 — 在 [a-z][A-Z], [A-Z]+[A-Z][a-z], '_' 等多处边界拆开
    v = v.replace("_", " ")
    v = re.sub(r"([a-z])([A-Z])", r"\1 \2", v)               # camelCase
    v = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", v)          # FOOBar → FOO Bar
    return v.strip()


_BRAND_FIELDS = {"brand", "manufacturer", "item model number"}


def select_clean_attrs(attrs: Dict[str, str], n_keep: int = 5) -> Dict[str, str]:
    """从 attrs dict 选前 n_keep 个,清洗粘连 attr,过滤清洗后仍异常字段。
    品牌字段 (Brand / Manufacturer) 跳过清洗保留原样。
    2026-09-19: 清洗后 tokenize 仍只有 1 个 token 且原值 >10 字符无空格 → 跳过。
    """
    out: Dict[str, str] = {}
    for k, v_raw in attrs.items():
        is_brand = k.lower() in _BRAND_FIELDS
        v_clean = clean_attr_value(v_raw, is_brand=is_brand)
        if not v_clean:
            continue
        toks = re.findall(r"\b\w+(?:['-]\w+)*\b", v_clean.lower())
        if not toks:
            continue
        v_text = " ".join(toks)
        if len(v_text) > 30:
            continue
        # 2026-09-19: 清洗后仍只 1 token 且原值无空格且 >10 字符 → 不可恢复异常, 跳过
        if len(toks) == 1 and " " not in v_raw and len(v_raw) > 10:
            continue
        out[k] = v_clean
        if len(out) >= n_keep:
            break
    return out


def filter_pass_queries(asin: str, queries: List[str],
                        attrs_all: Dict[str, Dict[str, str]]) -> List[str]:
    """对单个 ASIN 的 query 列表按 content_pass 过滤,只保留 pass=True。

    用户 2026-09-13:写入 pool_queries.json 前,去掉不合格 query。
    复刻 generate_candidates.check_content 的 tokenize + 全覆盖 (2026-09-18 去掉 no_repeat_token 检查)。
    cached pool 也走这条路径,因为旧产物没经过滤。
    2026-09-18: 选 attrs 前先经 select_clean_attrs 清洗,过滤掉粘连 / 异常 attr。
    """
    if not queries:
        return []
    if asin not in attrs_all:
        # attrs 缺失 → 全部不合格,跟 generate_candidates 行为一致(raise)
        raise ValueError(f"ASIN {asin} missing in product_attributes")
    attrs = select_clean_attrs(attrs_all[asin], n_keep=5)

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
        if not missing:  # 2026-09-18: 去掉 no_repeat_token 检查 (小词 "a"/"the" 重复导致 reject)
            kept.append(text)
    return kept


def select_eval_asins(attrs_all: Dict[str, Dict[str, str]],
                      n_asin: int) -> List[str]:
    """从 Stage 04 fitted cohort 中选择有属性的 ASIN, 仅保留 Stage 5 audit 中
    至少 1 个 user 合格的 ASIN。"""
    if not STAGE04_PATH.exists():
        raise FileNotFoundError(
            f"{STAGE04_PATH} 不存在, 请先跑 04_gaussian/fit_per_user_gaussian.py"
        )
    with open(STAGE04_PATH) as f:
        stage04 = json.load(f)
    asin_to_valid = stage04.get("cohort_gates")
    if not isinstance(asin_to_valid, dict) or not asin_to_valid:
        raise ValueError("Stage 04 artifact requires non-empty cohort_gates")

    STAGE05_PATH = REPO_ROOT / "result/05_gaussian_audit/raw_cov_validity.json"
    valid_uids: set[str] = set()
    if STAGE05_PATH.exists():
        with open(STAGE05_PATH) as f:
            stage05 = json.load(f)
        valid_uids = {uid for uid, r in stage05.get("per_user", {}).items()
                      if r.get("valid_gaussian")}
        log(f"  [stage05] loaded {len(valid_uids)} valid uids from {STAGE05_PATH}")
    else:
        log(f"  [stage05] WARNING: {STAGE05_PATH} not found; "
            f"using all Stage 04 fitted users")

    for asin, cohort in asin_to_valid.items():
        if not isinstance(cohort, dict) or len(cohort) < 2:
            raise ValueError(f"Stage 04 cohort {asin} has fewer than two users")

    excluded = set()
    if SFT_POOL_EXCLUDE_TRAIN:
        sft_train_path = REPO_ROOT / "result/06_training_model/sft_train_asins.json"
        if sft_train_path.exists():
            with open(sft_train_path) as f:
                excluded = set(json.load(f).get("train_asins", []))

    # valid ≥ 1 Stage5 user ∩ ≥ 2 Stage 4 cohort users ∩ attrs ≥ 5 ∩ 非 train
    candidates = []
    for asin, udict in asin_to_valid.items():
        if asin in excluded:
            continue
        if asin not in attrs_all or len(list(attrs_all[asin].items())[:5]) < 5:
            continue
        if valid_uids:
            if not any(uid in valid_uids for uid in udict.keys()):
                continue
        candidates.append((asin, len(udict)))
    log(f"  coverage ASINs (Stage 04 cohort_gates): {len(asin_to_valid)}")
    log(f"  after stage05-filter ∩ attrs≥5: {len(candidates)}")

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

    # 哪些 ASIN 已完整 (count >= MIN_KEPT_QUERIES) — 直接复用, 不重新生成
    reuse_asins = [a for a in eval_asins
                   if a in cached_pool and len(cached_pool[a]) >= MIN_KEPT_QUERIES]
    fresh_asins = [a for a in eval_asins if a not in reuse_asins]
    # 2026-09-18: 加载 raw_per_asin cache (前次崩溃留下来的 raw candidates)
    raw_cache_path = OUT_DIR / "raw_per_asin_cache.json"
    raw_per_asin_cache: dict[str, list] = {}
    if raw_cache_path.exists():
        try:
            with open(raw_cache_path) as _rcf:
                raw_per_asin_cache = json.load(_rcf)
            log(f"  [raw-cache] loaded raw candidates for "
                f"{len(raw_per_asin_cache)} ASINs from {raw_cache_path}")
        except Exception as e:
            log(f"  [raw-cache] load failed: {e}; ignore cache")
            raw_per_asin_cache = {}
    # 2026-09-19: cached_pool ≥30 完全 reuse; raw cache 中剩余 ASINs (cached_pool <30)
    # 走 fresh 路径, raw cache 提供 initial raw, regen 给 <30 的 ASINs 补齐。
    reuse_asins = list(set(reuse_asins))
    fresh_asins = [a for a in eval_asins if a not in reuse_asins]
    log(f"  [cache] reuse {len(reuse_asins)} ASINs (cached_pool ≥{MIN_KEPT_QUERIES} ∪ raw_cache), "
        f"regen {len(fresh_asins)} ASINs (raw_cache 提供 initial raw, "
        f"{len(raw_per_asin_cache)} ASINs 在 raw cache)")

    fresh_pool: Dict[str, List[str]] = {}
    if fresh_asins:
        attrs_list = [dict(list(attrs_all[a].items())[:5]) for a in fresh_asins]
        # 增量生成: 每个 ASIN 累计 filter-pass 后的 queries, 直到 ≥ MIN_KEPT_QUERIES
        #           或 regen 轮数耗尽。
        # raw 是 candidate dicts 列表 (text + metadata), kept 是 string 列表。
        raw_per_asin = [[] for _ in fresh_asins]
        # 2026-09-18: 恢复 raw candidates cache (崩溃恢复)
        for i, asin in enumerate(fresh_asins):
            if asin in raw_per_asin_cache:
                cached = raw_per_asin_cache[asin]
                # strip metadata, keep only text for re-encoding
                raw_per_asin[i] = [{"text": c if isinstance(c, str) else c["text"]}
                                    for c in cached]
        n_loaded = sum(1 for r in raw_per_asin if r)
        log(f"  [raw-cache] loaded {n_loaded}/{len(fresh_asins)} ASINs "
            f"with raw candidates")
        kept_per_asin: List[List[str]] = [[] for _ in fresh_asins]
        # initial filter for cached raw candidates (避免重做)
        for i, asin in enumerate(fresh_asins):
            if raw_per_asin[i]:
                kept_per_asin[i] = filter_pass_queries(
                    asin, [c["text"] for c in raw_per_asin[i]], attrs_all)
        n_total_regen_rounds = 0
        # initial SFT_POOL_ROUNDS of generation (warm-up batches)
        for round_idx in range(SFT_POOL_ROUNDS):
            log(f"  --- round {round_idx+1}/{SFT_POOL_ROUNDS} (initial) ---")
            cands = generate_candidates(attrs_list, SFT_POOL_K)
            for i, asin_cands in enumerate(cands):
                raw_per_asin[i].extend(asin_cands)
            n_total_regen_rounds += 1
            # 2026-09-18: 每个 round 完成立即 dump raw cache + pool (filter pass ≥ MIN_KEPT_QUERIES),
            # 防止崩溃丢失 + 让 Stage 8 可提前消费
            try:
                # 2026-09-19: dump 时合并已有 raw_per_asin_cache + fresh_asins 新 raw,
                # 避免覆盖丢失 reuse_asins 的 raw 数据。
                _dump_existing: Dict[str, list] = {}
                if (OUT_DIR / "raw_per_asin_cache.json").exists():
                    try:
                        _dump_existing = json.load(
                            open(OUT_DIR / "raw_per_asin_cache.json"))
                    except Exception:
                        _dump_existing = {}
                _dump = dict(_dump_existing)
                for i, asin in enumerate(fresh_asins):
                    _dump[asin] = [c["text"] for c in raw_per_asin[i]]
                with open(OUT_DIR / "raw_per_asin_cache.json", "w") as _dcf:
                    json.dump(_dump, _dcf, ensure_ascii=False)
                # 同时 dump filter pass 的部分 pool (fresh + reuse 都包含)
                # 2026-09-19: 同步更新 kept_per_asin 让 regen rounds 不再选已 pass ASINs
                _pool_partial: Dict[str, List[str]] = {}
                for i, asin in enumerate(fresh_asins):
                    kept = filter_pass_queries(
                        asin, [c["text"] for c in raw_per_asin[i]], attrs_all)
                    kept_per_asin[i] = kept  # 同步 regen 用的 kept 列表
                    if len(kept) >= MIN_KEPT_QUERIES:
                        _pool_partial[asin] = kept
                # 2026-09-19: reuse_asins (raw cache 中) 也要 filter pass 一次进 pool
                for asin in reuse_asins:
                    if asin in _dump_existing and asin not in _pool_partial:
                        kept = filter_pass_queries(
                            asin, _dump_existing[asin], attrs_all)
                        if len(kept) >= MIN_KEPT_QUERIES:
                            _pool_partial[asin] = kept
                # 合并: 已有 cached_pool (reuse_asins) + fresh 的合格子集
                _pool_merged: Dict[str, List[str]] = dict(cached_pool)
                _pool_merged.update(_pool_partial)
                with open(OUT_DIR / "pool_queries.json", "w") as _pcf:
                    # 2026-09-19: 写 reuse_asin_list 让 Stage 8 fit CORAL 时用全 ASINs 子集
                    _reuse_list = sorted(set(cached_pool.keys())
                                         | set(reuse_asins)
                                         | set(_pool_partial.keys()))
                    json.dump({
                        "config": {
                            "source": SFT_POOL_SOURCE,
                            "min_keep_queries_per_asin": MIN_KEPT_QUERIES,
                            "n_total_fresh": len(fresh_asins),
                            "n_pool_fresh_kept": len(_pool_partial),
                            "n_pool_reuse_kept": len(reuse_asins),
                            "round_idx": n_total_regen_rounds,
                            "reuse_asin_list": _reuse_list,
                        },
                        "pool": _pool_merged,
                    }, _pcf, indent=2, ensure_ascii=False)
                log(f"  [dump] round {n_total_regen_rounds}: "
                    f"raw={len(_dump)} ASINs, "
                    f"pool={len(_pool_merged)} ASINs ({len(_pool_partial)} fresh pass, "
                    f"reuse_list={len(_reuse_list)})")
            except Exception as _e:
                log(f"  [raw-cache] dump failed: {_e}")
        # incremental regen: per-ASIN 检查是否达到 MIN_KEPT_QUERIES
        # 一次 generate_candidates 调用处理一批需要补的 ASINs.
        n_regen_full = 0
        while True:
            need_more = [i for i, kept in enumerate(kept_per_asin)
                         if len(kept) < MIN_KEPT_QUERIES]
            if not need_more:
                break
            if n_total_regen_rounds >= SFT_POOL_ROUNDS + MAX_REGEN_ROUNDS:
                log(f"  [regen] max rounds {n_total_regen_rounds} reached; "
                    f"still short on {len(need_more)} ASINs")
                break
            batch_attrs = [attrs_list[i] for i in need_more]
            log(f"  [regen] batch size={len(need_more)}, round {n_total_regen_rounds+1}")
            cands = generate_candidates(batch_attrs, SFT_POOL_K)
            for batch_idx, asin_cands in enumerate(cands):
                i = need_more[batch_idx]
                raw_per_asin[i].extend(asin_cands)
            n_total_regen_rounds += 1
            n_regen_full += 1
            # after each new batch, recompute kept for the touched ASINs (cheap)
            for i in need_more:
                kept_per_asin[i] = filter_pass_queries(
                    fresh_asins[i],
                    [c["text"] for c in raw_per_asin[i]],
                    attrs_all,
                )
            # 2026-09-18: 周期性 dump raw cache + pool partial (防崩溃丢数据, 让 Stage 8 可提前消费)
            if POOL_DUMP_EVERY_N_ROUNDS > 0 and n_regen_full % POOL_DUMP_EVERY_N_ROUNDS == 0:
                try:
                    # 2026-09-19: dump 时合并已有 raw_per_asin_cache + fresh_asins 新 raw
                    _dump_existing: Dict[str, list] = {}
                    if (OUT_DIR / "raw_per_asin_cache.json").exists():
                        try:
                            _dump_existing = json.load(
                                open(OUT_DIR / "raw_per_asin_cache.json"))
                        except Exception:
                            _dump_existing = {}
                    dump = dict(_dump_existing)
                    for i, asin in enumerate(fresh_asins):
                        dump[asin] = [c["text"] for c in raw_per_asin[i]]
                    with open(OUT_DIR / "raw_per_asin_cache.json", "w") as _dcf:
                        json.dump(dump, _dcf, ensure_ascii=False)
                    _pool_partial = {fresh_asins[i]: kept_per_asin[i]
                                     for i in range(len(fresh_asins))
                                     if len(kept_per_asin[i]) >= MIN_KEPT_QUERIES}
                    # 2026-09-19: reuse_asins (raw cache 中) 也 filter pass 一次进 pool
                    for asin in reuse_asins:
                        if asin in _dump_existing and asin not in _pool_partial:
                            k = filter_pass_queries(
                                asin, _dump_existing[asin], attrs_all)
                            if len(k) >= MIN_KEPT_QUERIES:
                                _pool_partial[asin] = k
                    _pool_merged = dict(cached_pool)
                    _pool_merged.update(_pool_partial)
                    with open(OUT_DIR / "pool_queries.json", "w") as _pcf:
                        # 2026-09-19: 写 reuse_asin_list (cached_pool ∪ reuse_asins ∪ pool_partial)
                        # 让 Stage 8 fit CORAL global A 时用全 ASINs 子集算
                        _reuse_list = sorted(set(cached_pool.keys())
                                             | set(reuse_asins)
                                             | set(_pool_partial.keys()))
                        json.dump({
                            "config": {
                                "source": SFT_POOL_SOURCE,
                                "min_keep_queries_per_asin": MIN_KEPT_QUERIES,
                                "n_pool_fresh_kept": len(_pool_partial),
                                "n_pool_reuse_kept": len(reuse_asins),
                                "round_idx": n_total_regen_rounds,
                                "reuse_asin_list": _reuse_list,
                            },
                            "pool": _pool_merged,
                        }, _pcf, indent=2, ensure_ascii=False)
                    log(f"  [dump] round {n_total_regen_rounds}: "
                        f"raw={len(dump)} ASINs, "
                        f"pool={len(_pool_merged)} ASINs ({len(_pool_partial)} fresh pass, "
                        f"reuse_list={len(_reuse_list)})")
                except Exception as e:
                    log(f"  [raw-cache] dump failed: {e}")
        log(f"  [regen] total rounds: {n_total_regen_rounds} "
            f"(initial 2 + {n_regen_full} regen)")
        fresh_pool = {fresh_asins[i]: kept for i, kept in enumerate(kept_per_asin)}

    pool: Dict[str, List[str]] = {}
    n_dropped_total = 0
    n_zero_query_asins = 0
    n_below_min_query_asins = 0
    MIN_KEEP_QUERIES = 30  # 2026-09-18: 强制 ≥30 个 query; 不足则该 ASIN 不进 pool
    for asin in eval_asins:
        if asin in cached_pool and len(cached_pool[asin]) >= MIN_KEPT_QUERIES:
            # cached_pool hit (reuse_asins ∩ cached_pool)
            raw = list(cached_pool[asin])
        elif asin in raw_per_asin_cache:
            # raw cache hit (reuse_asins ∩ raw_per_asin_cache only, cached_pool 空)
            raw = list(raw_per_asin_cache[asin])
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
                "reuse_asin_list": reuse_asins,  # 2026-09-18: Stage 8 fit CORAL 时 global fallback A 只用 reuse_asins 子集算
                "fresh_asin_list": fresh_asins,
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
