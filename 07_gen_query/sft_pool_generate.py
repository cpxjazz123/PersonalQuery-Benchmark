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
  result/07_gen_query/pool.json
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

# 2026-09-05 修复: 抽样域限定为 valid Gaussian 用户覆盖的 ASIN (≥2 valid users)
# 来源: result/05_gaussian_audit/asin_coverage_valid_ge2.json
ASIN_COVERAGE_PATH = REPO_ROOT / "result/05_gaussian_audit/asin_coverage_valid_ge2.json"
SFT_POOL_SOURCE = "gaussian_valid_ge2"  # 标注抽样源

QWEN_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
SFT_ADAPTER_DIR = REPO_ROOT / "result/06_training_model/sft_lora"
ATTRIBUTES_PATH = REPO_ROOT / "result/01_attribute_extraction/product_attributes.json"

# --- Hardcoded hyperparams (Rule 3) ---
SFT_POOL_SEED = 42
SFT_POOL_SMOKE = False        # True=5 ASIN smoke, False=100 ASIN full (Rule 20)
SFT_POOL_N_ASIN = 5 if SFT_POOL_SMOKE else 1000  # 2026-09-06: 扩到 1000 ASIN
SFT_POOL_K = 100              # 2026-09-06: 每个 ASIN 100 candidates
SFT_POOL_TEMP = 0.7
SFT_POOL_TOP_P = 0.95
SFT_POOL_MAX_NEW_TOKENS = 40
SFT_POOL_EXCLUDE_TRAIN = True # 排除 SFT 训练 ASIN (用户 query_samples_100.json 用过)
SFT_POOL_MAX_EXTRAS = 3       # 用户 2026-09-05: 容忍 "I'd"/"like" 填充词

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


def select_eval_asins(attrs_all: Dict[str, Dict[str, str]],
                      n_asin: int) -> List[str]:
    """加载 n_asin 个 ASIN, 排除 SFT 训练用过的.

    2026-09-05 新版: 抽样域限定为 valid Gaussian 用户 (E1E2E3E4 全 pass) 覆盖
    且 ≥2 valid users 的 ASIN, 保证下游 08 selection 每个 ASIN 都有多个
    candidate target user 可选. 按 valid uid 数量降序取前 n_asin (高 density
    优先, cohort competitor 更充足).
    """
    if not ASIN_COVERAGE_PATH.exists():
        raise FileNotFoundError(
            f"{ASIN_COVERAGE_PATH} 不存在, 请先跑 05_gaussian_audit/raw_cov_validity.py"
        )
    with open(ASIN_COVERAGE_PATH) as f:
        coverage = json.load(f)
    asin_to_valid = coverage["asin_to_valid_uids"]

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
        gpu_memory_utilization=0.55,  # 2026-09-06: qwen_hidden_server 已停, 提至 0.55
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
    from common.grpo_reward import check_content, tokenize

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
            content_pass = (len(miss) == 0 and len(rep) == 0
                            and len(extras) <= SFT_POOL_MAX_EXTRAS)
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
    log(f"  N_ASIN={SFT_POOL_N_ASIN}  K={SFT_POOL_K}  temp={SFT_POOL_TEMP} "
        f"top_p={SFT_POOL_TOP_P}  max_new_tokens={SFT_POOL_MAX_NEW_TOKENS}")

    attrs_all = load_attributes()
    eval_asins = select_eval_asins(attrs_all, SFT_POOL_N_ASIN)
    log(f"  eval_asins (n={len(eval_asins)}): {eval_asins[:5]} ...")

    attrs_list = [dict(list(attrs_all[a].items())[:5]) for a in eval_asins]
    cands_per_asin = generate_candidates(attrs_list, SFT_POOL_K)

    # --- Build pool_queries.json (精简版: asin -> [query, ...]) ---
    pool = {entry_asin: [c["text"] for c in cands]
            for entry_asin, cands in zip(eval_asins, cands_per_asin)}
    n_total = sum(len(v) for v in pool.values())

    out_path = OUT_DIR / "pool_queries.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "model": QWEN_PATH,
                "lora_adapter": str(SFT_ADAPTER_DIR),
                "n_asin": len(eval_asins),
                "k_per_asin": SFT_POOL_K,
                "temp": SFT_POOL_TEMP,
                "top_p": SFT_POOL_TOP_P,
                "max_new_tokens": SFT_POOL_MAX_NEW_TOKENS,
                "source": SFT_POOL_SOURCE,
                "coverage_path": str(ASIN_COVERAGE_PATH),
            },
            "pool": pool,
        }, f, indent=2, ensure_ascii=False)
    log(f"  saved -> {out_path} (n_total={n_total})")
    log(f"  SUMMARY: {n_total} queries, {len(pool)} ASINs")
    log("=== sft_pool_generate DONE ===")


if __name__ == "__main__":
    main_pipeline()
