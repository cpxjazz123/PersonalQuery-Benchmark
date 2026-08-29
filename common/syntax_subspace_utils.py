"""Shared utilities for syntax subspace pipeline.

Centralized helpers + path constants used by the 4 sibling scripts:

  - gaussian/build_user.py                     (Phase 1 cohort + Phase 2 per-user Gaussian)
  - gen_query/syntax_subspace_pool_regen.py     (Stage 1 LLM pool generation)
  - select_query/syntax_subspace_select_strict_alignment.py  (Stage 2 features + Stage 4 strict alignment)
  - syntactic_evaluation/syntax_subspace_retrieval_unified.py  (Stage 5: 7-retriever + volatility)

Exposes:
  - Path constants: REPO_ROOT, RESULT, SCRATCH, ASINS_IN, REVIEW_GZ, META_FILE,
                    POOL_IN/OUT, FEAT_CACHE, GAUSSIANS_OUT, ASIN_TO_DOC_CACHE,
                    VLLM_URL, MODEL_NAME
  - Hyperparameters: PCA_DIM, PCA_SEED, LAMBDA, VAR_EPS, MIN_REVIEWS_FOR_PER_USER,
                    K_POOL, TEMP, MAX_TOKENS, MAX_QUERY_TOKENS, N_INPUT
  - `log`: timestamped log print
  - `feat_key`: sha1(text) feature cache key
  - `load_jsonl`: load JSONL as list[dict]
"""

from __future__ import annotations

import hashlib as _hl
import json
import time
from pathlib import Path

# SCRATCH only used for input data + intermediate cache (downstream stage reads)
RESIDUAL_SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")

# ---------------------------------------------------------------------------
# 路径常量 (全部 4 个兄弟脚本共享)
# ---------------------------------------------------------------------------
REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
RESULT = REPO_ROOT / "result"  # 最终聚合结果 (按 Rule 13/16 每个功能目录单一 JSON)
SCRATCH = RESIDUAL_SCRATCH  # 输入 + intermediate cache

# --- Inputs (只读) ---
ASINS_IN = SCRATCH / "stage8_5_asins.json"
REVIEW_GZ = REPO_ROOT / "data/Baby_Products_2023.jsonl.gz"
META_FILE = REPO_ROOT / "data/meta_Baby_Products_2023.jsonl.gz"

# --- Stage 1 (gen_query/) → result ---
POOL_OUT = RESULT / "gen_query/pool.json"
POOL_IN = POOL_OUT

# --- Stage 2 (intermediate cache) → select_query/ 目录下 (用户指令 2026-08-29)
# 用户指令 2026-08-29: 把 stage7b_query_features.jsonl.gz 从 scratch2 搬到
# select_query/ 下, 与 Stage 2 features + Stage 4 selection 同模块。
# Stage 3 Gaussian 通过 select_query.FEAT_CACHE 路径同步读取。
FEAT_CACHE = REPO_ROOT / "select_query" / "stage7b_query_features.jsonl.gz"

# --- Stage 3 (gaussian/) → scratch2 ---
# User instruction 2026-08-29: 移动到 scratch2/ 避免 git 100MB push limit
# (no-cohort 后 user_gaussians.json 涨到 2.9GB,远超 GitLab 100MiB 单文件上限)
GAUSSIANS_OUT = SCRATCH / "stage8_5_user_gaussians.json"

# --- Stage 5 corpus cache (Part A → Part B reuse) ---
ASIN_TO_DOC_CACHE = SCRATCH / "asin_to_doc.json"

VLLM_URL = "http://localhost:8800/v1/completions"
# 用户指令 2026-08-27: 切换到 Qwen2-1.5B-Instruct 加速 Stage 1 LLM 生成
# (~3-4× throughput vs 7B; query 质量略下降但 attrs 信息足够)
# 用户指令 2026-08-29: vLLM 当前 server 已加载 Qwen2-7B-Instruct (用户重启用 7B)
# (上次是 Qwen2.5-14B-Instruct, 2026-08-29 用户决策切回 7B)
MODEL_NAME = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"

# ---------------------------------------------------------------------------
# 共享超参数 (硬编码,所有脚本统一)
# ---------------------------------------------------------------------------
# PCA / Gaussian
PCA_DIM = 48
PCA_SEED = 2024
LAMBDA = 0.1
VAR_EPS = 1e-3
# 用户指令 2026-08-29: Gaussian 质量门控 = 3 项独立指标 (不显式使用评论数):
# Q1 non-degenerate: mean(sigma_diag) ≥ MIN_SIGMA_MEAN (去方差塌陷)
# Q2 non-floored:    r_floor = #{σ_d ≤ VAR_EPS}/48 ≤ MAX_R_FLOOR (去维度触底)
# Q3 self-consistent: inlier_frac = #{自己句子 d² ≤ R_95}/N ≥ MIN_INLIER_FRAC (Gaussian
#                    能合理解释自己历史)。这直接量化 "Gaussian 是不是可靠地描述该用户"。
# 用户可以只有 6 条 review 但 Gaussian 稳定可解释就保留; 反之 50 条 review 但 Gaussian
# 退化也丢弃。
MIN_REVIEWS_FOR_PER_USER = 1  # 仅用于排除 0-review 用户, 不作为质量信号
MIN_SIGMA_MEAN = 0.01       # Q1: 去方差塌陷
MAX_R_FLOOR = 0.3           # Q2: 触底维度比例 ≤ 30%
R_95 = 8.073                # √χ²(0.95, 48) — 用于 Q3 self-consistency 判断
MIN_INLIER_FRAC = 0.5       # Q3: ≥50% 自己历史落在 G_u 内
# 用户指令 2026-08-29: ASIN 必须有 ≥MIN_N_QUALITY_USERS_PER_ASIN 个通过全部 3 项
# Gaussian 质量门控的用户 (与评论数无关)。
MIN_N_QUALITY_USERS_PER_ASIN = 2  # 用户指令 2026-08-29: cohort ASIN 准入门槛降到 2 个 quality user
N_SENTENCES_THRESHOLDS = (5, 10, 20)  # 敏感性分析 sweep (用于 n_u 单变量 ablate, 不是主 filter)

# Query generation (Stage 1)
K_POOL = 50  # 用户指令 2026-08-29: 从 K=200 降到 K=50, 为剩余 4055 ASINs 补全 pool (vLLM batched 202K calls ~10-15min)
# 用户指令 2026-08-28: 0.7 → 0.5,降低 LLM 跑偏概率(strict rate ↑5-10pp,质量更确定)
TEMP = 0.5
# 用户指令 2026-08-27: first-person query 需要更多 tokens (10 attrs + "I'm looking for..." +
# 各种 paraphrase),从 80 提到 120 给足 buffer
MAX_TOKENS = 120

# 用户指令 2026-08-28: 输出 query token 数 ≤60 的严格过滤(N=5 + 1st-person prompt 下,模型偶尔会失控
# 写出 80+ token 的长尾自言自语/重复,污染 strict 池)
MAX_QUERY_TOKENS = 60

# 用户指令 2026-08-28: N=5 是生成甜点(N=4 太薄,N=7/10 模型失控)。
# Stage 1 从 product_attributes.json 里挑 top-N attrs 给 LLM
N_INPUT = 5


# ---------------------------------------------------------------------------
# 共享工具函数
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    """Timestamped log print."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def feat_key(t: str) -> str:
    """SHA-1 key used to dedupe syntactic features by text."""
    return _hl.sha1(t.strip().lower().encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file as a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]