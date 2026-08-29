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
  - `_syntax_subspace_prepare`: load 10k user sentence cache + scaler + user_to_indices
                                (only fields consumed by main pipeline)

Inputs (paths under `RESIDUAL_SCRATCH`):
  - sentences_for_rewrite_10k.jsonl
  - sentences_318d_cache.jsonl.gz

Outputs: dict with X / X_scaled / scaler / feature_names_ordered / train_idx /
user_to_indices / user_ids / asins (only fields consumed by main pipeline).
"""

from __future__ import annotations

import gzip
import hashlib as _hl
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

# Legacy alias kept for backwards-compatibility (used by _syntax_subspace_prepare)
# SCRATCH 仅用于输入数据 + 下游 stage 读取的中间 cache (per_query / selection intermediate)
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
MIN_REVIEWS_FOR_PER_USER = 1

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


def _syntax_subspace_prepare(
    sents_path: Path | None = None,
    feat_cache_path: Path | None = None,
    feature_names_ordered: list | None = None,
) -> dict[str, Any]:
    """Stage 1 / Stage 2 共用的数据准备 (加载 318d cache + 标准化 + 切分).

    参数:
      sents_path / feat_cache_path: 覆盖默认路径
      feature_names_ordered: 传入则强制使用该特征顺序 (保证与冻结 PCA 一致)

    返回 dict, 含:
      X (原始) / X_scaled / Y_probes / user_ids / asins / train_idx / scaler /
      rewrites_by_user_text / user_to_indices / user_id_list / pair_idx / raw_dist /
      top_asins / asin_to_label / label_y / leak_train / leak_test /
      probe_train_idx / probe_test_idx / PROBE_TARGETS / feature_names_ordered / rng
    """
    spacy_sents = Path(sents_path) if sents_path else RESIDUAL_SCRATCH / "sentences_for_rewrite_10k.jsonl"
    feat_cache = Path(feat_cache_path) if feat_cache_path else RESIDUAL_SCRATCH / "sentences_318d_cache.jsonl.gz"
    rewrites_path = RESIDUAL_SCRATCH / "rewrites_10k.jsonl"

    if not spacy_sents.exists():
        raise FileNotFoundError(f"missing: {spacy_sents}")
    if not feat_cache.exists():
        raise FileNotFoundError(f"missing: {feat_cache}")
    if not rewrites_path.exists():
        raise FileNotFoundError(f"missing: {rewrites_path}")

    # --- 1. Load feature cache ---
    log("[subspace] loading sentence feature cache...")
    feat_map: dict = {}
    with gzip.open(feat_cache, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            feat_map[rec["k"]] = rec["v"]

    # --- 2. Load sentences ---
    sents_raw = load_jsonl(spacy_sents)
    log(f"[subspace] {len(sents_raw)} sentences")

    PROBE_TARGETS = ["max_depth", "nest_max", "n_clause", "mean_dist", "depth_var", "n_tok"]

    # Build matrix + meta
    sents_meta: list = []
    feature_names_ordered_local: list = []
    for row in sents_raw:
        text = row.get("sentence_text", "")
        k = _hl.sha1(text.strip().lower().encode("utf-8")).hexdigest()
        feats = feat_map.get(k, {})
        if not feats:
            continue
        numeric = {n: float(v) for n, v in feats.items() if isinstance(v, (int, float))}
        if not feature_names_ordered_local:
            feature_names_ordered_local = sorted(numeric.keys())
        vec = np.array([numeric[n] for n in feature_names_ordered_local], dtype=np.float64)
        probes = [float(numeric.get(t, 0.0)) for t in PROBE_TARGETS]
        sents_meta.append({
            "user_id": row["user_id"],
            "asin": row.get("asin", ""),
            "text": text,
            "vec": vec,
            "probes": np.array(probes, dtype=np.float64),
        })
    log(f"[subspace] {len(sents_meta)} valid sentences")

    if feature_names_ordered is None:
        feature_names_ordered = feature_names_ordered_local

    X = np.stack([s["vec"] for s in sents_meta], axis=0)
    Y_probes = np.stack([s["probes"] for s in sents_meta], axis=0)
    user_ids = np.array([s["user_id"] for s in sents_meta])
    asins = np.array([s["asin"] for s in sents_meta])
    log(f"[subspace] X shape: {X.shape}, Y_probes shape: {Y_probes.shape}")

    # --- 3. StandardScaler fit on 5000 sentences ---
    rng = np.random.default_rng(42)
    train_idx = rng.choice(len(X), size=min(5000, len(X)), replace=False)
    scaler = StandardScaler()
    scaler.fit(X[train_idx])
    X_scaled = scaler.transform(X)
    log(f"[subspace] StandardScaler fitted on {len(train_idx)} sentences")

    # --- 4. Load rewrites → neutral sentence features ---
    rewrites_raw = load_jsonl(rewrites_path)
    log(f"[subspace] {len(rewrites_raw)} rewrites")

    rewrites_by_user_text: dict = defaultdict(dict)
    for r in rewrites_raw:
        uid = r["user_id"]
        orig = r["sentence_text"]
        rewrite_text = r["rewrite"]
        k = _hl.sha1(rewrite_text.strip().lower().encode("utf-8")).hexdigest()
        feats = feat_map.get(k)
        if feats is None:
            continue
        numeric = {n: float(v) for n, v in feats.items() if isinstance(v, (int, float))}
        vec = np.array([numeric[n] for n in feature_names_ordered], dtype=np.float64)
        rewrites_by_user_text[uid][orig] = {
            "vec": vec,
            "rewrite_text": rewrite_text,
        }
    log(f"[subspace] {sum(len(d) for d in rewrites_by_user_text.values())} rewrite features available")

    # --- 5. Per-user aggregation ---
    user_to_indices: dict = defaultdict(list)
    for i, uid in enumerate(user_ids):
        user_to_indices[uid].append(i)
    user_id_list = sorted(user_to_indices.keys())
    log(f"[subspace] {len(user_id_list)} unique users")

    # --- 6. Pair sample for syntax ρ ---
    N_PAIR_SAMPLE = 2000
    pair_idx = rng.choice(len(X_scaled), size=(N_PAIR_SAMPLE, 2), replace=True)
    pair_idx = pair_idx[pair_idx[:, 0] != pair_idx[:, 1]]
    pair_idx = pair_idx[:N_PAIR_SAMPLE]
    log(f"[subspace] pair sample: {pair_idx.shape}")
    raw_dist = np.linalg.norm(X_scaled[pair_idx[:, 0]] - X_scaled[pair_idx[:, 1]], axis=1)

    # --- 7. Top ASINs for content leakage probe ---
    asin_counts = Counter(asins)
    top_asins = [a for a, c in asin_counts.most_common(200) if c >= 50]
    log(f"[subspace] {len(top_asins)} asins with >=50 sents")
    asin_to_label = {a: i for i, a in enumerate(top_asins)}
    label_mask = np.array([a in asin_to_label for a in asins])
    label_y = np.array([asin_to_label[a] if a in asin_to_label else -1 for a in asins])
    leak_idx = np.where(label_mask)[0]
    rng.shuffle(leak_idx)
    leak_split = int(len(leak_idx) * 0.8)
    leak_train, leak_test = leak_idx[:leak_split], leak_idx[leak_split:]

    probe_split = int(len(train_idx) * 0.8)
    probe_train_idx = train_idx[:probe_split]
    probe_test_idx = train_idx[probe_split:]

    return {
        "X": X, "X_scaled": X_scaled, "Y_probes": Y_probes, "user_ids": user_ids, "asins": asins,
        "train_idx": train_idx, "scaler": scaler,
        "rewrites_by_user_text": rewrites_by_user_text,
        "user_to_indices": user_to_indices, "user_id_list": user_id_list,
        "pair_idx": pair_idx, "raw_dist": raw_dist,
        "top_asins": top_asins, "asin_to_label": asin_to_label, "label_y": label_y,
        "leak_train": leak_train, "leak_test": leak_test,
        "probe_train_idx": probe_train_idx, "probe_test_idx": probe_test_idx,
        "PROBE_TARGETS": PROBE_TARGETS, "feature_names_ordered": feature_names_ordered,
        "rng": rng,
    }