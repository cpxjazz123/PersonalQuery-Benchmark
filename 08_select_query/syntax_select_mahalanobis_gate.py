#!/usr/bin/env python3
"""Query selection: per-user Mahalanobis D² + validation-quantile gate.

Refactor 2026-09-06: per-user Gaussian 与 per-(asin, uid) cohort membership
由 Stage 04 的单一 canonical artifact 提供，08 只做 candidate query 编码、
选择判定与输出，不在运行时重新拟合 Gaussian。

流程:
  1. 加载 Stage 05 audit 的 E1–E4 valid_gaussian 用户与 ASIN coverage
  2. 从 Stage 04 读取这些有效用户对应的 Gaussian 参数与 gate_T
  3. 加载 frozen encoder + vocab, 编码候选 pool queries → 32d z_q
  4. 对每个 (asin, uid) task:
     a. D²(z_q, μ_u) ≤ u.d2_q95 (gate_T)   ← target 核心内
     b. ∀comp ∈ cohort: D²(z_q, μ_comp) > comp.d2_q95 ← 核心外
  5. 通过者中选 D² 最小
  6. 每 ASIN 给 Stage 05 valid cohort 中的用户各选 1 query, ≥2 unique user 才保留

复用资产:
  - 冻结 encoder: pcfg_cache/adaptive_encoder.pt (_SupEncoder 21737→256→32, eval)
  - vocab:        pcfg_cache/vocab.json (21737 规则)
  - Stage 05 audit: result/05_gaussian_audit/raw_cov_validity.json
  - Stage 05 coverage: result/05_gaussian_audit/asin_coverage_valid_ge2.json
  - Stage 04 Gaussian parameters: result/04_gaussian/user_gaussian_stats.json
  - pool queries: result/07_gen_query/pool_queries.json

用法 (Rule 3: 无参数):
  cd /home/wlia0047/ar57/wenyu/PersoanlQuery
  $PY 08_select_query/syntax_select_mahalanobis_gate.py

输出: result/08_select_query/selected_queries.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.linalg
from collections import defaultdict, Counter
import spacy
import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))

CACHE_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache")
POOL_PATH = REPO_ROOT / "result/07_gen_query/pool_queries.json"
ATTRS_PATH = REPO_ROOT / "result/01_attribute_extraction/product_attributes.json"
OUT_PATH = REPO_ROOT / "result/08_select_query/selected_queries.json"
SMOKE_OUT_PATH = Path(
    "/home/wlia0047/hj82_scratch2/wenyu/stage08_select_smoke.json"
)
# Stage 05 定义 E1–E4 valid_gaussian 用户与可用 ASIN cohort；Stage 04
# 仅提供这些有效用户的 Gaussian 参数。
STAGE04_PATH = REPO_ROOT / "result/04_gaussian/user_gaussian_stats.json"
STAGE05_AUDIT_PATH = REPO_ROOT / "result/05_gaussian_audit/raw_cov_validity.json"
STAGE05_COVERAGE_PATH = (
    REPO_ROOT / "result/05_gaussian_audit/asin_coverage_valid_ge2.json"
)
USER_STATS_PATH = STAGE04_PATH
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# CONTRASTIVE_5558 ABLATION: 5558 cohort, NT-Xent contrastive encoder + 5558 stats
CONTRASTIVE_5558 = False
if CONTRASTIVE_5558:
    CACHE_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache_5558")
    STAGE04_PATH = REPO_ROOT / "result/04_gaussian/user_gaussian_stats_5558.json"
    USER_STATS_PATH = STAGE04_PATH
    OUT_PATH = REPO_ROOT / "result/08_select_query/selected_queries_5558.json"
    SMOKE_OUT_PATH = Path(
        "/home/wlia0047/hj82_scratch2/wenyu/stage08_select_smoke_5558.json"
    )

# --- Hardcoded hyperparams (Rule 3) ---
FILTER_Q = 0.05
SEMANTIC_SIM_THRESHOLD = 0.85  # 2026-09-15: top-K 模式下同 (asin, uid) 多 query 之间
                                    # cos 通常 0.85-0.95; 用 0.85 保留 block 让更多 ASIN
                                    # 进入输出 (单 query ASIN 自动保留, n>=2 时按阈值过滤)
MIN_PROFILE_SENTS = 40
MIN_VAL_SENTS = 10
SMOKE = False
N_SMOKE_ASIN = 5
SPACY_MODEL = "en_core_web_sm"
SPACY_DISABLE = ["ner", "textcat", "lemmatizer"]

COMPETITOR_GATE_QUANTILE = FILTER_Q
BS_N_BOOTSTRAP = 29
BS_FRAC = 0.5

# USE_LIKELIHOOD_MARGIN = False: pure Mahalanobis D² gate (no OVL logic)
OVL_MAX = 1.0           # disabled (OVL filtering off)
OVL_PATH = REPO_ROOT / "result/08_select_query/gaussian_ovl_matrix.json"
USE_LIKELIHOOD_MARGIN = False
BS_P_THRESH = 0.95
BS_SEED = 42
USE_SINGLE_FIT = True    # single_fit_unique: 一次性固定高斯, 无 bootstrap
ENCODE_DEVICE = "cuda:0"
ENCODE_CHUNK_SIZE = 1024
SPACY_BATCH_SIZE = 256
MINILM_BATCH_SIZE = 512
# 2026-09-15: 每个 (asin, uid) task 保留的 query 数量上限（按 D² 升序）。
# 原逻辑: 每个 task 只选 D² 最小的 1 个 query，导致同一 ASIN 上多个 cohort 用户中
# 只有 1 个用户通过 gate 时, ASIN 仅 1 个 query。
# 新逻辑: 保留每个 task 的 top-K query（unique_pass & pass_gate），让同一 ASIN 上
# 多用户的多 query 累积; ASIN 聚合后用 max_pair_cos < SEMANTIC_SIM_THRESHOLD 控制去重。
# K=1 退化回原行为; K>=3 可显著提高 ASIN 上 unique query >=2 的覆盖率。
TOP_K_PER_USER = 3
# 2026-09-15: SWEEP 模式 — 空列表 = 正常单 q 运行; 非空 = 跑 sweep, 只输出每个 q 的
# query 数统计 (不写 selected_queries.json, 不跑 MiniLM sim filter)。Sweep 共享一次
# encoder 调用, 对每个 q 重算 selection。
SWEEP_Q_VALUES = []  # 2026-09-16: disabled; main pipeline uses FILTER_Q=0.95
# 2026-09-15: SWEEP_GATE_STRATEGY controls which gate_T source to use.
#   "theoretical"  → d2_qNN_theoretical (χ²(d=16, q) ppf, NOT data-dependent)
#   "empirical"    → d2_qNN (percentile of D² over val sentences, data-dependent)
# Empirical stats only stored at q=0.50, 0.75, 0.95 (no q=0.05/0.10/0.20 in stage 04),
# so empirical sweep will skip those q values.
SWEEP_GATE_STRATEGY = "empirical"  # set to "theoretical" for chi-square gate
# 2026-09-16: SWEEP_OVL_MAX filters out high-OVL users (cohort-overlap) before
# selection. Users with mean_ovl >= SWEEP_OVL_MAX are excluded from BOTH target
# and competitor positions. None = no OVL filter (all users included).
# Cohort OVL matrix at OVL_PATH only covers ~3000 users; users without OVL
# data are KEPT (treated as OVL=0 = unique style).
SWEEP_OVL_MAX = 0.5  # exclude users with mean_ovl >= 0.5
SWEEP_OUT_PATH = REPO_ROOT / "result/08_select_query/q_sweep_results.json"  # suffix added at write time based on SWEEP_GATE_STRATEGY
D2_GATE_ABS_TOL = 0.25
D2_GATE_REL_TOL = 1e-3
D2_BOUNDARY_RECHECK_TOL = 0.25


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def require_cuda() -> None:
    """要求 CUDA；禁止在性能关键路径静默回退 CPU。"""
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 08 requires CUDA for batched encoder inference")
    device = torch.device(ENCODE_DEVICE)
    if device.type != "cuda" or device.index != 0:
        raise ValueError(f"ENCODE_DEVICE must be cuda:0, got {ENCODE_DEVICE}")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA device 0 is unavailable")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    log(f"  CUDA device: {torch.cuda.get_device_name(0)}")


# ============================================================================
# 03_spacy_encode 模块加载 (_SupEncoder / extract_struct_rules)
# ============================================================================

_pcfg = None


def _load_pcfg():
    global _pcfg
    if _pcfg is None:
        spec = importlib.util.spec_from_file_location(
            "syntax_pcfg_pipeline",
            REPO_ROOT / "03_spacy_encode/syntax_pcfg_pipeline.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _pcfg = mod
    return _pcfg


# ============================================================================
# SWEEP helpers (2026-09-15)
# ============================================================================
def _q_to_gate_keys(q: float) -> tuple[str, str]:
    """Map q value → (user_gate_key, cohort_gate_key) for Stage 04 stats.

    Strategy is selected by SWEEP_GATE_STRATEGY (module-level):
      "theoretical" → χ²(d=16, q) ppf, NOT data-dependent.
                       Stage 04 cohort stats provide gate_T_low_theoretical (q=0.05)
                       and gate_T_high_theoretical (q=0.95); intermediate q falls
                       back to user-level d2_qNN_theoretical for both target and
                       competitor.
      "empirical"   → d2_qNN from val-sentence percentile (DATA-DEPENDENT).
                       Stage 04 only stores empirical quantiles at q=0.50, 0.75, 0.95.
                       For q not in {0.50, 0.75, 0.95}, raises ValueError.
                       Cohort gate_T is not stored empirically, so falls back to
                       user-level d2_qNN.
    """
    if SWEEP_GATE_STRATEGY == "theoretical":
        if abs(q - 0.05) < 1e-6:
            return "d2_q05_theoretical", "gate_T_low_theoretical"
        if abs(q - 0.95) < 1e-6:
            return "d2_q95_theoretical", "gate_T_high_theoretical"
        if abs(q - 0.50) < 1e-6:
            return "d2_q50_theoretical", "user_fallback"
        if abs(q - 0.75) < 1e-6:
            return "d2_q75_theoretical", "user_fallback"
        snaps = [(0.05, "d2_q05_theoretical", "gate_T_low_theoretical"),
                 (0.50, "d2_q50_theoretical", "user_fallback"),
                 (0.75, "d2_q75_theoretical", "user_fallback"),
                 (0.95, "d2_q95_theoretical", "gate_T_high_theoretical")]
        nearest = min(snaps, key=lambda s: abs(s[0] - q))
        return nearest[1], nearest[2]
    if SWEEP_GATE_STRATEGY == "empirical":
        if abs(q - 0.50) < 1e-6:
            return "d2_q50", "user_fallback"
        if abs(q - 0.75) < 1e-6:
            return "d2_q75", "user_fallback"
        if abs(q - 0.95) < 1e-6:
            return "d2_q95", "user_fallback"
        raise ValueError(
            f"empirical gate_T only stored for q in {{0.50, 0.75, 0.95}}; "
            f"got q={q}. Either set SWEEP_Q_VALUES to only those values, or "
            f"use SWEEP_GATE_STRATEGY='theoretical'.")
    raise ValueError(f"unknown SWEEP_GATE_STRATEGY={SWEEP_GATE_STRATEGY!r}")


def _gauss_from_stats_q(stats: Dict, user_key: str) -> Dict:
    """Mirror of _gauss_from_stats but reads gate_T from user_key."""
    required = ("mu", "sigma_inv", "n", "n_val", "d2_q50", user_key)
    missing = [k for k in required if k not in stats]
    if missing:
        raise ValueError(f"Stage 04 user stats missing fields: {missing}")
    mu = np.asarray(stats["mu"], dtype=np.float32)
    inv_sigma = np.asarray(stats["sigma_inv"], dtype=np.float32)
    if mu.ndim != 1 or mu.shape[0] not in (16, 32):
        raise ValueError(f"invalid Gaussian mu shape: {mu.shape}")
    if inv_sigma.ndim != 2 or inv_sigma.shape[0] != inv_sigma.shape[1] or inv_sigma.shape[0] not in (16, 32):
        raise ValueError(f"invalid Gaussian sigma_inv shape: {inv_sigma.shape}")
    gate_T = float(stats[user_key])
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(inv_sigma)):
        raise FloatingPointError("non-finite Stage 04 Gaussian parameters")
    if not np.isfinite(gate_T) or gate_T < 0:
        raise ValueError(f"invalid Stage 04 gate_T={gate_T}")
    return {
        "mu": mu,
        "inv_sigma": inv_sigma,
        "gate_T": gate_T,
        "d2_val_median": float(stats["d2_q50"]),
        "gate_quantile": user_key,
        "n": int(stats["n"]),
        "n_val": int(stats["n_val"]),
    }


def _build_cohort_gates_q(stage04: Dict, user_key: str, cohort_key: str,
                          valid_uids: set) -> Dict:
    """Build per-ASIN cohort gates with gate_T read from cohort_key.

    For cohort_key == 'user_fallback', gate_T is read from each user's
    d2_qNN_theoretical field (using user_key). Otherwise reads cohort gate_T_*.
    """
    cohort_gates: Dict[str, Dict[str, Dict]] = {}
    stage04_cohorts = stage04.get("cohort_gates", {})
    for asin, source_cohort in stage04_cohorts.items():
        fitted_uids = [uid for uid in source_cohort.keys() if uid in valid_uids]
        if len(fitted_uids) < 2:
            continue
        cohort_gates[asin] = {}
        for uid in fitted_uids:
            gate = source_cohort[uid]
            if cohort_key == "user_fallback":
                user_stats = stage04["users"][uid]
                if user_key not in user_stats:
                    raise ValueError(
                        f"Stage 04 user {uid} missing {user_key}")
                gate_T_val = float(user_stats[user_key])
            else:
                if not isinstance(gate, dict) or cohort_key not in gate:
                    raise ValueError(
                        f"Stage 04 cohort {asin}/{uid} missing {cohort_key}")
                gate_T_val = float(gate[cohort_key])
            if not np.isfinite(gate_T_val):
                raise ValueError(
                    f"Stage 04 cohort {asin}/{uid} gate_T not finite")
            cohort_gates[asin][uid] = {
                "gate_T": gate_T_val,
                "n_profile": gate.get("n_profile"),
                "n_val": gate.get("n_val"),
            }
    return cohort_gates


def _select_for_q(asin_to_Zq: Dict[str, np.ndarray],
                  gauss_cache: Dict[str, Dict],
                  cohort_gates_map: Dict, q: float) -> Dict:
    """Run vectorized selection loop for a single q value."""
    asin_work: Dict[str, Dict] = {}
    tasks: List[Dict] = []
    for asin, Z_q in asin_to_Zq.items():
        cohort = cohort_gates_map.get(asin)
        if cohort is None:
            continue
        all_uids = [uid for uid in sorted(cohort) if uid in gauss_cache]
        if len(all_uids) < 2:
            continue
        asin_work[asin] = {"asin": asin, "uids": all_uids,
                           "cohort_uids": all_uids, "cohort": cohort}
        for uid in all_uids:
            tasks.append({"asin": asin, "uid": uid})
    log(f"  (q={q}) tasks={len(tasks)} asins_with_cohort={len(asin_work)}")

    selections: List[Dict] = []
    no_pass = 0
    for asin, work in asin_work.items():
        Z_q = asin_to_Zq[asin]
        cohort_uids = work["cohort_uids"]
        mu = np.stack([gauss_cache[uid]["mu"] for uid in cohort_uids], axis=0)
        inv_sigma = np.stack([gauss_cache[uid]["inv_sigma"] for uid in cohort_uids], axis=0)
        gate_Ts = np.asarray([float(work["cohort"][uid]["gate_T"]) for uid in cohort_uids], dtype=np.float64)
        diff = Z_q[:, None, :] - mu[None, :, :]
        left = np.einsum("cud,ude->cue", diff, inv_sigma)
        d2_matrix = np.sum(left * diff, axis=2, dtype=np.float32)
        del diff, left
        target_index = {uid: i for i, uid in enumerate(cohort_uids)}
        for idx, uid in enumerate(cohort_uids):
            target_i = target_index[uid]
            target_d2 = d2_matrix[:, target_i]
            gate_T = float(gate_Ts[target_i])
            target_inside = target_d2 <= gate_T
            competitor_inside = d2_matrix <= gate_Ts[None, :]
            competitor_inside[:, target_i] = False
            pass_unique_mask = target_inside & (~competitor_inside.any(axis=1))
            n_unique = int(pass_unique_mask.sum())
            if n_unique > 0:
                selections.append({"asin": asin, "uid": uid, "n_unique": n_unique})
            else:
                no_pass += 1

    # Aggregate per ASIN (top-K per user, expanded to flat entries)
    asin_to_user_counts: Dict[str, Counter] = defaultdict(Counter)
    for s in selections:
        n_contrib = min(TOP_K_PER_USER, s["n_unique"])
        asin_to_user_counts[s["asin"]][s["uid"]] += n_contrib
    asin_to_total_q = {a: sum(c.values()) for a, c in asin_to_user_counts.items()}
    asin_to_unique_q = {a: len(c) for a, c in asin_to_user_counts.items()}

    n_asins = len(asin_to_unique_q)
    n_unique_q = list(asin_to_unique_q.values())
    n_total_q = list(asin_to_total_q.values())
    user_q_dist = Counter(n_unique_q)
    total_q_dist = Counter(n_total_q)
    return {
        "q": q,
        "user_gate_key": _q_to_gate_keys(q)[0],
        "cohort_gate_key": _q_to_gate_keys(q)[1],
        "n_tasks": len(tasks),
        "n_selections": len(selections),
        "n_no_pass_tasks": no_pass,
        "n_asins_with_at_least_1_query": n_asins,
        "n_total_queries_after_topk": sum(n_total_q),
        "n_unique_user_queries_per_asin_dist": dict(sorted(user_q_dist.items())),
        "n_total_queries_per_asin_dist": dict(sorted(total_q_dist.items())),
        "n_asins_ge1_unique_queries": sum(1 for n in n_unique_q if n >= 1),
        "n_asins_ge2_unique_queries": sum(1 for n in n_unique_q if n >= 2),
        "n_asins_ge3_unique_queries": sum(1 for n in n_unique_q if n >= 3),
        "note": "sweep counts raw (no MiniLM semantic_sim filter applied); "
                "filter only removes blocks where max_pair_cos>=0.85; effect "
                "minor since top-K queries come from distinct (uid,query_text)",
    }


def _run_sweep(asin_to_Zq: Dict[str, np.ndarray],
               stage04: Dict, users_all: Dict,
               valid_uids: set) -> None:
    """Run selection for each q in SWEEP_Q_VALUES and save summary."""
    # Load OVL data if filtering enabled
    user_ovl_filter: Dict[str, float] = {}
    if SWEEP_OVL_MAX is not None:
        if not OVL_PATH.exists():
            raise FileNotFoundError(
                f"SWEEP_OVL_MAX={SWEEP_OVL_MAX} requires {OVL_PATH}")
        with open(OVL_PATH) as f:
            _ovl_data = json.load(f)
        user_ovl_filter = _ovl_data.get("per_user_mean_ovl", {})
        log(f"  OVL filter: {len(user_ovl_filter)} users with OVL data, threshold={SWEEP_OVL_MAX}")
        log(f"    users WITHOUT OVL data (kept): {len(valid_uids - set(user_ovl_filter.keys()))}")
        log(f"    users WITH OVL data, kept (ovl < {SWEEP_OVL_MAX}): "
            f"{sum(1 for u in valid_uids if u in user_ovl_filter and user_ovl_filter[u] < SWEEP_OVL_MAX)}")
        log(f"    users EXCLUDED (ovl >= {SWEEP_OVL_MAX}): "
            f"{sum(1 for u in valid_uids if u in user_ovl_filter and user_ovl_filter[u] >= SWEEP_OVL_MAX)}")
    """Run selection for each q in SWEEP_Q_VALUES and save summary."""
    sweep_results = []
    for q in SWEEP_Q_VALUES:
        user_key, cohort_key = _q_to_gate_keys(q)
        log(f"--- q={q} (user={user_key}, cohort={cohort_key}) ---")
        gauss_cache: Dict[str, Dict] = {}
        n_excluded_ovl = 0
        for uid in valid_uids:
            # Apply OVL filter: drop users with high mean OVL
            if SWEEP_OVL_MAX is not None and uid in user_ovl_filter:
                if user_ovl_filter[uid] >= SWEEP_OVL_MAX:
                    n_excluded_ovl += 1
                    continue
            stats = users_all[uid]
            try:
                gauss_cache[uid] = _gauss_from_stats_q(stats, user_key)
            except (ValueError, FloatingPointError) as e:
                continue
        log(f"  gauss_cache: {len(gauss_cache)} users (OVL-excluded: {n_excluded_ovl})")
        cohort_gates_map = _build_cohort_gates_q(
            stage04, user_key, cohort_key, set(gauss_cache.keys())
        )
        log(f"  cohort_gates (after OVL filter): {len(cohort_gates_map)} ASINs")
        result = _select_for_q(asin_to_Zq, gauss_cache, cohort_gates_map, q)
        sweep_results.append(result)
        log(f"  → n_sel={result['n_selections']} n_asins={result['n_asins_with_at_least_1_query']} "
            f"ge2={result['n_asins_ge2_unique_queries']} ge3={result['n_asins_ge3_unique_queries']}")

    out = {
        "config": {
            "top_k_per_user": TOP_K_PER_USER,
            "semantic_sim_threshold": SEMANTIC_SIM_THRESHOLD,
            "q_values": SWEEP_Q_VALUES,
            "gate_strategy": SWEEP_GATE_STRATEGY,
            "ovl_max": SWEEP_OVL_MAX,
            "stage04_source": str(STAGE04_PATH),
            "pool_path": str(POOL_PATH),
            "n_pool_asins": len(asin_to_Zq),
            "n_valid_users": len(valid_uids),
            "note": "Sweep counts without MiniLM semantic_sim filter; "
                    "top-K queries per user come from distinct (uid,query) "
                    "pairs so filter impact is minor.",
        },
        "sweep": sweep_results,
    }
    # strategy + ovl-suffixed output to avoid overwriting
    ovl_suffix = f"_ovl{int(SWEEP_OVL_MAX * 100):02d}" if SWEEP_OVL_MAX is not None else "_ovlNone"
    out_path = SWEEP_OUT_PATH.with_name(
        SWEEP_OUT_PATH.stem + f"_{SWEEP_GATE_STRATEGY}" + ovl_suffix + SWEEP_OUT_PATH.suffix
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"=== sweep done, saved -> {out_path} ===")
    # Pretty table
    print()
    print(f"{'q':>6} {'user_key':<26} {'gate_T':>8} {'n_sel':>8} "
          f"{'n_asins':>8} {'ge1':>5} {'ge2':>5} {'ge3':>5} "
          f"{'uq_dist':<30} {'tq_dist':<30}")
    sample_uid = next(iter(valid_uids)) if valid_uids else None
    sample_stats = users_all.get(sample_uid, {}) if sample_uid else {}
    for r in sweep_results:
        uk = r["user_gate_key"]
        gate_T = sample_stats.get(uk, "n/a") if sample_uid else "n/a"
        print(f"{r['q']:>6.2f} {uk:<26} {gate_T:>8.3f} "
              f"{r['n_selections']:>8d} "
              f"{r['n_asins_with_at_least_1_query']:>8d} "
              f"{r['n_asins_ge1_unique_queries']:>5d} "
              f"{r['n_asins_ge2_unique_queries']:>5d} "
              f"{r['n_asins_ge3_unique_queries']:>5d} "
              f"{str(r['n_unique_user_queries_per_asin_dist']):<30} "
              f"{str(r['n_total_queries_per_asin_dist']):<30}")


# ============================================================================
# 冻结 encoder 加载 (用于 candidate query 编码)
# ============================================================================

def load_encoder() -> torch.nn.Module:
    # Stage 04 fit_per_user_gaussian 读 strict3_embeddings.npz, 故 Stage 08
    # 也必须用同一 encoder (strict3_encoder.pt)。CLAUDE.md Rule 14:
    # canonical encoder 严格训练一次, strict3_encoder.pt 是单一来源。
    encoder_path = CACHE_DIR / "strict3_encoder.pt"
    if not encoder_path.exists():
        raise FileNotFoundError(
            f"missing canonical encoder: {encoder_path} "
            "(strict3_encoder.pt 是 Stage 04/08 唯一合法 encoder 来源)"
        )
    ckpt = torch.load(encoder_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    _SupEncoder = _load_pcfg()._SupEncoder
    model = _SupEncoder(
        cfg["vocab_size"], cfg["z_dim"], tuple(cfg["hidden"]),
        cfg["n_users"], cfg["dropout"]
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(torch.device(ENCODE_DEVICE))
    model.eval()
    log(f"  encoder loaded: vocab={cfg['vocab_size']} z={cfg['z_dim']} "
        f"hidden={cfg['hidden']} n_users={cfg['n_users']} device={ENCODE_DEVICE}")
    return model


# ============================================================================
# Stage 04 产物加载 (per-user Gaussian + cohort gates)
# ============================================================================

def load_stage04() -> tuple[Dict[str, Dict], Dict[str, Dict[str, Dict]], dict]:
    """加载 Stage 04 canonical Gaussian (full Σ)，以 Stage 04 自带 Gaussian 完整性
    作为 valid 凭证 (Stage 04 fit_one_user 已做 MIN_PROFILE / E2a finite / E2b
    beats_diag / Cholesky invert 全部过滤才写入 users_full)。

    Stage 05 audit 在 64996 全量用户上跑 (用户空间不同于 Stage 04 的 2155
    h≥30 cohort),valid_uids 跟 Stage 04 Gaussian 不重合。Stage 08 直接消费
    Stage 04 的 Gaussian artifact,不再以 Stage 05 audit valid_gaussian 标记
    为 ground truth。
    """
    for path in (STAGE04_PATH,):
        if not path.exists():
            raise FileNotFoundError(f"missing required Stage 04 artifact: {path}")
    with open(STAGE04_PATH) as f:
        stage04 = json.load(f)
    if not isinstance(stage04, dict):
        raise ValueError("Stage 04 artifact must be a JSON object")

    users_all = stage04.get("users")
    # 2026-09-15: 严格只读嵌入在 user_gaussian_stats.json 里的 cohort_gates
    # (gate_T = d2_q95)。独立 result/04_gaussian/cohort_gates_q*.json 是
    # pre-restructure 旧产物 (gate_T = 真实 d2_qNN), 跟 d2_q95 语义不兼容, 不读。
    config = stage04.get("config")
    stage04_cohorts = stage04.get("cohort_gates")
    if not isinstance(stage04_cohorts, dict) or not stage04_cohorts:
        raise ValueError(
            "Stage 04 artifact missing embedded 'cohort_gates' — "
            "rerun 04_gaussian/fit_per_user_gaussian.py to regenerate")
    if not isinstance(users_all, dict) or not users_all:
        raise ValueError("Stage 04 artifact requires non-empty 'users'")
    if not isinstance(stage04_cohorts, dict) or not stage04_cohorts:
        raise ValueError("Stage 04 artifact requires non-empty 'cohort_gates'")
    # Stage 04 Gaussian 完整性 = sigma_inv 存在 ∧ mu/sigma_inv/d2_q95 全部 finite
    valid_uids: set = set()
    for uid, stats in users_all.items():
        if not isinstance(stats, dict):
            continue
        if "sigma_inv" not in stats:
            continue
        mu = stats.get("mu")
        sigma_inv = stats.get("sigma_inv")
        gate_t = stats.get("d2_q05_theoretical")  # 2026-09-15: 理论 χ²(d, 0.05)
        if not isinstance(mu, list) or len(mu) not in (16, 32):
            continue
        if not isinstance(sigma_inv, list) or len(sigma_inv) not in (16, 32):
            continue
        if not all(isinstance(row, list) and len(row) in (16, 32) for row in sigma_inv):
            continue
        flat = [float(x) for row in sigma_inv for x in row]
        if not all(np.isfinite(x) for x in flat):
            continue
        if not all(np.isfinite(float(x)) for x in mu):
            continue
        if gate_t is None or not np.isfinite(float(gate_t)) or float(gate_t) < 0:
            continue
        valid_uids.add(uid)

    users = {uid: users_all[uid] for uid in sorted(valid_uids)}
    if not users:
        raise RuntimeError("Stage 04 produced no users with complete finite Gaussian")

    cohort_gates: Dict[str, Dict[str, Dict]] = {}
    for asin, source_cohort in stage04_cohorts.items():
        if not isinstance(source_cohort, dict):
            raise ValueError(f"Stage 04 cohort_gates[{asin}] must be an object")
        fitted_uids = [uid for uid in source_cohort.keys() if uid in valid_uids]
        if len(fitted_uids) < 2:
            continue
        cohort_gates[asin] = {}
        for uid in fitted_uids:
            gate = source_cohort[uid]
            # 2026-09-15: 严格只读 gate_T_low_theoretical (理论 χ²(d, 0.05), 低侧 rejection)
            if not isinstance(gate, dict) or "gate_T_low_theoretical" not in gate:
                raise ValueError(
                    f"Stage 04 cohort {asin}/{uid} missing gate_T_low_theoretical — "
                    "rerun 04_gaussian/rewrite_gaussian_with_theoretical_gate.py")
            if not np.isfinite(float(gate["gate_T_low_theoretical"])):
                raise ValueError(
                    f"Stage 04 cohort {asin}/{uid} gate_T_low_theoretical not finite")
            gate = dict(gate)
            gate["gate_T"] = float(gate["gate_T_low_theoretical"])
            cohort_gates[asin][uid] = gate

    log(f"  loaded Stage 04 Gaussian: {len(users_all)} users, "
        f"{len(valid_uids)} with complete finite full-Σ Gaussian")
    log(f"  Stage 04 cohort_gates: {len(stage04_cohorts)} ASINs total, "
        f"{len(cohort_gates)} ASINs with >=2 valid users")
    log(f"  cohort pairs: {sum(len(c) for c in cohort_gates.values())}")
    config = dict(config)
    config["stage04_source"] = str(STAGE04_PATH)
    config["stage04_valid_users"] = len(users)
    config["stage04_valid_asins"] = len(cohort_gates)
    return users, cohort_gates, config


def _gauss_from_stats(stats: Dict) -> Dict:
    """把 Stage 04 user stats 转成 selection 所需的 numpy 结构。

    2026-09-15: gate_T 用理论 χ²(d, q=0.05) (来自 rewrite_gaussian_with_theoretical_gate.py
    写入的 d2_q05_theoretical 字段), 不依赖 val 句子的经验 percentile。
    含义 (FILTER_Q=0.05): q=0.05 → gate_T = 7.96 → D²(z_q, μ_u) ≤ 7.96 才算"核心内"。
    含义 (FILTER_Q=0.95): q=0.95 → gate_T = d2_q95 (empirical val-sentence 95th pct D²)。
    """
    if FILTER_Q == 0.05:
        q_key = "d2_q05_theoretical"
    elif FILTER_Q == 0.95:
        q_key = "d2_q95"
    else:
        raise ValueError(f"FILTER_Q={FILTER_Q} not supported in _gauss_from_stats")
    required = ("mu", "sigma_inv", "n", "n_val", "d2_q50", q_key)
    missing = [key for key in required if key not in stats]
    if missing:
        raise ValueError(f"Stage 04 user stats missing fields: {missing}")
    mu = np.asarray(stats["mu"], dtype=np.float32)
    inv_sigma = np.asarray(stats["sigma_inv"], dtype=np.float32)
    if mu.ndim != 1 or mu.shape[0] not in (16, 32):
        raise ValueError(f"invalid Gaussian mu shape: {mu.shape}")
    if inv_sigma.ndim != 2 or inv_sigma.shape[0] != inv_sigma.shape[1] or inv_sigma.shape[0] not in (16, 32):
        raise ValueError(f"invalid Gaussian sigma_inv shape: {inv_sigma.shape}")
    gate_T = float(stats[q_key])
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(inv_sigma)):
        raise FloatingPointError("non-finite Stage 04 Gaussian parameters")
    if not np.isfinite(gate_T) or gate_T < 0:
        raise ValueError(f"invalid Stage 04 gate_T={gate_T}")
    return {
        "mu": mu,
        "inv_sigma": inv_sigma,
        "gate_T": gate_T,
        "d2_val_median": float(stats["d2_q50"]),
        "gate_quantile": q_key,
        "n": int(stats["n"]),
        "n_val": int(stats["n_val"]),
    }


# ============================================================================
# candidate query 编码
# ============================================================================

def encode_texts(texts: List[str], nlp, rule_to_id: Dict[str, int],
                 vocab_size: int, encoder: torch.nn.Module) -> np.ndarray:
    """分块提取规则并编码，避免构造全量 ``n×vocab`` 矩阵。"""
    extract_struct_rules = _load_pcfg().extract_struct_rules
    if not texts:
        return np.empty((0, encoder.z_dim if hasattr(encoder, 'z_dim') else 16), dtype=np.float32)
    encoded_chunks: List[np.ndarray] = []
    total = len(texts)
    for start in range(0, total, ENCODE_CHUNK_SIZE):
        chunk_texts = texts[start:start + ENCODE_CHUNK_SIZE]
        counts = np.zeros((len(chunk_texts), vocab_size), dtype=np.float32)
        for i, doc in enumerate(nlp.pipe(chunk_texts, batch_size=SPACY_BATCH_SIZE)):
            for rule in extract_struct_rules(doc):
                j = rule_to_id.get(rule)
                if j is not None:
                    counts[i, j] = 1.0
        counts *= 0.5
        counts_tensor = torch.from_numpy(counts).to(
            device=torch.device(ENCODE_DEVICE), dtype=torch.float32
        )
        with torch.inference_mode():
            z, _ = encoder(counts_tensor)
        encoded_chunks.append(z.detach().cpu().numpy().astype(np.float32))
        del counts_tensor, counts, z
        log(f"    encoded {min(start + len(chunk_texts), total)}/{total}")
    return np.concatenate(encoded_chunks, axis=0)


# ============================================================================
# Mahalanobis 计算
# ============================================================================

def maha_d2(Z: np.ndarray, mu: np.ndarray, inv_sigma: np.ndarray) -> np.ndarray:
    diff = Z - mu
    return np.einsum("nd,de,ne->n", diff, inv_sigma, diff)


def maha_d2_one(z: np.ndarray, mu: np.ndarray, inv_sigma: np.ndarray) -> float:
    d = z - mu
    return float(d @ inv_sigma @ d)


# ============================================================================
# Log-likelihood under full-covariance Gaussian (NEW framework)
# ============================================================================

def _build_inv_chol(inv_sigma: np.ndarray) -> np.ndarray:
    """L_invT = (chol(inv_sigma))^{-T} such that Σ = L^{-T} L^{-1}."""
    try:
        L = scipy.linalg.cholesky(inv_sigma, lower=True)
        return scipy.linalg.inv(L).T
    except Exception:
        cov = np.linalg.pinv(inv_sigma) + 1e-8 * np.eye(inv_sigma.shape[0])
        L = scipy.linalg.cholesky(cov, lower=True)
        return scipy.linalg.inv(L).T


def _log_prob_batch(Z: np.ndarray, mu: np.ndarray,
                    L_invT: np.ndarray) -> np.ndarray:
    """Vectorized log N(Z | mu, Σ) given L_invT = (chol(Σ^{-1}))^{-T}.

    Z: (n_samples, d)
    mu: (d,)
    L_invT: (d, d)
    Returns: (n_samples,)
    """
    d = mu.shape[0]
    diff = Z - mu
    wh = diff @ L_invT
    mahal = np.sum(wh * wh, axis=1)
    log_det = -2.0 * np.sum(np.log(np.abs(np.diag(L_invT)) + 1e-12))
    return -0.5 * (d * np.log(2 * np.pi) - log_det + mahal)


def log_prob_one(z: np.ndarray, mu: np.ndarray,
                 L_invT: np.ndarray) -> float:
    """Scalar log N(z | mu, Σ)."""
    d = z - mu
    mahal = float(d @ L_invT @ L_invT.T @ d)
    log_det = -2.0 * np.sum(np.log(np.abs(np.diag(L_invT)) + 1e-12))
    return -0.5 * (mu.shape[0] * np.log(2 * np.pi) - log_det + mahal)


def _load_per_pair_ovl(ovl_path: Path) -> dict:
    """Load per-(target_uid, competitor_uid) OVL from gaussian_ovl_matrix.json.

    Returns pair_ovl: dict of "targetUID_competitorUID" -> float OVL.
    """
    if not ovl_path.exists():
        return {}
    with open(ovl_path) as f:
        data = json.load(f)
    pair_ovl = data.get("pair_ovl", {})
    # Normalize: if key is i_j (index-based), we need per-user_mean_ovl instead
    # Since same-ASIN OVL stores keys as uid_uid strings, use as-is
    return pair_ovl


def _get_ovl(target_uid: str, competitor_uid: str,
             per_pair_ovl: dict, per_user_ovl: dict) -> float:
    """Get OVL between target and competitor.

    Prefer per-pair OVL if available; fall back to min of per-user means.
    """
    key = f"{target_uid}_{competitor_uid}"
    rev_key = f"{competitor_uid}_{target_uid}"
    if key in per_pair_ovl:
        return float(per_pair_ovl[key])
    if rev_key in per_pair_ovl:
        return float(per_pair_ovl[rev_key])
    # Fallback: geometric mean of per-user means (symmetric approximation)
    t_ovl = float(per_user_ovl.get(target_uid, 0.5))
    c_ovl = float(per_user_ovl.get(competitor_uid, 0.5))
    return (t_ovl + c_ovl) * 0.5


# ============================================================================
# 选择判定 (single_fit_unique 主体)
# ============================================================================

def check_single_fit_unique(query_z: np.ndarray, target: Dict,
                            competitors: Dict[str, Dict]) -> Tuple[bool, Dict]:
    debug = {"d_target": None, "d_competitors": {}, "fail_stage": None}
    d_t = maha_d2_one(query_z, target["mu"], target["inv_sigma"])
    debug["d_target"] = d_t
    if d_t > target["gate_T"]:
        debug["fail_stage"] = "target_outside_core"
        return False, debug
    for uid, comp in competitors.items():
        d_c = maha_d2_one(query_z, comp["mu"], comp["inv_sigma"])
        debug["d_competitors"][uid] = d_c
        if d_c <= comp["gate_T"]:
            debug["fail_stage"] = f"inside_competitor_core:{uid}"
            return False, debug
    debug["fail_stage"] = None
    return True, debug


def check_exclusive(query_z: np.ndarray, target: Dict,
                    competitors: Dict[str, Dict],
                    B: int = 29, p_thresh: float = 0.95,
                    bs_seed: int = 42) -> Tuple[bool, float, Dict]:
    """bootstrap 变体 (USE_SINGLE_FIT=False 时启用).

    需要 target["Z_prof"] 用于重抽样. Stage 04 不缓存 Z_prof, 因此默认
    (USE_SINGLE_FIT=True) 不走此路径. 若强制启用且 Z_prof 缺失, 返回失败.
    """
    debug = {"d_target": None, "d_competitors": {}, "n_pass_B": 0, "B": B,
             "boot_d2s": None, "fail_stage": None}
    if "Z_prof" not in target:
        debug["fail_stage"] = "bootstrap_unavailable:Z_prof_missing"
        return False, 0.0, debug
    Z_prof = target["Z_prof"]
    d_t = maha_d2_one(query_z, target["mu"], target["inv_sigma"])
    debug["d_target"] = d_t
    if d_t > target["gate_T"]:
        debug["fail_stage"] = "target_outside_core"
        return False, 0.0, debug
    for uid, comp in competitors.items():
        d_c = maha_d2_one(query_z, comp["mu"], comp["inv_sigma"])
        debug["d_competitors"][uid] = d_c
        if d_c <= comp["gate_T"]:
            debug["fail_stage"] = f"inside_competitor_core:{uid}"
            return False, 0.0, debug
    n_take = max(int(len(Z_prof) * BS_FRAC), 1)
    rng = np.random.RandomState(bs_seed)
    n_pass = 0
    boot_d2s = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.choice(len(Z_prof), size=n_take, replace=False)
        Z_b = Z_prof[idx]
        mu_b = Z_b.mean(axis=0)
        centered = Z_b - mu_b
        cov_b = (centered.T @ centered) / max(len(Z_b) - 1, 1)
        try:
            inv_b = np.linalg.inv(cov_b)
        except np.linalg.LinAlgError:
            boot_d2s[b] = np.inf
            continue
        d_b = maha_d2_one(query_z, mu_b, inv_b)
        boot_d2s[b] = d_b
        if d_b <= target["gate_T"]:
            n_pass += 1
    P_pass = n_pass / B
    debug["n_pass_B"] = n_pass
    debug["boot_d2s"] = boot_d2s.tolist()
    if P_pass < p_thresh:
        debug["fail_stage"] = f"bootstrap_unstable:{n_pass}/{B}<{p_thresh}"
        return False, P_pass, debug
    return True, P_pass, debug


# ============================================================================
# 主流程
# ============================================================================

def main_pipeline():
    log("=== syntax_select_mahalanobis_gate ===")
    log(f"  FILTER_Q={FILTER_Q}  MIN_PROFILE={MIN_PROFILE_SENTS} "
        f"MIN_VAL={MIN_VAL_SENTS}  SMOKE={SMOKE}")
    if SWEEP_Q_VALUES:
        log(f"  SWEEP mode active: q_values={SWEEP_Q_VALUES}")
    require_cuda()
    if not USE_SINGLE_FIT:
        raise NotImplementedError(
            "vectorized Stage 08 selection requires USE_SINGLE_FIT=True"
        )

    # --- 加载 Stage 04 canonical artifact (per-user Gaussian + cohort gates) ---
    user_stats, cohort_gates_map, stage04_config = load_stage04()

    # OVL filtering disabled in pure D² gate mode
    high_ovl_uids: set = set()
    user_ovl: Dict[str, float] = {}
    log(f"  OVL filtering disabled (OVL_MAX={OVL_MAX}), high-OVL excluded: 0")

    # --- 加载冻结 encoder + vocab (用于编码 candidate queries) ---
    encoder = load_encoder()
    V = encoder.encoder[0].in_features
    with open(CACHE_DIR / "vocab.json") as f:
        vocab = json.load(f)
    if len(vocab) != V:
        raise ValueError(f"vocab ({len(vocab)}) != encoder input dim ({V})")
    rule_to_id = {r: i for i, r in enumerate(vocab)}
    nlp = spacy.load(SPACY_MODEL, disable=SPACY_DISABLE)

    # --- 候选池 + target user 选择 ---
    with open(POOL_PATH) as f:
        pool_data = json.load(f)
    if isinstance(pool_data, dict) and "pool" in pool_data \
            and isinstance(pool_data["pool"], dict):
        pool_queries = pool_data["pool"]
    else:
        pool_queries = pool_data
    with open(ATTRS_PATH) as f:
        attrs_all = json.load(f)

    def _attrs_for(asin: str) -> Dict[str, str]:
        a = attrs_all.get(asin)
        if a is None:
            raise KeyError(f"missing product attributes for selected ASIN {asin}")
        return dict(list(a.items())[:5])

    pool_entries = []
    for asin, queries in pool_queries.items():
        cands = [{"text": t, "pass": bool(len(t.split()) >= 3)}
                 for t in queries]
        pool_entries.append({"asin": asin, "attrs": _attrs_for(asin),
                             "candidates": cands})
    log(f"  pool queries loaded: {len(pool_entries)} ASINs, "
        f"total candidates={sum(len(p['candidates']) for p in pool_entries)}")

    if SMOKE:
        pool_entries = [p for p in pool_entries
                        if p["asin"] in cohort_gates_map][:N_SMOKE_ASIN]
    log(f"  pool entries to process: {len(pool_entries)}")

    # Stage 04 Gaussian 只转换一次；task 仅保存轻量引用。
    gauss_cache = {uid: _gauss_from_stats(stats)
                   for uid, stats in user_stats.items()}

    # 预计算 L_invT（用于 log-likelihood 计算）
    log(f"  Pre-computing L_invT for {len(gauss_cache)} users ...")
    for uid, g in gauss_cache.items():
        g["L_invT"] = _build_inv_chol(g["inv_sigma"].astype(np.float64))

    asin_work: Dict[str, Dict] = {}
    tasks: List[Dict] = []
    skipped: List[Dict] = []
    no_pass_unreliable: List[Dict] = []
    for entry in pool_entries:
        asin = entry["asin"]
        cohort = cohort_gates_map.get(asin)
        if cohort is None:
            skipped.append({"asin": asin, "reason": "no_stage04_cohort"})
            continue
        # Filter out high-OVL users from both target and competitor positions
        all_cohort_uids = [uid for uid in sorted(cohort)
                           if uid not in high_ovl_uids]
        if len(all_cohort_uids) < 2:
            skipped.append({"asin": asin, "reason": f"cohort_too_small_after_ovl_filter({len(all_cohort_uids)})"})
            continue
        # All remaining cohort users are valid targets and competitors
        asin_work[asin] = {
            "asin": asin,
            "attrs": entry["attrs"],
            "candidates": entry["candidates"],
            "uids": all_cohort_uids,          # targets: all non-high-OVL users
            "cohort_uids": all_cohort_uids,   # competitors: same set
            "cohort": cohort,
        }
        for uid in all_cohort_uids:
            tasks.append({"asin": asin, "uid": uid, "tier": "stage04_fitted"})
    log(f"  tasks: {len(tasks)}  skipped: {len(skipped)}")
    if not tasks:
        raise RuntimeError("no eligible (ASIN, target_user) tasks")

    for t in tasks[:5]:
        gauss = gauss_cache[t["uid"]]
        log(f"    {t['asin']} uid={t['uid']} gate_T={gauss['gate_T']:.2f} "
            f"(d2_q50={gauss['d2_val_median']:.2f})")
    if len(tasks) > 5:
        log(f"    ... and {len(tasks) - 5} more tasks")

    # --- 全局精确文本去重，再按 ASIN 恢复候选顺序 ---
    asins = list(asin_work)
    unique_text_to_idx: Dict[str, int] = {}
    unique_texts: List[str] = []
    asin_text_indices: Dict[str, np.ndarray] = {}
    for asin in asins:
        indices = []
        for candidate in asin_work[asin]["candidates"]:
            text = candidate["text"]
            index = unique_text_to_idx.get(text)
            if index is None:
                index = len(unique_texts)
                unique_text_to_idx[text] = index
                unique_texts.append(text)
            indices.append(index)
        asin_text_indices[asin] = np.asarray(indices, dtype=np.int64)
    log(f"  encoding {len(unique_texts)} unique texts for "
        f"{sum(len(asin_work[a]['candidates']) for a in asins)} candidates "
        f"across {len(asins)} ASINs")
    Z_unique = encode_texts(unique_texts, nlp, rule_to_id, V, encoder)
    z_dim = encoder.z_dim if hasattr(encoder, 'z_dim') else 16
    if Z_unique.shape != (len(unique_texts), z_dim):
        raise ValueError(f"encoder output shape mismatch: {Z_unique.shape}")
    asin_to_Zq = {
        asin: Z_unique[asin_text_indices[asin]]
        for asin in asins
    }
    if SWEEP_Q_VALUES:
        # SWEEP mode: 重用已编码的 Z_q 跑 sweep, 然后退出 (不写 selected_queries.json).
        # Load Stage 04 raw artifact for sweep.
        with open(STAGE04_PATH) as f:
            _stage04 = json.load(f)
        _users_all = _stage04.get("users", {})
        _valid_uids: set = set()
        for _uid, _stats in _users_all.items():
            if not isinstance(_stats, dict) or "sigma_inv" not in _stats:
                continue
            _mu = _stats.get("mu"); _si = _stats.get("sigma_inv")
            if not isinstance(_mu, list) or len(_mu) not in (16, 32): continue
            if not isinstance(_si, list) or len(_si) not in (16, 32): continue
            if not all(isinstance(r, list) and len(r) in (16, 32) for r in _si): continue
            _flat = [float(x) for r in _si for x in r]
            if not all(np.isfinite(x) for x in _flat): continue
            if not all(np.isfinite(float(x)) for x in _mu): continue
            if not all(f"d2_q{n:02d}_theoretical" in _stats for n in (5, 50, 75, 95)):
                continue
            _valid_uids.add(_uid)
        log(f"  SWEEP: {len(_users_all)} users, {len(_valid_uids)} valid for sweep")
        _run_sweep(asin_to_Zq, _stage04, _users_all, _valid_uids)
        return
    asin_to_cand_text = {
        asin: asin_work[asin]["candidates"]
        for asin in asins
    }

    # --- 选择循环 ---
    selections: List[Dict] = []
    no_pass: List[Dict] = []
    all_d2_round0: List[float] = []
    all_margin_round0: List[float] = []
    n_gate_pass_round0 = 0
    n_cand_round0 = 0

    for asin in asins:
        work = asin_work[asin]
        Z_q = asin_to_Zq[asin]
        cands_src = asin_to_cand_text[asin]
        uids = work["uids"]
        cohort_uids = work["cohort_uids"]
        n_cohort = len(cohort_uids)
        n_cand = Z_q.shape[0]

        if USE_LIKELIHOOD_MARGIN:
            # === NEW FRAMEWORK: OVL-adaptive likelihood margin ===
            # 1. Compute log-likelihood matrix (n_cand, n_cohort)
            log_like_matrix = np.zeros((n_cand, n_cohort), dtype=np.float64)
            L_invT_stack = np.stack(
                [gauss_cache[uid]["L_invT"].astype(np.float64) for uid in cohort_uids],
                axis=0
            )
            mu_stack = np.stack(
                [gauss_cache[uid]["mu"].astype(np.float64) for uid in cohort_uids],
                axis=0
            )
            for ci in range(n_cohort):
                log_like_matrix[:, ci] = _log_prob_batch(
                    Z_q.astype(np.float64),
                    mu_stack[ci],
                    L_invT_stack[ci]
                )

            # 2. Build OVL matrix for this cohort (n_cohort, n_cohort)
            ovl_matrix = np.ones((n_cohort, n_cohort), dtype=np.float64) * 0.5
            for ti, t_uid in enumerate(cohort_uids):
                for vi, v_uid in enumerate(cohort_uids):
                    if ti == vi:
                        ovl_matrix[ti, vi] = 0.0
                    else:
                        ovl_matrix[ti, vi] = _get_ovl(t_uid, v_uid, per_pair_ovl, user_ovl)

            # 3. Per-target selection: margin > gamma0 + lambda * OVL
            target_index = {uid: i for i, uid in enumerate(cohort_uids)}
            for uid in uids:
                t_i = target_index[uid]
                log_like_t = log_like_matrix[:, t_i:t_i+1]   # (n_cand, 1)

                # Margin matrix: (n_cand, n_cohort)
                margin_matrix = log_like_t - log_like_matrix   # M(q;t,v) for all v
                # Required margin per competitor: gamma0 + lambda * OVL(t,v)
                required = GAMMA0 + LAMBDA * ovl_matrix[t_i, :]   # (n_cohort,)
                # Query passes if ALL margins exceed required (set self-margin to +inf)
                margin_matrix[:, t_i] = np.inf
                required[t_i] = -np.inf
                pass_mask = np.all(margin_matrix > required[None, :], axis=1)

                # Record per-candidate
                best_margin = -np.inf
                best_record = None
                for i, (c, log_lt) in enumerate(zip(cands_src, log_like_matrix[:, t_i])):
                    in_gate = bool(pass_mask[i])
                    if in_gate:
                        margin = float(log_lt - np.max(
                            np.where(np.arange(n_cohort) != t_i,
                                     log_like_matrix[i, :], -np.inf)
                        ))
                        if margin > best_margin:
                            best_margin = margin
                            # Compute per-competitor margins for record
                            comp_margins = {
                                cohort_uids[vi]: float(
                                    log_like_matrix[i, t_i] - log_like_matrix[i, vi]
                                )
                                for vi in range(n_cohort) if vi != t_i
                            }
                            best_record = {
                                "text": c["text"],
                                "log_like_target": float(log_like_matrix[i, t_i]),
                                "content_pass": bool(c.get("pass", False)),
                                "round": 0,
                                "pass_gate": True,
                                "margin": margin,
                                "comp_margins": comp_margins,
                                "mean_ovl": user_ovl.get(uid, None),
                            }
                    all_margin_round0.append(float(log_like_matrix[i, t_i]))

                n_cand_round0 += n_cand
                if best_record is not None:
                    gauss = gauss_cache[uid]
                    selections.append({
                        "asin": asin, "uid": uid,
                        "tier": "stage04_fitted",
                        "selection_mode": "likelihood_margin_ovl_adaptive",
                        "gate_quantile_used": gauss["gate_quantile"],
                        "n_profile": gauss["n"], "n_val": gauss["n_val"],
                        "gate_T": gauss["gate_T"],
                        "d2_val_median": gauss["d2_val_median"],
                        "selected": best_record,
                        "n_pass_unique": int(pass_mask.sum()),
                        "n_competitors": n_cohort - 1,
                        "n_candidates": n_cand,
                        "gamma0": GAMMA0,
                        "lambda": LAMBDA,
                        "competitors": [
                            {"uid": cohort_uids[vi], "ovl": float(ovl_matrix[t_i, vi]),
                             "required_margin": float(required[vi])}
                            for vi in range(n_cohort) if vi != t_i
                        ],
                    })
                else:
                    no_pass.append({
                        "asin": asin, "uid": uid,
                        "tier": "stage04_fitted",
                        "selection_mode": "likelihood_margin_ovl_adaptive",
                        "gate_quantile_used": gauss_cache[uid]["gate_quantile"],
                        "gate_T": gauss_cache[uid]["gate_T"],
                        "n_candidates": n_cand,
                        "reason": "no_margin_pass",
                        "n_competitors": n_cohort - 1,
                    })
        else:
            # === LEGACY D² gate ===
            mu = np.stack([gauss_cache[uid]["mu"] for uid in cohort_uids], axis=0)
            inv_sigma = np.stack([gauss_cache[uid]["inv_sigma"] for uid in cohort_uids], axis=0)
            gate_Ts = np.asarray([float(work["cohort"][uid]["gate_T"])
                                  for uid in cohort_uids], dtype=np.float64)
            for idx, uid in enumerate(cohort_uids):
                expected_gate = gauss_cache[uid]["gate_T"]
                if not np.isfinite(expected_gate) or not np.isclose(
                        gate_Ts[idx], expected_gate, atol=1e-5, rtol=0.0):
                    raise ValueError(f"gate_T mismatch for {asin}/{uid}")
            diff = Z_q[:, None, :] - mu[None, :, :]
            left = np.einsum("cud,ude->cue", diff, inv_sigma)
            d2_matrix = np.sum(left * diff, axis=2, dtype=np.float32)
            if not np.all(np.isfinite(d2_matrix)):
                raise FloatingPointError(f"non-finite candidate D² for ASIN {asin}")
            near_gate = np.abs(d2_matrix - gate_Ts[None, :]) <= D2_BOUNDARY_RECHECK_TOL
            boundary_pairs = np.argwhere(near_gate)
            for candidate_i, cohort_i in boundary_pairs:
                candidate_i = int(candidate_i)
                cohort_i = int(cohort_i)
                d2_matrix[candidate_i, cohort_i] = maha_d2_one(
                    Z_q[candidate_i], gauss_cache[cohort_uids[cohort_i]]["mu"],
                    gauss_cache[cohort_uids[cohort_i]]["inv_sigma"]
                )
            target_index = {uid: i for i, uid in enumerate(cohort_uids)}
            for uid in uids:
                target_i = target_index[uid]
                target_d2 = d2_matrix[:, target_i]
                gate_T = float(gate_Ts[target_i])
                target_inside = target_d2 <= gate_T
                competitor_inside = d2_matrix <= gate_Ts[None, :]
                competitor_inside[:, target_i] = False
                pass_unique_mask = target_inside & (~competitor_inside.any(axis=1))
                cand_records = []
                for i, (c, d, in_gate, unique) in enumerate(zip(
                        cands_src, target_d2, target_inside, pass_unique_mask)):
                    if unique:
                        fail_stage = None
                    elif not in_gate:
                        fail_stage = "target_outside_core"
                    else:
                        competitor_indices = np.flatnonzero(competitor_inside[i])
                        if len(competitor_indices) == 0:
                            raise RuntimeError(
                                f"exclusive gate state inconsistent for {asin}/{uid}/{i}"
                            )
                        fail_stage = (
                            f"inside_competitor_core:"
                            f"{cohort_uids[int(competitor_indices[0])] }"
                        )
                    cand_records.append({
                        "text": c["text"],
                        "d2": float(d),
                        "pass_gate": bool(in_gate),
                        "content_pass": bool(c.get("pass", False)),
                        "round": 0,
                        "pass_unique": bool(unique),
                        "P_pass": None,
                        "fail_stage": fail_stage,
                        "mean_ovl": user_ovl.get(uid, None),
                    })
                unique_indices = np.flatnonzero(pass_unique_mask)
                unique_pass = [cand_records[int(i)] for i in unique_indices]
                n_cand_round0 += len(cand_records)
                n_gate_pass_round0 += int(target_inside.sum())
                all_d2_round0.extend(float(d) for d in target_d2)
            gauss = gauss_cache[uid]
            if SMOKE:
                competitors = {
                    comp_uid: gauss_cache[comp_uid]
                    for comp_uid in cohort_uids
                    if comp_uid != uid
                }
                scalar_unique = []
                for i, candidate in enumerate(cands_src):
                    scalar_ok, scalar_debug = check_single_fit_unique(
                        Z_q[i], gauss_cache[uid], competitors
                    )
                    if not np.isclose(
                        cand_records[i]["d2"], scalar_debug["d_target"],
                        # einsum 与标量 BLAS 的 float32 累加顺序不同；门控值使用显式误差界。
                        rtol=D2_GATE_REL_TOL, atol=D2_GATE_ABS_TOL,
                    ):
                        raise AssertionError(
                            f"vectorized/scalar target D² mismatch: {asin}/{uid}/{i} "
                            f"vectorized={cand_records[i]['d2']:.9g} "
                            f"scalar={scalar_debug['d_target']:.9g} "
                            f"abs={abs(cand_records[i]['d2'] - scalar_debug['d_target']):.9g}"
                        )
                    if cand_records[i]["pass_unique"] != scalar_ok:
                        raise AssertionError(
                            f"vectorized/scalar pass mismatch: {asin}/{uid}/{i}"
                        )
                    if cand_records[i]["fail_stage"] != scalar_debug["fail_stage"]:
                        raise AssertionError(
                            f"vectorized/scalar fail-stage mismatch: {asin}/{uid}/{i} "
                            f"vectorized={cand_records[i]['fail_stage']} "
                            f"scalar={scalar_debug['fail_stage']}"
                        )
                    if scalar_ok:
                        scalar_unique.append(i)
                if list(unique_indices.astype(int)) != scalar_unique:
                    raise AssertionError(
                        f"vectorized/scalar candidate set mismatch: {asin}/{uid}"
                    )

            if unique_pass:
                # 2026-09-15: 保留 top-K unique_pass queries（D² 升序），让 ASIN 内
                # 多 query 累积。每个 (asin, uid) 最多贡献 K 条 query 给 ASIN。
                sorted_unique = sorted(unique_pass, key=lambda c: c["d2"])
                top_k_queries = sorted_unique[:TOP_K_PER_USER]
                best = top_k_queries[0]
                selections.append({
                    "asin": asin, "uid": uid,
                    "tier": "stage04_fitted",
                    "selection_mode": "single_fit_unique_topk" if USE_SINGLE_FIT
                                      else "bootstrap_core_exclusive_topk",
                    "gate_quantile_used": gauss["gate_quantile"],
                    "n_profile": gauss["n"], "n_val": gauss["n_val"],
                    "gate_T": gate_T,
                    "d2_val_median": gauss["d2_val_median"],
                    "selected": best,
                    "top_k_queries": top_k_queries,
                    "n_pass_unique": len(unique_pass),
                    "n_pass_gate": int(target_inside.sum()),
                    "n_competitors": len(uids) - 1,
                    "n_candidates": len(cand_records),
                    "n_regen_rounds_used": 0,
                    "P_pass": None,
                    "candidates": cand_records,
                })
            else:
                fail_stages = [c["fail_stage"] for c in cand_records
                               if c.get("fail_stage")]
                no_pass.append({
                    "asin": asin, "uid": uid,
                    "tier": "stage04_fitted",
                    "selection_mode": "single_fit_unique" if USE_SINGLE_FIT
                                      else "bootstrap_core_exclusive",
                    "gate_quantile_used": gauss["gate_quantile"],
                    "gate_T": gate_T,
                    "min_d2": float(target_d2.min()),
                    "n_candidates": len(cand_records),
                    "n_regen_rounds_used": 0,
                    "reason": "no_unique",
                    "fail_stages": dict(Counter(fail_stages).most_common()),
                    "n_competitors": len(uids) - 1,
                    "candidates": cand_records,
                })
            if not USE_LIKELIHOOD_MARGIN:
                del diff, d2_matrix, mu, inv_sigma, gate_Ts

    d2_arr = np.array(all_d2_round0, dtype=np.float64)
    no_pass_total = no_pass + no_pass_unreliable
    asin_user_count = Counter(s["asin"] for s in selections)
    n_asins_ge2_unique_users = sum(1 for c in asin_user_count.values() if c >= 2)
    n_unique_asins = len(asin_user_count)
    summary = {
        "n_tasks": len(tasks),
        "n_selected": len(selections),
        "n_no_pass": len(no_pass_total),
        "n_skipped": len(skipped),
        "n_unreliable_no_select": len(no_pass_unreliable),
        "total_regen_rounds": 0,
        "round0_gate_pass_rate": (n_gate_pass_round0 / max(1, n_cand_round0)),
        "round0_d2_median": float(np.median(d2_arr)) if len(d2_arr) else None,
        "round0_d2_p25": float(np.quantile(d2_arr, 0.25)) if len(d2_arr) else None,
        "round0_d2_p75": float(np.quantile(d2_arr, 0.75)) if len(d2_arr) else None,
        "n_unique_asins": n_unique_asins,
        "n_asins_ge2_unique_users": n_asins_ge2_unique_users,
        "n_high_ovl_excluded": len(high_ovl_uids),
        "ovl_max_threshold": OVL_MAX,
        "n_users_with_ovl_data": len(user_ovl),
        "top_k_per_user": TOP_K_PER_USER,
    }
    log(f"  SUMMARY: tasks={summary['n_tasks']} selected={summary['n_selected']} "
        f"no_pass={summary['n_no_pass']} (unreliable={summary['n_unreliable_no_select']}) "
        f"skipped={summary['n_skipped']} "
        f"regen_rounds={summary['total_regen_rounds']} "
        f"round0_gate_pass={summary['round0_gate_pass_rate']*100:.1f}% "
        f"round0_d2_median={summary['round0_d2_median']} "
        f"unique_asins={n_unique_asins} asins_ge2_unique_users={n_asins_ge2_unique_users}")

    # 按 ASIN 聚合 (全部 selected users，≥1 即可)
    # 2026-09-15: 每个 (asin, uid) task 可贡献 top-K queries
    # (见 selections[*].top_k_queries); 展开为 (uid, query) pairs.
    asin_to_users_out = defaultdict(list)
    for s in selections:
        top_q = s.get("top_k_queries") or [s["selected"]]
        for q in top_q:
            asin_to_users_out[s["asin"]].append({
                "uid": s["uid"],
                "query": q["text"],
                "d2": q.get("d2"),
            })
    asin_blocks = [
        {"asin": a, "users": us}
        for a, us in sorted(asin_to_users_out.items())
        if len(us) >= 1
    ]
    # 同一 ASIN unique query max pair cos ≥ 0.9 (MiniLM)
    from sentence_transformers import SentenceTransformer
    st_device = ENCODE_DEVICE
    _st_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device=st_device
    )
    block_texts = [[u["query"] for u in blk["users"]] for blk in asin_blocks]
    flat_texts = [text for texts in block_texts for text in texts]
    filtered_blocks = []
    if flat_texts:
        flat_emb = _st_model.encode(
            flat_texts, batch_size=MINILM_BATCH_SIZE, convert_to_tensor=True,
            normalize_embeddings=True, show_progress_bar=False,
            device=st_device,
        )
        offset = 0
        for blk, texts in zip(asin_blocks, block_texts):
            n = len(texts)
            emb = flat_emb[offset:offset + n]
            offset += n
            cos = emb @ emb.T
            if n == 2:
                max_cos = float(cos[0, 1])
            else:
                mask = ~torch.eye(n, dtype=torch.bool, device=cos.device)
                masked = cos[mask]
                if masked.numel() == 0:
                    max_cos = 0.0
                else:
                    max_cos = float(masked.max())
            if max_cos >= SEMANTIC_SIM_THRESHOLD or n == 1:
                blk["max_pair_cos"] = max_cos if n > 1 else None
                filtered_blocks.append(blk)
        if offset != len(flat_texts):
            raise RuntimeError("MiniLM block/text offset mismatch")
    asin_blocks = filtered_blocks
    n_asin_blocks = len(asin_blocks)
    # 2026-09-15: 报告 ASIN 内 unique query 覆盖
    from collections import Counter as _C
    _q_per_asin = [len(set(u["query"] for u in b["users"])) for b in asin_blocks]
    _uq_counts = _C(_q_per_asin)
    n_asins_ge2_unique_queries = sum(1 for n in _q_per_asin if n >= 2)
    n_asins_ge3_unique_queries = sum(1 for n in _q_per_asin if n >= 3)
    summary["n_asin_blocks_kept_after_sim_filter"] = n_asin_blocks
    summary["n_asins_ge2_unique_queries"] = n_asins_ge2_unique_queries
    summary["n_asins_ge3_unique_queries"] = n_asins_ge3_unique_queries
    summary["unique_queries_per_asin_distribution"] = dict(sorted(_uq_counts.items()))
    log(f"  POST-AGG: asin_blocks={n_asin_blocks} asins_ge2_unique_queries={n_asins_ge2_unique_queries} asins_ge3_unique_queries={n_asins_ge3_unique_queries} uq_dist={dict(_uq_counts)}")

    out = {
        "config": {
            "filter_q": 0.05,
            "min_profile_sents": MIN_PROFILE_SENTS,
            "min_val_sents": MIN_VAL_SENTS,
            "smoke": SMOKE,
            "semantic_sim_threshold": SEMANTIC_SIM_THRESHOLD,
            "encoder_device": ENCODE_DEVICE,
            "encoder_chunk_size": ENCODE_CHUNK_SIZE,
            "encoder": str(CACHE_DIR / "adaptive_encoder.pt"),
            "stage04_source": str(STAGE04_PATH),
            "stage04_config": stage04_config,
            "pool_path": str(POOL_PATH),
            "spacy_model": SPACY_MODEL,
            "note": "per-user Gaussian and cohort membership are loaded from "
                    "the Stage 04 canonical artifact; gate_T=d2_q05_theoretical; "
                    "single_fit_unique_topk keeps top-K unique_pass queries per (asin, uid) "
                    "(default K=3) to increase per-ASIN unique query count",
        },
        "summary": summary,
        "selections": asin_blocks,
        "no_pass": [
            {"asin": record["asin"], "uid": record["uid"]}
            for record in no_pass_total
        ],
        "skipped": skipped,
    }
    output_path = SMOKE_OUT_PATH if SMOKE else OUT_PATH
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"  saved -> {output_path}")
    log("=== syntax_select_mahalanobis_gate DONE ===")


if __name__ == "__main__":
    main_pipeline()
