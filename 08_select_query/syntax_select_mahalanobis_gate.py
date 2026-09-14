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
from collections import defaultdict
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
SEMANTIC_SIM_THRESHOLD = 0.8
MIN_PROFILE_SENTS = 40
MIN_VAL_SENTS = 10
SMOKE = False
N_SMOKE_ASIN = 5
SPACY_MODEL = "en_core_web_sm"
SPACY_DISABLE = ["ner", "textcat", "lemmatizer"]

COMPETITOR_GATE_QUANTILE = FILTER_Q
BS_N_BOOTSTRAP = 29
BS_FRAC = 0.5
BS_P_THRESH = 0.95
BS_SEED = 42
USE_SINGLE_FIT = True    # single_fit_unique: 一次性固定高斯, 无 bootstrap
ENCODE_DEVICE = "cuda:0"
ENCODE_CHUNK_SIZE = 1024
SPACY_BATCH_SIZE = 256
MINILM_BATCH_SIZE = 512
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
    stage04_cohorts = stage04.get("cohort_gates")
    config = stage04.get("config")
    if not isinstance(users_all, dict) or not users_all:
        raise ValueError("Stage 04 artifact requires non-empty 'users'")
    if not isinstance(stage04_cohorts, dict) or not stage04_cohorts:
        raise ValueError("Stage 04 artifact requires non-empty 'cohort_gates'")
    if not isinstance(config, dict) or config.get("gate_quantile") != FILTER_Q:
        raise ValueError(
            f"Stage 04 gate_quantile must equal FILTER_Q={FILTER_Q}; "
            f"got {None if not isinstance(config, dict) else config.get('gate_quantile')}"
        )

    # Stage 04 Gaussian 完整性 = sigma_inv 存在 ∧ mu/sigma_inv/d2_q95 全部 finite
    valid_uids: set = set()
    for uid, stats in users_all.items():
        if not isinstance(stats, dict):
            continue
        if "sigma_inv" not in stats:
            continue
        mu = stats.get("mu")
        sigma_inv = stats.get("sigma_inv")
        gate_t = stats.get("d2_q95")  # Stage 04 stores gate_quantile value in d2_q95 field
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
            if not isinstance(gate, dict) or "gate_T" not in gate:
                raise ValueError(f"Stage 04 cohort {asin}/{uid} missing gate_T")
            if not np.isfinite(float(gate["gate_T"])):
                raise ValueError(f"Stage 04 cohort {asin}/{uid} gate_T not finite")
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
    """把 Stage 04 user stats 转成 selection 所需的 numpy 结构。"""
    required = ("mu", "sigma_inv", "n", "n_val", "d2_q50", "d2_q95")
    missing = [key for key in required if key not in stats]
    if missing:
        raise ValueError(f"Stage 04 user stats missing fields: {missing}")
    mu = np.asarray(stats["mu"], dtype=np.float32)
    inv_sigma = np.asarray(stats["sigma_inv"], dtype=np.float32)
    if mu.ndim != 1 or mu.shape[0] not in (16, 32):
        raise ValueError(f"invalid Gaussian mu shape: {mu.shape}")
    if inv_sigma.ndim != 2 or inv_sigma.shape[0] != inv_sigma.shape[1] or inv_sigma.shape[0] not in (16, 32):
        raise ValueError(f"invalid Gaussian sigma_inv shape: {inv_sigma.shape}")
    gate_T = float(stats["d2_q95"])
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(inv_sigma)):
        raise FloatingPointError("non-finite Stage 04 Gaussian parameters")
    if not np.isfinite(gate_T) or gate_T < 0:
        raise ValueError(f"invalid Stage 04 gate_T={gate_T}")
    return {
        "mu": mu,
        "inv_sigma": inv_sigma,
        "gate_T": gate_T,
        "d2_val_median": float(stats["d2_q50"]),
        "gate_quantile": FILTER_Q,
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
    require_cuda()
    if not USE_SINGLE_FIT:
        raise NotImplementedError(
            "vectorized Stage 08 selection requires USE_SINGLE_FIT=True"
        )

    # --- 加载 Stage 04 canonical artifact (per-user Gaussian + cohort gates) ---
    user_stats, cohort_gates_map, stage04_config = load_stage04()

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
        target_uids = sorted(cohort)
        cohort_uids = list(cohort)
        for uid in target_uids:
            if uid not in gauss_cache:
                raise ValueError(f"Stage 04 cohort {asin} references missing user {uid}")
            if len(cohort_uids) < 2:
                raise ValueError(f"Stage 04 cohort {asin} has no competitor for {uid}")
        asin_work[asin] = {
            "asin": asin,
            "attrs": entry["attrs"],
            "candidates": entry["candidates"],
            "uids": target_uids,
            "cohort_uids": cohort_uids,
            "cohort": cohort,
        }
        for uid in target_uids:
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
    asin_to_cand_text = {
        asin: asin_work[asin]["candidates"]
        for asin in asins
    }

    # --- 选择循环 ---
    selections: List[Dict] = []
    no_pass: List[Dict] = []
    all_d2_round0: List[float] = []
    n_gate_pass_round0 = 0
    n_cand_round0 = 0

    for asin in asins:
        work = asin_work[asin]
        Z_q = asin_to_Zq[asin]
        cands_src = asin_to_cand_text[asin]
        uids = work["uids"]
        cohort_uids = work["cohort_uids"]
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
        # 先做与 maha_d2_one 相同的左乘，再做行向量内积，
        # 比三操作数 einsum 更接近旧标量路径的累加顺序。
        left = np.einsum("cud,ude->cue", diff, inv_sigma)
        d2_matrix = np.sum(left * diff, axis=2, dtype=np.float32)
        if not np.all(np.isfinite(d2_matrix)):
            raise FloatingPointError(f"non-finite candidate D² for ASIN {asin}")
        # 仅对接近 gate 的元素用旧标量公式复核，消除不同 BLAS 累加顺序
        # 在边界处造成的判定漂移；远离边界的主体仍保持全向量化。
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
            if SMOKE:
                near_target = np.abs(target_d2 - gate_T) <= D2_BOUNDARY_RECHECK_TOL
                for boundary_i in np.flatnonzero(near_target):
                    scalar_target_d2 = maha_d2_one(
                        Z_q[boundary_i], gauss_cache[uid]["mu"],
                        gauss_cache[uid]["inv_sigma"]
                    )
                    target_inside[boundary_i] = scalar_target_d2 <= gate_T
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
                best = min(unique_pass, key=lambda c: c["d2"])
                selections.append({
                    "asin": asin, "uid": uid,
                    "tier": "stage04_fitted",
                    "selection_mode": "single_fit_unique" if USE_SINGLE_FIT
                                      else "bootstrap_core_exclusive",
                    "gate_quantile_used": gauss["gate_quantile"],
                    "n_profile": gauss["n"], "n_val": gauss["n_val"],
                    "gate_T": gate_T,
                    "d2_val_median": gauss["d2_val_median"],
                    "selected": best,
                    "n_pass_unique": len(unique_pass),
                    "n_pass_gate": int(target_inside.sum()),
                    "n_competitors": len(uids) - 1,
                    "n_candidates": len(cand_records),
                    "n_regen_rounds_used": 0,
                    "P_pass": None,
                    "candidates": cand_records,
                })
            else:
                from collections import Counter
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
        del diff, d2_matrix, mu, inv_sigma, gate_Ts

    d2_arr = np.array(all_d2_round0, dtype=np.float64)
    no_pass_total = no_pass + no_pass_unreliable
    from collections import Counter
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
    }
    log(f"  SUMMARY: tasks={summary['n_tasks']} selected={summary['n_selected']} "
        f"no_pass={summary['n_no_pass']} (unreliable={summary['n_unreliable_no_select']}) "
        f"skipped={summary['n_skipped']} "
        f"regen_rounds={summary['total_regen_rounds']} "
        f"round0_gate_pass={summary['round0_gate_pass_rate']*100:.1f}% "
        f"round0_d2_median={summary['round0_d2_median']} "
        f"unique_asins={n_unique_asins} asins_ge2_unique_users={n_asins_ge2_unique_users}")

    # 按 ASIN 聚合 (≥2 users)
    asin_to_users_out = defaultdict(list)
    for s in selections:
        asin_to_users_out[s["asin"]].append({
            "uid": s["uid"],
            "query": s["selected"]["text"],
        })
    asin_blocks = [
        {"asin": a, "users": us}
        for a, us in sorted(asin_to_users_out.items())
        if len(us) >= 2
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
                max_cos = float(cos[mask].max())
            if max_cos >= SEMANTIC_SIM_THRESHOLD:
                blk["max_pair_cos"] = max_cos
                filtered_blocks.append(blk)
        if offset != len(flat_texts):
            raise RuntimeError("MiniLM block/text offset mismatch")
    asin_blocks = filtered_blocks
    n_asin_blocks = len(asin_blocks)

    out = {
        "config": {
            "filter_q": FILTER_Q,
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
                    "the Stage 04 canonical artifact; gate_T=d2_q95; "
                    "single_fit_unique uses fixed full-covariance fits",
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
