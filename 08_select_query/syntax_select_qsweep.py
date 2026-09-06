#!/usr/bin/env python3
"""Query selection: per-user Mahalanobis D² + validation-quantile gate.

用户 2026-09-05 设计 (两层):
  1. 目标用户真实历史句子的句法向量 z_i (profile 段) 估计 μ_u 与正则化协方差
     Σ_u + λI;
  2. 同一冻结 encoder 编码候选 query 得 z_q, D²(q,u) = (z_q−μ_u)ᵀ Σ_u⁻¹ (z_q−μ_u);
  3. absolute gate 阈值不看测试候选, 从验证集 (val 段) 真实句子 self-D² 取
     分位数 T_u = Q_0.75;
  4. D²(q,u) ≤ T_u 才算落在用户正常句法范围内, 通过者中选 D² 最小;
  5. 每 ASIN 给所有 healthy user 各选 1 query (per-(asin, uid) task), 共享
     candidates, 不再 regen;
  6. 全程同一 encoder、同一归一化 (count/(1+count))、同一正则化 (λI)。

复用资产 (全部已核实):
  - 冻结 encoder: pcfg_cache/adaptive_encoder.pt (_SupEncoder 21737→256→32, eval)
  - 预编码 profile/val z: pcfg_cache/adaptive_embeddings.npz (3-way hash split)
  - 行→uid: pcfg_cache/user_n_sents.json (sum=816023=sent_vectors 行数)
  - 归一化: normalize_counts (03_spacy_encode/syntax_pcfg_pipeline.py:190)
  - 规则: extract_struct_rules (同 :147), spaCy en_core_web_sm (同 :72)
  - 候选池: result/06_gen_query/pool.json

用法 (Rule 3: no args):
  cd /home/wlia0047/ar57/wenyu/PersoanlQuery
  $PY 08_select_query/syntax_select_mahalanobis_gate.py

输出:
  result/08_select_query/selected_queries.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import spacy
import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))

CACHE_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache")
POOL_PATH = REPO_ROOT / "result/07_gen_query/pool_queries.json"
ATTRS_PATH = REPO_ROOT / "result/01_attribute_extraction/product_attributes.json"
ASIN_USERS_PATH = REPO_ROOT / "asin_users/asin_to_users.json"
OUT_PATH = REPO_ROOT / "result/08_select_query/selected_queries.json"
MULTI_AUDIT_PATH = REPO_ROOT / "result/05_gaussian_audit/raw_cov_validity.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- Hardcoded hyperparams (Rule 3) ---
LAMBDA = 0.0                # 2026-09-05: 与 audit 一致用 raw Σ, 不加 ridge
                            # (audit 用 raw_cov_validity 算 E1E2E3E4, 必须统一)
# 2026-09-06 用户锁定 Q=0.75 作为 inside-target strict 的过滤分位数
# (Q sweep 7 个分位数中 Q=75 是 Pareto 峰值: 59 tasks / 48 users / 52 ASINs)
FILTER_Q = 0.75
GATE_QUANTILE_BY_TIER = {   # val self-D² 分位数 gate (用户设计: 三档策略)
    "healthy": FILTER_Q,    # 健康用户: Q_0.75 (Stage 8 inside-target strict 主体)
    "usable": 0.85,         # 可用用户: 放宽到 Q_0.85 (允许更多候选通过)
    "unreliable": None,     # 不可靠用户: 不参选 (记 no_pass 但不 regen)
}
MIN_PROFILE_SENTS = 40      # 拟合 32d full cov 的最小 profile 句数
MIN_VAL_SENTS = 10          # 算 gate 的最小 val 句数
MAX_REGEN_ROUNDS = 3        # 无候选通过时最大重新生成轮数
REGEN_K = 8                 # 每轮重新生成的 candidates 数
SMOKE = False               # True=5 ASIN smoke, False=100 ASIN full (Rule 20)
N_SMOKE_ASIN = 5
SPACY_MODEL = "en_core_web_sm"   # 与 cache 一致 (syntax_pcfg_pipeline.py:72)
SPACY_DISABLE = ["ner", "textcat", "lemmatizer"]  # 同 :229

# 2026-09-05: 核心区域排他性 (Stage 8 升级)
# 2026-09-06: 与 FILTER_Q 同步 (user 锁定 Q=0.75)
COMPETITOR_GATE_QUANTILE = FILTER_Q   # competitor 的"核心外" = val D² > Q0.75
BS_N_BOOTSTRAP = 29               # target 核心区域稳定性重抽样数
BS_FRAC = 0.5                     # 重抽样比例 (与 audit E4 一致)
BS_P_THRESH = 0.95                # bootstrap 通过阈值 (≥95% 稳定)
BS_SEED = 42                      # bootstrap 随机种子 (可复现)
# 2026-09-05 (Stage 8 single-fit 变体): 跳过 bootstrap, 直接用一次性固定高斯筛选.
# 命名为 single_fit_unique (固定 Gaussian, 无 bootstrap 稳定性声明).
USE_SINGLE_FIT = True             # True=single_fit_unique, False=bootstrap core exclusive
ENCODE_DEVICE = "cpu"       # encoder 前向极小 (≤1000 短句), 不与 vLLM 抢 GPU


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================
# 03_spacy_encode 模块加载 (_SupEncoder / extract_struct_rules, Rule 19 不挪动)
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
# 冻结 encoder 加载 (与 03_spacy_encode stage_adaptive 保存格式对齐)
# ============================================================================

def load_encoder() -> torch.nn.Module:
    """重建 _SupEncoder 并加载 adaptive_encoder.pt (必须 eval, BatchNorm)."""
    ckpt = torch.load(CACHE_DIR / "adaptive_encoder.pt", map_location="cpu",
                      weights_only=False)
    cfg = ckpt["config"]
    _SupEncoder = _load_pcfg()._SupEncoder
    model = _SupEncoder(cfg["vocab_size"], cfg["z_dim"], tuple(cfg["hidden"]),
                        cfg["n_users"], cfg["dropout"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log(f"  encoder loaded: vocab={cfg['vocab_size']} z={cfg['z_dim']} "
        f"hidden={cfg['hidden']} n_users={cfg['n_users']}")
    return model


# ============================================================================
# 预编码 profile/val z + 行→uid 映射
# ============================================================================

def load_precomputed() -> Dict:
    """adaptive_embeddings.npz + user_n_sents.json → per-uid profile/val 索引."""
    npz = np.load(CACHE_DIR / "adaptive_embeddings.npz", allow_pickle=False)
    z_profile = npz["z_profile"]   # (407510, 32)
    z_val = npz["z_val"]           # (163543, 32)
    profile_idx = npz["profile_idx"]
    val_idx = npz["val_idx"]
    uid_list = [str(u) for u in npz["uid_list"]]

    with open(CACHE_DIR / "user_n_sents.json") as f:
        user_n_sents = json.load(f)
    n_total = int(sum(user_n_sents))
    if n_total <= max(int(profile_idx.max()), int(val_idx.max())):
        raise ValueError(
            f"row index out of range: n_total={n_total}, "
            f"max_idx={max(int(profile_idx.max()), int(val_idx.max()))}")
    if len(user_n_sents) != len(uid_list):
        raise ValueError(
            f"user_n_sents ({len(user_n_sents)}) != uid_list ({len(uid_list)})")

    # 行 → uid index (行按 uid_list 顺序连续排列)
    uid_idx_per_row = np.repeat(np.arange(len(uid_list)), user_n_sents)

    prof_row_uid = uid_idx_per_row[profile_idx]
    val_row_uid = uid_idx_per_row[val_idx]
    n_profile_per_uid = np.bincount(prof_row_uid, minlength=len(uid_list))
    n_val_per_uid = np.bincount(val_row_uid, minlength=len(uid_list))

    # 每 uid 的 profile/val 行 mask 预构建 (布尔矩阵 5000×816023 太大, 按需现算)
    uid_to_row = {u: i for i, u in enumerate(uid_list)}
    log(f"  precomputed z: profile={z_profile.shape}, val={z_val.shape}, "
        f"uids={len(uid_list)}")
    return {
        "z_profile": z_profile, "z_val": z_val,
        "profile_idx": profile_idx, "val_idx": val_idx,
        "prof_row_uid": prof_row_uid, "val_row_uid": val_row_uid,
        "n_profile_per_uid": n_profile_per_uid, "n_val_per_uid": n_val_per_uid,
        "uid_to_row": uid_to_row,
    }


# ============================================================================
# candidate query 编码 (同一 encoder / 同一归一化 / 同一规则)
# ============================================================================

def encode_texts(texts: List[str], nlp, rule_to_id: Dict[str, int],
                 vocab_size: int, encoder: torch.nn.Module) -> np.ndarray:
    """texts → spaCy doc → extract_struct_rules (set 语义, 与 cache 一致)
    → 21737 二值计数向量 → normalize (x/(1+x)) → encoder → z (n, 32)."""
    extract_struct_rules = _load_pcfg().extract_struct_rules

    n = len(texts)
    counts = np.zeros((n, vocab_size), dtype=np.float32)
    for i, doc in enumerate(nlp.pipe(texts, batch_size=256)):
        for r in extract_struct_rules(doc):
            j = rule_to_id.get(r)
            if j is not None:            # 与 cache 构建一致: vocab 外规则丢弃
                counts[i, j] = 1.0       # sent_vectors 为二值 (data max=1)
    counts = counts / (1.0 + counts)     # normalize_counts (pipeline :190)
    with torch.no_grad():
        z, _ = encoder(torch.tensor(counts, dtype=torch.float32,
                                    device=ENCODE_DEVICE))
    return z.cpu().numpy().astype(np.float32)


# ============================================================================
# per-user Gaussian + gate
# ============================================================================

def fit_user_gaussian(Z_prof: np.ndarray, Z_val: np.ndarray,
                       gate_quantile: float = 0.75) -> Dict:
    """μ_u, raw Σ_u (no ridge, 与 audit 一致), val self-D² 的 Q_gate gate.

    2026-09-05: 与 audit 阶段 raw_cov_validity.py 保持完全一致 (λ=0).
    inv 奇异 → 抛 RuntimeError (调用方 catch 后该 user 跳过).
    """
    mu = Z_prof.mean(axis=0)
    centered = Z_prof - mu
    cov = (centered.T @ centered) / max(len(Z_prof) - 1, 1)
    if LAMBDA > 0:
        cov = cov + LAMBDA * np.eye(cov.shape[0], dtype=cov.dtype)
    try:
        inv_sigma = np.linalg.inv(cov)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"Σ_u singular (λ={LAMBDA}): {e}") from e

    diff = Z_val - mu
    d2_val = np.einsum("nd,de,ne->n", diff, inv_sigma, diff)
    if np.any(d2_val < 0):
        raise RuntimeError(f"negative Mahalanobis D² (numerical): min={d2_val.min()}")
    gate_T = float(np.quantile(d2_val, gate_quantile))
    return {
        "mu": mu, "inv_sigma": inv_sigma,
        "d2_val": d2_val, "gate_T": gate_T,
        "d2_val_median": float(np.median(d2_val)),
        "gate_quantile": gate_quantile,
    }


def load_audit() -> Dict[str, str]:
    """加载 raw_cov_validity.json, 把 valid_gaussian==True 的 uid 标 healthy,
    其余当 usable. 若文件缺失则全用户当 usable (向后兼容).
    """
    if not MULTI_AUDIT_PATH.exists():
        log(f"  WARNING: {MULTI_AUDIT_PATH} 不存在, 全用户当 usable")
        return {}
    with open(MULTI_AUDIT_PATH) as f:
        au = json.load(f)
    out = {}
    for u, r in au["per_user"].items():
        out[u] = "healthy" if r.get("valid_gaussian") else "usable"
    return out


def maha_d2(Z: np.ndarray, mu: np.ndarray, inv_sigma: np.ndarray) -> np.ndarray:
    diff = Z - mu
    return np.einsum("nd,de,ne->n", diff, inv_sigma, diff)


def maha_d2_one(z: np.ndarray, mu: np.ndarray, inv_sigma: np.ndarray) -> float:
    """单点 Mahalanobis D² (target/competitor 核心区域判定用)."""
    d = z - mu
    return float(d @ inv_sigma @ d)


def cohort_competitor_gates(competitor_uids: List[str], pre: Dict,
                            audit_tier: Dict[str, str],
                            gate_quantile: float = 0.75) -> Dict[str, Dict]:
    """同一 ASIN 的 competitor healthy users 拟合 Gaussian, 返回 gate dict.

    与 target 用同一 gate_quantile (默认 Q0.75 competitor 核心 = 高包容
    "核心内" 区域, query 需落在其外). 奇异 Σ 的 competitor 跳过.
    """
    out: Dict[str, Dict] = {}
    for uid in competitor_uids:
        i = pre["uid_to_row"].get(uid)
        if i is None:
            continue
        if audit_tier.get(uid, "usable") != "healthy":
            continue
        if (pre["n_profile_per_uid"][i] < MIN_PROFILE_SENTS
                or pre["n_val_per_uid"][i] < MIN_VAL_SENTS):
            continue
        p_mask = pre["prof_row_uid"] == i
        v_mask = pre["val_row_uid"] == i
        Z_prof = pre["z_profile"][p_mask]
        Z_val = pre["z_val"][v_mask]
        try:
            gauss = fit_user_gaussian(Z_prof, Z_val, gate_quantile=gate_quantile)
        except RuntimeError:
            continue
        out[uid] = {**gauss, "Z_prof": Z_prof}
    return out


def check_single_fit_unique(query_z: np.ndarray, target: Dict,
                            competitors: Dict[str, Dict]) -> Tuple[bool, Dict]:
    """Stage 8 single-fit 变体: 一次性固定高斯筛选.

    不做 bootstrap 重抽样 / 重拟合. 仅依赖已 fit 的 μ/Σ/gate_T:
      - d_target ≤ target.gate_T (核心内)
      - ∀competitor: d_competitor > comp.gate_T (核心外)

    返回命名: single_fit_unique (与 bootstrap exclusive 区分)
    """
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
    """核心区域排他性 + bootstrap 稳定性.

    判定顺序 (任一失败即 false):
      1. d_target ≤ target.gate_T (target 核心内)
      2. ∀competitor: d_competitor > competitor.gate_T (落在所有 competitor 核心外)
      3. Bootstrap: 重抽样 target profile B 次, 重 fit Gaussian, 统计
         d_target ≤ target.gate_T 的稳定性 (P ≥ p_thresh)

    Returns: (stable, P_pass, debug_info)
    """
    debug = {"d_target": None, "d_competitors": {}, "n_pass_B": 0, "B": B,
             "boot_d2s": None, "fail_stage": None}

    # Step 1: target 核心内
    d_t = maha_d2_one(query_z, target["mu"], target["inv_sigma"])
    debug["d_target"] = d_t
    if d_t > target["gate_T"]:
        debug["fail_stage"] = "target_outside_core"
        return False, 0.0, debug

    # Step 2: 全 competitor 核心外
    for uid, comp in competitors.items():
        d_c = maha_d2_one(query_z, comp["mu"], comp["inv_sigma"])
        debug["d_competitors"][uid] = d_c
        if d_c <= comp["gate_T"]:
            debug["fail_stage"] = f"inside_competitor_core:{uid}"
            return False, 0.0, debug

    # Step 3: Bootstrap stability of target core region
    Z_prof = target["Z_prof"]
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
        if LAMBDA > 0:
            cov_b = cov_b + LAMBDA * np.eye(cov_b.shape[0], dtype=cov_b.dtype)
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
    debug["fail_stage"] = None
    return True, P_pass, debug


# ============================================================================
# 06 生成模块 (regen 复用)
# ============================================================================

def load_gen_module():
    spec = importlib.util.spec_from_file_location(
        "sft_pool_generate", REPO_ROOT / "07_gen_query/sft_pool_generate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================================
# 主流程
# ============================================================================

def main_pipeline():
    log("=== syntax_select_mahalanobis_gate ===")
    log(f"  FILTER_Q={FILTER_Q} (inside-target strict, Q sweep peak: 59 tasks / 48 users / 52 ASINs)")
    log(f"  λ={LAMBDA}  gate_by_tier={GATE_QUANTILE_BY_TIER}  "
        f"COMPETITOR_GATE_QUANTILE={COMPETITOR_GATE_QUANTILE}  "
        f"MIN_PROFILE={MIN_PROFILE_SENTS}  MIN_VAL={MIN_VAL_SENTS}  "
        f"MAX_REGEN_ROUNDS={MAX_REGEN_ROUNDS}  SMOKE={SMOKE}")

    # --- 加载冻结 encoder + 预编码 z ---
    encoder = load_encoder()
    pre = load_precomputed()
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
    # 适配两种结构: 顶层 dict (精简版) 或 {"pool": {asin: [...]}} (旧版)
    if isinstance(pool_data, dict) and "pool" in pool_data \
            and isinstance(pool_data["pool"], dict):
        pool_queries = pool_data["pool"]
    else:
        pool_queries = pool_data   # {asin: [text, ...]}
    with open(ATTRS_PATH) as f:
        attrs_all = json.load(f)
    with open(ASIN_USERS_PATH) as f:
        asin_to_users = json.load(f)

    def _attrs_for(asin: str) -> Dict[str, str]:
        a = attrs_all.get(asin, {})
        return dict(list(a.items())[:5])

    # 展开为旧格式 [{asin, attrs, candidates}]
    pool_entries = []
    for asin, queries in pool_queries.items():
        cands = [{"text": t, "pass": bool(len(t.split()) >= 3)}
                 for t in queries]
        pool_entries.append({"asin": asin, "attrs": _attrs_for(asin),
                             "candidates": cands})
    log(f"  pool queries loaded: {len(pool_entries)} ASINs, "
        f"total candidates={sum(len(p['candidates']) for p in pool_entries)}")

    audit_tier = load_audit()
    log(f"  raw_cov_validity audit: {len(audit_tier)} uids  tiers=" +
        str({t: sum(1 for v in audit_tier.values() if v == t)
             for t in ("healthy", "usable", "unreliable")}))

    if SMOKE:
        pool_entries = [p for p in pool_entries
                        if any(u in pre["uid_to_row"]
                               and pre["n_profile_per_uid"][pre["uid_to_row"][u]] >= MIN_PROFILE_SENTS
                               and pre["n_val_per_uid"][pre["uid_to_row"][u]] >= MIN_VAL_SENTS
                               for u in asin_to_users.get(p["asin"], []))][:N_SMOKE_ASIN]
    log(f"  pool entries to process: {len(pool_entries)}")

    tasks: List[Dict] = []
    skipped: List[Dict] = []
    no_pass_unreliable: List[Dict] = []
    no_pass_singular: List[Dict] = []
    # 2026-09-05: cohort 缓存, 按 asin 复用 competitor gates
    asin_to_competitors: Dict[str, Dict[str, Dict]] = {}
    for entry in pool_entries:
        asin = entry["asin"]
        # 2026-09-05: 每个 ASIN 给所有 healthy user 各选 1 query (per-(asin,uid) task)
        eligible_users: List[Tuple[str, int]] = []
        for u in asin_to_users.get(asin, []):
            i = pre["uid_to_row"].get(u)
            if i is None:
                continue
            if (pre["n_profile_per_uid"][i] >= MIN_PROFILE_SENTS
                    and pre["n_val_per_uid"][i] >= MIN_VAL_SENTS):
                eligible_users.append((u, i))
        if not eligible_users:
            skipped.append({"asin": asin, "reason": "no_eligible_cached_reviewer"})
            continue
        # 2026-09-05: 预先 fit 该 ASIN 所有 healthy competitor 的 Gaussian
        competitor_uids_all = [u for u, _ in eligible_users
                               if audit_tier.get(u, "usable") == "healthy"]
        if asin not in asin_to_competitors:
            asin_to_competitors[asin] = cohort_competitor_gates(
                competitor_uids_all, pre, audit_tier,
                gate_quantile=COMPETITOR_GATE_QUANTILE,
            )
            log(f"    {asin}: {len(asin_to_competitors[asin])} competitor gates fit "
                f"(from {len(competitor_uids_all)} healthy)")
        for uid, ui in eligible_users:
            tier = audit_tier.get(uid, "usable")
            # 仅 healthy user 参选 (用户决策: 质量优先)
            if tier != "healthy":
                continue
            gate_q = GATE_QUANTILE_BY_TIER[tier]
            p_mask = pre["prof_row_uid"] == ui
            v_mask = pre["val_row_uid"] == ui
            Z_prof = pre["z_profile"][p_mask]
            Z_val = pre["z_val"][v_mask]
            try:
                gauss = fit_user_gaussian(Z_prof, Z_val, gate_quantile=gate_q)
            except RuntimeError as e:
                # raw Σ 奇异 (audit E2/E3 应已挡掉, 这里兜底)
                no_pass_singular.append({
                    "asin": asin, "uid": uid, "tier": tier,
                    "reason": f"Σ singular: {e}",
                })
                continue
            # 2026-09-05: competitors = 同 ASIN 其他 healthy (排除自身)
            comps = {c_uid: c for c_uid, c in asin_to_competitors[asin].items()
                     if c_uid != uid}
            tasks.append({
                "asin": asin, "uid": uid, "tier": tier,
                "attrs": entry["attrs"],
                "candidates": list(entry["candidates"]),
                "gauss": gauss,
                "Z_prof": Z_prof,  # 给 check_exclusive bootstrap 用
                "competitors": comps,
                "n_profile": len(Z_prof), "n_val": len(Z_val),
            })
    log(f"  tasks: {len(tasks)}  skipped: {len(skipped)}  "
        f"unreliable_no_pass: {len(no_pass_unreliable)}  "
        f"singular_no_pass: {len(no_pass_singular)}")
    if not tasks:
        raise RuntimeError("no eligible (ASIN, target_user) tasks")
    for t in tasks:
        log(f"    {t['asin']} uid={t['uid']} n_prof={t['n_profile']} "
            f"n_val={t['n_val']} gate_T={t['gauss']['gate_T']:.2f} "
            f"(val D² median {t['gauss']['d2_val_median']:.2f})")

    gen_mod = None  # lazy: 共享 candidates, 不需要 regen

    # --- 选择循环 (per-(asin, uid) 独立 Mahalanobis + tier gate) ---
    # 2026-09-05: 每 ASIN 给所有 healthy user 各选 1 query.
    # candidates 是 user-agnostic 的 (内容级 query), 不随 user 改, 所以
    # 一次性按 asin 编码全部 candidates, 然后对每个 user 重算 D² + gate.
    selections: List[Dict] = []
    no_pass: List[Dict] = []
    all_d2_round0: List[float] = []
    n_gate_pass_round0 = 0
    n_cand_round0 = 0

    # 按 asin 分组 tasks
    asin_to_tasks: Dict[str, List[Dict]] = {}
    for t in tasks:
        asin_to_tasks.setdefault(t["asin"], []).append(t)

    # 一次性编码所有 ASIN 的 candidates (dedup by asin)
    asins = list(asin_to_tasks.keys())
    texts_flat: List[str] = []
    spans: List[Tuple[int, int, str]] = []  # (start, end, asin)
    for a in asins:
        # 用第一个 task 的 candidates (同一 ASIN 内 user 共用)
        cands = asin_to_tasks[a][0]["candidates"]
        start = len(texts_flat)
        texts_flat.extend(c["text"] for c in cands)
        spans.append((start, len(texts_flat), a))
    log(f"  encoding {len(texts_flat)} candidates across {len(asins)} ASINs")
    # 2026-09-06: Q-sweep 模式下, encoding 由 _run_q 预计算并缓存; 跳过重复
    if _QSWEEP_CACHE["Z_all"] is not None and set(_QSWEEP_CACHE["asins"]) == set(asins):
        Z_all = _QSWEEP_CACHE["Z_all"]
        asin_to_Zq = _QSWEEP_CACHE["asin_to_Zq"]
        asin_to_cand_text = _QSWEEP_CACHE["asin_to_cand_text"]
        log(f"  [Q-SWEEP] reusing cached encoding for {len(asins)} ASINs")
    else:
        Z_all = encode_texts(texts_flat, nlp, rule_to_id, V, encoder)
        asin_to_Zq = {}
        asin_to_cand_text = {}
        for s, e, a in spans:
            asin_to_Zq[a] = Z_all[s:e]
            asin_to_cand_text[a] = asin_to_tasks[a][0]["candidates"]
        # 缓存到 module global, 给后续 Q 重用
        _QSWEEP_CACHE["Z_all"] = Z_all
        _QSWEEP_CACHE["asins"] = asins
        _QSWEEP_CACHE["asin_to_Zq"] = asin_to_Zq
        _QSWEEP_CACHE["asin_to_cand_text"] = asin_to_cand_text

    # 对每个 task (per asin × user) 算 D² + 排他性 + bootstrap 选 best unique
    for t in tasks:
        a = t["asin"]
        Z_q = asin_to_Zq[a]
        cands_src = asin_to_cand_text[a]
        # per-task deepcopy: c["d2"] / c["pass_gate"] / c["pass_unique"] 必须 per-user
        cand_records = [{
            "text": c["text"],
            "d2": None,
            "pass_gate": False,
            "content_pass": bool(c.get("pass", False)),
            "round": 0,
        } for c in cands_src]
        d2 = maha_d2(Z_q, t["gauss"]["mu"], t["gauss"]["inv_sigma"])
        gate_T = t["gauss"]["gate_T"]
        # 填每条记录的 d2 (legacy, 报告用)
        for c, d in zip(cand_records, d2):
            c["d2"] = float(d)
            c["pass_gate"] = bool(d <= gate_T)
        # 2026-09-05: 核心区域排他性 + (可选) bootstrap stability 替换单一 gate
        competitors = t["competitors"]
        target_with_Z = {**t["gauss"], "Z_prof": t["Z_prof"]}
        unique_pass: List[Dict] = []   # 通过 3-way 排他性的 cand records
        for i, c in enumerate(cand_records):
            if USE_SINGLE_FIT:
                # single_fit_unique: 一次性固定高斯, 无 bootstrap 声明
                ok, dbg = check_single_fit_unique(Z_q[i], t["gauss"], competitors)
                c["pass_unique"] = bool(ok)
                c["P_pass"] = None
                c["fail_stage"] = dbg.get("fail_stage")
                if ok:
                    unique_pass.append(c)
            else:
                # 完整 bootstrap core exclusive
                stable, P_pass, dbg = check_exclusive(
                    Z_q[i], target_with_Z, competitors,
                    B=BS_N_BOOTSTRAP, p_thresh=BS_P_THRESH, bs_seed=BS_SEED,
                )
                c["pass_unique"] = bool(stable)
                c["P_pass"] = float(P_pass)
                c["fail_stage"] = dbg.get("fail_stage")
                if stable:
                    unique_pass.append(c)
        n_cand_round0 += len(cand_records)
        n_gate_pass_round0 += len([c for c in cand_records if c["pass_gate"]])
        all_d2_round0.extend(c["d2"] for c in cand_records)
        if unique_pass:
            best = min(unique_pass, key=lambda c: c["d2"])
            selections.append({
                "asin": a, "uid": t["uid"],
                "tier": t.get("tier", "usable"),
                "selection_mode": "single_fit_unique" if USE_SINGLE_FIT
                                  else "bootstrap_core_exclusive",
                "gate_quantile_used": t["gauss"].get("gate_quantile"),
                "n_profile": t["n_profile"], "n_val": t["n_val"],
                "gate_T": gate_T,
                "d2_val_median": t["gauss"]["d2_val_median"],
                "selected": best,
                "n_pass_unique": len(unique_pass),
                "n_pass_gate": len([c for c in cand_records if c["pass_gate"]]),
                "n_competitors": len(competitors),
                "n_candidates": len(cand_records),
                "n_regen_rounds_used": 0,
                "P_pass": best["P_pass"],
                "candidates": cand_records,
            })
        else:
            # 2026-09-05: 无 unique candidate → 记 no_unique, 不降级
            fail_stages = [c["fail_stage"] for c in cand_records
                           if c.get("fail_stage")]
            from collections import Counter
            fail_counter = Counter(fail_stages)
            no_pass.append({
                "asin": a, "uid": t["uid"],
                "tier": t.get("tier", "usable"),
                "selection_mode": "single_fit_unique" if USE_SINGLE_FIT
                                  else "bootstrap_core_exclusive",
                "gate_quantile_used": t["gauss"].get("gate_quantile"),
                "gate_T": gate_T,
                "min_d2": min(c["d2"] for c in cand_records),
                "n_candidates": len(cand_records),
                "n_regen_rounds_used": 0,
                "reason": "no_unique",
                "fail_stages": dict(fail_counter.most_common()),
                "n_competitors": len(competitors),
                "candidates": cand_records,
            })

    # --- 汇总输出 ---
    d2_arr = np.array(all_d2_round0, dtype=np.float64)
    # 合并 unreliable 用户到 no_pass (他们根本不参选)
    no_pass_total = no_pass + no_pass_unreliable
    # 2026-09-06: ASIN × unique-user 分布统计
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
        "total_regen_rounds": 0,  # 共享 candidates, 不 regen
        "round0_gate_pass_rate": (n_gate_pass_round0 / max(1, n_cand_round0)),
        "round0_d2_median": float(np.median(d2_arr)) if len(d2_arr) else None,
        "round0_d2_p25": float(np.quantile(d2_arr, 0.25)) if len(d2_arr) else None,
        "round0_d2_p75": float(np.quantile(d2_arr, 0.75)) if len(d2_arr) else None,
        "n_unique_asins": n_unique_asins,                          # 2026-09-06
        "n_asins_ge2_unique_users": n_asins_ge2_unique_users,      # 2026-09-06
    }
    log(f"  SUMMARY: tasks={summary['n_tasks']} selected={summary['n_selected']} "
        f"no_pass={summary['n_no_pass']} (unreliable={summary['n_unreliable_no_select']}) "
        f"skipped={summary['n_skipped']} "
        f"regen_rounds={summary['total_regen_rounds']} "
        f"round0_gate_pass={summary['round0_gate_pass_rate']*100:.1f}% "
        f"round0_d2_median={summary['round0_d2_median']} "
        f"unique_asins={n_unique_asins} asins_ge2_unique_users={n_asins_ge2_unique_users}")

    out = {
        "config": {
            "filter_q": FILTER_Q,         # 2026-09-06: 用户锁定 Q=0.75 (sweep 峰值)
            "lambda": LAMBDA,
            "gate_quantile_by_tier": GATE_QUANTILE_BY_TIER,
            "min_profile_sents": MIN_PROFILE_SENTS, "min_val_sents": MIN_VAL_SENTS,
            "max_regen_rounds": MAX_REGEN_ROUNDS, "regen_k": REGEN_K,
            "smoke": SMOKE,
            "encoder": str(CACHE_DIR / "adaptive_encoder.pt"),
            "precomputed_z": str(CACHE_DIR / "adaptive_embeddings.npz"),
            "pool_path": str(POOL_PATH),
            "spacy_model": SPACY_MODEL,
            "tier_source": str(MULTI_AUDIT_PATH),
            "note": "gate T_u = Q_tier of val self-D² (经验分位数, 不基于 χ² 理论分布); "
                    "selection = argmin D² among D²<=T_u; no fallback; "
                    "unreliable tier 用户不参选, 直接记 no_pass",
        },
        "summary": summary,
        "selections": selections,
        "no_pass": no_pass_total,
        "skipped": skipped,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"  saved -> {OUT_PATH}")
    log("=== syntax_select_mahalanobis_gate DONE ===")


# 2026-09-06: Q-sweep runner. 修改 FILTER_Q + COMPETITOR_GATE_QUANTILE + GATE_QUANTILE_BY_TIER
# 重跑 main_pipeline, 然后改回 0.75. 每个 Q 输出单独的 result 文件.
Q_SWEEP_VALUES = [0.95, 0.75, 0.55, 0.35, 0.25, 0.15, 0.05]
# 跨 Q 共享 encoding 缓存 (在 main_pipeline 内部写, 在 _run_q 入口清空除首次外)
_QSWEEP_CACHE = {"Z_all": None, "asins": None,
                "asin_to_Zq": None, "asin_to_cand_text": None}


def _run_q(q: float) -> None:
    global FILTER_Q, COMPETITOR_GATE_QUANTILE, GATE_QUANTILE_BY_TIER, OUT_PATH
    FILTER_Q = q
    COMPETITOR_GATE_QUANTILE = q
    GATE_QUANTILE_BY_TIER = {"healthy": q, "usable": q, "unreliable": q}
    OUT_PATH = REPO_ROOT / f"result/08_select_query/selected_queries_q{int(q*100):02d}.json"
    log(f"##### Q-SWEEP RUN  Q={q}  OUT={OUT_PATH} #####")
    main_pipeline()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--q":
        # 单 Q 模式: python select_qsweep.py --q 0.95
        q = float(sys.argv[2])
        _run_q(q)
    else:
        # 完整 sweep: 跑 7 个 Q 值, 不改主项目 FILTER_Q
        for q in Q_SWEEP_VALUES:
            _run_q(q)
