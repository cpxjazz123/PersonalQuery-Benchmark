#!/usr/bin/env python3
"""Query selection: per-user Mahalanobis D² + validation-quantile gate.

Refactor 2026-09-06: per-user Gaussian 与 per-(asin, uid) cohort membership
由 Stage 04 的单一 canonical artifact 提供，08 只做 candidate query 编码、
选择判定与输出，不在运行时重新拟合 Gaussian。

流程:
  1. 加载 result/04_gaussian/user_gaussian_stats.json 的 users/cohort_gates
  2. 加载 frozen encoder + vocab, 编码候选 pool queries → 32d z_q
  3. 对每个 (asin, uid) task:
     a. D²(z_q, μ_u) ≤ u.d2_q95 (gate_T)   ← target 核心内
     b. ∀comp ∈ cohort: D²(z_q, μ_comp) > comp.d2_q95 ← 核心外
  4. 通过者中选 D² 最小
  5. 每 ASIN 给 Stage 04 cohort 中的用户各选 1 query, ≥2 unique user 才保留

复用资产:
  - 冻结 encoder: pcfg_cache/adaptive_encoder.pt (_SupEncoder 21737→256→32, eval)
  - vocab:        pcfg_cache/vocab.json (21737 规则)
  - Stage 04 canonical Gaussian + cohort gates:
    result/04_gaussian/user_gaussian_stats.json
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
# Stage 04 是 Gaussian 与 cohort membership 的唯一 canonical source。
STAGE04_PATH = REPO_ROOT / "result/04_gaussian/user_gaussian_stats.json"
USER_STATS_PATH = STAGE04_PATH
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- Hardcoded hyperparams (Rule 3) ---
FILTER_Q = 0.95
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
ENCODE_DEVICE = "cpu"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
# Stage 04 产物加载 (per-user Gaussian + cohort gates)
# ============================================================================

def load_stage04() -> tuple[Dict[str, Dict], Dict[str, Dict[str, Dict]], dict]:
    """严格加载 Stage 04 canonical artifact 及其 users/cohort_gates。"""
    if not STAGE04_PATH.exists():
        raise FileNotFoundError(
            f"missing: {STAGE04_PATH} (run 04_gaussian/fit_per_user_gaussian.py)"
        )
    with open(STAGE04_PATH) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Stage 04 artifact must be a JSON object")
    users = data.get("users")
    cohort_gates = data.get("cohort_gates")
    config = data.get("config")
    if not isinstance(users, dict) or not users:
        raise ValueError("Stage 04 artifact requires non-empty 'users'")
    if not isinstance(cohort_gates, dict) or not cohort_gates:
        raise ValueError("Stage 04 artifact requires non-empty 'cohort_gates'")
    if not isinstance(config, dict) or config.get("gate_quantile") != FILTER_Q:
        raise ValueError(
            f"Stage 04 gate_quantile must equal FILTER_Q={FILTER_Q}; "
            f"got {None if not isinstance(config, dict) else config.get('gate_quantile')}"
        )
    for asin, cohort in cohort_gates.items():
        if not isinstance(cohort, dict) or len(cohort) < 2:
            raise ValueError(f"cohort {asin} must contain at least two users")
        for uid, gate in cohort.items():
            if uid not in users:
                raise ValueError(f"cohort {asin} references missing user {uid}")
            if not isinstance(gate, dict) or "gate_T" not in gate:
                raise ValueError(f"cohort {asin}/{uid} missing gate_T")
    log(f"  loaded Stage 04: {len(users)} users, {len(cohort_gates)} ASINs, "
        f"{sum(len(c) for c in cohort_gates.values())} cohort pairs")
    return users, cohort_gates, config


def _gauss_from_stats(stats: Dict) -> Dict:
    """把 Stage 04 user stats 转成 selection 所需的 numpy 结构。"""
    required = ("mu", "sigma_inv", "n", "n_val", "d2_q50", "d2_q95")
    missing = [key for key in required if key not in stats]
    if missing:
        raise ValueError(f"Stage 04 user stats missing fields: {missing}")
    mu = np.asarray(stats["mu"], dtype=np.float32)
    inv_sigma = np.asarray(stats["sigma_inv"], dtype=np.float32)
    if mu.shape != (32,) or inv_sigma.shape != (32, 32):
        raise ValueError(f"invalid Gaussian shape: mu={mu.shape}, sigma_inv={inv_sigma.shape}")
    return {
        "mu": mu,
        "inv_sigma": inv_sigma,
        "gate_T": float(stats["d2_q95"]),
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
    extract_struct_rules = _load_pcfg().extract_struct_rules
    n = len(texts)
    counts = np.zeros((n, vocab_size), dtype=np.float32)
    for i, doc in enumerate(nlp.pipe(texts, batch_size=256)):
        for r in extract_struct_rules(doc):
            j = rule_to_id.get(r)
            if j is not None:
                counts[i, j] = 1.0
    counts = counts / (1.0 + counts)
    with torch.no_grad():
        z, _ = encoder(torch.tensor(counts, dtype=torch.float32,
                                    device=ENCODE_DEVICE))
    return z.cpu().numpy().astype(np.float32)


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

    tasks: List[Dict] = []
    skipped: List[Dict] = []
    no_pass_unreliable: List[Dict] = []
    for entry in pool_entries:
        asin = entry["asin"]
        comps_for_asin = cohort_gates_map.get(asin)
        if comps_for_asin is None:
            skipped.append({"asin": asin, "reason": "no_stage04_cohort"})
            continue
        for uid in sorted(comps_for_asin):
            if uid not in user_stats:
                raise ValueError(f"Stage 04 cohort {asin} references missing user {uid}")
            gauss = _gauss_from_stats(user_stats[uid])
            comps = {
                c_uid: _gauss_from_stats(user_stats[c_uid])
                for c_uid in comps_for_asin
                if c_uid != uid
            }
            if not comps:
                raise ValueError(f"Stage 04 cohort {asin} has no competitor for {uid}")
            tasks.append({
                "asin": asin, "uid": uid, "tier": "stage04_fitted",
                "attrs": entry["attrs"],
                "candidates": list(entry["candidates"]),
                "gauss": gauss,
                "competitors": comps,
            })
    log(f"  tasks: {len(tasks)}  skipped: {len(skipped)}")
    if not tasks:
        raise RuntimeError("no eligible (ASIN, target_user) tasks")

    for t in tasks[:5]:
        log(f"    {t['asin']} uid={t['uid']} gate_T={t['gauss']['gate_T']:.2f} "
            f"(d2_q50={t['gauss']['d2_val_median']:.2f})")
    if len(tasks) > 5:
        log(f"    ... and {len(tasks) - 5} more tasks")

    # --- 一次性编码所有 ASIN 的 candidates (按 asin 分组) ---
    asin_to_tasks: Dict[str, List[Dict]] = {}
    for t in tasks:
        asin_to_tasks.setdefault(t["asin"], []).append(t)

    asins = list(asin_to_tasks.keys())
    texts_flat: List[str] = []
    spans: List[Tuple[int, int, str]] = []
    for a in asins:
        cands = asin_to_tasks[a][0]["candidates"]
        start = len(texts_flat)
        texts_flat.extend(c["text"] for c in cands)
        spans.append((start, len(texts_flat), a))
    log(f"  encoding {len(texts_flat)} candidates across {len(asins)} ASINs")
    Z_all = encode_texts(texts_flat, nlp, rule_to_id, V, encoder)

    asin_to_Zq: Dict[str, np.ndarray] = {}
    asin_to_cand_text: Dict[str, List[Dict]] = {}
    for s, e, a in spans:
        asin_to_Zq[a] = Z_all[s:e]
        asin_to_cand_text[a] = asin_to_tasks[a][0]["candidates"]

    # --- 选择循环 ---
    selections: List[Dict] = []
    no_pass: List[Dict] = []
    all_d2_round0: List[float] = []
    n_gate_pass_round0 = 0
    n_cand_round0 = 0

    for t in tasks:
        a = t["asin"]
        Z_q = asin_to_Zq[a]
        cands_src = asin_to_cand_text[a]
        cand_records = [{
            "text": c["text"],
            "d2": None,
            "pass_gate": False,
            "content_pass": bool(c.get("pass", False)),
            "round": 0,
        } for c in cands_src]
        d2 = maha_d2(Z_q, t["gauss"]["mu"], t["gauss"]["inv_sigma"])
        gate_T = t["gauss"]["gate_T"]
        for c, d in zip(cand_records, d2):
            c["d2"] = float(d)
            c["pass_gate"] = bool(d <= gate_T)
        competitors = t["competitors"]
        unique_pass: List[Dict] = []
        for i, c in enumerate(cand_records):
            if USE_SINGLE_FIT:
                ok, dbg = check_single_fit_unique(Z_q[i], t["gauss"], competitors)
                c["pass_unique"] = bool(ok)
                c["P_pass"] = None
                c["fail_stage"] = dbg.get("fail_stage")
                if ok:
                    unique_pass.append(c)
            else:
                stable, P_pass, dbg = check_exclusive(
                    Z_q[i], t["gauss"], competitors,
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
                "tier": t.get("tier", "stage04_fitted"),
                "selection_mode": "single_fit_unique" if USE_SINGLE_FIT
                                  else "bootstrap_core_exclusive",
                "gate_quantile_used": t["gauss"].get("gate_quantile"),
                "n_profile": t["gauss"]["n"], "n_val": t["gauss"]["n_val"],
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
            from collections import Counter
            fail_stages = [c["fail_stage"] for c in cand_records
                           if c.get("fail_stage")]
            fail_counter = Counter(fail_stages)
            no_pass.append({
                "asin": a, "uid": t["uid"],
                "tier": t.get("tier", "stage04_fitted"),
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
    _st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    filtered_blocks = []
    for blk in asin_blocks:
        texts = [u["query"] for u in blk["users"]]
        emb = _st_model.encode(texts, convert_to_tensor=True,
                               normalize_embeddings=True, show_progress_bar=False)
        cos = emb @ emb.T
        n = len(texts)
        if n == 2:
            max_cos = float(cos[0, 1])
        else:
            mask = ~torch.eye(n, dtype=torch.bool, device=cos.device)
            max_cos = float(cos[mask].max())
        if max_cos >= SEMANTIC_SIM_THRESHOLD:
            blk["max_pair_cos"] = max_cos
            filtered_blocks.append(blk)
    asin_blocks = filtered_blocks
    n_asin_blocks = len(asin_blocks)

    out = {
        "config": {
            "filter_q": FILTER_Q,
            "min_profile_sents": MIN_PROFILE_SENTS,
            "min_val_sents": MIN_VAL_SENTS,
            "smoke": SMOKE,
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
            {"asin": np["asin"], "uid": np["uid"]}
            for np in no_pass_total
        ],
        "skipped": skipped,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"  saved -> {OUT_PATH}")
    log("=== syntax_select_mahalanobis_gate DONE ===")


if __name__ == "__main__":
    main_pipeline()
