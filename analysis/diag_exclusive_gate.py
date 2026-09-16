#!/usr/bin/env python3
"""Stage 8 exclusive-gate 诊断:实测 cross-Gaussian overlap。

不在生产代码上改动。仅读 Stage 04 artifact + Stage 07 pool + Stage 02 sentences,
重算每个 ASIN 的 d2_matrix,按 cohort_size 分桶,统计

  competitor_inside.sum(axis=1)  的分布。

目的是回答"cohort_size↑ ⇒ 唯一通过率↓"这个经验规律是否真是
由于 competitor 命中数 K=any 引起的 multiplicity penalty。

输出: result/05_gaussian_audit/exclusive_gate_overlap.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "08_select_query"))

# 直接 import Stage 8 模块来复用 load_encoder / encode_texts / load_stage04 等
# 但不调用其 main_pipeline,只借用工具函数
import importlib.util

def _import_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

s8 = _import_module(
    "stage8_tools",
    str(REPO_ROOT / "08_select_query/syntax_select_mahalanobis_gate.py"),
)

STAGE04_PATH = REPO_ROOT / "result/04_gaussian/user_gaussian_stats.json"
POOL_PATH = REPO_ROOT / "result/07_gen_query/pool_queries.json"
SFT_TRAIN_PATH = REPO_ROOT / "result/06_training_model/sft_train_asins.json"
OUT_PATH = REPO_ROOT / "result/05_gaussian_audit/exclusive_gate_overlap.json"

# 采样规模:全量 205K tasks 太大,按 ASIN 抽样 + 控制 cohort_size 分布
SAMPLE_N_ASINS_PER_BUCKET = 60  # 每个 cohort_size bucket 取 60 个 ASIN
COHORT_BUCKETS = [2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 30]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_pool() -> Dict[str, List[str]]:
    with open(POOL_PATH) as f:
        doc = json.load(f)
    return doc["pool"]


def build_asin_work(
    pool: Dict[str, List[str]],
    users_all: Dict[str, Dict],
    cohort_gates: Dict[str, Dict[str, Dict]],
    excluded_asins: set,
):
    """复刻 Stage 8 asin_work 构造:每 ASIN × target uids."""
    asin_work = {}
    for asin, queries in pool.items():
        if asin in excluded_asins:
            continue
        if asin not in cohort_gates:
            continue
        cohort = cohort_gates[asin]
        target_uids = [u for u in cohort.keys() if u in users_all]
        if len(target_uids) < 2:
            continue
        asin_work[asin] = {
            "asin": asin,
            "candidates": [{"text": q, "pass": True} for q in queries],
            "uids": target_uids,
            "cohort_uids": target_uids,
            "cohort": cohort,
        }
    return asin_work


def main():
    log("=== diagnostic: Stage 8 exclusive gate cross-overlap ===")
    log(f"  loading Stage 04 ...")
    users_all, cohort_gates, stage04_cfg = s8.load_stage04()
    log(f"  Stage 04 users: {len(users_all)}  cohort_gates: {len(cohort_gates)}")

    log("  loading pool ...")
    pool = load_pool()
    log(f"  pool: {len(pool)} ASINs")

    excluded = set()
    if SFT_TRAIN_PATH.exists():
        with open(SFT_TRAIN_PATH) as f:
            excluded = set(json.load(f).get("train_asins", []))
    log(f"  excluded train asins: {len(excluded)}")

    asin_work = build_asin_work(pool, users_all, cohort_gates, excluded)
    log(f"  asin_work (pool ∩ cg ∩ users_all, ≥2 uids): {len(asin_work)} ASINs")

    # 按 cohort_size 分桶
    buckets: Dict[int, List[str]] = {b: [] for b in COHORT_BUCKETS}
    buckets["other"] = []
    for asin, work in asin_work.items():
        cs = len(work["cohort_uids"])
        placed = False
        for b in COHORT_BUCKETS:
            if cs == b or (b == 4 and 4 <= cs <= 5) or \
               (b == 10 and 6 <= cs <= 10) or \
               (b == 15 and 11 <= cs <= 15) or \
               (b == 20 and 16 <= cs <= 20) or \
               (b == 30 and 21 <= cs <= 30):
                buckets[b].append(asin)
                placed = True
                break
        if not placed:
            buckets["other"].append(asin)

    # 重新定义 bucket 语义为单一 cohort_size 桶
    bucket_def = [
        ("cs=2", 2, 2), ("cs=3", 3, 3), ("cs=4", 4, 4), ("cs=5", 5, 5),
        ("cs=6", 6, 6), ("cs=7", 7, 7), ("cs=8", 8, 8),
        ("cs=9-10", 9, 10), ("cs=11-15", 11, 15),
        ("cs=16-20", 16, 20), ("cs=21-30", 21, 30),
        ("cs>=31", 31, 10**9),
    ]
    sized = {name: [] for name, _, _ in bucket_def}
    for asin, work in asin_work.items():
        cs = len(work["cohort_uids"])
        for name, lo, hi in bucket_def:
            if lo <= cs <= hi:
                sized[name].append(asin)
                break

    log("  cohort_size distribution among sampled-eligible ASINs:")
    for name, _, _ in bucket_def:
        log(f"    {name}: {len(sized[name])} ASINs")

    # 抽样
    rng = np.random.default_rng(42)
    sample: Dict[str, List[str]] = {}
    for name, _, _ in bucket_def:
        asins = sized[name]
        n = min(SAMPLE_N_ASINS_PER_BUCKET, len(asins))
        if n == 0:
            continue
        idx = rng.choice(len(asins), size=n, replace=False)
        sample[name] = [asins[i] for i in idx]
    n_sample_asins = sum(len(v) for v in sample.values())
    n_sample_pairs = sum(len(asin_work[a]["uids"]) for v in sample.values() for a in v)
    log(f"  sampled: {n_sample_asins} ASINs, {n_sample_pairs} (asin,uid) tasks")

    # 收集 candidate texts 全局去重
    unique_text_to_idx: Dict[str, int] = {}
    unique_texts: List[str] = []
    asin_text_indices: Dict[str, np.ndarray] = {}
    for name, asins in sample.items():
        for asin in asins:
            indices = []
            for c in asin_work[asin]["candidates"]:
                t = c["text"]
                j = unique_text_to_idx.get(t)
                if j is None:
                    j = len(unique_texts)
                    unique_text_to_idx[t] = j
                    unique_texts.append(t)
                indices.append(j)
            asin_text_indices[asin] = np.asarray(indices, dtype=np.int64)
    log(f"  unique texts: {len(unique_texts)}")

    log("  loading encoder + spaCy ...")
    encoder = s8.load_encoder()
    V = encoder.encoder[0].in_features
    CACHE_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache")
    with open(CACHE_DIR / "vocab.json") as f:
        vocab = json.load(f)
    if len(vocab) != V:
        raise ValueError(f"vocab ({len(vocab)}) != encoder input dim ({V})")
    rule_to_id = {r: i for i, r in enumerate(vocab)}
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger"])
    log(f"  V={V}  vocab={len(vocab)}")

    log(f"  encoding {len(unique_texts)} texts ...")
    Z_unique = s8.encode_texts(unique_texts, nlp, rule_to_id, V, encoder)
    if Z_unique.shape != (len(unique_texts), 32):
        raise ValueError(f"encoder shape mismatch: {Z_unique.shape}")
    asin_to_Zq = {
        a: Z_unique[asin_text_indices[a]] for name, asins in sample.items() for a in asins
    }

    # 主诊断:按 bucket 统计 candidate 上的 competitor_inside.sum
    bucket_stats: Dict[str, dict] = {}
    for name, asins in sample.items():
        log(f"  === bucket {name} ({len(asins)} ASINs) ===")
        per_cand_competitor_hits: List[int] = []
        per_target_unique_pass: List[int] = []
        per_target_pass_gate: List[int] = []
        per_target_n_cands: List[int] = []
        per_asin_unique_count: Dict[str, int] = {}
        n_target_users_total = 0
        n_candidates_total = 0
        t0 = time.time()
        for asin in asins:
            Z_q = asin_to_Zq[asin]
            cohort_uids = asin_work[asin]["cohort_uids"]
            # 把 Stage 04 raw stats 转成 Stage 8 selection 用的 numpy 结构
            # (mu/sigma_inv 是 list;gate_T 来自 d2_q95,跟 _gauss_from_stats 一致)
            gauss_cache_local = {
                u: s8._gauss_from_stats(users_all[u]) for u in cohort_uids
            }
            mu = np.stack([gauss_cache_local[u]["mu"] for u in cohort_uids], axis=0)
            inv_sigma = np.stack([gauss_cache_local[u]["inv_sigma"] for u in cohort_uids], axis=0)
            gate_Ts = np.asarray(
                [gauss_cache_local[u]["gate_T"] for u in cohort_uids],
                dtype=np.float64,
            )
            diff = Z_q[:, None, :] - mu[None, :, :]
            left = np.einsum("cud,ude->cue", diff, inv_sigma)
            d2_matrix = np.sum(left * diff, axis=2, dtype=np.float32)

            target_index = {u: i for i, u in enumerate(cohort_uids)}
            asin_unique_users = 0
            for uid in cohort_uids:
                ti = target_index[uid]
                target_d2 = d2_matrix[:, ti]
                gate_T = float(gate_Ts[ti])
                target_inside = target_d2 <= gate_T
                competitor_inside = d2_matrix <= gate_Ts[None, :]
                competitor_inside[:, ti] = False
                pass_unique = target_inside & (~competitor_inside.any(axis=1))
                per_cand_competitor_hits.extend(int(x) for x in competitor_inside.sum(axis=1))
                per_target_pass_gate.append(int(target_inside.sum()))
                per_target_unique_pass.append(int(pass_unique.sum()))
                per_target_n_cands.append(len(Z_q))
                n_target_users_total += 1
                n_candidates_total += len(Z_q)
                if pass_unique.any():
                    asin_unique_users += 1
            per_asin_unique_count[asin] = asin_unique_users
        log(f"    computed in {time.time()-t0:.1f}s")

        # candidate-level: P(competitor_hits = k)
        hits = np.asarray(per_cand_competitor_hits, dtype=np.int32)
        n_total_cands = len(hits)
        max_k = int(hits.max()) if n_total_cands else 0
        k_dist = {int(k): int((hits == k).sum()) for k in range(max_k + 1)}
        # cumulative: P(hits <= K)
        cum = {}
        running = 0
        for k in range(max_k + 1):
            running += k_dist.get(k, 0)
            cum[k] = running
        n_unique_pass = sum(per_target_unique_pass)
        n_pass_gate = sum(per_target_pass_gate)
        n_cands_total = sum(per_target_n_cands)

        # ASIN-level: how many asins have ≥2 uids with ≥1 unique candidate?
        asin_ge2 = sum(1 for v in per_asin_unique_count.values() if v >= 2)
        asin_ge1 = sum(1 for v in per_asin_unique_count.values() if v >= 1)

        bucket_stats[name] = {
            "n_asins_sampled": len(asins),
            "n_target_users_total": n_target_users_total,
            "n_candidates_total": n_cands_total,
            "n_pass_gate": n_pass_gate,
            "n_pass_unique": n_unique_pass,
            "pass_gate_rate": n_pass_gate / max(1, n_cands_total),
            "pass_unique_rate": n_pass_unique / max(1, n_cands_total),
            "pass_unique_per_target_avg": n_unique_pass / max(1, n_target_users_total),
            "asin_unique_count_dist": dict(
                sorted(Counter(per_asin_unique_count.values()).items())),
            "asin_with_ge1_unique_user": asin_ge1,
            "asin_with_ge2_unique_users": asin_ge2,
            "competitor_hit_k_dist": k_dist,
            "competitor_hit_cum": cum,
            "p_competitor_hit_eq_0": k_dist.get(0, 0) / max(1, n_total_cands),
            "p_competitor_hit_eq_1": k_dist.get(1, 0) / max(1, n_total_cands),
            "p_competitor_hit_le_1": cum.get(1, 0) / max(1, n_total_cands),
            "p_competitor_hit_le_2": cum.get(2, 0) / max(1, n_total_cands),
            "p_competitor_hit_le_5": cum.get(5, 0) / max(1, n_total_cands),
        }
        log(f"    pass_gate_rate={bucket_stats[name]['pass_gate_rate']:.4f}  "
            f"pass_unique_rate={bucket_stats[name]['pass_unique_rate']:.6f}")
        log(f"    P(competitor_hits=0)={bucket_stats[name]['p_competitor_hit_eq_0']:.4f}  "
            f"P(<=1)={bucket_stats[name]['p_competitor_hit_le_1']:.4f}  "
            f"P(<=2)={bucket_stats[name]['p_competitor_hit_le_2']:.4f}  "
            f"P(<=5)={bucket_stats[name]['p_competitor_hit_le_5']:.4f}")
        log(f"    asin ≥1 unique user: {asin_ge1}/{len(asins)}  "
            f"≥2 unique users: {asin_ge2}/{len(asins)}")

    out = {
        "config": {
            "sample_n_per_bucket": SAMPLE_N_ASINS_PER_BUCKET,
            "buckets": [name for name, _, _ in bucket_def],
            "stage04_source": str(STAGE04_PATH),
            "pool_source": str(POOL_PATH),
            "n_unique_texts": len(unique_texts),
            "gate_quantile": stage04_cfg.get("gate_quantile", "Q=0.95"),
            "encoder": str(REPO_ROOT / "result/03_spacy_encode/adaptive_encoder.pt"),
        },
        "buckets": bucket_stats,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"  wrote → {OUT_PATH}")


if __name__ == "__main__":
    from collections import Counter
    main()
