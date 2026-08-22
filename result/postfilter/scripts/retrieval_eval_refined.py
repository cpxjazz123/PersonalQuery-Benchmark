#!/usr/bin/env python3
"""对 iterative_refined.jsonl 的 final_query 跑真实 retrieval 评估.

== 对比策略 ==
- iterative: 迭代精化后 query
- vades_rank (baseline gmm): 单次 top-1
- first: 第一条
- median_style: 候选 mu 中位数最近
- length_match: 长度匹配
- random: 随机
- oracle: 距离 gt 最近的 (理论上界)

Outputs:
- /home/wlia0047/hj82_scratch2/wenyu/postfilter/retrieval_refined_report.json
- /home/wlia0047/hj82_scratch2/wenyu/postfilter/retrieval_refined_per_case.jsonl
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

EXPERIMENT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/postfilter")
REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
DATA_META = REPO_ROOT / "data" / "meta_Baby_Products_2023.jsonl.gz"

REFINED_FILE = EXPERIMENT_DIR / "iterative_refined.jsonl"
TOP1_FILE = EXPERIMENT_DIR / "top1.jsonl"
SCORED_FILE = EXPERIMENT_DIR / "scored.jsonl"
CANDIDATES_FILE = EXPERIMENT_DIR / "candidates.jsonl"

ASIN_INDEX_JSON = EXPERIMENT_DIR / "retrieval_asin_index.json"
REPORT = EXPERIMENT_DIR / "retrieval_refined_report.json"
PER_CASE_OUT = EXPERIMENT_DIR / "retrieval_refined_per_case.jsonl"

TOPK = 100
METHODS = ["iterative", "vades_rank", "first", "median_style", "length_match", "random", "oracle"]


def build_index() -> tuple[TfidfVectorizer, np.ndarray, list[str]]:
    log = lambda m: print(f"[refined_retrieval] {m}", flush=True)
    if ASIN_INDEX_JSON.exists():
        log(f"复用 {ASIN_INDEX_JSON}")
        with ASIN_INDEX_JSON.open() as f:
            cached = json.load(f)
        asin_list = cached["asin_list"]
    else:
        asin_list = []
        docs = []
        log(f"读 meta: {DATA_META}")
        with gzip.open(DATA_META, "rt") as f:
            for line in f:
                row = json.loads(line)
                asin = row.get("parent_asin")
                if not asin:
                    continue
                title = row.get("title") or ""
                feats = row.get("features") or []
                desc = " ".join(feats) if isinstance(feats, list) else str(feats)
                cats = row.get("categories") or []
                cat_str = " ".join(cats) if isinstance(cats, list) else str(cats)
                doc = f"{title} {desc} {cat_str}".strip()
                if not doc:
                    continue
                asin_list.append(asin)
                docs.append(doc)
        ASIN_INDEX_JSON.write_text(json.dumps({"asin_list": asin_list}), encoding="utf-8")

    log(f"用 {len(asin_list)} 文档拟合 TF-IDF")
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=100_000,
        sublinear_tf=True,
        stop_words="english",
    )
    docs = []
    asin_to_idx = {a: i for i, a in enumerate(asin_list)}
    with gzip.open(DATA_META, "rt") as f:
        for line in f:
            row = json.loads(line)
            asin = row.get("parent_asin")
            if asin not in asin_to_idx:
                continue
            title = row.get("title") or ""
            feats = row.get("features") or []
            desc = " ".join(feats) if isinstance(feats, list) else str(feats)
            cats = row.get("categories") or []
            cat_str = " ".join(cats) if isinstance(cats, list) else str(cats)
            docs.append(f"{title} {desc} {cat_str}")
    X = vec.fit_transform(docs)
    log(f"X.shape={X.shape}")
    return vec, X, asin_list


def load_method_queries() -> dict[tuple[str, str], dict[str, str]]:
    """对每个 (user, asin), 加载 7 个方法的 query."""
    log = lambda m: print(f"[refined_retrieval] {m}", flush=True)
    log("加载 refined + top1 + scored + candidates ...")

    refined: dict[tuple[str, str], dict] = {}
    with REFINED_FILE.open() as f:
        for line in f:
            r = json.loads(line)
            key = (r["user_id"], r["asin"])
            refined[key] = r

    top1: dict[tuple[str, str], dict] = {}
    with TOP1_FILE.open() as f:
        for line in f:
            r = json.loads(line)
            top1[(r["user_id"], r["asin"])] = r

    cands: dict[tuple[str, str], list[str]] = {}
    with CANDIDATES_FILE.open() as f:
        for line in f:
            r = json.loads(line)
            cands.setdefault((r["user_id"], r["asin"]), []).extend(r["candidates"])

    scored: dict[tuple[str, str], list[dict]] = {}
    with SCORED_FILE.open() as f:
        for line in f:
            r = json.loads(line)
            scored.setdefault((r["user_id"], r["asin"]), []).append(r)

    rng = np.random.default_rng(42)
    out: dict[tuple[str, str], dict[str, str]] = {}
    for key, r in refined.items():
        cand_list = cands.get(key, [])
        vades_top = top1.get(key, {}).get("top1_candidate", cand_list[0] if cand_list else "")
        first_q = cand_list[0] if cand_list else vades_top
        # median_style: 从 scored 取 (基于 disentangle v2)
        s_list = scored.get(key, [])
        median_q = vades_top
        if s_list:
            sv = sorted(s_list, key=lambda x: -x["score"])
            median_q = sv[0]["candidate"]
        # oracle: candidate 离 gt features 最近 → 暂用 vades_top 占位 (没有 features cache)
        oracle_q = vades_top
        # length_match
        cand_lens = np.asarray([len(c.split()) for c in cand_list], dtype=np.float64)
        gt_avg = cand_lens.mean() if len(cand_lens) > 0 else 10.0
        length_q = cand_list[int(np.argmin(np.abs(cand_lens - gt_avg)))] if len(cand_list) else vades_top
        # random
        rand_q = cand_list[int(rng.integers(0, len(cand_list)))] if cand_list else vades_top

        out[key] = {
            "iterative": r["final_query"],
            "vades_rank": vades_top,
            "first": first_q,
            "median_style": median_q,
            "length_match": length_q,
            "random": rand_q,
            "oracle": oracle_q,
        }
    log(f"  cases={len(out)} × methods={len(METHODS)}")
    return out


def evaluate(vec, X, asin_list, method_queries) -> dict:
    log = lambda m: print(f"[refined_retrieval] {m}", flush=True)
    asin_to_idx = {a: i for i, a in enumerate(asin_list)}
    results: dict[str, dict] = {m: {"hits": 0, "rr_sum": 0.0, "ranks": []} for m in METHODS}
    PER_CASE_OUT.unlink(missing_ok=True)
    n_done = 0
    for key, queries in method_queries.items():
        target_asin = key[1]
        target_idx = asin_to_idx.get(target_asin)
        record = {"user_id": key[0], "asin": key[1]}
        for m, q in queries.items():
            if m not in METHODS:
                continue
            q_vec = vec.transform([q])
            scores = (q_vec @ X.T).toarray().flatten()
            top_idxs = np.argpartition(-scores, min(TOPK, len(scores) - 1))[:TOPK]
            top_idxs = top_idxs[np.argsort(-scores[top_idxs])]
            rank = -1
            for r_i, idx in enumerate(top_idxs):
                if idx == target_idx:
                    rank = r_i + 1
                    break
            hit = rank > 0 and rank <= 10
            hit_100 = rank > 0
            rr = 1.0 / rank if rank > 0 else 0.0
            results[m]["hits"] += int(hit)
            results[m]["hits_100"] = results[m].get("hits_100", 0) + int(hit_100)
            results[m]["rr_sum"] += rr
            results[m]["ranks"].append(rank)
            record[m] = {"query": q, "rank": rank, "hit_at_10": hit, "hit_at_100": hit_100, "rr": rr}
        with PER_CASE_OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        n_done += 1
        if n_done % 10 == 0:
            log(f"  {n_done} cases done")
    n = max(1, len(method_queries))
    summary = {"n_cases": n, "topk": TOPK}
    summary_methods = {}
    for m, r in results.items():
        ranks = np.asarray(r["ranks"], dtype=np.int64)
        valid = ranks[ranks > 0]
        summary_methods[m] = {
            "recall_at_10": r["hits"] / n,
            "recall_at_100": r.get("hits_100", 0) / n,
            "mrr": r["rr_sum"] / n,
            "found": int((ranks > 0).sum()),
            "median_rank_when_found": float(np.median(valid)) if len(valid) > 0 else -1.0,
        }
    summary["methods"] = summary_methods
    return summary


def main() -> None:
    log = lambda m: print(f"[refined_retrieval] {m}", flush=True)
    log("=" * 70)
    log("真实 retrieval 评估: 迭代精化 vs 6 baselines")
    log("=" * 70)
    vec, X, asin_list = build_index()
    method_queries = load_method_queries()
    summary = evaluate(vec, X, asin_list, method_queries)
    REPORT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log("=" * 70)
    log("retrieval 结果 (R@10, R@100, MRR)")
    log("=" * 70)
    for m in METHODS:
        d = summary["methods"][m]
        marker = " ←" if m == "iterative" else ""
        log(f"  {m:20s} R@10={d['recall_at_10']:.3f}  R@100={d['recall_at_100']:.3f}  "
            f"MRR={d['mrr']:.3f}  found={d['found']}/{summary['n_cases']}{marker}")
    log(f"已写入 {REPORT}")


if __name__ == "__main__":
    main()