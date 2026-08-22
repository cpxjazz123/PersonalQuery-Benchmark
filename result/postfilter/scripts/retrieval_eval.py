#!/usr/bin/env python3
"""真实 retrieval 评估: 看每个方法的 top-1 query 能否召回到目标 asin.

== 流程 ==
1. 用 TF-IDF 索引 meta_Baby_Products (217k asin, title + description + categories 拼接)
2. 对每个 (user, asin) case, 把 eval_per_case.jsonl 里 9 个方法的 top-1 query 喂进 TF-IDF
3. 计算 top-K ranking, 标记目标 asin 的位置
4. 汇总 recall@10, recall@100, MRR, NDCG@10

Outputs:
- /home/wlia0047/hj82_scratch2/wenyu/postfilter/retrieval_report.json
- /home/wlia0047/hj82_scratch2/wenyu/postfilter/retrieval_per_case.jsonl
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
EXPERIMENT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/postfilter")
DATA_META = REPO_ROOT / "data" / "meta_Baby_Products_2023.jsonl.gz"
EVAL_PER_CASE = EXPERIMENT_DIR / "eval_per_case.jsonl"
TOP1 = EXPERIMENT_DIR / "top1.jsonl"
SCORED = EXPERIMENT_DIR / "scored.jsonl"
CANDIDATES = EXPERIMENT_DIR / "candidates.jsonl"
REPORT = EXPERIMENT_DIR / "retrieval_report.json"
PER_CASE_OUT = EXPERIMENT_DIR / "retrieval_per_case.jsonl"
ASIN_INDEX_JSON = EXPERIMENT_DIR / "retrieval_asin_index.json"

TOPK = 100
METHODS = [
    "vades_rank", "vades_minus_central", "vades_top3_median",
    "tfidf", "length_match", "first",
    "random", "median_style", "oracle",
]


def build_index() -> tuple[TfidfVectorizer, np.ndarray, list[str]]:
    """meta_Baby_Products 全文 TF-IDF index."""
    log = lambda m: print(f"[retrieval] {m}", flush=True)
    if ASIN_INDEX_JSON.exists():
        log(f"复用 {ASIN_INDEX_JSON}")
        with ASIN_INDEX_JSON.open() as f:
            cached = json.load(f)
        asin_list = cached["asin_list"]
        # 注: TF-IDF 矩阵本身不缓存 (太大), 重新拟合 (1 min)
    else:
        log(f"读 meta: {DATA_META}")
        asin_list = []
        docs = []
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
        log(f"docs={len(docs)}")
        ASIN_INDEX_JSON.write_text(json.dumps({"asin_list": asin_list}), encoding="utf-8")
        log(f"已写入 {ASIN_INDEX_JSON}")

    log(f"用 {len(asin_list)} 文档拟合 TF-IDF")
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=100_000,
        sublinear_tf=True,
        stop_words="english",
    )
    # 重新读 docs 拟合 (索引文件只存 asin)
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


def load_method_top1s() -> dict[tuple[str, str], dict[str, str]]:
    """对每个 (user,asin), 对每个 method 算 top-1 query (基于 eval_per_case.jsonl)."""
    log = lambda m: print(f"[retrieval] {m}", flush=True)
    log("加载 eval_per_case.jsonl")
    per_case: dict[tuple[str, str], dict] = {}
    with EVAL_PER_CASE.open() as f:
        for line in f:
            r = json.loads(line)
            key = (r["user_id"], r["asin"])
            per_case[key] = r

    # 配 reading: 把 eval_per_case 里每个 method 的 candidate 拿出来
    # 1. vades_rank: top1.top1_candidate
    # 2. vades_minus_central: 需要从 scored.jsonl 推
    # 3. vades_top3_median: 同样需要推
    # 4. tfidf/length_match/median_style/first: index of top-1 candidate
    # 5. random: 直接使用 eval_per_case.record 或 random sample
    # 6. oracle: 从 eval_per_case.oracle candidate

    # 预读 scored.jsonl + candidates.jsonl
    scored: dict[tuple[str, str], list[dict]] = {}
    with SCORED.open() as f:
        for line in f:
            r = json.loads(line)
            scored.setdefault((r["user_id"], r["asin"]), []).append(r)
    cands: dict[tuple[str, str], list[str]] = {}
    with CANDIDATES.open() as f:
        for line in f:
            r = json.loads(line)
            cands.setdefault((r["user_id"], r["asin"]), []).extend(r["candidates"])

    # 补 oracle/length_match/vades_minus_central/text_idx -> actual candidate text
    # eval_per_case.jsonl 只存了 dist_to_gt 没存 candidate, 需重算
    # 用 ground truth feats (gt_features) 反向
    log("加载 ground truth 特征")
    from evaluate import load_ground_truth_features, load_all_candidates
    gt_features = load_ground_truth_features()
    cand_dict, mu_dict = load_all_candidates()
    # 重抽 cand_features
    log("重抽 cand features for retrieval")
    import spacy
    from extract_clause_features_single_query import extract_clause_features_from_doc
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger"])
    with FEATURE_NAMES_FILE.open() as f:
        feature_names = json.load(f)
    cand_feats_dict: dict[tuple[str, str], np.ndarray] = {}
    for key, cs in cand_dict.items():
        feats = []
        for doc in nlp.pipe(cs, batch_size=128):
            try:
                ff = extract_clause_features_from_doc(doc, doc.text)
                feats.append([float(ff[n]) for n in feature_names])
            except Exception:
                feats.append([0.0] * len(feature_names))
        cand_feats_dict[key] = np.asarray(feats, dtype=np.float64)

    rng = np.random.default_rng(42)
    method_top1: dict[tuple[str, str], dict[str, str]] = {}
    for key, r in per_case.items():
        gt = gt_features.get(key)
        if gt is None:
            continue
        cands_list = cands.get(key, [])
        feats = cand_feats_dict.get(key, np.zeros((0, len(feature_names))))
        if len(cands_list) == 0:
            continue
        d_to_gt = np.linalg.norm(feats - gt, axis=1)
        mu_arr = np.asarray(mu_dict[key])
        cand_mean = feats.mean(axis=0)

        # 1. vades_rank: 从 scored.jsonl 排序
        s_list = scored.get(key, [])
        if s_list:
            sv = sorted(s_list, key=lambda x: -x["score"])
            vades_top_text = sv[0]["candidate"]
        else:
            vades_top_text = cands_list[0]
        # vades_minus_central: top-5 离 cand_mean 最远
        if len(s_list) >= 5:
            top5 = sv[:5]
            top5_idxs = [cands_list.index(t["candidate"]) for t in top5 if t["candidate"] in cands_list]
            d_to_mean_top5 = np.linalg.norm(feats[top5_idxs] - cand_mean, axis=1)
            vm_text = cands_list[top5_idxs[int(np.argmax(d_to_mean_top5))]]
        else:
            vm_text = vades_top_text
        # vades_top3_median
        if len(s_list) >= 3:
            top3 = sv[:3]
            top3_idxs = [cands_list.index(t["candidate"]) for t in top3 if t["candidate"] in cands_list]
            if top3_idxs:
                top3_mu = mu_arr[top3_idxs]
                top3_med = np.median(top3_mu, axis=0)
                d_t3m = np.linalg.norm(top3_mu - top3_med, axis=1)
                v3m_text = cands_list[top3_idxs[int(np.argmin(d_t3m))]]
            else:
                v3m_text = vades_top_text
        else:
            v3m_text = vades_top_text
        # length_match
        lens = np.asarray([len(c.split()) for c in cands_list], dtype=np.float64)
        gt_texts = r.get("gt_reviews", [])
        gt_avg_len = float(np.mean([len(t.split()) for t in gt_texts])) if gt_texts else float(np.mean(lens))
        len_idx = int(np.argmin(np.abs(lens - gt_avg_len)))
        length_text = cands_list[len_idx]
        # median_style
        if len(mu_arr) > 0:
            mu_med = np.median(mu_arr, axis=0)
            d_to_med = np.linalg.norm(mu_arr - mu_med, axis=1)
            med_idx = int(np.argmin(d_to_med))
        else:
            med_idx = 0
        median_text = cands_list[med_idx]
        # first
        first_text = cands_list[0]
        # random (固定一个 seed 抽)
        rand_idx = int(rng.integers(0, len(cands_list)))
        random_text = cands_list[rand_idx]
        # oracle
        oracle_idx = int(np.argmin(d_to_gt))
        oracle_text = cands_list[oracle_idx]

        method_top1[key] = {
            "vades_rank": vades_top_text,
            "vades_minus_central": vm_text,
            "vades_top3_median": v3m_text,
            "tfidf": r.get("tfidf", {}).get("candidate", oracle_text),  # 占位
            "length_match": length_text,
            "first": first_text,
            "random": random_text,
            "median_style": median_text,
            "oracle": oracle_text,
        }
    return method_top1


def evaluate(vec: TfidfVectorizer, X, asin_list: list[str], method_top1: dict) -> dict:
    """对每个 method 跑 top-K retrieval, 求 recall@K, MRR."""
    log = lambda m: print(f"[retrieval] {m}", flush=True)
    asin_to_idx = {a: i for i, a in enumerate(asin_list)}
    results: dict[str, dict] = {m: {"hits": 0, "rr_sum": 0.0, "ranks": []} for m in METHODS}

    PER_CASE_OUT.unlink(missing_ok=True)
    n_done = 0
    for key, method_queries in method_top1.items():
        target_asin = key[1]
        target_idx = asin_to_idx.get(target_asin)
        per_case_record = {"user_id": key[0], "asin": key[1]}
        for m, q in method_queries.items():
            if m not in METHODS:
                continue
            q_vec = vec.transform([q])
            scores = (q_vec @ X.T).toarray().flatten()
            top_idxs = np.argpartition(-scores, min(TOPK, len(scores)-1))[:TOPK]
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
            per_case_record[m] = {"query": q, "rank": rank, "hit_at_10": hit, "hit_at_100": hit_100, "rr": rr}
        with PER_CASE_OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(per_case_record, ensure_ascii=False) + "\n")
        n_done += 1
        if n_done % 5 == 0:
            log(f"  {n_done} cases done")

    n = max(1, len(method_top1))
    summary = {"n_cases": n, "topk": TOPK}
    summary_methods = {}
    for m, r in results.items():
        ranks = np.asarray(r["ranks"], dtype=np.int64)
        valid_ranks = ranks[ranks > 0]
        summary_methods[m] = {
            "recall_at_10": r["hits"] / n,
            "recall_at_100": r.get("hits_100", 0) / n,
            "mrr": r["rr_sum"] / n,
            "found": int((ranks > 0).sum()),
            "median_rank_when_found": float(np.median(valid_ranks)) if len(valid_ranks) > 0 else -1.0,
        }
    summary["methods"] = summary_methods
    return summary


def main() -> None:
    log = lambda m: print(f"[retrieval] {m}", flush=True)
    log("=" * 70)
    log("真实 retrieval 评估: TF-IDF over meta_Baby_Products")
    log("=" * 70)

    vec, X, asin_list = build_index()
    method_top1 = load_method_top1s()
    log(f"case×method: {len(method_top1)} × {len(METHODS)}")

    summary = evaluate(vec, X, asin_list, method_top1)
    REPORT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log("=" * 70)
    log("retrieval 结果 (recall@10, recall@100, MRR)")
    log("=" * 70)
    for m in METHODS:
        d = summary["methods"][m]
        marker = " ←" if m == "vades_rank" else ""
        log(f"  {m:25s} R@10={d['recall_at_10']:.3f}  R@100={d['recall_at_100']:.3f}  "
            f"MRR={d['mrr']:.3f}  found={d['found']}/{summary['n_cases']}{marker}")
    log(f"已写入 {REPORT}")


# 复制 evaluate.py 里的常量
FEATURE_NAMES_FILE = EXPERIMENT_DIR / "_feature_names.json"
if not FEATURE_NAMES_FILE.exists():
    FEATURE_NAMES_FILE.write_text(json.dumps([
        "max_dependency_depth", "mean_dependency_depth", "dependency_tree_height",
        "depth_variance", "acl_count", "relcl_count", "ccomp_count", "xcomp_count",
        "advcl_count", "clause_nesting_depth", "mean_dependency_distance",
        "max_dependency_distance", "long_dependency_ratio", "amod_count",
        "advmod_count", "nmod_count", "compound_count", "modifier_density",
        "coordination_count", "max_branching_factor",
    ]))


if __name__ == "__main__":
    main()
