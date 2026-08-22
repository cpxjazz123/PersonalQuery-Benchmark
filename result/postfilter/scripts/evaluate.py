#!/usr/bin/env python3
"""评估 Generate-then-Rank: VADES top-1 vs 4 个 baseline.

评估指标:
- 候选与用户 ground-truth reviews 的句法特征距离 (mean L2 across 20-d)
   (候选 vs gt_review_mean = 1/N_gt * sum features(gt_review_i))

策略:
1. VADES-rank: top-1 via user_mu (主策略)
2. Random: 随机抽 1 条, 重复 100 次取均值
3. First: 取候选列表里第一条
4. Median-style: 选候选里 mu_query 离所有候选 mu_query 中位数最近的一条
5. Oracle: 选候选里 features 离 gt_reviews mean features 最近的一条 (风格理论上界)

Outputs:
- /home/wlia0047/hj82_scratch2/wenyu/postfilter/eval_report.json
- /home/wlia0047/hj82_scratch2/wenyu/postfilter/eval_per_case.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
SYNTAX_DIR = REPO_ROOT / "syntactic_analysis"
sys.path.insert(0, str(SYNTAX_DIR))
from extract_clause_features_single_query import extract_clause_features  # noqa: E402

EXPERIMENT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/postfilter")
TEST_CASES = EXPERIMENT_DIR / "test_cases.jsonl"
SCORED = EXPERIMENT_DIR / "scored.jsonl"
TOP1 = EXPERIMENT_DIR / "top1.jsonl"
EVAL_REPORT = EXPERIMENT_DIR / "eval_report.json"
EVAL_PER_CASE = EXPERIMENT_DIR / "eval_per_case.jsonl"

FEATURE_NAMES = [
    "max_dependency_depth", "mean_dependency_depth", "dependency_tree_height",
    "depth_variance", "acl_count", "relcl_count", "ccomp_count", "xcomp_count",
    "advcl_count", "clause_nesting_depth", "mean_dependency_distance",
    "max_dependency_distance", "long_dependency_ratio", "amod_count",
    "advmod_count", "nmod_count", "compound_count", "modifier_density",
    "coordination_count", "max_branching_factor",
]
N_RANDOM = 100
RANDOM_SEED = 42


def load_ground_truth_features() -> dict[tuple[str, str], np.ndarray]:
    """对每个 (user, asin), 用 spaCy 提取 gt_reviews 句法特征, 求 mean (20-d)."""
    log = lambda m: print(f"[eval] {m}", flush=True)
    log("提取 ground truth 特征 ...")
    cases: dict[tuple[str, str], dict] = {}
    with TEST_CASES.open() as f:
        for line in f:
            row = json.loads(line)
            cases[(row["user_id"], row["asin"])] = row

    # 批量 spaCy 解析
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger"])
    from extract_clause_features_single_query import extract_clause_features_from_doc

    gt_features: dict[tuple[str, str], np.ndarray] = {}
    for (uid, asin), case in cases.items():
        gts = case.get("gt_reviews", [])
        feats = []
        for doc in nlp.pipe(gts, batch_size=64):
            try:
                f = extract_clause_features_from_doc(doc, doc.text)
                feats.append([float(f[name]) for name in FEATURE_NAMES])
            except Exception:
                continue
        if not feats:
            continue
        gt_features[(uid, asin)] = np.mean(np.asarray(feats), axis=0)
    log(f"  gt_features: {len(gt_features)} cases")
    return gt_features


def load_all_candidates() -> dict[tuple[str, str], list[str]]:
    """从 scored.jsonl 还原 per (user, asin) 的全部候选 + mu_query."""
    cand: dict[tuple[str, str], list[str]] = defaultdict(list)
    mu: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    with SCORED.open() as f:
        for line in f:
            row = json.loads(line)
            key = (row["user_id"], row["asin"])
            cand[key].append(row["candidate"])
            mu[key].append(np.asarray(row["mu_query"], dtype=np.float64))
    return cand, mu


def extract_features_for_candidates(candidates: list[str]) -> np.ndarray:
    """(复跑) 候选 → 20-d 句法特征 (未标准化, 用于和 gt_features 比较)."""
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger"])
    from extract_clause_features_single_query import extract_clause_features_from_doc
    feats = []
    for doc in nlp.pipe(candidates, batch_size=128):
        try:
            f = extract_clause_features_from_doc(doc, doc.text)
            feats.append([float(f[name]) for name in FEATURE_NAMES])
        except Exception:
            feats.append([0.0] * len(FEATURE_NAMES))
    return np.asarray(feats, dtype=np.float64)


def load_top1_picks() -> dict[tuple[str, str], dict]:
    picks: dict[tuple[str, str], dict] = {}
    with TOP1.open() as f:
        for line in f:
            row = json.loads(line)
            picks[(row["user_id"], row["asin"])] = row
    return picks


def load_scored_dict() -> dict[tuple[str, str], list[dict]]:
    """scored.jsonl → {key: [{candidate, score, ...}, ...]}"""
    out: dict[tuple[str, str], list[dict]] = {}
    with SCORED.open() as f:
        for line in f:
            row = json.loads(line)
            out.setdefault((row["user_id"], row["asin"]), []).append(row)
    return out


def tfidf_cosine_to_gt(candidate: str, gt_reviews: list[str]) -> float:
    """候选 vs gt reviews: TF-IDF cosine similarity (取 max over each gt).
    越高越好, 转化成 L2 距离用 1 - max.
    """
    if not gt_reviews:
        return 0.0
    try:
        docs = [candidate] + gt_reviews
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=2000).fit_transform(docs)
        sims = cosine_similarity(vec[0:1], vec[1:]).flatten()
        return float(sims.max())
    except Exception:
        return 0.0


def main() -> None:
    log = lambda m: print(f"[eval] {m}", flush=True)
    log("=" * 70)
    log("评估 Generate-then-Rank: VADES-rank vs 4 baselines")
    log("=" * 70)

    rng = np.random.default_rng(RANDOM_SEED)
    gt_features = load_ground_truth_features()
    cand_dict, mu_dict = load_all_candidates()
    top1_picks = load_top1_picks()
    scored_dict = load_scored_dict()
    log(f"cases 上有 cands: {len(cand_dict)}, top1 picks: {len(top1_picks)}, scored: {len(scored_dict)}")

    # 加载 test_cases 用于 length-match / tfidf gt 复用
    test_cases = {}
    with TEST_CASES.open() as f:
        for line in f:
            r = json.loads(line)
            test_cases[(r["user_id"], r["asin"])] = r

    # === 提取所有候选的 (未标准化) 句法特征, 用于与 gt_features 比较 ===
    all_cands_list = []
    all_cands_keys = []
    for key, cs in cand_dict.items():
        all_cands_list.extend(cs)
        all_cands_keys.extend([key] * len(cs))
    log(f"总候选 {len(all_cands_list)} 条, 提取 20-d 句法特征 ...")
    cand_features = extract_features_for_candidates(all_cands_list)
    log(f"  cand_features.shape={cand_features.shape}")

    # 按 key split
    cand_feats_dict: dict[tuple[str, str], np.ndarray] = {}
    cursor = 0
    for key, cs in cand_dict.items():
        n = len(cs)
        cand_feats_dict[key] = cand_features[cursor : cursor + n]
        cursor += n

    # === 评估每个 case ===
    EVAL_PER_CASE.unlink(missing_ok=True)
    per_case_metrics = []
    methods = ["vades_rank", "random", "first", "median_style", "oracle",
               "length_match", "tfidf", "vades_top3_median", "vades_minus_central", "oracle_minus_central"]

    for key, gt in gt_features.items():
        uid, asin = key
        cands = cand_dict.get(key, [])
        cand_feats = cand_feats_dict.get(key, np.zeros((0, len(FEATURE_NAMES))))

        if len(cands) == 0:
            continue

        # 1. VADES-rank top-1
        top1 = top1_picks.get(key, {})
        top1_text = top1.get("top1_candidate", "")
        if top1_text:
            i = cands.index(top1_text) if top1_text in cands else 0
            vades_feat = cand_feats[i]
        else:
            vades_feat = cand_feats[0]

        # 2. Random: 100 次取均值
        idxs = rng.integers(0, len(cands), size=N_RANDOM)
        rand_dists = [float(np.linalg.norm(cand_feats[i] - gt)) for i in idxs]
        random_mean_dist = float(np.mean(rand_dists))
        random_min_dist = float(np.min(rand_dists))

        # 3. First: 第一条
        first_dist = float(np.linalg.norm(cand_feats[0] - gt))

        # 4. Median-style: 选 mu_query 离所有候选 mu 中位数最近
        mu_arr = np.asarray(mu_dict[key])
        if len(mu_arr) > 0:
            mu_median = np.median(mu_arr, axis=0)
            d_to_median = np.linalg.norm(mu_arr - mu_median, axis=1)
            median_idx = int(np.argmin(d_to_median))
        else:
            median_idx = 0
        median_dist = float(np.linalg.norm(cand_feats[median_idx] - gt))

        # 5. Oracle: 选 candidate 离 gt_features 最近的 (理论上界)
        d_to_gt = np.linalg.norm(cand_feats - gt, axis=1)
        oracle_idx = int(np.argmin(d_to_gt))
        oracle_dist = float(d_to_gt[oracle_idx])

        # 6. Length-match: 选 word count 与用户 gt 平均长度最接近的
        cand_lens = np.asarray([len(c.split()) for c in cands], dtype=np.float64)
        gt_texts = test_cases.get(key, {}).get("gt_reviews", [])
        gt_avg_len = float(np.mean([len(t.split()) for t in gt_texts])) if gt_texts else float(np.mean(cand_lens))
        d_to_len = np.abs(cand_lens - gt_avg_len)
        length_idx = int(np.argmin(d_to_len))
        length_dist = float(np.linalg.norm(cand_feats[length_idx] - gt))

        # 7. TF-IDF cosine to gt_reviews (semantic baseline): 取 cosine 最高
        tfidf_sims = np.asarray([tfidf_cosine_to_gt(c, gt_texts) for c in cands], dtype=np.float64)
        tfidf_idx = int(np.argmax(tfidf_sims))
        tfidf_dist = float(np.linalg.norm(cand_feats[tfidf_idx] - gt))

        # 8. VADES top-3 by user_mu, then median-style
        scored = scored_dict.get(key, [])
        if len(scored) >= 3:
            top3 = sorted(scored, key=lambda r: -r["score"])[:3]
            top3_texts = [r["candidate"] for r in top3]
            top3_idxs = [cands.index(t) for t in top3_texts if t in cands]
            if top3_idxs:
                top3_mu = mu_arr[top3_idxs]
                top3_median = np.median(top3_mu, axis=0)
                d_to_t3m = np.linalg.norm(top3_mu - top3_median, axis=1)
                v3m_idx = top3_idxs[int(np.argmin(d_to_t3m))]
            else:
                v3m_idx = top1_idx if (top1_idx := (cands.index(top1_text) if top1_text in cands else 0)) else 0
        else:
            v3m_idx = cands.index(top1_text) if top1_text in cands else 0
        v3m_dist = float(np.linalg.norm(cand_feats[v3m_idx] - gt))

        # 9. VADES - central: 在 VADES top-5 中挑离候选均值最远的 (尝试多样性)
        if len(scored) >= 5:
            top5 = sorted(scored, key=lambda r: -r["score"])[:5]
            top5_texts = [r["candidate"] for r in top5]
            top5_idxs = [cands.index(t) for t in top5_texts if t in cands]
            cand_mean = np.mean(cand_feats, axis=0)
            d_to_mean = np.linalg.norm(cand_feats[top5_idxs] - cand_mean, axis=1)
            vm_idx = top5_idxs[int(np.argmax(d_to_mean))]
        else:
            vm_idx = cands.index(top1_text) if top1_text in cands else 0
        vm_dist = float(np.linalg.norm(cand_feats[vm_idx] - gt))

        # 10. Oracle - central: oracle top-3 中挑离候选均值最远的
        oracle_top3_idxs = np.argsort(d_to_gt)[:3].tolist()
        cand_mean = np.mean(cand_feats, axis=0)
        d_ot = np.linalg.norm(cand_feats[oracle_top3_idxs] - cand_mean, axis=1)
        oc_idx = oracle_top3_idxs[int(np.argmax(d_ot))]
        oc_dist = float(np.linalg.norm(cand_feats[oc_idx] - gt))

        vades_dist = float(np.linalg.norm(vades_feat - gt))

        record = {
            "user_id": uid,
            "asin": asin,
            "n_candidates": len(cands),
            "vades_rank": {
                "candidate": top1.get("top1_candidate", cands[0]),
                "dist_to_gt": vades_dist,
            },
            "random": {
                "dist_mean_over_100": random_mean_dist,
                "dist_min_over_100": random_min_dist,
            },
            "first": {"dist_to_gt": first_dist},
            "median_style": {"dist_to_gt": median_dist},
            "oracle": {"dist_to_gt": oracle_dist},
            "length_match": {"dist_to_gt": length_dist},
            "tfidf": {"dist_to_gt": tfidf_dist},
            "vades_top3_median": {"dist_to_gt": v3m_dist},
            "vades_minus_central": {"dist_to_gt": vm_dist},
            "oracle_minus_central": {"dist_to_gt": oc_dist},
        }
        per_case_metrics.append(record)
        with EVAL_PER_CASE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # === 汇总 ===
    log(f"汇总 {len(per_case_metrics)} cases ...")
    method_names = ["vades_rank", "random", "first", "median_style", "oracle",
                    "length_match", "tfidf", "vades_top3_median",
                    "vades_minus_central", "oracle_minus_central"]
    method_dists: dict[str, list[float]] = {}
    for m in method_names:
        if m == "random":
            method_dists[m] = [r[m]["dist_mean_over_100"] for r in per_case_metrics]
        else:
            method_dists[m] = [r[m]["dist_to_gt"] for r in per_case_metrics]

    summary = {
        "n_cases": len(per_case_metrics),
        "metric": "L2 distance between candidate style features (20d) and user gt_reviews mean features (20d)",
        "methods": {m: {
            "mean_dist": float(np.mean(method_dists[m])),
            "median_dist": float(np.median(method_dists[m])),
            "std_dist": float(np.std(method_dists[m])),
        } for m in method_names},
        "random_min": {
            "mean_dist": float(np.mean([r["random"]["dist_min_over_100"] for r in per_case_metrics])),
            "median_dist": float(np.median([r["random"]["dist_min_over_100"] for r in per_case_metrics])),
        },
    }
    vades_dists = method_dists["vades_rank"]
    random_mean = method_dists["random"]
    oracle_dists = method_dists["oracle"]
    summary["vades_vs_random_pct"] = {
        "delta_mean": float(np.mean(vades_dists) - np.mean(random_mean)),
        "ratio_mean": float(np.mean(vades_dists) / np.mean(random_mean)),
    }
    summary["vades_vs_oracle_gap"] = {
        "delta_mean": float(np.mean(vades_dists) - np.mean(oracle_dists)),
        "ratio_mean": float(np.mean(vades_dists) / np.mean(oracle_dists)),
    }

    EVAL_REPORT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log("=" * 70)
    log("评估结果 (mean L2 distance to user gt, 越低越好)")
    log("=" * 70)
    for m in method_names:
        d = summary["methods"][m]["mean_dist"]
        marker = " ←" if m == "vades_rank" else ""
        log(f"  {m:25s} {d:8.4f}{marker}")
    log(f"  random_min (100 次抽样最小值): {summary['random_min']['mean_dist']:8.4f}")
    log(f"")
    log(f"VADES-rank 比 random 改善: {summary['vades_vs_random_pct']['delta_mean']:+.4f} "
        f"({summary['vades_vs_random_pct']['ratio_mean']:.2%})")
    log(f"VADES-rank vs oracle gap: {summary['vades_vs_oracle_gap']['delta_mean']:+.4f} "
        f"({summary['vades_vs_oracle_gap']['ratio_mean']:.2%})")
    log(f"已写入 {EVAL_REPORT}")


if __name__ == "__main__":
    main()
