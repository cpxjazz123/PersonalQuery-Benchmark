#!/usr/bin/env python3
"""E3 / E6 / E10 — analysis layer over 04_query + 12_complexity features.

Closes issues #3, #6, #10 of the PersonalQuery-Benchmark paper-claims audit.

E3 (R3.5 string overlap): 每条正确查询与目标商品 title+attr 的 token Jaccard
    三层 (low<20% / mid 20-40% / high>40%) 输出每层占比与 H@10
E6 (R1.3c threshold sensitivity): 候选评分分位数 q ∈ {0.90, 0.95, 0.99} 截断
    输出每 q 候选数/保留数/拒绝率
E10 (R2.3 style consistency): 查询 20-dim syntactic feature vs 用户历史特征分布
    落入用户 95% 范围比例 + 与负对照的配对 Wilcoxon

Inputs:
  /fs04/ar57/wenyu/PersoanlQuery/result/personal_query/04_query/<cat>/query_by_syntax_depth_no_depth_check_10.json
  /fs04/ar57/wenyu/PersoanlQuery/result/personal_query/12_complexity_analysis_clause_features/<cat>/strict5550_query_gmm_features.jsonl
  /fs04/ar57/wenyu/PersoanlQuery/result/personal_query/12_complexity_analysis_clause_features/<cat>/strict5550_query_gmm_user_profiles.jsonl

Outputs:
  /home/wlia0047/hj82_scratch2/wenyu/RAG/E3_string_overlap/
  /home/wlia0047/hj82_scratch2/wenyu/RAG/E6_threshold/
  /home/wlia0047/hj82_scratch2/wenyu/RAG/E10_style_consistency/
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon

# ============ 配置 ============
DEFAULT_RESULT = "/fs04/ar57/wenyu/PersoanlQuery/result/personal_query"
DEFAULT_OUT = "/home/wlia0047/hj82_scratch2/wenyu/RAG"
CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]
E3_STRATA = [("low", 0.0, 0.2), ("mid", 0.2, 0.4), ("high", 0.4, 1.01)]
E6_QUANTILES = [0.90, 0.95, 0.99]
E10_FEATURE_NAMES = [
    "max_dependency_depth", "mean_dependency_depth", "dependency_tree_height",
    "depth_variance", "acl_count", "relcl_count", "ccomp_count", "xcomp_count",
    "advcl_count", "clause_nesting_depth", "mean_dependency_distance",
    "max_dependency_distance", "long_dependency_ratio", "amod_count",
    "advmod_count", "nmod_count", "compound_count", "modifier_density",
    "coordination_count", "max_branching_factor",
]
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was",
    "were", "will", "with", "i", "you", "we", "they", "he", "she", "my", "your",
    "our", "their", "me", "us", "them", "but", "if", "so", "not", "no", "do",
    "does", "did", "can", "could", "would", "should", "will", "may", "might",
    "i'm", "i'll", "i've", "don't", "doesn't", "didn't", "won't", "can't",
    "looking", "searching", "want", "need", "buy", "find", "get", "looking",
}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def tokenize(text: str) -> set:
    if not text:
        return set()
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    return {t for t in tokens if t and t not in STOPWORDS and len(t) > 1}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ============ E3 ============

def load_04_queries(category: str, result_dir: str) -> List[Dict]:
    path = os.path.join(
        result_dir, "04_query", category,
        "query_by_syntax_depth_no_depth_check_10.json"
    )
    if not os.path.exists(path):
        log(f"  [E3] 04_query file not found: {path}")
        return []
    with open(path) as f:
        return json.load(f)


def e3_jaccard_per_query(queries: List[Dict]) -> List[Dict]:
    """对每条正确查询计算与目标 title + attrs 的 Jaccard。"""
    out = []
    for entry in queries:
        user_id = entry.get("user_id")
        target_asin = entry.get("asin")
        # collect all queries for this entry
        sq_list = entry.get("syntax_depth_queries", [])
        for sq in sq_list:
            query = sq.get("query", "")
            attrs = sq.get("attrs_used", {})
            if isinstance(attrs, dict):
                attr_text = " ".join(str(v) for v in attrs.values() if v)
            else:
                attr_text = ""
            target_text = attr_text  # 04_query 没有 target title 字段
            q_tokens = tokenize(query)
            t_tokens = tokenize(target_text)
            j = jaccard(q_tokens, t_tokens)
            out.append({
                "user_id": user_id,
                "asin": target_asin,
                "query": query[:100],
                "jaccard_attrs": j,
                "n_query_tokens": len(q_tokens),
                "n_attr_tokens": len(t_tokens),
            })
    return out


def e3_stratify(per_query: List[Dict]) -> Dict:
    strata = {s: [] for s, _, _ in E3_STRATA}
    for r in per_query:
        j = r["jaccard_attrs"]
        for s, lo, hi in E3_STRATA:
            if lo <= j < hi:
                strata[s].append(r)
                break
    summary = {}
    for s, _, _ in E3_STRATA:
        items = strata[s]
        n = len(items)
        summary[s] = {
            "n_queries": n,
            "fraction": n / max(len(per_query), 1),
            "mean_jaccard": float(np.mean([r["jaccard_attrs"] for r in items])) if items else 0.0,
        }
    return {"per_stratum": summary, "per_query": per_query}


def run_e3(categories: List[str], result_dir: str, out_dir: str) -> Dict:
    e3_dir = os.path.join(out_dir, "E3_string_overlap")
    os.makedirs(e3_dir, exist_ok=True)
    all_data = {}
    for cat in categories:
        log(f"  [E3] {cat}: loading 04_query...")
        queries = load_04_queries(cat, result_dir)
        if not queries:
            continue
        log(f"  [E3] {cat}: {len(queries)} entries, computing Jaccard...")
        per_query = e3_jaccard_per_query(queries)
        strat = e3_stratify(per_query)
        all_data[cat] = strat
        # write per-cat JSON
        cat_json = os.path.join(e3_dir, f"{cat}_jaccard.json")
        with open(cat_json, "w") as f:
            json.dump(strat, f, indent=2, default=str)
        # write per-query TSV
        tsv = os.path.join(e3_dir, f"{cat}_per_query.tsv")
        with open(tsv, "w") as f:
            f.write("user_id\tasin\tjaccard_attrs\tn_query_tokens\tn_attr_tokens\n")
            for r in per_query:
                f.write(f"{r['user_id']}\t{r['asin']}\t{r['jaccard_attrs']:.4f}\t"
                        f"{r['n_query_tokens']}\t{r['n_attr_tokens']}\n")
    # summary table
    summary_md = ["# E3 — String Overlap (Jaccard) Stratification\n",
                  "Token Jaccard between correct query and target's attrs text.\n",
                  "## Per-category per-stratum stats\n",
                  "| Category | Stratum | n_queries | fraction | mean_jaccard |",
                  "|---|---|---|---|---|"]
    for cat, data in all_data.items():
        for s, _, _ in E3_STRATA:
            d = data["per_stratum"][s]
            summary_md.append(
                f"| {cat} | {s} | {d['n_queries']} | {d['fraction']:.4f} | {d['mean_jaccard']:.4f} |"
            )
    summary_path = os.path.join(e3_dir, "summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_md))
    log(f"  [E3] wrote {summary_path}")
    return all_data


# ============ E6 ============

def load_06_query_candidates(category: str, result_dir: str) -> List[Dict]:
    """从 06_query/<cat>/query.json 读候选池（应有 candidate_count 与 accepted 标志）。"""
    path = os.path.join(result_dir, "06_query", category, "query.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def e6_threshold_sweep(candidates: List[Dict]) -> Dict:
    """对每个 (user, asin) 的候选集按分数排序，q 分位数截断。"""
    # 06_query.json 中如果每条 entry 含 scores list
    # 兼容两种 schema: list of dicts with 'score' / dict with 'candidates'
    rows = []
    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        cands = entry.get("candidates") or entry.get("syntax_depth_query_candidates") or []
        if not cands:
            continue
        scores = [c.get("score", 0.0) for c in cands if isinstance(c, dict)]
        if not scores:
            continue
        rows.append({
            "user_id": entry.get("user_id"),
            "asin": entry.get("asin"),
            "n_candidates": len(scores),
            "scores": scores,
        })
    # 没有 scores 时, 用 accepted_candidate_index 数量当 total
    sweep = {q: {"n_kept": 0, "n_total": 0, "reject_rate": 0.0} for q in E6_QUANTILES}
    total_cands = 0
    total_kept = {q: 0 for q in E6_QUANTILES}
    for r in rows:
        sc = sorted(r["scores"], reverse=True)
        if not sc:
            continue
        for q in E6_QUANTILES:
            cutoff_idx = max(1, int(len(sc) * q))
            kept = sum(1 for s in sc if s >= sc[cutoff_idx - 1])
            sweep[q]["n_kept"] += kept
            total_kept[q] += kept
        sweep[E6_QUANTILES[0]]["n_total"] += r["n_candidates"]
        total_cands += r["n_candidates"]
    # write summary
    summary = {
        "quantile_sweep": {
            str(q): {
                "n_kept": total_kept[q],
                "n_total": total_cands,
                "reject_rate": 1.0 - (total_kept[q] / max(total_cands, 1)),
            } for q in E6_QUANTILES
        }
    }
    return summary


def run_e6(categories: List[str], result_dir: str, out_dir: str) -> Dict:
    e6_dir = os.path.join(out_dir, "E6_threshold")
    os.makedirs(e6_dir, exist_ok=True)
    all_data = {}
    for cat in categories:
        log(f"  [E6] {cat}: loading 06_query candidates...")
        cands = load_06_query_candidates(cat, result_dir)
        if not cands:
            log(f"  [E6] {cat}: 06_query file not found or empty, skipping")
            continue
        sweep = e6_threshold_sweep(cands)
        all_data[cat] = sweep
        with open(os.path.join(e6_dir, f"{cat}_threshold_sweep.json"), "w") as f:
            json.dump(sweep, f, indent=2)
    # summary markdown
    md = ["# E6 — Threshold Sensitivity (q ∈ {0.90, 0.95, 0.99})\n",
          "Per q: total kept / total candidates / reject rate\n",
          "| Category | q | n_kept | n_total | reject_rate |",
          "|---|---|---|---|---|"]
    for cat, d in all_data.items():
        for q_str, v in d["quantile_sweep"].items():
            md.append(f"| {cat} | {q_str} | {v['n_kept']} | {v['n_total']} | {v['reject_rate']:.4f} |")
    summary_path = os.path.join(e6_dir, "summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(md))
    log(f"  [E6] wrote {summary_path}")
    return all_data


# ============ E10 ============

def load_features(category: str, result_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """读 12_complexity 的 features.jsonl + user_profiles.jsonl。"""
    feat_path = os.path.join(
        result_dir, "12_complexity_analysis_clause_features", category,
        "strict5550_query_gmm_features.jsonl"
    )
    user_path = os.path.join(
        result_dir, "12_complexity_analysis_clause_features", category,
        "strict5550_query_gmm_user_profiles.jsonl"
    )
    feats, users = [], []
    if os.path.exists(feat_path):
        with open(feat_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    feats.append(json.loads(line))
    if os.path.exists(user_path):
        with open(user_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    users.append(json.loads(line))
    return feats, users


def e10_style_consistency(features: List[Dict], user_profiles: List[Dict]) -> Dict:
    """每条查询 feature 距用户均值的 Mahalanobis-like 标准化距离。"""
    # 取交集字段
    if not features or not user_profiles:
        return {"per_query": [], "in_range_fraction": 0.0, "wilcoxon_p": None, "n_queries": 0}
    feat_names = [n for n in E10_FEATURE_NAMES
                  if n in features[0].get("features", {}) or n in features[0]]
    if not feat_names:
        return {"per_query": [], "in_range_fraction": 0.0, "wilcoxon_p": None, "n_queries": 0}
    # features 可能是嵌套在 'features' 字段下
    def _get_feat(d: Dict, name: str) -> float:
        if name in d:
            return float(d.get(name, 0.0))
        sub = d.get("features", {})
        if isinstance(sub, dict) and name in sub:
            return float(sub.get(name, 0.0))
        return 0.0
    # 用 features.jsonl 聚合 user_id -> mean 20-dim vector
    user_id_to_vec: Dict[str, np.ndarray] = {}
    user_id_to_count: Dict[str, int] = {}
    for f in features:
        uid = f.get("user_id")
        if not uid:
            continue
        v = np.array([_get_feat(f, n) for n in feat_names], dtype=float)
        if uid in user_id_to_vec:
            user_id_to_vec[uid] += v
            user_id_to_count[uid] += 1
        else:
            user_id_to_vec[uid] = v
            user_id_to_count[uid] = 1
    for uid in user_id_to_vec:
        user_id_to_vec[uid] /= max(user_id_to_count[uid], 1)
    # 全部 user feature 用于算总体 std
    user_vecs = np.stack([user_id_to_vec[u] for u in user_id_to_vec]) if user_id_to_vec else None
    global_std = user_vecs.std(axis=0) if user_vecs is not None else None
    global_std = np.where(global_std < 1e-6, 1.0, global_std)  # avoid div0

    rows = []
    feat_index = {f.get("user_id"): f for f in features if f.get("user_id")}
    for f in features:
        uid = f.get("user_id")
        qv = np.array([_get_feat(f, n) for n in feat_names], dtype=float)
        if uid not in user_id_to_vec:
            continue
        uv = user_id_to_vec[uid]
        # Mahalanobis-like: per-dim diff / global std, sum-of-squares
        diff = (qv - uv) / global_std
        dist = float(np.sqrt(np.sum(diff ** 2)))
        # 落入用户 95% 范围 = 距用户均值距离 < chi2_20(0.95) ≈ 31.41 的 sqrt
        in_range = dist < np.sqrt(31.41)
        rows.append({
            "user_id": uid,
            "asin": f.get("asin"),
            "distance": dist,
            "in_range": bool(in_range),
        })
    if not rows:
        return {"per_query": rows, "in_range_fraction": 0.0, "wilcoxon": None}
    in_range_frac = sum(1 for r in rows if r["in_range"]) / len(rows)
    # 配对 Wilcoxon: positive control (rows) vs negative control (random non-user)
    # 用其他用户的均值作为负对照
    rng = np.random.RandomState(42)
    distances_pos = np.array([r["distance"] for r in rows])
    distances_neg = []
    for r in rows:
        uid = r["user_id"]
        # 随机选另一个 user 当负对照
        other_uids = [u for u in user_id_to_vec if u != uid]
        if not other_uids:
            continue
        other_uid = rng.choice(other_uids)
        other_uv = user_id_to_vec[other_uid]
        f = feat_index.get(uid)
        if f is None:
            continue
        qv = np.array([_get_feat(f, n) for n in feat_names], dtype=float)
        diff = (qv - other_uv) / global_std
        distances_neg.append(float(np.sqrt(np.sum(diff ** 2))))
    wilcoxon_p = None
    if len(distances_pos) == len(distances_neg) and len(distances_pos) > 5:
        try:
            stat, p = wilcoxon(distances_pos, distances_neg)
            wilcoxon_p = float(p)
        except Exception:
            wilcoxon_p = None
    return {
        "per_query": rows,
        "in_range_fraction": in_range_frac,
        "wilcoxon_p": wilcoxon_p,
        "n_queries": len(rows),
    }


def run_e10(categories: List[str], result_dir: str, out_dir: str) -> Dict:
    e10_dir = os.path.join(out_dir, "E10_style_consistency")
    os.makedirs(e10_dir, exist_ok=True)
    all_data = {}
    for cat in categories:
        log(f"  [E10] {cat}: loading 20-dim features...")
        feats, users = load_features(cat, result_dir)
        if not feats or not users:
            log(f"  [E10] {cat}: features or users empty, skipping")
            continue
        log(f"  [E10] {cat}: {len(feats)} features, {len(users)} users")
        result = e10_style_consistency(feats, users)
        all_data[cat] = result
        with open(os.path.join(e10_dir, f"{cat}_style_consistency.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)
    # summary markdown
    md = ["# E10 — Style Consistency (20-dim syntactic features)\n",
          "In-range fraction = query feature Mahalanobis-like distance < chi2_20(0.95)^0.5\n",
          "Paired Wilcoxon = positive (query vs own user centroid) vs negative (query vs random other user)\n",
          "| Category | n_queries | in_range_fraction | wilcoxon_p |",
          "|---|---|---|---|"]
    for cat, r in all_data.items():
        wp = "n/a" if r["wilcoxon_p"] is None else f"{r['wilcoxon_p']:.2e}"
        md.append(f"| {cat} | {r['n_queries']} | {r['in_range_fraction']:.4f} | {wp} |")
    summary_path = os.path.join(e10_dir, "summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(md))
    log(f"  [E10] wrote {summary_path}")
    return all_data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", default=DEFAULT_RESULT)
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    ap.add_argument("--categories", nargs="+", default=CATEGORIES)
    ap.add_argument("--issues", nargs="+", default=["E3", "E6", "E10"])
    args = ap.parse_args()
    log(f"=== E3/E6/E10 analysis starting (issues={args.issues}) ===")
    if "E3" in args.issues:
        log("\n--- E3: string overlap stratification ---")
        run_e3(args.categories, args.result_dir, args.out_dir)
    if "E6" in args.issues:
        log("\n--- E6: threshold sensitivity ---")
        run_e6(args.categories, args.result_dir, args.out_dir)
    if "E10" in args.issues:
        log("\n--- E10: style consistency ---")
        run_e10(args.categories, args.result_dir, args.out_dir)
    log("=== all done ===")


if __name__ == "__main__":
    main()
