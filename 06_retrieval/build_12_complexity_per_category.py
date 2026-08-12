#!/usr/bin/env python3
"""为 Grocery / Pet 跑 12_complexity 20-dim features.

Reads 04_query/query_by_syntax_depth_no_depth_check_10.json, extracts 20-dim syntactic
features via spacy, runs GMM clustering, writes features.jsonl + summary.json to
12_complexity_analysis_clause_features/<cat>/.

直接复用 cluster_strict5550_query_gmm_and_attach_retrieval 的核心逻辑, 但跳过
attach_retrieval 步骤 (因为我们只需要 features 本身).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "10_complexity_analysis" / "common"))

from cluster_strict5550_query_gmm_and_attach_retrieval import (
    extract_feature_matrix,
    load_query_rows,
    run_pca_selection,
    select_best_gmm,
    remap_cluster_labels,
    build_feature_summaries,
    write_jsonl,
    log,
)


REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
CATEGORIES = ["Grocery_and_Gourmet_Food", "Pet_Supplies"]


def load_query_rows_from_syntax_depth(query_file: Path) -> list[dict]:
    """04_query 输出格式: [{user_id, asin, syntax_depth_queries: [{query, target_depth, ...}], ...}]

    每行展开为 N 个 query (N = len(syntax_depth_queries)), 每个 query 一个 row.
    """
    data = json.loads(query_file.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"{query_file} 必须是非空列表")
    rows = []
    for r in data:
        user_id = r.get("user_id")
        asin = r.get("asin")
        syntax_depth_queries = r.get("syntax_depth_queries", [])
        if not (user_id and asin and syntax_depth_queries):
            raise ValueError(f"{query_file} 缺少 user_id/asin/syntax_depth_queries: {r}")
        for sq in syntax_depth_queries:
            query_text = sq.get("query")
            if not query_text:
                continue
            rows.append({
                "original_row": r,
                "user_id": user_id,
                "asin": asin,
                "query_text": query_text,
                "word_count": int(sq.get("word_count", len(query_text.split()))),
                "target_depth": sq.get("target_depth"),
                "user_avg_depth": sq.get("user_avg_depth"),
            })
    if not rows:
        raise ValueError(f"{query_file} 展开后无有效 query")
    return rows


def run_for_category(category: str) -> dict:
    clause_dir = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / category
    clause_dir.mkdir(parents=True, exist_ok=True)
    query_file = REPO_ROOT / "result" / "personal_query" / "04_query" / category / "query_by_syntax_depth_no_depth_check_10.json"
    if not query_file.exists():
        raise FileNotFoundError(f"缺少 query 文件: {query_file}")

    log(f"[{category}] 读取 query: {query_file}")
    query_rows = load_query_rows_from_syntax_depth(query_file)
    log(f"[{category}] n_rows = {len(query_rows)}")
    feature_names, feature_matrix, feature_rows = extract_feature_matrix(query_rows)
    log(f"[{category}] feature_names (20-dim) = {feature_names}")
    log(f"[{category}] feature_matrix shape = {feature_matrix.shape}")
    embedding, _scaler, _pca, pca_summary = run_pca_selection(feature_matrix)
    log(f"[{category}] PCA dim = {pca_summary['selected_dim']}, cumvar = {pca_summary['explained_variance_ratio_sum']:.4f}")
    raw_labels, _gmm, selection_summary = select_best_gmm(embedding)
    cluster_labels, _remap, cluster_counts = remap_cluster_labels(raw_labels)
    cluster_feature_summaries = build_feature_summaries(feature_names, feature_matrix, cluster_labels)
    log(f"[{category}] selected_k = {selection_summary['selected_k']}, cluster_counts = {cluster_counts}")

    feature_output_rows = []
    user_output_rows = []
    for idx, row in enumerate(feature_rows):
        cluster_index = int(cluster_labels[idx])
        cluster_label = f"cluster_{cluster_index}"
        feature_output_rows.append({
            "user_id": row["user_id"],
            "asin": row["asin"],
            "cluster_label": cluster_label,
            "cluster_index": cluster_index,
            "query_text": row["query_text"],
            "word_count": row["word_count"],
            "target_depth": row["target_depth"],
            "user_avg_depth": row["user_avg_depth"],
            "features": row["features"],
            "pca_embedding": embedding[idx].tolist(),
        })
        user_output_rows.append({
            "user_id": row["user_id"],
            "asin": row["asin"],
            "cluster_label": cluster_label,
            "cluster_index": cluster_index,
            "query_text": row["query_text"],
            "word_count": row["word_count"],
            "target_depth": row["target_depth"],
            "user_avg_depth": row["user_avg_depth"],
            "pca_embedding": embedding[idx].tolist(),
        })

    feature_file = clause_dir / "strict5550_query_gmm_features.jsonl"
    user_file = clause_dir / "strict5550_query_gmm_user_profiles.jsonl"
    summary_file = clause_dir / "strict5550_query_gmm_summary.json"
    write_jsonl(feature_file, feature_output_rows)
    write_jsonl(user_file, user_output_rows)

    summary = {
        "category": category,
        "method": "gmm_query_syntax_feature_clustering",
        "query_file": str(query_file),
        "feature_file": str(feature_file),
        "user_file": str(user_file),
        "feature_names": feature_names,
        "pca": pca_summary,
        "cluster_selection": selection_summary,
        "cluster_counts": cluster_counts,
        "cluster_feature_summaries": cluster_feature_summaries,
        "n_rows_input": len(query_rows),
    }
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[{category}] 写入: {feature_file}")
    log(f"[{category}] 写入: {user_file}")
    log(f"[{category}] 写入: {summary_file}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=CATEGORIES)
    args = ap.parse_args()
    log(f"=== build 12_complexity features for {args.categories} ===")
    for cat in args.categories:
        run_for_category(cat)
    log("=== done ===")


if __name__ == "__main__":
    main()