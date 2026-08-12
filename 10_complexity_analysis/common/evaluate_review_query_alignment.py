#!/usr/bin/env python3
"""Evaluate alignment between review-derived style and selected-query style."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import kruskal, pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path("/fs04/ar57/wenyu")
CATEGORY = os.environ.get("PQ_CATEGORY", "Baby_Products")
OUTPUT_TAG = os.environ.get("VADES_OUTPUT_TAG", "vades_lite_sentence_user_distribution_train10_holdout10")
CLAUSE_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / CATEGORY

SENTENCE_FILE = CLAUSE_DIR / f"{OUTPUT_TAG}_sentences.jsonl"
SELECTED_QUERY_FILE = CLAUSE_DIR / f"{OUTPUT_TAG}_selected_query_records.jsonl"
GMM_USER_FILE = CLAUSE_DIR / "strict5550_query_gmm_user_profiles.jsonl"

REVIEW_USER_STYLE_ALIGNMENT_FILE = CLAUSE_DIR / "review_query_user_style_alignment_summary.json"
REVIEW_USER_STYLE_ALIGNMENT_USER_RECORDS_FILE = CLAUSE_DIR / "review_query_user_style_alignment_user_records.jsonl"
REVIEW_SHARED_PCA_USER_FILE = CLAUSE_DIR / "review_query_shared_pca_user_profiles.jsonl"
REVIEW_SHARED_PCA_QUERY_FILE = CLAUSE_DIR / "review_query_shared_pca_query_records.jsonl"
REVIEW_SHARED_PCA_SUMMARY_FILE = CLAUSE_DIR / "review_query_shared_pca_summary.json"
SELECTED_QUERY_SHARED_PCA_ALIGNMENT_FILE = CLAUSE_DIR / "selected_query_shared_pca_alignment_summary.json"
SELECTED_QUERY_SHARED_PCA_USER_FILE = CLAUSE_DIR / "selected_query_shared_pca_user_profiles.jsonl"
SELECTED_QUERY_SHARED_PCA_RECORD_FILE = CLAUSE_DIR / "selected_query_shared_pca_records.jsonl"


from common_utils import log  # 统一 log 函数


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"{path} 为空")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def summarize_array(values: np.ndarray) -> dict:
    if len(values) == 0:
        raise ValueError("无法汇总空数组")
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.5)),
        "q75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
    }


def load_gmm_cluster_index_by_user() -> dict[str, int]:
    rows = load_jsonl(GMM_USER_FILE)
    cluster_by_user: dict[str, int] = {}
    for row in rows:
        user_id = row.get("user_id")
        cluster_index = row.get("cluster_index")
        if user_id is None or cluster_index is None:
            raise ValueError(f"{GMM_USER_FILE} 缺少 user_id 或 cluster_index")
        cluster_by_user[user_id] = int(cluster_index)
    return cluster_by_user


def load_review_sentence_feature_matrix() -> tuple[list[dict], list[str], np.ndarray]:
    rows = load_jsonl(SENTENCE_FILE)
    feature_names: list[str] | None = None
    matrix: list[list[float]] = []
    normalized_rows: list[dict] = []
    for row in rows:
        user_id = row.get("user_id")
        features = row.get("features")
        if user_id is None or features is None:
            raise ValueError(f"{SENTENCE_FILE} 行缺少 user_id 或 features")
        if feature_names is None:
            feature_names = list(features.keys())
        elif list(features.keys()) != feature_names:
            raise ValueError(f"{SENTENCE_FILE} 特征字段顺序不一致: user_id={user_id}")
        feature_vector = [float(features[name]) for name in feature_names]
        normalized_rows.append(
            {
                "user_id": user_id,
                "review_index": row.get("review_index"),
                "sentence_index": row.get("sentence_index"),
                "sentence_text": row.get("sentence_text"),
                "word_count": row.get("word_count"),
                "feature_vector": feature_vector,
            }
        )
        matrix.append(feature_vector)
    if feature_names is None:
        raise ValueError(f"{SENTENCE_FILE} 未读取到特征字段")
    return normalized_rows, feature_names, np.asarray(matrix, dtype=np.float64)


def load_selected_query_feature_rows(feature_names: list[str]) -> tuple[list[dict], np.ndarray]:
    rows = load_jsonl(SELECTED_QUERY_FILE)
    normalized_rows: list[dict] = []
    matrix: list[list[float]] = []
    for row in rows:
        user_id = row.get("user_id")
        features = row.get("features")
        if user_id is None or features is None:
            raise ValueError(f"{SELECTED_QUERY_FILE} 行缺少 user_id 或 features")
        if list(features.keys()) != feature_names:
            raise ValueError(f"{SELECTED_QUERY_FILE} 特征字段顺序不一致: user_id={user_id}")
        feature_vector = [float(features[name]) for name in feature_names]
        normalized_row = dict(row)
        normalized_row["feature_vector"] = feature_vector
        normalized_rows.append(normalized_row)
        matrix.append(feature_vector)
    return normalized_rows, np.asarray(matrix, dtype=np.float64)


def assign_axis_sign(
    review_rows: list[dict],
    review_scores: np.ndarray,
    query_rows: list[dict],
    query_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    review_scores_by_user: dict[str, list[float]] = {}
    for row, score in zip(review_rows, review_scores, strict=True):
        review_scores_by_user.setdefault(row["user_id"], []).append(float(score))

    query_scores_by_user: dict[str, float] = {}
    for row, score in zip(query_rows, query_scores, strict=True):
        user_id = row["user_id"]
        if user_id in query_scores_by_user:
            raise ValueError(f"selected query 中用户重复: {user_id}")
        query_scores_by_user[user_id] = float(score)

    shared_user_ids = sorted(set(review_scores_by_user.keys()) & set(query_scores_by_user.keys()))
    if not shared_user_ids:
        raise ValueError("review/query 之间没有重叠用户，无法确定 PCA 方向")

    review_user_means = np.asarray(
        [float(np.mean(review_scores_by_user[user_id])) for user_id in shared_user_ids],
        dtype=np.float64,
    )
    query_user_values = np.asarray(
        [query_scores_by_user[user_id] for user_id in shared_user_ids],
        dtype=np.float64,
    )
    corr, _ = pearsonr(review_user_means, query_user_values)
    if corr < 0:
        return -review_scores, -query_scores, -corr
    return review_scores, query_scores, corr


def build_user_review_profiles(review_rows: list[dict], review_scores: np.ndarray) -> tuple[list[dict], dict[str, dict]]:
    scores_by_user: dict[str, list[float]] = {}
    for row, score in zip(review_rows, review_scores, strict=True):
        scores_by_user.setdefault(row["user_id"], []).append(float(score))

    user_rows: list[dict] = []
    user_lookup: dict[str, dict] = {}
    for user_id in sorted(scores_by_user.keys()):
        score_array = np.asarray(scores_by_user[user_id], dtype=np.float64)
        summary = summarize_array(score_array)
        record = {
            "user_id": user_id,
            "review_sentence_count": int(len(score_array)),
            "shared_pca_score_summary": summary,
            "shared_pca_score_mean": float(summary["mean"]),
            "shared_pca_score_std": float(summary["std"]),
            "shared_pca_score_p25": float(summary["q25"]),
            "shared_pca_score_p75": float(summary["q75"]),
        }
        user_rows.append(record)
        user_lookup[user_id] = record
    return user_rows, user_lookup


def build_query_records(
    query_rows: list[dict],
    query_scores: np.ndarray,
    review_profile_lookup: dict[str, dict],
    cluster_by_user: dict[str, int],
) -> list[dict]:
    records: list[dict] = []
    for row, score in zip(query_rows, query_scores, strict=True):
        user_id = row["user_id"]
        review_profile = review_profile_lookup.get(user_id)
        if review_profile is None:
            raise ValueError(f"query 用户缺少 review profile: {user_id}")
        cluster_index = cluster_by_user.get(user_id)
        if cluster_index is None:
            raise ValueError(f"GMM 分组缺少用户: {user_id}")
        records.append(
            {
                "user_id": user_id,
                "asin": row.get("asin"),
                "query_type": "expression_style_query",
                "target_depth": row.get("target_depth"),
                "actual_depth": row.get("actual_depth"),
                "user_avg_depth": row.get("user_avg_depth"),
                "query": row.get("query"),
                "cluster_index": int(cluster_index),
                "shared_pca_score": float(score),
                "review_sentence_count": int(review_profile["review_sentence_count"]),
                "review_shared_pca_score_summary": review_profile["shared_pca_score_summary"],
            }
        )
    return records


def build_selected_query_records(
    query_rows: list[dict],
    query_scores: np.ndarray,
    review_profile_lookup: dict[str, dict],
    cluster_by_user: dict[str, int],
) -> list[dict]:
    records: list[dict] = []
    for row, score in zip(query_rows, query_scores, strict=True):
        user_id = row["user_id"]
        review_profile = review_profile_lookup.get(user_id)
        if review_profile is None:
            raise ValueError(f"selected query 用户缺少 review profile: {user_id}")
        cluster_index = cluster_by_user.get(user_id)
        if cluster_index is None:
            raise ValueError(f"GMM 分组缺少用户: {user_id}")
        normalized = dict(row)
        normalized.pop("feature_vector", None)
        normalized["cluster_index"] = int(cluster_index)
        normalized["shared_pca_score"] = float(score)
        normalized["review_sentence_count"] = int(review_profile["review_sentence_count"])
        normalized["review_shared_pca_score_summary"] = review_profile["shared_pca_score_summary"]
        records.append(normalized)
    return records


def build_user_alignment_rows(query_records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in query_records:
        review_summary = row["review_shared_pca_score_summary"]
        rows.append(
            {
                "user_id": row["user_id"],
                "cluster_index": int(row["cluster_index"]),
                "review_user_shared_pca_score": float(review_summary["mean"]),
                "query_shared_pca_score": float(row["shared_pca_score"]),
            }
        )
    return rows


def build_gmm_cluster_groups(user_rows: list[dict]) -> dict:
    groups: dict[int, dict[str, list[float]]] = {}
    for row in user_rows:
        cluster_index = int(row["cluster_index"])
        groups.setdefault(cluster_index, {"review": [], "query": []})
        groups[cluster_index]["review"].append(float(row["review_user_shared_pca_score"]))
        groups[cluster_index]["query"].append(float(row["query_shared_pca_score"]))

    cluster_indices = sorted(groups.keys())
    group_summary: dict[str, dict] = {}
    query_means: list[float] = []
    for cluster_index in cluster_indices:
        review_values = np.asarray(groups[cluster_index]["review"], dtype=np.float64)
        query_values = np.asarray(groups[cluster_index]["query"], dtype=np.float64)
        group_summary[f"cluster_{cluster_index}"] = {
            "cluster_index": cluster_index,
            "user_count": int(len(query_values)),
            "review_user_pca_summary": summarize_array(review_values),
            "query_user_pca_summary": summarize_array(query_values),
        }
        query_means.append(float(np.mean(query_values)))

    kruskal_stat, kruskal_pvalue = kruskal(
        *[np.asarray(groups[idx]["query"], dtype=np.float64) for idx in cluster_indices]
    )
    return {
        "grouping_source": str(GMM_USER_FILE),
        "cluster_indices": cluster_indices,
        "groups": group_summary,
        "query_user_pca_mean_by_cluster": {
            f"cluster_{idx}": float(mean_value) for idx, mean_value in zip(cluster_indices, query_means, strict=True)
        },
        "kruskal_wallis": {
            "statistic": float(kruskal_stat),
            "p_value": float(kruskal_pvalue),
        },
    }


def build_correlation(review_values: np.ndarray, query_values: np.ndarray) -> dict:
    pearson_r, pearson_p = pearsonr(review_values, query_values)
    spearman_rho, spearman_p = spearmanr(review_values, query_values)
    slope, intercept = np.polyfit(review_values, query_values, 1)
    return {
        "pearson_r": float(pearson_r),
        "pearson_p_value": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p_value": float(spearman_p),
        "linear_fit_slope": float(slope),
        "linear_fit_intercept": float(intercept),
    }


def build_relative_to_reviews_summary(
    query_records: list[dict],
    user_lookup: dict[str, dict],
) -> dict:
    percentiles: list[float] = []
    shifts: list[float] = []
    zscores: list[float] = []
    above_mean = 0
    above_p75 = 0
    below_p25 = 0

    for row in query_records:
        review_summary = user_lookup[row["user_id"]]["shared_pca_score_summary"]
        query_score = float(row["shared_pca_score"])
        p25 = float(review_summary["q25"])
        p75 = float(review_summary["q75"])
        mean = float(review_summary["mean"])
        std = float(review_summary["std"])
        min_value = float(review_summary["min"])
        max_value = float(review_summary["max"])

        if max_value == min_value:
            percentile = 0.5
        elif query_score <= min_value:
            percentile = 0.0
        elif query_score >= max_value:
            percentile = 1.0
        else:
            percentile = (query_score - min_value) / (max_value - min_value)
        percentiles.append(float(percentile))

        shift = query_score - mean
        shifts.append(float(shift))
        zscore = shift / std if std != 0 else 0.0
        zscores.append(float(zscore))

        if query_score > mean:
            above_mean += 1
        if query_score > p75:
            above_p75 += 1
        if query_score < p25:
            below_p25 += 1

    total = len(query_records)
    if total == 0:
        raise ValueError("query_records 为空")

    return {
        "mean_percentile": float(np.mean(percentiles)),
        "median_percentile": float(np.median(percentiles)),
        "mean_shift": float(np.mean(shifts)),
        "median_shift": float(np.median(shifts)),
        "mean_zscore": float(np.mean(zscores)),
        "median_zscore": float(np.median(zscores)),
        "share_above_user_review_mean": float(above_mean / total),
        "share_above_user_review_p75": float(above_p75 / total),
        "share_below_user_review_p25": float(below_p25 / total),
    }


def main() -> None:
    cluster_by_user = load_gmm_cluster_index_by_user()
    review_rows, feature_names, review_matrix = load_review_sentence_feature_matrix()
    query_rows, query_matrix = load_selected_query_feature_rows(feature_names)

    scaler = StandardScaler()
    review_matrix_scaled = scaler.fit_transform(review_matrix)
    query_matrix_scaled = scaler.transform(query_matrix)

    pca = PCA(n_components=1, random_state=42)
    review_scores = pca.fit_transform(review_matrix_scaled).reshape(-1)
    query_scores = pca.transform(query_matrix_scaled).reshape(-1)
    review_scores, query_scores, anchor_correlation = assign_axis_sign(
        review_rows=review_rows,
        review_scores=review_scores,
        query_rows=query_rows,
        query_scores=query_scores,
    )

    review_user_rows, review_user_lookup = build_user_review_profiles(review_rows, review_scores)
    query_records = build_query_records(query_rows, query_scores, review_user_lookup, cluster_by_user)
    selected_query_records = build_selected_query_records(query_rows, query_scores, review_user_lookup, cluster_by_user)
    user_alignment_rows = build_user_alignment_rows(query_records)

    review_user_values = np.asarray([row["review_user_shared_pca_score"] for row in user_alignment_rows], dtype=np.float64)
    query_user_values = np.asarray([row["query_shared_pca_score"] for row in user_alignment_rows], dtype=np.float64)
    correlation = build_correlation(review_user_values, query_user_values)
    alignment_by_gmm_cluster = build_gmm_cluster_groups(user_alignment_rows)

    write_jsonl(REVIEW_SHARED_PCA_USER_FILE, review_user_rows)
    write_jsonl(REVIEW_SHARED_PCA_QUERY_FILE, query_records)
    write_jsonl(REVIEW_USER_STYLE_ALIGNMENT_USER_RECORDS_FILE, user_alignment_rows)
    write_jsonl(SELECTED_QUERY_SHARED_PCA_USER_FILE, review_user_rows)
    write_jsonl(SELECTED_QUERY_SHARED_PCA_RECORD_FILE, selected_query_records)

    review_summary = {
        "category": CATEGORY,
        "review_profile_file": str(REVIEW_SHARED_PCA_USER_FILE),
        "query_record_file": str(REVIEW_SHARED_PCA_QUERY_FILE),
        "num_review_profiles": int(len(review_user_rows)),
        "num_query_records": int(len(query_records)),
        "num_query_users": int(len(user_alignment_rows)),
        "user_level_review_pca_summary": summarize_array(review_user_values),
        "user_level_query_pca_summary": summarize_array(query_user_values),
        "correlation": correlation,
        "alignment_by_gmm_cluster": alignment_by_gmm_cluster,
    }
    REVIEW_USER_STYLE_ALIGNMENT_FILE.write_text(
        json.dumps(review_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    shared_pca_summary = {
        "category": CATEGORY,
        "review_sentence_file": str(SENTENCE_FILE),
        "selected_query_file": str(SELECTED_QUERY_FILE),
        "shared_pca_fit_source": "review_sentences_only",
        "num_review_sentence_rows": int(len(review_rows)),
        "num_selected_query_rows": int(len(selected_query_records)),
        "num_selected_query_users": int(len(user_alignment_rows)),
        "explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
        "anchor_correlation": float(anchor_correlation),
        "review_shared_pca_score_summary": summarize_array(review_scores),
        "query_shared_pca_score_summary": summarize_array(query_scores),
        "query_relative_to_user_reviews": build_relative_to_reviews_summary(query_records, review_user_lookup),
        "user_level_alignment": {
            "num_query_users": int(len(user_alignment_rows)),
            "user_level_review_pca_summary": summarize_array(review_user_values),
            "user_level_query_pca_summary": summarize_array(query_user_values),
            "correlation": correlation,
            "alignment_by_gmm_cluster": alignment_by_gmm_cluster,
        },
    }
    REVIEW_SHARED_PCA_SUMMARY_FILE.write_text(
        json.dumps(shared_pca_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    SELECTED_QUERY_SHARED_PCA_ALIGNMENT_FILE.write_text(
        json.dumps(shared_pca_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(f"已写入: {REVIEW_USER_STYLE_ALIGNMENT_FILE}")
    log(f"已写入: {REVIEW_SHARED_PCA_SUMMARY_FILE}")
    log(f"已写入: {SELECTED_QUERY_SHARED_PCA_ALIGNMENT_FILE}")


if __name__ == "__main__":
    main()