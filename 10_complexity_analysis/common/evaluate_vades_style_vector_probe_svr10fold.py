#!/usr/bin/env python3
"""Probe learned user style vectors against real syntactic style features."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR


REPO_ROOT = Path("/fs04/ar57/wenyu")
CATEGORY = os.environ.get("PQ_CATEGORY", "Baby_Products")
OUTPUT_TAG = os.environ.get("VADES_OUTPUT_TAG", "vades_lite_sentence_user_distribution_train10_holdout10")
CLAUSE_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / CATEGORY

REPRESENTATION_FIELD = "user_mu"
N_SPLITS = 10
RANDOM_STATE = 42
MODEL_NAME = "svr_rbf"
MAX_USERS_OVERRIDE = os.environ.get("STYLE_PROBE_MAX_USERS")

USER_PROFILE_FILE = CLAUSE_DIR / f"{OUTPUT_TAG}_user_profiles.jsonl"
SENTENCE_FILE = CLAUSE_DIR / f"{OUTPUT_TAG}_sentences.jsonl"
SUMMARY_FILE = CLAUSE_DIR / f"{OUTPUT_TAG}_style_vector_eval_summary.json"
PER_FEATURE_FILE = CLAUSE_DIR / f"{OUTPUT_TAG}_style_vector_eval_per_feature.jsonl"
FOLD_FILE = CLAUSE_DIR / f"{OUTPUT_TAG}_style_vector_eval_fold_metrics.jsonl"


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


def load_user_style_vectors() -> tuple[list[str], np.ndarray]:
    log(f"开始读取 learned style vectors: {USER_PROFILE_FILE}")
    rows = load_jsonl(USER_PROFILE_FILE)
    user_ids: list[str] = []
    vectors: list[list[float]] = []
    for row in rows:
        vector = row.get(REPRESENTATION_FIELD)
        if vector is None:
            raise ValueError(f"{USER_PROFILE_FILE} 缺少字段 {REPRESENTATION_FIELD}: user_id={row.get('user_id')}")
        user_id = row.get("user_id")
        if user_id is None:
            raise ValueError(f"{USER_PROFILE_FILE} 缺少 user_id")
        user_ids.append(user_id)
        vectors.append(vector)
    return user_ids, np.asarray(vectors, dtype=np.float64)


def load_true_style_targets() -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    log(f"开始读取真实句法风格特征并按用户聚合: {SENTENCE_FILE}")
    rows = load_jsonl(SENTENCE_FILE)
    feature_names: list[str] | None = None
    user_feature_rows: dict[str, list[list[float]]] = {}
    sentence_counts: list[int] = []
    for row in rows:
        user_id = row.get("user_id")
        features = row.get("features")
        if user_id is None or features is None:
            raise ValueError(f"{SENTENCE_FILE} 行缺少 user_id 或 features")
        if feature_names is None:
            feature_names = list(features.keys())
        else:
            current_feature_names = list(features.keys())
            if current_feature_names != feature_names:
                raise ValueError(f"{SENTENCE_FILE} 特征字段顺序不一致: user_id={user_id}")
        user_feature_rows.setdefault(user_id, []).append([float(features[name]) for name in feature_names])

    if feature_names is None:
        raise ValueError(f"{SENTENCE_FILE} 未读取到特征字段")

    user_ids = sorted(user_feature_rows.keys())
    targets: list[list[float]] = []
    for user_id in user_ids:
        feature_matrix = np.asarray(user_feature_rows[user_id], dtype=np.float64)
        sentence_counts.append(int(feature_matrix.shape[0]))
        targets.append(feature_matrix.mean(axis=0).tolist())

    return (
        user_ids,
        np.asarray(targets, dtype=np.float64),
        feature_names,
        np.asarray(sentence_counts, dtype=np.float64),
    )


def align_x_y(
    x_user_ids: list[str],
    x_vectors: np.ndarray,
    y_user_ids: list[str],
    y_targets: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    x_by_user = {user_id: x_vectors[idx] for idx, user_id in enumerate(x_user_ids)}
    shared_user_ids = [user_id for user_id in y_user_ids if user_id in x_by_user]
    if not shared_user_ids:
        raise ValueError("X 和 y 没有重叠用户")
    x_aligned = np.asarray([x_by_user[user_id] for user_id in shared_user_ids], dtype=np.float64)
    y_by_user = {user_id: y_targets[idx] for idx, user_id in enumerate(y_user_ids)}
    y_aligned = np.asarray([y_by_user[user_id] for user_id in shared_user_ids], dtype=np.float64)
    if MAX_USERS_OVERRIDE is not None:
        max_users = int(MAX_USERS_OVERRIDE)
        if max_users <= 0:
            raise ValueError("STYLE_PROBE_MAX_USERS 必须是正整数")
        shared_user_ids = shared_user_ids[:max_users]
        x_aligned = x_aligned[:max_users]
        y_aligned = y_aligned[:max_users]
    return shared_user_ids, x_aligned, y_aligned


def evaluate_probe(x_matrix: np.ndarray, y_matrix: np.ndarray, feature_names: list[str]) -> tuple[dict, list[dict], list[dict]]:
    log(f"开始执行 {MODEL_NAME} 的 {N_SPLITS}-fold cross validation")
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    fold_rows: list[dict] = []
    per_feature_mse_values: dict[str, list[float]] = {name: [] for name in feature_names}
    overall_mse_values: list[float] = []

    for fold_index, (train_idx, test_idx) in enumerate(kfold.split(x_matrix), start=1):
        model = MultiOutputRegressor(SVR(kernel="rbf", C=1.0, epsilon=0.1))
        model.fit(x_matrix[train_idx], y_matrix[train_idx])
        prediction = model.predict(x_matrix[test_idx])
        mse_by_feature = np.mean((prediction - y_matrix[test_idx]) ** 2, axis=0)
        overall_mse = float(np.mean(mse_by_feature))
        overall_mse_values.append(overall_mse)
        fold_rows.append(
            {
                "model": MODEL_NAME,
                "fold_index": fold_index,
                "train_user_count": int(len(train_idx)),
                "test_user_count": int(len(test_idx)),
                "overall_mse": overall_mse,
            }
        )
        for feature_index, feature_name in enumerate(feature_names):
            per_feature_mse_values[feature_name].append(float(mse_by_feature[feature_index]))

    per_feature_rows = [
        {
            "model": MODEL_NAME,
            "feature_name": feature_name,
            "mse": float(np.mean(values)),
        }
        for feature_name, values in per_feature_mse_values.items()
    ]
    summary = {
        "model": MODEL_NAME,
        "n_splits": N_SPLITS,
        "overall_mse_mean": float(np.mean(overall_mse_values)),
        "overall_mse_std": float(np.std(overall_mse_values)),
        "per_feature_mse_summary": summarize_array(
            np.asarray([row["mse"] for row in per_feature_rows], dtype=np.float64)
        ),
    }
    return summary, per_feature_rows, fold_rows


def main() -> None:
    x_user_ids, x_vectors = load_user_style_vectors()
    y_user_ids, y_targets, feature_names, sentence_counts = load_true_style_targets()
    shared_user_ids, x_aligned, y_aligned = align_x_y(x_user_ids, x_vectors, y_user_ids, y_targets)

    summary, per_feature_rows, fold_rows = evaluate_probe(x_aligned, y_aligned, feature_names)
    write_jsonl(PER_FEATURE_FILE, per_feature_rows)
    write_jsonl(FOLD_FILE, fold_rows)

    result = {
        "category": CATEGORY,
        "x_source": str(USER_PROFILE_FILE),
        "x_representation_field": REPRESENTATION_FIELD,
        "y_source": str(SENTENCE_FILE),
        "y_aggregation": "mean_over_user_sentences",
        "evaluation_type": "10-fold cross validation on users",
        "style_vector_dim": int(x_aligned.shape[1]),
        "style_feature_dim": int(y_aligned.shape[1]),
        "user_count": int(len(shared_user_ids)),
        "feature_names": feature_names,
        "sentence_count_per_user_summary": summarize_array(sentence_counts),
        "probe_model": MODEL_NAME,
        "main_reported_metric": "mse",
        "cross_validation": "10-fold",
        "summary": summary,
        "per_feature_file": str(PER_FEATURE_FILE),
        "fold_file": str(FOLD_FILE),
    }
    SUMMARY_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"已写入: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()