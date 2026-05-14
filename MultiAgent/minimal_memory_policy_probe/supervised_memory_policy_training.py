#!/usr/bin/env python3
"""Export teacher trajectories and train a small supervised memory-policy model."""

from __future__ import annotations

import gzip
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from llm_memory_policy_distillation_probe import (
    ACTION_NAMES,
    COMPRESS_NAMES,
    FORGET_NAMES,
    WRITE_NAMES,
    ExperimentConfig,
    Session,
    TeacherDecision,
    load_reviews,
    teacher_payload,
    teacher_policy,
)


PROBE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROBE_DIR / "supervised_memory_policy"
DATASET_JSONL = OUTPUT_DIR / "teacher_trajectories.jsonl"
MODEL_JSON = OUTPUT_DIR / "small_policy_model.json"
RESULT_JSON = OUTPUT_DIR / "small_policy_result.json"


@dataclass(frozen=True)
class TrainingConfig:
    review_file: Path = Path("/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2023/Baby_Products.jsonl.gz")
    meta_file: Path = Path("/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2023/meta_Baby_Products.jsonl.gz")
    users: int = 500
    sessions_per_user: int = 4
    max_candidate_items: int = 8
    train_ratio: float = 0.8
    epochs: int = 40
    learning_rate: float = 0.08
    feature_hash_size: int = 512
    min_train_examples: int = 10


class BinaryLogisticModel:
    def __init__(self, feature_size: int, learning_rate: float, epochs: int) -> None:
        if feature_size <= 0:
            raise ValueError("feature_size must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        self.weights = [0.0 for _ in range(feature_size)]
        self.bias = 0.0
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, rows: list[dict[int, float]], labels: list[int]) -> None:
        if not rows:
            raise ValueError("rows cannot be empty")
        if len(rows) != len(labels):
            raise ValueError("rows and labels length mismatch")
        for label in labels:
            if label not in (0, 1):
                raise ValueError(f"invalid binary label: {label}")

        for _ in range(self.epochs):
            for features, label in zip(rows, labels):
                pred = self.predict_proba(features)
                error = pred - label
                for idx, value in features.items():
                    self.weights[idx] -= self.learning_rate * error * value
                self.bias -= self.learning_rate * error

    def predict(self, features: dict[int, float]) -> int:
        return 1 if self.predict_proba(features) >= 0.5 else 0

    def predict_proba(self, features: dict[int, float]) -> float:
        score = self.bias
        for idx, value in features.items():
            score += self.weights[idx] * value
        if score >= 0:
            z = math.exp(-score)
            return 1.0 / (1.0 + z)
        z = math.exp(score)
        return z / (1.0 + z)

    def to_json(self) -> dict[str, Any]:
        return {"weights": self.weights, "bias": self.bias}


def open_gzip_text(path: Path):
    if path.suffix != ".gz":
        raise ValueError(f"Expected gzip file: {path}")
    return gzip.open(path, "rt", encoding="utf-8")


def load_topic_lookup(meta_file: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    with open_gzip_text(meta_file) as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if "parent_asin" not in record:
                continue
            if "categories" not in record:
                continue
            categories = record["categories"]
            if not isinstance(categories, list) or len(categories) < 2:
                continue
            topic = categories[1]
            if not isinstance(topic, str) or not topic.strip():
                continue
            lookup[str(record["parent_asin"])] = topic.strip()
    if not lookup:
        raise ValueError(f"No usable second-level topics loaded from {meta_file}")
    return lookup


def build_supervised_sessions(config: TrainingConfig) -> list[Session]:
    topic_lookup = load_topic_lookup(config.meta_file)
    reviews = load_reviews(config.review_file)
    reviews_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for review in reviews:
        asin = str(review["asin"])
        if asin in topic_lookup:
            reviews_by_user[str(review["user_id"])].append(review)

    sessions: list[Session] = []
    for source_user_id in sorted(reviews_by_user):
        if len(sessions) >= config.users:
            break
        user_reviews = reviews_by_user[source_user_id]
        if len(user_reviews) < config.sessions_per_user + 1:
            continue
        user_reviews.sort(key=lambda item: item["timestamp"] if "timestamp" in item else 0)
        history_slice = user_reviews[: config.sessions_per_user]
        target_review = user_reviews[config.sessions_per_user]

        history_topic_counts: dict[str, int] = defaultdict(int)
        for review in history_slice:
            history_topic_counts[topic_lookup[str(review["asin"])]] += 1
        if not history_topic_counts:
            raise ValueError(f"Expected non-empty history topics for user {source_user_id}")

        current_topic = topic_lookup[str(target_review["asin"])]
        memory_topic = max(history_topic_counts, key=history_topic_counts.get)
        memory_strength = history_topic_counts[memory_topic]
        candidate_items = tuple(
            dict.fromkeys(
                [str(target_review["asin"])]
                + [str(review["asin"]) for review in history_slice]
                + [
                    str(review["asin"])
                    for review in user_reviews[
                        config.sessions_per_user + 1 : config.sessions_per_user + 1 + config.max_candidate_items
                    ]
                    if str(review["asin"]) in topic_lookup
                ]
            )
        )
        if len(candidate_items) < 2:
            continue

        sessions.append(
            Session(
                user_id=len(sessions),
                history_topic_counts=dict(history_topic_counts),
                current_topic=current_topic,
                drift_topic=current_topic if current_topic != memory_topic else "",
                candidate_items=candidate_items[: config.max_candidate_items],
                target_item=str(target_review["asin"]),
                target_topic=current_topic,
                is_drift=current_topic != memory_topic,
                memory_topic=memory_topic,
                memory_strength=memory_strength,
            )
        )

    if len(sessions) < config.min_train_examples:
        raise ValueError(f"Only built {len(sessions)} sessions; need at least {config.min_train_examples}")
    return sessions


def session_payload(session: Session) -> dict[str, Any]:
    return {
        "user_id": session.user_id,
        "history_topic_counts": session.history_topic_counts,
        "current_topic_signal": session.current_topic,
        "candidate_items": list(session.candidate_items),
        "external_memory": {
            "dominant_topic": session.memory_topic,
            "strength": session.memory_strength,
        },
    }


def supervised_record(session: Session, decision: TeacherDecision) -> dict[str, Any]:
    return {
        "input": session_payload(session),
        "label": teacher_payload(decision),
        "target_item": session.target_item,
        "target_topic": session.target_topic,
        "is_drift": session.is_drift,
    }


def export_teacher_trajectories(sessions: list[Session], decisions: list[TeacherDecision], path: Path) -> None:
    if len(sessions) != len(decisions):
        raise ValueError("sessions and decisions length mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for session, decision in zip(sessions, decisions):
            handle.write(json.dumps(supervised_record(session, decision), ensure_ascii=False, sort_keys=True) + "\n")


def add_feature(features: dict[int, float], key: str, value: float, size: int) -> None:
    idx = hash(key) % size
    features[idx] = features.get(idx, 0.0) + value


def featurize_policy(session: Session, size: int) -> dict[int, float]:
    total = sum(session.history_topic_counts.values())
    if total <= 0:
        raise ValueError("history total must be positive")
    features: dict[int, float] = {}
    add_feature(features, "bias", 1.0, size)
    add_feature(features, f"current_topic={session.current_topic}", 1.0, size)
    add_feature(features, f"memory_topic={session.memory_topic}", 1.0, size)
    add_feature(features, f"memory_matches={session.memory_topic == session.current_topic}", 1.0, size)
    add_feature(features, f"is_drift={session.is_drift}", 1.0, size)
    add_feature(features, "memory_strength_ratio", session.memory_strength / total, size)
    add_feature(features, "history_total", total / 20.0, size)
    for topic, count in session.history_topic_counts.items():
        add_feature(features, f"history_topic={topic}", count / total, size)
    return features


def featurize_candidate(session: Session, candidate: str, size: int) -> dict[int, float]:
    features = featurize_policy(session, size)
    add_feature(features, f"candidate={candidate}", 1.0, size)
    add_feature(features, f"candidate_seen={candidate in session.candidate_items}", 1.0, size)
    return features


def accuracy(predicted: list[int], labels: list[int]) -> float:
    if len(predicted) != len(labels):
        raise ValueError("predicted and labels length mismatch")
    if not labels:
        raise ValueError("labels cannot be empty")
    return sum(1 for pred, label in zip(predicted, labels) if pred == label) / len(labels)


def recommendation_accuracy(sessions: list[Session], predicted_items: list[str]) -> float:
    if len(sessions) != len(predicted_items):
        raise ValueError("sessions and predicted_items length mismatch")
    return sum(1 for session, item in zip(sessions, predicted_items) if item == session.target_item) / len(sessions)


def fit_action_model(
    train_sessions: list[Session],
    train_decisions: list[TeacherDecision],
    label_getter,
    config: TrainingConfig,
) -> BinaryLogisticModel:
    model = BinaryLogisticModel(config.feature_hash_size, config.learning_rate, config.epochs)
    rows = [featurize_policy(session, config.feature_hash_size) for session in train_sessions]
    labels = [label_getter(decision) for decision in train_decisions]
    model.fit(rows, labels)
    return model


def fit_candidate_ranker(
    train_sessions: list[Session],
    config: TrainingConfig,
) -> BinaryLogisticModel:
    model = BinaryLogisticModel(config.feature_hash_size, config.learning_rate, config.epochs)
    rows: list[dict[int, float]] = []
    labels: list[int] = []
    for session in train_sessions:
        for candidate in session.candidate_items:
            rows.append(featurize_candidate(session, candidate, config.feature_hash_size))
            labels.append(1 if candidate == session.target_item else 0)
    model.fit(rows, labels)
    return model


def predict_recommendation(model: BinaryLogisticModel, session: Session, config: TrainingConfig) -> str:
    if not session.candidate_items:
        raise ValueError("candidate_items cannot be empty")
    scored = [
        (model.predict_proba(featurize_candidate(session, candidate, config.feature_hash_size)), candidate)
        for candidate in session.candidate_items
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def train_and_evaluate(config: TrainingConfig) -> dict[str, Any]:
    sessions = build_supervised_sessions(config)
    decisions = [teacher_policy(session) for session in sessions]
    export_teacher_trajectories(sessions, decisions, DATASET_JSONL)

    split = int(len(sessions) * config.train_ratio)
    if split <= 0 or split >= len(sessions):
        raise ValueError("train_ratio creates an invalid split")
    train_sessions = sessions[:split]
    test_sessions = sessions[split:]
    train_decisions = decisions[:split]
    test_decisions = decisions[split:]

    action_models = {
        "tool_vs_memory": fit_action_model(train_sessions, train_decisions, lambda d: d.select_action, config),
        "memory_write": fit_action_model(train_sessions, train_decisions, lambda d: d.write_action, config),
        "memory_compress": fit_action_model(train_sessions, train_decisions, lambda d: d.compress_action, config),
        "memory_forget": fit_action_model(train_sessions, train_decisions, lambda d: d.forget_action, config),
    }
    ranker = fit_candidate_ranker(train_sessions, config)

    test_features = [featurize_policy(session, config.feature_hash_size) for session in test_sessions]
    predictions = {
        name: [model.predict(features) for features in test_features]
        for name, model in action_models.items()
    }
    labels = {
        "tool_vs_memory": [decision.select_action for decision in test_decisions],
        "memory_write": [decision.write_action for decision in test_decisions],
        "memory_compress": [decision.compress_action for decision in test_decisions],
        "memory_forget": [decision.forget_action for decision in test_decisions],
    }
    predicted_items = [predict_recommendation(ranker, session, config) for session in test_sessions]

    metrics = {
        "tool_vs_memory_accuracy": accuracy(predictions["tool_vs_memory"], labels["tool_vs_memory"]),
        "memory_write_accuracy": accuracy(predictions["memory_write"], labels["memory_write"]),
        "memory_compress_accuracy": accuracy(predictions["memory_compress"], labels["memory_compress"]),
        "memory_forget_accuracy": accuracy(predictions["memory_forget"], labels["memory_forget"]),
        "recommendation_top1": recommendation_accuracy(test_sessions, predicted_items),
    }
    model_payload = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "action_models": {name: model.to_json() for name, model in action_models.items()},
        "candidate_ranker": ranker.to_json(),
        "label_names": {
            "tool_vs_memory": list(ACTION_NAMES),
            "memory_write": list(WRITE_NAMES),
            "memory_compress": list(COMPRESS_NAMES),
            "memory_forget": list(FORGET_NAMES),
        },
    }
    MODEL_JSON.write_text(json.dumps(model_payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "config": model_payload["config"],
        "dataset_jsonl": str(DATASET_JSONL),
        "model_json": str(MODEL_JSON),
        "sizes": {
            "total": len(sessions),
            "train": len(train_sessions),
            "test": len(test_sessions),
            "teacher_trajectory_records": len(decisions),
        },
        "label_distribution": {
            "tool_vs_memory": dict(Counter(ACTION_NAMES[decision.select_action] for decision in decisions)),
            "memory_write": dict(Counter(WRITE_NAMES[decision.write_action] for decision in decisions)),
            "memory_compress": dict(Counter(COMPRESS_NAMES[decision.compress_action] for decision in decisions)),
            "memory_forget": dict(Counter(FORGET_NAMES[decision.forget_action] for decision in decisions)),
            "drift": dict(Counter("drift" if session.is_drift else "stable" for session in sessions)),
        },
        "metrics": metrics,
        "examples": [
            {
                "input": session_payload(session),
                "target_item": session.target_item,
                "predicted_item": predicted_item,
                "teacher": teacher_payload(decision),
                "predicted_policy": {
                    "tool_vs_memory": ACTION_NAMES[predictions["tool_vs_memory"][idx]],
                    "memory_write": WRITE_NAMES[predictions["memory_write"][idx]],
                    "memory_compress": COMPRESS_NAMES[predictions["memory_compress"][idx]],
                    "memory_forget": FORGET_NAMES[predictions["memory_forget"][idx]],
                },
            }
            for idx, (session, decision, predicted_item) in enumerate(
                zip(test_sessions[:3], test_decisions[:3], predicted_items[:3])
            )
        ],
    }


def main() -> None:
    config = TrainingConfig()
    result = train_and_evaluate(config)
    RESULT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
