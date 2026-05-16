#!/usr/bin/env python3
"""Minimal probe for memory-policy distillation in recommendation.

The experiment is intentionally small and CPU-only. It uses synthetic
recommendation sessions where a teacher emits explicit memory operations:
read, write, compress, forget, and tool-vs-memory selection. A student model
then learns these decisions from teacher trajectories and is compared against
a context-only baseline that can read static memory but cannot learn when to
use, update, compress, or forget it.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TOPICS = ("sci_fi", "fantasy", "mystery", "romance", "history", "cooking")
ITEMS_BY_TOPIC = {
    topic: tuple(f"{topic}_item_{idx}" for idx in range(8)) for topic in TOPICS
}
ACTION_NAMES = ("memory", "tool")
WRITE_NAMES = ("skip_write", "write")
COMPRESS_NAMES = ("skip_compress", "compress")
FORGET_NAMES = ("keep", "forget")


@dataclass(frozen=True)
class Session:
    user_id: int
    history_topic_counts: dict[str, int]
    current_topic: str
    drift_topic: str
    candidate_items: tuple[str, ...]
    target_item: str
    target_topic: str
    is_drift: bool
    memory_topic: str
    memory_strength: int


@dataclass(frozen=True)
class TeacherDecision:
    select_action: int
    write_action: int
    compress_action: int
    forget_action: int
    recommendation: str
    trajectory: dict[str, object]


class LinearSoftmaxClassifier:
    def __init__(self, num_features: int, num_classes: int, learning_rate: float, epochs: int) -> None:
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1")
        self.weights = [[0.0 for _ in range(num_features)] for _ in range(num_classes)]
        self.bias = [0.0 for _ in range(num_classes)]
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, features: list[list[float]], labels: list[int]) -> None:
        if not features:
            raise ValueError("features cannot be empty")
        if len(features) != len(labels):
            raise ValueError("features and labels must have the same length")
        expected_dim = len(self.weights[0])
        for row in features:
            if len(row) != expected_dim:
                raise ValueError("feature dimension mismatch")

        for _ in range(self.epochs):
            for x, label in zip(features, labels):
                probs = self.predict_proba(x)
                for cls_idx in range(len(self.weights)):
                    error = probs[cls_idx] - (1.0 if cls_idx == label else 0.0)
                    for feat_idx, value in enumerate(x):
                        self.weights[cls_idx][feat_idx] -= self.learning_rate * error * value
                    self.bias[cls_idx] -= self.learning_rate * error

    def predict(self, x: list[float]) -> int:
        probs = self.predict_proba(x)
        return max(range(len(probs)), key=lambda idx: probs[idx])

    def predict_proba(self, x: list[float]) -> list[float]:
        scores = []
        for cls_weights, cls_bias in zip(self.weights, self.bias):
            score = cls_bias + sum(weight * value for weight, value in zip(cls_weights, x))
            scores.append(score)
        max_score = max(scores)
        exps = [math.exp(score - max_score) for score in scores]
        total = sum(exps)
        if total <= 0:
            raise ValueError("softmax denominator must be positive")
        return [value / total for value in exps]


class MemoryPolicyStudent:
    def __init__(self, num_features: int, learning_rate: float, epochs: int) -> None:
        self.select_model = LinearSoftmaxClassifier(num_features, 2, learning_rate, epochs)
        self.write_model = LinearSoftmaxClassifier(num_features, 2, learning_rate, epochs)
        self.compress_model = LinearSoftmaxClassifier(num_features, 2, learning_rate, epochs)
        self.forget_model = LinearSoftmaxClassifier(num_features, 2, learning_rate, epochs)

    def fit(self, features: list[list[float]], decisions: list[TeacherDecision]) -> None:
        self.select_model.fit(features, [decision.select_action for decision in decisions])
        self.write_model.fit(features, [decision.write_action for decision in decisions])
        self.compress_model.fit(features, [decision.compress_action for decision in decisions])
        self.forget_model.fit(features, [decision.forget_action for decision in decisions])

    def predict(self, features: list[float]) -> tuple[int, int, int, int]:
        return (
            self.select_model.predict(features),
            self.write_model.predict(features),
            self.compress_model.predict(features),
            self.forget_model.predict(features),
        )


class StaticMemoryContextBaseline:
    def predict(self, session: Session) -> tuple[int, int, int, int]:
        # This baseline has memory as external context but no learned memory policy.
        if session.memory_strength >= 2:
            select_action = 0
        else:
            select_action = 1
        return select_action, 0, 0, 0


def topic_from_item(item: str) -> str:
    for topic in TOPICS:
        prefix = f"{topic}_item_"
        if item.startswith(prefix):
            return topic
    raise ValueError(f"Unknown item topic: {item}")


def build_tool_call_trace(
    *,
    agent_name: str,
    tool_name: str,
    tool_args: dict[str, Any],
    tool_result: dict[str, Any],
    observation: str,
    latency_ms: float | None = None,
) -> dict[str, Any]:
    trace = {
        "agent": agent_name,
        "tool_name": tool_name,
        "tool_args": tool_args,
        "tool_result": tool_result,
        "observation": observation,
        "status": "success",
    }
    if latency_ms is not None:
        trace["latency_ms"] = latency_ms
    return trace


def choose_item(topic: str, rng: random.Random) -> str:
    if topic not in ITEMS_BY_TOPIC:
        raise ValueError(f"Unknown topic: {topic}")
    return rng.choice(ITEMS_BY_TOPIC[topic])


def make_candidate_items(target_topic: str, rng: random.Random, candidate_count: int) -> tuple[str, ...]:
    if candidate_count < 2:
        raise ValueError("candidate_count must be at least 2")
    target = choose_item(target_topic, rng)
    candidates = [target]
    other_topics = [topic for topic in TOPICS if topic != target_topic]
    while len(candidates) < candidate_count:
        topic = rng.choice(other_topics)
        item = choose_item(topic, rng)
        if item not in candidates:
            candidates.append(item)
    rng.shuffle(candidates)
    return tuple(candidates)


def generate_sessions(num_users: int, sessions_per_user: int, seed: int) -> list[Session]:
    if num_users <= 0 or sessions_per_user <= 0:
        raise ValueError("num_users and sessions_per_user must be positive")
    rng = random.Random(seed)
    sessions: list[Session] = []

    for user_id in range(num_users):
        base_topic = rng.choice(TOPICS)
        alternate_topic = rng.choice([topic for topic in TOPICS if topic != base_topic])
        history_counts = {topic: 0 for topic in TOPICS}
        history_counts[base_topic] = rng.randint(4, 7)
        history_counts[alternate_topic] = rng.randint(1, 3)
        memory_topic = max(history_counts, key=history_counts.get)
        memory_strength = history_counts[memory_topic]

        for step in range(sessions_per_user):
            drift_probability = 0.22 + 0.05 * (step % 3)
            is_drift = rng.random() < drift_probability
            current_topic = rng.choice([topic for topic in TOPICS if topic != memory_topic]) if is_drift else memory_topic
            target_topic = current_topic
            candidates = make_candidate_items(target_topic, rng, candidate_count=6)
            target = next(item for item in candidates if topic_from_item(item) == target_topic)
            sessions.append(
                Session(
                    user_id=user_id,
                    history_topic_counts=dict(history_counts),
                    current_topic=current_topic,
                    drift_topic=current_topic if is_drift else "",
                    candidate_items=candidates,
                    target_item=target,
                    target_topic=target_topic,
                    is_drift=is_drift,
                    memory_topic=memory_topic,
                    memory_strength=memory_strength,
                )
            )

            history_counts[current_topic] += 1
            sorted_topics = sorted(history_counts.items(), key=lambda item: item[1], reverse=True)
            memory_topic = sorted_topics[0][0]
            memory_strength = sorted_topics[0][1]

    return sessions


def teacher_policy(session: Session) -> TeacherDecision:
    memory_matches = session.memory_topic == session.current_topic
    history_total = sum(session.history_topic_counts.values())
    topic_count = session.history_topic_counts[session.current_topic]
    recent_is_weak = topic_count <= 1

    use_tool = (not memory_matches) or recent_is_weak
    select_action = 1 if use_tool else 0
    write_action = 1 if session.is_drift or topic_count <= 2 else 0
    compress_action = 1 if history_total >= 10 and session.memory_strength >= 6 else 0
    forget_action = 1 if session.is_drift and session.memory_strength >= 6 and not memory_matches else 0

    candidate_topics = {item: topic_from_item(item) for item in session.candidate_items}
    current_matches = [item for item, topic in candidate_topics.items() if topic == session.current_topic]
    memory_topic_matches = [item for item, topic in candidate_topics.items() if topic == session.memory_topic]
    shortlist = current_matches if current_matches else (
        memory_topic_matches if memory_topic_matches else list(session.candidate_items[:3])
    )
    tool_calls = []
    if use_tool:
        tool_calls.append(
            build_tool_call_trace(
                agent_name="MemoryAgent",
                tool_name="lookup_candidate_topic_alignment",
                tool_args={
                    "candidate_items": list(session.candidate_items),
                    "current_topic": session.current_topic,
                    "memory_topic": session.memory_topic,
                },
                tool_result={
                    "candidate_topics": candidate_topics,
                    "current_topic_matches": current_matches,
                    "memory_topic_matches": memory_topic_matches,
                    "shortlist": shortlist,
                    "topic_alignment": "current_topic" if current_matches else ("memory_topic" if memory_topic_matches else "none"),
                    "memory_conflict": not memory_matches,
                },
                observation=(
                    f"current_topic_matches={current_matches}; memory_topic_matches={memory_topic_matches}; "
                    f"shortlist={shortlist}"
                ),
            )
        )

    recommendation_topic = session.current_topic if use_tool else session.memory_topic
    recommendation = first_candidate_for_topic(session.candidate_items, recommendation_topic)
    trajectory = {
        "memory_plan": {
            "read_key": f"user:{session.user_id}:dominant_topic",
            "tool_vs_memory": ACTION_NAMES[select_action],
            "reason": "drift_or_sparse_signal" if use_tool else "stable_memory_match",
        },
        "memory_read": {
            "topic": session.memory_topic,
            "strength": session.memory_strength,
        },
        "tool_calls_raw": tool_calls,
        "memory_write": {
            "decision": WRITE_NAMES[write_action],
            "topic": session.current_topic if write_action else "",
        },
        "memory_compress": {
            "decision": COMPRESS_NAMES[compress_action],
        },
        "memory_forget": {
            "decision": FORGET_NAMES[forget_action],
            "topic": session.memory_topic if forget_action else "",
        },
        "recommend": recommendation,
    }
    return TeacherDecision(select_action, write_action, compress_action, forget_action, recommendation, trajectory)


def first_candidate_for_topic(candidates: Iterable[str], topic: str) -> str:
    for item in candidates:
        if topic_from_item(item) == topic:
            return item
    raise ValueError(f"No candidate for topic: {topic}")


def featurize(session: Session) -> list[float]:
    total = sum(session.history_topic_counts.values())
    if total <= 0:
        raise ValueError("history must be non-empty")
    features = []
    for topic in TOPICS:
        features.append(session.history_topic_counts[topic] / total)
    features.extend(1.0 if session.current_topic == topic else 0.0 for topic in TOPICS)
    features.append(session.memory_strength / total)
    features.append(1.0 if session.memory_topic == session.current_topic else 0.0)
    features.append(1.0 if session.is_drift else 0.0)
    features.append(total / 20.0)
    return features


def recommend_with_policy(session: Session, policy: tuple[int, int, int, int]) -> str:
    select_action, write_action, _, forget_action = policy
    if select_action == 1:
        topic = session.current_topic
    elif write_action == 1:
        topic = session.current_topic
    elif forget_action == 1:
        topic = session.current_topic
    else:
        topic = session.memory_topic
    for item in session.candidate_items:
        if topic_from_item(item) == topic:
            return item
    return f"missing_candidate_for_{topic}"


def top1_accuracy(sessions: list[Session], policies: list[tuple[int, int, int, int]]) -> float:
    if len(sessions) != len(policies):
        raise ValueError("sessions and policies length mismatch")
    correct = 0
    for session, policy in zip(sessions, policies):
        if recommend_with_policy(session, policy) == session.target_item:
            correct += 1
    return correct / len(sessions)


def action_accuracy(
    predicted: list[tuple[int, int, int, int]],
    teacher: list[TeacherDecision],
    action_index: int,
) -> float:
    if len(predicted) != len(teacher):
        raise ValueError("prediction and teacher length mismatch")
    correct = 0
    for pred, decision in zip(predicted, teacher):
        labels = (
            decision.select_action,
            decision.write_action,
            decision.compress_action,
            decision.forget_action,
        )
        if pred[action_index] == labels[action_index]:
            correct += 1
    return correct / len(teacher)


def summarize_policy_counts(decisions: list[TeacherDecision]) -> dict[str, dict[str, int]]:
    return {
        "tool_vs_memory": dict(Counter(ACTION_NAMES[d.select_action] for d in decisions)),
        "write": dict(Counter(WRITE_NAMES[d.write_action] for d in decisions)),
        "compress": dict(Counter(COMPRESS_NAMES[d.compress_action] for d in decisions)),
        "forget": dict(Counter(FORGET_NAMES[d.forget_action] for d in decisions)),
    }


def grouped_accuracy_by_drift(
    sessions: list[Session],
    policies: list[tuple[int, int, int, int]],
) -> dict[str, float]:
    groups: dict[str, list[tuple[Session, tuple[int, int, int, int]]]] = defaultdict(list)
    for session, policy in zip(sessions, policies):
        groups["drift" if session.is_drift else "stable"].append((session, policy))
    result = {}
    for name, rows in groups.items():
        row_sessions = [row[0] for row in rows]
        row_policies = [row[1] for row in rows]
        result[name] = top1_accuracy(row_sessions, row_policies)
    return result


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    sessions = generate_sessions(args.users, args.sessions_per_user, args.seed)
    split = int(len(sessions) * args.train_ratio)
    if split <= 0 or split >= len(sessions):
        raise ValueError("train_ratio creates an invalid split")
    train_sessions = sessions[:split]
    test_sessions = sessions[split:]

    train_decisions = [teacher_policy(session) for session in train_sessions]
    test_decisions = [teacher_policy(session) for session in test_sessions]
    train_features = [featurize(session) for session in train_sessions]
    test_features = [featurize(session) for session in test_sessions]

    student = MemoryPolicyStudent(len(train_features[0]), args.learning_rate, args.epochs)
    student.fit(train_features, train_decisions)

    student_policies = [student.predict(features) for features in test_features]
    baseline = StaticMemoryContextBaseline()
    baseline_policies = [baseline.predict(session) for session in test_sessions]
    teacher_policies = [
        (
            decision.select_action,
            decision.write_action,
            decision.compress_action,
            decision.forget_action,
        )
        for decision in test_decisions
    ]

    metrics = {
        "teacher_top1": top1_accuracy(test_sessions, teacher_policies),
        "student_top1": top1_accuracy(test_sessions, student_policies),
        "context_only_top1": top1_accuracy(test_sessions, baseline_policies),
        "student_tool_vs_memory_accuracy": action_accuracy(student_policies, test_decisions, 0),
        "student_write_accuracy": action_accuracy(student_policies, test_decisions, 1),
        "student_compress_accuracy": action_accuracy(student_policies, test_decisions, 2),
        "student_forget_accuracy": action_accuracy(student_policies, test_decisions, 3),
        "context_only_tool_vs_memory_accuracy": action_accuracy(baseline_policies, test_decisions, 0),
        "context_only_write_accuracy": action_accuracy(baseline_policies, test_decisions, 1),
        "context_only_compress_accuracy": action_accuracy(baseline_policies, test_decisions, 2),
        "context_only_forget_accuracy": action_accuracy(baseline_policies, test_decisions, 3),
    }
    metrics["student_gain_over_context_only"] = metrics["student_top1"] - metrics["context_only_top1"]

    if metrics["student_top1"] < args.min_student_top1:
        raise AssertionError(
            f"student_top1={metrics['student_top1']:.4f} below threshold {args.min_student_top1:.4f}"
        )
    if metrics["student_gain_over_context_only"] < args.min_gain:
        raise AssertionError(
            "student gain over context-only baseline is too small: "
            f"{metrics['student_gain_over_context_only']:.4f} < {args.min_gain:.4f}"
        )
    if metrics["student_tool_vs_memory_accuracy"] < args.min_policy_accuracy:
        raise AssertionError(
            "student tool-vs-memory policy accuracy is too low: "
            f"{metrics['student_tool_vs_memory_accuracy']:.4f} < {args.min_policy_accuracy:.4f}"
        )
    if metrics["student_write_accuracy"] < args.min_policy_accuracy:
        raise AssertionError(
            "student write policy accuracy is too low: "
            f"{metrics['student_write_accuracy']:.4f} < {args.min_policy_accuracy:.4f}"
        )

    examples = []
    for session, decision in zip(test_sessions[: args.example_count], test_decisions[: args.example_count]):
        examples.append(
            {
                "user_id": session.user_id,
                "current_topic": session.current_topic,
                "target_item": session.target_item,
                "teacher_trajectory": decision.trajectory,
            }
        )

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }

    return {
        "config": config,
        "sizes": {
            "train": len(train_sessions),
            "test": len(test_sessions),
        },
        "teacher_policy_counts": summarize_policy_counts(train_decisions + test_decisions),
        "metrics": metrics,
        "student_by_group": grouped_accuracy_by_drift(test_sessions, student_policies),
        "context_only_by_group": grouped_accuracy_by_drift(test_sessions, baseline_policies),
        "examples": examples,
        "claim": (
            "The learned student internalizes memory-operation decisions and improves recommendation "
            "accuracy over a context-only memory baseline on held-out synthetic sessions."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=80)
    parser.add_argument("--sessions-per-user", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--min-student-top1", type=float, default=0.75)
    parser.add_argument("--min-gain", type=float, default=0.10)
    parser.add_argument("--min-policy-accuracy", type=float, default=0.75)
    parser.add_argument("--example-count", type=int, default=3)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment(args)
    output = json.dumps(result, indent=2, sort_keys=True)
    print(output)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
