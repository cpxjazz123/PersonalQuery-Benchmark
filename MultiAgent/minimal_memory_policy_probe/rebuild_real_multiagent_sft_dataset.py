#!/usr/bin/env python3
"""Rebuild compact SFT data from validated real multi-agent teacher JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from llm_memory_policy_distillation_probe import (
    ACTION_NAMES,
    COMPRESS_NAMES,
    FORGET_NAMES,
    WRITE_NAMES,
    Session,
    TeacherDecision,
    teacher_payload,
)
from real_multiagent_teacher_distillation import (
    EVAL_JSONL,
    LLAMA_FACTORY_SFT_JSON,
    RESULT_JSON,
    TRAIN_JSONL,
    export_real_llamafactory_sft_dataset,
)
from supervised_memory_policy_training import session_payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file does not exist: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} in {path} is not a JSON object")
            records.append(record)
    if not records:
        raise ValueError(f"JSONL file is empty: {path}")
    return records


def decision_from_label(label: dict[str, Any]) -> TeacherDecision:
    required = ("tool_vs_memory", "memory_write", "memory_compress", "memory_forget", "recommend", "trajectory")
    missing = [key for key in required if key not in label]
    if missing:
        raise ValueError(f"teacher label missing required keys: {missing}")

    tool_vs_memory = str(label["tool_vs_memory"])
    memory_write = str(label["memory_write"])
    memory_compress = str(label["memory_compress"])
    memory_forget = str(label["memory_forget"])
    recommendation = str(label["recommend"])
    trajectory = label["trajectory"]

    if tool_vs_memory not in ACTION_NAMES:
        raise ValueError(f"invalid teacher tool_vs_memory: {tool_vs_memory}")
    if memory_write not in WRITE_NAMES:
        raise ValueError(f"invalid teacher memory_write: {memory_write}")
    if memory_compress not in COMPRESS_NAMES:
        raise ValueError(f"invalid teacher memory_compress: {memory_compress}")
    if memory_forget not in FORGET_NAMES:
        raise ValueError(f"invalid teacher memory_forget: {memory_forget}")
    if not isinstance(trajectory, dict):
        raise ValueError("teacher trajectory must be a JSON object")

    return TeacherDecision(
        select_action=ACTION_NAMES.index(tool_vs_memory),
        write_action=WRITE_NAMES.index(memory_write),
        compress_action=COMPRESS_NAMES.index(memory_compress),
        forget_action=FORGET_NAMES.index(memory_forget),
        recommendation=recommendation,
        trajectory=trajectory,
    )


def session_from_record(record: dict[str, Any]) -> Session:
    required = ("input", "label", "target_item", "target_topic", "is_drift")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"teacher record missing required keys: {missing}")

    input_payload = record["input"]
    if not isinstance(input_payload, dict):
        raise ValueError("record input must be a JSON object")
    required_input = ("user_id", "history_topic_counts", "current_topic_signal", "candidate_items", "external_memory")
    missing_input = [key for key in required_input if key not in input_payload]
    if missing_input:
        raise ValueError(f"record input missing required keys: {missing_input}")

    external_memory = input_payload["external_memory"]
    if not isinstance(external_memory, dict):
        raise ValueError("external_memory must be a JSON object")
    if "dominant_topic" not in external_memory or "strength" not in external_memory:
        raise ValueError("external_memory missing dominant_topic or strength")

    candidate_items = tuple(str(item) for item in input_payload["candidate_items"])
    if not candidate_items:
        raise ValueError("candidate_items cannot be empty")

    target_item = str(record["target_item"])
    if target_item not in candidate_items:
        raise ValueError(f"target_item is not in candidate_items: {target_item}")

    recommendation = str(record["label"]["recommend"])
    if recommendation not in candidate_items:
        raise ValueError(f"teacher recommendation is not in candidate_items: {recommendation}")

    current_topic = str(input_payload["current_topic_signal"])
    is_drift = bool(record["is_drift"])
    return Session(
        user_id=int(input_payload["user_id"]),
        history_topic_counts={str(key): int(value) for key, value in input_payload["history_topic_counts"].items()},
        current_topic=current_topic,
        drift_topic=current_topic if is_drift else "",
        candidate_items=candidate_items,
        target_item=target_item,
        target_topic=str(record["target_topic"]),
        is_drift=is_drift,
        memory_topic=str(external_memory["dominant_topic"]),
        memory_strength=int(external_memory["strength"]),
    )


def load_teacher_jsonl(path: Path) -> tuple[list[Session], list[TeacherDecision]]:
    records = read_jsonl(path)
    sessions = [session_from_record(record) for record in records]
    decisions = [decision_from_label(record["label"]) for record in records]
    if len(sessions) != len(decisions):
        raise ValueError("sessions and decisions length mismatch")
    return sessions, decisions


def teacher_policy_distribution(decisions: list[TeacherDecision]) -> dict[str, dict[str, int]]:
    return {
        "tool_vs_memory": dict(Counter(ACTION_NAMES[decision.select_action] for decision in decisions)),
        "memory_write": dict(Counter(WRITE_NAMES[decision.write_action] for decision in decisions)),
        "memory_compress": dict(Counter(COMPRESS_NAMES[decision.compress_action] for decision in decisions)),
        "memory_forget": dict(Counter(FORGET_NAMES[decision.forget_action] for decision in decisions)),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, default=TRAIN_JSONL)
    parser.add_argument("--eval-jsonl", type=Path, default=EVAL_JSONL)
    parser.add_argument("--output-json", type=Path, default=LLAMA_FACTORY_SFT_JSON)
    parser.add_argument("--result-json", type=Path, default=RESULT_JSON)
    parser.add_argument("--expected-train-count", type=int)
    parser.add_argument("--expected-eval-count", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_sessions, train_decisions = load_teacher_jsonl(args.train_jsonl)
    eval_sessions, eval_decisions = load_teacher_jsonl(args.eval_jsonl)

    if args.expected_train_count is not None and len(train_sessions) != args.expected_train_count:
        raise ValueError(f"Expected {args.expected_train_count} train records, got {len(train_sessions)}")
    if args.expected_eval_count is not None and len(eval_sessions) != args.expected_eval_count:
        raise ValueError(f"Expected {args.expected_eval_count} eval records, got {len(eval_sessions)}")

    train_ids = {session.user_id for session in train_sessions}
    eval_ids = {session.user_id for session in eval_sessions}
    overlap = train_ids.intersection(eval_ids)
    if overlap:
        raise ValueError(f"Train and eval users overlap: {sorted(overlap)}")

    export_real_llamafactory_sft_dataset(train_sessions, train_decisions, args.output_json)

    result = {
        "artifacts": {
            "eval_jsonl": str(args.eval_jsonl),
            "llamafactory_sft_json": str(args.output_json),
            "train_jsonl": str(args.train_jsonl),
        },
        "source": "validated_jsonl_rebuild",
        "sizes": {
            "eval": len(eval_sessions),
            "total_teacher_calls": 4 * (len(train_sessions) + len(eval_sessions)),
            "train": len(train_sessions),
        },
        "train_label_distribution": teacher_policy_distribution(train_decisions),
        "eval_label_distribution": teacher_policy_distribution(eval_decisions),
        "examples": [
            {
                "session": session_payload(session),
                "target_item": session.target_item,
                "teacher": teacher_payload(decision),
            }
            for session, decision in zip(eval_sessions[:2], eval_decisions[:2])
        ],
    }
    args.result_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    error_path = args.result_json.parent / "teacher_generation_error.json"
    if error_path.exists():
        error_path.unlink()

    result = {
        "train_jsonl": str(args.train_jsonl),
        "eval_jsonl": str(args.eval_jsonl),
        "output_json": str(args.output_json),
        "result_json": str(args.result_json),
        "train_records": len(train_sessions),
        "eval_records": len(eval_sessions),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
