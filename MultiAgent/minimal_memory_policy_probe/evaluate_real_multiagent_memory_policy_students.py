#!/usr/bin/env python3
"""Evaluate memory-policy students against real multi-agent LLM teacher labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from llm_memory_policy_distillation_probe import (
    ACTION_NAMES,
    COMPRESS_NAMES,
    FORGET_NAMES,
    PREFERENCE_TYPES,
    WRITE_NAMES,
    Session,
    TeacherDecision,
    ranking_metrics,
    teacher_payload,
)
from real_multiagent_teacher_distillation import EVAL_JSONL, TRAIN_JSONL
from supervised_memory_policy_training import TrainingConfig, session_payload
from evaluate_memory_policy_students import (
    ParsedPrediction,
    logistic_predictions,
    minimax_predictions,
    summarize_predictions,
)


PROBE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_JSON = PROBE_DIR / "real_multiagent_teacher" / "real_multiagent_eval_result.json"


@dataclass(frozen=True)
class RealEvalConfig:
    train_jsonl: Path = TRAIN_JSONL
    eval_jsonl: Path = EVAL_JSONL
    output_json: Path = DEFAULT_OUTPUT_JSON
    minimax_model: str | None = None
    minimax_max_tokens: int = 1024
    minimax_max_retries: int = 5
    include_minimax_prompt_only: bool = True
    include_logistic_baseline: bool = True


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
    required = (
        "tool_vs_memory",
        "memory_write",
        "memory_compress",
        "memory_forget",
        "preference_type",
        "ranking",
        "trajectory",
    )
    missing = [key for key in required if key not in label]
    if missing:
        raise ValueError(f"teacher label missing required keys: {missing}")

    tool_vs_memory = str(label["tool_vs_memory"])
    memory_write = str(label["memory_write"])
    memory_compress = str(label["memory_compress"])
    memory_forget = str(label["memory_forget"])
    preference_type = str(label["preference_type"])
    ranking_raw = label["ranking"]
    trajectory = label["trajectory"]

    if tool_vs_memory not in ACTION_NAMES:
        raise ValueError(f"invalid teacher tool_vs_memory: {tool_vs_memory}")
    if memory_write not in WRITE_NAMES:
        raise ValueError(f"invalid teacher memory_write: {memory_write}")
    if memory_compress not in COMPRESS_NAMES:
        raise ValueError(f"invalid teacher memory_compress: {memory_compress}")
    if memory_forget not in FORGET_NAMES:
        raise ValueError(f"invalid teacher memory_forget: {memory_forget}")
    if preference_type not in PREFERENCE_TYPES:
        raise ValueError(f"invalid teacher preference_type: {preference_type}")
    if not isinstance(ranking_raw, list):
        raise ValueError("teacher ranking must be a JSON array")
    ranking = tuple(str(item) for item in ranking_raw)
    if not ranking:
        raise ValueError("teacher ranking cannot be empty")
    if not isinstance(trajectory, dict):
        raise ValueError("teacher trajectory must be a JSON object")
    required_trajectory = ("profile_agent", "memory_agent", "recommendation_agent", "critic_agent")
    missing_trajectory = [key for key in required_trajectory if key not in trajectory]
    if missing_trajectory:
        raise ValueError(f"teacher trajectory missing required keys: {missing_trajectory}")
    for key in ("profile_agent", "memory_agent", "recommendation_agent", "critic_agent"):
        if not isinstance(trajectory[key], str):
            raise ValueError(f"teacher trajectory.{key} must be a string")

    return TeacherDecision(
        select_action=ACTION_NAMES.index(tool_vs_memory),
        write_action=WRITE_NAMES.index(memory_write),
        compress_action=COMPRESS_NAMES.index(memory_compress),
        forget_action=FORGET_NAMES.index(memory_forget),
        recommendation=ranking[0],
        trajectory=trajectory,
        preference_type=preference_type,
        ranking=ranking,
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
    candidate_topics_raw = input_payload.get("candidate_topics")
    current_topic_matches_raw = input_payload.get("current_topic_matches")
    memory_topic_matches_raw = input_payload.get("memory_topic_matches")
    shortlist_raw = input_payload.get("shortlist")
    if not isinstance(candidate_topics_raw, dict):
        raise ValueError("candidate_topics must be a JSON object")
    if not isinstance(current_topic_matches_raw, list):
        raise ValueError("current_topic_matches must be a JSON array")
    if not isinstance(memory_topic_matches_raw, list):
        raise ValueError("memory_topic_matches must be a JSON array")
    if not isinstance(shortlist_raw, list):
        raise ValueError("shortlist must be a JSON array")
    target_item = str(record["target_item"])
    if target_item not in candidate_items:
        raise ValueError(f"target_item is not in candidate_items: {target_item}")

    current_topic = str(input_payload["current_topic_signal"])
    is_drift = bool(record["is_drift"])
    return Session(
        user_id=int(input_payload["user_id"]),
        history_topic_counts={str(key): int(value) for key, value in input_payload["history_topic_counts"].items()},
        current_topic=current_topic,
        drift_topic=current_topic if is_drift else "",
        candidate_items=candidate_items,
        candidate_topics={str(key): str(value) for key, value in candidate_topics_raw.items()},
        current_topic_matches=tuple(str(item) for item in current_topic_matches_raw),
        memory_topic_matches=tuple(str(item) for item in memory_topic_matches_raw),
        shortlist=tuple(str(item) for item in shortlist_raw),
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
        "preference_type": dict(Counter(decision.preference_type for decision in decisions)),
    }


def prediction_examples(
    sessions: list[Session],
    decisions: list[TeacherDecision],
    predictions_by_model: dict[str, list[ParsedPrediction]],
    count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, (session, decision) in enumerate(zip(sessions[:count], decisions[:count])):
        row = {
            "index": idx,
            "session": session_payload(session),
            "teacher": teacher_payload(decision),
            "models": {},
        }
        for model_name, predictions in predictions_by_model.items():
            prediction = predictions[idx]
            row["models"][model_name] = {
                "policy": list(prediction.policy) if prediction.policy is not None else None,
                "preference_type": prediction.preference_type,
                "ranking": list(prediction.ranking) if prediction.ranking is not None else None,
                "parse_error": prediction.parse_error,
                "parsed_json": prediction.parsed_json,
                "raw_text": prediction.raw_text[:1200],
            }
        rows.append(row)
    return rows


def run_eval(config: RealEvalConfig) -> dict[str, Any]:
    train_sessions, train_decisions = load_teacher_jsonl(config.train_jsonl)
    eval_sessions, eval_decisions = load_teacher_jsonl(config.eval_jsonl)

    train_ids = {session.user_id for session in train_sessions}
    eval_ids = {session.user_id for session in eval_sessions}
    overlap = train_ids.intersection(eval_ids)
    if overlap:
        raise ValueError(f"Train and eval users overlap: {sorted(overlap)}")

    predictions_by_model: dict[str, list[ParsedPrediction]] = {}

    if config.include_minimax_prompt_only:
        predictions_by_model["minimax_prompt_only"] = minimax_predictions(
            eval_sessions,
            config.minimax_model,
            config.minimax_max_tokens,
            config.minimax_max_retries,
        )

    if config.include_logistic_baseline:
        baseline_config = TrainingConfig(users=len(train_sessions))
        predictions_by_model["logistic_baseline"] = logistic_predictions(
            train_sessions,
            train_decisions,
            eval_sessions,
            baseline_config,
        )

    metrics_by_model = {
        name: summarize_predictions(eval_sessions, eval_decisions, predictions)
        for name, predictions in predictions_by_model.items()
    }

    result = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "sizes": {
            "train_teacher_records": len(train_sessions),
            "eval_teacher_records": len(eval_sessions),
        },
        "teacher_metrics": {
            "teacher_rank_at_1": ranking_metrics(eval_sessions, [decision.ranking for decision in eval_decisions])["rank_at_1"],
            "teacher_mrr": ranking_metrics(eval_sessions, [decision.ranking for decision in eval_decisions])["mrr"],
            "teacher_ndcg@5": ranking_metrics(eval_sessions, [decision.ranking for decision in eval_decisions])["ndcg@5"],
        },
        "train_label_distribution": teacher_policy_distribution(train_decisions),
        "eval_label_distribution": teacher_policy_distribution(eval_decisions),
        "metrics": metrics_by_model,
        "examples": prediction_examples(eval_sessions, eval_decisions, predictions_by_model, count=min(3, len(eval_sessions))),
    }
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> RealEvalConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, default=RealEvalConfig.train_jsonl)
    parser.add_argument("--eval-jsonl", type=Path, default=RealEvalConfig.eval_jsonl)
    parser.add_argument("--output-json", type=Path, default=RealEvalConfig.output_json)
    parser.add_argument("--minimax-model", default=RealEvalConfig.minimax_model)
    parser.add_argument("--minimax-max-tokens", type=int, default=RealEvalConfig.minimax_max_tokens)
    parser.add_argument("--minimax-max-retries", type=int, default=RealEvalConfig.minimax_max_retries)
    parser.add_argument("--skip-minimax-prompt-only", action="store_true")
    parser.add_argument("--skip-logistic-baseline", action="store_true")
    args = parser.parse_args()
    return RealEvalConfig(
        train_jsonl=args.train_jsonl,
        eval_jsonl=args.eval_jsonl,
        output_json=args.output_json,
        minimax_model=args.minimax_model,
        minimax_max_tokens=args.minimax_max_tokens,
        minimax_max_retries=args.minimax_max_retries,
        include_minimax_prompt_only=not args.skip_minimax_prompt_only,
        include_logistic_baseline=not args.skip_logistic_baseline,
    )


def main() -> None:
    config = parse_args()
    result = run_eval(config)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
