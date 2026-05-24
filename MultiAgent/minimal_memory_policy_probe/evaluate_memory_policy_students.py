#!/usr/bin/env python3
"""Evaluate free-generation memory-policy students on unseen recommendation sessions."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from llm_memory_policy_distillation_probe import (
    ACTION_NAMES,
    COMPRESS_NAMES,
    FORGET_NAMES,
    build_candidate_alignment,
    PREFERENCE_TYPES,
    preference_vector_for_type,
    preference_type_for_session,
    rank_candidates,
    ranking_metrics,
    WRITE_NAMES,
    Session,
    TeacherDecision,
    create_llm_client,
    load_reviews,
    teacher_payload,
)
from supervised_memory_policy_training import (
    TrainingConfig,
    featurize_policy,
    fit_action_model,
    fit_preference_model,
    load_topic_lookup,
    predict_ranking,
    session_payload,
    sft_input,
    sft_instruction,
    sft_system_prompt,
)


PROBE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_JSON = PROBE_DIR / "supervised_memory_policy" / "free_generation_eval_result.json"


@dataclass(frozen=True)
class EvalConfig:
    output_json: Path = DEFAULT_OUTPUT_JSON
    train_users: int = 500
    eval_count: int = 20
    eval_offset: int = 500
    sessions_per_user: int = 4
    max_candidate_items: int = 8
    minimax_model: str | None = None
    minimax_max_tokens: int = 1024
    minimax_max_retries: int = 5
    logistic_baseline_enabled: bool = True


@dataclass(frozen=True)
class ParsedPrediction:
    policy: tuple[int, int, int, int] | None
    preference_type: str | None
    ranking: tuple[str, ...] | None
    recommendation: str | None
    parsed_json: dict[str, Any] | None
    raw_text: str
    parse_error: str | None


def build_supervised_sessions_with_offset(
    config: TrainingConfig,
    offset: int,
    count: int,
) -> list[Session]:
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if count <= 0:
        raise ValueError("count must be positive")

    topic_lookup = load_topic_lookup(config.meta_file)
    reviews = load_reviews(config.review_file)
    reviews_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for review in reviews:
        asin = str(review["asin"])
        if asin in topic_lookup:
            reviews_by_user[str(review["user_id"])].append(review)

    sessions: list[Session] = []
    eligible_index = 0
    for source_user_id in sorted(reviews_by_user):
        if len(sessions) >= count:
            break
        user_reviews = reviews_by_user[source_user_id]
        if len(user_reviews) < config.sessions_per_user + 1:
            continue
        user_reviews.sort(key=lambda item: item["timestamp"] if "timestamp" in item else 0)
        if eligible_index < offset:
            eligible_index += 1
            continue

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
        )[: config.max_candidate_items]
        eligible_index += 1
        if len(candidate_items) < 2:
            continue

        candidate_topics = {item: topic_lookup[item] for item in candidate_items}
        candidate_topics, current_topic_matches, memory_topic_matches, shortlist = build_candidate_alignment(
            candidate_items,
            candidate_topics,
            current_topic,
            memory_topic,
        )

        sessions.append(
            Session(
                user_id=offset + len(sessions),
                history_topic_counts=dict(history_topic_counts),
                current_topic=current_topic,
                drift_topic=current_topic if current_topic != memory_topic else "",
                candidate_items=candidate_items,
                candidate_topics=candidate_topics,
                current_topic_matches=current_topic_matches,
                memory_topic_matches=memory_topic_matches,
                shortlist=shortlist,
                target_item=current_topic_matches[0],
                target_topic=current_topic,
                is_drift=current_topic != memory_topic,
                memory_topic=memory_topic,
                memory_strength=memory_strength,
            )
        )

    if len(sessions) != count:
        raise ValueError(f"Only built {len(sessions)} eval sessions; expected {count}")
    return sessions


def extract_json_object(text: str) -> dict[str, Any]:
    if not text.strip():
        raise ValueError("model returned an empty response")
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    raw = fenced.group(1) if fenced else text
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"response does not contain a JSON object: {text[:300]}")
    return json.loads(raw[start : end + 1])


def parse_generated_policy(text: str, session: Session) -> ParsedPrediction:
    try:
        parsed = extract_json_object(text)
        required = (
            "tool_vs_memory",
            "memory_write",
            "memory_compress",
            "memory_forget",
            "preference_type",
            "ranking",
            "trajectory",
        )
        missing = [key for key in required if key not in parsed]
        if missing:
            raise ValueError(f"missing required keys: {missing}")

        tool_vs_memory = str(parsed["tool_vs_memory"])
        memory_write = str(parsed["memory_write"])
        memory_compress = str(parsed["memory_compress"])
        memory_forget = str(parsed["memory_forget"])
        preference_type = str(parsed["preference_type"])
        ranking_raw = parsed["ranking"]

        if tool_vs_memory not in ACTION_NAMES:
            raise ValueError(f"invalid tool_vs_memory: {tool_vs_memory}")
        if memory_write not in WRITE_NAMES:
            raise ValueError(f"invalid memory_write: {memory_write}")
        if memory_compress not in COMPRESS_NAMES:
            raise ValueError(f"invalid memory_compress: {memory_compress}")
        if memory_forget not in FORGET_NAMES:
            raise ValueError(f"invalid memory_forget: {memory_forget}")
        if preference_type not in PREFERENCE_TYPES:
            raise ValueError(f"invalid preference_type: {preference_type}")
        if not isinstance(ranking_raw, list):
            raise ValueError("ranking must be a JSON array")
        ranking = tuple(str(item) for item in ranking_raw)
        if len(ranking) != len(session.candidate_items):
            raise ValueError("ranking must contain every candidate item exactly once")
        if Counter(ranking) != Counter(session.candidate_items):
            raise ValueError("ranking items do not match candidate_items")
        if preference_type in ("aligned", "current_priority") and ranking[0] not in session.current_topic_matches:
            raise ValueError(f"ranking top item must come from current_topic_matches for {preference_type}: {ranking[0]}")
        if preference_type == "memory_priority" and ranking[0] not in session.memory_topic_matches:
            raise ValueError(f"ranking top item must come from memory_topic_matches for memory_priority: {ranking[0]}")
        trajectory = parsed["trajectory"]
        if not isinstance(trajectory, dict):
            raise ValueError("trajectory must be a JSON object")
        trajectory_required = ("memory_plan", "memory_read", "memory_write", "memory_compress", "memory_forget", "preference", "ranking")
        trajectory_missing = [key for key in trajectory_required if key not in trajectory]
        if trajectory_missing:
            raise ValueError(f"trajectory missing required keys: {trajectory_missing}")
        if not isinstance(trajectory["memory_plan"], dict):
            raise ValueError("trajectory.memory_plan must be a JSON object")
        if not isinstance(trajectory["memory_read"], dict):
            raise ValueError("trajectory.memory_read must be a JSON object")
        if not isinstance(trajectory["memory_write"], dict):
            raise ValueError("trajectory.memory_write must be a JSON object")
        if not isinstance(trajectory["memory_compress"], dict):
            raise ValueError("trajectory.memory_compress must be a JSON object")
        if not isinstance(trajectory["memory_forget"], dict):
            raise ValueError("trajectory.memory_forget must be a JSON object")
        if not isinstance(trajectory["preference"], dict):
            raise ValueError("trajectory.preference must be a JSON object")
        if not isinstance(trajectory["ranking"], list):
            raise ValueError("trajectory.ranking must be a JSON array")

        return ParsedPrediction(
            policy=(
                ACTION_NAMES.index(tool_vs_memory),
                WRITE_NAMES.index(memory_write),
                COMPRESS_NAMES.index(memory_compress),
                FORGET_NAMES.index(memory_forget),
            ),
            preference_type=preference_type,
            ranking=ranking,
            recommendation=ranking[0],
            parsed_json=parsed,
            raw_text=text,
            parse_error=None,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        return ParsedPrediction(
            policy=None,
            preference_type=None,
            ranking=None,
            recommendation=None,
            parsed_json=None,
            raw_text=text,
            parse_error=str(exc),
        )


def build_minimax_prompt_parts(session: Session) -> tuple[str, str]:
    return (
        "\n\n".join([sft_system_prompt(), sft_instruction()]),
        f"Target input:\n{sft_input(session)}",
    )


def teacher_policy_real(session: Session, topic_lookup: dict[str, str]) -> TeacherDecision:
    memory_matches = session.memory_topic == session.current_topic
    history_total = sum(session.history_topic_counts.values())
    topic_count = session.history_topic_counts.get(session.current_topic, 0)
    recent_is_weak = topic_count <= 1

    use_tool = (not memory_matches) or recent_is_weak
    select_action = 1 if use_tool else 0
    write_action = 1 if session.is_drift or topic_count <= 2 else 0
    compress_action = 1 if history_total >= 10 and session.memory_strength >= 6 else 0
    forget_action = 1 if session.is_drift and session.memory_strength >= 6 and not memory_matches else 0

    if not session.current_topic_matches:
        raise ValueError(f"No current topic matches for session {session.user_id}")
    preference_type = preference_type_for_session(session)
    ranking = rank_candidates(session, preference_type)
    recommendation = ranking[0]

    trajectory = {
        "memory_plan": {
            "read_key": f"user:{session.user_id}:dominant_topic",
            "tool_vs_memory": ACTION_NAMES[select_action],
            "reason": "drift_or_sparse_signal" if use_tool else "stable_memory_match",
        },
        "preference": {
            "type": preference_type,
            "vector": list(preference_vector_for_type(preference_type)),
        },
        "memory_read": {
            "topic": session.memory_topic,
            "strength": session.memory_strength,
        },
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
        "ranking": list(ranking),
    }
    return TeacherDecision(
        select_action,
        write_action,
        compress_action,
        forget_action,
        recommendation,
        trajectory,
        preference_type=preference_type,
        ranking=ranking,
    )


def teacher_policy_tuple(decision: TeacherDecision) -> tuple[int, int, int, int]:
    return (
        decision.select_action,
        decision.write_action,
        decision.compress_action,
        decision.forget_action,
    )


def prediction_ranking_metrics(
    sessions: list[Session],
    predictions: list[ParsedPrediction],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    if len(sessions) != len(predictions):
        raise ValueError("sessions and predictions length mismatch")
    if not sessions:
        raise ValueError("sessions cannot be empty")
    metrics: dict[str, float] = {f"rank_at_{k}": 0.0 for k in ks}
    metrics["mrr"] = 0.0
    for k in ks:
        metrics[f"ndcg@{k}"] = 0.0

    total = float(len(sessions))
    for session, prediction in zip(sessions, predictions):
        ranking = prediction.ranking
        if ranking is None:
            continue
        position = ranking.index(session.target_item) + 1
        metrics["mrr"] += 1.0 / position
        for k in ks:
            if position <= k:
                metrics[f"rank_at_{k}"] += 1.0
                metrics[f"ndcg@{k}"] += 1.0 / math.log2(position + 1)

    for key in metrics:
        metrics[key] /= total
    return metrics


def summarize_predictions(
    sessions: list[Session],
    decisions: list[TeacherDecision],
    predictions: list[ParsedPrediction],
) -> dict[str, Any]:
    if len(sessions) != len(decisions) or len(sessions) != len(predictions):
        raise ValueError("sessions, decisions, and predictions length mismatch")

    labels = [teacher_policy_tuple(decision) for decision in decisions]
    teacher_rankings = [decision.ranking for decision in decisions]
    metrics: dict[str, Any] = {
        "json_parse_success_rate": sum(pred.parse_error is None for pred in predictions) / len(predictions),
        **prediction_ranking_metrics(sessions, predictions),
        "teacher_rank_at_1": ranking_metrics(sessions, teacher_rankings)["rank_at_1"],
        "teacher_mrr": ranking_metrics(sessions, teacher_rankings)["mrr"],
        "teacher_ndcg@5": ranking_metrics(sessions, teacher_rankings)["ndcg@5"],
        "exact_policy_accuracy": sum(pred.policy == label for pred, label in zip(predictions, labels)) / len(predictions),
        "tool_vs_memory_accuracy": sum(
            pred.policy is not None and pred.policy[0] == label[0] for pred, label in zip(predictions, labels)
        )
        / len(predictions),
        "memory_write_accuracy": sum(
            pred.policy is not None and pred.policy[1] == label[1] for pred, label in zip(predictions, labels)
        )
        / len(predictions),
        "memory_compress_accuracy": sum(
            pred.policy is not None and pred.policy[2] == label[2] for pred, label in zip(predictions, labels)
        )
        / len(predictions),
        "memory_forget_accuracy": sum(
            pred.policy is not None and pred.policy[3] == label[3] for pred, label in zip(predictions, labels)
        )
        / len(predictions),
        "preference_type_accuracy": sum(
            pred.preference_type == decision.preference_type for pred, decision in zip(predictions, decisions)
        )
        / len(predictions),
        "parse_errors": dict(Counter(pred.parse_error for pred in predictions if pred.parse_error is not None)),
    }
    return metrics


def logistic_predictions(
    train_sessions: list[Session],
    train_decisions: list[TeacherDecision],
    eval_sessions: list[Session],
    config: TrainingConfig,
) -> list[ParsedPrediction]:
    action_models = {
        "tool_vs_memory": fit_action_model(train_sessions, train_decisions, lambda d: d.select_action, config),
        "memory_write": fit_action_model(train_sessions, train_decisions, lambda d: d.write_action, config),
        "memory_compress": fit_action_model(train_sessions, train_decisions, lambda d: d.compress_action, config),
        "memory_forget": fit_action_model(train_sessions, train_decisions, lambda d: d.forget_action, config),
    }
    preference_model = fit_preference_model(train_sessions, train_decisions, config)
    predictions: list[ParsedPrediction] = []
    for session in eval_sessions:
        features = featurize_policy(session, config.feature_hash_size)
        policy = (
            action_models["tool_vs_memory"].predict(features),
            action_models["memory_write"].predict(features),
            action_models["memory_compress"].predict(features),
            action_models["memory_forget"].predict(features),
        )
        preference_type = PREFERENCE_TYPES[preference_model.predict(features)]
        ranking = predict_ranking(session, preference_type)
        parsed_json = {
            "tool_vs_memory": ACTION_NAMES[policy[0]],
            "memory_write": WRITE_NAMES[policy[1]],
            "memory_compress": COMPRESS_NAMES[policy[2]],
            "memory_forget": FORGET_NAMES[policy[3]],
            "preference_type": preference_type,
            "ranking": list(ranking),
        }
        predictions.append(
            ParsedPrediction(
                policy=policy,
                preference_type=preference_type,
                ranking=ranking,
                recommendation=ranking[0],
                parsed_json=parsed_json,
                raw_text=json.dumps(parsed_json, ensure_ascii=False, sort_keys=True),
                parse_error=None,
            )
        )
    return predictions


def minimax_predictions(
    sessions: list[Session],
    model: str | None,
    max_tokens: int,
    max_retries: int,
) -> list[ParsedPrediction]:
    client = create_llm_client(model)
    predictions: list[ParsedPrediction] = []
    for session in sessions:
        system_base, user_content = build_minimax_prompt_parts(session)
        text, _ = client.call_with_cache(
            system_base=system_base,
            user_content=user_content,
            max_tokens=max_tokens,
            temperature=0.0,
            max_retries=max_retries,
        )
        predictions.append(parse_generated_policy(text, session))
    return predictions


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
            pred = predictions[idx]
            row["models"][model_name] = {
                "policy": list(pred.policy) if pred.policy is not None else None,
                "preference_type": pred.preference_type,
                "ranking": list(pred.ranking) if pred.ranking is not None else None,
                "parse_error": pred.parse_error,
                "parsed_json": pred.parsed_json,
                "raw_text": pred.raw_text[:1200],
            }
        rows.append(row)
    return rows


def run_eval(config: EvalConfig) -> dict[str, Any]:
    train_config = TrainingConfig(
        users=config.train_users,
        sessions_per_user=config.sessions_per_user,
        max_candidate_items=config.max_candidate_items,
    )
    topic_lookup = load_topic_lookup(train_config.meta_file)
    train_sessions = build_supervised_sessions_with_offset(train_config, offset=0, count=config.train_users)
    train_decisions = [teacher_policy_real(session, topic_lookup) for session in train_sessions]
    eval_sessions = build_supervised_sessions_with_offset(
        train_config,
        offset=config.eval_offset,
        count=config.eval_count,
    )
    eval_decisions = [teacher_policy_real(session, topic_lookup) for session in eval_sessions]

    predictions_by_model: dict[str, list[ParsedPrediction]] = {}
    predictions_by_model["minimax_prompt_only"] = minimax_predictions(
        eval_sessions,
        config.minimax_model,
        config.minimax_max_tokens,
        config.minimax_max_retries,
    )
    if config.logistic_baseline_enabled:
        predictions_by_model["logistic_baseline"] = logistic_predictions(train_sessions, train_decisions, eval_sessions, train_config)
    metrics_by_model = {
        name: summarize_predictions(eval_sessions, eval_decisions, predictions)
        for name, predictions in predictions_by_model.items()
    }

    result = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "sizes": {
            "train_sessions": len(train_sessions),
            "eval_sessions": len(eval_sessions),
        },
        "eval_label_distribution": {
            "tool_vs_memory": dict(Counter(ACTION_NAMES[d.select_action] for d in eval_decisions)),
            "memory_write": dict(Counter(WRITE_NAMES[d.write_action] for d in eval_decisions)),
            "memory_compress": dict(Counter(COMPRESS_NAMES[d.compress_action] for d in eval_decisions)),
            "memory_forget": dict(Counter(FORGET_NAMES[d.forget_action] for d in eval_decisions)),
            "drift": dict(Counter("drift" if session.is_drift else "stable" for session in eval_sessions)),
        },
        "metrics": metrics_by_model,
        "examples": prediction_examples(eval_sessions, eval_decisions, predictions_by_model, count=min(3, config.eval_count)),
    }
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-count", type=int, default=20)
    parser.add_argument("--eval-offset", type=int, default=500)
    parser.add_argument("--train-users", type=int, default=500)
    parser.add_argument("--minimax-max-tokens", type=int, default=1024)
    parser.add_argument("--minimax-max-retries", type=int, default=5)
    parser.add_argument("--minimax-model", default=None)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()
    return EvalConfig(
        output_json=args.output_json,
        train_users=args.train_users,
        eval_count=args.eval_count,
        eval_offset=args.eval_offset,
        minimax_model=args.minimax_model,
        minimax_max_tokens=args.minimax_max_tokens,
        minimax_max_retries=args.minimax_max_retries,
    )


def main() -> None:
    config = parse_args()
    result = run_eval(config)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
