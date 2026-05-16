#!/usr/bin/env python3
"""Evaluate free-generation memory-policy students on unseen recommendation sessions."""

from __future__ import annotations

import argparse
import gc
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_memory_policy_distillation_probe import (
    ACTION_NAMES,
    COMPRESS_NAMES,
    FORGET_NAMES,
    WRITE_NAMES,
    Session,
    TeacherDecision,
    create_llm_client,
    load_reviews,
    teacher_payload,
    teacher_policy,
)
from supervised_memory_policy_training import (
    LLAMA_FACTORY_DIR,
    TrainingConfig,
    featurize_policy,
    fit_action_model,
    fit_candidate_ranker,
    load_topic_lookup,
    predict_recommendation,
    session_payload,
    sft_input,
    sft_instruction,
    sft_system_prompt,
)


PROBE_DIR = Path(__file__).resolve().parent
DEFAULT_ADAPTER_PATH = (
    LLAMA_FACTORY_DIR / "saves" / "memory_policy_trajectory" / "qwen2.5-0.5b" / "lora" / "sft"
)
DEFAULT_OUTPUT_JSON = PROBE_DIR / "supervised_memory_policy" / "free_generation_eval_result.json"


@dataclass(frozen=True)
class EvalConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_path: Path = DEFAULT_ADAPTER_PATH
    output_json: Path = DEFAULT_OUTPUT_JSON
    train_users: int = 500
    eval_count: int = 20
    eval_offset: int = 500
    sessions_per_user: int = 4
    max_candidate_items: int = 8
    max_new_tokens: int = 768
    minimax_provider: str = "minimax"
    minimax_model: str | None = None
    minimax_max_tokens: int = 1024
    minimax_max_retries: int = 5


@dataclass(frozen=True)
class ParsedPrediction:
    policy: tuple[int, int, int, int] | None
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
        )
        eligible_index += 1
        if len(candidate_items) < 2:
            continue

        sessions.append(
            Session(
                user_id=offset + len(sessions),
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
        required = ("tool_vs_memory", "memory_write", "memory_compress", "memory_forget", "recommend", "trajectory")
        missing = [key for key in required if key not in parsed]
        if missing:
            raise ValueError(f"missing required keys: {missing}")

        tool_vs_memory = str(parsed["tool_vs_memory"])
        memory_write = str(parsed["memory_write"])
        memory_compress = str(parsed["memory_compress"])
        memory_forget = str(parsed["memory_forget"])
        recommendation = str(parsed["recommend"])

        if tool_vs_memory not in ACTION_NAMES:
            raise ValueError(f"invalid tool_vs_memory: {tool_vs_memory}")
        if memory_write not in WRITE_NAMES:
            raise ValueError(f"invalid memory_write: {memory_write}")
        if memory_compress not in COMPRESS_NAMES:
            raise ValueError(f"invalid memory_compress: {memory_compress}")
        if memory_forget not in FORGET_NAMES:
            raise ValueError(f"invalid memory_forget: {memory_forget}")
        if recommendation not in session.candidate_items:
            raise ValueError(f"recommendation is not in candidate_items: {recommendation}")
        trajectory = parsed["trajectory"]
        if not isinstance(trajectory, dict):
            raise ValueError("trajectory must be a JSON object")
        trajectory_required = ("memory_plan", "memory_read", "tool_calls", "memory_write", "memory_compress", "memory_forget", "recommend")
        trajectory_missing = [key for key in trajectory_required if key not in trajectory]
        if trajectory_missing:
            raise ValueError(f"trajectory missing required keys: {trajectory_missing}")
        if not isinstance(trajectory["memory_plan"], dict):
            raise ValueError("trajectory.memory_plan must be a JSON object")
        if not isinstance(trajectory["memory_read"], dict):
            raise ValueError("trajectory.memory_read must be a JSON object")
        if not isinstance(trajectory["tool_calls"], list):
            raise ValueError("trajectory.tool_calls must be a JSON array")
        if tool_vs_memory == "tool" and not trajectory["tool_calls"]:
            raise ValueError("tool_vs_memory=tool requires at least one executed tool call")
        for index, tool_call in enumerate(trajectory["tool_calls"]):
            if not isinstance(tool_call, dict):
                raise ValueError(f"trajectory.tool_calls[{index}] must be a JSON object")
            required_tool_call_keys = ("agent", "tool_name", "tool_args", "tool_result", "observation")
            missing_tool_call_keys = [key for key in required_tool_call_keys if key not in tool_call]
            if missing_tool_call_keys:
                raise ValueError(
                    f"trajectory.tool_calls[{index}] missing required keys: {missing_tool_call_keys}"
                )
        if not isinstance(trajectory["memory_write"], dict):
            raise ValueError("trajectory.memory_write must be a JSON object")
        if not isinstance(trajectory["memory_compress"], dict):
            raise ValueError("trajectory.memory_compress must be a JSON object")
        if not isinstance(trajectory["memory_forget"], dict):
            raise ValueError("trajectory.memory_forget must be a JSON object")
        if not isinstance(trajectory["recommend"], str):
            raise ValueError("trajectory.recommend must be a string")

        return ParsedPrediction(
            policy=(
                ACTION_NAMES.index(tool_vs_memory),
                WRITE_NAMES.index(memory_write),
                COMPRESS_NAMES.index(memory_compress),
                FORGET_NAMES.index(memory_forget),
            ),
            recommendation=recommendation,
            parsed_json=parsed,
            raw_text=text,
            parse_error=None,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        return ParsedPrediction(
            policy=None,
            recommendation=None,
            parsed_json=None,
            raw_text=text,
            parse_error=str(exc),
        )


def build_messages(session: Session) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": sft_system_prompt()},
        {"role": "user", "content": f"{sft_instruction()}\n{sft_input(session)}"},
    ]


def build_minimax_prompt(session: Session) -> str:
    return "\n\n".join(
        [
            f"System:\n{sft_system_prompt()}",
            f"User:\n{sft_instruction()}\n{sft_input(session)}",
            "Assistant:",
        ]
    )


def generate_with_qwen(
    model,
    tokenizer,
    session: Session,
    max_new_tokens: int,
) -> ParsedPrediction:
    messages = build_messages(session)
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0, input_ids.shape[-1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return parse_generated_policy(text, session)


def load_qwen_base(model_name_or_path: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen free-generation evaluation")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")
    model.eval()
    return tokenizer, model


def release_model(model) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def teacher_policy_tuple(decision: TeacherDecision) -> tuple[int, int, int, int]:
    return (
        decision.select_action,
        decision.write_action,
        decision.compress_action,
        decision.forget_action,
    )


def summarize_predictions(
    sessions: list[Session],
    decisions: list[TeacherDecision],
    predictions: list[ParsedPrediction],
) -> dict[str, Any]:
    if len(sessions) != len(decisions) or len(sessions) != len(predictions):
        raise ValueError("sessions, decisions, and predictions length mismatch")

    labels = [teacher_policy_tuple(decision) for decision in decisions]
    metrics: dict[str, Any] = {
        "json_parse_success_rate": sum(pred.parse_error is None for pred in predictions) / len(predictions),
        "recommendation_top1": sum(
            pred.recommendation == session.target_item for session, pred in zip(sessions, predictions)
        )
        / len(predictions),
        "recommendation_teacher_match": sum(
            pred.recommendation == decision.recommendation for decision, pred in zip(decisions, predictions)
        )
        / len(predictions),
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
    ranker = fit_candidate_ranker(train_sessions, config)
    predictions: list[ParsedPrediction] = []
    for session in eval_sessions:
        features = featurize_policy(session, config.feature_hash_size)
        policy = (
            action_models["tool_vs_memory"].predict(features),
            action_models["memory_write"].predict(features),
            action_models["memory_compress"].predict(features),
            action_models["memory_forget"].predict(features),
        )
        recommendation = predict_recommendation(ranker, session, config)
        parsed_json = {
            "tool_vs_memory": ACTION_NAMES[policy[0]],
            "memory_write": WRITE_NAMES[policy[1]],
            "memory_compress": COMPRESS_NAMES[policy[2]],
            "memory_forget": FORGET_NAMES[policy[3]],
            "recommend": recommendation,
        }
        predictions.append(
            ParsedPrediction(
                policy=policy,
                recommendation=recommendation,
                parsed_json=parsed_json,
                raw_text=json.dumps(parsed_json, ensure_ascii=False, sort_keys=True),
                parse_error=None,
            )
        )
    return predictions


def minimax_predictions(
    sessions: list[Session],
    provider: str,
    model: str | None,
    max_tokens: int,
    max_retries: int,
) -> list[ParsedPrediction]:
    client = create_llm_client(provider, model)
    predictions: list[ParsedPrediction] = []
    for session in sessions:
        _, text = client.call_with_thinking(
            prompt=build_minimax_prompt(session),
            max_tokens=max_tokens,
            temperature=0.0,
            max_retries=max_retries,
        )
        predictions.append(parse_generated_policy(text, session))
    return predictions


def qwen_predictions(
    sessions: list[Session],
    model_name_or_path: str,
    adapter_path: Path | None,
    max_new_tokens: int,
) -> list[ParsedPrediction]:
    tokenizer, model = load_qwen_base(model_name_or_path)
    if adapter_path is not None:
        if not adapter_path.is_dir():
            raise FileNotFoundError(f"adapter path does not exist: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
    predictions = [generate_with_qwen(model, tokenizer, session, max_new_tokens) for session in sessions]
    release_model(model)
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
                "recommendation": pred.recommendation,
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
    train_sessions = build_supervised_sessions_with_offset(train_config, offset=0, count=config.train_users)
    train_decisions = [teacher_policy(session) for session in train_sessions]
    eval_sessions = build_supervised_sessions_with_offset(
        train_config,
        offset=config.eval_offset,
        count=config.eval_count,
    )
    eval_decisions = [teacher_policy(session) for session in eval_sessions]

    predictions_by_model = {
        "qwen_lora_sft": qwen_predictions(
            eval_sessions,
            config.model_name_or_path,
            config.adapter_path,
            config.max_new_tokens,
        ),
        "qwen_base": qwen_predictions(
            eval_sessions,
            config.model_name_or_path,
            None,
            config.max_new_tokens,
        ),
        "minimax_prompt_only": minimax_predictions(
            eval_sessions,
            config.minimax_provider,
            config.minimax_model,
            config.minimax_max_tokens,
            config.minimax_max_retries,
        ),
        "logistic_baseline": logistic_predictions(train_sessions, train_decisions, eval_sessions, train_config),
    }
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
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--minimax-max-tokens", type=int, default=1024)
    parser.add_argument("--minimax-max-retries", type=int, default=5)
    parser.add_argument("--minimax-provider", default="minimax")
    parser.add_argument("--minimax-model", default=None)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()
    return EvalConfig(
        model_name_or_path=args.model_name_or_path,
        adapter_path=args.adapter_path,
        output_json=args.output_json,
        train_users=args.train_users,
        eval_count=args.eval_count,
        eval_offset=args.eval_offset,
        max_new_tokens=args.max_new_tokens,
        minimax_provider=args.minimax_provider,
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
