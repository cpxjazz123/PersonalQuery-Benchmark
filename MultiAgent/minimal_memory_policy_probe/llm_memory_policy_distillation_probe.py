#!/usr/bin/env python3
"""LLM probe for memory-policy distillation in recommendation.

This script keeps the synthetic recommendation environment from the minimal
probe, but replaces the linear student with a real LLM from ../llm_client.py.
The "distilled" LLM receives teacher trajectories containing memory read,
write, compress, forget, and tool-vs-memory decisions. The context-only LLM
receives the same task state without teacher memory-policy trajectories.
"""

from __future__ import annotations

import importlib.util
import json
import gzip
import re
import sys
from dataclasses import asdict, dataclass
from collections import defaultdict
from pathlib import Path
from typing import Any

PROBE_DIR = Path(__file__).resolve().parent
MULTIAGENT_DIR = PROBE_DIR.parent
OUTPUT_JSON = PROBE_DIR / "latest_llm_result.json"
if str(PROBE_DIR) not in sys.path:
    sys.path.insert(0, str(PROBE_DIR))

from memory_policy_distillation_probe import (  # noqa: E402
    ACTION_NAMES,
    COMPRESS_NAMES,
    FORGET_NAMES,
    build_tool_call_trace,
    topic_from_item,
    WRITE_NAMES,
    Session,
    TeacherDecision,
)


@dataclass(frozen=True)
class ExperimentConfig:
    provider: str = "minimax"
    model: str | None = None
    review_file: Path = Path("/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2023/Baby_Products.jsonl.gz")
    meta_file: Path = Path("/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2023/meta_Baby_Products.jsonl.gz")
    users: int = 8
    sessions_per_user: int = 4
    seed: int = 11
    train_ratio: float = 0.7
    eval_count: int = 2
    teacher_examples: int = 4
    temperature: float = 0.0
    max_tokens: int = 2048
    max_retries: int = 5
    min_distilled_top1: float = 0.0
    min_gain: float = -1.0
    min_policy_accuracy: float = 0.0
    example_count: int = 2
    output_json: Path = OUTPUT_JSON


def load_llm_client_class(provider: str):
    module_path = MULTIAGENT_DIR / "llm_client.py"
    spec = importlib.util.spec_from_file_location("multiagent_llm_client", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load llm_client module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    providers = {
        "minimax": "MiniMaxAnthropicClient",
        "minimax_io": "MiniMaxIOAnthropicClient",
    }
    class_name = providers.get(provider)
    if class_name is None:
        raise ValueError(f"Unsupported provider: {provider}")
    if not hasattr(module, class_name):
        raise RuntimeError(f"llm_client.py does not define {class_name}")
    return getattr(module, class_name)


def create_llm_client(provider: str, model: str | None):
    client_cls = load_llm_client_class(provider)
    if model is None:
        return client_cls()
    return client_cls(model=model)


def session_payload(session: Session) -> dict[str, Any]:
    return {
        "user_id": session.user_id,
        "history_topic_counts": session.history_topic_counts,
        "current_topic_signal": session.current_topic,
        "candidate_items": session.candidate_items,
        "external_memory": {
            "dominant_topic": session.memory_topic,
            "strength": session.memory_strength,
        },
    }


def open_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def load_meta_lookup(meta_file: Path) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    with open_maybe_gzip(meta_file) as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            asin = record.get("asin") or record.get("parent_asin")
            if asin is not None:
                lookup[str(asin)] = record
    if not lookup:
        raise ValueError(f"No meta records loaded from {meta_file}")
    return lookup


def meta_topic(meta: dict[str, Any]) -> str:
    categories = meta.get("categories")
    if isinstance(categories, list) and categories:
        first = categories[0]
        if isinstance(first, list) and first:
            return str(first[0])
        if isinstance(first, str):
            return str(first)
    title = meta.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    raise ValueError("Meta record is missing both categories and title")


def load_reviews(review_file: Path) -> list[dict[str, Any]]:
    reviews: list[dict[str, Any]] = []
    with open_maybe_gzip(review_file) as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("user_id") is None or record.get("asin") is None:
                continue
            reviews.append(record)
    if not reviews:
        raise ValueError(f"No reviews loaded from {review_file}")
    return reviews


def teacher_policy(session: Session) -> TeacherDecision:
    memory_matches = session.memory_topic == session.current_topic
    history_total = sum(session.history_topic_counts.values())
    topic_count = session.history_topic_counts.get(session.current_topic)
    if topic_count is None:
        topic_count = 0
    recent_is_weak = topic_count <= 1

    use_tool = (not memory_matches) or recent_is_weak
    select_action = 1 if use_tool else 0
    write_action = 1 if session.is_drift or topic_count <= 2 else 0
    compress_action = 1 if history_total >= 10 and session.memory_strength >= 6 else 0
    forget_action = 1 if session.is_drift and session.memory_strength >= 6 and not memory_matches else 0

    recommendation = session.target_item
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
        "tool_calls": tool_calls,
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


def build_real_sessions(config: ExperimentConfig) -> list[Session]:
    meta_lookup = load_meta_lookup(config.meta_file)
    reviews = load_reviews(config.review_file)
    reviews_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for review in reviews:
        reviews_by_user[str(review["user_id"])].append(review)

    eligible_users: list[tuple[str, list[dict[str, Any]]]] = []
    for user_id, user_reviews in reviews_by_user.items():
        usable_reviews = [review for review in user_reviews if str(review["asin"]) in meta_lookup]
        if len(usable_reviews) < config.sessions_per_user + 1:
            continue
        usable_reviews.sort(key=lambda item: item.get("timestamp", 0))
        eligible_users.append((user_id, usable_reviews))

    if not eligible_users:
        raise ValueError("No eligible users found in real review data")

    sessions: list[Session] = []
    for user_index, (user_id, user_reviews) in enumerate(eligible_users[: config.users]):
        history_slice = user_reviews[: config.sessions_per_user]
        target_review = user_reviews[config.sessions_per_user]
        history_topic_counts: dict[str, int] = defaultdict(int)
        for review in history_slice:
            meta = meta_lookup[str(review["asin"])]
            topic = meta_topic(meta)
            history_topic_counts[topic] += 1

        if not history_topic_counts:
            raise ValueError(f"No usable history topics for user {user_id}")

        target_meta = meta_lookup[str(target_review["asin"])]
        current_topic = meta_topic(target_meta)

        memory_topic = max(history_topic_counts, key=history_topic_counts.get)
        memory_strength = history_topic_counts[memory_topic]
        candidate_items = tuple(
            dict.fromkeys(
                [str(target_review["asin"])]
                + [str(review["asin"]) for review in history_slice[:5]]
                + [str(review["asin"]) for review in user_reviews[config.sessions_per_user + 1 : config.sessions_per_user + 6]]
            )
        )
        if len(candidate_items) < 2:
            continue

        sessions.append(
            Session(
                user_id=user_index,
                history_topic_counts=dict(history_topic_counts),
                current_topic=current_topic,
                drift_topic=current_topic if current_topic != memory_topic else "",
                candidate_items=candidate_items,
                target_item=str(target_review["asin"]),
                target_topic=current_topic,
                is_drift=current_topic != memory_topic,
                memory_topic=memory_topic,
                memory_strength=memory_strength,
            )
        )
    if not sessions:
        raise ValueError("No sessions built from real recommendation data")
    return sessions


def teacher_payload(decision: TeacherDecision) -> dict[str, Any]:
    return {
        "tool_vs_memory": ACTION_NAMES[decision.select_action],
        "memory_write": WRITE_NAMES[decision.write_action],
        "memory_compress": COMPRESS_NAMES[decision.compress_action],
        "memory_forget": FORGET_NAMES[decision.forget_action],
        "recommend": decision.recommendation,
        "trajectory": decision.trajectory,
    }


def build_distilled_cached_prompt_parts(
    examples: list[tuple[Session, TeacherDecision]], session: Session
) -> tuple[str, str]:
    example_blocks = []
    for idx, (example_session, decision) in enumerate(examples, start=1):
        example_blocks.append(
            json.dumps(
                {
                    "example_id": idx,
                    "input": session_payload(example_session),
                    "teacher_memory_policy_trajectory": teacher_payload(decision),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    system_base = f"""
You are a recommendation model distilled from a multi-agent teacher.

Task:
Given a user state, output a JSON object with exactly these keys:
tool_vs_memory, memory_write, memory_compress, memory_forget, recommend.

Valid values:
tool_vs_memory: "memory" or "tool"
memory_write: "skip_write" or "write"
memory_compress: "skip_compress" or "compress"
memory_forget: "keep" or "forget"
recommend: one item copied exactly from candidate_items

Learn the memory decision policy from these teacher trajectories.
Do not explain. Return JSON only.
Return compact one-line JSON. Do not use markdown fences.

Teacher examples:
{chr(10).join(example_blocks)}
""".strip()

    user_content = f"""
Target input:
{json.dumps(session_payload(session), ensure_ascii=False, indent=2)}
""".strip()
    return system_base, user_content


def build_context_only_cached_prompt_parts(session: Session) -> tuple[str, str]:
    system_base = """
You are a recommendation model with access to external memory as context.

Task:
Given a user state, output a JSON object with exactly these keys:
tool_vs_memory, memory_write, memory_compress, memory_forget, recommend.

Valid values:
tool_vs_memory: "memory" or "tool"
memory_write: "skip_write" or "write"
memory_compress: "skip_compress" or "compress"
memory_forget: "keep" or "forget"
recommend: one item copied exactly from candidate_items

Use the provided external memory as context. Do not infer hidden labels.
Do not explain. Return JSON only.
Return compact one-line JSON. Do not use markdown fences.
""".strip()

    user_content = f"""
Target input:
{json.dumps(session_payload(session), ensure_ascii=False, indent=2)}
""".strip()
    return system_base, user_content


def extract_json_object(text: str) -> dict[str, Any]:
    if not text.strip():
        raise ValueError("LLM returned an empty response")
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    raw = fenced.group(1) if fenced else text
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM response does not contain a JSON object: {text[:500]}")
    return json.loads(raw[start : end + 1])


def parse_policy(response: dict[str, Any], session: Session) -> tuple[int, int, int, int, str]:
    required = ("tool_vs_memory", "memory_write", "memory_compress", "memory_forget", "recommend")
    missing = [key for key in required if key not in response]
    if missing:
        raise ValueError(f"LLM JSON missing required keys: {missing}")

    tool_vs_memory = str(response["tool_vs_memory"])
    memory_write = str(response["memory_write"])
    memory_compress = str(response["memory_compress"])
    memory_forget = str(response["memory_forget"])
    recommendation = str(response["recommend"])

    if tool_vs_memory not in ACTION_NAMES:
        raise ValueError(f"Invalid tool_vs_memory: {tool_vs_memory}")
    if memory_write not in WRITE_NAMES:
        raise ValueError(f"Invalid memory_write: {memory_write}")
    if memory_compress not in COMPRESS_NAMES:
        raise ValueError(f"Invalid memory_compress: {memory_compress}")
    if memory_forget not in FORGET_NAMES:
        raise ValueError(f"Invalid memory_forget: {memory_forget}")
    if recommendation not in session.candidate_items:
        raise ValueError(f"Recommendation is not in candidate_items: {recommendation}")

    return (
        ACTION_NAMES.index(tool_vs_memory),
        WRITE_NAMES.index(memory_write),
        COMPRESS_NAMES.index(memory_compress),
        FORGET_NAMES.index(memory_forget),
        recommendation,
    )


def call_policy_with_cache(
    client,
    system_base: str,
    user_content: str,
    session: Session,
    max_tokens: int,
    temperature: float,
    max_retries: int,
):
    if not hasattr(client, "call_with_cache"):
        raise RuntimeError("Selected LLM client does not support call_with_cache")
    text, cache_info = client.call_with_cache(
        system_base=system_base,
        user_content=user_content,
        max_tokens=max_tokens,
        temperature=temperature,
        max_retries=max_retries,
    )
    parsed = extract_json_object(text)
    policy = parse_policy(parsed, session)
    return policy, parsed, text, cache_info


def select_balanced_examples(sessions: list[Session], decisions: list[TeacherDecision], count: int):
    if count <= 0:
        raise ValueError("example count must be positive")
    drift_rows = [(s, d) for s, d in zip(sessions, decisions) if s.is_drift]
    stable_rows = [(s, d) for s, d in zip(sessions, decisions) if not s.is_drift]
    examples = []
    half = count // 2
    if stable_rows:
        examples.extend(stable_rows[: max(1, half)])
    if len(examples) < count and drift_rows:
        examples.extend(drift_rows[: max(1, count - len(examples))])
    if not examples:
        raise ValueError("training sessions must include at least one example")
    return examples[:count]


def policy_tuple(policy_with_recommendation: tuple[int, int, int, int, str]) -> tuple[int, int, int, int]:
    return policy_with_recommendation[:4]


def recommendation_accuracy(sessions: list[Session], predicted_items: list[str]) -> float:
    if len(sessions) != len(predicted_items):
        raise ValueError("sessions and predicted_items length mismatch")
    return sum(1 for s, item in zip(sessions, predicted_items) if item == s.target_item) / len(sessions)


def grouped_recommendation_accuracy_by_drift(
    sessions: list[Session],
    predicted_items: list[str],
) -> dict[str, float]:
    if len(sessions) != len(predicted_items):
        raise ValueError("sessions and predicted_items length mismatch")
    groups: dict[str, list[tuple[Session, str]]] = defaultdict(list)
    for session, predicted_item in zip(sessions, predicted_items):
        groups["drift" if session.is_drift else "stable"].append((session, predicted_item))
    result: dict[str, float] = {}
    for name, rows in groups.items():
        correct = sum(1 for session, predicted_item in rows if predicted_item == session.target_item)
        result[name] = correct / len(rows)
    return result


def action_accuracy(
    predicted: list[tuple[int, int, int, int]],
    teacher: list[TeacherDecision],
    action_index: int,
) -> float:
    labels = [
        (d.select_action, d.write_action, d.compress_action, d.forget_action)
        for d in teacher
    ]
    return sum(1 for p, y in zip(predicted, labels) if p[action_index] == y[action_index]) / len(labels)


def summarize_cache_infos(cache_infos: list[dict[str, Any]]) -> dict[str, int]:
    keys = (
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "input_tokens",
        "output_tokens",
    )
    summary = {key: 0 for key in keys}
    for info in cache_infos:
        for key in keys:
            value = info.get(key, 0)
            if not isinstance(value, int):
                raise ValueError(f"cache info {key} must be int, got {type(value).__name__}")
            summary[key] += value
    return summary


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    sessions = build_real_sessions(config)
    split = int(len(sessions) * config.train_ratio)
    if split <= 0 or split >= len(sessions):
        raise ValueError("train_ratio creates an invalid split")
    train_sessions = sessions[:split]
    test_sessions = sessions[split : split + config.eval_count]
    if len(test_sessions) != config.eval_count:
        raise ValueError("eval_count exceeds available test sessions")

    train_decisions = [teacher_policy(session) for session in train_sessions]
    test_decisions = [teacher_policy(session) for session in test_sessions]
    examples = select_balanced_examples(train_sessions, train_decisions, config.teacher_examples)
    client = create_llm_client(config.provider, config.model)

    distilled_policies = []
    distilled_items = []
    context_only_policies = []
    context_only_items = []
    raw_examples = []
    distilled_cache_infos = []
    context_only_cache_infos = []

    for idx, session in enumerate(test_sessions):
        distilled_system_base, distilled_user_content = build_distilled_cached_prompt_parts(examples, session)
        distilled_policy, distilled_json, distilled_raw, distilled_cache_info = call_policy_with_cache(
            client,
            distilled_system_base,
            distilled_user_content,
            session,
            config.max_tokens,
            config.temperature,
            config.max_retries,
        )
        context_system_base, context_user_content = build_context_only_cached_prompt_parts(session)
        context_policy, context_json, context_raw, context_cache_info = call_policy_with_cache(
            client,
            context_system_base,
            context_user_content,
            session,
            config.max_tokens,
            config.temperature,
            config.max_retries,
        )

        distilled_policies.append(policy_tuple(distilled_policy))
        distilled_items.append(distilled_policy[4])
        context_only_policies.append(policy_tuple(context_policy))
        context_only_items.append(context_policy[4])
        distilled_cache_infos.append(distilled_cache_info)
        context_only_cache_infos.append(context_cache_info)

        if idx < config.example_count:
            raw_examples.append(
                {
                    "session": session_payload(session),
                    "target_item": session.target_item,
                    "teacher": teacher_payload(test_decisions[idx]),
                    "distilled_llm_json": distilled_json,
                    "context_only_llm_json": context_json,
                    "distilled_cache_info": distilled_cache_info,
                    "context_only_cache_info": context_cache_info,
                    "distilled_raw": distilled_raw[:1000],
                    "context_only_raw": context_raw[:1000],
                }
            )

    teacher_policies = [
        (d.select_action, d.write_action, d.compress_action, d.forget_action)
        for d in test_decisions
    ]
    teacher_items = [d.recommendation for d in test_decisions]

    metrics = {
        "teacher_top1": recommendation_accuracy(test_sessions, teacher_items),
        "distilled_llm_top1": recommendation_accuracy(test_sessions, distilled_items),
        "context_only_llm_top1": recommendation_accuracy(test_sessions, context_only_items),
        "distilled_gain_over_context_only": (
            recommendation_accuracy(test_sessions, distilled_items)
            - recommendation_accuracy(test_sessions, context_only_items)
        ),
        "distilled_tool_vs_memory_accuracy": action_accuracy(distilled_policies, test_decisions, 0),
        "distilled_write_accuracy": action_accuracy(distilled_policies, test_decisions, 1),
        "distilled_compress_accuracy": action_accuracy(distilled_policies, test_decisions, 2),
        "distilled_forget_accuracy": action_accuracy(distilled_policies, test_decisions, 3),
        "context_only_tool_vs_memory_accuracy": action_accuracy(context_only_policies, test_decisions, 0),
        "context_only_write_accuracy": action_accuracy(context_only_policies, test_decisions, 1),
        "context_only_compress_accuracy": action_accuracy(context_only_policies, test_decisions, 2),
        "context_only_forget_accuracy": action_accuracy(context_only_policies, test_decisions, 3),
    }

    if metrics["distilled_llm_top1"] < config.min_distilled_top1:
        raise AssertionError(
            f"distilled_llm_top1={metrics['distilled_llm_top1']:.4f} below {config.min_distilled_top1:.4f}"
        )
    if metrics["distilled_gain_over_context_only"] < config.min_gain:
        raise AssertionError(
            "distilled LLM gain over context-only LLM is too small: "
            f"{metrics['distilled_gain_over_context_only']:.4f} < {config.min_gain:.4f}"
        )
    if metrics["distilled_tool_vs_memory_accuracy"] < config.min_policy_accuracy:
        raise AssertionError(
            "distilled tool-vs-memory accuracy is too low: "
            f"{metrics['distilled_tool_vs_memory_accuracy']:.4f} < {config.min_policy_accuracy:.4f}"
        )

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }
    return {
        "config": config,
        "sizes": {"train": len(train_sessions), "eval": len(test_sessions)},
        "metrics": metrics,
        "distilled_by_group": grouped_recommendation_accuracy_by_drift(test_sessions, distilled_items),
        "context_only_by_group": grouped_recommendation_accuracy_by_drift(test_sessions, context_only_items),
        "cache_summary": {
            "distilled": summarize_cache_infos(distilled_cache_infos),
            "context_only": summarize_cache_infos(context_only_cache_infos),
        },
        "teacher_policy_counts": {
            "tool": sum(d.select_action for d in train_decisions + test_decisions),
            "write": sum(d.write_action for d in train_decisions + test_decisions),
            "compress": sum(d.compress_action for d in train_decisions + test_decisions),
            "forget": sum(d.forget_action for d in train_decisions + test_decisions),
        },
        "examples": raw_examples,
        "claim": (
            "A real LLM using distilled teacher memory-policy trajectories should learn when to "
            "read/write/compress/forget and when to use tools, outperforming a prompt that treats "
            "memory only as static context."
        ),
    }


def main() -> None:
    config = ExperimentConfig()
    result = run_experiment(config)
    output = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)
    print(output)
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
