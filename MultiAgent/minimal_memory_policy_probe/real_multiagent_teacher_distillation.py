#!/usr/bin/env python3
"""Generate real multi-agent LLM teacher trajectories for memory-policy SFT."""

from __future__ import annotations

import argparse
import json
import signal
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from llm_memory_policy_distillation_probe import (
    ACTION_NAMES,
    COMPRESS_NAMES,
    FORGET_NAMES,
    PREFERENCE_TYPES,
    preference_type_for_session,
    preference_vector_for_type,
    rank_candidates,
    ranking_metrics,
    WRITE_NAMES,
    Session,
    TeacherDecision,
    build_candidate_alignment,
    create_llm_client,
    teacher_payload,
)
from supervised_memory_policy_training import (
    LLAMA_FACTORY_DIR,
    export_teacher_trajectories,
    load_topic_lookup,
    open_gzip_text,
    sft_input,
    sft_instruction,
    sft_system_prompt,
    session_payload,
)


PROBE_DIR = Path(__file__).resolve().parent
MULTIAGENT_DIR = PROBE_DIR.parent
OUTPUT_DIR = PROBE_DIR / "real_multiagent_teacher"
TRAIN_JSONL = OUTPUT_DIR / "teacher_train_trajectories.jsonl"
EVAL_JSONL = OUTPUT_DIR / "teacher_eval_trajectories.jsonl"
RESULT_JSON = OUTPUT_DIR / "teacher_generation_result.json"
LLAMA_FACTORY_SFT_JSON = LLAMA_FACTORY_DIR / "data" / "real_multiagent_memory_policy_sft.json"


@dataclass(frozen=True)
class RealTeacherConfig:
    review_file: Path = Path("/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2023/Baby_Products.jsonl.gz")
    meta_file: Path = Path("/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2023/meta_Baby_Products.jsonl.gz")
    train_count: int = 12
    eval_count: int = 4
    eval_offset: int = 500
    candidate_multiplier: int = 8
    require_teacher_target_hit: bool = True
    sessions_per_user: int = 4
    max_candidate_items: int = 8
    teacher_model: str | None = None
    max_tokens: int = 32768
    temperature: float = 0.0
    max_retries: int = 5
    call_timeout_seconds: int = 300
    output_dir: Path = OUTPUT_DIR
    train_jsonl: Path = TRAIN_JSONL
    eval_jsonl: Path = EVAL_JSONL
    result_json: Path = RESULT_JSON
    llamafactory_sft_json: Path = LLAMA_FACTORY_SFT_JSON


class AgentCallTimeoutError(TimeoutError):
    pass


def _agent_call_timeout_handler(signum, frame):
    raise AgentCallTimeoutError("real multi-agent teacher call timed out")


def create_real_teacher_client(model: str | None):
    return create_llm_client(model)


def extract_json_object(text: str) -> dict[str, Any]:
    if not text.strip():
        raise ValueError("LLM response is empty")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM response does not contain a JSON object: {text[:500]}")
    return json.loads(text[start : end + 1])


def call_json_agent(
    client,
    agent_name: str,
    prompt: str,
    config: RealTeacherConfig,
) -> dict[str, Any]:
    print(f"[real-teacher] calling {agent_name}", flush=True)
    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _agent_call_timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, config.call_timeout_seconds)
    try:
        text, _ = client.call_with_cache(
            system_base="You are a precise JSON-only assistant.",
            user_content=prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            max_retries=config.max_retries,
            stream=True,
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
    parsed = extract_json_object(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"{agent_name} output must be a JSON object")
    print(f"[real-teacher] {agent_name} returned valid JSON", flush=True)
    return parsed


def require_keys(payload: dict[str, Any], keys: tuple[str, ...], context: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(f"{context} missing required keys: {missing}")


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def require_mapping(payload: Any, context: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be a JSON object")
    return payload


def compact_agent_text(payload: dict[str, Any], keys: tuple[str, ...], context: str) -> str:
    selected = {}
    for key in keys:
        if key not in payload:
            raise ValueError(f"{context} missing required compact key: {key}")
        selected[key] = payload[key]
    return compact_json(selected)


def compact_teacher_payload(decision: TeacherDecision) -> dict[str, Any]:
    trajectory = require_mapping(decision.trajectory, "teacher trajectory")
    profile_agent = require_mapping(trajectory["profile_agent_raw"], "profile_agent_raw trajectory")
    memory_agent = require_mapping(trajectory["memory_agent_raw"], "memory_agent_raw trajectory")
    recommendation_agent = require_mapping(trajectory["recommendation_agent_raw"], "recommendation_agent_raw trajectory")
    critic_agent = require_mapping(trajectory["critic_agent_raw"], "critic_agent_raw trajectory")
    require_keys(profile_agent, ("current_signal", "uncertainty"), "profile_agent trajectory")
    require_keys(memory_agent, ("tool_vs_memory", "memory_write", "memory_compress", "memory_forget"), "memory_agent trajectory")
    require_keys(
        recommendation_agent,
        ("preference_type", "ranking", "ranking_reason", "rejected_candidates"),
        "recommendation_agent trajectory",
    )
    require_keys(
        critic_agent,
        ("tool_vs_memory", "memory_write", "memory_compress", "memory_forget", "preference_type", "ranking"),
        "critic_agent trajectory",
    )
    if not isinstance(decision.preference_type, str) or not decision.preference_type:
        raise ValueError("decision preference_type must be populated")
    if not isinstance(decision.ranking, tuple) or not decision.ranking:
        raise ValueError("decision ranking must be populated")
    return {
        "tool_vs_memory": ACTION_NAMES[decision.select_action],
        "memory_write": WRITE_NAMES[decision.write_action],
        "memory_compress": COMPRESS_NAMES[decision.compress_action],
        "memory_forget": FORGET_NAMES[decision.forget_action],
        "preference_type": decision.preference_type,
        "ranking": list(decision.ranking),
        "trajectory": {
            "profile_agent": (
                f"ProfileAgent read current_signal={profile_agent['current_signal']} "
                f"with uncertainty={profile_agent['uncertainty']}."
            ),
            "memory_agent": (
                f"MemoryAgent set tool_vs_memory={memory_agent['tool_vs_memory']}; "
                f"memory_write={memory_agent['memory_write']}, "
                f"memory_compress={memory_agent['memory_compress']}, "
                f"memory_forget={memory_agent['memory_forget']}."
            ),
            "recommendation_agent": (
                f"RecommendationAgent selected preference_type={recommendation_agent['preference_type']} "
                f"with top ranking item {recommendation_agent['ranking'][0]}."
            ),
            "critic_agent": (
                f"CriticAgent validated tool_vs_memory={critic_agent['tool_vs_memory']}, "
                f"memory_write={critic_agent['memory_write']}, "
                f"memory_compress={critic_agent['memory_compress']}, "
                f"memory_forget={critic_agent['memory_forget']}, preference_type={critic_agent['preference_type']}, "
                f"and ranking top item {critic_agent['ranking'][0]}."
            ),
        },
    }


def real_sft_record(session: Session, decision: TeacherDecision) -> dict[str, str]:
    return {
        "instruction": sft_instruction(),
        "input": sft_input(session),
        "output": json.dumps(compact_teacher_payload(decision), ensure_ascii=False, indent=2, sort_keys=True),
        "system": sft_system_prompt(),
    }


def export_real_llamafactory_sft_dataset(sessions: list[Session], decisions: list[TeacherDecision], path: Path) -> None:
    if len(sessions) != len(decisions):
        raise ValueError("sessions and decisions length mismatch")
    records = [real_sft_record(session, decision) for session, decision in zip(sessions, decisions)]
    if not records:
        raise ValueError("real SFT records cannot be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_profile_prompt(session: Session) -> str:
    return f"""
You are ProfileAgent in a multi-agent recommendation teacher team.

Input user state:
{json.dumps(session_payload(session), ensure_ascii=False, indent=2, sort_keys=True)}

Return JSON only with exactly these keys:
stable_preferences, current_signal, memory_state, candidate_notes, uncertainty.

Rules:
- Do not choose the final recommendation.
- Do not use markdown fences.
- Every value must be concise.
""".strip()


def build_memory_prompt(session: Session, profile: dict[str, Any]) -> str:
    return f"""
You are MemoryAgent in a multi-agent recommendation teacher team.

Input user state:
{json.dumps(session_payload(session), ensure_ascii=False, indent=2, sort_keys=True)}

ProfileAgent output:
{json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True)}

Return JSON only with exactly these keys:
tool_vs_memory, memory_write, memory_compress, memory_forget, read_key, write_topic, forget_topic.

Valid values:
tool_vs_memory: {list(ACTION_NAMES)}
memory_write: {list(WRITE_NAMES)}
memory_compress: {list(COMPRESS_NAMES)}
memory_forget: {list(FORGET_NAMES)}

Rules:
- Use "tool" when current signal differs from memory or memory is too weak.
- Use "write" when the current signal should update user memory.
- Use "compress" only when memory is strong and history is repetitive.
- Use "forget" only when old memory conflicts with the current signal.
- Do not use markdown fences.
""".strip()


def build_recommendation_prompt(
    session: Session,
    profile: dict[str, Any],
    memory_plan: dict[str, Any],
) -> str:
    candidate_items_json = json.dumps(list(session.candidate_items), ensure_ascii=False)
    return f"""
You are RecommendationAgent in a multi-agent recommendation teacher team.

Input user state:
{json.dumps(session_payload(session), ensure_ascii=False, indent=2, sort_keys=True)}

ProfileAgent output:
{json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True)}

MemoryAgent output:
{json.dumps(memory_plan, ensure_ascii=False, indent=2, sort_keys=True)}

Return JSON only with exactly these keys:
preference_type, ranking, ranking_reason, rejected_candidates.

Required JSON shape:
{{
  "preference_type": "aligned | current_priority | memory_priority",
  "ranking": ["candidate string", "candidate string"],
  "ranking_reason": "concise reason",
  "rejected_candidates": ["candidate string", "candidate string"]
}}

Valid candidate_items:
{candidate_items_json}

Rules:
- preference_type must be one of aligned, current_priority, memory_priority.
- ranking must contain every candidate item exactly once.
- ranking must never be a JSON array with duplicates or missing items.
- ranking[0] must reflect the selected preference_type.
- rejected_candidates must be a JSON array.
- Every rejected_candidates item must be copied from candidate_items.
- Do not add product titles, explanations, bullets, or markdown.
- Do not use markdown fences.
""".strip()


def build_critic_prompt(
    session: Session,
    profile: dict[str, Any],
    memory_plan: dict[str, Any],
    recommendation: dict[str, Any],
) -> str:
    candidate_items_json = json.dumps(list(session.candidate_items), ensure_ascii=False)
    return f"""
You are CriticAgent. Validate and finalize a multi-agent teacher trajectory.

Input user state:
{json.dumps(session_payload(session), ensure_ascii=False, indent=2, sort_keys=True)}

ProfileAgent output:
{json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True)}

MemoryAgent output:
{json.dumps(memory_plan, ensure_ascii=False, indent=2, sort_keys=True)}

RecommendationAgent output:
{json.dumps(recommendation, ensure_ascii=False, indent=2, sort_keys=True)}

    Return JSON only with exactly these top-level keys:
    tool_vs_memory, memory_write, memory_compress, memory_forget, preference_type, ranking, profile_agent, memory_agent, recommendation_agent, critic_agent.

    Required JSON shape:
    {{
      "tool_vs_memory": "memory or tool",
      "memory_write": "skip_write or write",
      "memory_compress": "skip_compress or compress",
      "memory_forget": "keep or forget",
      "preference_type": "aligned | current_priority | memory_priority",
      "ranking": ["candidate string", "candidate string"],
      "profile_agent": "concise summary",
      "memory_agent": "concise summary",
      "recommendation_agent": "concise summary",
      "critic_agent": "concise summary"
    }}

    Valid values:
    tool_vs_memory: {list(ACTION_NAMES)}
    memory_write: {list(WRITE_NAMES)}
    memory_compress: {list(COMPRESS_NAMES)}
    memory_forget: {list(FORGET_NAMES)}
    preference_type: {list(PREFERENCE_TYPES)}
    ranking: every candidate item exactly once

Valid candidate_items:
{candidate_items_json}

    Rules:
    - Preserve the MemoryAgent policy decisions unless they are invalid.
    - ranking must never be a JSON array with duplicates or missing items.
- Do not add product titles, explanations, bullets, or markdown.
- Do not explain outside JSON.
- Do not use markdown fences.
""".strip()


def validate_final_teacher_payload(
    payload: dict[str, Any],
    session: Session,
) -> TeacherDecision:
    require_keys(
        payload,
        (
            "tool_vs_memory",
            "memory_write",
            "memory_compress",
            "memory_forget",
            "preference_type",
            "ranking",
            "profile_agent",
            "memory_agent",
            "recommendation_agent",
            "critic_agent",
        ),
        "final teacher payload",
    )

    tool_vs_memory = str(payload["tool_vs_memory"])
    memory_write = str(payload["memory_write"])
    memory_compress = str(payload["memory_compress"])
    memory_forget = str(payload["memory_forget"])
    preference_type = str(payload["preference_type"])
    ranking_raw = payload["ranking"]

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
    profile_agent = payload["profile_agent"]
    memory_agent = payload["memory_agent"]
    recommendation_agent = payload["recommendation_agent"]
    critic_agent = payload["critic_agent"]
    if not isinstance(profile_agent, str):
        raise ValueError("profile_agent must be a string")
    if not isinstance(memory_agent, str):
        raise ValueError("memory_agent must be a string")
    if not isinstance(recommendation_agent, str):
        raise ValueError("recommendation_agent must be a string")
    if not isinstance(critic_agent, str):
        raise ValueError("critic_agent must be a string")

    normalized_trajectory = {
        "profile_agent": profile_agent,
        "memory_agent": memory_agent,
        "recommendation_agent": recommendation_agent,
        "critic_agent": critic_agent,
        "preference_type": preference_type,
        "ranking": list(ranking),
        "multiagent_teacher": True,
    }

    return TeacherDecision(
        select_action=ACTION_NAMES.index(tool_vs_memory),
        write_action=WRITE_NAMES.index(memory_write),
        compress_action=COMPRESS_NAMES.index(memory_compress),
        forget_action=FORGET_NAMES.index(memory_forget),
        recommendation=ranking[0],
        trajectory=normalized_trajectory,
        preference_type=preference_type,
        ranking=ranking,
    )


def run_real_multiagent_teacher(
    client,
    session: Session,
    config: RealTeacherConfig,
    topic_lookup: dict[str, str],
) -> TeacherDecision:
    profile = call_json_agent(client, "ProfileAgent", build_profile_prompt(session), config)
    require_keys(profile, ("stable_preferences", "current_signal", "memory_state", "candidate_notes", "uncertainty"), "ProfileAgent")

    memory_plan = call_json_agent(client, "MemoryAgent", build_memory_prompt(session, profile), config)
    require_keys(
        memory_plan,
        (
            "tool_vs_memory",
            "memory_write",
            "memory_compress",
            "memory_forget",
            "read_key",
            "write_topic",
            "forget_topic",
        ),
        "MemoryAgent",
    )

    recommendation = call_json_agent(
        client,
        "RecommendationAgent",
        build_recommendation_prompt(session, profile, memory_plan),
        config,
    )
    require_keys(
        recommendation,
        ("preference_type", "ranking", "ranking_reason", "rejected_candidates"),
        "RecommendationAgent",
    )

    final_payload = call_json_agent(
        client,
        "CriticAgent",
        build_critic_prompt(session, profile, memory_plan, recommendation),
        config,
    )
    decision = validate_final_teacher_payload(final_payload, session)

    decision.trajectory["profile_agent_raw"] = profile
    decision.trajectory["memory_agent_raw"] = memory_plan
    decision.trajectory["recommendation_agent_raw"] = recommendation
    decision.trajectory["critic_agent_raw"] = final_payload
    return decision


def teacher_hits_target(session: Session, decision: TeacherDecision) -> bool:
    return decision.ranking[0] == session.target_item



def build_candidate_sessions(
    config: RealTeacherConfig,
    total_users: int,
    topic_lookup: dict[str, str],
) -> list[Session]:
    if total_users <= 0:
        raise ValueError("total_users must be positive")
    reviews_by_user: dict[str, list[tuple[Any, str]]] = defaultdict(list)
    with open_gzip_text(config.review_file) as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            user_id = record.get("user_id")
            asin = record.get("asin")
            if user_id is None or asin is None:
                continue
            asin_str = str(asin)
            if asin_str not in topic_lookup:
                continue
            timestamp = record["timestamp"] if "timestamp" in record else 0
            reviews_by_user[str(user_id)].append((timestamp, asin_str))

    sessions: list[Session] = []
    for source_user_id in sorted(reviews_by_user):
        if len(sessions) >= total_users:
            break
        user_reviews = reviews_by_user[source_user_id]
        if len(user_reviews) < config.sessions_per_user + 1:
            continue
        user_reviews.sort(key=lambda item: item[0])
        history_slice = user_reviews[: config.sessions_per_user]
        target_review = user_reviews[config.sessions_per_user]

        history_topic_counts: dict[str, int] = defaultdict(int)
        for _, asin in history_slice:
            history_topic_counts[topic_lookup[asin]] += 1
        if not history_topic_counts:
            raise ValueError(f"Expected non-empty history topics for user {source_user_id}")

        current_topic = topic_lookup[target_review[1]]
        memory_topic = max(history_topic_counts, key=history_topic_counts.get)
        memory_strength = history_topic_counts[memory_topic]
        candidate_items = tuple(
            dict.fromkeys(
                [target_review[1]]
                + [asin for _, asin in history_slice]
                + [
                    asin
                    for _, asin in user_reviews[
                        config.sessions_per_user + 1 : config.sessions_per_user + 1 + config.max_candidate_items
                    ]
                ]
            )
        )[: config.max_candidate_items]
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
                user_id=len(sessions),
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

    if len(sessions) < total_users:
        raise ValueError(f"Only built {len(sessions)} candidate sessions; expected at least {total_users}")
    return sessions


def select_sessions(config: RealTeacherConfig, topic_lookup: dict[str, str]) -> tuple[list[Session], list[Session]]:
    if config.candidate_multiplier <= 0:
        raise ValueError("candidate_multiplier must be positive")
    train_pool_size = config.train_count * config.candidate_multiplier
    eval_pool_size = config.eval_count * config.candidate_multiplier
    total_users = config.eval_offset + eval_pool_size
    sessions = build_candidate_sessions(config, total_users, topic_lookup)
    train_sessions = sessions[:train_pool_size]
    eval_sessions = sessions[config.eval_offset : config.eval_offset + eval_pool_size]
    if len(train_sessions) != train_pool_size:
        raise ValueError(f"Expected {train_pool_size} train candidate sessions, got {len(train_sessions)}")
    if len(eval_sessions) != eval_pool_size:
        raise ValueError(f"Expected {eval_pool_size} eval candidate sessions, got {len(eval_sessions)}")
    train_ids = {session.user_id for session in train_sessions}
    eval_ids = {session.user_id for session in eval_sessions}
    overlap = train_ids.intersection(eval_ids)
    if overlap:
        raise ValueError(f"Train and eval users overlap: {sorted(overlap)}")
    return train_sessions, eval_sessions


def summarize_decisions(decisions: list[TeacherDecision]) -> dict[str, dict[str, int]]:
    return {
        "tool_vs_memory": dict(Counter(ACTION_NAMES[decision.select_action] for decision in decisions)),
        "memory_write": dict(Counter(WRITE_NAMES[decision.write_action] for decision in decisions)),
        "memory_compress": dict(Counter(COMPRESS_NAMES[decision.compress_action] for decision in decisions)),
        "memory_forget": dict(Counter(FORGET_NAMES[decision.forget_action] for decision in decisions)),
        "preference_type": dict(Counter(decision.preference_type for decision in decisions)),
    }


def filter_teacher_candidates(
    client,
    candidate_sessions: list[Session],
    required_count: int,
    split_name: str,
    config: RealTeacherConfig,
    topic_lookup: dict[str, str],
) -> tuple[list[Session], list[TeacherDecision], dict[str, Any]]:
    if required_count <= 0:
        raise ValueError("required_count must be positive")
    kept_sessions: list[Session] = []
    kept_decisions: list[TeacherDecision] = []
    attempted_decisions: list[TeacherDecision] = []
    filtered_out_examples: list[dict[str, Any]] = []
    for attempt_index, session in enumerate(candidate_sessions, start=1):
        decision = run_real_multiagent_teacher(client, session, config, topic_lookup)
        attempted_decisions.append(decision)
        is_hit = teacher_hits_target(session, decision)
        print(
            f"[real-teacher] {split_name} attempt {attempt_index}/{len(candidate_sessions)} "
            f"target_hit={is_hit} ranking_top={decision.ranking[0]} target={session.target_item}",
            flush=True,
        )
        if config.require_teacher_target_hit and not is_hit:
            if len(filtered_out_examples) < 5:
                filtered_out_examples.append(
                    {
                        "session": session_payload(session),
                        "target_item": session.target_item,
                        "teacher_ranking_top": decision.ranking[0],
                    }
                )
            continue
        kept_sessions.append(session)
        kept_decisions.append(decision)
        if len(kept_sessions) == required_count:
            break

    if len(kept_sessions) != required_count:
        raise ValueError(
            f"Only kept {len(kept_sessions)} {split_name} sessions out of {len(candidate_sessions)} "
            f"candidate sessions after teacher filtering; need {required_count}"
        )

    attempted_hits = sum(
        teacher_hits_target(session, decision)
        for session, decision in zip(candidate_sessions[: len(attempted_decisions)], attempted_decisions)
    )
    stats = {
        "candidate_pool_size": len(candidate_sessions),
        "attempted": len(attempted_decisions),
        "kept": len(kept_sessions),
        "filtered_out": len(attempted_decisions) - len(kept_sessions),
        "attempt_rank_at_1": attempted_hits / len(attempted_decisions),
        "kept_rank_at_1": sum(
            teacher_hits_target(session, decision)
            for session, decision in zip(kept_sessions, kept_decisions)
        )
        / len(kept_sessions),
        "filtered_out_examples": filtered_out_examples,
    }
    return kept_sessions, kept_decisions, stats


def run_generation(config: RealTeacherConfig) -> dict[str, Any]:
    if config.train_count <= 0 or config.eval_count <= 0:
        raise ValueError("train_count and eval_count must be positive")
    if config.candidate_multiplier <= 0:
        raise ValueError("candidate_multiplier must be positive")
    if config.eval_offset < config.train_count * config.candidate_multiplier:
        raise ValueError(
            "eval_offset must be greater than or equal to train_count * candidate_multiplier "
            f"({config.train_count * config.candidate_multiplier})"
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    topic_lookup = load_topic_lookup(config.meta_file)
    train_candidate_sessions, eval_candidate_sessions = select_sessions(config, topic_lookup)
    client = create_real_teacher_client(config.teacher_model)

    train_sessions, train_decisions, train_filter_stats = filter_teacher_candidates(
        client,
        train_candidate_sessions,
        config.train_count,
        "train",
        config,
        topic_lookup,
    )
    eval_sessions, eval_decisions, eval_filter_stats = filter_teacher_candidates(
        client,
        eval_candidate_sessions,
        config.eval_count,
        "eval",
        config,
        topic_lookup,
    )

    export_teacher_trajectories(train_sessions, train_decisions, config.train_jsonl)
    export_teacher_trajectories(eval_sessions, eval_decisions, config.eval_jsonl)
    export_real_llamafactory_sft_dataset(train_sessions, train_decisions, config.llamafactory_sft_json)

    result = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "artifacts": {
            "train_jsonl": str(config.train_jsonl),
            "eval_jsonl": str(config.eval_jsonl),
            "llamafactory_sft_json": str(config.llamafactory_sft_json),
        },
        "sizes": {
            "train_candidate_pool": len(train_candidate_sessions),
            "eval_candidate_pool": len(eval_candidate_sessions),
            "train": len(train_sessions),
            "eval": len(eval_sessions),
            "total_teacher_calls": 4 * (train_filter_stats["attempted"] + eval_filter_stats["attempted"]),
        },
        "teacher_filtering": {
            "enabled": config.require_teacher_target_hit,
            "criterion": "teacher_ranking_top_equals_target_item",
            "train": train_filter_stats,
            "eval": eval_filter_stats,
        },
        "train_label_distribution": summarize_decisions(train_decisions),
        "eval_label_distribution": summarize_decisions(eval_decisions),
        "teacher_metrics": {
            "train_rank_at_1": ranking_metrics(train_sessions, [decision.ranking for decision in train_decisions])["rank_at_1"],
            "eval_rank_at_1": ranking_metrics(eval_sessions, [decision.ranking for decision in eval_decisions])["rank_at_1"],
        },
        "examples": [
            {
                "session": session_payload(session),
                "target_item": session.target_item,
                "teacher": teacher_payload(decision),
            }
            for session, decision in zip(eval_sessions[:2], eval_decisions[:2])
        ],
    }
    config.result_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    error_path = config.output_dir / "teacher_generation_error.json"
    if error_path.exists():
        error_path.unlink()
    return result


def parse_args() -> RealTeacherConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-file", type=Path, default=RealTeacherConfig.review_file)
    parser.add_argument("--meta-file", type=Path, default=RealTeacherConfig.meta_file)
    parser.add_argument("--train-count", type=int, default=RealTeacherConfig.train_count)
    parser.add_argument("--eval-count", type=int, default=RealTeacherConfig.eval_count)
    parser.add_argument("--eval-offset", type=int, default=RealTeacherConfig.eval_offset)
    parser.add_argument("--candidate-multiplier", type=int, default=RealTeacherConfig.candidate_multiplier)
    parser.add_argument("--disable-teacher-target-hit-filter", action="store_true")
    parser.add_argument("--sessions-per-user", type=int, default=RealTeacherConfig.sessions_per_user)
    parser.add_argument("--max-candidate-items", type=int, default=RealTeacherConfig.max_candidate_items)
    parser.add_argument("--teacher-model", default=RealTeacherConfig.teacher_model)
    parser.add_argument("--max-tokens", type=int, default=RealTeacherConfig.max_tokens)
    parser.add_argument("--temperature", type=float, default=RealTeacherConfig.temperature)
    parser.add_argument("--max-retries", type=int, default=RealTeacherConfig.max_retries)
    parser.add_argument("--call-timeout-seconds", type=int, default=RealTeacherConfig.call_timeout_seconds)
    parser.add_argument("--output-dir", type=Path, default=RealTeacherConfig.output_dir)
    parser.add_argument("--train-jsonl", type=Path, default=RealTeacherConfig.train_jsonl)
    parser.add_argument("--eval-jsonl", type=Path, default=RealTeacherConfig.eval_jsonl)
    parser.add_argument("--result-json", type=Path, default=RealTeacherConfig.result_json)
    parser.add_argument("--llamafactory-sft-json", type=Path, default=RealTeacherConfig.llamafactory_sft_json)
    args = parser.parse_args()
    return RealTeacherConfig(
        review_file=args.review_file,
        meta_file=args.meta_file,
        train_count=args.train_count,
        eval_count=args.eval_count,
        eval_offset=args.eval_offset,
        candidate_multiplier=args.candidate_multiplier,
        require_teacher_target_hit=not args.disable_teacher_target_hit_filter,
        sessions_per_user=args.sessions_per_user,
        max_candidate_items=args.max_candidate_items,
        teacher_model=args.teacher_model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_retries=args.max_retries,
        call_timeout_seconds=args.call_timeout_seconds,
        output_dir=args.output_dir,
        train_jsonl=args.train_jsonl,
        eval_jsonl=args.eval_jsonl,
        result_json=args.result_json,
        llamafactory_sft_json=args.llamafactory_sft_json,
    )


def main() -> None:
    config = parse_args()
    try:
        result = run_generation(config)
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    except Exception as exc:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        error_payload = {
            "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
            "error": {
                "type": f"{type(exc).__module__}.{type(exc).__name__}",
                "message": str(exc),
            },
        }
        error_path = config.output_dir / "teacher_generation_error.json"
        error_path.write_text(json.dumps(error_payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
